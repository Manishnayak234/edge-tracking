#include "edge_tracking/inference/trt_engine.hpp"

#include <NvInferRuntime.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iterator>

namespace edge_tracking::inference {

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
};

Logger g_logger;

}  // namespace

struct TrtEngine::Impl {
    // Destroyed in reverse order: context, engine, runtime.
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    TensorInfo input;
    TensorInfo output;

    struct Graph {
        const float* input;
        float* output;
        cudaStream_t stream;
        cudaGraphExec_t exec;
    };
    static constexpr size_t kMaxGraphs = 16;  // beyond this, new buffer pairs use direct enqueue
    std::vector<Graph> graphs;
    bool use_graphs = true;
    bool graphs_failed = false;
    bool warmed_up = false;  // TensorRT wants one normal enqueue before graph capture

    // TensorRT takes void* for all bindings; the input is only read.
    bool bind(const float* in, float* out) {
        return context->setTensorAddress(input.name.c_str(), const_cast<float*>(in)) &&
               context->setTensorAddress(output.name.c_str(), out);
    }

    // Records one enqueueV3 on `stream` into a graph. Returns null if capture is not possible.
    cudaGraphExec_t capture(cudaStream_t stream) {
        cudaGraph_t graph = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return nullptr;
        const bool enqueued = context->enqueueV3(stream);
        const cudaError_t end = cudaStreamEndCapture(stream, &graph);
        cudaGraphExec_t exec = nullptr;
        if (enqueued && end == cudaSuccess && graph && cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
            exec = nullptr;
        }
        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();  // a failed capture leaves a sticky error; the caller falls back
        return exec;
    }

    void destroy_graphs() {
        for (Graph& g : graphs) cudaGraphExecDestroy(g.exec);
        graphs.clear();
    }
};

TrtEngine::TrtEngine() : impl_(std::make_unique<Impl>()) {}

TrtEngine::~TrtEngine() {
    impl_->destroy_graphs();
    impl_->context.reset();
    impl_->engine.reset();
    impl_->runtime.reset();
}

bool TrtEngine::load(const std::string& engine_path, std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };

    std::ifstream file(engine_path, std::ios::binary);
    if (!file) return fail("cannot open engine file " + engine_path);
    const std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    Impl& s = *impl_;
    s.destroy_graphs();
    s.warmed_up = false;
    s.graphs_failed = false;
    s.context.reset();  // release a previously loaded engine in dependency order
    s.engine.reset();
    s.runtime.reset(nvinfer1::createInferRuntime(g_logger));
    if (!s.runtime) return fail("createInferRuntime failed");
    s.engine.reset(s.runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!s.engine) return fail("cannot deserialize engine (built with a different TensorRT version?)");
    s.context.reset(s.engine->createExecutionContext());
    if (!s.context) return fail("createExecutionContext failed");

    int inputs = 0;
    int outputs = 0;
    for (int i = 0; i < s.engine->getNbIOTensors(); ++i) {
        const char* name = s.engine->getIOTensorName(i);
        if (s.engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            return fail(std::string("tensor ") + name + " is not float32");
        }
        TensorInfo info;
        info.name = name;
        const nvinfer1::Dims dims = s.engine->getTensorShape(name);
        info.elements = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] < 0) return fail(std::string("tensor ") + name + " has a dynamic shape");
            info.shape.push_back(dims.d[d]);
            info.elements *= static_cast<size_t>(dims.d[d]);
        }
        if (s.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            s.input = info;
            ++inputs;
        } else {
            s.output = info;
            ++outputs;
        }
    }
    if (inputs != 1 || outputs != 1) return fail("engine must have exactly one input and one output");
    return true;
}

const TensorInfo& TrtEngine::input() const { return impl_->input; }
const TensorInfo& TrtEngine::output() const { return impl_->output; }

bool TrtEngine::enqueue(const float* input, float* output, cudaStream_t stream) {
    Impl& s = *impl_;
    if (!s.context) return false;
    if (s.use_graphs && !s.graphs_failed) {
        for (const Impl::Graph& g : s.graphs) {
            if (g.input == input && g.output == output && g.stream == stream) {
                return cudaGraphLaunch(g.exec, stream) == cudaSuccess;
            }
        }
    }
    if (!s.bind(input, output)) return false;
    const bool can_capture = s.use_graphs && !s.graphs_failed && stream && s.graphs.size() < Impl::kMaxGraphs;
    if (!can_capture) return s.context->enqueueV3(stream);

    if (!s.warmed_up) {
        if (!s.context->enqueueV3(stream) || cudaStreamSynchronize(stream) != cudaSuccess) return false;
        s.warmed_up = true;
    }
    const cudaGraphExec_t exec = s.capture(stream);
    if (!exec) {
        std::fprintf(stderr, "[TrtEngine] CUDA graph capture failed, using direct enqueue\n");
        s.graphs_failed = true;
        return s.context->enqueueV3(stream);
    }
    s.graphs.push_back({input, output, stream, exec});
    return cudaGraphLaunch(exec, stream) == cudaSuccess;
}

void TrtEngine::set_use_cuda_graphs(bool enabled) { impl_->use_graphs = enabled; }

bool TrtEngine::cuda_graphs_active() const { return impl_->use_graphs && !impl_->graphs_failed; }

}  // namespace edge_tracking::inference
