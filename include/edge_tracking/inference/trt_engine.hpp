#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace edge_tracking::inference {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    size_t elements = 0;  // product of shape
};

// Runs a serialized TensorRT engine with one float32 input and one float32 output
// (e.g. YOLOv8: "images" 1x3x640x640 -> "output0" 1x84x8400). Buffers are owned by the
// caller, so the preprocessing output can be passed in directly without a copy.
class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    bool load(const std::string& engine_path, std::string* error = nullptr);

    const TensorInfo& input() const;
    const TensorInfo& output() const;

    // Enqueues one inference on `stream`. `input` and `output` are device buffers holding
    // input().elements and output().elements floats. Returns false if enqueueing failed.
    // With CUDA graphs enabled (default), the first call for each (input, output, stream)
    // records TensorRT's kernel launches once; later calls replay them with a single launch,
    // which saves most of the CPU time spent launching kernels. Reusing a few fixed buffers
    // (like the pipeline's slots) keeps the number of recorded graphs small.
    bool enqueue(const float* input, float* output, cudaStream_t stream);

    void set_use_cuda_graphs(bool enabled);
    // False if recording a graph failed and the engine fell back to direct launches.
    bool cuda_graphs_active() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_tracking::inference
