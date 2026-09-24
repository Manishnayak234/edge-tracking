// Checks and times TensorRT inference.
//  1. Accuracy: runs the engine on models/ref_input.f32 and compares with ONNX Runtime's
//     FP32 output models/ref_output.f32 (made on the PC from the same camera tensor).
//  2. Live: camera -> preprocess -> inference, timing each GPU stage per frame.
// Usage: infer_check [engine] [num_frames]
//        (defaults: models/yolov8n_fp16.engine, 300; run from the project root)

#include "edge_tracking/camera/argus_gpu_camera.hpp"
#include "edge_tracking/inference/trt_engine.hpp"
#include "edge_tracking/preprocess/letterbox.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t status_ = (call);                                                \
        if (status_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_));   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

bool read_floats(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    return file && file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float)) &&
           file.gcount() == static_cast<std::streamsize>(data.size() * sizeof(float));
}

struct Stats {
    std::vector<float> ms;
    void print(const char* name) {
        std::sort(ms.begin(), ms.end());
        double sum = 0.0;
        for (float t : ms) sum += t;
        std::printf("  %-11s mean %6.3f ms, median %6.3f ms, p99 %6.3f ms, max %6.3f ms\n", name, sum / ms.size(),
                    ms[ms.size() / 2], ms[ms.size() * 99 / 100], ms.back());
    }
};

// YOLOv8 output is [84][num_candidates]: rows 0-3 = cx, cy, w, h; rows 4-83 = class scores.
struct Best {
    int candidate = -1;
    int cls = -1;
    float score = 0.0f;
};

Best best_candidate(const std::vector<float>& out, int num_candidates, int num_classes) {
    Best best;
    for (int c = 0; c < num_classes; ++c) {
        const float* row = out.data() + static_cast<size_t>(4 + c) * num_candidates;
        for (int i = 0; i < num_candidates; ++i) {
            if (row[i] > best.score) best = {i, c, row[i]};
        }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    namespace cam = edge_tracking::camera;
    namespace pre = edge_tracking::preprocess;
    namespace inf = edge_tracking::inference;

    const char* engine_path = argc > 1 ? argv[1] : "models/yolov8n_fp16.engine";
    const long num_frames = argc > 2 ? std::strtol(argv[2], nullptr, 10) : 300;

    inf::TrtEngine engine;
    std::string error;
    if (!engine.load(engine_path, &error)) {
        std::fprintf(stderr, "engine: %s\n", error.c_str());
        return 1;
    }
    const inf::TensorInfo& in = engine.input();
    const inf::TensorInfo& out = engine.output();
    if (in.shape.size() != 4 || out.shape.size() != 3) {
        std::fprintf(stderr, "unexpected tensor ranks for a YOLOv8 engine\n");
        return 1;
    }
    const int input_size = static_cast<int>(in.shape[3]);
    const int num_candidates = static_cast<int>(out.shape[2]);
    const int num_classes = static_cast<int>(out.shape[1]) - 4;
    std::printf("engine %s: %s [1,3,%d,%d] -> %s [1,%d,%d]\n", engine_path, in.name.c_str(), input_size, input_size,
                out.name.c_str(), num_classes + 4, num_candidates);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, in.elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, out.elements * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<float> host_output(out.elements);

    // 1. Accuracy against ONNX Runtime FP32.
    std::vector<float> ref_input(in.elements);
    std::vector<float> ref_output(out.elements);
    if (read_floats("models/ref_input.f32", ref_input) && read_floats("models/ref_output.f32", ref_output)) {
        CUDA_CHECK(cudaMemcpy(d_input, ref_input.data(), in.elements * sizeof(float), cudaMemcpyHostToDevice));
        if (!engine.enqueue(d_input, d_output, stream)) return 1;
        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), d_output, out.elements * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        double max_score_diff = 0.0;
        double max_box_diff_confident = 0.0;  // box error only matters where something is detected
        int confident = 0;
        int class_agree = 0;
        for (int i = 0; i < num_candidates; ++i) {
            int ref_cls = 0;
            int trt_cls = 0;
            float ref_best = -1.0f;
            float trt_best = -1.0f;
            for (int c = 0; c < num_classes; ++c) {
                const size_t k = static_cast<size_t>(4 + c) * num_candidates + i;
                max_score_diff = std::max(max_score_diff, static_cast<double>(std::abs(host_output[k] - ref_output[k])));
                if (ref_output[k] > ref_best) ref_best = ref_output[k], ref_cls = c;
                if (host_output[k] > trt_best) trt_best = host_output[k], trt_cls = c;
            }
            if (ref_best > 0.25f) {
                ++confident;
                if (ref_cls == trt_cls) ++class_agree;
                for (int r = 0; r < 4; ++r) {
                    const size_t k = static_cast<size_t>(r) * num_candidates + i;
                    max_box_diff_confident =
                        std::max(max_box_diff_confident, static_cast<double>(std::abs(host_output[k] - ref_output[k])));
                }
            }
        }
        const Best ref_top = best_candidate(ref_output, num_candidates, num_classes);
        const Best trt_top = best_candidate(host_output, num_candidates, num_classes);
        std::printf("accuracy vs ONNX Runtime FP32 (same input tensor):\n");
        std::printf("  max class-score diff %.4f; candidates with score > 0.25: %d, same class: %d\n",
                    max_score_diff, confident, class_agree);
        std::printf("  max box diff on those candidates: %.2f px\n", max_box_diff_confident);
        std::printf("  top: ONNX class %d score %.3f (candidate %d) | TensorRT class %d score %.3f (candidate %d)\n",
                    ref_top.cls, ref_top.score, ref_top.candidate, trt_top.cls, trt_top.score, trt_top.candidate);
    } else {
        std::printf("models/ref_input.f32 or ref_output.f32 missing: skipping accuracy check\n");
    }

    // 2. Live pipeline timing.
    cam::CameraConfig config;
    cam::ArgusGpuCamera camera(config);
    if (!camera.start(&error)) {
        std::fprintf(stderr, "camera: %s\n", error.c_str());
        return 1;
    }
    cam::GpuFrame frame;
    if (!camera.read(frame, std::chrono::milliseconds(5000))) {
        std::fprintf(stderr, "camera: %s\n", camera.last_error().c_str());
        return 1;
    }
    const pre::LetterboxParams lb = pre::compute_letterbox(frame.width, frame.height, input_size, input_size);
    const pre::TensorFormat format;

    cudaEvent_t e_start, e_pre, e_infer;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_pre));
    CUDA_CHECK(cudaEventCreate(&e_infer));
    Stats pre_stats, infer_stats, frame_stats;

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto last_print = start;
    for (long i = 0; i < num_frames; ++i) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "camera: %s\n", camera.last_error().c_str());
            break;
        }
        const auto t0 = Clock::now();
        CUDA_CHECK(cudaEventRecord(e_start, stream));
        CUDA_CHECK(pre::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch, frame.color_space, lb,
                                              format, d_input, stream));
        CUDA_CHECK(cudaEventRecord(e_pre, stream));
        if (!engine.enqueue(d_input, d_output, stream)) {
            std::fprintf(stderr, "enqueue failed\n");
            return 1;
        }
        CUDA_CHECK(cudaEventRecord(e_infer, stream));
        CUDA_CHECK(cudaEventSynchronize(e_infer));  // frame is released on the next read()
        frame_stats.ms.push_back(std::chrono::duration<float, std::milli>(Clock::now() - t0).count());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_pre));
        pre_stats.ms.push_back(ms);
        CUDA_CHECK(cudaEventElapsedTime(&ms, e_pre, e_infer));
        infer_stats.ms.push_back(ms);

        if (Clock::now() - last_print >= std::chrono::seconds(1)) {
            last_print = Clock::now();
            CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, out.elements * sizeof(float), cudaMemcpyDeviceToHost));
            const Best top = best_candidate(host_output, num_candidates, num_classes);
            std::printf("frame %4llu: top candidate class %2d, score %.2f\n",
                        static_cast<unsigned long long>(frame.index), top.cls, top.score);
        }
    }
    const double total_s = std::chrono::duration<double>(Clock::now() - start).count();

    std::printf("live: %zu frames in %.2f s -> %.1f fps (camera %d fps)\n", frame_stats.ms.size(), total_s,
                frame_stats.ms.size() / total_s, config.fps);
    pre_stats.print("preprocess");
    infer_stats.print("inference");
    frame_stats.print("per frame");

    camera.stop();
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_pre);
    cudaEventDestroy(e_infer);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
