// Checks and times YOLOv8 postprocessing (decode + NMS).
//  1. Decoder alone: feeds ONNX Runtime outputs (models/*_output.f32) to YoloDecoder and
//     compares with Ultralytics non_max_suppression results (models/*_detections.txt).
//  2. Engine + decoder: runs TensorRT on models/bus_input.f32 and compares the detections.
//  3. Live: camera -> preprocess -> inference -> decode, timing each stage.
// Usage: postprocess_check [engine] [num_frames]
//        (defaults: models/yolov8n_fp16.engine, 300; run from the project root)

#include "edge_tracking/camera/argus_gpu_camera.hpp"
#include "edge_tracking/inference/trt_engine.hpp"
#include "edge_tracking/postprocess/yolo_decoder.hpp"
#include "edge_tracking/preprocess/letterbox.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

namespace post = edge_tracking::postprocess;
namespace pre = edge_tracking::preprocess;

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t status_ = (call);                                                \
        if (status_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_));   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

bool read_floats(const std::string& path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    return file && file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float)) &&
           file.gcount() == static_cast<std::streamsize>(data.size() * sizeof(float));
}

// Lines of "class score x1 y1 x2 y2", as written by the Python reference script.
std::vector<post::Detection> read_detections(const std::string& path) {
    std::vector<post::Detection> dets;
    std::ifstream file(path);
    post::Detection d;
    while (file >> d.class_id >> d.score >> d.x1 >> d.y1 >> d.x2 >> d.y2) dets.push_back(d);
    return dets;
}

float iou(const post::Detection& a, const post::Detection& b) {
    const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    const float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (w <= 0.0f || h <= 0.0f) return 0.0f;
    const float inter = w * h;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

std::string name_of(const std::vector<std::string>& names, int id) {
    return id >= 0 && id < static_cast<int>(names.size()) ? names[id] : "class " + std::to_string(id);
}

// Matches each expected detection to the unused actual one of the same class with the
// highest IoU. Passes if counts are equal and every match has IoU >= min_iou.
bool compare(const char* label, const std::vector<post::Detection>& expected, const std::vector<post::Detection>& actual,
             float min_iou, const std::vector<std::string>& names) {
    std::vector<bool> used(actual.size(), false);
    float worst_iou = 1.0f;
    float worst_score_diff = 0.0f;
    int matched = 0;
    for (const post::Detection& e : expected) {
        int best = -1;
        float best_iou = 0.0f;
        for (size_t j = 0; j < actual.size(); ++j) {
            if (used[j] || actual[j].class_id != e.class_id) continue;
            const float v = iou(e, actual[j]);
            if (v > best_iou) best_iou = v, best = static_cast<int>(j);
        }
        if (best < 0) {
            std::printf("    missing: %s %.3f\n", name_of(names, e.class_id).c_str(), e.score);
            worst_iou = 0.0f;
            continue;
        }
        used[best] = true;
        ++matched;
        worst_iou = std::min(worst_iou, best_iou);
        worst_score_diff = std::max(worst_score_diff, std::abs(e.score - actual[best].score));
    }
    const bool pass = matched == static_cast<int>(expected.size()) && actual.size() == expected.size() &&
                      worst_iou >= min_iou;
    std::printf("  %-34s expected %zu, got %zu, matched %d, worst IoU %.4f, max score diff %.4f -> %s\n", label,
                expected.size(), actual.size(), matched, worst_iou, worst_score_diff, pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    namespace cam = edge_tracking::camera;
    namespace inf = edge_tracking::inference;

    const char* engine_path = argc > 1 ? argv[1] : "models/yolov8n_fp16.engine";
    const long num_frames = argc > 2 ? std::strtol(argv[2], nullptr, 10) : 300;
    const std::vector<std::string> names = post::load_class_names("models/coco_names.txt");

    inf::TrtEngine engine;
    std::string error;
    if (!engine.load(engine_path, &error)) {
        std::fprintf(stderr, "engine: %s\n", error.c_str());
        return 1;
    }
    const inf::TensorInfo& in = engine.input();
    const inf::TensorInfo& out = engine.output();
    const int input_size = static_cast<int>(in.shape[3]);
    const int num_candidates = static_cast<int>(out.shape[2]);
    const int num_classes = static_cast<int>(out.shape[1]) - 4;

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, in.elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, out.elements * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    post::YoloDecoder decoder(num_candidates, num_classes);
    // Reference files are in network-input coordinates: identity letterbox.
    const pre::LetterboxParams identity = pre::compute_letterbox(input_size, input_size, input_size, input_size);

    bool all_pass = true;
    std::printf("checks against Ultralytics (conf 0.25, IoU 0.7):\n");
    std::vector<float> host_in(in.elements);
    std::vector<float> host_out(out.elements);
    for (const char* name : {"ref", "bus"}) {
        const std::string prefix = std::string("models/") + name;
        const auto expected = read_detections(prefix + "_detections.txt");
        if (!read_floats(prefix + "_output.f32", host_out) || expected.empty()) {
            std::printf("  %s: reference files missing, skipped\n", name);
            continue;
        }
        CUDA_CHECK(cudaMemcpy(d_output, host_out.data(), out.elements * sizeof(float), cudaMemcpyHostToDevice));
        const auto decoded = decoder.decode(d_output, identity, stream);
        const std::string label = std::string(name) + ": decoder on ONNX output (" +
                                  std::to_string(decoder.last_candidate_count()) + " cand.)";
        all_pass &= compare(label.c_str(), expected, decoded, 0.999f, names);
    }
    {
        const auto expected = read_detections("models/bus_detections.txt");
        if (read_floats("models/bus_input.f32", host_in) && !expected.empty()) {
            CUDA_CHECK(cudaMemcpy(d_input, host_in.data(), in.elements * sizeof(float), cudaMemcpyHostToDevice));
            if (!engine.enqueue(d_input, d_output, stream)) return 1;
            const auto detected = decoder.decode(d_output, identity, stream);
            all_pass &= compare("bus: TensorRT FP16 + decoder", expected, detected, 0.98f, names);
            for (const post::Detection& d : detected) {
                std::printf("    %-8s %.3f  (%.1f, %.1f, %.1f, %.1f)\n", name_of(names, d.class_id).c_str(), d.score,
                            d.x1, d.y1, d.x2, d.y2);
            }
        }
    }

    // Live pipeline.
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

    using Clock = std::chrono::steady_clock;
    std::vector<float> infer_ms, decode_ms, frame_ms;
    const auto start = Clock::now();
    auto last_print = start;
    for (long i = 0; i < num_frames; ++i) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "camera: %s\n", camera.last_error().c_str());
            break;
        }
        const auto t0 = Clock::now();
        CUDA_CHECK(pre::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch, frame.color_space, lb,
                                              format, d_input, stream));
        if (!engine.enqueue(d_input, d_output, stream)) return 1;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t1 = Clock::now();
        const auto detections = decoder.decode(d_output, lb, stream);
        const auto t2 = Clock::now();
        infer_ms.push_back(std::chrono::duration<float, std::milli>(t1 - t0).count());
        decode_ms.push_back(std::chrono::duration<float, std::milli>(t2 - t1).count());
        frame_ms.push_back(std::chrono::duration<float, std::milli>(t2 - t0).count());

        if (t2 - last_print >= std::chrono::seconds(1)) {
            last_print = t2;
            std::printf("frame %4llu: %d candidates -> %zu detections",
                        static_cast<unsigned long long>(frame.index), decoder.last_candidate_count(),
                        detections.size());
            for (size_t k = 0; k < std::min<size_t>(detections.size(), 3); ++k) {
                const post::Detection& d = detections[k];
                std::printf("  | %s %.2f [%.0f,%.0f,%.0f,%.0f]", name_of(names, d.class_id).c_str(), d.score, d.x1,
                            d.y1, d.x2, d.y2);
            }
            std::printf("\n");
        }
    }
    const double total_s = std::chrono::duration<double>(Clock::now() - start).count();

    auto print_stats = [](const char* label, std::vector<float> ms) {
        std::sort(ms.begin(), ms.end());
        double sum = 0.0;
        for (float t : ms) sum += t;
        std::printf("  %-22s mean %6.3f ms, median %6.3f ms, p99 %6.3f ms\n", label, sum / ms.size(),
                    ms[ms.size() / 2], ms[ms.size() * 99 / 100]);
    };
    std::printf("live: %zu frames in %.2f s -> %.1f fps\n", frame_ms.size(), total_s, frame_ms.size() / total_s);
    print_stats("preprocess+inference", infer_ms);
    print_stats("decode+NMS", decode_ms);
    print_stats("per frame", frame_ms);

    camera.stop();
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    return all_pass ? 0 : 1;
}
