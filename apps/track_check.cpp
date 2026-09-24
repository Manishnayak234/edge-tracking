// Tracker checks, compared against Ultralytics with tools/bytetrack_reference.py.
//   track_check record <frames> <detections.txt>   live camera detections (score >= 0.1) to a file
//   track_check replay <detections.txt> <tracks.txt>   run ByteTracker over a detection file
// File formats (frames count from 1):
//   detections: "F <frame> <n>" then n lines "<class> <score> <x1> <y1> <x2> <y2>"
//   tracks:     "F <frame> <n>" then n lines "<id> <class> <score> <x1> <y1> <x2> <y2>"
// Run from the project root.

#include "edge_tracking/camera/argus_gpu_camera.hpp"
#include "edge_tracking/inference/trt_engine.hpp"
#include "edge_tracking/postprocess/yolo_decoder.hpp"
#include "edge_tracking/preprocess/letterbox.hpp"
#include "edge_tracking/tracking/byte_tracker.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace post = edge_tracking::postprocess;

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t status_ = (call);                                                \
        if (status_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_));   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

int record(long num_frames, const char* path) {
    namespace cam = edge_tracking::camera;
    namespace pre = edge_tracking::preprocess;
    namespace inf = edge_tracking::inference;

    inf::TrtEngine engine;
    std::string error;
    if (!engine.load("models/yolov8n_fp16.engine", &error)) {
        std::fprintf(stderr, "engine: %s\n", error.c_str());
        return 1;
    }
    const int input_size = static_cast<int>(engine.input().shape[3]);
    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, engine.input().elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, engine.output().elements * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    post::DecoderConfig decoder_config;
    decoder_config.score_threshold = 0.1f;  // tracking needs the low-score boxes
    post::YoloDecoder decoder(static_cast<int>(engine.output().shape[2]),
                              static_cast<int>(engine.output().shape[1]) - 4, decoder_config);

    cam::ArgusGpuCamera camera(cam::CameraConfig{});
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

    std::FILE* out = std::fopen(path, "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    long total = 0;
    for (long f = 1; f <= num_frames; ++f) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "camera: %s\n", camera.last_error().c_str());
            break;
        }
        CUDA_CHECK(pre::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch, frame.color_space, lb,
                                              pre::TensorFormat{}, d_input, stream));
        if (!engine.enqueue(d_input, d_output, stream)) return 1;
        const auto dets = decoder.decode(d_output, lb, stream);
        std::fprintf(out, "F %ld %zu\n", f, dets.size());
        for (const auto& d : dets) {
            std::fprintf(out, "%d %.6f %.4f %.4f %.4f %.4f\n", d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
        }
        total += static_cast<long>(dets.size());
    }
    std::fclose(out);
    std::printf("recorded %ld frames, %ld detections -> %s\n", num_frames, total, path);
    camera.stop();
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

int replay(const char* in_path, const char* out_path) {
    std::ifstream in(in_path);
    std::FILE* out = std::fopen(out_path, "w");
    if (!in || !out) {
        std::fprintf(stderr, "cannot open %s or %s\n", in_path, out_path);
        return 1;
    }
    edge_tracking::tracking::ByteTracker tracker;
    std::vector<double> update_us;
    std::string tag;
    long frame = 0;
    size_t count = 0;
    int max_id = 0;
    while (in >> tag >> frame >> count) {
        std::vector<post::Detection> dets(count);
        for (auto& d : dets) in >> d.class_id >> d.score >> d.x1 >> d.y1 >> d.x2 >> d.y2;
        const auto t0 = std::chrono::steady_clock::now();
        const auto tracks = tracker.update(dets);
        update_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        std::fprintf(out, "F %ld %zu\n", frame, tracks.size());
        for (const auto& t : tracks) {
            std::fprintf(out, "%d %d %.6f %.4f %.4f %.4f %.4f\n", t.id, t.class_id, t.score, t.x1, t.y1, t.x2, t.y2);
            max_id = std::max(max_id, t.id);
        }
    }
    std::fclose(out);
    std::sort(update_us.begin(), update_us.end());
    double sum = 0.0;
    for (double t : update_us) sum += t;
    std::printf("replayed %zu frames, %d track ids -> %s; update() mean %.1f us, max %.1f us\n", update_us.size(),
                max_id, out_path, update_us.empty() ? 0.0 : sum / update_us.size(),
                update_us.empty() ? 0.0 : update_us.back());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "record") == 0) return record(std::strtol(argv[2], nullptr, 10), argv[3]);
    if (argc == 4 && std::strcmp(argv[1], "replay") == 0) return replay(argv[2], argv[3]);
    std::fprintf(stderr,
                 "usage: track_check record <frames> <detections.txt>\n"
                 "       track_check replay <detections.txt> <tracks.txt>\n");
    return 2;
}
