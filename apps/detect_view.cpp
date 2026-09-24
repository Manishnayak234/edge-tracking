// Live detection viewer: camera -> preprocess -> TensorRT -> decode, drawn with OpenCV.
// Debugging tool: drawing copies each frame to the CPU, the detection path stays on the GPU.
// Usage: detect_view [seconds] [engine] [snapshot.png]
//   seconds: 0 = run until q/Esc/Ctrl+C. Press s to save /tmp/detect_view_<n>.png.
//   snapshot.png: if given, the last shown frame is saved there on exit. Run from project root.
// Over SSH, start it through scripts/on_display.sh so the window opens on the Jetson screen.

#include "edge_tracking/camera/argus_gpu_camera.hpp"
#include "edge_tracking/inference/trt_engine.hpp"
#include "edge_tracking/postprocess/yolo_decoder.hpp"
#include "edge_tracking/preprocess/letterbox.hpp"

#include "view_utils.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t status_ = (call);                                                \
        if (status_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_));   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    namespace cam = edge_tracking::camera;
    namespace pre = edge_tracking::preprocess;
    namespace inf = edge_tracking::inference;
    namespace post = edge_tracking::postprocess;
    using Clock = std::chrono::steady_clock;

    const double run_seconds = argc > 1 ? std::strtod(argv[1], nullptr) : 0.0;
    const char* engine_path = argc > 2 ? argv[2] : "models/yolov8n_fp16.engine";
    const char* exit_snapshot = argc > 3 ? argv[3] : nullptr;
    std::signal(SIGINT, on_signal);

    inf::TrtEngine engine;
    std::string error;
    if (!engine.load(engine_path, &error)) {
        std::fprintf(stderr, "engine: %s\n", error.c_str());
        return 1;
    }
    const int input_size = static_cast<int>(engine.input().shape[3]);
    const int num_candidates = static_cast<int>(engine.output().shape[2]);
    const int num_classes = static_cast<int>(engine.output().shape[1]) - 4;
    const std::vector<std::string> names = post::load_class_names("models/coco_names.txt");

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, engine.input().elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, engine.output().elements * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    post::YoloDecoder decoder(num_candidates, num_classes);

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

    const char* window = "edge-tracking detections";
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1);
    cv::Mat bgr;

    const auto start = Clock::now();
    auto fps_window_start = start;
    int fps_window_frames = 0;
    double shown_fps = 0.0;
    long frames = 0;
    // Frames the camera produced but this loop never saw (dropped by the appsink while we
    // were busy), counted from gaps in the capture timestamps.
    const double frame_interval_ns = 1e9 / config.fps;
    int64_t prev_pts = frame.pts_ns;
    long skipped = 0;
    int snapshots = 0;

    while (!g_stop) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "camera: %s\n", camera.last_error().c_str());
            break;
        }
        if (prev_pts >= 0 && frame.pts_ns >= 0) {
            skipped += std::max<long>(std::lround((frame.pts_ns - prev_pts) / frame_interval_ns) - 1, 0);
        }
        prev_pts = frame.pts_ns;

        const auto t0 = Clock::now();
        CUDA_CHECK(pre::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch, frame.color_space, lb,
                                              format, d_input, stream));
        if (!engine.enqueue(d_input, d_output, stream)) return 1;
        const auto detections = decoder.decode(d_output, lb, stream);  // synchronizes the stream
        const double detect_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // Copy the frame for drawing before the next read() hands the buffer back.
        CUDA_CHECK(cudaMemcpy2D(nv12.data, frame.width, frame.y, frame.y_pitch, frame.width, frame.height,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(nv12.data + static_cast<size_t>(frame.width) * frame.height, frame.width, frame.uv,
                                frame.uv_pitch, frame.width, frame.height / 2, cudaMemcpyDeviceToHost));
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);  // BT.601: slight color shift, fine for viewing

        for (const post::Detection& d : detections) {
            const cv::Scalar color = edge_tracking::apps::palette_color(d.class_id);
            const cv::Point p1(static_cast<int>(d.x1), static_cast<int>(d.y1));
            const cv::Point p2(static_cast<int>(d.x2), static_cast<int>(d.y2));
            cv::rectangle(bgr, p1, p2, color, 2);
            const std::string name = d.class_id < static_cast<int>(names.size()) ? names[d.class_id]
                                                                                 : std::to_string(d.class_id);
            char label[96];
            std::snprintf(label, sizeof(label), "%s %.2f", name.c_str(), d.score);
            edge_tracking::apps::draw_label(bgr, label, p1, color);
        }

        ++frames;
        ++fps_window_frames;
        const auto now = Clock::now();
        const double window_s = std::chrono::duration<double>(now - fps_window_start).count();
        if (window_s >= 1.0) {
            shown_fps = fps_window_frames / window_s;
            fps_window_start = now;
            fps_window_frames = 0;
        }
        char stats[160];
        std::snprintf(stats, sizeof(stats), "%.1f fps | detect %.1f ms | %zu objects | camera frames skipped %ld",
                      shown_fps, detect_ms, detections.size(), skipped);
        edge_tracking::apps::draw_status(bgr, stats);

        cv::imshow(window, bgr);
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == 's') {
            const std::string path = "/tmp/detect_view_" + std::to_string(snapshots++) + ".png";
            cv::imwrite(path, bgr);
            std::printf("saved %s\n", path.c_str());
        }
        if (run_seconds > 0 && std::chrono::duration<double>(now - start).count() >= run_seconds) break;
    }

    const double total_s = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("shown %ld frames in %.1f s -> %.1f fps, camera frames skipped: %ld\n", frames, total_s,
                frames / total_s, skipped);
    if (exit_snapshot && !bgr.empty() && cv::imwrite(exit_snapshot, bgr)) std::printf("saved %s\n", exit_snapshot);

    cv::destroyAllWindows();
    camera.stop();
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
