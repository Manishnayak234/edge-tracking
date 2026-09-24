// Measures zero-copy capture with ArgusGpuCamera and proves the GPU pointers are right:
// a CUDA kernel averages the Y plane of every frame, and every 120 frames the result is
// checked against the same average computed on the CPU from a copy of the plane.
// Usage: camera_gpu_fps [num_frames]   (default 600, Ctrl+C to stop early)

#include "edge_tracking/camera/argus_gpu_camera.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// Sums all luma values of a pitch-linear plane into *out (grid-stride loop + warp reduce).
__global__ void sum_luma(const uint8_t* y, size_t pitch, int width, int height,
                         unsigned long long* out) {
    unsigned long long local = 0;
    const int total = width * height;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int row = i / width;
        const int col = i - row * width;
        local += y[static_cast<size_t>(row) * pitch + col];
    }
    for (int offset = 16; offset > 0; offset /= 2) local += __shfl_down_sync(0xffffffffu, local, offset);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, local);
}

}  // namespace

int main(int argc, char** argv) {
    using edge_tracking::camera::ArgusGpuCamera;
    using edge_tracking::camera::CameraConfig;
    using edge_tracking::camera::GpuFrame;
    using Clock = std::chrono::steady_clock;

    const long target_frames = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 600;
    std::signal(SIGINT, on_signal);

    CameraConfig config;
    ArgusGpuCamera camera(config);
    std::printf("pipeline: %s\n", camera.pipeline_description().c_str());

    std::string error;
    if (!camera.start(&error)) {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    GpuFrame frame;
    if (!camera.read(frame, std::chrono::milliseconds(5000))) {
        std::fprintf(stderr, "no first frame: %s\n", camera.last_error().c_str());
        return 1;
    }
    std::printf("first frame: %dx%d NV12 on GPU, Y pitch %zu, UV pitch %zu\n", frame.width, frame.height,
                frame.y_pitch, frame.uv_pitch);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    unsigned long long* d_sum = nullptr;
    unsigned long long h_sum = 0;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(*d_sum)));
    std::vector<uint8_t> host_y;

    const double expected_gap_ms = 1000.0 / config.fps;
    long frames = 0;
    long late_frames = 0;  // gaps longer than 1.5x the expected interval, i.e. likely drops
    long checks = 0;
    long check_failures = 0;
    double max_gap_ms = 0.0;
    int64_t prev_pts = frame.pts_ns;

    const auto start = Clock::now();
    auto window_start = start;
    long window_frames = 0;

    while (!g_stop && frames < target_frames) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "read failed: %s\n", camera.last_error().c_str());
            break;
        }
        ++frames;
        ++window_frames;

        const double pixels = static_cast<double>(frame.width) * frame.height;
        CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(*d_sum), stream));
        sum_luma<<<64, 256, 0, stream>>>(frame.y, frame.y_pitch, frame.width, frame.height, d_sum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost, stream));
        // The frame is released on the next read(), so GPU work on it must finish first.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (frames % 120 == 0) {
            host_y.resize(static_cast<size_t>(frame.width) * frame.height);
            CUDA_CHECK(cudaMemcpy2D(host_y.data(), frame.width, frame.y, frame.y_pitch, frame.width,
                                    frame.height, cudaMemcpyDeviceToHost));
            unsigned long long cpu_sum = 0;
            for (uint8_t v : host_y) cpu_sum += v;
            const bool ok = cpu_sum == h_sum;
            ++checks;
            if (!ok) ++check_failures;
            std::printf("  check frame %llu: GPU mean luma %.2f, CPU mean luma %.2f -> %s\n",
                        static_cast<unsigned long long>(frame.index), h_sum / pixels, cpu_sum / pixels,
                        ok ? "match" : "MISMATCH");
        }

        if (prev_pts >= 0 && frame.pts_ns >= 0) {
            const double gap_ms = (frame.pts_ns - prev_pts) / 1e6;
            max_gap_ms = std::max(max_gap_ms, gap_ms);
            if (gap_ms > 1.5 * expected_gap_ms) ++late_frames;
        }
        prev_pts = frame.pts_ns;

        const auto now = Clock::now();
        if (now - window_start >= std::chrono::seconds(1)) {
            const double secs = std::chrono::duration<double>(now - window_start).count();
            std::printf("%6.1f fps  (frame %llu, mean luma %.1f)\n", window_frames / secs,
                        static_cast<unsigned long long>(frame.index), h_sum / pixels);
            window_start = now;
            window_frames = 0;
        }
    }

    const double total_s = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("\nframes: %ld in %.2f s -> %.1f fps (target %d)\n", frames, total_s,
                total_s > 0 ? frames / total_s : 0.0, config.fps);
    std::printf("max frame gap: %.1f ms (expected %.1f ms), late/dropped: %ld\n", max_gap_ms,
                expected_gap_ms, late_frames);
    std::printf("GPU/CPU checks: %ld passed, %ld failed\n", checks - check_failures, check_failures);

    camera.stop();
    cudaFree(d_sum);
    cudaStreamDestroy(stream);
    return frames > 0 && check_failures == 0 ? 0 : 1;
}
