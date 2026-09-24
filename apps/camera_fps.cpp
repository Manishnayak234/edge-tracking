// Measures capture rate and frame timing of ArgusCamera.
// Usage: camera_fps [num_frames]   (default 600, Ctrl+C to stop early)

#include "edge_tracking/camera/argus_camera.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }
}  // namespace

int main(int argc, char** argv) {
    using edge_tracking::camera::ArgusCamera;
    using edge_tracking::camera::CameraConfig;
    using edge_tracking::camera::Frame;
    using Clock = std::chrono::steady_clock;

    const long target_frames = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 600;
    std::signal(SIGINT, on_signal);

    CameraConfig config;
    ArgusCamera camera(config);
    std::printf("pipeline: %s\n", camera.pipeline_description().c_str());

    std::string error;
    if (!camera.start(&error)) {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    Frame frame;
    // First frame includes Argus startup (~1-2 s), so timing starts after it.
    if (!camera.read(frame, std::chrono::milliseconds(5000))) {
        std::fprintf(stderr, "no first frame: %s\n", camera.last_error().c_str());
        return 1;
    }
    std::printf("first frame: %dx%d NV12, %zu bytes\n", frame.width, frame.height, frame.nv12.size());

    const double expected_gap_ms = 1000.0 / config.fps;
    long frames = 0;
    long late_frames = 0;  // gaps longer than 1.5x the expected interval, i.e. likely drops
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

        if (prev_pts >= 0 && frame.pts_ns >= 0) {
            const double gap_ms = (frame.pts_ns - prev_pts) / 1e6;
            max_gap_ms = std::max(max_gap_ms, gap_ms);
            if (gap_ms > 1.5 * expected_gap_ms) ++late_frames;
        }
        prev_pts = frame.pts_ns;

        const auto now = Clock::now();
        if (now - window_start >= std::chrono::seconds(1)) {
            const double secs = std::chrono::duration<double>(now - window_start).count();
            std::printf("%6.1f fps  (frame %llu)\n", window_frames / secs,
                        static_cast<unsigned long long>(frame.index));
            window_start = now;
            window_frames = 0;
        }
    }

    const double total_s = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("\nframes: %ld in %.2f s -> %.1f fps (target %d)\n", frames, total_s,
                total_s > 0 ? frames / total_s : 0.0, config.fps);
    std::printf("max frame gap: %.1f ms (expected %.1f ms), late/dropped: %ld\n", max_gap_ms,
                expected_gap_ms, late_frames);

    camera.stop();
    return frames > 0 ? 0 : 1;
}
