// Runs the threaded pipeline (camera -> detection -> tracking) and prints telemetry every second.
// Usage: pipeline_run [seconds] [view|headless|spin] [snapshot.png]
//   seconds:  0 = until Ctrl+C (or q in the window)
//   view:     draw tracks in a window (start through scripts/on_display.sh over SSH)
//   spin:     headless, with CUDA spin-waiting instead of blocking sync (for comparison)
//   snapshot.png: with view, the last shown frame is saved there on exit
// Run from the project root.

#include "edge_tracking/pipeline/pipeline.hpp"
#include "view_utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

void print_totals(const edge_tracking::telemetry::Snapshot& s) {
    using edge_tracking::telemetry::Counter;
    using edge_tracking::telemetry::Stage;
    std::printf("\nsummary over %.1f s\n", s.window_s);
    std::printf("  camera frames %ld (%.1f fps), results %ld (%.1f fps)\n", s.counter(Counter::CameraFrames),
                s.rate(Counter::CameraFrames), s.counter(Counter::Results), s.rate(Counter::Results));
    std::printf("  dropped: by camera %ld, at ingest (no free slot) %ld, at sink (sink busy) %ld\n",
                s.counter(Counter::CameraDropped), s.counter(Counter::IngestDropped), s.counter(Counter::SinkDropped));
    const struct {
        const char* name;
        Stage stage;
    } rows[] = {{"capture", Stage::Capture},
                {"inference", Stage::Inference},
                {"tracking", Stage::Tracking},
                {"sink", Stage::Sink},
                {"latency (end to end)", Stage::Latency}};
    for (const auto& row : rows) {
        const auto& st = s.stage(row.stage);
        std::printf("  %-21s mean %7.3f ms, p50 %7.3f, p99 %7.3f, max %7.3f  (%ld samples)\n", row.name, st.mean_ms,
                    st.p50_ms, st.p99_ms, st.max_ms, st.count);
    }
}

}  // namespace

int main(int argc, char** argv) {
    namespace pl = edge_tracking::pipeline;
    namespace view = edge_tracking::apps;

    const double run_seconds = argc > 1 ? std::strtod(argv[1], nullptr) : 0.0;
    const bool show = argc > 2 && std::strcmp(argv[2], "view") == 0;
    const char* snapshot_path = argc > 3 ? argv[3] : nullptr;
    std::signal(SIGINT, on_signal);

    pl::PipelineConfig config;
    config.keep_image = show;
    config.blocking_sync = !(argc > 2 && std::strcmp(argv[2], "spin") == 0);
    const std::vector<std::string> names = edge_tracking::postprocess::load_class_names("models/coco_names.txt");

    // Sink state, only touched on the sink thread (and by main after stop() has joined it).
    view::TrailHistory trails(45, config.tracker.track_buffer);
    cv::Mat bgr;
    bool window_open = false;
    std::string status = "starting";
    std::mutex status_mutex;

    pl::Pipeline pipeline(config);
    auto on_result = [&](const pl::FrameResult& r) {
        if (!show) return;
        const cv::Mat nv12(r.height * 3 / 2, r.width, CV_8UC1, const_cast<uint8_t*>(r.nv12));
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);  // BT.601: slight color shift, fine for viewing
        trails.update(r.tracks);
        view::draw_tracks(bgr, r.tracks, trails, names);
        {
            std::lock_guard<std::mutex> lock(status_mutex);
            view::draw_status(bgr, status);
        }
        if (!window_open) {
            cv::namedWindow("edge-tracking pipeline", cv::WINDOW_AUTOSIZE);
            window_open = true;
        }
        cv::imshow("edge-tracking pipeline", bgr);
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) g_stop = true;
    };

    std::string error;
    if (!pipeline.start(on_result, &error)) {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    pipeline.telemetry().take_window();
    pipeline.telemetry().reset_totals();
    while (!g_stop && pipeline.running()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto window = pipeline.telemetry().take_window();
        const std::string line = edge_tracking::telemetry::format(window);
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        {
            std::lock_guard<std::mutex> lock(status_mutex);
            char text[160];
            std::snprintf(text, sizeof(text), "%.1f fps | infer %.1f ms | latency %.1f ms | dropped %ld",
                          window.rate(edge_tracking::telemetry::Counter::Results),
                          window.stage(edge_tracking::telemetry::Stage::Inference).mean_ms,
                          window.stage(edge_tracking::telemetry::Stage::Latency).mean_ms,
                          window.counter(edge_tracking::telemetry::Counter::CameraDropped) +
                              window.counter(edge_tracking::telemetry::Counter::IngestDropped) +
                              window.counter(edge_tracking::telemetry::Counter::SinkDropped));
            status = text;
        }
        if (run_seconds > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= run_seconds) {
            break;
        }
    }

    const std::string failure = pipeline.error();
    const auto totals = pipeline.telemetry().totals();
    pipeline.stop();
    if (!failure.empty()) std::fprintf(stderr, "pipeline error: %s\n", failure.c_str());
    print_totals(totals);

    if (show) {
        if (snapshot_path && !bgr.empty() && cv::imwrite(snapshot_path, bgr)) std::printf("saved %s\n", snapshot_path);
        cv::destroyAllWindows();
    }
    return failure.empty() ? 0 : 1;
}
