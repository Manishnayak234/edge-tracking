#pragma once

#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace edge_tracking::telemetry {

// Durations measured per frame.
enum class Stage {
    Capture,    // frame arrived -> preprocessed (+ optional image copy), camera buffer free again
    Inference,  // TensorRT + decode + NMS
    Tracking,   // ByteTracker::update
    Sink,       // user callback
    Latency,    // camera capture time -> result handed to the sink (end to end)
    Count_
};

// Events counted per window.
enum class Counter {
    CameraFrames,    // frames delivered by the camera
    CameraDropped,   // frames the camera produced but we never received (timestamp gaps)
    IngestDropped,   // frames dropped because every buffer slot was busy
    SinkDropped,     // results dropped because the sink queue was full
    Results,         // results delivered to the sink
    Count_
};

struct StageSummary {
    long count = 0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

struct Snapshot {
    double window_s = 0.0;
    std::array<StageSummary, static_cast<size_t>(Stage::Count_)> stages{};
    std::array<long, static_cast<size_t>(Counter::Count_)> counters{};

    const StageSummary& stage(Stage s) const { return stages[static_cast<size_t>(s)]; }
    long counter(Counter c) const { return counters[static_cast<size_t>(c)]; }
    double rate(Counter c) const { return window_s > 0 ? counter(c) / window_s : 0.0; }
};

// Thread-safe collector. Stages report durations, the owner takes a snapshot per window
// (e.g. once a second), which also starts the next window.
class Recorder {
public:
    Recorder();

    void add(Stage stage, double ms);
    void count(Counter counter, long n = 1);

    // Summary since the previous take_window() (or construction), then resets.
    Snapshot take_window();
    // Summary since construction (or reset_totals()); does not reset anything. Keeps at most
    // kMaxTotalSamples durations per stage (~4.6 h at 60 fps); counters are always exact.
    Snapshot totals() const;
    void reset_totals();

    static constexpr size_t kMaxTotalSamples = 1'000'000;

private:
    struct Window {
        double start_s = 0.0;
        std::array<std::vector<float>, static_cast<size_t>(Stage::Count_)> samples;
        std::array<long, static_cast<size_t>(Counter::Count_)> counters{};
    };
    static Snapshot summarize(const Window& w, double now_s);

    mutable std::mutex mutex_;
    Window window_;
    Window totals_;
};

// One-line summary, e.g. for printing once per window.
std::string format(const Snapshot& s);

}  // namespace edge_tracking::telemetry
