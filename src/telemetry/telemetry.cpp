#include "edge_tracking/telemetry/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace edge_tracking::telemetry {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

Recorder::Recorder() {
    window_.start_s = now_s();
    totals_.start_s = window_.start_s;
}

void Recorder::add(Stage stage, double ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    window_.samples[static_cast<size_t>(stage)].push_back(static_cast<float>(ms));
    auto& total = totals_.samples[static_cast<size_t>(stage)];
    if (total.size() < kMaxTotalSamples) total.push_back(static_cast<float>(ms));
}

void Recorder::count(Counter counter, long n) {
    std::lock_guard<std::mutex> lock(mutex_);
    window_.counters[static_cast<size_t>(counter)] += n;
    totals_.counters[static_cast<size_t>(counter)] += n;
}

Snapshot Recorder::summarize(const Window& w, double now) {
    Snapshot s;
    s.window_s = now - w.start_s;
    s.counters = w.counters;
    for (size_t i = 0; i < w.samples.size(); ++i) {
        std::vector<float> v = w.samples[i];
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        double sum = 0.0;
        for (float x : v) sum += x;
        StageSummary& out = s.stages[i];
        out.count = static_cast<long>(v.size());
        out.mean_ms = sum / v.size();
        out.p50_ms = v[v.size() / 2];
        out.p99_ms = v[std::min(v.size() - 1, v.size() * 99 / 100)];
        out.max_ms = v.back();
    }
    return s;
}

Snapshot Recorder::take_window() {
    std::lock_guard<std::mutex> lock(mutex_);
    const double now = now_s();
    Snapshot s = summarize(window_, now);
    window_ = Window{};
    window_.start_s = now;
    return s;
}

Snapshot Recorder::totals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return summarize(totals_, now_s());
}

void Recorder::reset_totals() {
    std::lock_guard<std::mutex> lock(mutex_);
    totals_ = Window{};
    totals_.start_s = now_s();
}

std::string format(const Snapshot& s) {
    auto st = [&](Stage stage) { return s.stage(stage); };
    char line[512];
    std::snprintf(line, sizeof(line),
                  "in %5.1f fps, out %5.1f fps | capture %.2f | infer %.2f (p99 %.2f) | track %.3f | sink %.2f ms | "
                  "latency %.1f ms (p99 %.1f) | dropped: camera %ld, ingest %ld, sink %ld",
                  s.rate(Counter::CameraFrames), s.rate(Counter::Results), st(Stage::Capture).mean_ms,
                  st(Stage::Inference).mean_ms, st(Stage::Inference).p99_ms, st(Stage::Tracking).mean_ms,
                  st(Stage::Sink).mean_ms, st(Stage::Latency).mean_ms, st(Stage::Latency).p99_ms,
                  s.counter(Counter::CameraDropped), s.counter(Counter::IngestDropped),
                  s.counter(Counter::SinkDropped));
    return line;
}

}  // namespace edge_tracking::telemetry
