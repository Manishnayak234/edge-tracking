// Sweeps an SG90 servo on an Arduino for as long as a person is in the camera's view.
// Usage: servo_track [seconds] [device]
//   seconds:  0 = until Ctrl+C (default)
//   device:   Arduino serial port (default /dev/ttyACM0)
//   servo_track selftest [device]   sweeps for 5 s without the camera, to check the wiring
// Flash arduino/servo_sweep/servo_sweep.ino first. Run from the project root.

#include "edge_tracking/pipeline/pipeline.hpp"
#include "serial_port.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

constexpr int kPersonClassId = 0;      // first line of models/coco_names.txt
constexpr float kPersonScore = 0.40f;  // the tracker also keeps low-score boxes; only react to confident ones
constexpr int kFramesToStart = 5;      // ~80 ms at 60 fps, so one stray detection does not start the servo
constexpr int kFramesToStop = 10;     // ~0.5 s at 60 fps, so the sweep continues through a brief miss
constexpr auto kHeartbeat = std::chrono::milliseconds(500);  // repeat the state; the sketch stops without it

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int count_people(const edge_tracking::pipeline::FrameResult& result) {
    int people = 0;
    for (const auto& track : result.tracks) {
        if (track.class_id == kPersonClassId && track.score >= kPersonScore) ++people;
    }
    return people;
}

// Prints the Arduino's replies ("ready", "sweep", "idle", ...), one per line.
void drain(edge_tracking::apps::SerialPort& serial) {
    std::string text = serial.read_available();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    if (!text.empty()) std::printf("  arduino: %s\n", text.c_str());
}

bool open_arduino(edge_tracking::apps::SerialPort& serial, const char* device) {
    std::string error;
    if (!serial.open(device, &error)) {
        std::fprintf(stderr, "cannot open %s: %s\n", device, error.c_str());
        std::fprintf(stderr, "  plug the Arduino into the Jetson and check: ls /dev/ttyACM* /dev/ttyUSB*\n");
        std::fprintf(stderr, "  permission denied? add yourself to the dialout group, then log in again:\n");
        std::fprintf(stderr, "    sudo usermod -aG dialout $USER\n");
        return false;
    }
    // Opening the port pulls DTR low, which resets an Uno/Nano into its bootloader.
    std::printf("opened %s, waiting for the board to reboot\n", device);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    drain(serial);
    return true;
}

int selftest(const char* device) {
    edge_tracking::apps::SerialPort serial;
    if (!open_arduino(serial, device)) return 1;
    std::printf("sweeping for 5 s\n");
    for (int i = 0; i < 10 && !g_stop; ++i) {  // heartbeat, or the sketch stops after 3 s
        serial.write_char('S');
        std::this_thread::sleep_for(kHeartbeat);
        drain(serial);
    }
    serial.write_char('X');
    std::printf("centring\n");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    drain(serial);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    namespace pl = edge_tracking::pipeline;
    std::signal(SIGINT, on_signal);

    if (argc > 1 && std::strcmp(argv[1], "selftest") == 0) return selftest(argc > 2 ? argv[2] : "/dev/ttyACM0");

    const double run_seconds = argc > 1 ? std::strtod(argv[1], nullptr) : 0.0;
    const char* device = argc > 2 ? argv[2] : "/dev/ttyACM0";

    edge_tracking::apps::SerialPort serial;
    if (!open_arduino(serial, device)) return 1;

    // Sink-thread state: the callback owns the serial writes, so there is only one writer.
    int on_streak = 0;
    int off_streak = 0;
    bool sweeping = false;
    auto last_send = std::chrono::steady_clock::now() - kHeartbeat;
    std::atomic<int> people_now{0};
    std::atomic<bool> sweeping_now{false};

    pl::PipelineConfig config;
    pl::Pipeline pipeline(config);
    auto on_result = [&](const pl::FrameResult& result) {
        const int people = count_people(result);
        people_now = people;
        if (people > 0) {
            ++on_streak;
            off_streak = 0;
        } else {
            ++off_streak;
            on_streak = 0;
        }
        // Hysteresis: start quickly, stop slowly, so the servo does not flicker on and off.
        const bool want = sweeping ? off_streak <  : on_streak >= kFramesToStart;

        const auto now = std::chrono::steady_clock::now();
        if (want != sweeping || now - last_send >= kHeartbeat) {
            if (want != sweeping) std::printf("%s\n", want ? "person -> sweep" : "no person -> stop");
            sweeping = want;
            sweeping_now = want;
            last_send = now;
            if (!serial.write_char(want ? 'S' : 'X')) std::fprintf(stderr, "serial write failed\n");
        }
    };

    std::string error;
    if (!pipeline.start(on_result, &error)) {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }
    std::printf("running, Ctrl+C to stop\n");

    const auto start = std::chrono::steady_clock::now();
    pipeline.telemetry().take_window();
    while (!g_stop && pipeline.running()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto window = pipeline.telemetry().take_window();
        std::printf("people %d | servo %s | %.1f fps | latency %.1f ms\n", people_now.load(),
                    sweeping_now.load() ? "sweeping" : "idle",
                    window.rate(edge_tracking::telemetry::Counter::Results),
                    window.stage(edge_tracking::telemetry::Stage::Latency).mean_ms);
        drain(serial);
        std::fflush(stdout);
        if (run_seconds > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= run_seconds) {
            break;
        }
    }

    const std::string failure = pipeline.error();
    pipeline.stop();  // joins the sink thread, so main can use the port again
    serial.write_char('X');
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    drain(serial);
    if (!failure.empty()) {
        std::fprintf(stderr, "pipeline error: %s\n", failure.c_str());
        return 1;
    }
    return 0;
}
