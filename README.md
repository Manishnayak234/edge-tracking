# edge-tracking

Real-time camera object detection and multi-object tracking on an NVIDIA Jetson, in C++/CUDA.

A CSI camera feeds YOLOv8n (TensorRT, FP16) and ByteTrack. Frames stay in GPU memory from the camera to the detector, and the stages run in separate threads. On a Jetson Orin Nano Super it runs at **60 fps (1280×720)**, and tracks are ready **~10–15 ms after capture**.

Every stage is checked against a reference implementation (OpenCV, ONNX Runtime, Ultralytics), and the tools for those checks are in this repository (see [Verification](#verification)).

```
IMX219 ──► Argus ISP ──► VIC (pitch-linear) ──► CUDA preprocess ──► TensorRT YOLOv8n ──► decode + NMS ──► ByteTrack ──► your callback
          nvarguscamerasrc   nvvidconv (NVMM)     letterbox 640×640     FP16, CUDA graph    GPU + CPU         CPU
```

## Contents

- [Hardware and software](#hardware-and-software)
- [Project layout](#project-layout)
- [Build](#build)
- [Model setup](#model-setup)
- [Running](#running)
- [Using the pipeline in code](#using-the-pipeline-in-code)
- [Architecture](#architecture)
- [Performance](#performance)
- [Verification](#verification)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Hardware and software

Developed and tested on:

| | |
|---|---|
| Board | Jetson Orin Nano Engineering Reference Developer Kit Super |
| Power mode | MAXN_SUPER |
| L4T / JetPack | R36.4.7 (JetPack 6.2 series) |
| CUDA | 12.6 |
| TensorRT | 10.3 |
| cuDNN | 9.3 |
| OpenCV | 4.8 (NVIDIA build, no CUDA modules), only for checks and viewers |
| Camera | Sony IMX219 on CSI (`/dev/video0`), sensor mode 4: 1280×720 @ 60 fps |
| Build tools | CMake ≥ 3.18 (tested 3.22), GCC 11 |

The build also needs the GStreamer development packages (`gstreamer-1.0`, `gstreamer-app-1.0`, `gstreamer-video-1.0`) and the Jetson Multimedia API headers in `/usr/src/jetson_multimedia_api` (for `NvBufSurface`).

Exporting the model needs a separate machine (or venv) with Python and `ultralytics`, because PyTorch is not installed on the Jetson (see [Model setup](#model-setup)).

## Project layout

```
include/edge_tracking/    public headers, one folder per module
src/
  camera/                 ArgusCamera (CPU copy) and ArgusGpuCamera (zero-copy NVMM -> CUDA)
  preprocess/             NV12 -> letterboxed RGB tensor, one CUDA kernel
  inference/              TrtEngine: TensorRT runtime with CUDA-graph replay
  postprocess/            YoloDecoder: GPU decode + score filter, CPU NMS
  tracking/               ByteTracker (port of Ultralytics), Kalman filter, assignment solver
  pipeline/               threaded Pipeline: capture / inference+tracking / sink
  telemetry/              per-stage timings, counters, end-to-end latency
apps/                     benchmarks, checks and viewers (see below)
scripts/                  on_display.sh, preview.sh
tools/                    bytetrack_reference.py (runs on the PC, compares with Ultralytics)
models/                   ONNX model, class names, reference data (the TensorRT engine is built here, not committed)
docs/                     camera baseline notes and captures
tracks/                   detection/track files from the tracker checks
configs/ tests/ export/ results/   reserved, currently empty
```

Each module is a separate CMake library (`et_camera`, `et_preprocess`, `et_inference`, `et_postprocess`, `et_tracking`, `et_telemetry`, `et_pipeline`), so an application links only the modules it uses.

## Build

On the Jetson:

```bash
cd ~/edge-tracking
cmake -S . -B build            # Release by default; CUDA arch 87 (Orin)
cmake --build build -j6
```

Notes:

- `nvcc` is picked up from `/usr/local/cuda/bin/nvcc` even when it is not on `PATH`.
- OpenCV is found through its CMake config and **must be 4.8**. Ubuntu's OpenCV 4.5 is installed too, and NVIDIA's `opencv4.pc` has a wrong `/usr/local` prefix, so pkg-config is not used. If OpenCV 4.8 is missing, the libraries and the benchmark/check apps still build, but `preprocess_check`, the viewers and `pipeline_run` (which has a view mode) are skipped.

## Model setup

The engine file is specific to the Jetson and TensorRT version, so build it on the target. The repository includes `models/yolov8n.onnx` and `models/coco_names.txt` but not the engine, so after cloning you can go straight to step 3.

**1. Export ONNX** (on a PC with Python):

```bash
python3 -m venv yolo-venv
yolo-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
yolo-venv/bin/pip install ultralytics onnx onnxslim onnxruntime lap
yolo-venv/bin/yolo export model=yolov8n.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False batch=1
```

This gives `yolov8n.onnx`: input `images` [1,3,640,640], output `output0` [1,84,8400] (4 box values + 80 COCO class scores per candidate). Tested with ultralytics 8.4.160.

**2. Copy it to the Jetson** into `models/`, together with `coco_names.txt` (one class name per line, in class-id order).

**3. Build the FP16 engine** on the Jetson (takes ~7 minutes on the Orin Nano):

```bash
cd ~/edge-tracking/models
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine \
    --fp16 --memPoolSize=workspace:1024
```

The apps expect `models/yolov8n_fp16.engine` and `models/coco_names.txt` and must be **run from the project root**.

## Running

The camera can only be used by one program at a time. Stop a running preview or viewer before starting another app (see [Troubleshooting](#troubleshooting)).

### Main application

```bash
./build/pipeline_run 30                                  # 30 s headless, telemetry every second
./build/pipeline_run 0                                   # until Ctrl+C
./scripts/on_display.sh ./build/pipeline_run 0 view      # live window with tracks (q / Esc to quit)
./scripts/on_display.sh ./build/pipeline_run 20 view /tmp/last.png   # save the last frame on exit
```

Arguments: `pipeline_run [seconds] [view|headless|spin] [snapshot.png]`. `spin` runs headless with CUDA spin-waiting instead of blocking sync, for comparison.

Example telemetry line:

```
in  60.0 fps, out  60.0 fps | capture 0.88 | infer 9.12 (p99 9.29) | track 0.024 | sink 0.00 ms | latency 12.8 ms (p99 13.1) | dropped: camera 0, ingest 0, sink 0
```

### Servo output (Arduino)

`servo_track` runs the same pipeline headless and drives an SG90 servo on an Arduino over USB
serial: while a person (COCO class 0) is in view the servo sweeps back and forth, otherwise it
centres and releases. Flash `arduino/servo_sweep/servo_sweep.ino` first (servo signal on D9).

```bash
./build/servo_track selftest            # sweep for 5 s with no camera, to check the wiring
./build/servo_track 0                   # until Ctrl+C, default device /dev/ttyACM0
./build/servo_track 60 /dev/ttyUSB0     # 60 s on a clone board with a CH340/FTDI chip
```

The Jetson sends one character, `S` (sweep) or `X` (stop), on every change and again every 500 ms.
The sketch stops the sweep if 3 s pass with no command, so the servo cannot keep running when the
app exits or the cable is pulled. Detection is debounced: 5 frames with a person (~80 ms) start the
sweep, 90 frames without one (~1.5 s) stop it, so a momentary miss does not interrupt the motion.
Reading the serial port needs membership of the `dialout` group (`sudo usermod -aG dialout $USER`,
then log in again).

### Showing windows on the Jetson screen over SSH

`scripts/on_display.sh <command>` runs any command with its windows on the Jetson's own monitor. It finds the logged-in desktop's X display. **Someone has to be logged in on the Jetson desktop**: the login screen cannot be used over SSH.

```bash
./scripts/preview.sh          # plain live camera preview (GStreamer -> nv3dsink), Ctrl+C to stop
./scripts/preview.sh 10       # 10 seconds
```

### All apps

| App | What it does |
|---|---|
| `camera_fps [frames]` | CPU-copy capture benchmark: fps, frame gaps, drops |
| `camera_gpu_fps [frames]` | zero-copy capture benchmark; checks GPU pixels against a CPU copy |
| `preprocess_check [frames] [out.png]` | preprocess kernel timing; compares with OpenCV; writes the tensor as an image |
| `infer_check [engine] [frames]` | TensorRT output vs ONNX Runtime reference, then live stage timing |
| `postprocess_check [engine] [frames]` | decode + NMS vs Ultralytics references, then live timing |
| `track_check record <frames> <dets.txt>` | record live detections (score ≥ 0.1) to a file |
| `track_check replay <dets.txt> <tracks.txt>` | run ByteTracker over a detection file |
| `detect_view [seconds] [engine] [snap.png]` | single-loop viewer, detections only (s = snapshot) |
| `track_view [seconds] [engine] [snap.png]` | single-loop viewer with track ids and trails (s = snapshot) |
| `pipeline_run [seconds] [view\|headless\|spin] [snap.png]` | the threaded pipeline (main application) |
| `servo_track [seconds] [device]` | sweeps an Arduino servo while a person is in view (`selftest` checks the wiring) |

The viewers copy each frame to the CPU and draw with OpenCV. They are for debugging; the detection path itself never touches pixels on the CPU. They convert the image for display with OpenCV's BT.601 formula, so colours in the window are slightly off; the detector's input uses the camera's real colour space (BT.709).

## Using the pipeline in code

```cpp
#include "edge_tracking/pipeline/pipeline.hpp"

namespace pl = edge_tracking::pipeline;

pl::PipelineConfig config;               // defaults: 1280x720@60, models/yolov8n_fp16.engine
pl::Pipeline pipeline(config);

std::string error;
bool ok = pipeline.start([](const pl::FrameResult& r) {
    // Runs on the sink thread, once per processed frame. `r` is valid only during the call.
    for (const auto& t : r.tracks) {
        // t.id, t.class_id, t.score, t.x1, t.y1, t.x2, t.y2 (camera pixels)
    }
}, &error);

// ... later, e.g. once per second:
auto window = pipeline.telemetry().take_window();
std::printf("%s\n", edge_tracking::telemetry::format(window).c_str());

pipeline.stop();
```

Link against `et_pipeline`. `FrameResult` also contains the raw `detections`, `frame_index`, `capture_time_ns` and, with `config.keep_image = true`, a host copy of the NV12 image (`nv12`).

## Architecture

### Threads and buffers

```
camera ─► [et-capture]  read frame, GPU preprocess into a free slot (~1 ms), release camera buffer
              │  slot pool (4 slots); if none is free, the new frame is dropped (bounded latency)
          [et-infer]    TensorRT (CUDA graph) + decode/NMS + ByteTrack, in frame order
              │  sink queue (2); if full, the result is dropped
          [et-sink]     user callback
```

- **Why capture and preprocessing share a thread:** the camera buffer is only valid until the next `read()`, so preprocessing must finish before the next frame is read. Preprocessing takes about 0.2 ms, and afterwards the frame lives in its own slot.
- **Why tracking shares the inference thread:** it takes ~30 µs and must see frames in order, so a separate thread would only add a hand-off.
- **Why the sink has its own thread:** a slow consumer, such as the OpenCV viewer at ~7 ms per frame, never stalls detection.
- Threads are named `et-capture`, `et-infer` and `et-sink`, so they show up in `top -H` and `htop`.

### Camera (`src/camera`)

`ArgusGpuCamera` uses the pipeline `nvarguscamerasrc → nvvidconv bl-output=false output-buffers=8 → appsink` with NVMM buffers throughout:

- **Block-linear conversion:** Argus outputs block-linear (tiled) surfaces. `nvvidconv` converts them to pitch-linear on the **VIC** hardware converter, with no CPU copy and no GPU compute.
- **CUDA mapping:** each `NvBufSurface` is mapped into CUDA through EGL, once per pool buffer. The mapping is cached and reused, because mapping every frame cost ~0.5 s of CPU per 600 frames.
- **Pool size:** `output-buffers=8`, because the default 4 starved the pool and dropped frames.
- **Frame metadata:** each frame reports its colour space from the buffer's `colorFormat` (BT.709 limited for this camera) and its capture time on `steady_clock`, which is used for end-to-end latency.

`ArgusCamera` is the simpler CPU-copy version, kept as a baseline.

### Preprocessing (`src/preprocess`)

A single CUDA kernel goes from NV12 to a `1×3×640×640` float RGB tensor in [0, 1], matching Ultralytics:

- **Letterbox:** the image is scaled to 640×360 and padded with grey (114) to 640×640.
- **Resize:** bilinear, with the same pixel mapping as OpenCV `INTER_LINEAR`.
- **Colour:** YUV→RGB coefficients follow the frame's colour space.
- **Mapping back:** `LetterboxParams::to_src_x/y` map network coordinates back to camera pixels, on the host or the GPU.

### Inference (`src/inference`)

`TrtEngine` loads the serialized engine and checks it has one float32 input and one float32 output with fixed shapes.

`enqueue()` records TensorRT's kernel launches as a **CUDA graph** the first time it sees an (input, output, stream) combination, and replays it after that. This cut the inference thread's CPU from 25% to 3% of a core; launching the kernels one by one cost 3.6 ms of CPU per frame. The engine keeps at most 16 graphs, and it falls back to direct launches if recording fails.

### Postprocessing (`src/postprocess`)

`YoloDecoder` works in two parts:

- **GPU:** one thread per candidate (8400) picks its best class, applies the score threshold, and converts the box to camera pixels.
- **CPU:** class-aware greedy NMS runs on the survivors, which are few.

Defaults match Ultralytics `predict()`: score 0.25, IoU 0.7, at most 300 detections, boxes clipped after NMS. The pipeline uses score 0.1, as Ultralytics' tracking mode does, because ByteTrack needs the low-score boxes.

### Tracking (`src/tracking`)

`ByteTracker` is a port of Ultralytics `BYTETracker` with the `bytetrack.yaml` defaults:

| Setting | Value |
|---|---|
| High / low / new-track thresholds | 0.25 / 0.1 / 0.25 |
| `track_buffer` | 30 frames (0.5 s at 60 fps) |
| `match_thresh` | 0.8 |
| `fuse_score` | on |

It has three parts:

- `KalmanFilterXYAH`: constant velocity over box centre, aspect ratio and height.
- `linear_assignment`: a Hungarian solver with the same result as `lap.lapjv(extend_cost=True, cost_limit=...)`.
- The ByteTrack update itself: two-stage association, confirmation of new tracks, the lost-track buffer, and duplicate removal.

Association ignores class, as in Ultralytics.

**About track IDs:** every new tentative track takes an ID, even if it's never confirmed. Shown IDs therefore have gaps (#1, #2, #48, …). Ultralytics behaves the same way.

### Telemetry (`src/telemetry`)

`Recorder` is thread-safe:

- **Stage timings:** capture, inference, tracking, sink, and end-to-end latency (capture time → result handed to the sink).
- **Counters:** camera frames, frames dropped by the camera (timestamp gaps), at ingest (no free slot) and at the sink (queue full), and results delivered.

`take_window()` summarises and resets a window; `totals()` covers the whole run. Totals keep at most 1,000,000 samples per stage, about 4.6 h at 60 fps; the counters are always exact.

## Performance

Measured on the Jetson Orin Nano Super, MAXN_SUPER, default clock governors, 1280×720 @ 60 fps, 20 s runs.

| | Result |
|---|---|
| Camera capture (zero-copy) | 60 fps, 0–1 dropped frames per 600 |
| Preprocessing kernel | 0.23–0.28 ms |
| TensorRT YOLOv8n FP16 | 3.96 ms at full GPU clock (trtexec); 8–11 ms in the pipeline (see below) |
| Decode + NMS | ~0.35 ms |
| ByteTrack update | 7–9 µs replayed, ~30 µs live |
| Pipeline output | 59.3 fps (the only drops were 1–3 frames in the first second, during startup) |
| Capture → tracks latency | 12.5–15 ms mean, p99 ~16 ms (headless); ~10 ms with the viewer running |
| Process CPU | ~40% of one core; ~33% of that is NVIDIA's Argus camera plugin |

**GPU clock matters most.** The GPU governor (`nvhost_podgov`) keeps the GPU at **306–408 MHz** of its 1020 MHz maximum, because the load is light, and inference time follows that choice. For the lowest and most stable latency, lock the clocks with `sudo jetson_clocks`. This uses more power, and the setting resets on reboot.

## Verification

Each stage was checked against an independent reference. The data files are in `models/` and `tracks/`.

| Stage | Reference | Result |
|---|---|---|
| Zero-copy capture | CPU copy of the same GPU frame | pixel sums match on all checks |
| Preprocessing | OpenCV `cvtColor` + `resize INTER_LINEAR` + pad 114 (Ultralytics' method) | max diff 0.98 / 255, none > 1 |
| TensorRT FP16 | ONNX Runtime FP32 on the same tensor | same class for all confident candidates, box diff ≤ 0.22 px |
| Decode + NMS | Ultralytics `non_max_suppression` (camera frame and `bus.jpg`) | identical detections; FP16 end to end IoU ≥ 0.998 |
| ByteTrack | Ultralytics `BYTETracker` on a synthetic sequence (400 frames) and a live recording (600 frames) | identical ids in every frame, box diff ≤ 0.0001 px |

To repeat the tracker comparison (the Python part runs on the PC with `ultralytics` and `lap`):

```bash
# Jetson: record live detections and track them in C++
./build/track_check record 600 tracks/live_dets.txt
./build/track_check replay tracks/live_dets.txt tracks/live_tracks_cpp.txt

# PC: copy both files over, then
python tools/bytetrack_reference.py track live_dets.txt live_tracks_ultra.txt
python tools/bytetrack_reference.py compare live_tracks_ultra.txt live_tracks_cpp.txt

# synthetic sequence covering crossings, occlusion, low-score frames, false positives:
python tools/bytetrack_reference.py synth synth_dets.txt    # then replay on the Jetson and compare
```

## Configuration

Everything is set in code through config structs with documented defaults:

| Struct | Main fields |
|---|---|
| `camera::CameraConfig` | `sensor_id`, `sensor_mode` (4), `width`, `height`, `fps` |
| `postprocess::DecoderConfig` | `score_threshold`, `iou_threshold`, `max_detections`, `class_agnostic` |
| `tracking::TrackerConfig` | the ByteTrack thresholds, `track_buffer`, `match_thresh`, `fuse_score` |
| `pipeline::PipelineConfig` | all of the above + `engine_path`, `num_slots` (4), `sink_queue_capacity` (2), `keep_image`, `blocking_sync` |

IMX219 sensor modes are listed in `docs/camera_modes.txt`. Changing the camera size or rate means changing `sensor_mode` together with `width`, `height` and `fps`.

## Troubleshooting

**`Failed to create CaptureSession` / `timed out waiting for a frame`.** Another program is using the camera. Find it and stop it:

```bash
ps -eo pid,etime,args | grep -E "gst-launch|preview|_view|_check|pipeline_run|camera_" | grep -v grep
```

If nothing is running and it still fails, restart the camera service: `sudo systemctl restart nvargus-daemon`.

**`Cannot open display` / `Authorization required`.** Log in on the Jetson desktop, then start the app through `scripts/on_display.sh`.

**`cannot deserialize engine`.** The engine was built with a different TensorRT version or on a different device. Rebuild it with `trtexec` on this Jetson (see [Model setup](#model-setup)).

**`CUDA graph capture failed, using direct enqueue`.** Inference still works correctly, just with higher CPU use (~25% of a core for the inference thread).

**Inference slower than expected.** Check the GPU clock while the pipeline runs:

```bash
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq
```

See [Performance](#performance) for why it stays low, and `jetson_clocks`.

**Wrong-looking classes.** YOLOv8n uses the 80 COCO classes; objects outside them get the closest class (e.g. a white wardrobe shows up as "refrigerator"). Fix this by fine-tuning on your own data, then re-export and rebuild the engine.

## License

[AGPL-3.0](LICENSE). The tracker is a port of Ultralytics' `BYTETracker`, and the detector is Ultralytics YOLOv8n; both are AGPL-3.0.
