# Camera baseline

## Hardware

- Sensor: Sony IMX219
- Interface: MIPI CSI-2
- Device: `/dev/video0`
- Video-input driver: `tegra-video`

## Educational raw-capture path

- Interface: V4L2
- Format: `RG10` — 10-bit Bayer RGRG/GBGB
- Purpose: learn capture buffers, stride, raw formats, and camera diagnostics.

## Deployment capture path

- Interface: NVIDIA Argus through `nvarguscamerasrc`
- Sensor mode: 4
- Capture resolution: 1280 × 720
- Capture rate: 60 FPS
- Output format: NV12
- Memory: NVMM

## Validation

Captured 120 frames through:

```text
IMX219 → Argus ISP → NV12/NVMM → fakesink