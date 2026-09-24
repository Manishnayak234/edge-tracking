#pragma once

#include "edge_tracking/frame/color_space.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define ET_HOST_DEVICE __host__ __device__
#else
#define ET_HOST_DEVICE
#endif

namespace edge_tracking::preprocess {

// Geometry of an aspect-preserving resize into a fixed network input, with the unused
// border padded (Ultralytics "letterbox"). Also used to map detections back to the frame.
struct LetterboxParams {
    int src_width = 0;
    int src_height = 0;
    int dst_width = 0;
    int dst_height = 0;
    int resized_width = 0;   // size of the image area inside dst
    int resized_height = 0;
    int pad_x = 0;           // left padding in dst pixels
    int pad_y = 0;           // top padding in dst pixels

    // Map a point from network-input pixels back to source-frame pixels (host or device).
    ET_HOST_DEVICE float to_src_x(float x) const {
        return (x - pad_x) * src_width / static_cast<float>(resized_width);
    }
    ET_HOST_DEVICE float to_src_y(float y) const {
        return (y - pad_y) * src_height / static_cast<float>(resized_height);
    }
};

// Same rounding as Ultralytics LetterBox (center padding).
inline LetterboxParams compute_letterbox(int src_width, int src_height, int dst_width, int dst_height) {
    LetterboxParams p;
    p.src_width = src_width;
    p.src_height = src_height;
    p.dst_width = dst_width;
    p.dst_height = dst_height;
    const float scale = std::min(static_cast<float>(dst_width) / src_width,
                                 static_cast<float>(dst_height) / src_height);
    p.resized_width = static_cast<int>(std::lround(src_width * scale));
    p.resized_height = static_cast<int>(std::lround(src_height * scale));
    p.pad_x = static_cast<int>(std::lround((dst_width - p.resized_width) / 2.0f - 0.1f));
    p.pad_y = static_cast<int>(std::lround((dst_height - p.resized_height) / 2.0f - 0.1f));
    return p;
}

// YOLOv8 input convention: RGB, values scaled to [0, 1], no mean/std, padding 114.
struct TensorFormat {
    float scale = 1.0f / 255.0f;
    float pad_value = 114.0f;  // in 0..255 units, before scaling
};

// Converts a pitch-linear NV12 frame in device memory into a letterboxed RGB float tensor,
// planar NCHW with N = 1 (dst holds 3 * dst_width * dst_height floats), in one kernel:
// YUV -> RGB per source pixel, then bilinear resize with OpenCV INTER_LINEAR sampling.
// Asynchronous on `stream`; returns launch errors only.
cudaError_t nv12_to_rgb_letterbox(const uint8_t* y, size_t y_pitch, const uint8_t* uv, size_t uv_pitch,
                                  ColorSpace color_space, const LetterboxParams& params,
                                  const TensorFormat& format, float* dst, cudaStream_t stream);

}  // namespace edge_tracking::preprocess
