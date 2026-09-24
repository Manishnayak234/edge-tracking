#include "edge_tracking/preprocess/letterbox.hpp"

namespace edge_tracking::preprocess {

namespace {

// RGB = y_scale * (Y - y_offset) + matrix * (U - 128, V - 128)
struct YuvToRgb {
    float y_offset;
    float y_scale;
    float rv;  // R from V
    float gu;  // G from U (subtracted)
    float gv;  // G from V (subtracted)
    float bu;  // B from U
};

YuvToRgb coefficients_for(ColorSpace cs) {
    switch (cs) {
        case ColorSpace::Bt601Full: return {0.0f, 1.0f, 1.402f, 0.344136f, 0.714136f, 1.772f};
        case ColorSpace::Bt709Limited: return {16.0f, 1.164383f, 1.792741f, 0.213249f, 0.532909f, 2.112402f};
        case ColorSpace::Bt709Full: return {0.0f, 1.0f, 1.5748f, 0.187324f, 0.468124f, 1.8556f};
        case ColorSpace::Bt601Limited:
        case ColorSpace::Unknown:  // most common for camera NV12
            break;
    }
    return {16.0f, 1.164383f, 1.596027f, 0.391762f, 0.812968f, 2.017232f};
}

struct Rgb {
    float r, g, b;
};

__device__ Rgb load_rgb(const uint8_t* y_plane, size_t y_pitch, const uint8_t* uv_plane, size_t uv_pitch,
                        int x, int y, const YuvToRgb& c) {
    // NV12: one interleaved U,V pair per 2x2 block of Y samples.
    const float luma = (y_plane[static_cast<size_t>(y) * y_pitch + x] - c.y_offset) * c.y_scale;
    const uint8_t* uv = uv_plane + static_cast<size_t>(y / 2) * uv_pitch + (x / 2) * 2;
    const float u = uv[0] - 128.0f;
    const float v = uv[1] - 128.0f;
    return {fminf(fmaxf(luma + c.rv * v, 0.0f), 255.0f),
            fminf(fmaxf(luma - c.gu * u - c.gv * v, 0.0f), 255.0f),
            fminf(fmaxf(luma + c.bu * u, 0.0f), 255.0f)};
}

__global__ void nv12_to_rgb_letterbox_kernel(const uint8_t* y_plane, size_t y_pitch, const uint8_t* uv_plane,
                                             size_t uv_pitch, YuvToRgb coeffs, LetterboxParams p,
                                             TensorFormat format, float* dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dst_width || y >= p.dst_height) return;

    const size_t plane_size = static_cast<size_t>(p.dst_width) * p.dst_height;
    const size_t idx = static_cast<size_t>(y) * p.dst_width + x;

    const int rx = x - p.pad_x;
    const int ry = y - p.pad_y;
    if (rx < 0 || ry < 0 || rx >= p.resized_width || ry >= p.resized_height) {
        const float pad = format.pad_value * format.scale;
        dst[idx] = pad;
        dst[idx + plane_size] = pad;
        dst[idx + 2 * plane_size] = pad;
        return;
    }

    // Source position with pixel centers at +0.5, as OpenCV INTER_LINEAR (negatives clamp to 0).
    const float sx = fmaxf((rx + 0.5f) * p.src_width / p.resized_width - 0.5f, 0.0f);
    const float sy = fmaxf((ry + 0.5f) * p.src_height / p.resized_height - 0.5f, 0.0f);
    const int x0 = min(static_cast<int>(sx), p.src_width - 1);
    const int y0 = min(static_cast<int>(sy), p.src_height - 1);
    const int x1 = min(x0 + 1, p.src_width - 1);
    const int y1 = min(y0 + 1, p.src_height - 1);
    const float fx = sx - x0;
    const float fy = sy - y0;

    const Rgb a = load_rgb(y_plane, y_pitch, uv_plane, uv_pitch, x0, y0, coeffs);
    const Rgb b = load_rgb(y_plane, y_pitch, uv_plane, uv_pitch, x1, y0, coeffs);
    const Rgb c = load_rgb(y_plane, y_pitch, uv_plane, uv_pitch, x0, y1, coeffs);
    const Rgb d = load_rgb(y_plane, y_pitch, uv_plane, uv_pitch, x1, y1, coeffs);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;
    dst[idx] = (a.r * w00 + b.r * w01 + c.r * w10 + d.r * w11) * format.scale;
    dst[idx + plane_size] = (a.g * w00 + b.g * w01 + c.g * w10 + d.g * w11) * format.scale;
    dst[idx + 2 * plane_size] = (a.b * w00 + b.b * w01 + c.b * w10 + d.b * w11) * format.scale;
}

}  // namespace

cudaError_t nv12_to_rgb_letterbox(const uint8_t* y, size_t y_pitch, const uint8_t* uv, size_t uv_pitch,
                                  ColorSpace color_space, const LetterboxParams& params,
                                  const TensorFormat& format, float* dst, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((params.dst_width + block.x - 1) / block.x, (params.dst_height + block.y - 1) / block.y);
    nv12_to_rgb_letterbox_kernel<<<grid, block, 0, stream>>>(y, y_pitch, uv, uv_pitch,
                                                             coefficients_for(color_space), params, format, dst);
    return cudaGetLastError();
}

}  // namespace edge_tracking::preprocess
