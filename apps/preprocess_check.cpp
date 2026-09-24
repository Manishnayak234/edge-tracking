// Checks and times the YOLOv8 preprocessing kernel on live camera frames.
//  - timing: GPU time of nv12_to_rgb_letterbox per frame (CUDA events)
//  - correctness: runs the kernel on the last frame with BT.601 limited coefficients (the
//    only ones OpenCV's NV12 conversion supports) and compares against the Ultralytics-style
//    OpenCV reference (cvtColor NV12->RGB, resize INTER_LINEAR, pad 114)
//  - writes the real tensor (camera's own color space) as an image so it can be inspected
// Usage: preprocess_check [num_frames] [output.png]   (defaults: 300, /tmp/et_letterbox.png)

#include "edge_tracking/camera/argus_gpu_camera.hpp"
#include "edge_tracking/preprocess/letterbox.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t status_ = (call);                                                \
        if (status_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_));   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

constexpr int kInputSize = 640;       // YOLOv8n input
constexpr int kWarmupFrames = 60;     // let auto-exposure settle

}  // namespace

int main(int argc, char** argv) {
    namespace cam = edge_tracking::camera;
    namespace pre = edge_tracking::preprocess;

    const long num_frames = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 300;
    const char* output_path = argc > 2 ? argv[2] : "/tmp/et_letterbox.png";

    cam::CameraConfig config;
    cam::ArgusGpuCamera camera(config);
    std::string error;
    if (!camera.start(&error)) {
        std::fprintf(stderr, "start failed: %s\n", error.c_str());
        return 1;
    }

    cam::GpuFrame frame;
    for (int i = 0; i < kWarmupFrames; ++i) {
        if (!camera.read(frame, std::chrono::milliseconds(5000))) {
            std::fprintf(stderr, "read failed: %s\n", camera.last_error().c_str());
            return 1;
        }
    }

    const pre::LetterboxParams lb = pre::compute_letterbox(frame.width, frame.height, kInputSize, kInputSize);
    const pre::TensorFormat format;
    std::printf("camera %dx%d, %s -> tensor 1x3x%dx%d, image area %dx%d at (%d, %d)\n", frame.width,
                frame.height, edge_tracking::to_string(frame.color_space), lb.dst_height, lb.dst_width,
                lb.resized_width, lb.resized_height, lb.pad_x, lb.pad_y);

    const size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    float* d_tensor = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tensor, 3 * plane * sizeof(float)));
    cudaStream_t stream;
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<float> times_ms;
    for (long i = 0; i < num_frames; ++i) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "read failed: %s\n", camera.last_error().c_str());
            return 1;
        }
        CUDA_CHECK(cudaEventRecord(begin, stream));
        CUDA_CHECK(pre::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch, frame.color_space,
                                              lb, format, d_tensor, stream));
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        times_ms.push_back(ms);
    }

    std::sort(times_ms.begin(), times_ms.end());
    double sum = 0.0;
    for (float t : times_ms) sum += t;
    std::printf("preprocess GPU time over %zu frames: mean %.3f ms, median %.3f ms, p99 %.3f ms, max %.3f ms\n",
                times_ms.size(), sum / times_ms.size(), times_ms[times_ms.size() / 2],
                times_ms[times_ms.size() * 99 / 100], times_ms.back());

    // Reference on the last frame (still held by the camera until the next read()).
    cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy2D(nv12.data, frame.width, frame.y, frame.y_pitch, frame.width, frame.height,
                            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(nv12.data + static_cast<size_t>(frame.width) * frame.height, frame.width, frame.uv,
                            frame.uv_pitch, frame.width, frame.height / 2, cudaMemcpyDeviceToHost));
    std::vector<float> tensor(3 * plane);
    CUDA_CHECK(cudaMemcpy(tensor.data(), d_tensor, tensor.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> tensor_601(3 * plane);
    CUDA_CHECK(pre::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch,
                                          edge_tracking::ColorSpace::Bt601Limited, lb, format, d_tensor, stream));
    CUDA_CHECK(cudaMemcpyAsync(tensor_601.data(), d_tensor, tensor_601.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cv::Mat rgb, resized, reference;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);  // OpenCV uses BT.601 limited range
    cv::resize(rgb, resized, cv::Size(lb.resized_width, lb.resized_height), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, reference, lb.pad_y, lb.dst_height - lb.resized_height - lb.pad_y, lb.pad_x,
                       lb.dst_width - lb.resized_width - lb.pad_x, cv::BORDER_CONSTANT,
                       cv::Scalar::all(format.pad_value));

    cv::Mat gpu_image(kInputSize, kInputSize, CV_8UC3);
    double max_diff = 0.0;
    double total_diff = 0.0;
    long over_1 = 0;
    double max_cs_diff = 0.0;
    for (int yy = 0; yy < kInputSize; ++yy) {
        for (int xx = 0; xx < kInputSize; ++xx) {
            const cv::Vec3b& ref = reference.at<cv::Vec3b>(yy, xx);
            cv::Vec3b& out = gpu_image.at<cv::Vec3b>(yy, xx);
            for (int c = 0; c < 3; ++c) {
                const size_t i = c * plane + static_cast<size_t>(yy) * kInputSize + xx;
                const double value_601 = tensor_601[i] / format.scale;
                const double diff = std::abs(value_601 - ref[c]);
                max_diff = std::max(max_diff, diff);
                total_diff += diff;
                if (diff > 1.0) ++over_1;
                const double value = tensor[i] / format.scale;
                max_cs_diff = std::max(max_cs_diff, std::abs(value - value_601));
                out[2 - c] = cv::saturate_cast<uint8_t>(value);  // RGB tensor -> BGR image
            }
        }
    }
    std::printf("BT.601 kernel vs OpenCV reference (0..255 units): max diff %.2f, mean diff %.3f, "
                "off by >1: %ld of %zu\n",
                max_diff, total_diff / (3.0 * plane), over_1, 3 * plane);
    std::printf("camera color space (%s) vs BT.601: max diff %.2f\n", edge_tracking::to_string(frame.color_space),
                max_cs_diff);

    cv::imwrite(output_path, gpu_image);
    std::printf("wrote %s\n", output_path);

    camera.stop();
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    cudaFree(d_tensor);
    return 0;
}
