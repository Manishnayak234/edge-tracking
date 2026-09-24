#include "edge_tracking/postprocess/yolo_decoder.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace edge_tracking::postprocess {

namespace {

// One thread per candidate: best class, threshold, cx/cy/w/h in network pixels ->
// x1/y1/x2/y2 in frame pixels. Boxes are not clipped yet: the letterbox mapping is a
// uniform scale plus shift, so IoU (and NMS) is the same as in network coordinates,
// matching Ultralytics, which clips after NMS.
__global__ void decode_kernel(const float* output, int num_candidates, int num_classes, float score_threshold,
                              preprocess::LetterboxParams lb, int capacity, int* count, Detection* candidates) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_candidates) return;

    float best = output[static_cast<size_t>(4) * num_candidates + i];
    int best_class = 0;
    for (int c = 1; c < num_classes; ++c) {
        const float s = output[static_cast<size_t>(4 + c) * num_candidates + i];
        if (s > best) {
            best = s;
            best_class = c;
        }
    }
    if (best <= score_threshold) return;

    const int slot = atomicAdd(count, 1);
    if (slot >= capacity) return;  // counted, so the host can report the overflow

    const float cx = output[i];
    const float cy = output[num_candidates + i];
    const float half_w = output[2 * num_candidates + i] * 0.5f;
    const float half_h = output[3 * num_candidates + i] * 0.5f;
    Detection d;
    d.x1 = lb.to_src_x(cx - half_w);
    d.y1 = lb.to_src_y(cy - half_h);
    d.x2 = lb.to_src_x(cx + half_w);
    d.y2 = lb.to_src_y(cy + half_h);
    d.score = best;
    d.class_id = best_class;
    candidates[slot] = d;
}

float iou(const Detection& a, const Detection& b) {
    const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    const float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    if (w <= 0.0f || h <= 0.0f) return 0.0f;
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

}  // namespace

struct YoloDecoder::Impl {
    int num_candidates = 0;
    int num_classes = 0;
    DecoderConfig config;

    // Candidate count and list on the device, with pinned host copies for fast readback.
    int* d_count = nullptr;
    Detection* d_candidates = nullptr;
    int* h_count = nullptr;
    Detection* h_candidates = nullptr;

    int last_count = 0;
};

YoloDecoder::YoloDecoder(int num_candidates, int num_classes, DecoderConfig config)
    : impl_(std::make_unique<Impl>()) {
    Impl& s = *impl_;
    s.num_candidates = num_candidates;
    s.num_classes = num_classes;
    s.config = config;
    check(cudaMalloc(&s.d_count, sizeof(int)), "cudaMalloc");
    check(cudaMalloc(&s.d_candidates, sizeof(Detection) * config.max_candidates), "cudaMalloc");
    check(cudaMallocHost(&s.h_count, sizeof(int)), "cudaMallocHost");
    check(cudaMallocHost(&s.h_candidates, sizeof(Detection) * config.max_candidates), "cudaMallocHost");
}

YoloDecoder::~YoloDecoder() {
    Impl& s = *impl_;
    cudaFree(s.d_count);
    cudaFree(s.d_candidates);
    cudaFreeHost(s.h_count);
    cudaFreeHost(s.h_candidates);
}

std::vector<Detection> YoloDecoder::decode(const float* output, const preprocess::LetterboxParams& letterbox,
                                           cudaStream_t stream) {
    Impl& s = *impl_;
    check(cudaMemsetAsync(s.d_count, 0, sizeof(int), stream), "cudaMemsetAsync");
    const int block = 256;
    decode_kernel<<<(s.num_candidates + block - 1) / block, block, 0, stream>>>(
        output, s.num_candidates, s.num_classes, s.config.score_threshold, letterbox, s.config.max_candidates,
        s.d_count, s.d_candidates);
    check(cudaGetLastError(), "decode_kernel");
    check(cudaMemcpyAsync(s.h_count, s.d_count, sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    s.last_count = *s.h_count;
    const int n = std::min(s.last_count, s.config.max_candidates);
    if (n > 0) {
        check(cudaMemcpyAsync(s.h_candidates, s.d_candidates, sizeof(Detection) * n, cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    // Greedy NMS, highest score first. Sorting also makes the result independent of the
    // order in which GPU threads claimed slots (except for exactly equal scores).
    std::vector<Detection> candidates(s.h_candidates, s.h_candidates + n);
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const Detection& c : candidates) {
        bool suppressed = false;
        for (const Detection& k : kept) {
            if ((s.config.class_agnostic || k.class_id == c.class_id) && iou(k, c) > s.config.iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        kept.push_back(c);
        if (static_cast<int>(kept.size()) >= s.config.max_detections) break;
    }

    // Clip to the frame after NMS (as Ultralytics does).
    const float w = static_cast<float>(letterbox.src_width);
    const float h = static_cast<float>(letterbox.src_height);
    for (Detection& d : kept) {
        d.x1 = std::clamp(d.x1, 0.0f, w);
        d.x2 = std::clamp(d.x2, 0.0f, w);
        d.y1 = std::clamp(d.y1, 0.0f, h);
        d.y2 = std::clamp(d.y2, 0.0f, h);
    }
    return kept;
}

int YoloDecoder::last_candidate_count() const { return impl_->last_count; }
bool YoloDecoder::last_overflowed() const { return impl_->last_count > impl_->config.max_candidates; }

std::vector<std::string> load_class_names(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    return names;
}

}  // namespace edge_tracking::postprocess
