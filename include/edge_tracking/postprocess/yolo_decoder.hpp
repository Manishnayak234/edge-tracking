#pragma once

#include "edge_tracking/preprocess/letterbox.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace edge_tracking::postprocess {

// Axis-aligned box in source-frame pixels (camera coordinates).
struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = -1;
};

// Defaults match Ultralytics predict().
struct DecoderConfig {
    float score_threshold = 0.25f;
    float iou_threshold = 0.7f;
    int max_detections = 300;
    bool class_agnostic = false;  // false: NMS only suppresses boxes of the same class
    int max_candidates = 4096;    // capacity for boxes passing the score threshold
};

// Turns the raw YOLOv8 output [4 + num_classes][num_candidates] into detections.
// GPU: per-candidate best class, score threshold, box conversion to frame pixels.
// CPU: NMS over the few surviving boxes.
class YoloDecoder {
public:
    YoloDecoder(int num_candidates, int num_classes, DecoderConfig config = {});
    ~YoloDecoder();

    YoloDecoder(const YoloDecoder&) = delete;
    YoloDecoder& operator=(const YoloDecoder&) = delete;

    // Decodes `output` (device memory) on `stream` and synchronizes the stream.
    // Returns detections sorted by score, highest first.
    std::vector<Detection> decode(const float* output, const preprocess::LetterboxParams& letterbox,
                                  cudaStream_t stream);

    // Candidates above the score threshold in the last decode(), before NMS.
    int last_candidate_count() const;
    // True if the last decode() had more candidates than max_candidates (extra ones dropped).
    bool last_overflowed() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Reads one class name per line; returns an empty list if the file cannot be read.
std::vector<std::string> load_class_names(const std::string& path);

}  // namespace edge_tracking::postprocess
