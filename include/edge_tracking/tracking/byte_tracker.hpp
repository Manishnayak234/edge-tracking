#pragma once

#include "edge_tracking/postprocess/yolo_decoder.hpp"

#include <memory>
#include <vector>

namespace edge_tracking::tracking {

// Defaults match Ultralytics cfg/trackers/bytetrack.yaml. Detections should come from the
// decoder with score_threshold 0.1 (as Ultralytics track mode does), because the second
// association stage uses the low-score boxes between track_low_thresh and track_high_thresh.
struct TrackerConfig {
    float track_high_thresh = 0.25f;  // first association uses detections with score >= this
    float track_low_thresh = 0.1f;    // second association uses scores in (low, high)
    float new_track_thresh = 0.25f;   // unmatched detections start a track only above this
    int track_buffer = 30;            // frames a lost track is kept (0.5 s at 60 fps)
    float match_thresh = 0.8f;        // max association cost (1 - IoU * score) in the first stage
    bool fuse_score = true;           // weight IoU by detection score in first/unconfirmed stages
};

struct Track {
    int id = 0;                  // stable across frames, starts at 1
    float x1 = 0.0f, y1 = 0.0f;  // Kalman-filtered box in frame pixels
    float x2 = 0.0f, y2 = 0.0f;
    float score = 0.0f;          // score of the last matched detection
    int class_id = -1;           // class of the last matched detection
    int start_frame = 0;         // frame the track was started in (frames count from 1)
};

// C++ port of Ultralytics BYTETracker (ByteTrack: two-stage IoU association with a
// Kalman motion model). Association ignores class, as in Ultralytics.
class ByteTracker {
public:
    explicit ByteTracker(TrackerConfig config = {});
    ~ByteTracker();

    ByteTracker(const ByteTracker&) = delete;
    ByteTracker& operator=(const ByteTracker&) = delete;

    // Feeds one frame of detections; returns the confirmed tracks visible in this frame.
    std::vector<Track> update(const std::vector<postprocess::Detection>& detections);

    // Forgets all tracks and restarts ids at 1.
    void reset();

    int frame_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_tracking::tracking
