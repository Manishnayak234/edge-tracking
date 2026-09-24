#include "edge_tracking/tracking/byte_tracker.hpp"

#include "kalman_filter.hpp"
#include "linear_assignment.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <set>

namespace edge_tracking::tracking {

namespace {

enum class TrackState { New, Tracked, Lost, Removed };

// One detection or track (as in Ultralytics, a detection becomes a track by activation).
struct STrack {
    int track_id = 0;
    TrackState state = TrackState::New;
    bool is_activated = false;
    float score = 0.0f;
    int class_id = -1;
    int frame_id = 0;  // last frame this track was updated in
    int start_frame = 0;
    int tracklet_len = 0;

    std::array<float, 4> det_tlwh{};  // detection box (left, top, w, h) until the filter starts
    bool has_state = false;
    Vec8 mean{};
    Mat8 covariance{};

    // (left, top, w, h): the filtered state once the track exists, else the detection box.
    std::array<double, 4> tlwh() const {
        if (!has_state) return {det_tlwh[0], det_tlwh[1], det_tlwh[2], det_tlwh[3]};
        const double w = mean[2] * mean[3];
        const double h = mean[3];
        return {mean[0] - w / 2, mean[1] - h / 2, w, h};
    }

    std::array<float, 4> xyxy() const {
        const auto t = tlwh();
        return {static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[0] + t[2]),
                static_cast<float>(t[1] + t[3])};
    }
};

using TrackPtr = std::shared_ptr<STrack>;

Vec4 to_xyah(const std::array<float, 4>& tlwh) {
    // In float like Ultralytics (the detection box is float32 there), then widened.
    const float cx = tlwh[0] + tlwh[2] / 2;
    const float cy = tlwh[1] + tlwh[3] / 2;
    const float a = tlwh[2] / tlwh[3];
    return {cx, cy, a, tlwh[3]};
}

// 1 - IoU for every (track, detection) pair, row-major. float, eps 1e-7, as Ultralytics.
std::vector<float> iou_distance(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b) {
    std::vector<float> dist(a.size() * b.size());
    std::vector<std::array<float, 4>> bb(b.size());
    for (size_t j = 0; j < b.size(); ++j) bb[j] = b[j]->xyxy();
    for (size_t i = 0; i < a.size(); ++i) {
        const auto p = a[i]->xyxy();
        const float area_p = (p[2] - p[0]) * (p[3] - p[1]);
        for (size_t j = 0; j < b.size(); ++j) {
            const auto& q = bb[j];
            const float iw = std::max(std::min(p[2], q[2]) - std::max(p[0], q[0]), 0.0f);
            const float ih = std::max(std::min(p[3], q[3]) - std::max(p[1], q[1]), 0.0f);
            const float inter = iw * ih;
            const float area_q = (q[2] - q[0]) * (q[3] - q[1]);
            dist[i * b.size() + j] = 1.0f - inter / (area_q + area_p - inter + 1e-7f);
        }
    }
    return dist;
}

// cost = 1 - IoU * detection score
void fuse_score(std::vector<float>& dist, size_t rows, const std::vector<TrackPtr>& detections) {
    const size_t cols = detections.size();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float& d = dist[i * cols + j];
            d = 1.0f - (1.0f - d) * detections[j]->score;
        }
    }
}

// a + tracks of b whose id is not in a (order preserved).
std::vector<TrackPtr> joint(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b) {
    std::set<int> ids;
    for (const auto& t : a) ids.insert(t->track_id);
    std::vector<TrackPtr> out = a;
    for (const auto& t : b) {
        if (!ids.count(t->track_id)) out.push_back(t);
    }
    return out;
}

// a without tracks whose id is in `ids`.
std::vector<TrackPtr> without_ids(const std::vector<TrackPtr>& a, const std::set<int>& ids) {
    std::vector<TrackPtr> out;
    for (const auto& t : a) {
        if (!ids.count(t->track_id)) out.push_back(t);
    }
    return out;
}

std::set<int> ids_of(const std::vector<TrackPtr>& tracks) {
    std::set<int> ids;
    for (const auto& t : tracks) ids.insert(t->track_id);
    return ids;
}

}  // namespace

struct ByteTracker::Impl {
    TrackerConfig config;
    KalmanFilterXYAH kalman;
    int frame_id = 0;
    int next_id = 0;

    std::vector<TrackPtr> tracked;
    std::vector<TrackPtr> lost;
    std::deque<int> removed_ids;  // most recent 1000, as Ultralytics' removed_stracks buffer

    void activate(STrack& t) {
        t.track_id = ++next_id;
        kalman.initiate(to_xyah(t.det_tlwh), t.mean, t.covariance);
        t.has_state = true;
        t.tracklet_len = 0;
        t.state = TrackState::Tracked;
        if (frame_id == 1) t.is_activated = true;  // later tracks need a second match first
        t.frame_id = frame_id;
        t.start_frame = frame_id;
    }

    void update_track(STrack& t, const STrack& det) {
        t.frame_id = frame_id;
        ++t.tracklet_len;
        kalman.update(t.mean, t.covariance, to_xyah(det.det_tlwh));
        t.state = TrackState::Tracked;
        t.is_activated = true;
        t.score = det.score;
        t.class_id = det.class_id;
    }

    void reactivate(STrack& t, const STrack& det) {
        kalman.update(t.mean, t.covariance, to_xyah(det.det_tlwh));
        t.tracklet_len = 0;
        t.state = TrackState::Tracked;
        t.is_activated = true;
        t.frame_id = frame_id;
        t.score = det.score;
        t.class_id = det.class_id;
    }

    void predict(STrack& t) const {
        if (t.state != TrackState::Tracked) t.mean[7] = 0.0;  // stop height change while not seen
        kalman.predict(t.mean, t.covariance);
    }

    std::vector<float> dists(const std::vector<TrackPtr>& tracks, const std::vector<TrackPtr>& dets) const {
        std::vector<float> d = iou_distance(tracks, dets);
        if (config.fuse_score) fuse_score(d, tracks.size(), dets);
        return d;
    }

    void apply_match(const TrackPtr& track, const STrack& det, std::vector<TrackPtr>& activated,
                     std::vector<TrackPtr>& refind) {
        if (track->state == TrackState::Tracked) {
            update_track(*track, det);
            activated.push_back(track);
        } else {
            reactivate(*track, det);
            refind.push_back(track);
        }
    }
};

ByteTracker::ByteTracker(TrackerConfig config) : impl_(std::make_unique<Impl>()) { impl_->config = config; }

ByteTracker::~ByteTracker() = default;

void ByteTracker::reset() {
    const TrackerConfig config = impl_->config;
    impl_ = std::make_unique<Impl>();
    impl_->config = config;
}

int ByteTracker::frame_id() const { return impl_->frame_id; }

std::vector<Track> ByteTracker::update(const std::vector<postprocess::Detection>& input) {
    Impl& s = *impl_;
    const TrackerConfig& cfg = s.config;
    ++s.frame_id;
    std::vector<TrackPtr> activated, refind, lost, removed;

    // Split detections into high-score and low-score sets.
    std::vector<TrackPtr> high, low;
    for (const postprocess::Detection& d : input) {
        const float w = d.x2 - d.x1;
        const float h = d.y2 - d.y1;
        if (!(w > 0.0f && h > 0.0f)) continue;
        auto t = std::make_shared<STrack>();
        t->det_tlwh = {d.x1, d.y1, w, h};
        t->score = d.score;
        t->class_id = d.class_id;
        if (d.score >= cfg.track_high_thresh) {
            high.push_back(t);
        } else if (d.score > cfg.track_low_thresh) {
            low.push_back(t);
        }
    }

    // Unconfirmed: started last frame, not matched since. Pool: confirmed + lost, predicted.
    std::vector<TrackPtr> unconfirmed, confirmed;
    for (const auto& t : s.tracked) (t->is_activated ? confirmed : unconfirmed).push_back(t);
    const std::vector<TrackPtr> pool = joint(confirmed, s.lost);
    for (const auto& t : pool) s.predict(*t);

    // 1st association: pool vs high-score detections, IoU fused with score.
    const AssignmentResult first =
        linear_assignment(s.dists(pool, high), static_cast<int>(pool.size()), static_cast<int>(high.size()),
                          cfg.match_thresh);
    for (const auto& [it, id] : first.matches) s.apply_match(pool[it], *high[id], activated, refind);

    // 2nd association: still-tracked leftovers vs low-score detections, plain IoU.
    std::vector<TrackPtr> remaining;
    for (int it : first.unmatched_rows) {
        if (pool[it]->state == TrackState::Tracked) remaining.push_back(pool[it]);
    }
    std::vector<int> still_unmatched;
    if (!remaining.empty() && !low.empty()) {
        const AssignmentResult second = linear_assignment(iou_distance(remaining, low),
                                                          static_cast<int>(remaining.size()),
                                                          static_cast<int>(low.size()), 0.5);
        for (const auto& [it, id] : second.matches) s.apply_match(remaining[it], *low[id], activated, refind);
        still_unmatched = second.unmatched_rows;
    } else {
        for (int i = 0; i < static_cast<int>(remaining.size()); ++i) still_unmatched.push_back(i);
    }
    for (int it : still_unmatched) {
        const TrackPtr& t = remaining[it];
        if (t->state != TrackState::Lost) {
            t->state = TrackState::Lost;
            lost.push_back(t);
        }
    }

    // Unconfirmed tracks vs the high-score detections left over; unmatched ones are dropped.
    std::vector<TrackPtr> left;
    for (int id : first.unmatched_cols) left.push_back(high[id]);
    std::vector<int> new_candidates;
    if (unconfirmed.empty()) {
        for (int i = 0; i < static_cast<int>(left.size()); ++i) new_candidates.push_back(i);
    } else {
        const AssignmentResult third =
            linear_assignment(s.dists(unconfirmed, left), static_cast<int>(unconfirmed.size()),
                              static_cast<int>(left.size()), 0.7);
        for (const auto& [it, id] : third.matches) {
            s.update_track(*unconfirmed[it], *left[id]);
            activated.push_back(unconfirmed[it]);
        }
        for (int it : third.unmatched_rows) {
            unconfirmed[it]->state = TrackState::Removed;
            removed.push_back(unconfirmed[it]);
        }
        new_candidates = third.unmatched_cols;
    }

    // Start new tracks from confident unmatched detections.
    for (int id : new_candidates) {
        const TrackPtr& t = left[id];
        if (t->score < cfg.new_track_thresh) continue;
        s.activate(*t);
        activated.push_back(t);
    }

    // Drop tracks lost for too long.
    for (const auto& t : s.lost) {
        if (s.frame_id - t->frame_id > cfg.track_buffer) {
            t->state = TrackState::Removed;
            removed.push_back(t);
        }
    }

    // Merge pools (Ultralytics merge_track_pools).
    std::vector<TrackPtr> tracked;
    for (const auto& t : s.tracked) {
        if (t->state == TrackState::Tracked) tracked.push_back(t);
    }
    tracked = joint(tracked, activated);
    tracked = joint(tracked, refind);
    std::vector<TrackPtr> lost_pool = without_ids(s.lost, ids_of(tracked));
    lost_pool.insert(lost_pool.end(), lost.begin(), lost.end());
    std::set<int> removed_set(s.removed_ids.begin(), s.removed_ids.end());
    lost_pool = without_ids(lost_pool, removed_set);

    // Remove duplicates: a tracked and a lost track on the same spot, keep the older one.
    const std::vector<float> pdist = iou_distance(tracked, lost_pool);
    std::set<size_t> drop_tracked, drop_lost;
    for (size_t p = 0; p < tracked.size(); ++p) {
        for (size_t q = 0; q < lost_pool.size(); ++q) {
            if (pdist[p * lost_pool.size() + q] >= 0.15f) continue;
            const int time_p = tracked[p]->frame_id - tracked[p]->start_frame;
            const int time_q = lost_pool[q]->frame_id - lost_pool[q]->start_frame;
            if (time_p > time_q) {
                drop_lost.insert(q);
            } else {
                drop_tracked.insert(p);
            }
        }
    }
    s.tracked.clear();
    for (size_t p = 0; p < tracked.size(); ++p) {
        if (!drop_tracked.count(p)) s.tracked.push_back(tracked[p]);
    }
    s.lost.clear();
    for (size_t q = 0; q < lost_pool.size(); ++q) {
        if (!drop_lost.count(q)) s.lost.push_back(lost_pool[q]);
    }
    for (const auto& t : removed) s.removed_ids.push_back(t->track_id);
    while (s.removed_ids.size() > 1000) s.removed_ids.pop_front();

    std::vector<Track> output;
    for (const auto& t : s.tracked) {
        if (!t->is_activated) continue;
        const auto b = t->xyxy();
        output.push_back({t->track_id, b[0], b[1], b[2], b[3], t->score, t->class_id, t->start_frame});
    }
    return output;
}

}  // namespace edge_tracking::tracking
