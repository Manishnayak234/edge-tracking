#pragma once

// Drawing helpers shared by the OpenCV debug viewers.

#include "edge_tracking/tracking/byte_tracker.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace edge_tracking::apps {

// Ultralytics' 20-color palette (BGR). Index by class id (as Ultralytics plots) or track id.
inline cv::Scalar palette_color(int index) {
    static const unsigned rgb[] = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                   0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                   0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
    const unsigned c = rgb[index % (sizeof(rgb) / sizeof(rgb[0]))];
    return {static_cast<double>(c & 0xFF), static_cast<double>((c >> 8) & 0xFF), static_cast<double>(c >> 16)};
}

inline void draw_label(cv::Mat& image, const std::string& text, cv::Point origin, const cv::Scalar& color) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    const int top = std::max(origin.y - size.height - 6, 0);
    cv::rectangle(image, {origin.x, top}, {origin.x + size.width + 6, top + size.height + 6}, color, cv::FILLED);
    cv::putText(image, text, {origin.x + 3, top + size.height + 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1,
                cv::LINE_AA);
}

// Status line with a dark outline so it stays readable on any background.
inline void draw_status(cv::Mat& image, const std::string& text) {
    cv::putText(image, text, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 4, cv::LINE_AA);
    cv::putText(image, text, {10, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 1, cv::LINE_AA);
}

// Recent bottom-center positions of each track, for drawing motion trails.
class TrailHistory {
public:
    // length: points kept per track (45 = 0.75 s at 60 fps).
    // forget_after: updates without the track before its trail is dropped (tracker buffer).
    explicit TrailHistory(size_t length = 45, int forget_after = 30) : length_(length), forget_after_(forget_after) {}

    // Call once per shown frame with that frame's tracks.
    void update(const std::vector<tracking::Track>& tracks) {
        ++update_count_;
        for (const tracking::Track& t : tracks) {
            Trail& trail = trails_[t.id];
            trail.points.emplace_back(static_cast<int>((t.x1 + t.x2) / 2), static_cast<int>(t.y2));
            if (trail.points.size() > length_) trail.points.pop_front();
            trail.last_seen = update_count_;
        }
        for (auto it = trails_.begin(); it != trails_.end();) {
            it = update_count_ - it->second.last_seen > forget_after_ ? trails_.erase(it) : std::next(it);
        }
    }

    const std::deque<cv::Point>& points(int id) const {
        static const std::deque<cv::Point> empty;
        const auto it = trails_.find(id);
        return it == trails_.end() ? empty : it->second.points;
    }

private:
    struct Trail {
        std::deque<cv::Point> points;  // oldest first
        long last_seen = 0;
    };
    size_t length_;
    int forget_after_;
    long update_count_ = 0;
    std::map<int, Trail> trails_;
};

// Box, "class #id score" label and trail per track; colors by track id.
inline void draw_tracks(cv::Mat& image, const std::vector<tracking::Track>& tracks, const TrailHistory& trails,
                        const std::vector<std::string>& names) {
    for (const tracking::Track& t : tracks) {
        const cv::Scalar color = palette_color(t.id);
        const std::deque<cv::Point>& trail = trails.points(t.id);
        for (size_t i = 1; i < trail.size(); ++i) {
            const int thickness = 1 + static_cast<int>(3 * i / trail.size());  // thicker = newer
            cv::line(image, trail[i - 1], trail[i], color, thickness, cv::LINE_AA);
        }
        const cv::Point p1(static_cast<int>(t.x1), static_cast<int>(t.y1));
        cv::rectangle(image, p1, {static_cast<int>(t.x2), static_cast<int>(t.y2)}, color, 2);
        const std::string name = t.class_id >= 0 && t.class_id < static_cast<int>(names.size())
                                     ? names[t.class_id]
                                     : std::to_string(t.class_id);
        char label[96];
        std::snprintf(label, sizeof(label), "%s #%d %.2f", name.c_str(), t.id, t.score);
        draw_label(image, label, p1, color);
    }
}

}  // namespace edge_tracking::apps
