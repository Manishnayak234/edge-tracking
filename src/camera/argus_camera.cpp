#include "edge_tracking/camera/argus_camera.hpp"

#include "gst_utils.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstring>
#include <sstream>

namespace edge_tracking::camera {

struct ArgusCamera::Impl {
    CameraConfig config;
    GstElement* pipeline = nullptr;
    GstElement* sink = nullptr;
    uint64_t frame_count = 0;
    std::string last_error;
};

ArgusCamera::ArgusCamera(CameraConfig config) : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

ArgusCamera::~ArgusCamera() { stop(); }

std::string ArgusCamera::pipeline_description() const {
    const CameraConfig& c = impl_->config;
    std::ostringstream s;
    s << "nvarguscamerasrc sensor-id=" << c.sensor_id << " sensor-mode=" << c.sensor_mode
      << " ! video/x-raw(memory:NVMM),width=" << c.width << ",height=" << c.height
      << ",framerate=" << c.fps << "/1,format=NV12"
      << " ! nvvidconv ! video/x-raw,format=NV12"
      // Keep at most 2 frames queued and drop the oldest, so a slow consumer
      // sees fresh frames instead of building up latency.
      << " ! appsink name=sink max-buffers=2 drop=true sync=false";
    return s.str();
}

bool ArgusCamera::start(std::string* error) {
    if (impl_->pipeline) return true;
    gst_init(nullptr, nullptr);

    GError* err = nullptr;
    impl_->pipeline = gst_parse_launch(pipeline_description().c_str(), &err);
    if (!impl_->pipeline) {
        impl_->last_error = err ? err->message : "gst_parse_launch failed";
        g_clear_error(&err);
        if (error) *error = impl_->last_error;
        return false;
    }
    g_clear_error(&err);  // parse can succeed with a non-fatal warning

    impl_->sink = gst_bin_get_by_name(GST_BIN(impl_->pipeline), "sink");
    if (gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::string bus_error = pop_bus_error(impl_->pipeline);
        impl_->last_error = bus_error.empty()
                                ? "failed to start pipeline (is nvargus-daemon running and the camera free?)"
                                : bus_error;
        if (error) *error = impl_->last_error;
        stop();
        return false;
    }
    impl_->frame_count = 0;
    return true;
}

void ArgusCamera::stop() {
    if (!impl_->pipeline) return;
    gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    if (impl_->sink) gst_object_unref(impl_->sink);
    gst_object_unref(impl_->pipeline);
    impl_->sink = nullptr;
    impl_->pipeline = nullptr;
}

bool ArgusCamera::read(Frame& frame, std::chrono::milliseconds timeout) {
    if (!impl_->sink) {
        impl_->last_error = "camera not started";
        return false;
    }

    GstSample* sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(impl_->sink), static_cast<GstClockTime>(timeout.count()) * GST_MSECOND);
    if (!sample) {
        std::string bus_error = pop_bus_error(impl_->pipeline);
        impl_->last_error = bus_error.empty() ? "timed out waiting for a frame" : bus_error;
        return false;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo info;
    GstVideoFrame vframe;
    if (!buffer || !caps || !gst_video_info_from_caps(&info, caps) ||
        !gst_video_frame_map(&vframe, &info, buffer, GST_MAP_READ)) {
        gst_sample_unref(sample);
        impl_->last_error = "failed to map frame";
        return false;
    }

    const int width = GST_VIDEO_INFO_WIDTH(&info);
    const int height = GST_VIDEO_INFO_HEIGHT(&info);
    frame.width = width;
    frame.height = height;
    frame.nv12.resize(static_cast<size_t>(width) * height * 3 / 2);

    // Copy row by row: source rows can be padded (stride > width), the output is packed.
    // Plane 0 is Y (height rows), plane 1 is interleaved UV (height / 2 rows); both are
    // `width` bytes per row.
    uint8_t* dst = frame.nv12.data();
    for (int plane = 0; plane < 2; ++plane) {
        const auto* src = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&vframe, plane));
        const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vframe, plane);
        const int rows = plane == 0 ? height : height / 2;
        for (int r = 0; r < rows; ++r) {
            std::memcpy(dst, src + static_cast<size_t>(r) * stride, width);
            dst += width;
        }
    }

    frame.pts_ns = GST_BUFFER_PTS_IS_VALID(buffer) ? static_cast<int64_t>(GST_BUFFER_PTS(buffer)) : -1;
    frame.index = impl_->frame_count++;

    gst_video_frame_unmap(&vframe);
    gst_sample_unref(sample);
    return true;
}

const std::string& ArgusCamera::last_error() const { return impl_->last_error; }

}  // namespace edge_tracking::camera
