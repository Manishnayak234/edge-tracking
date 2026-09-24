#pragma once

#include <gst/gst.h>

#include <string>

namespace edge_tracking::camera {

// Returns the first pending error or end-of-stream message on the pipeline bus, if any.
inline std::string pop_bus_error(GstElement* pipeline) {
    GstBus* bus = gst_element_get_bus(pipeline);
    std::string message;
    GstMessage* msg =
        gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (msg) {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
            message = "end of stream";
        } else {
            GError* err = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(msg, &err, &debug);
            message = err ? err->message : "unknown pipeline error";
            g_clear_error(&err);
            g_free(debug);
        }
        gst_message_unref(msg);
    }
    gst_object_unref(bus);
    return message;
}

}  // namespace edge_tracking::camera
