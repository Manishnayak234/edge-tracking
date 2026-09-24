#include "edge_tracking/camera/argus_gpu_camera.hpp"

#include "gst_utils.hpp"

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <nvbufsurface.h>

#include <chrono>
#include <sstream>
#include <vector>

namespace edge_tracking::camera {

namespace {

ColorSpace color_space_of(NvBufSurfaceColorFormat format) {
    switch (format) {
        case NVBUF_COLOR_FORMAT_NV12: return ColorSpace::Bt601Limited;
        case NVBUF_COLOR_FORMAT_NV12_ER: return ColorSpace::Bt601Full;
        case NVBUF_COLOR_FORMAT_NV12_709: return ColorSpace::Bt709Limited;
        case NVBUF_COLOR_FORMAT_NV12_709_ER: return ColorSpace::Bt709Full;
        default: return ColorSpace::Unknown;
    }
}

std::string cu_error(const char* what, CUresult result) {
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    return std::string(what) + ": " + (text ? text : "unknown CUDA error");
}

}  // namespace

struct ArgusGpuCamera::Impl {
    CameraConfig config;
    GstElement* pipeline = nullptr;
    GstElement* sink = nullptr;
    uint64_t frame_count = 0;
    std::string last_error;
    CUcontext cuda_context = nullptr;  // primary context, made current in whichever thread reads

    // Frame currently handed out to the caller, released on the next read() or stop().
    GstSample* sample = nullptr;
    GstBuffer* buffer = nullptr;
    GstMapInfo map{};
    bool buffer_mapped = false;

    // nvvidconv recycles a small pool of NVMM buffers. Mapping one into CUDA (EGLImage +
    // CUDA registration) costs syscalls, so each buffer is mapped once, on first sight, and
    // the mapping is reused every time that buffer comes back. Unmapped in stop().
    struct CudaMapping {
        NvBufSurface* surface = nullptr;
        uint64_t buffer_desc = 0;  // dmabuf fd; with `surface` identifies the pool buffer
        CUgraphicsResource resource = nullptr;
        CUeglFrame egl_frame{};
    };
    std::vector<CudaMapping> mappings;

    // Returns the CUDA mapping for `surface`, creating it if this buffer is new.
    const CudaMapping* mapping_for(NvBufSurface* surface) {
        const NvBufSurfaceParams& params = surface->surfaceList[0];
        for (const CudaMapping& m : mappings) {
            if (m.surface == surface && m.buffer_desc == params.bufferDesc) return &m;
        }

        CudaMapping m;
        m.surface = surface;
        m.buffer_desc = params.bufferDesc;
        if (NvBufSurfaceMapEglImage(surface, 0) != 0) {
            last_error = "NvBufSurfaceMapEglImage failed";
            return nullptr;
        }
        CUresult result = cuGraphicsEGLRegisterImage(&m.resource, params.mappedAddr.eglImage,
                                                     CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (result == CUDA_SUCCESS) result = cuGraphicsResourceGetMappedEglFrame(&m.egl_frame, m.resource, 0, 0);
        if (result != CUDA_SUCCESS) {
            last_error = cu_error("mapping NVMM buffer into CUDA", result);
            if (m.resource) cuGraphicsUnregisterResource(m.resource);
            NvBufSurfaceUnMapEglImage(surface, 0);
            return nullptr;
        }
        if (m.egl_frame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
            last_error = "EGL frame is not pitch-linear";
            cuGraphicsUnregisterResource(m.resource);
            NvBufSurfaceUnMapEglImage(surface, 0);
            return nullptr;
        }
        mappings.push_back(m);
        return &mappings.back();
    }

    // Must run while the pipeline still owns its buffer pool (before state NULL).
    void unmap_all() {
        for (CudaMapping& m : mappings) {
            cuGraphicsUnregisterResource(m.resource);
            NvBufSurfaceUnMapEglImage(m.surface, 0);
        }
        mappings.clear();
    }

    void release_frame() {
        if (buffer_mapped) {
            gst_buffer_unmap(buffer, &map);
            buffer_mapped = false;
        }
        buffer = nullptr;
        if (sample) {
            gst_sample_unref(sample);
            sample = nullptr;
        }
    }

    bool fail(std::string message) {
        last_error = std::move(message);
        release_frame();
        return false;
    }

    bool fail_keep_error() {
        release_frame();
        return false;
    }
};

ArgusGpuCamera::ArgusGpuCamera(CameraConfig config) : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

ArgusGpuCamera::~ArgusGpuCamera() { stop(); }

std::string ArgusGpuCamera::pipeline_description() const {
    const CameraConfig& c = impl_->config;
    std::ostringstream s;
    // Argus outputs block-linear (tiled) surfaces. nvvidconv converts them to pitch-linear
    // on the VIC engine, NVMM to NVMM: no CPU copy and no GPU compute, and CUDA kernels
    // can then index pixels as plain rows. Its pool needs more than the default 4 buffers:
    // the appsink queues up to 2 and the caller holds 1, which starved it and dropped frames.
    s << "nvarguscamerasrc sensor-id=" << c.sensor_id << " sensor-mode=" << c.sensor_mode
      << " ! video/x-raw(memory:NVMM),width=" << c.width << ",height=" << c.height
      << ",framerate=" << c.fps << "/1,format=NV12"
      << " ! nvvidconv bl-output=false output-buffers=8 ! video/x-raw(memory:NVMM),format=NV12"
      << " ! appsink name=sink max-buffers=2 drop=true sync=false";
    return s.str();
}

bool ArgusGpuCamera::start(std::string* error) {
    Impl& s = *impl_;
    if (s.pipeline) return true;

    // Create the CUDA primary context now; the driver-API EGL calls in read() use it.
    const cudaError_t cuda_status = cudaFree(nullptr);
    if (cuda_status != cudaSuccess) {
        s.last_error = std::string("CUDA init failed: ") + cudaGetErrorString(cuda_status);
        if (error) *error = s.last_error;
        return false;
    }

    cuCtxGetCurrent(&s.cuda_context);

    gst_init(nullptr, nullptr);
    GError* err = nullptr;
    s.pipeline = gst_parse_launch(pipeline_description().c_str(), &err);
    if (!s.pipeline) {
        s.last_error = err ? err->message : "gst_parse_launch failed";
        g_clear_error(&err);
        if (error) *error = s.last_error;
        return false;
    }
    g_clear_error(&err);  // parse can succeed with a non-fatal warning

    s.sink = gst_bin_get_by_name(GST_BIN(s.pipeline), "sink");
    if (gst_element_set_state(s.pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::string bus_error = pop_bus_error(s.pipeline);
        s.last_error = bus_error.empty()
                           ? "failed to start pipeline (is nvargus-daemon running and the camera free?)"
                           : bus_error;
        if (error) *error = s.last_error;
        stop();
        return false;
    }
    s.frame_count = 0;
    return true;
}

void ArgusGpuCamera::stop() {
    Impl& s = *impl_;
    s.release_frame();
    s.unmap_all();
    if (!s.pipeline) return;
    gst_element_set_state(s.pipeline, GST_STATE_NULL);
    if (s.sink) gst_object_unref(s.sink);
    gst_object_unref(s.pipeline);
    s.sink = nullptr;
    s.pipeline = nullptr;
}

bool ArgusGpuCamera::read(GpuFrame& frame, std::chrono::milliseconds timeout) {
    Impl& s = *impl_;
    if (!s.sink) return s.fail("camera not started");
    s.release_frame();

    s.sample = gst_app_sink_try_pull_sample(GST_APP_SINK(s.sink),
                                            static_cast<GstClockTime>(timeout.count()) * GST_MSECOND);
    if (!s.sample) {
        std::string bus_error = pop_bus_error(s.pipeline);
        return s.fail(bus_error.empty() ? "timed out waiting for a frame" : bus_error);
    }

    // Mapping an NVMM GstBuffer gives an NvBufSurface descriptor, not pixel data.
    s.buffer = gst_sample_get_buffer(s.sample);
    if (!s.buffer || !gst_buffer_map(s.buffer, &s.map, GST_MAP_READ)) {
        return s.fail("failed to map NVMM buffer");
    }
    s.buffer_mapped = true;
    auto* surface = reinterpret_cast<NvBufSurface*>(s.map.data);
    if (!surface || surface->numFilled < 1) return s.fail("buffer holds no NvBufSurface");

    const NvBufSurfaceParams& params = surface->surfaceList[0];
    if (params.layout != NVBUF_LAYOUT_PITCH) {
        return s.fail("camera surface is block-linear; pitch-linear expected");
    }

    // NvBufSurface -> EGLImage -> CUDA: gives device pointers to the same memory. The driver
    // API calls need a current context, which a fresh thread does not have yet.
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != s.cuda_context) cuCtxSetCurrent(s.cuda_context);
    const Impl::CudaMapping* mapping = s.mapping_for(surface);
    if (!mapping) return s.fail_keep_error();
    const CUeglFrame& egl_frame = mapping->egl_frame;

    frame.width = static_cast<int>(params.width);
    frame.height = static_cast<int>(params.height);
    frame.y = static_cast<const uint8_t*>(egl_frame.frame.pPitch[0]);
    frame.uv = static_cast<const uint8_t*>(egl_frame.frame.pPitch[1]);
    frame.y_pitch = params.planeParams.pitch[0];
    frame.uv_pitch = params.planeParams.pitch[1];
    frame.color_space = color_space_of(params.colorFormat);
    frame.pts_ns = GST_BUFFER_PTS_IS_VALID(s.buffer) ? static_cast<int64_t>(GST_BUFFER_PTS(s.buffer)) : -1;
    frame.capture_time_ns = -1;
    if (frame.pts_ns >= 0) {
        // pipeline clock time of capture = base time + PTS; shift it into steady_clock time.
        if (GstClock* clock = gst_element_get_clock(s.pipeline)) {
            const auto gst_now = static_cast<int64_t>(gst_clock_get_time(clock));
            const int64_t steady_now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now().time_since_epoch())
                                           .count();
            const int64_t gst_capture = static_cast<int64_t>(gst_element_get_base_time(s.pipeline)) + frame.pts_ns;
            frame.capture_time_ns = steady_now - (gst_now - gst_capture);
            gst_object_unref(clock);
        }
    }
    frame.index = s.frame_count++;
    return true;
}

const std::string& ArgusGpuCamera::last_error() const { return impl_->last_error; }

}  // namespace edge_tracking::camera
