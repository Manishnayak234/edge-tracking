cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
endif()

project(edge_tracking LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
find_package(CUDAToolkit REQUIRED)

# Jetson Multimedia API: NvBufSurface (NVMM buffers)
set(JETSON_MMAPI_DIR /usr/src/jetson_multimedia_api CACHE PATH "Jetson Multimedia API root")
find_library(NVBUFSURFACE_LIB nvbufsurface PATHS /usr/lib/aarch64-linux-gnu/nvidia REQUIRED)

# Camera capture (Argus via GStreamer): CPU-copy and zero-copy GPU paths
add_library(et_camera
  src/camera/argus_camera.cpp
  src/camera/argus_gpu_camera.cpp)
target_include_directories(et_camera
  PUBLIC include
  PRIVATE ${JETSON_MMAPI_DIR}/include)
target_link_libraries(et_camera PRIVATE PkgConfig::GST CUDA::cudart CUDA::cuda_driver ${NVBUFSURFACE_LIB})
target_compile_options(et_camera PRIVATE -Wall -Wextra)

# Preprocessing (CUDA): NV12 -> letterboxed RGB tensor for the detector
add_library(et_preprocess src/preprocess/letterbox.cu)
target_include_directories(et_preprocess PUBLIC include)
target_link_libraries(et_preprocess PUBLIC CUDA::cudart)

# Inference (TensorRT)
find_library(NVINFER_LIB nvinfer REQUIRED)
add_library(et_inference src/inference/trt_engine.cpp)
target_include_directories(et_inference PUBLIC include)
target_link_libraries(et_inference PUBLIC CUDA::cudart PRIVATE ${NVINFER_LIB})
target_compile_options(et_inference PRIVATE -Wall -Wextra)

# Postprocessing: YOLOv8 decode (CUDA) + NMS
add_library(et_postprocess src/postprocess/yolo_decoder.cu)
target_include_directories(et_postprocess PUBLIC include)
target_link_libraries(et_postprocess PUBLIC CUDA::cudart)

# Tracking: ByteTrack (CPU)
add_library(et_tracking
  src/tracking/byte_tracker.cpp
  src/tracking/kalman_filter.cpp
  src/tracking/linear_assignment.cpp)
target_include_directories(et_tracking PUBLIC include)
target_link_libraries(et_tracking PUBLIC et_postprocess)
target_compile_options(et_tracking PRIVATE -Wall -Wextra)

# Telemetry: per-stage timings and counters
add_library(et_telemetry src/telemetry/telemetry.cpp)
target_include_directories(et_telemetry PUBLIC include)
target_compile_options(et_telemetry PRIVATE -Wall -Wextra)

# Pipeline: capture / inference+tracking / sink threads
find_package(Threads REQUIRED)
add_library(et_pipeline src/pipeline/pipeline.cpp)
target_include_directories(et_pipeline PUBLIC include)
target_link_libraries(et_pipeline
  PUBLIC et_postprocess et_tracking et_telemetry
  PRIVATE et_camera et_preprocess et_inference Threads::Threads)
target_compile_options(et_pipeline PRIVATE -Wall -Wextra)

# Apps
add_executable(camera_fps apps/camera_fps.cpp)
target_link_libraries(camera_fps PRIVATE et_camera)
target_compile_options(camera_fps PRIVATE -Wall -Wextra)

add_executable(camera_gpu_fps apps/camera_gpu_fps.cu)
target_link_libraries(camera_gpu_fps PRIVATE et_camera CUDA::cudart)

add_executable(infer_check apps/infer_check.cpp)
target_link_libraries(infer_check PRIVATE et_camera et_preprocess et_inference)
target_compile_options(infer_check PRIVATE -Wall -Wextra)

add_executable(postprocess_check apps/postprocess_check.cpp)
target_link_libraries(postprocess_check PRIVATE et_camera et_preprocess et_inference et_postprocess)
target_compile_options(postprocess_check PRIVATE -Wall -Wextra)

add_executable(track_check apps/track_check.cpp)
target_link_libraries(track_check PRIVATE et_camera et_preprocess et_inference et_postprocess et_tracking)
target_compile_options(track_check PRIVATE -Wall -Wextra)

# Servo control: sweeps an Arduino-driven servo while a person is in view (no OpenCV needed)
add_executable(servo_track apps/servo_track.cpp)
target_link_libraries(servo_track PRIVATE et_pipeline)
target_compile_options(servo_track PRIVATE -Wall -Wextra)

# OpenCV is only used by checks (as an independent reference) and the debug viewers. Require NVIDIA's 4.8 build:
# Ubuntu's 4.5 is installed too, and NVIDIA's opencv4.pc has a wrong prefix, so use CMake config.
find_package(OpenCV 4.8 QUIET COMPONENTS core imgproc imgcodecs highgui)
if(OpenCV_FOUND)
  add_executable(detect_view apps/detect_view.cpp)
  target_link_libraries(detect_view PRIVATE et_camera et_preprocess et_inference et_postprocess ${OpenCV_LIBS})
  target_compile_options(detect_view PRIVATE -Wall -Wextra)
  add_executable(track_view apps/track_view.cpp)
  target_link_libraries(track_view PRIVATE et_camera et_preprocess et_inference et_postprocess et_tracking
                                           ${OpenCV_LIBS})
  target_compile_options(track_view PRIVATE -Wall -Wextra)
  add_executable(pipeline_run apps/pipeline_run.cpp)
  target_link_libraries(pipeline_run PRIVATE et_pipeline ${OpenCV_LIBS})
  target_compile_options(pipeline_run PRIVATE -Wall -Wextra)

  add_executable(preprocess_check apps/preprocess_check.cpp)
  target_link_libraries(preprocess_check PRIVATE et_camera et_preprocess ${OpenCV_LIBS})
  target_compile_options(preprocess_check PRIVATE -Wall -Wextra)
endif()
