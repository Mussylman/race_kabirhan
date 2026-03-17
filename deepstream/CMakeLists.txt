cmake_minimum_required(VERSION 3.18)
project(race_vision_deepstream LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# ── DeepStream SDK ──────────────────────────────────────────────────

set(DEEPSTREAM_SDK "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK path")

if(NOT EXISTS ${DEEPSTREAM_SDK})
    message(FATAL_ERROR "DeepStream SDK not found at ${DEEPSTREAM_SDK}. "
                        "Set -DDEEPSTREAM_SDK=/path/to/deepstream")
endif()

# ── Find packages ──────────────────────────────────────────────────

find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu
          /usr/local/include
          ${CUDA_TOOLKIT_ROOT_DIR}/include)

find_library(TENSORRT_LIB nvinfer
    PATHS /usr/lib/x86_64-linux-gnu
          /usr/local/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_library(TENSORRT_PARSERS_LIB nvparsers
    PATHS /usr/lib/x86_64-linux-gnu
          /usr/local/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# nlohmann/json (header-only)
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp
    PATHS /usr/include
          /usr/local/include
          ${CMAKE_SOURCE_DIR}/third_party)

if(NOT JSON_INCLUDE_DIR)
    message(STATUS "nlohmann/json not found, will download...")
    include(FetchContent)
    FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(json)
    set(JSON_INCLUDE_DIR ${json_SOURCE_DIR}/include)
    set(JSON_TARGET nlohmann_json::nlohmann_json)
endif()

# ── Include directories ────────────────────────────────────────────

set(DS_INCLUDE_DIRS
    ${DEEPSTREAM_SDK}/sources/includes
    ${DEEPSTREAM_SDK}/sources/libs/nvdsinfer
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${DS_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
)

# ── Source files ───────────────────────────────────────────────────

set(SOURCES
    src/main.cpp
    src/pipeline.cpp
    src/trigger_pipeline.cpp
    src/analysis_pipeline.cpp
    src/dual_pipeline.cpp
    src/trigger_shm.cpp
    src/yolo_parser.cpp
    src/color_infer.cpp
    src/shm_writer.cpp
    src/diag_logger.cpp
)

# color_infer.cpp contains CUDA kernels — compile as .cu
set_source_files_properties(src/color_infer.cpp PROPERTIES LANGUAGE CUDA)

# ── Executable ─────────────────────────────────────────────────────

add_executable(race_vision_deepstream ${SOURCES})

target_link_directories(race_vision_deepstream PRIVATE
    ${DEEPSTREAM_SDK}/lib
    /usr/local/cuda/lib64/stubs
    /usr/local/cuda/lib64
)

target_link_libraries(race_vision_deepstream PRIVATE
    # GStreamer
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}

    # DeepStream
    nvdsgst_meta
    nvds_meta
    nvds_infer
    nvbufsurface
    nvbufsurftransform

    # TensorRT + CUDA
    ${TENSORRT_LIB}
    ${CUDA_LIBRARIES}
    cudart
    cuda

    # POSIX shared memory
    rt
    pthread

    # Filesystem (C++17)
    stdc++fs
)

if(JSON_TARGET)
    target_link_libraries(race_vision_deepstream PRIVATE ${JSON_TARGET})
endif()

# ── Custom parser shared library (for nvinfer plugin) ──────────────

add_library(nvdsinfer_yolov8_parser SHARED
    src/yolo_parser.cpp
)

target_include_directories(nvdsinfer_yolov8_parser PRIVATE
    ${DS_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_directories(nvdsinfer_yolov8_parser PRIVATE
    ${DEEPSTREAM_SDK}/lib
)

target_link_libraries(nvdsinfer_yolov8_parser PRIVATE
    nvds_infer
)

# ── Install ────────────────────────────────────────────────────────

install(TARGETS race_vision_deepstream DESTINATION bin)
install(TARGETS nvdsinfer_yolov8_parser DESTINATION lib)

message(STATUS "")
message(STATUS "Race Vision DeepStream Configuration:")
message(STATUS "  DeepStream SDK: ${DEEPSTREAM_SDK}")
message(STATUS "  CUDA:           ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  TensorRT:       ${TENSORRT_LIB}")
message(STATUS "  GStreamer:       ${GSTREAMER_VERSION}")
message(STATUS "")
