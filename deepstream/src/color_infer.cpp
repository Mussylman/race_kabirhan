/**
 * color_infer.cpp — TensorRT color classifier + CUDA crop/resize/normalize.
 */

#include "color_infer.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include "nvbufsurface.h"

namespace rv {

// ── CUDA kernel: crop + resize + normalize ─────────────────────────

// ImageNet normalization constants
static __device__ __constant__ float MEAN[3] = {0.485f, 0.456f, 0.406f};
static __device__ __constant__ float STD[3]  = {0.229f, 0.224f, 0.225f};
static constexpr int CROP_SIZE = 64;  // v1: SimpleColorCNN expects 64x64

/**
 * CUDA kernel: crop + resize + NV12→RGB + ImageNet normalize.
 * Reads directly from NV12 surface (Y plane + UV interleaved plane).
 * No nvvideoconvert needed — converts only the small crop region.
 *
 * Output layout: [batch_idx, channel, y, x] (NCHW)
 */
__global__ void crop_resize_normalize_nv12_kernel(
    const uint8_t* __restrict__ y_plane,   // Y plane (luminance)
    const uint8_t* __restrict__ uv_plane,  // UV interleaved plane (chrominance)
    int src_width, int src_height,
    int y_pitch, int uv_pitch,
    const int* __restrict__ rois,          // [N, 4] = {x1, y1, x2, y2}
    float* __restrict__ dst,               // [N, 3, CROP_SIZE, CROP_SIZE]
    int num_crops)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = num_crops * CROP_SIZE * CROP_SIZE;
    if (tid >= total_pixels) return;

    int crop_idx = tid / (CROP_SIZE * CROP_SIZE);
    int pixel_idx = tid % (CROP_SIZE * CROP_SIZE);
    int out_y = pixel_idx / CROP_SIZE;
    int out_x = pixel_idx % CROP_SIZE;

    // Get ROI for this crop
    int roi_x1 = rois[crop_idx * 4 + 0];
    int roi_y1 = rois[crop_idx * 4 + 1];
    int roi_x2 = rois[crop_idx * 4 + 2];
    int roi_y2 = rois[crop_idx * 4 + 3];
    int roi_w = roi_x2 - roi_x1;
    int roi_h = roi_y2 - roi_y1;

    if (roi_w <= 0 || roi_h <= 0) return;

    // Map output pixel to source coordinate
    float sx = roi_x1 + (out_x + 0.5f) * roi_w / CROP_SIZE;
    float sy = roi_y1 + (out_y + 0.5f) * roi_h / CROP_SIZE;

    int ix = min(max(static_cast<int>(sx), 0), src_width - 1);
    int iy = min(max(static_cast<int>(sy), 0), src_height - 1);

    // Read NV12: Y from Y-plane, U/V from interleaved UV-plane
    int Y = y_plane[iy * y_pitch + ix];
    int uv_row = iy / 2;
    int uv_col = (ix / 2) * 2;  // UV is interleaved: U0 V0 U1 V1 ...
    int U = uv_plane[uv_row * uv_pitch + uv_col]     - 128;
    int V = uv_plane[uv_row * uv_pitch + uv_col + 1] - 128;

    // BT.601 NV12 → RGB (same formula as diag_logger)
    float r = min(max((Y + ((359 * V) >> 8))              / 255.0f, 0.0f), 1.0f);
    float g = min(max((Y - ((88 * U + 183 * V) >> 8))     / 255.0f, 0.0f), 1.0f);
    float b = min(max((Y + ((454 * U) >> 8))               / 255.0f, 0.0f), 1.0f);

    // ImageNet normalize
    r = (r - MEAN[0]) / STD[0];
    g = (g - MEAN[1]) / STD[1];
    b = (b - MEAN[2]) / STD[2];

    // Write to NCHW layout
    int plane_size = CROP_SIZE * CROP_SIZE;
    int base = crop_idx * 3 * plane_size + pixel_idx;
    dst[base + 0 * plane_size] = r;
    dst[base + 1 * plane_size] = g;
    dst[base + 2 * plane_size] = b;
}

/**
 * CUDA kernel: crop + resize + RGBA→RGB + ImageNet normalize (fallback).
 */
__global__ void crop_resize_normalize_rgba_kernel(
    const uint8_t* __restrict__ src,
    int src_width, int src_height, int src_pitch,
    const int* __restrict__ rois,
    float* __restrict__ dst,
    int num_crops)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = num_crops * CROP_SIZE * CROP_SIZE;
    if (tid >= total_pixels) return;

    int crop_idx = tid / (CROP_SIZE * CROP_SIZE);
    int pixel_idx = tid % (CROP_SIZE * CROP_SIZE);
    int out_y = pixel_idx / CROP_SIZE;
    int out_x = pixel_idx % CROP_SIZE;

    int roi_x1 = rois[crop_idx * 4 + 0];
    int roi_y1 = rois[crop_idx * 4 + 1];
    int roi_x2 = rois[crop_idx * 4 + 2];
    int roi_y2 = rois[crop_idx * 4 + 3];
    int roi_w = roi_x2 - roi_x1;
    int roi_h = roi_y2 - roi_y1;
    if (roi_w <= 0 || roi_h <= 0) return;

    float sx = roi_x1 + (out_x + 0.5f) * roi_w / CROP_SIZE;
    float sy = roi_y1 + (out_y + 0.5f) * roi_h / CROP_SIZE;
    int ix = min(max(static_cast<int>(sx), 0), src_width - 1);
    int iy = min(max(static_cast<int>(sy), 0), src_height - 1);

    const uint8_t* pixel = src + iy * src_pitch + ix * 4;
    float r = pixel[0] / 255.0f;
    float g = pixel[1] / 255.0f;
    float b = pixel[2] / 255.0f;

    r = (r - MEAN[0]) / STD[0];
    g = (g - MEAN[1]) / STD[1];
    b = (b - MEAN[2]) / STD[2];

    int plane_size = CROP_SIZE * CROP_SIZE;
    int base = crop_idx * 3 * plane_size + pixel_idx;
    dst[base + 0 * plane_size] = r;
    dst[base + 1 * plane_size] = g;
    dst[base + 2 * plane_size] = b;
}

// ── ColorInfer implementation ──────────────────────────────────────

ColorInfer::ColorInfer() = default;

ColorInfer::~ColorInfer() {
    delete context_;
    delete engine_;
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (d_rois_)   cudaFree(d_rois_);
    if (stream_)   cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

bool ColorInfer::load(const std::string& engine_path) {
    // Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        fprintf(stderr, "[ColorInfer] Cannot open engine: %s\n", engine_path.c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Deserialize — TRT 10+ logger
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                fprintf(stderr, "[TRT] %s\n", msg);
        }
    };
    static Logger trt_logger;

    auto runtime = nvinfer1::createInferRuntime(trt_logger);
    if (!runtime) {
        fprintf(stderr, "[ColorInfer] Failed to create TRT runtime\n");
        return false;
    }

    engine_ = runtime->deserializeCudaEngine(engine_data.data(), size);
    delete runtime;
    if (!engine_) {
        fprintf(stderr, "[ColorInfer] Failed to deserialize engine\n");
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        fprintf(stderr, "[ColorInfer] Failed to create execution context\n");
        return false;
    }

    // Allocate GPU buffers (pre-allocated once, reused every frame)
    size_t input_bytes  = max_batch_ * 3 * CROP_SIZE * CROP_SIZE * sizeof(float);
    size_t output_bytes = max_batch_ * MODEL_NUM_CLASSES * sizeof(float);
    size_t rois_bytes   = max_batch_ * 4 * sizeof(int);
    cudaMalloc(&d_input_,  input_bytes);
    cudaMalloc(&d_output_, output_bytes);
    cudaMalloc(&d_rois_,   rois_bytes);

    // TRT 10: set tensor addresses (required for enqueueV3)
    context_->setTensorAddress("input", d_input_);
    context_->setTensorAddress("output", d_output_);

    cudaStream_t s;
    cudaStreamCreate(&s);
    stream_ = s;

    fprintf(stderr, "[ColorInfer] Loaded engine: %s (max_batch=%d, classes=%d→%d)\n",
            engine_path.c_str(), max_batch_, MODEL_NUM_CLASSES, NUM_COLORS);
    return true;
}

TorsoROI ColorInfer::compute_torso_roi(float x1, float y1, float x2, float y2,
                                       int frame_w, int frame_h,
                                       int det_index, int cam_index) {
    float bw = x2 - x1;
    float bh = y2 - y1;

    TorsoROI roi;
    roi.x1 = std::max(0,       static_cast<int>(x1 + bw * TORSO_LEFT));
    roi.y1 = std::max(0,       static_cast<int>(y1 + bh * TORSO_TOP));
    roi.x2 = std::min(frame_w, static_cast<int>(x2 - bw * TORSO_RIGHT));
    roi.y2 = std::min(frame_h, static_cast<int>(y1 + bh * TORSO_BOTTOM));
    roi.det_index = det_index;
    roi.cam_index = cam_index;
    return roi;
}

void ColorInfer::preprocess_crops(const NvBufSurface* surface,
                                  const std::vector<TorsoROI>& rois) {
    if (rois.empty() || !surface) return;

    int num_crops = std::min(static_cast<int>(rois.size()), max_batch_);
    auto stream = static_cast<cudaStream_t>(stream_);

    // Upload all ROIs to pre-allocated GPU buffer (no malloc!)
    std::vector<int> h_rois(num_crops * 4);
    for (int i = 0; i < num_crops; ++i) {
        h_rois[i * 4 + 0] = rois[i].x1;
        h_rois[i * 4 + 1] = rois[i].y1;
        h_rois[i * 4 + 2] = rois[i].x2;
        h_rois[i * 4 + 3] = rois[i].y2;
    }
    cudaMemcpyAsync(d_rois_, h_rois.data(), num_crops * 4 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Group crops by camera and launch batched kernels
    int threads = 256;
    int pixels_per_crop = CROP_SIZE * CROP_SIZE;

    // Find runs of same surface_index (rois are typically grouped by camera)
    int i = 0;
    while (i < num_crops) {
        int surf_idx = rois[i].surface_index;
        if (surf_idx >= static_cast<int>(surface->numFilled)) { i++; continue; }

        // Count consecutive crops from same surface
        int batch_start = i;
        int batch_count = 0;
        while (i < num_crops && rois[i].surface_index == surf_idx) {
            i++;
            batch_count++;
        }

        const auto& params = surface->surfaceList[surf_idx];
        const uint8_t* src = static_cast<const uint8_t*>(params.dataPtr);
        int src_w = params.width;
        int src_h = params.height;

        // Total pixels for this batch
        int total_pixels = batch_count * pixels_per_crop;
        int blocks = (total_pixels + threads - 1) / threads;

        float* dst_ptr = static_cast<float*>(d_input_) + batch_start * 3 * pixels_per_crop;

        auto fmt = params.colorFormat;
        bool is_nv12 = (fmt == NVBUF_COLOR_FORMAT_NV12 ||
                        fmt == NVBUF_COLOR_FORMAT_NV12_709 ||
                        fmt == NVBUF_COLOR_FORMAT_NV12_709_ER ||
                        fmt == NVBUF_COLOR_FORMAT_NV12_2020 ||
                        fmt == NVBUF_COLOR_FORMAT_NV12_ER);

        if (is_nv12) {
            int y_pitch = params.planeParams.pitch[0];
            int uv_pitch = params.planeParams.pitch[1];
            const uint8_t* y_ptr = src + params.planeParams.offset[0];
            const uint8_t* uv_ptr = src + params.planeParams.offset[1];

            // One kernel launch for ALL crops from this camera
            crop_resize_normalize_nv12_kernel<<<blocks, threads, 0, stream>>>(
                y_ptr, uv_ptr, src_w, src_h, y_pitch, uv_pitch,
                d_rois_ + batch_start * 4, dst_ptr, batch_count);
        } else {
            crop_resize_normalize_rgba_kernel<<<blocks, threads, 0, stream>>>(
                src, src_w, src_h, params.pitch,
                d_rois_ + batch_start * 4, dst_ptr, batch_count);
        }
    }
}

void ColorInfer::softmax_and_parse(int num_crops, std::vector<ColorResult>& results) {
    // Copy output from GPU (model outputs MODEL_NUM_CLASSES per crop)
    std::vector<float> h_output(num_crops * MODEL_NUM_CLASSES);
    cudaMemcpyAsync(h_output.data(), d_output_,
                    num_crops * MODEL_NUM_CLASSES * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));

    results.resize(num_crops);
    for (int i = 0; i < num_crops; ++i) {
        const float* logits = h_output.data() + i * MODEL_NUM_CLASSES;

        // Softmax over MODEL_NUM_CLASSES
        float max_val = *std::max_element(logits, logits + MODEL_NUM_CLASSES);
        float sum = 0.0f;
        float model_probs[MODEL_NUM_CLASSES];
        for (int c = 0; c < MODEL_NUM_CLASSES; ++c) {
            model_probs[c] = std::exp(logits[c] - max_val);
            sum += model_probs[c];
        }
        for (int c = 0; c < MODEL_NUM_CLASSES; ++c) {
            model_probs[c] /= sum;
        }

        // Find best model class
        int best_model = 0;
        for (int c = 1; c < MODEL_NUM_CLASSES; ++c) {
            if (model_probs[c] > model_probs[best_model]) best_model = c;
        }

        // Map model classes → SHM 5-slot format
        results[i].color_id   = MODEL_TO_SHM[best_model];
        results[i].confidence = model_probs[best_model];
        // Fill SHM probs: zero for unused slots (blue, purple)
        std::memset(results[i].probs, 0, sizeof(results[i].probs));
        for (int c = 0; c < MODEL_NUM_CLASSES; ++c) {
            results[i].probs[MODEL_TO_SHM[c]] = model_probs[c];
        }
    }
}

void ColorInfer::classify(const NvBufSurface* surface,
                          const std::vector<TorsoROI>& rois,
                          std::vector<ColorResult>& results) {
    if (rois.empty() || !engine_) {
        results.clear();
        return;
    }

    int num_crops = static_cast<int>(rois.size());
    auto stream = static_cast<cudaStream_t>(stream_);

    // Process in batches of max_batch_
    results.clear();
    results.reserve(num_crops);

    for (int offset = 0; offset < num_crops; offset += max_batch_) {
        int batch_size = std::min(max_batch_, num_crops - offset);

        // Preprocess: crop + resize + normalize on GPU
        std::vector<TorsoROI> batch_rois(rois.begin() + offset,
                                         rois.begin() + offset + batch_size);
        preprocess_crops(surface, batch_rois);

        // Set dynamic batch size if using dynamic shapes
        context_->setInputShape("input",
            nvinfer1::Dims4{batch_size, 3, CROP_SIZE, CROP_SIZE});

        // Run inference (tensor addresses set in load())
        context_->enqueueV3(stream);

        // Parse results
        std::vector<ColorResult> batch_results;
        softmax_and_parse(batch_size, batch_results);
        results.insert(results.end(), batch_results.begin(), batch_results.end());
    }
}

void ColorInfer::classify_preprocessed(const float* crops, int num_crops,
                                       std::vector<ColorResult>& results) {
    if (!engine_ || num_crops <= 0) {
        results.clear();
        return;
    }

    auto stream = static_cast<cudaStream_t>(stream_);
    results.clear();
    results.reserve(num_crops);

    for (int offset = 0; offset < num_crops; offset += max_batch_) {
        int batch_size = std::min(max_batch_, num_crops - offset);
        size_t input_bytes = batch_size * 3 * CROP_SIZE * CROP_SIZE * sizeof(float);

        // Copy preprocessed crops to GPU
        cudaMemcpyAsync(d_input_, crops + offset * 3 * CROP_SIZE * CROP_SIZE,
                        input_bytes, cudaMemcpyHostToDevice, stream);

        // Set batch size
        context_->setInputShape("input",
            nvinfer1::Dims4{batch_size, 3, CROP_SIZE, CROP_SIZE});

        // Run inference (tensor addresses set in load())
        context_->enqueueV3(stream);

        // Parse results
        std::vector<ColorResult> batch_results;
        softmax_and_parse(batch_size, batch_results);
        results.insert(results.end(), batch_results.begin(), batch_results.end());
    }
}

} // namespace rv
