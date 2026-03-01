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
static constexpr int CROP_SIZE = 64;

/**
 * CUDA kernel: For each output pixel (in 64x64 crop), bilinear sample
 * from the source RGBA frame, convert to RGB, and apply ImageNet normalization.
 *
 * Output layout: [batch_idx, channel, y, x] (NCHW)
 */
__global__ void crop_resize_normalize_kernel(
    const uint8_t* __restrict__ src,     // source frame (RGBA, pitch-linear)
    int src_width, int src_height, int src_pitch,
    const int* __restrict__ rois,        // [N, 4] = {x1, y1, x2, y2}
    float* __restrict__ dst,             // [N, 3, 64, 64]
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

    // Map output pixel to source coordinate (bilinear)
    float sx = roi_x1 + (out_x + 0.5f) * roi_w / CROP_SIZE;
    float sy = roi_y1 + (out_y + 0.5f) * roi_h / CROP_SIZE;

    // Nearest-neighbor (good enough for 64x64 target)
    int ix = min(max(static_cast<int>(sx), 0), src_width - 1);
    int iy = min(max(static_cast<int>(sy), 0), src_height - 1);

    // Read RGBA pixel
    const uint8_t* pixel = src + iy * src_pitch + ix * 4;
    float r = pixel[0] / 255.0f;
    float g = pixel[1] / 255.0f;
    float b = pixel[2] / 255.0f;

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

// ── ColorInfer implementation ──────────────────────────────────────

ColorInfer::ColorInfer() = default;

ColorInfer::~ColorInfer() {
    delete context_;
    delete engine_;
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
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

    // Allocate GPU buffers
    size_t input_bytes  = max_batch_ * 3 * CROP_SIZE * CROP_SIZE * sizeof(float);
    size_t output_bytes = max_batch_ * NUM_COLORS * sizeof(float);
    cudaMalloc(&d_input_,  input_bytes);
    cudaMalloc(&d_output_, output_bytes);

    cudaStream_t s;
    cudaStreamCreate(&s);
    stream_ = s;

    fprintf(stderr, "[ColorInfer] Loaded engine: %s (max_batch=%d)\n",
            engine_path.c_str(), max_batch_);
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

    int num_crops = static_cast<int>(rois.size());
    auto stream = static_cast<cudaStream_t>(stream_);

    // Upload ROIs to GPU
    int* d_rois = nullptr;
    cudaMalloc(&d_rois, num_crops * 4 * sizeof(int));
    std::vector<int> h_rois(num_crops * 4);
    for (int i = 0; i < num_crops; ++i) {
        h_rois[i * 4 + 0] = rois[i].x1;
        h_rois[i * 4 + 1] = rois[i].y1;
        h_rois[i * 4 + 2] = rois[i].x2;
        h_rois[i * 4 + 3] = rois[i].y2;
    }
    cudaMemcpyAsync(d_rois, h_rois.data(), num_crops * 4 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Process crops per camera surface
    // Each batch element in nvstreammux maps to a surface index
    for (int i = 0; i < num_crops; ++i) {
        int cam_idx = rois[i].cam_index;
        if (cam_idx >= static_cast<int>(surface->numFilled)) continue;

        const auto& params = surface->surfaceList[cam_idx];
        const uint8_t* src = static_cast<const uint8_t*>(params.dataPtr);
        int src_w = params.width;
        int src_h = params.height;
        int src_pitch = params.pitch;

        // Launch kernel for crops belonging to this camera
        int total_pixels = CROP_SIZE * CROP_SIZE;  // per crop
        int threads = 256;
        int blocks = (total_pixels + threads - 1) / threads;

        // Note: in production, we batch all crops from same camera
        // For simplicity, process one crop at a time here
        crop_resize_normalize_kernel<<<blocks, threads, 0, stream>>>(
            src, src_w, src_h, src_pitch,
            d_rois + i * 4,
            static_cast<float*>(d_input_) + i * 3 * CROP_SIZE * CROP_SIZE,
            1);
    }

    cudaFree(d_rois);
}

void ColorInfer::softmax_and_parse(int num_crops, std::vector<ColorResult>& results) {
    // Copy output from GPU
    std::vector<float> h_output(num_crops * NUM_COLORS);
    cudaMemcpyAsync(h_output.data(), d_output_,
                    num_crops * NUM_COLORS * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));

    results.resize(num_crops);
    for (int i = 0; i < num_crops; ++i) {
        const float* logits = h_output.data() + i * NUM_COLORS;

        // Softmax
        float max_val = *std::max_element(logits, logits + NUM_COLORS);
        float sum = 0.0f;
        float probs[NUM_COLORS];
        for (int c = 0; c < NUM_COLORS; ++c) {
            probs[c] = std::exp(logits[c] - max_val);
            sum += probs[c];
        }
        for (int c = 0; c < NUM_COLORS; ++c) {
            probs[c] /= sum;
        }

        // Find best
        int best = 0;
        for (int c = 1; c < NUM_COLORS; ++c) {
            if (probs[c] > probs[best]) best = c;
        }

        results[i].color_id   = static_cast<uint32_t>(best);
        results[i].confidence = probs[best];
        std::memcpy(results[i].probs, probs, sizeof(probs));
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

        // Run inference
        void* bindings[2] = {d_input_, d_output_};
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

        // Run inference
        void* bindings[2] = {d_input_, d_output_};
        context_->enqueueV3(stream);

        // Parse results
        std::vector<ColorResult> batch_results;
        softmax_and_parse(batch_size, batch_results);
        results.insert(results.end(), batch_results.begin(), batch_results.end());
    }
}

} // namespace rv
