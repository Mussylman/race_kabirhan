#pragma once
/**
 * color_infer.h — TensorRT color classifier with CUDA torso crop.
 *
 * Workflow per frame batch:
 *   1. For each person detection, compute torso ROI (10-40% height, 20% side margins)
 *   2. CUDA kernel: crop + resize 128x128 + NV12/RGBA→RGB + ImageNet normalize
 *   3. Batch TRT inference → softmax → color_id + confidence
 */

#include "config.h"

#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace nvinfer1 {
    class ICudaEngine;
    class IExecutionContext;
}

struct NvBufSurface;

namespace rv {

// Result of classifying one torso crop
struct ColorResult {
    uint32_t color_id   = COLOR_UNKNOWN;
    float    confidence = 0.0f;
    float    probs[NUM_COLORS] = {0};
};

// Torso ROI in pixel coordinates (within a single camera frame)
struct TorsoROI {
    int x1, y1, x2, y2;       // crop region in frame
    int det_index;             // index into detection array (for writing back)
    int cam_index;             // which camera (source_id)
    int surface_index;         // index into surfaceList (batch_id)
};

class ColorInfer {
public:
    ColorInfer();
    ~ColorInfer();

    /**
     * Load TensorRT engine for SimpleColorCNN.
     * @param engine_path Path to .engine file
     * @return true on success
     */
    bool load(const std::string& engine_path);

    /**
     * Classify a batch of torso crops from GPU surface.
     *
     * @param surface   NvBufSurface containing decoded frames (GPU memory)
     * @param rois      Torso ROIs to crop and classify
     * @param results   Output: one ColorResult per ROI
     */
    void classify(const NvBufSurface* surface,
                  const std::vector<TorsoROI>& rois,
                  std::vector<ColorResult>& results);

    /**
     * Classify pre-extracted CPU crops (for testing/validation).
     * @param crops     float[N][3][128][128] in ImageNet-normalized RGB
     * @param num_crops Number of crops
     * @param results   Output: one ColorResult per crop
     */
    void classify_preprocessed(const float* crops, int num_crops,
                               std::vector<ColorResult>& results);

    bool is_loaded() const { return engine_ != nullptr; }

    // Crop extraction parameters: torso sub-region (matches Python pipeline/analyzer.py)
    static constexpr float TORSO_TOP    = 0.10f;
    static constexpr float TORSO_BOTTOM = 0.40f;
    static constexpr float TORSO_LEFT   = 0.20f;
    static constexpr float TORSO_RIGHT  = 0.20f;

    /**
     * Compute torso ROI from person bounding box.
     * @param x1,y1,x2,y2  Person bbox in pixel coordinates
     * @param frame_w,frame_h  Frame dimensions (for clamping)
     * @return TorsoROI with clamped coordinates
     */
    static TorsoROI compute_torso_roi(float x1, float y1, float x2, float y2,
                                      int frame_w, int frame_h,
                                      int det_index = 0, int cam_index = 0);

private:
    // TensorRT engine and context
    nvinfer1::ICudaEngine*      engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // GPU buffers (pre-allocated once in load())
    void* d_input_  = nullptr;   // float[max_batch][3][CROP_SIZE][CROP_SIZE]
    void* d_output_ = nullptr;   // float[max_batch][model_num_classes_]
    int*  d_rois_   = nullptr;   // int[max_batch][4] — pre-allocated ROI buffer
    int   max_batch_ = 128;

    // v1: 5 classes (blue=0, green=1, purple=2, red=3, yellow=4) — matches SHM directly
    static constexpr int MODEL_NUM_CLASSES = 5;
    static constexpr uint32_t MODEL_TO_SHM[MODEL_NUM_CLASSES] = {
        COLOR_BLUE,    // model 0 → blue   (SHM 0)
        COLOR_GREEN,   // model 1 → green  (SHM 1)
        COLOR_PURPLE,  // model 2 → purple (SHM 2)
        COLOR_RED,     // model 3 → red    (SHM 3)
        COLOR_YELLOW,  // model 4 → yellow (SHM 4)
    };

    // CUDA stream for async ops
    void* stream_ = nullptr;     // cudaStream_t

    // Internal: run preprocessing CUDA kernel
    void preprocess_crops(const NvBufSurface* surface,
                          const std::vector<TorsoROI>& rois);

    // Internal: run softmax on output
    void softmax_and_parse(int num_crops, std::vector<ColorResult>& results);
};

} // namespace rv
