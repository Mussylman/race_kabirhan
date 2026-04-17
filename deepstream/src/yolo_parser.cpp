/**
 * yolo_parser.cpp — YOLOv8 output parser for DeepStream nvinfer.
 *
 * YOLOv8 output tensor shape: [batch, 84, 8400]
 *   - 84 = 4 (cx, cy, w, h) + 80 COCO classes
 *   - 8400 = sum of detection grid cells at 3 scales
 *   - We only keep class 0 (person) detections
 *
 * The output is transposed vs YOLOv5: features are in dim=1, not dim=2.
 */

#include "yolo_parser.h"

#include <cmath>
#include <algorithm>
#include <cstring>

namespace rv {

// Target class index (person in COCO, jockey in custom — both class 0)
static constexpr int TARGET_CLASS = 0;
// Number of bbox params (cx, cy, w, h)
static constexpr int BBOX_PARAMS = 4;
// NMS IoU threshold
static constexpr float NMS_IOU_THRESH = 0.50f;

struct RawDetection {
    float x1, y1, x2, y2;
    float conf;
};

static float iou(const RawDetection& a, const RawDetection& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static void nms(std::vector<RawDetection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const RawDetection& a, const RawDetection& b) {
                  return a.conf > b.conf;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<RawDetection> result;
    result.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    dets = std::move(result);
}

bool parseYoloV8(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayers.empty()) return false;

    const NvDsInferLayerInfo& layer = outputLayers[0];
    const float* data = static_cast<const float*>(layer.buffer);
    if (!data) return false;

    // YOLOv8 output: [num_features, num_dets] or [num_dets, num_features]
    //   COCO:   num_features=84  (4 bbox + 80 classes)
    //   Custom: num_features=5   (4 bbox + 1 class)
    int dim0 = layer.inferDims.d[0];
    int dim1 = layer.inferDims.d[1];
    int ndims = layer.inferDims.numDims;

    // Auto-detect layout: features dim is the smaller one (>= 5)
    int num_features, num_dets;
    bool transposed = false;
    if (dim0 >= BBOX_PARAMS + 1 && dim0 < dim1) {
        // Normal: [features, num_dets] e.g. [84, 8400] or [5, 13125]
        num_features = dim0;
        num_dets = dim1;
    } else if (dim1 >= BBOX_PARAMS + 1 && dim1 < dim0) {
        // Transposed: [num_dets, features]
        num_features = dim1;
        num_dets = dim0;
        transposed = true;
    } else {
        num_features = dim0;
        num_dets = dim1;
    }

    int num_classes = num_features - BBOX_PARAMS;

    // Debug logging (first call only)
    static bool debug_done = false;
    if (!debug_done) {
        fprintf(stderr, "[YoloParser] ndims=%d dim0=%d dim1=%d => features=%d dets=%d classes=%d transposed=%d\n",
                ndims, dim0, dim1, num_features, num_dets, num_classes, transposed);
        debug_done = true;
    }

    if (num_features < BBOX_PARAMS + 1) return false;

    const float conf_thresh = detectionParams.perClassPreclusterThreshold.size() > 0
        ? detectionParams.perClassPreclusterThreshold[0]
        : 0.35f;

    std::vector<RawDetection> raw_dets;
    raw_dets.reserve(256);

    for (int i = 0; i < num_dets; ++i) {
        float cx, cy, w, h, conf;
        if (!transposed) {
            // [84, num_dets] layout
            cx   = data[0 * num_dets + i];
            cy   = data[1 * num_dets + i];
            w    = data[2 * num_dets + i];
            h    = data[3 * num_dets + i];
            conf = data[(BBOX_PARAMS + TARGET_CLASS) * num_dets + i];
        } else {
            // [num_dets, 84] layout
            cx   = data[i * num_features + 0];
            cy   = data[i * num_features + 1];
            w    = data[i * num_features + 2];
            h    = data[i * num_features + 3];
            conf = data[i * num_features + BBOX_PARAMS + TARGET_CLASS];
        }

        if (conf < conf_thresh) continue;

        // Convert cx,cy,w,h to x1,y1,x2,y2 in PIXEL coords (network scale)
        // DeepStream expects coords in network input resolution, NOT normalized!
        RawDetection det;
        det.x1   = cx - w / 2.0f;
        det.y1   = cy - h / 2.0f;
        det.x2   = cx + w / 2.0f;
        det.y2   = cy + h / 2.0f;
        det.conf = conf;

        // Clamp to network dimensions
        det.x1 = std::max(0.0f, std::min(static_cast<float>(networkInfo.width),  det.x1));
        det.y1 = std::max(0.0f, std::min(static_cast<float>(networkInfo.height), det.y1));
        det.x2 = std::max(0.0f, std::min(static_cast<float>(networkInfo.width),  det.x2));
        det.y2 = std::max(0.0f, std::min(static_cast<float>(networkInfo.height), det.y2));

        raw_dets.push_back(det);
    }

    // Apply NMS
    nms(raw_dets, NMS_IOU_THRESH);

    // Convert to DeepStream format (pixel coords in network input space)
    for (const auto& d : raw_dets) {
        NvDsInferObjectDetectionInfo obj;
        obj.classId       = TARGET_CLASS;
        obj.detectionConfidence = d.conf;
        obj.left   = d.x1;
        obj.top    = d.y1;
        obj.width  = d.x2 - d.x1;
        obj.height = d.y2 - d.y1;
        objectList.push_back(obj);
    }

    return true;
}

} // namespace rv

// C-linkage wrapper for nvinfer plugin
extern "C" bool NvDsInferParseYoloV8(
    const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    return rv::parseYoloV8(outputLayersInfo, networkInfo, detectionParams, objectList);
}
