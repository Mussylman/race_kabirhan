#pragma once
/**
 * yolo_parser.h — Custom YOLOv8 output parser for DeepStream nvinfer.
 *
 * Parses the raw TensorRT output tensor from YOLOv8s into NvDsInferObjectDetectionInfo.
 * YOLOv8 output format: [batch, 84, 8400] where 84 = 4 bbox + 80 classes.
 * We only care about class 0 (person).
 */

#include <vector>
#include "nvdsinfer_custom_impl.h"

namespace rv {

/**
 * Parse YOLOv8 detection output.
 * Registered as custom bounding-box parser in nvinfer config.
 *
 * @param outputLayers    nvinfer output layer info
 * @param networkInfo     input dimensions
 * @param detectionParams confidence threshold etc.
 * @param objectList      output detection list
 * @return true on success
 */
bool parseYoloV8(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

} // namespace rv

// C-linkage function for nvinfer plugin registration
extern "C" bool NvDsInferParseYoloV8(
    const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);
