/**
 * color_classifier_parser.cpp — Custom classifier output parser for SGIE.
 *
 * DeepStream 9.0 auto-parser does not emit NvDsInferAttribute for our
 * color_classifier_v4 ONNX (output shape [batch, 5] — 2D, not the 4D
 * [batch, C, 1, 1] that the built-in parser expects). This function
 * manually softmaxes the logits and emits one attribute per detection.
 *
 * Registered in sgie_color.txt via
 *   parse-classifier-func-name=NvDsInferClassiferParseCustomColor
 *   custom-lib-path=.../libnvdsinfer_racevision.so
 */

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Order MUST match deepstream/configs/labels_color.txt
static const char* kColorLabels[5] = {"blue", "green", "purple", "red", "yellow"};

extern "C" bool NvDsInferClassiferParseCustomColor(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string& descString)
{
    static int call_counter = 0;
    if (call_counter++ < 6) {
        fprintf(stderr, "[ColorParse #%d] layers=%zu threshold=%.3f\n",
                call_counter, outputLayersInfo.size(), classifierThreshold);
        fflush(stderr);
    }
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* logits = reinterpret_cast<const float*>(layer.buffer);
    if (!logits) return false;

    constexpr int N = 5;
    // Softmax (numerically stable)
    float maxv = logits[0];
    for (int i = 1; i < N; ++i) if (logits[i] > maxv) maxv = logits[i];
    float probs[N];
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        probs[i] = std::exp(logits[i] - maxv);
        sum += probs[i];
    }
    if (sum <= 0.0f) return false;
    for (int i = 0; i < N; ++i) probs[i] /= sum;

    int best = 0;
    for (int i = 1; i < N; ++i) if (probs[i] > probs[best]) best = i;
    if (probs[best] < classifierThreshold) return true;  // OK, just no attr

    if (call_counter <= 6) {
        fprintf(stderr, "[ColorParse] best=%d (%s) prob=%.3f\n",
                best, kColorLabels[best], probs[best]);
        fflush(stderr);
    }

    // Encode confidence into the label string ("red|0.95") because
    // pyservicemaker does NOT expose attr.attributeConfidence via Python.
    // Probe splits on '|' and strips before OSD.
    char labelbuf[32];
    std::snprintf(labelbuf, sizeof(labelbuf), "%s|%.3f",
                  kColorLabels[best], probs[best]);

    NvDsInferAttribute attr;
    attr.attributeIndex = 0;
    attr.attributeValue = static_cast<unsigned int>(best);
    attr.attributeConfidence = probs[best];
    attr.attributeLabel = strdup(labelbuf);
    attrList.emplace_back(attr);
    descString = kColorLabels[best];  // clean label for OSD/descString
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseCustomColor);
