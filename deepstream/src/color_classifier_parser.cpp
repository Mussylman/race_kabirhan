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

// Additional filter thresholds (env-driven, read once):
//   RV_MIN_LOGIT   — drop when max raw logit < this (all classes weak).
//                    Default -9999 = disabled.
//   RV_MIN_MARGIN  — drop when (top1_logit - top2_logit) < this. Computed
//                    on LOGITS (not softmax probs) because softmax saturates
//                    and prob-margin is almost always ~1.0. Default 0 = off.
static float rv_env_float(const char* name, float dflt) {
    const char* s = std::getenv(name);
    if (!s || !*s) return dflt;
    try { return std::stof(s); } catch (...) { return dflt; }
}
// Disabled by default — filtering is now done in TimeTracker
// (per-color logit check inside pack confirmation).
static float g_min_logit  = rv_env_float("RV_MIN_LOGIT",  -9999.0f);
static float g_min_margin = rv_env_float("RV_MIN_MARGIN", -9999.0f);

extern "C" bool NvDsInferClassiferParseCustomColor(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string& descString)
{
    static int call_counter = 0;
    if (call_counter++ < 6) {
        fprintf(stderr, "[ColorParse #%d] layers=%zu threshold=%.3f min_logit=%.2f min_margin=%.2f\n",
                call_counter, outputLayersInfo.size(), classifierThreshold,
                g_min_logit, g_min_margin);
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

    // Margin on LOGITS (softmax saturates → prob-margin nearly useless).
    //   top1_logit = logits[best]
    //   top2_logit = max logit among the other 4 classes
    float top1_logit = logits[best];
    float top2_logit = -1e30f;
    for (int i = 0; i < N; ++i)
        if (i != best && logits[i] > top2_logit) top2_logit = logits[i];
    float logit_margin = top1_logit - top2_logit;

    if (maxv < g_min_logit) return true;           // weak activation overall
    if (logit_margin < g_min_margin) return true;  // ambiguous pick
    if (probs[best] < classifierThreshold) return true;

    if (call_counter <= 6) {
        fprintf(stderr, "[ColorParse] best=%d (%s) prob=%.3f logit=%.2f margin=%.2f\n",
                best, kColorLabels[best], probs[best], maxv, logit_margin);
        fflush(stderr);
    }

    // Encode "color|prob|logit" into the label string because
    // pyservicemaker does NOT expose attr.attributeConfidence via Python.
    // Probe splits on '|' and strips before OSD.
    char labelbuf[48];
    std::snprintf(labelbuf, sizeof(labelbuf), "%s|%.3f|%.2f",
                  kColorLabels[best], probs[best], maxv);

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
