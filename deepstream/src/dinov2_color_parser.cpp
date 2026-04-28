/**
 * dinov2_color_parser.cpp — SGIE classifier parser для DINOv2-base
 * embedding-based color classification.
 *
 * Заменяет SimpleColorCNN (color_classifier_v4) который имел 0% reject
 * на not_jockey. DINOv2 prototype-based @ similarity threshold даёт
 * 98.8% reject rate с сохранением ~80% jockey accuracy (см. verdict.md).
 *
 * Input  : layer "embedding" [1, 768] FP32 (от DINOv2 TRT engine)
 * Output : NvDsInferAttribute с class_id из labels_color.txt
 *          (blue=0, green=1, purple=2, red=3, yellow=4)
 *          ИЛИ class_id=255 (COLOR_UNKNOWN) при reject (cos < threshold)
 *
 * Logic:
 *   1. L2-normalize input embedding
 *   2. Cosine similarity к 4 hardcoded prototypes (kDinov2Protos)
 *   3. Argmax + threshold check → class_id или 255
 *
 * Prototypes built from cam-24 reference video, L2-normalized FP32.
 * Threshold via env RV_DINOV2_MIN_SIM (default 0.55) AND classifier-threshold
 * from sgie config (we apply max of two).
 *
 * Registered in sgie_color.txt via
 *   parse-classifier-func-name=NvDsInferClassifierParseCustomDinov2
 *   custom-lib-path=.../libnvdsinfer_racevision.so
 */

#include "nvdsinfer_custom_impl.h"
#include "dinov2_prototypes_v1.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

constexpr unsigned int COLOR_UNKNOWN_ID = 255;

static float rv_env_float(const char* name, float dflt) {
    const char* s = std::getenv(name);
    if (!s || !*s) return dflt;
    try { return std::stof(s); } catch (...) { return dflt; }
}

// Reject если max cosine similarity < этого порога.
// 0.55 — best composite на zero-shot Phase 3 (см. eval_report.md).
static const float g_min_sim = rv_env_float("RV_DINOV2_MIN_SIM", 0.55f);

extern "C" bool NvDsInferClassifierParseCustomDinov2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string& descString)
{
    static int call_counter = 0;
    if (call_counter++ < 6) {
        std::fprintf(stderr,
            "[Dinov2Parse #%d] layers=%zu cfg_thr=%.3f env_min_sim=%.3f\n",
            call_counter, outputLayersInfo.size(), classifierThreshold, g_min_sim);
        std::fflush(stderr);
    }
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* emb = reinterpret_cast<const float*>(layer.buffer);
    if (!emb) return false;

    // 1. L2 norm входного embedding
    float sumsq = 0.0f;
    for (int i = 0; i < rv::DINOV2_DIM; ++i) sumsq += emb[i] * emb[i];
    const float norm = std::sqrt(sumsq) + 1e-12f;

    // 2. Cosine similarity к 4 prototypes (они уже L2-normalized в .h)
    float sims[rv::DINOV2_NUM_PROTOS];
    for (int p = 0; p < rv::DINOV2_NUM_PROTOS; ++p) {
        float dot = 0.0f;
        for (int i = 0; i < rv::DINOV2_DIM; ++i) dot += emb[i] * rv::kDinov2Protos[p][i];
        sims[p] = dot / norm;
    }

    // 3. Argmax
    int best = 0;
    for (int p = 1; p < rv::DINOV2_NUM_PROTOS; ++p)
        if (sims[p] > sims[best]) best = p;
    const float max_sim = sims[best];

    // 4. Reject path: emit attribute с class_id=255, conf=0
    const float effective_thr = std::max(g_min_sim, classifierThreshold);
    if (max_sim < effective_thr) {
        if (call_counter <= 6) {
            std::fprintf(stderr,
                "[Dinov2Parse] REJECT max_sim=%.3f < thr=%.3f (winner_was=%s)\n",
                max_sim, effective_thr, rv::kDinov2ProtoLabels[best]);
            std::fflush(stderr);
        }
        NvDsInferAttribute attr;
        attr.attributeIndex = 0;
        attr.attributeValue = COLOR_UNKNOWN_ID;
        attr.attributeConfidence = 0.0f;
        attr.attributeLabel = strdup("unknown|0.000|0.00");
        attrList.emplace_back(attr);
        descString = "unknown";
        return true;
    }

    // 5. Map наш prototype index → labels_color.txt class_id
    const unsigned int color_id = rv::kDinov2ClassMapping[best];

    if (call_counter <= 6) {
        std::fprintf(stderr,
            "[Dinov2Parse] best=%d (%s, mapped=%u) max_sim=%.3f\n",
            best, rv::kDinov2ProtoLabels[best], color_id, max_sim);
        std::fflush(stderr);
    }

    // 6. Emit attribute. Label format совместим с pipeline.py probe split на '|':
    //    "<color>|<confidence>|<logit>" — для DINOv2 logit = max_sim (тот же скор)
    char labelbuf[48];
    std::snprintf(labelbuf, sizeof(labelbuf), "%s|%.3f|%.2f",
                  rv::kDinov2ProtoLabels[best], max_sim, max_sim);

    NvDsInferAttribute attr;
    attr.attributeIndex = 0;
    attr.attributeValue = color_id;
    attr.attributeConfidence = max_sim;
    attr.attributeLabel = strdup(labelbuf);
    attrList.emplace_back(attr);
    descString = rv::kDinov2ProtoLabels[best];
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassifierParseCustomDinov2);
