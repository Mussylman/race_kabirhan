/**
 * osnet_color_parser.cpp — SGIE classifier parser для OSNet x1.0 (market1501)
 * embedding-based color classification.
 *
 * Заменяет DINOv2 prototype-based, который имел 43% REF/PROD disagreement
 * из-за preprocessing math drift. OSNet pretrained показал 100% argmax
 * stability на DeepStream-симулированном preprocessing (median cos sim
 * REF↔PROD 0.9988) и 98.2% jockey accuracy на golden cam-13 test
 * (см. ~/race_vision_bench/osnet_test/).
 *
 * Input  : layer "embedding" [1, 512] FP32 (от OSNet TRT engine)
 * Output : NvDsInferAttribute с class_id из labels_color.txt
 *          (blue=0, green=1, purple=2, red=3, yellow=4)
 *          ИЛИ class_id=255 (COLOR_UNKNOWN) при reject (cos < threshold)
 *
 * Logic:
 *   1. L2-normalize input embedding
 *   2. Cosine similarity к 4 hardcoded prototypes (kOsnetProtos)
 *   3. Argmax + threshold check → class_id или 255
 *
 * Prototypes built from prototypes_osnet_combined.npz (1444 crops,
 * 17 cams, 5 sessions; multicam + color_dataset merged), L2-normalized FP32.
 * Threshold via env RV_OSNET_MIN_SIM (default 0.75 — выше чем DINOv2 0.55,
 * т.к. OSNet даёт более уверенные predictions с большим запасом над NJ)
 * AND classifier-threshold from sgie config (we apply max of two).
 *
 * Registered in sgie_color.txt via
 *   parse-classifier-func-name=NvDsInferClassifierParseCustomOsnet
 *   custom-lib-path=.../libnvdsinfer_racevision.so
 */

#include "nvdsinfer_custom_impl.h"
#include "osnet_prototypes_v1.h"

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
// 0.75 — best balanced на golden cam-13 (jockey_acc=98.2%, NJ_reject=99.2%).
static const float g_min_sim = rv_env_float("RV_OSNET_MIN_SIM", 0.75f);

extern "C" bool NvDsInferClassifierParseCustomOsnet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& /*networkInfo*/,
    float classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string& descString)
{
    static int call_counter = 0;
    if (call_counter++ < 6) {
        std::fprintf(stderr,
            "[OsnetParse #%d] layers=%zu cfg_thr=%.3f env_min_sim=%.3f\n",
            call_counter, outputLayersInfo.size(), classifierThreshold, g_min_sim);
        std::fflush(stderr);
    }
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* emb = reinterpret_cast<const float*>(layer.buffer);
    if (!emb) return false;

    // 1. L2 norm входного embedding
    float sumsq = 0.0f;
    for (int i = 0; i < rv::OSNET_DIM; ++i) sumsq += emb[i] * emb[i];
    const float norm = std::sqrt(sumsq) + 1e-12f;

    // 2. Cosine similarity к 4 prototypes (они уже L2-normalized в .h)
    float sims[rv::OSNET_NUM_PROTOS];
    for (int p = 0; p < rv::OSNET_NUM_PROTOS; ++p) {
        float dot = 0.0f;
        for (int i = 0; i < rv::OSNET_DIM; ++i) dot += emb[i] * rv::kOsnetProtos[p][i];
        sims[p] = dot / norm;
    }

    // 3. Argmax
    int best = 0;
    for (int p = 1; p < rv::OSNET_NUM_PROTOS; ++p)
        if (sims[p] > sims[best]) best = p;
    const float max_sim = sims[best];

    // 4. Reject path: emit attribute с class_id=255, conf=0
    const float effective_thr = std::max(g_min_sim, classifierThreshold);
    if (max_sim < effective_thr) {
        if (call_counter <= 6) {
            std::fprintf(stderr,
                "[OsnetParse] REJECT max_sim=%.3f < thr=%.3f (winner_was=%s)\n",
                max_sim, effective_thr, rv::kOsnetProtoLabels[best]);
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
    const unsigned int color_id = rv::kOsnetClassMapping[best];

    if (call_counter <= 6) {
        std::fprintf(stderr,
            "[OsnetParse] best=%d (%s, mapped=%u) max_sim=%.3f\n",
            best, rv::kOsnetProtoLabels[best], color_id, max_sim);
        std::fflush(stderr);
    }

    // 6. Emit attribute. Label format совместим с pipeline.py probe split на '|':
    //    "<color>|<confidence>|<logit>" — для OSNet logit = max_sim (тот же скор)
    char labelbuf[48];
    std::snprintf(labelbuf, sizeof(labelbuf), "%s|%.3f|%.2f",
                  rv::kOsnetProtoLabels[best], max_sim, max_sim);

    NvDsInferAttribute attr;
    attr.attributeIndex = 0;
    attr.attributeValue = color_id;
    attr.attributeConfidence = max_sim;
    attr.attributeLabel = strdup(labelbuf);
    attrList.emplace_back(attr);
    descString = "osnet";
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassifierParseCustomOsnet);
