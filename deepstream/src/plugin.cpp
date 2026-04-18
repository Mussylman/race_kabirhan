/**
 * plugin.cpp — C API entry points for the Race Vision plugin.
 *
 * This file exposes C-callable wrappers around ColorInfer and ShmWriter
 * so they can be loaded from Python via ctypes (used by the new
 * pyservicemaker-based pipeline in deepstream/pipeline.py).
 *
 * The YOLO parser (yolo_parser.cpp) already exports NvDsInferParseYoloV8
 * with C linkage and does not need wrapping here.
 *
 * Linkage:
 *   libnvdsinfer_racevision.so contains:
 *     - NvDsInferParseYoloV8       (used by nvinfer via parse-bbox-func-name)
 *     - rv_color_*                 (used by Python ctypes)
 *     - rv_shm_*                   (used by Python ctypes)
 */

#include "color_infer.h"
#include "shm_writer.h"
#include "config.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <memory>

using rv::ColorInfer;
using rv::ColorResult;
using rv::ShmWriter;
using rv::CameraSlot;

extern "C" {

// ── Color inference ─────────────────────────────────────────────────

/**
 * Create a ColorInfer instance and load the TRT engine.
 * @return opaque handle, or nullptr on failure
 */
void* rv_color_create(const char* engine_path) {
    if (!engine_path) return nullptr;
    auto* ci = new ColorInfer();
    if (!ci->load(engine_path)) {
        delete ci;
        return nullptr;
    }
    return static_cast<void*>(ci);
}

void rv_color_destroy(void* handle) {
    if (!handle) return;
    delete static_cast<ColorInfer*>(handle);
}

/**
 * Classify pre-extracted CPU crops.
 *
 * @param handle      ColorInfer instance from rv_color_create
 * @param crops_ptr   float[num_crops][3][128][128] in ImageNet-normalized RGB
 * @param num_crops   number of crops
 * @param out_ids     uint32_t[num_crops] — output color IDs (0=blue,1=green,2=purple,3=red,4=yellow)
 * @param out_confs   float[num_crops] — output top-1 confidences
 * @return 0 on success, -1 on failure
 */
int rv_color_classify(void* handle, const float* crops_ptr, int num_crops,
                       uint32_t* out_ids, float* out_confs) {
    if (!handle || !crops_ptr || num_crops <= 0) return -1;
    auto* ci = static_cast<ColorInfer*>(handle);
    if (!ci->is_loaded()) return -1;

    std::vector<ColorResult> results;
    ci->classify_preprocessed(crops_ptr, num_crops, results);

    if (static_cast<int>(results.size()) != num_crops) return -1;
    for (int i = 0; i < num_crops; ++i) {
        if (out_ids)   out_ids[i]   = results[i].color_id;
        if (out_confs) out_confs[i] = results[i].confidence;
    }
    return 0;
}

// ── SHM writer ─────────────────────────────────────────────────────

/**
 * Create the /rv_detections shared memory segment.
 *
 * @param num_cameras  number of camera slots to allocate
 * @param cam_ids      char* array of length num_cameras with camera IDs
 * @return opaque handle, or nullptr on failure
 */
void* rv_shm_create(uint32_t num_cameras, const char* const* cam_ids) {
    if (num_cameras == 0 || !cam_ids) return nullptr;
    auto* sw = new ShmWriter();
    std::vector<std::string> ids;
    ids.reserve(num_cameras);
    for (uint32_t i = 0; i < num_cameras; ++i) {
        ids.emplace_back(cam_ids[i] ? cam_ids[i] : "");
    }
    if (!sw->create(num_cameras, ids)) {
        delete sw;
        return nullptr;
    }
    return static_cast<void*>(sw);
}

void rv_shm_destroy(void* handle) {
    if (!handle) return;
    auto* sw = static_cast<ShmWriter*>(handle);
    sw->destroy();
    delete sw;
}

/**
 * Write a CameraSlot's raw bytes for one camera.
 *
 * @param handle     ShmWriter instance
 * @param cam_index  slot index [0, num_cameras)
 * @param slot_data  pointer to a populated CameraSlot struct
 *                   (Python ctypes Structure with the same layout as config.h)
 */
void rv_shm_write_camera(void* handle, uint32_t cam_index, const void* slot_data) {
    if (!handle || !slot_data) return;
    auto* sw = static_cast<ShmWriter*>(handle);
    const CameraSlot* slot = static_cast<const CameraSlot*>(slot_data);
    sw->write_camera(cam_index, *slot);
}

void rv_shm_commit(void* handle) {
    if (!handle) return;
    static_cast<ShmWriter*>(handle)->commit();
}

// ── Constants exported for Python sanity checks ─────────────────────

uint32_t rv_get_max_cameras()    { return rv::MAX_CAMERAS; }
uint32_t rv_get_max_detections() { return rv::MAX_DETECTIONS; }
uint32_t rv_get_num_colors()     { return rv::NUM_COLORS; }
uint32_t rv_get_camera_slot_size() { return sizeof(CameraSlot); }
uint32_t rv_get_detection_size()   { return sizeof(rv::Detection); }

} // extern "C"
