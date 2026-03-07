#pragma once
/**
 * config.h — Shared memory struct definitions for Race Vision.
 *
 * These structs define the binary protocol between the DeepStream C++
 * pipeline and the Python FastAPI server via POSIX shared memory.
 *
 * IMPORTANT: Keep in sync with pipeline/shm_reader.py
 */

#include <cstdint>
#include <cstring>
#include <string>

namespace rv {

// ── Constants ───────────────────────────────────────────────────────

static constexpr int MAX_CAMERAS      = 25;
static constexpr int MAX_DETECTIONS   = 20;
static constexpr int NUM_COLORS       = 5;
static constexpr int CAM_ID_LEN       = 16;

static constexpr const char* SHM_NAME = "/rv_detections";
static constexpr const char* SEM_NAME = "/rv_detections_sem";

// Color indices (must match Python ALL_COLORS order)
enum ColorId : uint32_t {
    COLOR_BLUE   = 0,
    COLOR_GREEN  = 1,
    COLOR_PURPLE = 2,
    COLOR_RED    = 3,
    COLOR_YELLOW = 4,
    COLOR_UNKNOWN = 255,
};

static constexpr const char* COLOR_NAMES[NUM_COLORS] = {
    "blue", "green", "purple", "red", "yellow"
};

// ── Detection (single person detection within one camera) ───────────

struct __attribute__((packed)) Detection {
    float    x1;                    // bbox left
    float    y1;                    // bbox top
    float    x2;                    // bbox right
    float    y2;                    // bbox bottom
    float    center_x;             // (x1+x2)/2
    float    det_conf;             // YOLO detection confidence
    uint32_t color_id;             // ColorId enum
    float    color_conf;           // top-1 color confidence
    float    color_probs[NUM_COLORS]; // softmax probabilities [blue,green,purple,red,yellow]
    uint32_t track_id;             // nvtracker persistent object ID (0 = untracked)
};
static_assert(sizeof(Detection) == 56, "Detection must be 56 bytes");

// ── CameraSlot (one camera's detection results) ─────────────────────

struct __attribute__((packed)) CameraSlot {
    char     cam_id[CAM_ID_LEN];   // null-terminated camera ID, e.g. "cam-01"
    uint64_t timestamp_us;         // microseconds since epoch
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t num_detections;       // 0..MAX_DETECTIONS
    uint32_t _pad;
    Detection detections[MAX_DETECTIONS];
};
static_assert(sizeof(CameraSlot) == 16 + 8 + 4 + 4 + 4 + 4 + 56 * MAX_DETECTIONS,
              "CameraSlot layout mismatch");

// ── ShmHeader (top-level shared memory layout) ──────────────────────

struct __attribute__((packed)) ShmHeader {
    uint64_t write_seq;            // monotonic counter, atomically incremented after write
    uint32_t num_cameras;          // actual number of cameras configured
    uint32_t _reserved;
    CameraSlot cameras[MAX_CAMERAS];
};

static constexpr size_t SHM_SIZE = sizeof(ShmHeader);

// ── Trigger shared memory (lightweight camera activation state) ──

static constexpr const char* TRIGGER_SHM_NAME = "/rv_trigger";
static constexpr const char* TRIGGER_SEM_NAME = "/rv_trigger_sem";

struct __attribute__((packed)) TriggerShmHeader {
    uint64_t write_seq;                    // monotonic counter
    uint32_t active_mask;                  // bitmask of active cameras (bit i = camera i)
    uint32_t num_cameras;                  // configured camera count
    uint32_t detection_counts[MAX_CAMERAS]; // person detections per camera
    uint64_t timestamp_us;                 // microseconds since epoch
};

static constexpr size_t TRIGGER_SHM_SIZE = sizeof(TriggerShmHeader);

// ── Camera configuration ──────────────────────────────────────────────

struct CameraConfig {
    std::string id;
    std::string url;
    float track_start = 0.0f;
    float track_end   = 100.0f;
};

// ── Helpers ─────────────────────────────────────────────────────────

inline void init_camera_slot(CameraSlot& slot, const char* cam_id) {
    std::memset(&slot, 0, sizeof(CameraSlot));
    std::strncpy(slot.cam_id, cam_id, CAM_ID_LEN - 1);
    slot.cam_id[CAM_ID_LEN - 1] = '\0';
}

inline void init_detection(Detection& det) {
    std::memset(&det, 0, sizeof(Detection));
    det.color_id = COLOR_UNKNOWN;
}

} // namespace rv
