#pragma once
/**
 * diag_logger.h — Diagnostic logger for DeepStream pipeline.
 *
 * Saves per-detection CSV log and annotated frame snapshots (JPG)
 * for offline analysis. Auto-creates experiment directories:
 *   <log_dir>/exp1/, exp2/, exp3/, ...
 *
 * CSV columns:
 *   batch,cam_id,frame_w,frame_h,det_idx,x1,y1,x2,y2,center_x,
 *   det_conf,color,color_conf,prob_blue,prob_green,prob_purple,prob_red,prob_yellow,track_id
 */

#include "config.h"

#include <string>
#include <vector>
#include <cstdio>
#include <mutex>

struct NvBufSurface;

namespace rv {

struct DiagDetection {
    std::string cam_id;
    int frame_w, frame_h;
    int det_idx;
    float x1, y1, x2, y2;
    float center_x;
    float det_conf;
    uint32_t color_id;
    float color_conf;
    float color_probs[NUM_COLORS];
    uint32_t track_id;
};

class DiagLogger {
public:
    DiagLogger() = default;
    ~DiagLogger();

    /**
     * Initialize logger. Creates exp_dir = <base_dir>/expN/
     * @param base_dir  Root log directory (e.g. "ds_results")
     * @param snap_interval  Save JPG every N batches (0 = every batch with detections)
     * @return true on success
     */
    bool init(const std::string& base_dir, int snap_interval = 10);

    /**
     * Log detections from one batch. Thread-safe.
     * @param batch_num  Current batch counter
     * @param detections  All detections in this batch
     */
    void log_detections(int batch_num, const std::vector<DiagDetection>& detections);

    /**
     * Save annotated frame snapshot from GPU surface.
     * Downloads frame from GPU, draws bboxes+labels, saves as JPG.
     * @param batch_num   Batch counter (for filename)
     * @param surface     NvBufSurface with decoded frames
     * @param cam_index   Which camera surface to save
     * @param cam_id      Camera name
     * @param detections  Detections to draw on this camera's frame
     */
    void save_snapshot(int batch_num, const NvBufSurface* surface,
                       int cam_index, const std::string& cam_id,
                       const std::vector<DiagDetection>& cam_dets);

    bool is_enabled() const { return enabled_; }
    std::string get_exp_dir() const { return exp_dir_; }

private:
    bool enabled_ = false;
    std::string exp_dir_;
    FILE* csv_file_ = nullptr;
    std::mutex mutex_;
    int snap_interval_ = 10;
    int snap_counter_ = 0;
};

} // namespace rv
