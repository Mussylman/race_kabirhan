#pragma once
/**
 * pipeline.h — DeepStream GStreamer pipeline for multi-camera RTSP decode + inference.
 *
 * Pipeline topology:
 *   N x uridecodebin (RTSP → NVDEC hw decode)
 *     → nvstreammux (batch all cameras)
 *       → nvinfer (YOLOv8s TRT FP16)
 *         → nvtracker (IOU tracker for persistent object IDs)
 *           → probe callback (filter + crop + color classify + SHM write)
 *             → fakesink
 */

#include "config.h"
#include "shm_writer.h"
#include "color_infer.h"
#include "diag_logger.h"

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <gst/gst.h>

namespace rv {

struct PipelineConfig {
    std::vector<CameraConfig> cameras;
    std::string yolo_engine_path;
    std::string color_engine_path;
    int batch_size       = 25;
    int mux_width        = 800;
    int mux_height       = 800;
    float det_conf       = 0.35f;
    int mux_batched_push_timeout = 40000; // microseconds (40ms for RTSP, override for files)
    bool live_source     = true;   // false for file:// playback
    bool display         = false;  // --display: show video with OSD + tiler
    bool display_only    = false;  // --display-only: video grid without inference
    int display_width    = 1280;
    int display_height   = 720;
    std::string log_dir;           // --log-dir: diagnostic logging (CSV + JPG snapshots)
    int snap_interval    = 10;     // save snapshot every N batches (0=every batch with dets)
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    /**
     * Build the GStreamer pipeline from config.
     * @return true on success
     */
    bool build(const PipelineConfig& config);

    /**
     * Start the pipeline (begins RTSP decode + inference).
     */
    void start();

    /**
     * Stop the pipeline.
     */
    void stop();

    /**
     * Get the GMainLoop (for g_main_loop_run in main thread).
     */
    GMainLoop* get_main_loop() const { return loop_; }

    /**
     * Check if pipeline is running.
     */
    bool is_running() const { return running_; }

private:
    GstElement*  pipeline_   = nullptr;
    GstElement*  streammux_  = nullptr;
    GstElement*  tiler_      = nullptr;   // for show-source focus
    GMainLoop*   loop_       = nullptr;
    bool         running_    = false;

    PipelineConfig config_;
    ShmWriter      shm_writer_;
    ColorInfer     color_infer_;
    DiagLogger     diag_logger_;

    // Focus tracking: which camera to show fullscreen (-1 = grid)
    int focused_source_ = -1;

    // camera index → source element (for reconnect tracking)
    std::map<int, GstElement*> sources_;

    // ── Camera activation (only process cameras with recent detections) ──
    std::map<int, GstElement*> valves_;          // cam_index → valve element
    static std::map<int, int>  cam_last_det_;    // cam_index → last frame with detection
    static constexpr int CAM_DEACTIVATE_FRAMES = 75;  // ~3 sec at 25fps
    void update_camera_activation(int current_frame);

    // ── RTSP reconnection state ─────────────────────────────────────
    struct ReconnectInfo {
        int cam_index   = -1;
        int retry_count = 0;
        bool pending    = false;
        guint timer_id  = 0;
    };
    std::map<int, ReconnectInfo> reconnect_info_;

    // ── Pipeline construction helpers ───────────────────────────────

    bool add_sources();
    bool add_inference();
    bool add_sink();

    // ── Pad callbacks ───────────────────────────────────────────────

    static void on_pad_added(GstElement* src, GstPad* pad, gpointer data);
    static void on_child_added(GstChildProxy* proxy, GObject* object,
                               gchar* name, gpointer data);

    // ── Probe callback (the core detection + classification logic) ──

    static GstPadProbeReturn inference_probe(GstPad* pad,
                                             GstPadProbeInfo* info,
                                             gpointer data);

    // ── Bus message handler ────────────────────────────────────────

    static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data);

    // ── RTSP source reconnection ────────────────────────────────────

    int find_source_index(GstElement* element);
    void attempt_reconnect(int cam_index);
    static gboolean reconnect_cb(gpointer data);

    // ── ROI polygon filter (point-in-polygon) ───────────────────────
    static bool point_in_polygon(float px, float py,
                                 const std::vector<Point2f>& poly) {
        bool inside = false;
        int n = static_cast<int>(poly.size());
        for (int i = 0, j = n - 1; i < n; j = i++) {
            if (((poly[i].y > py) != (poly[j].y > py)) &&
                (px < (poly[j].x - poly[i].x) * (py - poly[i].y) /
                       (poly[j].y - poly[i].y) + poly[i].x))
                inside = !inside;
        }
        return inside;
    }

    static bool point_in_any_roi(float norm_x, float norm_y,
                                  const std::vector<std::vector<Point2f>>& zones) {
        if (zones.empty()) return true;  // no ROI = accept all
        for (const auto& poly : zones) {
            if (point_in_polygon(norm_x, norm_y, poly)) return true;
        }
        return false;
    }

    // ── Detection filtering ────────────────────────────────────────

    // Min bbox height in pixels, edge margin (0 = no edge filter)
    static constexpr int   MIN_BBOX_HEIGHT   = 35;   // pixels at 800x800; filters distant noise
    static constexpr float MIN_ASPECT_RATIO  = 0.25f; // width/height; reject very thin slivers
    static constexpr float MAX_ASPECT_RATIO  = 2.5f; // reject very wide slivers
    static constexpr int   EDGE_MARGIN       = 0;    // disabled — jockeys enter/exit at frame edges
    static constexpr int   MIN_CROP_PIXELS   = 200;
    static constexpr int   MAX_CROP_PIXELS   = 20000;

    // ── Static object filter (reject non-moving false positives) ────
    // Track center_x history per (cam_index, track_id). If an object
    // has been tracked for N+ frames and moved less than MIN_TRAVEL_PX
    // total, it's a static background object (pole, equipment, etc.)
    static constexpr int   STATIC_HISTORY_FRAMES = 50;  // 2 sec at 25fps — reject stationary objects
    static constexpr float MIN_TRAVEL_PX         = 10.0f; // min pixels moved to be considered alive

    struct TrackHistory {
        float first_cx = 0;
        float last_cx  = 0;
        int   frames   = 0;
    };
    // key: (cam_index << 16) | (track_id & 0xFFFF)
    static std::map<uint32_t, TrackHistory> track_history_;

    // ── Color smoothing per track (exponential moving average) ──────
    // Smooths out per-frame classifier noise for stable color display
    static constexpr float COLOR_EMA_ALPHA = 0.35f;  // weight of new observation (lower = smoother)
    struct ColorSmoother {
        float probs[NUM_COLORS] = {};  // smoothed probability per class
        int   frames            = 0;   // how many frames accumulated
    };
    // key: same as track_history_ — (cam_index << 16) | (track_id & 0xFFFF)
    static std::map<uint32_t, ColorSmoother> color_smooth_;
};

} // namespace rv
