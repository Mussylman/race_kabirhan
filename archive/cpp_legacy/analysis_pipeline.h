#pragma once
/**
 * analysis_pipeline.h — Heavy analysis pipeline (YOLOv8s @ 800x800).
 *
 * All 25 cameras are connected at startup. A GStreamer "valve" element
 * per camera gates which frames reach inference. activate()/deactivate()
 * flip the valve instantly — no RTSP reconnection delay.
 *
 * Pipeline topology per camera:
 *   uridecodebin → valve(drop=TRUE) → nvstreammux 800×800
 *     → nvinfer YOLOv8s → inference_probe → fakesink
 */

#include "config.h"
#include "shm_writer.h"
#include "color_infer.h"

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <gst/gst.h>

namespace rv {

struct AnalysisConfig {
    std::vector<CameraConfig> cameras;    // all cameras (indexed by cam_index)
    std::string nvinfer_config = "configs/nvinfer_yolov8s_analysis.txt";
    std::string color_engine_path;
    int   mux_width   = 800;
    int   mux_height  = 800;
    int   max_batch   = 8;
};

class AnalysisPipeline {
public:
    AnalysisPipeline();
    ~AnalysisPipeline();

    /**
     * Build the analysis pipeline with all cameras connected.
     * All valves start closed (drop=TRUE). Models are loaded at build time.
     */
    bool build(const AnalysisConfig& config, GMainLoop* shared_loop);

    void start();
    void stop();

    /**
     * Open valve for camera — frames start flowing to inference instantly.
     * Thread-safe (GObject property set is atomic).
     */
    void activate(int cam_index);

    /**
     * Close valve for camera — frames are dropped before inference.
     * Also zeros the SHM slot for this camera.
     */
    void deactivate(int cam_index);

    bool is_running() const { return running_; }

private:
    GstElement*  pipeline_   = nullptr;
    GstElement*  streammux_  = nullptr;
    GMainLoop*   loop_       = nullptr;  // borrowed, not owned
    bool         running_    = false;

    AnalysisConfig config_;
    ShmWriter      shm_writer_;
    ColorInfer     color_infer_;

    // Per-camera elements
    std::map<int, GstElement*> sources_;  // uridecodebin
    std::map<int, GstElement*> valves_;   // valve (drop property gates frames)

    // ── Pipeline construction ──────────────────────────────────────
    bool add_sources();
    bool add_inference();
    bool add_sink();

    // ── Pad callbacks ──────────────────────────────────────────────
    static void on_pad_added(GstElement* src, GstPad* pad, gpointer data);
    static void on_child_added(GstChildProxy* proxy, GObject* object,
                               gchar* name, gpointer data);

    // ── Inference probe (filter + color classify + SHM write) ──────
    static GstPadProbeReturn inference_probe(GstPad* pad,
                                             GstPadProbeInfo* info,
                                             gpointer data);

    // ── Bus handler ────────────────────────────────────────────────
    static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data);

    // ── RTSP reconnection ──────────────────────────────────────────
    struct ReconnectInfo {
        int cam_index   = -1;
        int retry_count = 0;
        bool pending    = false;
        guint timer_id  = 0;
    };
    std::map<int, ReconnectInfo> reconnect_info_;

    int find_source_index(GstElement* element);
    void attempt_reconnect(int cam_index);
    static gboolean reconnect_cb(gpointer data);

    // Detection filtering (same thresholds as single pipeline)
    static constexpr int   MIN_BBOX_HEIGHT   = 65;
    static constexpr float MIN_ASPECT_RATIO  = 0.8f;
    static constexpr int   EDGE_MARGIN       = 10;
    static constexpr int   MIN_CROP_PIXELS   = 400;
    static constexpr int   MAX_CROP_PIXELS   = 15000;
};

} // namespace rv
