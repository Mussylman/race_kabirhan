#pragma once
/**
 * pipeline.h — DeepStream GStreamer pipeline for multi-camera RTSP decode + inference.
 *
 * Pipeline topology:
 *   N x uridecodebin (RTSP → NVDEC hw decode)
 *     → nvstreammux (batch all cameras)
 *       → nvinfer (YOLOv8s TRT FP16)
 *         → probe callback (filter + crop + color classify + SHM write)
 *           → fakesink
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

struct PipelineConfig {
    std::vector<CameraConfig> cameras;
    std::string yolo_engine_path;
    std::string color_engine_path;
    int batch_size       = 25;
    int mux_width        = 1280;
    int mux_height       = 1280;
    float det_conf       = 0.35f;
    int mux_batched_push_timeout = 40000; // microseconds
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
    GMainLoop*   loop_       = nullptr;
    bool         running_    = false;

    PipelineConfig config_;
    ShmWriter      shm_writer_;
    ColorInfer     color_infer_;

    // camera index → source element (for reconnect tracking)
    std::map<int, GstElement*> sources_;

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

    // ── Detection filtering ────────────────────────────────────────

    // Min bbox height in pixels, min aspect ratio, edge margin
    static constexpr int   MIN_BBOX_HEIGHT   = 100;
    static constexpr float MIN_ASPECT_RATIO  = 1.2f;
    static constexpr int   EDGE_MARGIN       = 10;
    static constexpr int   MIN_CROP_PIXELS   = 400;
    static constexpr int   MAX_CROP_PIXELS   = 15000;
};

} // namespace rv
