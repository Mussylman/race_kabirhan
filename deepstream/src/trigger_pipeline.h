#pragma once
/**
 * trigger_pipeline.h — Lightweight trigger pipeline (YOLOv8n @ 640x640).
 *
 * Runs all 25 cameras through a small YOLO model to detect which cameras
 * have horses/jockeys. Manages activation state with cooldown and max_active
 * enforcement. Fires a callback when the active set changes.
 *
 * Pipeline topology:
 *   25 x uridecodebin → nvstreammux 640×640 → nvinfer YOLOv8n → trigger_probe → fakesink
 */

#include "config.h"
#include "trigger_shm.h"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <functional>
#include <chrono>
#include <utility>
#include <gst/gst.h>

namespace rv {

struct TriggerConfig {
    std::vector<CameraConfig> cameras;
    std::string nvinfer_config = "configs/nvinfer_yolov8n_trigger.txt";
    int   mux_width   = 640;
    int   mux_height  = 640;
    int   max_active  = 8;
    float cooldown_s  = 3.0f;

    // Trigger detection filters (lighter than analysis)
    int   min_bbox_height = 50;
    float min_aspect_ratio = 0.8f;
};

// Callback: (newly_activated, newly_deactivated)
using ActivationCallback = std::function<void(
    const std::set<int>& activated,
    const std::set<int>& deactivated)>;

class TriggerPipeline {
public:
    TriggerPipeline();
    ~TriggerPipeline();

    /**
     * Build the trigger pipeline. Does NOT create a GMainLoop —
     * expects a shared loop from the coordinator.
     */
    bool build(const TriggerConfig& config, GMainLoop* shared_loop);

    void start();
    void stop();

    void set_activation_callback(ActivationCallback cb);

    std::set<int> get_active_cameras() const;

    bool is_running() const { return running_; }

private:
    GstElement*  pipeline_   = nullptr;
    GstElement*  streammux_  = nullptr;
    GMainLoop*   loop_       = nullptr;  // borrowed, not owned
    bool         running_    = false;

    TriggerConfig config_;
    TriggerShmWriter trigger_shm_;

    // Source tracking
    std::map<int, GstElement*> sources_;

    // ── Activation state ───────────────────────────────────────────
    mutable std::mutex activation_mutex_;
    std::set<int> active_cameras_;
    // Last detection time per camera (for cooldown)
    std::map<int, std::chrono::steady_clock::time_point> last_detection_;
    ActivationCallback activation_cb_;

    // ── Reconnection ───────────────────────────────────────────────
    struct ReconnectInfo {
        int cam_index   = -1;
        int retry_count = 0;
        bool pending    = false;
        guint timer_id  = 0;
    };
    std::map<int, ReconnectInfo> reconnect_info_;

    // ── Pipeline construction ──────────────────────────────────────
    bool add_sources();
    bool add_inference();
    bool add_sink();

    // ── Callbacks ──────────────────────────────────────────────────
    static void on_pad_added(GstElement* src, GstPad* pad, gpointer data);
    static void on_child_added(GstChildProxy* proxy, GObject* object,
                               gchar* name, gpointer data);
    static GstPadProbeReturn trigger_probe(GstPad* pad,
                                           GstPadProbeInfo* info,
                                           gpointer data);
    static gboolean bus_call(GstBus* bus, GstMessage* msg, gpointer data);

    // ── Activation logic ───────────────────────────────────────────
    void update_activations(const uint32_t detection_counts[MAX_CAMERAS]);

    // ── Reconnection ───────────────────────────────────────────────
    int find_source_index(GstElement* element);
    void attempt_reconnect(int cam_index);
    static gboolean reconnect_cb(gpointer data);
};

} // namespace rv
