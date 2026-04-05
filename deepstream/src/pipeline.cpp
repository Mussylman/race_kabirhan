/**
 * pipeline.cpp — DeepStream GStreamer pipeline implementation.
 *
 * Builds and manages the full pipeline:
 *   uridecodebin(s) → nvstreammux → nvinfer → probe → fakesink
 *
 * Split into 3 files:
 *   pipeline.cpp          — Pipeline construction (build, add_sources, add_inference, add_sink),
 *                           start(), stop(), bus_call()
 *   probe_handler.cpp     — inference_probe() callback (detection/classification logic)
 *   camera_activation.cpp — update_camera_activation(), reconnect, pad callbacks
 */

#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include "nvbufsurface.h"

namespace rv {

// Static member definitions
std::map<uint32_t, Pipeline::TrackHistory>   Pipeline::track_history_;
std::map<uint32_t, Pipeline::ColorSmoother>  Pipeline::color_smooth_;
std::map<int, int>                           Pipeline::cam_last_det_;

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
}

// ── Build ──────────────────────────────────────────────────────────

bool Pipeline::build(const PipelineConfig& config) {
    config_ = config;

    gst_init(nullptr, nullptr);
    loop_ = g_main_loop_new(nullptr, FALSE);

    pipeline_ = gst_pipeline_new("race-vision-pipeline");
    if (!pipeline_) {
        fprintf(stderr, "[Pipeline] Failed to create pipeline\n");
        return false;
    }

    // Create streammux
    streammux_ = gst_element_factory_make("nvstreammux", "streammux");
    if (!streammux_) {
        fprintf(stderr, "[Pipeline] Failed to create nvstreammux\n");
        return false;
    }

    // Optimized streammux settings for high FPS
    g_object_set(G_OBJECT(streammux_),
        "batch-size",             config_.batch_size,
        "width",                  config_.mux_width,
        "height",                 config_.mux_height,
        "batched-push-timeout",   config_.mux_batched_push_timeout,
        "live-source",            config_.live_source ? TRUE : FALSE,
        "enable-padding",         TRUE,
        "num-surfaces-per-frame", 1,
        "attach-sys-ts",          TRUE,
        NULL);

    fprintf(stderr, "[Pipeline] live-source=%s\n",
            config_.live_source ? "TRUE" : "FALSE");

    gst_bin_add(GST_BIN(pipeline_), streammux_);

    if (!config_.display_only) {
        // Initialize shared memory
        std::vector<std::string> cam_ids;
        for (const auto& cam : config_.cameras) {
            cam_ids.push_back(cam.id);
        }
        if (!shm_writer_.create(static_cast<uint32_t>(config_.cameras.size()), cam_ids)) {
            fprintf(stderr, "[Pipeline] Failed to create shared memory\n");
            return false;
        }

        // Load color classifier
        if (!config_.color_engine_path.empty()) {
            if (!color_infer_.load(config_.color_engine_path)) {
                fprintf(stderr, "[Pipeline] Warning: color classifier not loaded\n");
            }
        }
    }

    // Initialize diagnostic logger
    if (!config_.log_dir.empty()) {
        if (!diag_logger_.init(config_.log_dir, config_.snap_interval)) {
            fprintf(stderr, "[Pipeline] Warning: diagnostic logger init failed\n");
        }
    }

    if (!add_sources()) return false;
    if (!config_.display_only) {
        if (!add_inference()) return false;
    }
    if (!add_sink()) return false;

    // Bus watch
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    gst_bus_add_watch(bus, bus_call, this);
    gst_object_unref(bus);

    fprintf(stderr, "[Pipeline] Built successfully (%zu cameras)\n", config_.cameras.size());
    return true;
}

bool Pipeline::add_sources() {
    for (size_t i = 0; i < config_.cameras.size(); ++i) {
        const auto& cam = config_.cameras[i];
        char name[64];
        snprintf(name, sizeof(name), "source-%zu", i);

        GstElement* source = gst_element_factory_make("uridecodebin", name);
        if (!source) {
            fprintf(stderr, "[Pipeline] Failed to create uridecodebin for %s\n", cam.id.c_str());
            return false;
        }

        g_object_set(G_OBJECT(source), "uri", cam.url.c_str(), NULL);

        // Connect pad-added signal for dynamic linking to streammux
        g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), this);
        g_signal_connect(source, "child-added", G_CALLBACK(on_child_added), this);

        gst_bin_add(GST_BIN(pipeline_), source);
        sources_[static_cast<int>(i)] = source;

        fprintf(stderr, "[Pipeline] Added source %s: %s\n", cam.id.c_str(), cam.url.c_str());
    }
    return true;
}

bool Pipeline::add_inference() {
    GstElement* nvinfer = gst_element_factory_make("nvinfer", "yolo-infer");
    if (!nvinfer) {
        fprintf(stderr, "[Pipeline] Failed to create nvinfer\n");
        return false;
    }

    // nvinfer config file path — set via config property
    // The config file points to the engine, batch size, custom parser, etc.
    std::string nvinfer_config = config_.yolo_engine_path;
    // If config path ends with .txt, use it as nvinfer config
    // Otherwise, use the default config path
    if (nvinfer_config.find(".txt") != std::string::npos) {
        g_object_set(G_OBJECT(nvinfer), "config-file-path", nvinfer_config.c_str(), NULL);
    } else {
        // Use default config that references the engine path
        g_object_set(G_OBJECT(nvinfer),
            "config-file-path", "configs/nvinfer_yolov8s.txt",
            NULL);
    }

    // unique-id must be set as GObject property (not in config file)
    g_object_set(G_OBJECT(nvinfer), "unique-id", 1, NULL);

    gst_bin_add(GST_BIN(pipeline_), nvinfer);

    // Link streammux → nvinfer
    if (!gst_element_link(streammux_, nvinfer)) {
        fprintf(stderr, "[Pipeline] Failed to link streammux → nvinfer\n");
        return false;
    }

    // ── nvtracker (IOU tracker for persistent object IDs) ──
    GstElement* tracker = gst_element_factory_make("nvtracker", "tracker");
    if (!tracker) {
        fprintf(stderr, "[Pipeline] Failed to create nvtracker\n");
        return false;
    }

    g_object_set(G_OBJECT(tracker),
        "tracker-width",  config_.mux_width,
        "tracker-height", config_.mux_height,
        "ll-lib-file",    "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file", "configs/tracker_iou.yml",
        NULL);

    gst_bin_add(GST_BIN(pipeline_), tracker);

    // Link nvinfer → nvtracker
    if (!gst_element_link(nvinfer, tracker)) {
        fprintf(stderr, "[Pipeline] Failed to link nvinfer → nvtracker\n");
        return false;
    }

    fprintf(stderr, "[Pipeline] nvtracker added (IOU tracker, %dx%d)\n",
            config_.mux_width, config_.mux_height);

    // Add probe on nvtracker src pad (color kernel handles NV12 directly)
    GstPad* src_pad = gst_element_get_static_pad(tracker, "src");
    if (src_pad) {
        gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          inference_probe, this, nullptr);
        gst_object_unref(src_pad);
    }

    return true;
}

bool Pipeline::add_sink() {
    // Source element to link from: nvinfer (normal) or streammux (display-only)
    GstElement* link_from = nullptr;
    if (config_.display_only) {
        link_from = streammux_;
        g_object_ref(link_from);  // balance unref below
    } else {
        // Link from nvtracker (last element before sink)
        link_from = gst_bin_get_by_name(GST_BIN(pipeline_), "tracker");
        if (!link_from) {
            link_from = gst_bin_get_by_name(GST_BIN(pipeline_), "yolo-infer");
        }
        if (!link_from) {
            fprintf(stderr, "[Pipeline] Cannot find tracker/nvinfer element\n");
            return false;
        }
    }

    if (config_.display || config_.display_only) {
        // ── Display mode: link_from → [nvdsosd] → tiler → conv → sink ──

        // Tiler: tile N cameras into one output
        GstElement* tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
        if (!tiler) {
            fprintf(stderr, "[Pipeline] Failed to create nvmultistreamtiler\n");
            gst_object_unref(link_from);
            return false;
        }
        int n = static_cast<int>(config_.cameras.size());
        int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
        int rows = (n + cols - 1) / cols;
        g_object_set(G_OBJECT(tiler),
            "rows",   rows,
            "columns", cols,
            "width",  config_.display_width,
            "height", config_.display_height,
            "show-source", 0,           // start with cam-01, switch on detections
            "gpu-id", 0,
            NULL);
        tiler_ = tiler;  // save for dynamic focus

        // Video converter
        GstElement* conv = gst_element_factory_make("nvvideoconvert", "display-conv");
        if (!conv) {
            fprintf(stderr, "[Pipeline] Failed to create nvvideoconvert\n");
            gst_object_unref(link_from);
            return false;
        }

        // Display sink — try nv3dsink first, fallback to nveglglessink
        GstElement* sink = gst_element_factory_make("nv3dsink", "display-sink");
        if (!sink) {
            sink = gst_element_factory_make("nveglglessink", "display-sink");
        }
        if (!sink) {
            fprintf(stderr, "[Pipeline] Failed to create display sink\n");
            gst_object_unref(link_from);
            return false;
        }
        g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);  // TRUE = real-time playback at source fps

        // Queue decouples inference from display — display runs at source fps,
        // inference runs as fast as it can. Leaky=2 (downstream) drops old frames.
        GstElement* disp_queue = gst_element_factory_make("queue", "display-queue");
        if (disp_queue) {
            g_object_set(G_OBJECT(disp_queue),
                "max-size-buffers", 2,
                "leaky", 2,        // downstream: drop oldest if full
                NULL);
        }

        if (config_.display && !config_.display_only) {
            // Full display: link_from → queue → osd → tiler → conv → sink
            GstElement* osd = gst_element_factory_make("nvdsosd", "osd");
            if (!osd) {
                fprintf(stderr, "[Pipeline] Failed to create nvdsosd\n");
                gst_object_unref(link_from);
                return false;
            }
            g_object_set(G_OBJECT(osd),
                "process-mode", 1,
                "display-text", TRUE,
                NULL);

            if (disp_queue) {
                gst_bin_add_many(GST_BIN(pipeline_), disp_queue, osd, tiler, conv, sink, NULL);
                if (!gst_element_link_many(link_from, disp_queue, osd, tiler, conv, sink, NULL)) {
                    fprintf(stderr, "[Pipeline] Failed to link display pipeline\n");
                    gst_object_unref(link_from);
                    return false;
                }
            } else {
                gst_bin_add_many(GST_BIN(pipeline_), osd, tiler, conv, sink, NULL);
                if (!gst_element_link_many(link_from, osd, tiler, conv, sink, NULL)) {
                    fprintf(stderr, "[Pipeline] Failed to link display pipeline\n");
                    gst_object_unref(link_from);
                    return false;
                }
            }
        } else {
            // Display-only: link_from → queue → tiler → conv → sink (no OSD)
            if (disp_queue) {
                gst_bin_add_many(GST_BIN(pipeline_), disp_queue, tiler, conv, sink, NULL);
                if (!gst_element_link_many(link_from, disp_queue, tiler, conv, sink, NULL)) {
                    fprintf(stderr, "[Pipeline] Failed to link display-only pipeline\n");
                    gst_object_unref(link_from);
                    return false;
                }
            } else {
                gst_bin_add_many(GST_BIN(pipeline_), tiler, conv, sink, NULL);
                if (!gst_element_link_many(link_from, tiler, conv, sink, NULL)) {
                    fprintf(stderr, "[Pipeline] Failed to link display-only pipeline\n");
                    gst_object_unref(link_from);
                    return false;
                }
            }
        }

        fprintf(stderr, "[Pipeline] Display mode: %dx%d tiler (%dx%d grid)%s\n",
                config_.display_width, config_.display_height, cols, rows,
                config_.display_only ? " [VIDEO ONLY]" : " [+YOLO+OSD]");
    } else {
        // ── Headless mode: link_from → fakesink ──
        GstElement* sink = gst_element_factory_make("fakesink", "fakesink");
        if (!sink) {
            fprintf(stderr, "[Pipeline] Failed to create fakesink\n");
            gst_object_unref(link_from);
            return false;
        }
        g_object_set(G_OBJECT(sink), "sync", TRUE, "async", FALSE, NULL);
        gst_bin_add(GST_BIN(pipeline_), sink);
        gst_element_link(link_from, sink);
    }

    gst_object_unref(link_from);
    return true;
}

// ── Bus message handler ────────────────────────────────────────────

gboolean Pipeline::bus_call(GstBus* bus, GstMessage* msg, gpointer data) {
    auto* self = static_cast<Pipeline*>(data);

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        fprintf(stderr, "[Pipeline] End of stream\n");
        g_main_loop_quit(self->loop_);
        break;

    case GST_MESSAGE_ERROR: {
        gchar* debug = nullptr;
        GError* error = nullptr;
        gst_message_parse_error(msg, &error, &debug);
        fprintf(stderr, "[Pipeline] ERROR from %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
        if (debug) {
            fprintf(stderr, "[Pipeline]   Debug: %s\n", debug);
            g_free(debug);
        }
        g_error_free(error);

        // Schedule reconnect for the failed RTSP source
        int cam_idx = self->find_source_index(GST_ELEMENT(msg->src));
        if (cam_idx >= 0 && !self->reconnect_info_[cam_idx].pending) {
            auto& info = self->reconnect_info_[cam_idx];
            info.cam_index = cam_idx;
            info.pending = true;
            int delay = std::min(5 * (1 << std::min(info.retry_count, 3)), 30);
            fprintf(stderr, "[Pipeline] Scheduling reconnect for source-%d in %ds (attempt %d)\n",
                    cam_idx, delay, info.retry_count + 1);
            info.timer_id = g_timeout_add_seconds(delay, reconnect_cb,
                new std::pair<Pipeline*, int>(self, cam_idx));
        }
        break;
    }

    case GST_MESSAGE_WARNING: {
        gchar* debug = nullptr;
        GError* error = nullptr;
        gst_message_parse_warning(msg, &error, &debug);
        // Suppress "buffers are being dropped" from display queue (expected with 25 cams)
        if (error && strstr(error->message, "drop") == nullptr) {
            fprintf(stderr, "[Pipeline] WARNING from %s: %s\n",
                    GST_OBJECT_NAME(msg->src), error->message);
        }
        if (debug) g_free(debug);
        g_error_free(error);
        break;
    }

    case GST_MESSAGE_STATE_CHANGED:
        if (GST_MESSAGE_SRC(msg) == GST_OBJECT(self->pipeline_)) {
            GstState old_state, new_state, pending;
            gst_message_parse_state_changed(msg, &old_state, &new_state, &pending);
            fprintf(stderr, "[Pipeline] State: %s → %s\n",
                    gst_element_state_get_name(old_state),
                    gst_element_state_get_name(new_state));
        }
        break;

    default:
        break;
    }

    return TRUE;
}

// ── Start / Stop ───────────────────────────────────────────────────

void Pipeline::start() {
    if (!pipeline_) return;

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[Pipeline] Failed to set pipeline to PLAYING\n");
        return;
    }

    running_ = true;
    fprintf(stderr, "[Pipeline] Started (PLAYING)\n");
}

void Pipeline::stop() {
    running_ = false;

    // Cancel all pending reconnect timers
    for (auto& [idx, info] : reconnect_info_) {
        if (info.timer_id > 0) {
            g_source_remove(info.timer_id);
            info.timer_id = 0;
        }
    }
    reconnect_info_.clear();

    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_  = nullptr;
        streammux_ = nullptr;
    }

    if (loop_) {
        if (g_main_loop_is_running(loop_)) {
            g_main_loop_quit(loop_);
        }
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }

    shm_writer_.destroy();
    fprintf(stderr, "[Pipeline] Stopped\n");
}

} // namespace rv
