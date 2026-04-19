/**
 * pipeline.cpp — DeepStream GStreamer pipeline implementation.
 *
 * Builds and manages the full pipeline:
 *   uridecodebin(s) → nvstreammux → nvinfer → probe → fakesink
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

void Pipeline::update_camera_activation(int current_frame) {
    if (valves_.empty()) return;

    int n_cams = static_cast<int>(config_.cameras.size());
    int active = 0, deactivated = 0;

    for (int i = 0; i < n_cams; ++i) {
        auto it = valves_.find(i);
        if (it == valves_.end()) continue;

        int last_det = (cam_last_det_.count(i)) ? cam_last_det_[i] : 0;
        bool has_recent = (current_frame - last_det) < CAM_DEACTIVATE_FRAMES;

        // Also activate neighbors of active cameras
        bool neighbor_active = false;
        if (i > 0 && cam_last_det_.count(i - 1) && (current_frame - cam_last_det_[i - 1]) < CAM_DEACTIVATE_FRAMES)
            neighbor_active = true;
        if (i < n_cams - 1 && cam_last_det_.count(i + 1) && (current_frame - cam_last_det_[i + 1]) < CAM_DEACTIVATE_FRAMES)
            neighbor_active = true;

        bool should_be_active = has_recent || neighbor_active;

        // First 150 frames (6 sec) — all active (waiting for race start)
        if (current_frame < 150) should_be_active = true;

        g_object_set(G_OBJECT(it->second), "drop", should_be_active ? FALSE : TRUE, NULL);

        if (should_be_active) active++;
        else deactivated++;
    }

    static int last_log_frame = 0;
    if (current_frame - last_log_frame >= 50) {
        fprintf(stderr, "[CAMS] frm=%-5d active=%d/%d\n", current_frame, active, n_cams);
        last_log_frame = current_frame;
    }
}

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

// ── RTSP source reconnection ───────────────────────────────────────

int Pipeline::find_source_index(GstElement* element) {
    GstElement* cur = element;
    while (cur && cur != pipeline_) {
        gchar* name = gst_element_get_name(cur);
        int idx = -1;
        if (sscanf(name, "source-%d", &idx) == 1) {
            g_free(name);
            return idx;
        }
        g_free(name);

        GstObject* parent = gst_element_get_parent(cur);
        if (!parent || parent == GST_OBJECT(pipeline_)) {
            if (parent) gst_object_unref(parent);
            break;
        }
        cur = GST_ELEMENT(parent);
        gst_object_unref(parent);
    }
    return -1;
}

gboolean Pipeline::reconnect_cb(gpointer data) {
    auto* p = static_cast<std::pair<Pipeline*, int>*>(data);
    p->first->attempt_reconnect(p->second);
    delete p;
    return FALSE;  // one-shot timer
}

void Pipeline::attempt_reconnect(int cam_idx) {
    if (cam_idx < 0 || cam_idx >= static_cast<int>(config_.cameras.size()))
        return;

    auto& info = reconnect_info_[cam_idx];
    info.timer_id = 0;

    const auto& cam = config_.cameras[cam_idx];
    fprintf(stderr, "[Pipeline] Reconnecting source-%d (%s), attempt %d\n",
            cam_idx, cam.id.c_str(), info.retry_count + 1);

    // 1. Remove old source element
    GstElement* old_source = sources_.count(cam_idx) ? sources_[cam_idx] : nullptr;
    if (old_source) {
        gst_element_set_state(old_source, GST_STATE_NULL);

        // Release the streammux sink pad for this source
        char pad_name[32];
        snprintf(pad_name, sizeof(pad_name), "sink_%d", cam_idx);
        GstPad* mux_pad = gst_element_get_static_pad(streammux_, pad_name);
        if (mux_pad) {
            gst_element_release_request_pad(streammux_, mux_pad);
            gst_object_unref(mux_pad);
        }

        gst_bin_remove(GST_BIN(pipeline_), old_source);
        sources_.erase(cam_idx);
    }

    // 2. Create new uridecodebin
    char name[64];
    snprintf(name, sizeof(name), "source-%d", cam_idx);
    GstElement* new_source = gst_element_factory_make("uridecodebin", name);
    if (!new_source) {
        fprintf(stderr, "[Pipeline] Failed to create uridecodebin for reconnect of source-%d\n",
                cam_idx);
        info.retry_count++;
        int delay = std::min(5 * (1 << std::min(info.retry_count, 3)), 30);
        info.timer_id = g_timeout_add_seconds(delay, reconnect_cb,
            new std::pair<Pipeline*, int>(this, cam_idx));
        return;
    }

    g_object_set(G_OBJECT(new_source), "uri", cam.url.c_str(), NULL);
    g_signal_connect(new_source, "pad-added", G_CALLBACK(on_pad_added), this);
    g_signal_connect(new_source, "child-added", G_CALLBACK(on_child_added), this);

    // 3. Add to pipeline and sync state
    gst_bin_add(GST_BIN(pipeline_), new_source);
    gst_element_sync_state_with_parent(new_source);

    sources_[cam_idx] = new_source;
    info.retry_count++;

    fprintf(stderr, "[Pipeline] source-%d reconnect initiated\n", cam_idx);
}

// ── Pad callbacks ──────────────────────────────────────────────────

void Pipeline::on_pad_added(GstElement* src, GstPad* pad, gpointer data) {
    auto* self = static_cast<Pipeline*>(data);

    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);

    const GstStructure* str = gst_caps_get_structure(caps, 0);
    const char* name = gst_structure_get_name(str);

    // Only link video pads
    if (g_str_has_prefix(name, "video/")) {
        // Determine which source this is
        gchar* src_name = gst_element_get_name(src);
        int src_idx = 0;
        if (sscanf(src_name, "source-%d", &src_idx) == 1) {
            // Request a sink pad from streammux
            char pad_name[32];
            snprintf(pad_name, sizeof(pad_name), "sink_%d", src_idx);

            GstPad* mux_pad = gst_element_request_pad_simple(
                self->streammux_, pad_name);
            if (mux_pad) {
                if (gst_pad_link(pad, mux_pad) == GST_PAD_LINK_OK) {
                    // Reset reconnect state on successful link
                    if (self->reconnect_info_.count(src_idx) &&
                        self->reconnect_info_[src_idx].retry_count > 0) {
                        fprintf(stderr, "[Pipeline] source-%d reconnected after %d attempts\n",
                                src_idx, self->reconnect_info_[src_idx].retry_count);
                        self->reconnect_info_[src_idx].retry_count = 0;
                        self->reconnect_info_[src_idx].pending = false;
                    }
                } else {
                    fprintf(stderr, "[Pipeline] Failed to link %s pad to streammux\n",
                            src_name);
                }
                gst_object_unref(mux_pad);
            }
        }
        g_free(src_name);
    }

    gst_caps_unref(caps);
}

void Pipeline::on_child_added(GstChildProxy* proxy, GObject* object,
                              gchar* name, gpointer data) {
    auto* self = static_cast<Pipeline*>(data);

    // Configure RTSP source properties for low latency
    if (g_str_has_prefix(name, "source")) {
        // Check if this element has the "drop-on-latency" property (rtspsrc-specific)
        GParamSpec* pspec = g_object_class_find_property(
            G_OBJECT_GET_CLASS(object), "drop-on-latency");
        if (pspec) {
            g_object_set(object,
                "drop-on-latency", TRUE,
                "latency",         100,   // ms
                NULL);
        }
    }

    // Optimize decoder (nvv4l2decoder) for throughput
    if (g_strrstr(name, "dec")) {
        GParamSpec* pspec = g_object_class_find_property(
            G_OBJECT_GET_CLASS(object), "drop-frame-interval");
        if (pspec && self->config_.live_source) {
            // Drop every other frame for 2x speedup on file playback
            g_object_set(object, "drop-frame-interval", 0, NULL);
        }

        // Enable low-latency mode if available
        pspec = g_object_class_find_property(
            G_OBJECT_GET_CLASS(object), "enable-max-performance");
        if (pspec) {
            g_object_set(object, "enable-max-performance", TRUE, NULL);
        }
    }
}

// ── Inference probe (main detection + classification logic) ────────

GstPadProbeReturn Pipeline::inference_probe(GstPad* pad,
                                            GstPadProbeInfo* info,
                                            gpointer data) {
    auto* self = static_cast<Pipeline*>(data);

    auto t_probe_start = std::chrono::steady_clock::now();

    // Measure time BETWEEN probes (= display pipeline time)
    static auto last_probe_exit = std::chrono::steady_clock::now();
    float between_probes_ms = std::chrono::duration<float, std::milli>(t_probe_start - last_probe_exit).count();
    static float sum_between_ms = 0;
    sum_between_ms += between_probes_ms;

    static int batch_counter = 0;
    batch_counter++;

    // Throttle to ~25fps for stable display (live-source=TRUE runs uncapped)
    static auto last_frame_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time).count();
    constexpr long TARGET_FRAME_US = 40000;  // 40ms = 25fps
    long sleep_us = 0;
    if (elapsed_us < TARGET_FRAME_US) {
        sleep_us = TARGET_FRAME_US - elapsed_us;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
    last_frame_time = std::chrono::steady_clock::now();

    auto t_after_throttle = std::chrono::steady_clock::now();

    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    // Get NvDs batch metadata
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    // Get the surface for GPU crop access
    GstMapInfo map_info;
    NvBufSurface* surface = nullptr;
    if (gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        surface = reinterpret_cast<NvBufSurface*>(map_info.data);
        gst_buffer_unmap(buf, &map_info);
    }

    // Debug: log surface params once
    static bool surface_logged = false;
    if (!surface_logged && surface && surface->numFilled > 0) {
        fprintf(stderr, "[SURFACE] numFilled=%d batchSize=%d\n", surface->numFilled, surface->batchSize);
        for (uint32_t si = 0; si < surface->numFilled && si < 5; ++si) {
            auto& p = surface->surfaceList[si];
            fprintf(stderr, "  [%d] %dx%d pitch=%d fmt=%d dataPtr=%p dataSize=%d numPlanes=%d\n",
                    si, p.width, p.height, p.pitch, p.colorFormat, p.dataPtr, p.dataSize, p.planeParams.num_planes);
            for (uint32_t pi = 0; pi < p.planeParams.num_planes && pi < 4; ++pi) {
                fprintf(stderr, "    plane[%d]: w=%d h=%d pitch=%d offset=%d psize=%d\n",
                        pi, p.planeParams.width[pi], p.planeParams.height[pi],
                        p.planeParams.pitch[pi], p.planeParams.offset[pi], p.planeParams.psize[pi]);
            }
        }
        surface_logged = true;
    }

    // Collect torso ROIs across all cameras
    std::vector<TorsoROI> all_rois;
    // Map from ROI index → (cam_index, det_index in camera slot)
    struct RoiMapping {
        int cam_index;
        int det_index;
        float x1, y1, x2, y2;
        float center_x;
        float det_conf;
        bool tracker_only;
    };
    std::vector<RoiMapping> roi_mappings;

    // Temporary storage for per-camera detections
    struct CamDets {
        CameraSlot slot;
        int cam_index;
    };
    std::vector<CamDets> cam_dets_list;

    // Iterate over frames in the batch
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr; l_frame = l_frame->next) {

        NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
        int cam_index = frame_meta->source_id;

        if (cam_index < 0 || cam_index >= static_cast<int>(self->config_.cameras.size()))
            continue;

        const auto& cam_config = self->config_.cameras[cam_index];
        // Use mux dimensions for bbox coordinates (nvinfer outputs in mux space)
        int fw = self->config_.mux_width;
        int fh = self->config_.mux_height;

        CamDets cam_dets;
        cam_dets.cam_index = cam_index;
        init_camera_slot(cam_dets.slot, cam_config.id.c_str());
        cam_dets.slot.frame_width  = fw;
        cam_dets.slot.frame_height = fh;
        cam_dets.slot.timestamp_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );

        int det_count = 0;
        int raw_count = 0;
        int filt_class = 0, filt_edge = 0, filt_height = 0, filt_aspect = 0, filt_tracker = 0;

        // Count total objects in frame
        int total_objs = 0;
        for (NvDsMetaList* tmp = frame_meta->obj_meta_list; tmp; tmp = tmp->next) total_objs++;

        // Iterate over detected objects
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list;
             l_obj != nullptr && det_count < MAX_DETECTIONS;
             l_obj = l_obj->next) {

            NvDsObjectMeta* obj_meta = static_cast<NvDsObjectMeta*>(l_obj->data);

            // Only person class
            if (obj_meta->class_id != 0) { filt_class++; continue; }
            raw_count++;

            float x1 = obj_meta->rect_params.left;
            float y1 = obj_meta->rect_params.top;
            float x2 = x1 + obj_meta->rect_params.width;
            float y2 = y1 + obj_meta->rect_params.height;
            // confidence < 0 means tracker-interpolated (no YOLO detection this frame)
            // Use absolute value for display, still process for tracking continuity
            float det_conf = obj_meta->confidence;
            bool tracker_only = (det_conf < 0.0f);
            if (tracker_only) det_conf = 0.5f;  // assign moderate confidence for tracker objects

            float bw = x2 - x1;
            float bh = y2 - y1;
            float center_x = (x1 + x2) / 2.0f;

            // Filter: edge margin (skip if fully outside frame)
            if (EDGE_MARGIN > 0 && (x1 <= EDGE_MARGIN || x2 >= fw - EDGE_MARGIN)) { filt_edge++; continue; }

            // Filter: min height (reject tiny distant objects / noise)
            if (bh < MIN_BBOX_HEIGHT) { filt_height++; continue; }

            // Filter: aspect ratio (reject thin slivers / edge artifacts)
            float aspect = bw / std::max(bh, 1.0f);
            if (aspect < MIN_ASPECT_RATIO || aspect > MAX_ASPECT_RATIO) { filt_aspect++; continue; }

            // Filter: ROI polygon (bbox center must be inside camera's ROI zone)
            {
                float norm_cx = center_x / static_cast<float>(fw);
                float norm_cy = ((y1 + y2) / 2.0f) / static_cast<float>(fh);
                if (!point_in_any_roi(norm_cx, norm_cy, cam_config.roi_zones)) continue;
            }

            // Filter: static object (not moving = not a racing horse)
            // STATIC_HISTORY_FRAMES=0 disables this filter entirely
            if (STATIC_HISTORY_FRAMES > 0) {
                uint32_t tid = static_cast<uint32_t>(obj_meta->object_id);
                uint32_t key = (static_cast<uint32_t>(cam_index) << 16) | (tid & 0xFFFF);
                auto& th = track_history_[key];
                if (th.frames == 0) {
                    th.first_cx = center_x;
                }
                th.last_cx = center_x;
                th.frames++;

                if (th.frames >= STATIC_HISTORY_FRAMES) {
                    float travel = std::abs(th.last_cx - th.first_cx);
                    if (travel < MIN_TRAVEL_PX) {
                        continue; // static object, skip
                    }
                }
            }

            // Crop ROI: torso only (10%..40% height, 20% side margins)
            // Focuses on chest/torso where jockey color is most visible
            TorsoROI roi = ColorInfer::compute_torso_roi(x1, y1, x2, y2, fw, fh, det_count, cam_index);
            roi.surface_index = frame_meta->batch_id;

            int crop_pixels = (roi.x2 - roi.x1) * (roi.y2 - roi.y1);
            if (crop_pixels < MIN_CROP_PIXELS || crop_pixels > MAX_CROP_PIXELS) continue;

            // (debug logging removed — see summary every 50 frames)

            // Store detection (tracker-only objects kept for visual continuity)
            Detection& det = cam_dets.slot.detections[det_count];
            init_detection(det);
            det.x1       = x1;
            det.y1       = y1;
            det.x2       = x2;
            det.y2       = y2;
            det.center_x = center_x;
            det.det_conf = det_conf;
            det.track_id = static_cast<uint32_t>(obj_meta->object_id);

            // Queue for color classification only on active cameras
            // (cameras with recent detections or neighbors of active cameras)
            bool cam_active = true;
            if (cam_last_det_.count(cam_index)) {
                int frames_since = batch_counter - cam_last_det_[cam_index];
                cam_active = (frames_since < CAM_DEACTIVATE_FRAMES);
            }
            // Also active if neighbor has recent detections
            if (!cam_active) {
                for (int delta : {-1, 1}) {
                    int nb = cam_index + delta;
                    if (nb >= 0 && nb < static_cast<int>(self->config_.cameras.size()) &&
                        cam_last_det_.count(nb) && (batch_counter - cam_last_det_[nb]) < CAM_DEACTIVATE_FRAMES) {
                        cam_active = true;
                        break;
                    }
                }
            }
            // First 150 frames = all active (waiting for race start)
            if (batch_counter < 150) cam_active = true;

            if (cam_active) {
                all_rois.push_back(roi);
                roi_mappings.push_back({cam_index, det_count, x1, y1, x2, y2, center_x, det_conf, tracker_only});
            }
            det_count++;
        }

        cam_dets.slot.num_detections = det_count;
        cam_dets_list.push_back(std::move(cam_dets));
    }

    auto t_after_filter = std::chrono::steady_clock::now();

    // Batch color classification (all objects including tracker-only)
    auto t_classify_start = std::chrono::steady_clock::now();
    int n_classified = 0;
    if (!all_rois.empty() && self->color_infer_.is_loaded() && surface) {
        n_classified = static_cast<int>(all_rois.size());
        std::vector<ColorResult> color_results;
        self->color_infer_.classify(surface, all_rois, color_results);

        // Map color results back + apply EMA smoothing per track_id
        for (size_t ri = 0; ri < roi_mappings.size() && ri < color_results.size(); ++ri) {
            const auto& rm = roi_mappings[ri];
            const auto& cr = color_results[ri];
            for (auto& cd : cam_dets_list) {
                if (cd.cam_index != rm.cam_index) continue;
                if (rm.det_index >= static_cast<int>(cd.slot.num_detections)) continue;

                Detection& det = cd.slot.detections[rm.det_index];

                // EMA smoothing: blend raw probs with history
                uint32_t key = (static_cast<uint32_t>(rm.cam_index) << 16)
                             | (det.track_id & 0xFFFF);
                auto& sm = color_smooth_[key];

                if (sm.frames == 0) {
                    // First observation: init with raw probs
                    std::memcpy(sm.probs, cr.probs, NUM_COLORS * sizeof(float));
                } else {
                    // EMA: smooth[i] = alpha*raw[i] + (1-alpha)*smooth[i]
                    for (int c = 0; c < NUM_COLORS; ++c) {
                        sm.probs[c] = COLOR_EMA_ALPHA * cr.probs[c]
                                    + (1.0f - COLOR_EMA_ALPHA) * sm.probs[c];
                    }
                }
                sm.frames++;

                // Pick best class from smoothed probs
                uint32_t best_id   = COLOR_UNKNOWN;
                float    best_prob = 0.0f;
                for (int c = 0; c < NUM_COLORS; ++c) {
                    if (sm.probs[c] > best_prob) {
                        best_prob = sm.probs[c];
                        best_id   = static_cast<uint32_t>(c);
                    }
                }

                // Require minimum confidence — below threshold show as UNKNOWN
                static constexpr float MIN_COLOR_CONF = 0.60f;
                det.color_id   = (best_prob >= MIN_COLOR_CONF) ? best_id : COLOR_UNKNOWN;
                det.color_conf = best_prob;
                std::memcpy(det.color_probs, sm.probs, sizeof(sm.probs));
                break;
            }
        }
    }

    // FPS counter (batch_counter declared at top of probe)

    // ── Diagnostic logging (CSV + JPG snapshots) ────────────────────
    if (self->diag_logger_.is_enabled()) {
        // Collect all detections for CSV
        std::vector<DiagDetection> all_diag_dets;
        for (const auto& cd : cam_dets_list) {
            std::vector<DiagDetection> cam_diag_dets;
            for (uint32_t di = 0; di < cd.slot.num_detections; ++di) {
                const auto& det = cd.slot.detections[di];
                DiagDetection dd;
                dd.cam_id = cd.slot.cam_id;
                dd.frame_w = cd.slot.frame_width;
                dd.frame_h = cd.slot.frame_height;
                dd.det_idx = static_cast<int>(di);
                dd.x1 = det.x1; dd.y1 = det.y1;
                dd.x2 = det.x2; dd.y2 = det.y2;
                dd.center_x = det.center_x;
                dd.det_conf = det.det_conf;
                dd.color_id = det.color_id;
                dd.color_conf = det.color_conf;
                std::memcpy(dd.color_probs, det.color_probs, sizeof(dd.color_probs));
                dd.track_id = det.track_id;
                all_diag_dets.push_back(dd);
                cam_diag_dets.push_back(dd);
            }

            // Save snapshot for cameras with detections
            if (!cam_diag_dets.empty() && surface) {
                // snap_interval check: save every N batches
                int si = self->config_.snap_interval;
                if (si <= 0 || (batch_counter % si == 0)) {
                    self->diag_logger_.save_snapshot(
                        batch_counter, surface,
                        cd.cam_index, std::string(cd.slot.cam_id),
                        cam_diag_dets);
                }
            }
        }

        // Log all detections to CSV (every batch)
        if (!all_diag_dets.empty()) {
            self->diag_logger_.log_detections(batch_counter, all_diag_dets);
        }
    }

    // Periodic cleanup of stale track/color history (every 100 batches)
    if (batch_counter % 100 == 0) {
        track_history_.clear();
        color_smooth_.clear();  // active tracks re-accumulate in a few frames
    }

    // Draw camera label on each frame
    if (self->config_.display) {
        for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
             l_frame; l_frame = l_frame->next) {
            NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
            int cam_index = frame_meta->source_id;

            NvDsDisplayMeta* dmeta = nvds_acquire_display_meta_from_pool(batch_meta);
            if (dmeta) {
                // Zero-init to avoid garbage fields causing crash
                std::memset(dmeta->text_params, 0, sizeof(dmeta->text_params));
                dmeta->num_labels = 0;
                dmeta->num_rects = 0;
                dmeta->num_lines = 0;
                dmeta->num_arrows = 0;
                dmeta->num_circles = 0;

                NvOSD_TextParams& tp = dmeta->text_params[0];
                char cam_label[32];
                snprintf(cam_label, sizeof(cam_label), "CAM-%02d", cam_index + 1);
                tp.display_text = g_strdup(cam_label);
                tp.x_offset = 10;
                tp.y_offset = 10;
                tp.font_params.font_name = const_cast<char*>("Serif");
                tp.font_params.font_size = 16;
                tp.font_params.font_color = {1.0f, 1.0f, 0.0f, 1.0f};
                tp.set_bg_clr = 1;
                tp.text_bg_clr = {0.0f, 0.0f, 0.0f, 0.7f};
                dmeta->num_labels = 1;
                nvds_add_display_meta_to_frame(frame_meta, dmeta);
            }
        }
    }

    // Set OSD labels on obj_meta for display mode
    if (self->config_.display) {
        // Map color_id to OSD colors (R,G,B,A)
        static const float OSD_COLORS[][4] = {
            {0.0f, 0.4f, 1.0f, 1.0f},  // blue
            {0.0f, 0.9f, 0.0f, 1.0f},  // green
            {0.8f, 0.0f, 0.8f, 1.0f},  // purple
            {1.0f, 0.0f, 0.0f, 1.0f},  // red
            {1.0f, 0.9f, 0.0f, 1.0f},  // yellow
        };

        for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
             l_frame; l_frame = l_frame->next) {
            NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
            int cam_index = frame_meta->source_id;

            // Find our CamDets for this camera
            CamDets* found_cd = nullptr;
            for (auto& cd : cam_dets_list) {
                if (cd.cam_index == cam_index) { found_cd = &cd; break; }
            }

            for (NvDsMetaList* l_obj = frame_meta->obj_meta_list;
                 l_obj; l_obj = l_obj->next) {
                NvDsObjectMeta* obj_meta = static_cast<NvDsObjectMeta*>(l_obj->data);
                if (obj_meta->class_id != 0) continue;

                // Find matching detection by bbox (only filtered detections are in cam_dets)
                int color_id = COLOR_UNKNOWN;
                float color_conf = 0.0f;
                if (found_cd) {
                    float ox1 = obj_meta->rect_params.left;
                    float oy1 = obj_meta->rect_params.top;
                    for (uint32_t di = 0; di < found_cd->slot.num_detections; ++di) {
                        const auto& det = found_cd->slot.detections[di];
                        if (std::abs(det.x1 - ox1) < 2.0f && std::abs(det.y1 - oy1) < 2.0f) {
                            color_id = det.color_id;
                            color_conf = det.color_conf;
                            break;
                        }
                    }
                }

                // Set bbox color and label
                float r, g, b;
                char label[64];
                uint64_t tid = obj_meta->object_id;
                float det_conf_pct = obj_meta->confidence * 100.0f;

                if (color_id != COLOR_UNKNOWN && color_id < NUM_COLORS) {
                    // Classified: colored bbox
                    r = OSD_COLORS[color_id][0];
                    g = OSD_COLORS[color_id][1];
                    b = OSD_COLORS[color_id][2];
                    const char* cname = COLOR_NAMES[color_id];
                    if (tid > 0) {
                        snprintf(label, sizeof(label), "%s %d%% #%lu", cname,
                                 static_cast<int>(color_conf * 100), (unsigned long)tid);
                    } else {
                        snprintf(label, sizeof(label), "%s %d%%", cname,
                                 static_cast<int>(color_conf * 100));
                    }
                } else {
                    // Not classified: white bbox with YOLO conf
                    r = 1.0f; g = 1.0f; b = 1.0f;
                    if (tid > 0) {
                        snprintf(label, sizeof(label), "YOLO %.0f%% #%lu",
                                 det_conf_pct, (unsigned long)tid);
                    } else {
                        snprintf(label, sizeof(label), "YOLO %.0f%%", det_conf_pct);
                    }
                }

                obj_meta->rect_params.border_color = {r, g, b, 1.0f};
                obj_meta->rect_params.border_width = 2;

                if (obj_meta->text_params.display_text)
                    g_free(obj_meta->text_params.display_text);
                obj_meta->text_params.display_text = g_strdup(label);
                obj_meta->text_params.x_offset = static_cast<int>(obj_meta->rect_params.left);
                obj_meta->text_params.y_offset = std::max(0, static_cast<int>(obj_meta->rect_params.top) - 14);
                obj_meta->text_params.font_params.font_name = const_cast<char*>("Sans");
                obj_meta->text_params.font_params.font_size = 10;
                obj_meta->text_params.font_params.font_color = {0.0f, 0.0f, 0.0f, 1.0f};  // black text
                obj_meta->text_params.set_bg_clr = 1;
                obj_meta->text_params.text_bg_clr = {r, g, b, 0.85f};
            }
        }
    }

    // Write all camera slots to shared memory + track last detection frame
    int total_dets = 0;
    int best_cam = -1;
    int best_det_count = 0;
    for (const auto& cd : cam_dets_list) {
        self->shm_writer_.write_camera(cd.cam_index, cd.slot);
        total_dets += cd.slot.num_detections;
        if (cd.slot.num_detections > 0) {
            cam_last_det_[cd.cam_index] = batch_counter;
        }
        if (static_cast<int>(cd.slot.num_detections) > best_det_count) {
            best_det_count = cd.slot.num_detections;
            best_cam = cd.cam_index;
        }
    }

    // Log active cameras
    {
        int n_active = 0;
        int n_cams = static_cast<int>(self->config_.cameras.size());
        for (int i = 0; i < n_cams; ++i) {
            if (cam_last_det_.count(i) && (batch_counter - cam_last_det_[i]) < CAM_DEACTIVATE_FRAMES)
                n_active++;
        }
        if (batch_counter % 50 == 0 && batch_counter > 0)
            fprintf(stderr, "[CAMS] frm=%-5d classify_active=%d/%d\n", batch_counter, n_active, n_cams);
    }

    // Dynamic focus: show camera with most detections fullscreen
    if (self->tiler_ && self->config_.display) {
        // Always show single camera (full grid = too heavy for 25 cams)
        // When no dets, keep showing last active camera (or cam 0)
        int target = (best_det_count > 0) ? best_cam : self->focused_source_;
        if (target < 0) target = 0;  // default to cam-01
        if (target != self->focused_source_) {
            g_object_set(G_OBJECT(self->tiler_), "show-source", target, NULL);
            self->focused_source_ = target;
            if (target >= 0) {
                fprintf(stderr, "[Focus] → %s (%d dets)\n",
                        self->config_.cameras[target].id.c_str(), best_det_count);
            }
        }
    }

    // Commit (increment seq + signal semaphore)
    self->shm_writer_.commit();

    // FPS counter + periodic logging
    static int fps_frame_count = 0;
    static auto fps_start = std::chrono::steady_clock::now();
    static float current_fps = 0.0f;

    fps_frame_count++;

    auto now_tp = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(now_tp - fps_start).count();
    if (elapsed_sec >= 2.0) {
        current_fps = static_cast<float>(fps_frame_count / elapsed_sec);
        fps_frame_count = 0;
        fps_start = now_tp;
    }

    // Timing: color classification
    auto t_classify_end = std::chrono::steady_clock::now();
    float classify_ms = std::chrono::duration<float, std::milli>(t_classify_end - t_classify_start).count();

    // Accumulate stats
    static int sum_dets = 0, sum_yolo = 0, sum_tracker = 0, sum_classified = 0;
    static float sum_classify_ms = 0;
    static int active_cams = 0;
    static const char* COLOR_LABEL[6] = {"BLUE","GREEN","PURPLE","RED","YELLOW","???"};

    int frame_yolo = 0, frame_tracker = 0;
    active_cams = 0;
    for (const auto& cd : cam_dets_list) {
        if (cd.slot.num_detections > 0) active_cams++;
        for (uint32_t di = 0; di < cd.slot.num_detections; ++di) {
            if (cd.slot.detections[di].det_conf == 0.5f)
                frame_tracker++;
            else
                frame_yolo++;
        }
    }
    sum_dets += total_dets;
    sum_yolo += frame_yolo;
    sum_tracker += frame_tracker;
    sum_classified += n_classified;
    sum_classify_ms += classify_ms;

    // Timing for this frame
    auto t_probe_end = std::chrono::steady_clock::now();
    float probe_total_ms = std::chrono::duration<float, std::milli>(t_probe_end - t_probe_start).count();
    float throttle_ms = std::chrono::duration<float, std::milli>(t_after_throttle - t_probe_start).count();
    float filter_ms = std::chrono::duration<float, std::milli>(t_after_filter - t_after_throttle).count();

    static float sum_probe_ms = 0, sum_throttle_ms = 0, sum_filter_ms = 0;
    sum_probe_ms += probe_total_ms;
    sum_throttle_ms += throttle_ms;
    sum_filter_ms += filter_ms;

    // Summary every 50 frames
    if (batch_counter % 50 == 0) {
        fprintf(stderr, "[STAT] frm=%-5d fps=%.1f | dets=%d (yolo=%d trk=%d) | classify=%d %.1fms | cams=%d/%zu\n",
                batch_counter, current_fps,
                sum_dets, sum_yolo, sum_tracker,
                sum_classified, sum_classify_ms,
                active_cams, self->config_.cameras.size());
        float work_ms = sum_probe_ms - sum_throttle_ms;
        float display_ms = sum_between_ms;
        float total_ms = sum_probe_ms + sum_between_ms;
        fprintf(stderr, "  timing per 50frm: total=%.0fms | probe=%.0fms (throttle=%.0fms work=%.0fms) | display_pipe=%.0fms\n",
                total_ms, sum_probe_ms, sum_throttle_ms, work_ms, display_ms);
        fprintf(stderr, "  per-frame avg: probe=%.1fms (throttle=%.1f work=%.1f classify=%.1f) display=%.1fms | budget=40ms\n",
                sum_probe_ms/50, sum_throttle_ms/50, work_ms/50, sum_classify_ms/50, display_ms/50);

        // Per-camera color summary
        for (const auto& cd : cam_dets_list) {
            if (cd.slot.num_detections == 0) continue;
            fprintf(stderr, "  %s:", cd.slot.cam_id);
            for (uint32_t di = 0; di < cd.slot.num_detections; ++di) {
                const auto& det = cd.slot.detections[di];
                const char* cname = (det.color_id < NUM_COLORS) ? COLOR_LABEL[det.color_id] : "UNK";
                fprintf(stderr, " t%u=%s(%.0f%%)", det.track_id, cname, det.color_conf * 100);
            }
            fprintf(stderr, "\n");
        }

        sum_dets = sum_yolo = sum_tracker = sum_classified = 0;
        sum_classify_ms = 0;
        sum_probe_ms = sum_throttle_ms = sum_filter_ms = 0;
        sum_between_ms = 0;
    }

    last_probe_exit = std::chrono::steady_clock::now();
    return GST_PAD_PROBE_OK;
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
