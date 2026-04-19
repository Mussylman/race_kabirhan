/**
 * analysis_pipeline.cpp — Heavy analysis pipeline with valve-gated sources.
 *
 * All 25 cameras connect at startup (RTSP sessions stay open).
 * Each camera has a valve element: drop=TRUE (inactive) or drop=FALSE (active).
 * Trigger flips the valve → frames flow to inference instantly, no reconnect delay.
 */

#include "analysis_pipeline.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <algorithm>

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include "nvbufsurface.h"

namespace rv {

AnalysisPipeline::AnalysisPipeline() = default;

AnalysisPipeline::~AnalysisPipeline() {
    stop();
}

// ── Build ──────────────────────────────────────────────────────────

bool AnalysisPipeline::build(const AnalysisConfig& config, GMainLoop* shared_loop) {
    config_ = config;
    loop_ = shared_loop;

    pipeline_ = gst_pipeline_new("analysis-pipeline");
    if (!pipeline_) {
        fprintf(stderr, "[Analysis] Failed to create pipeline\n");
        return false;
    }

    // Create streammux
    streammux_ = gst_element_factory_make("nvstreammux", "analysis-streammux");
    if (!streammux_) {
        fprintf(stderr, "[Analysis] Failed to create nvstreammux\n");
        return false;
    }

    int num_cams = static_cast<int>(config_.cameras.size());
    if (num_cams > MAX_CAMERAS) num_cams = MAX_CAMERAS;

    g_object_set(G_OBJECT(streammux_),
        "batch-size",             num_cams,
        "width",                  config_.mux_width,
        "height",                 config_.mux_height,
        "batched-push-timeout",   40000,
        "live-source",            TRUE,
        "enable-padding",         TRUE,
        NULL);

    gst_bin_add(GST_BIN(pipeline_), streammux_);

    // Initialize SHM (all camera IDs registered)
    std::vector<std::string> cam_ids;
    for (const auto& cam : config_.cameras) {
        cam_ids.push_back(cam.id);
    }
    if (!shm_writer_.create(static_cast<uint32_t>(config_.cameras.size()), cam_ids)) {
        fprintf(stderr, "[Analysis] Failed to create SHM\n");
        return false;
    }

    // Load color classifier
    if (!config_.color_engine_path.empty()) {
        if (!color_infer_.load(config_.color_engine_path)) {
            fprintf(stderr, "[Analysis] Warning: color classifier not loaded\n");
        }
    }

    if (!add_sources()) return false;
    if (!add_inference()) return false;
    if (!add_sink()) return false;

    // Bus watch
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    gst_bus_add_watch(bus, bus_call, this);
    gst_object_unref(bus);

    fprintf(stderr, "[Analysis] Built (%d cameras, all valves closed, models loaded)\n",
            num_cams);
    return true;
}

bool AnalysisPipeline::add_sources() {
    for (size_t i = 0; i < config_.cameras.size(); ++i) {
        const auto& cam = config_.cameras[i];
        int idx = static_cast<int>(i);

        // Create valve (starts closed — all frames dropped)
        char valve_name[64];
        snprintf(valve_name, sizeof(valve_name), "analysis-valve-%d", idx);

        GstElement* valve = gst_element_factory_make("valve", valve_name);
        if (!valve) {
            fprintf(stderr, "[Analysis] Failed to create valve for cam %d\n", idx);
            return false;
        }
        g_object_set(G_OBJECT(valve), "drop", TRUE, NULL);
        gst_bin_add(GST_BIN(pipeline_), valve);

        // Link valve src → streammux sink pad
        char pad_name[32];
        snprintf(pad_name, sizeof(pad_name), "sink_%d", idx);

        GstPad* valve_src = gst_element_get_static_pad(valve, "src");
        GstPad* mux_sink = gst_element_request_pad_simple(streammux_, pad_name);
        if (valve_src && mux_sink) {
            if (gst_pad_link(valve_src, mux_sink) != GST_PAD_LINK_OK) {
                fprintf(stderr, "[Analysis] Failed to link valve-%d → streammux\n", idx);
            }
        }
        if (valve_src) gst_object_unref(valve_src);
        if (mux_sink) gst_object_unref(mux_sink);

        valves_[idx] = valve;

        // Create uridecodebin (RTSP connects at PLAYING)
        char src_name[64];
        snprintf(src_name, sizeof(src_name), "analysis-source-%d", idx);

        GstElement* source = gst_element_factory_make("uridecodebin", src_name);
        if (!source) {
            fprintf(stderr, "[Analysis] Failed to create uridecodebin for %s\n",
                    cam.id.c_str());
            return false;
        }

        g_object_set(G_OBJECT(source), "uri", cam.url.c_str(), NULL);
        g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), this);
        g_signal_connect(source, "child-added", G_CALLBACK(on_child_added), this);

        gst_bin_add(GST_BIN(pipeline_), source);
        sources_[idx] = source;
    }
    return true;
}

bool AnalysisPipeline::add_inference() {
    GstElement* nvinfer = gst_element_factory_make("nvinfer", "analysis-infer");
    if (!nvinfer) {
        fprintf(stderr, "[Analysis] Failed to create nvinfer\n");
        return false;
    }

    g_object_set(G_OBJECT(nvinfer),
        "config-file-path", config_.nvinfer_config.c_str(),
        "unique-id",        1,
        NULL);

    gst_bin_add(GST_BIN(pipeline_), nvinfer);

    if (!gst_element_link(streammux_, nvinfer)) {
        fprintf(stderr, "[Analysis] Failed to link streammux → nvinfer\n");
        return false;
    }

    GstPad* src_pad = gst_element_get_static_pad(nvinfer, "src");
    if (src_pad) {
        gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          inference_probe, this, nullptr);
        gst_object_unref(src_pad);
    }

    return true;
}

bool AnalysisPipeline::add_sink() {
    GstElement* sink = gst_element_factory_make("fakesink", "analysis-fakesink");
    if (!sink) {
        fprintf(stderr, "[Analysis] Failed to create fakesink\n");
        return false;
    }
    g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);
    gst_bin_add(GST_BIN(pipeline_), sink);

    GstElement* nvinfer = gst_bin_get_by_name(GST_BIN(pipeline_), "analysis-infer");
    if (nvinfer) {
        gst_element_link(nvinfer, sink);
        gst_object_unref(nvinfer);
    }

    return true;
}

// ── Activate / Deactivate (instant — just flip valve) ──────────────

void AnalysisPipeline::activate(int cam_index) {
    auto it = valves_.find(cam_index);
    if (it == valves_.end()) return;

    g_object_set(G_OBJECT(it->second), "drop", FALSE, NULL);
    fprintf(stderr, "[Analysis] Activated cam-%d\n", cam_index);
}

void AnalysisPipeline::deactivate(int cam_index) {
    auto it = valves_.find(cam_index);
    if (it == valves_.end()) return;

    g_object_set(G_OBJECT(it->second), "drop", TRUE, NULL);

    // Zero SHM slot
    if (cam_index < static_cast<int>(config_.cameras.size())) {
        CameraSlot empty_slot;
        init_camera_slot(empty_slot, config_.cameras[cam_index].id.c_str());
        shm_writer_.write_camera(cam_index, empty_slot);
        shm_writer_.commit();
    }

    fprintf(stderr, "[Analysis] Deactivated cam-%d\n", cam_index);
}

// ── Pad callbacks ──────────────────────────────────────────────────

void AnalysisPipeline::on_pad_added(GstElement* src, GstPad* pad, gpointer data) {
    auto* self = static_cast<AnalysisPipeline*>(data);

    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);

    const GstStructure* str = gst_caps_get_structure(caps, 0);
    const char* name = gst_structure_get_name(str);

    if (g_str_has_prefix(name, "video/")) {
        gchar* src_name = gst_element_get_name(src);
        int src_idx = 0;
        if (sscanf(src_name, "analysis-source-%d", &src_idx) == 1) {
            // Link uridecodebin video pad → valve sink
            char valve_name[64];
            snprintf(valve_name, sizeof(valve_name), "analysis-valve-%d", src_idx);

            GstElement* valve = gst_bin_get_by_name(GST_BIN(self->pipeline_), valve_name);
            if (valve) {
                GstPad* valve_sink = gst_element_get_static_pad(valve, "sink");
                if (valve_sink) {
                    if (gst_pad_link(pad, valve_sink) == GST_PAD_LINK_OK) {
                        // Reset reconnect state on successful link
                        if (self->reconnect_info_.count(src_idx) &&
                            self->reconnect_info_[src_idx].retry_count > 0) {
                            fprintf(stderr, "[Analysis] source-%d reconnected after %d attempts\n",
                                    src_idx, self->reconnect_info_[src_idx].retry_count);
                            self->reconnect_info_[src_idx].retry_count = 0;
                            self->reconnect_info_[src_idx].pending = false;
                        }
                    } else {
                        fprintf(stderr, "[Analysis] Failed to link %s → valve\n", src_name);
                    }
                    gst_object_unref(valve_sink);
                }
                gst_object_unref(valve);
            }
        }
        g_free(src_name);
    }

    gst_caps_unref(caps);
}

void AnalysisPipeline::on_child_added(GstChildProxy* proxy, GObject* object,
                                      gchar* name, gpointer data) {
    if (g_str_has_prefix(name, "source")) {
        g_object_set(object,
            "drop-on-latency", TRUE,
            "latency",         100,
            NULL);
    }
}

// ── Inference probe (filter + color classify + SHM write) ──────────

GstPadProbeReturn AnalysisPipeline::inference_probe(GstPad* pad,
                                                     GstPadProbeInfo* info,
                                                     gpointer data) {
    auto* self = static_cast<AnalysisPipeline*>(data);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    // Get surface for GPU crop access
    GstMapInfo map_info;
    NvBufSurface* surface = nullptr;
    if (gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        surface = reinterpret_cast<NvBufSurface*>(map_info.data);
        gst_buffer_unmap(buf, &map_info);
    }

    std::vector<TorsoROI> all_rois;
    struct RoiMapping {
        int cam_index;
        int det_index;
        float x1, y1, x2, y2;
        float center_x;
        float det_conf;
    };
    std::vector<RoiMapping> roi_mappings;

    struct CamDets {
        CameraSlot slot;
        int cam_index;
    };
    std::vector<CamDets> cam_dets_list;

    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr; l_frame = l_frame->next) {

        NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
        int cam_index = frame_meta->source_id;

        if (cam_index < 0 || cam_index >= static_cast<int>(self->config_.cameras.size()))
            continue;

        const auto& cam_config = self->config_.cameras[cam_index];
        int fw = frame_meta->source_frame_width;
        int fh = frame_meta->source_frame_height;

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

        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list;
             l_obj != nullptr && det_count < MAX_DETECTIONS;
             l_obj = l_obj->next) {

            NvDsObjectMeta* obj_meta = static_cast<NvDsObjectMeta*>(l_obj->data);
            if (obj_meta->class_id != 0) continue;

            float x1 = obj_meta->rect_params.left;
            float y1 = obj_meta->rect_params.top;
            float x2 = x1 + obj_meta->rect_params.width;
            float y2 = y1 + obj_meta->rect_params.height;
            float det_conf = obj_meta->confidence;
            float bw = x2 - x1;
            float bh = y2 - y1;
            float center_x = (x1 + x2) / 2.0f;

            if (x1 <= EDGE_MARGIN || x2 >= fw - EDGE_MARGIN) continue;
            if (bh < MIN_BBOX_HEIGHT) continue;
            if (bh / std::max(bw, 1.0f) < MIN_ASPECT_RATIO) continue;

            TorsoROI roi = ColorInfer::compute_torso_roi(
                x1, y1, x2, y2, fw, fh, det_count, cam_index);

            int crop_pixels = (roi.x2 - roi.x1) * (roi.y2 - roi.y1);
            if (crop_pixels < MIN_CROP_PIXELS || crop_pixels > MAX_CROP_PIXELS) continue;

            Detection& det = cam_dets.slot.detections[det_count];
            init_detection(det);
            det.x1       = x1;
            det.y1       = y1;
            det.x2       = x2;
            det.y2       = y2;
            det.center_x = center_x;
            det.det_conf = det_conf;

            all_rois.push_back(roi);
            roi_mappings.push_back({cam_index, det_count, x1, y1, x2, y2, center_x, det_conf});
            det_count++;
        }

        cam_dets.slot.num_detections = det_count;
        cam_dets_list.push_back(std::move(cam_dets));
    }

    // Batch color classification
    if (!all_rois.empty() && self->color_infer_.is_loaded() && surface) {
        std::vector<ColorResult> color_results;
        self->color_infer_.classify(surface, all_rois, color_results);

        for (size_t i = 0; i < color_results.size() && i < roi_mappings.size(); ++i) {
            const auto& cr = color_results[i];
            const auto& rm = roi_mappings[i];

            for (auto& cd : cam_dets_list) {
                if (cd.cam_index == rm.cam_index &&
                    rm.det_index < static_cast<int>(cd.slot.num_detections)) {
                    Detection& det = cd.slot.detections[rm.det_index];
                    det.color_id   = cr.color_id;
                    det.color_conf = cr.confidence;
                    std::memcpy(det.color_probs, cr.probs, sizeof(cr.probs));
                    break;
                }
            }
        }
    }

    // Write all camera slots to SHM
    for (const auto& cd : cam_dets_list) {
        self->shm_writer_.write_camera(cd.cam_index, cd.slot);
    }
    self->shm_writer_.commit();

    return GST_PAD_PROBE_OK;
}

// ── Bus handler ────────────────────────────────────────────────────

gboolean AnalysisPipeline::bus_call(GstBus* bus, GstMessage* msg, gpointer data) {
    auto* self = static_cast<AnalysisPipeline*>(data);

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        fprintf(stderr, "[Analysis] End of stream\n");
        break;

    case GST_MESSAGE_ERROR: {
        gchar* debug = nullptr;
        GError* error = nullptr;
        gst_message_parse_error(msg, &error, &debug);
        fprintf(stderr, "[Analysis] ERROR from %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
        if (debug) {
            fprintf(stderr, "[Analysis]   Debug: %s\n", debug);
            g_free(debug);
        }
        g_error_free(error);

        // Schedule reconnect for failed source
        int cam_idx = self->find_source_index(GST_ELEMENT(msg->src));
        if (cam_idx >= 0 && !self->reconnect_info_[cam_idx].pending) {
            auto& ri = self->reconnect_info_[cam_idx];
            ri.cam_index = cam_idx;
            ri.pending = true;
            int delay = std::min(5 * (1 << std::min(ri.retry_count, 3)), 30);
            fprintf(stderr, "[Analysis] Scheduling reconnect for source-%d in %ds\n",
                    cam_idx, delay);
            ri.timer_id = g_timeout_add_seconds(delay, reconnect_cb,
                new std::pair<AnalysisPipeline*, int>(self, cam_idx));
        }
        break;
    }

    case GST_MESSAGE_WARNING: {
        gchar* debug = nullptr;
        GError* error = nullptr;
        gst_message_parse_warning(msg, &error, &debug);
        fprintf(stderr, "[Analysis] WARNING from %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
        if (debug) g_free(debug);
        g_error_free(error);
        break;
    }

    case GST_MESSAGE_STATE_CHANGED:
        if (GST_MESSAGE_SRC(msg) == GST_OBJECT(self->pipeline_)) {
            GstState old_state, new_state, pending;
            gst_message_parse_state_changed(msg, &old_state, &new_state, &pending);
            fprintf(stderr, "[Analysis] State: %s → %s\n",
                    gst_element_state_get_name(old_state),
                    gst_element_state_get_name(new_state));
        }
        break;

    default:
        break;
    }

    return TRUE;
}

// ── RTSP reconnection ──────────────────────────────────────────────

int AnalysisPipeline::find_source_index(GstElement* element) {
    GstElement* cur = element;
    while (cur && cur != pipeline_) {
        gchar* name = gst_element_get_name(cur);
        int idx = -1;
        if (sscanf(name, "analysis-source-%d", &idx) == 1) {
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

gboolean AnalysisPipeline::reconnect_cb(gpointer data) {
    auto* p = static_cast<std::pair<AnalysisPipeline*, int>*>(data);
    p->first->attempt_reconnect(p->second);
    delete p;
    return FALSE;
}

void AnalysisPipeline::attempt_reconnect(int cam_idx) {
    if (cam_idx < 0 || cam_idx >= static_cast<int>(config_.cameras.size()))
        return;

    auto& ri = reconnect_info_[cam_idx];
    ri.timer_id = 0;

    const auto& cam = config_.cameras[cam_idx];
    fprintf(stderr, "[Analysis] Reconnecting source-%d (%s), attempt %d\n",
            cam_idx, cam.id.c_str(), ri.retry_count + 1);

    // Remove old source (keep valve in place)
    GstElement* old_source = sources_.count(cam_idx) ? sources_[cam_idx] : nullptr;
    if (old_source) {
        gst_element_set_state(old_source, GST_STATE_NULL);

        // Unlink from valve
        GstElement* valve = valves_.count(cam_idx) ? valves_[cam_idx] : nullptr;
        if (valve) {
            GstPad* valve_sink = gst_element_get_static_pad(valve, "sink");
            if (valve_sink) {
                GstPad* peer = gst_pad_get_peer(valve_sink);
                if (peer) {
                    gst_pad_unlink(peer, valve_sink);
                    gst_object_unref(peer);
                }
                gst_object_unref(valve_sink);
            }
        }

        gst_bin_remove(GST_BIN(pipeline_), old_source);
        sources_.erase(cam_idx);
    }

    // Create new uridecodebin
    char name[64];
    snprintf(name, sizeof(name), "analysis-source-%d", cam_idx);
    GstElement* new_source = gst_element_factory_make("uridecodebin", name);
    if (!new_source) {
        fprintf(stderr, "[Analysis] Failed to create uridecodebin for reconnect source-%d\n",
                cam_idx);
        ri.retry_count++;
        int delay = std::min(5 * (1 << std::min(ri.retry_count, 3)), 30);
        ri.timer_id = g_timeout_add_seconds(delay, reconnect_cb,
            new std::pair<AnalysisPipeline*, int>(this, cam_idx));
        return;
    }

    g_object_set(G_OBJECT(new_source), "uri", cam.url.c_str(), NULL);
    g_signal_connect(new_source, "pad-added", G_CALLBACK(on_pad_added), this);
    g_signal_connect(new_source, "child-added", G_CALLBACK(on_child_added), this);

    gst_bin_add(GST_BIN(pipeline_), new_source);
    gst_element_sync_state_with_parent(new_source);

    sources_[cam_idx] = new_source;
    ri.retry_count++;

    fprintf(stderr, "[Analysis] source-%d reconnect initiated\n", cam_idx);
}

// ── Start / Stop ───────────────────────────────────────────────────

void AnalysisPipeline::start() {
    if (!pipeline_) return;

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[Analysis] Failed to set pipeline to PLAYING\n");
        return;
    }

    running_ = true;
    fprintf(stderr, "[Analysis] Started (PLAYING, all 25 RTSP connected, valves closed)\n");
}

void AnalysisPipeline::stop() {
    running_ = false;

    for (auto& [idx, ri] : reconnect_info_) {
        if (ri.timer_id > 0) {
            g_source_remove(ri.timer_id);
            ri.timer_id = 0;
        }
    }
    reconnect_info_.clear();

    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_  = nullptr;
        streammux_ = nullptr;
    }

    sources_.clear();
    valves_.clear();

    shm_writer_.destroy();
    fprintf(stderr, "[Analysis] Stopped\n");
}

} // namespace rv
