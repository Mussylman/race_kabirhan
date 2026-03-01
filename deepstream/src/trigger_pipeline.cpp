/**
 * trigger_pipeline.cpp — Lightweight trigger pipeline implementation.
 *
 * Runs YOLOv8n on all 25 cameras at 640x640. The probe counts person
 * detections per camera and manages the active camera set with cooldown.
 */

#include "trigger_pipeline.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

#include <gst/gst.h>
#include <gstnvdsmeta.h>

namespace rv {

TriggerPipeline::TriggerPipeline() = default;

TriggerPipeline::~TriggerPipeline() {
    stop();
}

// ── Build ──────────────────────────────────────────────────────────

bool TriggerPipeline::build(const TriggerConfig& config, GMainLoop* shared_loop) {
    config_ = config;
    loop_ = shared_loop;  // borrowed

    pipeline_ = gst_pipeline_new("trigger-pipeline");
    if (!pipeline_) {
        fprintf(stderr, "[Trigger] Failed to create pipeline\n");
        return false;
    }

    // Create streammux (640x640, batch=num_cameras)
    streammux_ = gst_element_factory_make("nvstreammux", "trigger-streammux");
    if (!streammux_) {
        fprintf(stderr, "[Trigger] Failed to create nvstreammux\n");
        return false;
    }

    int batch = static_cast<int>(config_.cameras.size());
    if (batch > MAX_CAMERAS) batch = MAX_CAMERAS;

    g_object_set(G_OBJECT(streammux_),
        "batch-size",             batch,
        "width",                  config_.mux_width,
        "height",                 config_.mux_height,
        "batched-push-timeout",   40000,
        "live-source",            TRUE,
        "enable-padding",         TRUE,
        NULL);

    gst_bin_add(GST_BIN(pipeline_), streammux_);

    // Initialize trigger SHM
    if (!trigger_shm_.create(static_cast<uint32_t>(config_.cameras.size()))) {
        fprintf(stderr, "[Trigger] Failed to create trigger SHM\n");
        return false;
    }

    if (!add_sources()) return false;
    if (!add_inference()) return false;
    if (!add_sink()) return false;

    // Bus watch
    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    gst_bus_add_watch(bus, bus_call, this);
    gst_object_unref(bus);

    fprintf(stderr, "[Trigger] Built successfully (%zu cameras, %dx%d)\n",
            config_.cameras.size(), config_.mux_width, config_.mux_height);
    return true;
}

bool TriggerPipeline::add_sources() {
    for (size_t i = 0; i < config_.cameras.size(); ++i) {
        const auto& cam = config_.cameras[i];
        char name[64];
        snprintf(name, sizeof(name), "trigger-source-%zu", i);

        GstElement* source = gst_element_factory_make("uridecodebin", name);
        if (!source) {
            fprintf(stderr, "[Trigger] Failed to create uridecodebin for %s\n",
                    cam.id.c_str());
            return false;
        }

        g_object_set(G_OBJECT(source), "uri", cam.url.c_str(), NULL);
        g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), this);
        g_signal_connect(source, "child-added", G_CALLBACK(on_child_added), this);

        gst_bin_add(GST_BIN(pipeline_), source);
        sources_[static_cast<int>(i)] = source;
    }
    return true;
}

bool TriggerPipeline::add_inference() {
    GstElement* nvinfer = gst_element_factory_make("nvinfer", "trigger-infer");
    if (!nvinfer) {
        fprintf(stderr, "[Trigger] Failed to create nvinfer\n");
        return false;
    }

    g_object_set(G_OBJECT(nvinfer),
        "config-file-path", config_.nvinfer_config.c_str(),
        "unique-id",        2,  // different from analysis (1)
        NULL);

    gst_bin_add(GST_BIN(pipeline_), nvinfer);

    if (!gst_element_link(streammux_, nvinfer)) {
        fprintf(stderr, "[Trigger] Failed to link streammux → nvinfer\n");
        return false;
    }

    // Add trigger probe on nvinfer src pad
    GstPad* src_pad = gst_element_get_static_pad(nvinfer, "src");
    if (src_pad) {
        gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          trigger_probe, this, nullptr);
        gst_object_unref(src_pad);
    }

    return true;
}

bool TriggerPipeline::add_sink() {
    GstElement* sink = gst_element_factory_make("fakesink", "trigger-fakesink");
    if (!sink) {
        fprintf(stderr, "[Trigger] Failed to create fakesink\n");
        return false;
    }
    g_object_set(G_OBJECT(sink), "sync", FALSE, "async", FALSE, NULL);
    gst_bin_add(GST_BIN(pipeline_), sink);

    GstElement* nvinfer = gst_bin_get_by_name(GST_BIN(pipeline_), "trigger-infer");
    if (nvinfer) {
        gst_element_link(nvinfer, sink);
        gst_object_unref(nvinfer);
    }

    return true;
}

// ── Pad callbacks ──────────────────────────────────────────────────

void TriggerPipeline::on_pad_added(GstElement* src, GstPad* pad, gpointer data) {
    auto* self = static_cast<TriggerPipeline*>(data);

    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);

    const GstStructure* str = gst_caps_get_structure(caps, 0);
    const char* name = gst_structure_get_name(str);

    if (g_str_has_prefix(name, "video/")) {
        gchar* src_name = gst_element_get_name(src);
        int src_idx = 0;
        if (sscanf(src_name, "trigger-source-%d", &src_idx) == 1) {
            char pad_name[32];
            snprintf(pad_name, sizeof(pad_name), "sink_%d", src_idx);

            GstPad* mux_pad = gst_element_request_pad_simple(
                self->streammux_, pad_name);
            if (mux_pad) {
                if (gst_pad_link(pad, mux_pad) == GST_PAD_LINK_OK) {
                    if (self->reconnect_info_.count(src_idx) &&
                        self->reconnect_info_[src_idx].retry_count > 0) {
                        fprintf(stderr, "[Trigger] source-%d reconnected after %d attempts\n",
                                src_idx, self->reconnect_info_[src_idx].retry_count);
                        self->reconnect_info_[src_idx].retry_count = 0;
                        self->reconnect_info_[src_idx].pending = false;
                    }
                } else {
                    fprintf(stderr, "[Trigger] Failed to link %s to streammux\n", src_name);
                }
                gst_object_unref(mux_pad);
            }
        }
        g_free(src_name);
    }

    gst_caps_unref(caps);
}

void TriggerPipeline::on_child_added(GstChildProxy* proxy, GObject* object,
                                     gchar* name, gpointer data) {
    if (g_str_has_prefix(name, "source")) {
        g_object_set(object,
            "drop-on-latency", TRUE,
            "latency",         100,
            NULL);
    }
}

// ── Trigger probe (count detections, manage activations) ───────────

GstPadProbeReturn TriggerPipeline::trigger_probe(GstPad* pad,
                                                  GstPadProbeInfo* info,
                                                  gpointer data) {
    auto* self = static_cast<TriggerPipeline*>(data);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    uint32_t detection_counts[MAX_CAMERAS] = {0};

    // Iterate frames in batch — count person detections per camera
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr; l_frame = l_frame->next) {

        NvDsFrameMeta* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
        int cam_index = frame_meta->source_id;

        if (cam_index < 0 || cam_index >= static_cast<int>(self->config_.cameras.size()))
            continue;

        uint32_t count = 0;

        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list;
             l_obj != nullptr; l_obj = l_obj->next) {

            NvDsObjectMeta* obj_meta = static_cast<NvDsObjectMeta*>(l_obj->data);

            // Only person class (class_id 0)
            if (obj_meta->class_id != 0) continue;

            float bw = obj_meta->rect_params.width;
            float bh = obj_meta->rect_params.height;

            // Lightweight filters
            if (bh < self->config_.min_bbox_height) continue;
            if (bh / std::max(bw, 1.0f) < self->config_.min_aspect_ratio) continue;

            count++;
        }

        detection_counts[cam_index] = count;
    }

    // Update activations (thread-safe)
    self->update_activations(detection_counts);

    return GST_PAD_PROBE_OK;
}

// ── Activation logic ───────────────────────────────────────────────

void TriggerPipeline::update_activations(const uint32_t detection_counts[MAX_CAMERAS]) {
    auto now = std::chrono::steady_clock::now();

    std::set<int> newly_activated;
    std::set<int> newly_deactivated;

    {
        std::lock_guard<std::mutex> lock(activation_mutex_);

        int num_cams = static_cast<int>(config_.cameras.size());

        // Update last_detection_ timestamps for cameras with detections
        for (int i = 0; i < num_cams; ++i) {
            if (detection_counts[i] > 0) {
                last_detection_[i] = now;
            }
        }

        // Check for deactivations (cooldown expired)
        auto cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<float>(config_.cooldown_s));

        std::set<int> still_active;
        for (int cam : active_cameras_) {
            if (last_detection_.count(cam) &&
                (now - last_detection_[cam]) < cooldown) {
                still_active.insert(cam);
            } else {
                newly_deactivated.insert(cam);
            }
        }

        // Check for new activations
        for (int i = 0; i < num_cams; ++i) {
            if (detection_counts[i] > 0 && !still_active.count(i)) {
                // New camera wants activation
                if (static_cast<int>(still_active.size()) < config_.max_active) {
                    still_active.insert(i);
                    newly_activated.insert(i);
                } else {
                    // Evict oldest active camera to make room
                    int oldest = -1;
                    auto oldest_time = now;
                    for (int c : still_active) {
                        if (last_detection_.count(c) && last_detection_[c] < oldest_time) {
                            oldest_time = last_detection_[c];
                            oldest = c;
                        }
                    }
                    if (oldest >= 0) {
                        still_active.erase(oldest);
                        newly_deactivated.insert(oldest);
                        still_active.insert(i);
                        newly_activated.insert(i);
                    }
                }
            }
        }

        active_cameras_ = still_active;

        // Build bitmask and write trigger SHM
        uint32_t active_mask = 0;
        for (int cam : active_cameras_) {
            if (cam < 32) active_mask |= (1u << cam);
        }
        trigger_shm_.write(active_mask, detection_counts);
    }

    // Fire callback outside mutex
    if ((!newly_activated.empty() || !newly_deactivated.empty()) && activation_cb_) {
        activation_cb_(newly_activated, newly_deactivated);
    }
}

void TriggerPipeline::set_activation_callback(ActivationCallback cb) {
    std::lock_guard<std::mutex> lock(activation_mutex_);
    activation_cb_ = std::move(cb);
}

std::set<int> TriggerPipeline::get_active_cameras() const {
    std::lock_guard<std::mutex> lock(activation_mutex_);
    return active_cameras_;
}

// ── Bus message handler ────────────────────────────────────────────

gboolean TriggerPipeline::bus_call(GstBus* bus, GstMessage* msg, gpointer data) {
    auto* self = static_cast<TriggerPipeline*>(data);

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        fprintf(stderr, "[Trigger] End of stream\n");
        if (self->loop_) g_main_loop_quit(self->loop_);
        break;

    case GST_MESSAGE_ERROR: {
        gchar* debug = nullptr;
        GError* error = nullptr;
        gst_message_parse_error(msg, &error, &debug);
        fprintf(stderr, "[Trigger] ERROR from %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
        if (debug) {
            fprintf(stderr, "[Trigger]   Debug: %s\n", debug);
            g_free(debug);
        }
        g_error_free(error);

        int cam_idx = self->find_source_index(GST_ELEMENT(msg->src));
        if (cam_idx >= 0 && !self->reconnect_info_[cam_idx].pending) {
            auto& ri = self->reconnect_info_[cam_idx];
            ri.cam_index = cam_idx;
            ri.pending = true;
            int delay = std::min(5 * (1 << std::min(ri.retry_count, 3)), 30);
            fprintf(stderr, "[Trigger] Scheduling reconnect for source-%d in %ds\n",
                    cam_idx, delay);
            ri.timer_id = g_timeout_add_seconds(delay, reconnect_cb,
                new std::pair<TriggerPipeline*, int>(self, cam_idx));
        }
        break;
    }

    case GST_MESSAGE_WARNING: {
        gchar* debug = nullptr;
        GError* error = nullptr;
        gst_message_parse_warning(msg, &error, &debug);
        fprintf(stderr, "[Trigger] WARNING from %s: %s\n",
                GST_OBJECT_NAME(msg->src), error->message);
        if (debug) g_free(debug);
        g_error_free(error);
        break;
    }

    case GST_MESSAGE_STATE_CHANGED:
        if (GST_MESSAGE_SRC(msg) == GST_OBJECT(self->pipeline_)) {
            GstState old_state, new_state, pending;
            gst_message_parse_state_changed(msg, &old_state, &new_state, &pending);
            fprintf(stderr, "[Trigger] State: %s → %s\n",
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

int TriggerPipeline::find_source_index(GstElement* element) {
    GstElement* cur = element;
    while (cur && cur != pipeline_) {
        gchar* name = gst_element_get_name(cur);
        int idx = -1;
        if (sscanf(name, "trigger-source-%d", &idx) == 1) {
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

gboolean TriggerPipeline::reconnect_cb(gpointer data) {
    auto* p = static_cast<std::pair<TriggerPipeline*, int>*>(data);
    p->first->attempt_reconnect(p->second);
    delete p;
    return FALSE;
}

void TriggerPipeline::attempt_reconnect(int cam_idx) {
    if (cam_idx < 0 || cam_idx >= static_cast<int>(config_.cameras.size()))
        return;

    auto& ri = reconnect_info_[cam_idx];
    ri.timer_id = 0;

    const auto& cam = config_.cameras[cam_idx];
    fprintf(stderr, "[Trigger] Reconnecting source-%d (%s), attempt %d\n",
            cam_idx, cam.id.c_str(), ri.retry_count + 1);

    // Remove old source
    GstElement* old_source = sources_.count(cam_idx) ? sources_[cam_idx] : nullptr;
    if (old_source) {
        gst_element_set_state(old_source, GST_STATE_NULL);

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

    // Create new source
    char name[64];
    snprintf(name, sizeof(name), "trigger-source-%d", cam_idx);
    GstElement* new_source = gst_element_factory_make("uridecodebin", name);
    if (!new_source) {
        fprintf(stderr, "[Trigger] Failed to create uridecodebin for reconnect source-%d\n",
                cam_idx);
        ri.retry_count++;
        int delay = std::min(5 * (1 << std::min(ri.retry_count, 3)), 30);
        ri.timer_id = g_timeout_add_seconds(delay, reconnect_cb,
            new std::pair<TriggerPipeline*, int>(this, cam_idx));
        return;
    }

    g_object_set(G_OBJECT(new_source), "uri", cam.url.c_str(), NULL);
    g_signal_connect(new_source, "pad-added", G_CALLBACK(on_pad_added), this);
    g_signal_connect(new_source, "child-added", G_CALLBACK(on_child_added), this);

    gst_bin_add(GST_BIN(pipeline_), new_source);
    gst_element_sync_state_with_parent(new_source);

    sources_[cam_idx] = new_source;
    ri.retry_count++;

    fprintf(stderr, "[Trigger] source-%d reconnect initiated\n", cam_idx);
}

// ── Start / Stop ───────────────────────────────────────────────────

void TriggerPipeline::start() {
    if (!pipeline_) return;

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[Trigger] Failed to set pipeline to PLAYING\n");
        return;
    }

    running_ = true;
    fprintf(stderr, "[Trigger] Started (PLAYING)\n");
}

void TriggerPipeline::stop() {
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

    // Don't unref loop_ — we don't own it

    trigger_shm_.destroy();
    fprintf(stderr, "[Trigger] Stopped\n");
}

} // namespace rv
