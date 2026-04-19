/**
 * dual_pipeline.cpp — Coordinator wiring trigger → analysis.
 */

#include "dual_pipeline.h"

#include <cstdio>

namespace rv {

DualPipeline::DualPipeline() = default;

DualPipeline::~DualPipeline() {
    stop();
}

bool DualPipeline::build(const DualPipelineConfig& config) {
    gst_init(nullptr, nullptr);
    loop_ = g_main_loop_new(nullptr, FALSE);

    // Build trigger pipeline (shared loop)
    if (!trigger_.build(config.trigger, loop_)) {
        fprintf(stderr, "[DualPipeline] Failed to build trigger pipeline\n");
        return false;
    }

    // Build analysis pipeline (shared loop)
    if (!analysis_.build(config.analysis, loop_)) {
        fprintf(stderr, "[DualPipeline] Failed to build analysis pipeline\n");
        return false;
    }

    // Wire trigger activations → analysis valve control
    trigger_.set_activation_callback(
        [this](const std::set<int>& activated, const std::set<int>& deactivated) {
            for (int cam : deactivated) {
                analysis_.deactivate(cam);
            }
            for (int cam : activated) {
                analysis_.activate(cam);
            }
        }
    );

    fprintf(stderr, "[DualPipeline] Built (trigger + analysis)\n");
    return true;
}

void DualPipeline::start() {
    trigger_.start();
    analysis_.start();
    fprintf(stderr, "[DualPipeline] Both pipelines started\n");
}

void DualPipeline::stop() {
    trigger_.stop();
    analysis_.stop();

    if (loop_) {
        if (g_main_loop_is_running(loop_)) {
            g_main_loop_quit(loop_);
        }
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }

    fprintf(stderr, "[DualPipeline] Stopped\n");
}

} // namespace rv
