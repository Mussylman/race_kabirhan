#pragma once
/**
 * dual_pipeline.h — Coordinator for trigger + analysis pipelines.
 *
 * Creates a shared GMainLoop and wires the trigger's activation callback
 * to the analysis pipeline's add/remove source methods.
 */

#include "trigger_pipeline.h"
#include "analysis_pipeline.h"

#include <gst/gst.h>

namespace rv {

struct DualPipelineConfig {
    TriggerConfig   trigger;
    AnalysisConfig  analysis;
};

class DualPipeline {
public:
    DualPipeline();
    ~DualPipeline();

    bool build(const DualPipelineConfig& config);
    void start();
    void stop();

    GMainLoop* get_main_loop() const { return loop_; }

private:
    GMainLoop*        loop_ = nullptr;
    TriggerPipeline   trigger_;
    AnalysisPipeline  analysis_;
};

} // namespace rv
