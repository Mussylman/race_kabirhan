"""
pipeline_test.py — Phase 4 minimal smoke test for the new combined plugin.

Pipeline: filesrc -> h264parse -> nvv4l2decoder -> nvstreammux -> nvinfer
          (libnvdsinfer_racevision.so) -> probe(count) -> fakesink

Goal: prove that
  1. pyservicemaker can build and run a DS 9.0 pipeline
  2. nvinfer loads our new libnvdsinfer_racevision.so
  3. NvDsInferParseYoloV8 parser produces detections
  4. A Python probe receives BatchMetadata with the detections

No color classification or SHM writing yet — those come in step 4.5+.
"""

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator
from multiprocessing import Process
import argparse
import os
import sys

CONFIG_FILE = "/home/ipodrom/Рабочий стол/Ipodrom-Project/user/race_vision/deepstream/configs/nvinfer_racevision.txt"
STREAM_WIDTH  = 1280
STREAM_HEIGHT = 720


class DetectionCounter(BatchMetadataOperator):
    """Counts persons (class 0) per frame and prints summary."""

    def __init__(self):
        super().__init__()
        self.frames = 0
        self.total_persons = 0

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            person_count = 0
            for obj in frame_meta.object_items:
                if obj.class_id == 0:
                    person_count += 1
            self.frames += 1
            self.total_persons += person_count
            if self.frames % 25 == 0 or self.frames <= 5:
                avg = self.total_persons / max(self.frames, 1)
                print(f"[probe] frame={frame_meta.frame_number:5d}  "
                      f"persons_in_frame={person_count:2d}  "
                      f"total={self.total_persons:5d}  avg={avg:.2f}")


def main(video_path: str):
    if not os.path.isfile(video_path):
        sys.exit(f"video not found: {video_path}")

    uri = f"file://{video_path}"
    print(f"[pipeline_test] uri={uri}")
    print(f"[pipeline_test] config={CONFIG_FILE}")

    counter = DetectionCounter()

    pipe = (
        Pipeline("rv-test")
        .add("uridecodebin",  "src",   {"uri": uri})
        .add("nvstreammux",   "mux",
             {"batch-size": 1, "width": STREAM_WIDTH, "height": STREAM_HEIGHT,
              "batched-push-timeout": 40000})
        .add("nvinfer",       "infer", {"config-file-path": CONFIG_FILE, "unique-id": 1})
        .add("fakesink",      "sink",  {"sync": False, "async": False})
        .link(("src", "mux"), ("", "sink_%u"))
        .link("mux", "infer", "sink")
        .attach("infer", Probe("counter", counter))
    )

    print("[pipeline_test] starting pipeline...")
    pipe.start().wait()
    print(f"[pipeline_test] done. processed={counter.frames} total_persons={counter.total_persons}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to .mp4 file with H264 video")
    args = ap.parse_args()

    # multiprocessing wrapper so Ctrl-C is handled cleanly (per pyservicemaker examples)
    p = Process(target=main, args=(args.video,))
    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        p.terminate()
