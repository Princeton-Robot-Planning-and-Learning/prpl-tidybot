"""Wall time per localize() call, one process, on 12 frame sets (6 per spot)."""
import json, sys, time, pickle
import cv2 as cv
import numpy as np
import data, detectors as D
from evaluate import Baseline

threads = int(sys.argv[1])
cv.setNumThreads(threads)
R = "/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/analysis/2026-10-20-pilot/bin_detector/results"
cams = data.cameras()
items = [it for it in data.png_items() if it["spot"] in ("close", "far")]
sel = [it for it in items if it["spot"] == "close"][:6] + [it for it in items if it["spot"] == "far"][:6]
frames = [(data.load(it), it["exclude"]) for it in sel]
dets = {"sat130": Baseline(130.0), "edge": D.Detector("edge", cams),
        "learn": D.Detector("learn", cams, pickle.load(open(f"{R}/learn_model.pkl", "rb"))),
        "contrast": D.Detector("contrast", cams)}
out = {}
for name, det in dets.items():
    det.localize(*frames[0])  # warm-up
    ts = []
    for fr, ex in frames:
        t = time.perf_counter(); det.localize(fr, ex); ts.append(time.perf_counter() - t)
    out[name] = dict(median=float(np.median(ts)), max=float(np.max(ts)))
    print(threads, name, f"median {np.median(ts):.2f} s, max {np.max(ts):.2f} s", flush=True)
json.dump(out, open(f"{R}/timing_threads{threads}.json", "w"))
