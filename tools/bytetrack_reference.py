"""Reference checks for the C++ ByteTracker against Ultralytics BYTETracker.

Needs a Python environment with ultralytics and lap (e.g. on the PC used for model export).

  python bytetrack_reference.py synth <detections.txt>             synthetic sequence covering tracker edge cases
  python bytetrack_reference.py track <detections.txt> <tracks.txt>  run Ultralytics BYTETracker (bytetrack.yaml)
  python bytetrack_reference.py compare <expected.txt> <actual.txt>  per-frame comparison of two track files

File formats match apps/track_check.cpp (frames count from 1):
  detections: "F <frame> <n>" then n lines "<class> <score> <x1> <y1> <x2> <y2>"
  tracks:     "F <frame> <n>" then n lines "<id> <class> <score> <x1> <y1> <x2> <y2>"
"""

import sys

import numpy as np

FRAME_W, FRAME_H = 1280, 720


def read_frames(path):
    frames = []
    with open(path) as f:
        lines = iter(f.read().split("\n"))
        for line in lines:
            if not line.startswith("F "):
                continue
            _, frame, n = line.split()
            rows = [list(map(float, next(lines).split())) for _ in range(int(n))]
            frames.append((int(frame), rows))
    return frames


def write_frames(path, frames, fmt):
    with open(path, "w") as f:
        for frame, rows in frames:
            f.write(f"F {frame} {len(rows)}\n")
            for r in rows:
                f.write(fmt(r) + "\n")


def synth(path, num_frames=400, seed=7):
    """Moving boxes exercising: crossing paths, occlusion (lost -> re-found), low-score
    frames (second association), one-frame false positives (unconfirmed -> removed),
    a static object, an object leaving for good (removed after the buffer), clutter."""
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(1, num_frames + 1):
        dets = []

        def add(cls, score, cx, cy, w, h, jitter=2.0):
            cx, cy = cx + rng.normal(0, jitter), cy + rng.normal(0, jitter)
            w, h = w + rng.normal(0, jitter), h + rng.normal(0, jitter)
            x1, y1 = max(cx - w / 2, 0), max(cy - h / 2, 0)
            x2, y2 = min(cx + w / 2, FRAME_W), min(cy + h / 2, FRAME_H)
            if x2 > x1 and y2 > y1:
                dets.append([cls, float(np.clip(score, 0.0, 1.0)), x1, y1, x2, y2])

        # A: walks right; hidden (no detection) for frames 120-139.
        if not 120 <= f < 140:
            add(0, 0.85 + rng.normal(0, 0.03), 100 + 2.5 * f, 400, 120, 300)
        # B: walks left, crosses A around frame 180; weak (0.12-0.24) for frames 200-229.
        score_b = rng.uniform(0.12, 0.24) if 200 <= f < 230 else 0.8 + rng.normal(0, 0.03)
        add(0, score_b, 1150 - 2.2 * f, 420, 110, 290)
        # C: static object with a mid score.
        add(72, 0.5 + rng.normal(0, 0.05), 300, 250, 400, 450, jitter=1.0)
        # D: enters at frame 60, moves diagonally, leaves for good at frame 260.
        if 60 <= f < 260:
            add(2, 0.7 + rng.normal(0, 0.05), 200 + 3 * (f - 60), 100 + 1.5 * (f - 60), 160, 90)
        # E: one-frame false positives.
        if rng.random() < 0.05:
            add(rng.integers(0, 80), rng.uniform(0.26, 0.4), rng.uniform(100, 1180), rng.uniform(100, 620), 60, 60)
        # F: low-score clutter below the new-track threshold.
        for _ in range(rng.integers(0, 3)):
            add(rng.integers(0, 80), rng.uniform(0.1, 0.24), rng.uniform(50, 1230), rng.uniform(50, 670), 50, 50)
        frames.append((f, dets))
    write_frames(path, frames, lambda r: f"{int(r[0])} {r[1]:.6f} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f}")
    print(f"wrote {len(frames)} frames, {sum(len(d) for _, d in frames)} detections -> {path}")


def track(det_path, out_path):
    from ultralytics.cfg import ROOT
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace

    cfg = IterableSimpleNamespace(**YAML.load(ROOT / "cfg/trackers/bytetrack.yaml"))
    tracker = BYTETracker(cfg)
    frames = []
    for frame, rows in read_frames(det_path):
        # Boxes rows: x1 y1 x2 y2 conf cls
        data = np.array([[r[2], r[3], r[4], r[5], r[1], r[0]] for r in rows], dtype=np.float32).reshape(-1, 6)
        out = tracker.update(Boxes(data, orig_shape=(FRAME_H, FRAME_W)))
        # out rows: x1 y1 x2 y2 id score cls idx
        frames.append((frame, [[r[4], r[6], r[5], r[0], r[1], r[2], r[3]] for r in out.tolist()]))
    write_frames(out_path, frames, lambda r: f"{int(r[0])} {int(r[1])} {r[2]:.6f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f} {r[6]:.4f}")
    print(f"Ultralytics: {len(frames)} frames, {max((int(r[0]) for _, rs in frames for r in rs), default=0)} track ids -> {out_path}")


def compare(expected_path, actual_path):
    expected, actual = read_frames(expected_path), read_frames(actual_path)
    if len(expected) != len(actual):
        print(f"FAIL: {len(expected)} vs {len(actual)} frames")
        return 1
    bad_frames, max_box, max_score, total = [], 0.0, 0.0, 0
    for (fe, re_), (fa, ra) in zip(expected, actual):
        e = {int(r[0]): r for r in re_}
        a = {int(r[0]): r for r in ra}
        total += len(e)
        if fe != fa or e.keys() != a.keys() or any(int(e[k][1]) != int(a[k][1]) for k in e):
            bad_frames.append(fe)
            continue
        for k in e:
            max_box = max(max_box, max(abs(x - y) for x, y in zip(e[k][3:], a[k][3:])))
            max_score = max(max_score, abs(e[k][2] - a[k][2]))
    ok = not bad_frames and max_box < 0.01
    print(f"{len(expected)} frames, {total} track boxes: frames with different ids/classes: {len(bad_frames)}"
          f"{' (first: ' + str(bad_frames[:5]) + ')' if bad_frames else ''}; "
          f"max box diff {max_box:.4f} px, max score diff {max_score:.6f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "synth" and len(sys.argv) == 3:
        synth(sys.argv[2])
    elif cmd == "track" and len(sys.argv) == 4:
        track(sys.argv[2], sys.argv[3])
    elif cmd == "compare" and len(sys.argv) == 4:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
