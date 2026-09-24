"""Charts for the README from a pipeline_run log.

Needs Python 3 with matplotlib (e.g. on the PC used for model export).

  ./build/pipeline_run 30 > run.txt                                    on the Jetson
  python tools/plot_telemetry.py run.txt docs/images                   on the PC

Writes a light and a dark version of each chart, for GitHub's light and dark themes:
  latency_breakdown-{light,dark}.svg   mean time per stage, capture to tracks
  latency_timeline-{light,dark}.svg    per-second mean and p99 end-to-end latency
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

WINDOW = re.compile(
    r"in\s+([\d.]+) fps, out\s+([\d.]+) fps \| capture ([\d.]+) \| infer ([\d.]+) \(p99 ([\d.]+)\) \| "
    r"track ([\d.]+) \| sink ([\d.]+) ms \| latency ([\d.]+) ms \(p99 ([\d.]+)\)")
TOTAL = re.compile(r"^\s+(capture|inference|tracking|sink|latency \(end to end\))\s+mean\s+([\d.]+) ms")
FRAMES = re.compile(r"results (\d+) \(([\d.]+) fps\)")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926"]),
}
DPI = 100
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "font.size": 11})


def parse(path):
    windows, totals, fps = [], {}, None
    for line in Path(path).read_text().splitlines():
        if m := WINDOW.search(line):
            windows.append(dict(out_fps=float(m[2]), latency=float(m[8]), latency_p99=float(m[9])))
        elif m := TOTAL.match(line):
            totals[m[1].split()[0]] = float(m[2])
        elif m := FRAMES.search(line):
            fps = float(m[2])
    if not windows or len(totals) < 5:
        sys.exit(f"{path}: no pipeline_run telemetry found")
    return windows, totals, fps


def new_figure(t, width_px, height_px):
    fig = plt.figure(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)
    fig.patch.set_facecolor(t["surface"])
    return fig


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)


def titles(fig, t, title, subtitle):
    fig.text(0.03, 0.93, title, color=t["ink"], fontsize=15, fontweight="bold", va="top")
    fig.text(0.03, 0.83, subtitle, color=t["ink2"], fontsize=10.5, va="top")


def breakdown(totals, fps, t, out):
    # The stages don't cover the whole latency: the rest is the camera delivering the frame
    # (ISP, VIC conversion, appsink) and the hand-offs between threads.
    stages = [("Capture + preprocess", totals["capture"]),
              ("TensorRT + decode + NMS", totals["inference"]),
              ("ByteTrack", totals["tracking"]),
              ("Camera delivery + hand-offs",
               totals["latency"] - totals["capture"] - totals["inference"] - totals["tracking"])]
    fig = new_figure(t, 820, 300)
    titles(fig, t, f"Where the {totals['latency']:.1f} ms goes",
           f"Mean time per stage from capture to tracks, 1280×720, {fps:.1f} fps output, Jetson Orin Nano Super")
    ax = fig.add_axes([0.30, 0.12, 0.60, 0.58])
    style_axes(ax, t)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    # Data units are pixels here, so the bar shape is exact: 22 px thick, 4 px rounded data end,
    # square at the baseline.
    width_px, height_px = ax.get_window_extent().width, ax.get_window_extent().height
    row_px = height_px / len(stages)
    px_per_ms = width_px / (max(v for _, v in stages) * 1.12)
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_yticks([(i + 0.5) * row_px for i in range(len(stages))], [name for name, _ in stages],
                  color=t["ink2"], fontsize=10.5)
    bar, r = 22, 4
    for i, (_, value) in enumerate(stages):
        length, y = value * px_per_ms, (i + 0.5) * row_px - bar / 2
        if length > 2 * r:
            ax.add_patch(FancyBboxPatch((0, y), length, bar, boxstyle=f"round,pad=0,rounding_size={r}", lw=0,
                                        fc=t["series"][0]))
            ax.add_patch(Rectangle((0, y), r, bar, lw=0, fc=t["series"][0]))
        else:
            ax.add_patch(Rectangle((0, y), max(length, 2), bar, lw=0, fc=t["series"][0]))
        label = f"{value:.2f} ms" if value < 1 else f"{value:.1f} ms"
        ax.text(max(length, 2) + 8, y + bar / 2, label, va="center", color=t["ink"], fontsize=10.5)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def timeline(windows, t, out):
    # The first window includes startup (TensorRT warm-up, first camera buffers), so it is left out.
    steady = windows[1:]
    seconds = list(range(2, len(steady) + 2))
    mean = [w["latency"] for w in steady]
    p99 = [w["latency_p99"] for w in steady]
    fig = new_figure(t, 820, 380)
    titles(fig, t, "End-to-end latency, second by second",
           f"Capture to tracks, per 1 s window over {len(windows)} s (first second, startup, left out)")
    ax = fig.add_axes([0.08, 0.14, 0.76, 0.58])
    style_axes(ax, t)
    ax.set_ylim(0, max(max(p99) * 1.15, 20))
    ax.set_xlim(seconds[0] - 0.5, seconds[-1] + 0.5)
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.set_ylabel("ms", color=t["muted"], fontsize=10, rotation=0, labelpad=14, va="center")
    ax.set_xlabel("seconds", color=t["muted"], fontsize=10)
    for values, name, color, z in ((p99, "p99", t["series"][1], 2), (mean, "Mean", t["series"][0], 3)):
        ax.plot(seconds, values, color=color, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=z,
                label=name)
        ax.plot(seconds[-1], values[-1], "o", ms=8, color=color, mec=t["surface"], mew=2, zorder=z + 1)
    # Direct end labels; nudged apart only when the two ends would overlap.
    ymax = ax.get_ylim()[1]
    y_p99, y_mean = p99[-1], mean[-1]
    if (y_p99 - y_mean) < ymax * 0.07:
        y_p99 = y_mean + ymax * 0.07
    ax.text(seconds[-1] + 0.8, y_p99, f"p99 {p99[-1]:.1f} ms", va="center", color=t["ink"], fontsize=10.5)
    ax.text(seconds[-1] + 0.8, y_mean, f"mean {mean[-1]:.1f} ms", va="center", color=t["ink"], fontsize=10.5)
    handles, names = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], names[::-1], loc="lower left", frameon=False, ncol=2, fontsize=10,
                       handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(t["ink2"])
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    windows, totals, fps = parse(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        breakdown(totals, fps, theme, out_dir / f"latency_breakdown-{name}.svg")
        timeline(windows, theme, out_dir / f"latency_timeline-{name}.svg")
    print(f"wrote 4 charts to {out_dir}")


if __name__ == "__main__":
    main()
