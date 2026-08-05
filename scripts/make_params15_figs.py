#!/usr/bin/env python3
"""Publication figures for the params15 (7- vs 15-parameter) study.

Reads the committed result tables — never hardcoded numbers — and renders
three figures in light and dark variants:

  1. params15_aggregate      grouped bars, 4 aggregates x 5 fixed methods
  2. params15_per_device     per-device dumbbells, faceted by family
  3. params15_fd_recovery    raw -> FD slope, showing the 7/15 crossing

Encoding is deliberately doubled so identity never rests on hue alone:
colour carries the parameter count (blue = 7, orange = 15), hatching carries
the search stage (hatched = raw, solid = FD-polished), and the published-card
baseline stays neutral grey as chart chrome. Palette values are the dataviz
reference palette; they pass the lightness/chroma/CVD/contrast checks in both
modes (worst CVD dE 24.7 light, 26.8 dark; >= 8 required).

  python scripts/make_params15_figs.py            # light + dark PNG
  python scripts/make_params15_figs.py --pdf      # also vector PDF
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_CSV = ROOT / "out" / "tables" / "comparison.csv"
PARAMS15_JSON = ROOT / "out" / "tables" / "params15_study.json"
FIGS = ROOT / "figs"

# --- theme -----------------------------------------------------------------
# Two selected modes (dark is re-stepped for the dark surface, not an inversion).
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "p7": "#2a78d6", "p15": "#eb6834", "base": "#898781",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "p7": "#3987e5", "p15": "#d95926", "base": "#898781",
    },
}

AGGREGATES = ("nMOS (8)", "pMOS (10)", "Combined", "All-device (18)")


def device_label(tag: str) -> str:
    """``nmos_L0p15_W1p6`` -> ``L=0.15  W=1.6`` (family is the panel title).

    Only the ``p`` inside a numeric field is a decimal point — a blanket
    replace would turn ``pmos`` into ``.mos``.
    """
    _, length, width = tag.split("_")
    return f"L={length[1:].replace('p', '.')}   W={width[1:].replace('p', '.')}"


# --- data ------------------------------------------------------------------
def load_results() -> dict:
    """Per-device RRMS for the five fixed series, plus family aggregates."""
    for path in (COMPARISON_CSV, PARAMS15_JSON):
        if not path.exists():
            raise SystemExit(f"missing {path}; run the pipeline first")

    with COMPARISON_CSV.open() as f:
        rows7 = {r["device"]: r for r in csv.DictReader(f)}
    p15 = {r["device"]: r
           for r in json.loads(PARAMS15_JSON.read_text())["per_device"]}

    devices = sorted(rows7, key=lambda d: (not d.startswith("nmos"), d))
    series = {
        "paper": [float(rows7[d]["paper_params_ngspice"]) for d in devices],
        "p7_raw": [float(rows7[d]["surrogate_search_raw"]) for d in devices],
        "p7_fd": [float(rows7[d]["surrogate_search_plus_fd"]) for d in devices],
        "p15_raw": [float(p15[d]["emu_search_rrms"]) for d in devices],
        "p15_fd": [float(p15[d]["emu_search_fd_rrms"]) for d in devices],
    }
    fam = np.array([d.split("_")[0] for d in devices])
    agg = {}
    for key, vals in series.items():
        v = np.asarray(vals)
        n, p = v[fam == "nmos"].mean(), v[fam == "pmos"].mean()
        agg[key] = np.array([n, p, (n + p) / 2, v.mean()])
    return {"devices": devices, "family": fam,
            "series": {k: np.asarray(v) for k, v in series.items()},
            "aggregates": agg}


def apply_theme(t: dict) -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        # DejaVu ships with matplotlib: deterministic rendering everywhere,
        # and it has the arrow/minus glyphs the labels use.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
        "axes.titleweight": "semibold", "figure.dpi": 120,
        "legend.frameon": False,
    })


def _titles(fig, title: str, subtitle: str, t: dict, y=0.975, ys=0.93) -> None:
    fig.suptitle(title, fontsize=14, color=t["ink"], y=y, x=0.012, ha="left",
                 fontweight="semibold")
    fig.text(0.012, ys, subtitle, fontsize=9.5, color=t["ink2"], ha="left",
             va="top")


# --- figure 1: aggregate grouped bars --------------------------------------
def fig_aggregate(data: dict, t: dict):
    agg = data["aggregates"]
    bars = [
        ("paper", "Published card", t["base"], None),
        ("p7_raw", "7-param, raw", t["p7"], "////"),
        ("p7_fd", "7-param + FD", t["p7"], None),
        ("p15_raw", "15-param, raw", t["p15"], "////"),
        ("p15_fd", "15-param + FD", t["p15"], None),
    ]
    x = np.arange(len(AGGREGATES))
    width = 0.155

    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.075, right=0.985)

    for i, (key, label, colour, hatch) in enumerate(bars):
        offset = (i - (len(bars) - 1) / 2) * width
        ax.bar(x + offset, agg[key], width * 0.92, label=label,
               facecolor=colour, edgecolor=t["surface"], linewidth=1.4,
               hatch=hatch, zorder=3)
        for xi, val in zip(x + offset, agg[key]):
            ax.text(xi, val + 0.006, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.1, color=t["ink2"], rotation=90, zorder=4)

    ax.set_xticks(x, AGGREGATES, fontsize=10.5, color=t["ink"])
    ax.set_ylabel("RRMS  (lower is better ↓)", fontsize=10.5)
    ax.set_ylim(0, max(agg["paper"].max(), agg["p15_raw"].max()) * 1.20)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, left=True, bottom=False)
    ax.tick_params(axis="x", length=0, pad=6)

    _titles(fig,
            "Extending the tuned BSIM4 set from 7 to 15 parameters lowers "
            "RRMS on every aggregate",
            "SKY130 at 77 K, all 18 Table-6 devices. Identical protocol, box, "
            "metric and search recipe; only the theta vector differs.\n"
            "Colour = parameter count, hatching = before FD polish. Frozen "
            "published-card curve inclusion; every value is a real NGSpice "
            "re-simulation.", t)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=5,
              fontsize=9.5, labelcolor=t["ink2"], handlelength=1.5,
              columnspacing=1.5, handleheight=1.1)
    fig.text(0.075, 0.035,
             "Combined = (nMOS mean + pMOS mean) / 2, the upstream headline "
             "convention.  All-device weights all 18 transistors equally.  "
             "The two are not interchangeable.",
             fontsize=8.2, color=t["muted"], ha="left")
    return fig


# --- figure 2: per-device dumbbells -----------------------------------------
def fig_per_device(data: dict, t: dict):
    s, devices, fam = data["series"], data["devices"], data["family"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.115, right=0.985,
                        wspace=0.42)

    for ax, family, name in zip(axes, ("nmos", "pmos"), ("nMOS", "pMOS")):
        idx = [i for i, f in enumerate(fam) if f == family]
        idx.sort(key=lambda i: s["paper"][i])          # worst at top
        y = np.arange(len(idx))
        paper = s["paper"][idx]
        p7 = s["p7_fd"][idx]
        p15 = s["p15_fd"][idx]

        ax.hlines(y, np.minimum(p15, paper), np.maximum(p15, paper),
                  color=t["axis"], linewidth=1.6, zorder=2)
        # Sizes descend with draw order so near-coincident markers (a device
        # the 7-param run barely moved) still show every series.
        ax.scatter(paper, y, s=86, color=t["base"], marker="o", zorder=3,
                   edgecolor=t["surface"], linewidth=1.1,
                   label="Published card")
        ax.scatter(p7, y, s=46, color=t["p7"], marker="s", zorder=4,
                   edgecolor=t["surface"], linewidth=1.1,
                   label="7-param + FD")
        ax.scatter(p15, y, s=58, color=t["p15"], marker="D", zorder=5,
                   edgecolor=t["surface"], linewidth=1.1,
                   label="15-param + FD")

        ax.set_yticks(y, [device_label(devices[i]) for i in idx], fontsize=9)
        ax.set_ylim(-0.8, len(idx) - 0.2)
        ax.set_xlim(0, paper.max() * 1.30)
        ax.set_xlabel("RRMS  (lower is better ↓)", fontsize=10)
        ax.set_title(f"{name}  ({len(idx)} devices)", fontsize=11.5,
                     color=t["ink"], loc="left", pad=8)
        ax.yaxis.grid(False)
        sns.despine(ax=ax, left=True, bottom=False)
        ax.tick_params(axis="y", length=0)

        for yi, pa, pn in zip(y, paper, p15):
            ax.text(ax.get_xlim()[1] * 0.995, yi,
                    f"−{100 * (pa - pn) / pa:.0f}%", ha="right",
                    va="center", fontsize=8.5, color=t["p15"],
                    fontweight="semibold")

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.115), ncol=3,
                   fontsize=10, labelcolor=t["ink2"])
    _titles(fig,
            "The 15-parameter card wins on 18 of 18 devices",
            "Each row is one transistor geometry (L and W in µm); the bar "
            "spans the published card and the 15-parameter result. "
            "Percentages are the RRMS reduction versus the published card.\n"
            "Note the different x-scales: the pMOS family carries most of the "
            "residual error, and gains the most.", t, y=0.965, ys=0.915)
    return fig


# --- figure 3: raw -> FD slope ----------------------------------------------
def fig_fd_recovery(data: dict, t: dict):
    agg = data["aggregates"]
    panels = [("All 18 devices (arithmetic mean)", 3),
              ("pMOS family (10 devices)", 1)]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.6))
    fig.subplots_adjust(top=0.74, bottom=0.13, left=0.085, right=0.975,
                        wspace=0.26)

    for ax, (name, col) in zip(axes, panels):
        for key_raw, key_fd, colour, label in (
                ("p7_raw", "p7_fd", t["p7"], "7-parameter"),
                ("p15_raw", "p15_fd", t["p15"], "15-parameter")):
            ys = [agg[key_raw][col], agg[key_fd][col]]
            ax.plot([0, 1], ys, color=colour, linewidth=2.4, marker="o",
                    markersize=9, markeredgecolor=t["surface"],
                    markeredgewidth=1.4, label=label, zorder=4)
            for xi, val, ha in ((0, ys[0], "right"), (1, ys[1], "left")):
                ax.text(xi + (-0.06 if ha == "right" else 0.06), val,
                        f"{val:.4f}", ha=ha, va="center", fontsize=10,
                        color=colour, fontweight="semibold", zorder=5)

        base = agg["paper"][col]
        ax.axhline(base, color=t["base"], linestyle=(0, (5, 4)), linewidth=1.6,
                   zorder=2)
        # Sits just above the rule at the right edge, where no series line
        # runs, so nothing has to be masked out.
        ax.annotate(f"published card  {base:.4f}", (1.36, base),
                    xytext=(0, 4), textcoords="offset points", ha="right",
                    va="bottom", fontsize=8.8, color=t["ink2"])

        ax.set_xticks([0, 1], ["surrogate search\n(raw)", "+ FD polish"],
                      fontsize=10.5, color=t["ink"])
        ax.set_xlim(-0.38, 1.38)
        lo = min(agg["p7_fd"][col], agg["p15_fd"][col])
        hi = max(agg["p7_raw"][col], agg["p15_raw"][col], base)
        pad = (hi - lo) * 0.28
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(name, fontsize=11.5, color=t["ink"], loc="left", pad=8)
        ax.set_ylabel("RRMS  (lower is better ↓)", fontsize=10.5)
        ax.xaxis.grid(False)
        sns.despine(ax=ax, left=True, bottom=False)
        ax.tick_params(axis="x", length=0, pad=6)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.13, -0.115), ncol=2,
                   fontsize=10.5, labelcolor=t["ink2"])
    _titles(fig,
            "In 15 dimensions the surrogate only finds the basin — FD "
            "polish does the precision work",
            "With the same 10,000-sample budget the raw 15-parameter search is "
            "worse than the raw 7-parameter search, yet ends better after "
            "identical FD polish.\nThe lines cross: more parameters need the "
            "numerical stage more, not less.", t, y=0.965, ys=0.915)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="also write vector PDF")
    ap.add_argument("--modes", default="light,dark")
    args = ap.parse_args()

    data = load_results()
    FIGS.mkdir(exist_ok=True)
    builders = {"params15_aggregate": fig_aggregate,
                "params15_per_device": fig_per_device,
                "params15_fd_recovery": fig_fd_recovery}

    for mode in args.modes.split(","):
        theme = THEMES[mode.strip()]
        apply_theme(theme)
        for stem, builder in builders.items():
            fig = builder(data, theme)
            suffix = "" if mode == "light" else "_dark"
            for ext in (["png", "pdf"] if args.pdf else ["png"]):
                out = FIGS / f"{stem}{suffix}.{ext}"
                fig.savefig(out, dpi=220, bbox_inches="tight",
                            facecolor=theme["surface"])
                print(f"wrote {out.relative_to(ROOT)}")
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
