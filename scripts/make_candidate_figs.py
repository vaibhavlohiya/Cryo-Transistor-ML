#!/usr/bin/env python3
"""Figures for the candidate-parameter screen (docs/CANDIDATE_PARAMETERS.md).

Reads the committed screen outputs — never hardcoded numbers:

  out/tables/candidate_sensitivity.json     Step B, 4 devices x 61 parameters
  out/tables/candidate_showmod_survey.json  Step A, 18 bins x 61 parameters

Renders four figures in light and dark variants:

  1. candidate_sensitivity        dot plot, every parameter x 4 devices
  2. candidate_length_dependence  dumbbells, short vs long channel
  3. candidate_vsat               published vsat per bin + sensitivity vs value
  4. candidate_box_locked         active parameters frozen at zero by the box

Encoding is doubled so identity never rests on hue alone: colour carries the
transistor family (blue = nMOS, orange = pMOS) and marker fill carries channel
length (filled = short, hollow = long). Palette values are the same theme as
make_params15_figs.py, already validated against both surfaces (worst CVD dE
24.7 light, 26.8 dark; >= 8 required); this figure uses a two-hue subset of
that set, so every pairwise margin is at least as large.

Sensitivity spans twelve decades, so every magnitude axis is logarithmic.
Two states cannot sit on a log axis and are drawn at the floor instead:
exactly-inert parameters (plotted as a ring on the floor line) and
box-frozen ones with no in-box test (plotted as a grey x) -- both are called
out in the legend rather than silently clipped.

  python scripts/make_candidate_figs.py            # light + dark PNG
  python scripts/make_candidate_figs.py --pdf      # also vector PDF
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SENS_JSON = ROOT / "out" / "tables" / "candidate_sensitivity.json"
SURVEY_JSON = ROOT / "out" / "tables" / "candidate_showmod_survey.json"
FIGS = ROOT / "figs"

# --- theme -----------------------------------------------------------------
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#2b2a28", "ink2": "#5c5a55",
        "muted": "#a3a099", "grid": "#eceae4", "axis": "#d5d3cb",
        "band": "#f4f2ec",
        "nmos": "#2a78d6", "pmos": "#eb6834", "base": "#b0ada5",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#8f8d86", "grid": "#2c2c2a", "axis": "#3f3f3c",
        "band": "#232322",
        "nmos": "#3987e5", "pmos": "#d95926", "base": "#7d7a74",
    },
}

# Probed devices, in reading order. ``short`` drives marker fill.
DEVICES = [
    ("nmos_L0p15_W1p6", "nMOS  L=0.15 um", "nmos", True),
    ("nmos_L1_W3", "nMOS  L=1 um", "nmos", False),
    ("pmos_L0p35_W1p6", "pMOS  L=0.35 um", "pmos", True),
    ("pmos_L4_W7", "pMOS  L=4 um", "pmos", False),
]

FLOOR = 1e-13          # log-axis floor; below this we draw the special states
C_LIGHT = 2.998e8      # m/s, the physicality line on the vsat panel

# Verdict bands, from docs/CANDIDATE_PARAMETERS.md
BANDS = [(FLOOR, 1e-6, "negligible"), (1e-6, 1e-3, "weak"),
         (1e-3, 1e-2, "moderate"), (1e-2, 10.0, "strong")]


def apply_theme(t: dict) -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
        "axes.titleweight": "semibold", "figure.dpi": 120,
        "legend.frameon": False,
    })


def _titles(fig, title: str, subtitle: str, t: dict, y=0.975, ys=0.945,
            x=0.012) -> None:
    fig.suptitle(title, fontsize=14, color=t["ink"], y=y, x=x, ha="left",
                 fontweight="semibold")
    fig.text(x, ys, subtitle, fontsize=9.5, color=t["ink2"], ha="left",
             va="top")


# --- data ------------------------------------------------------------------
def load() -> dict:
    for p in (SENS_JSON, SURVEY_JSON):
        if not p.exists():
            raise SystemExit(f"missing {p}; run the candidate screen first")
    rows = json.loads(SENS_JSON.read_text())
    survey = json.loads(SURVEY_JSON.read_text())

    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("status") == "not_echoed_by_showmod":
            continue
        by.setdefault(r["param"], {})[r["device"]] = r

    params = []
    for p, recs in by.items():
        sens = [recs.get(d, {}).get("d_rrms_pm10") for d, *_ in DEVICES]
        mrel = [recs.get(d, {}).get("maxrel_large") for d, *_ in DEVICES]
        live = [s for s in sens if s is not None]
        mrl = [m for m in mrel if m is not None]
        params.append({
            "param": p, "sens": sens, "maxrel": mrel,
            "max_sens": max(live) if live else None,
            "inert": bool(mrl) and max(mrl) == 0.0,
            "in15": any(recs.get(d, {}).get("in_set15") for d, *_ in DEVICES),
            "recs": recs,
        })
    return {"params": params, "survey": survey}


def _plot_state(ax, x, y, t, kind: str):
    """Draw the two states that cannot live on a log axis."""
    if kind == "inert":       # exactly zero response
        ax.plot(FLOOR * 2.2, y, "o", ms=5.5, mfc="none", mec=t["muted"],
                mew=1.4, zorder=3, clip_on=False)
    else:                     # published 0 -> no in-box test possible
        ax.plot(FLOOR * 2.2, y, "x", ms=5.5, color=t["muted"], mew=1.4,
                zorder=3, clip_on=False)


# --- figure 1: master sensitivity dot plot ---------------------------------
def fig_sensitivity(data: dict, t: dict):
    ps = [p for p in data["params"] if not p["inert"]]
    ps.sort(key=lambda p: (p["max_sens"] is None, -(p["max_sens"] or 0)))
    inert = sorted(p["param"] for p in data["params"] if p["inert"])

    half = (len(ps) + 1) // 2
    cols = [ps[:half], ps[half:]]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 0.30 * half + 2.6))

    for ax, col in zip(axes, cols):
        n = len(col)
        for lo, hi, name in BANDS:
            ax.axvspan(lo, hi, color=t["band"], zorder=0,
                       alpha=0.0 if name in ("weak", "strong") else 1.0)
        for i, p in enumerate(col):
            y = n - 1 - i
            ax.plot([FLOOR, 10], [y, y], color=t["grid"], lw=0.7, zorder=1)
            for (dtag, _lbl, fam, short), s in zip(DEVICES, p["sens"]):
                if s is None:
                    _plot_state(ax, None, y, t, "frozen")
                    continue
                if s <= FLOOR:
                    _plot_state(ax, None, y, t, "inert")
                    continue
                ax.plot(s, y, "o", ms=7.5, zorder=4,
                        mfc=t[fam] if short else t["surface"],
                        mec=t[fam], mew=1.8)
        ax.set_yticks(range(n), [p["param"] for p in reversed(col)],
                      fontsize=8.8)
        for lbl, p in zip(ax.get_yticklabels(), reversed(col)):
            if p["in15"]:
                lbl.set_fontweight("bold")
                lbl.set_color(t["ink"])
        ax.set_xscale("log")
        ax.set_xlim(FLOOR, 10)
        ax.set_ylim(-0.9, n - 0.1)
        ax.set_xlabel("max |ΔRRMS| under a ±10% perturbation  (log)",
                      fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    for ax, col in zip(axes, cols):
        for lo, hi, name in BANDS:
            ax.text(np.sqrt(lo * hi), len(col) - 0.45, name, ha="center",
                    va="bottom", fontsize=8, color=t["muted"])

    handles = [
        Line2D([], [], marker="o", ls="", ms=7.5, mfc=t["nmos"],
               mec=t["nmos"], mew=1.8, label="nMOS, short channel"),
        Line2D([], [], marker="o", ls="", ms=7.5, mfc=t["surface"],
               mec=t["nmos"], mew=1.8, label="nMOS, long channel"),
        Line2D([], [], marker="o", ls="", ms=7.5, mfc=t["pmos"],
               mec=t["pmos"], mew=1.8, label="pMOS, short channel"),
        Line2D([], [], marker="o", ls="", ms=7.5, mfc=t["surface"],
               mec=t["pmos"], mew=1.8, label="pMOS, long channel"),
        Line2D([], [], marker="o", ls="", ms=5.5, mfc="none", mec=t["muted"],
               mew=1.4, label="exactly zero response"),
        Line2D([], [], marker="x", ls="", ms=5.5, color=t["muted"], mew=1.4,
               label="published 0 — frozen by the box"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.5, columnspacing=1.6)

    _titles(fig, "Which BSIM4 parameters actually move the 77 K fit",
            "Sensitivity of the frozen-inclusion RRMS to a ±10% perturbation, "
            "measured in real NGSpice on four devices.  Bold = already one of "
            "the 15 tuned parameters.", t, y=0.985, ys=0.962)
    fig.text(0.012, 0.038,
             "Inert on every device (exactly zero response, omitted above):  "
             + ", ".join(inert),
             fontsize=8.6, color=t["muted"], ha="left")
    fig.tight_layout(rect=[0, 0.075, 1, 0.945])
    return fig


# --- figure 2: channel-length dependence -----------------------------------
def fig_length(data: dict, t: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    panels = [("nmos", "nMOS", "nmos_L0p15_W1p6", "nmos_L1_W3",
               "L=0.15 um", "L=1 um"),
              ("pmos", "pMOS", "pmos_L0p35_W1p6", "pmos_L4_W7",
               "L=0.35 um", "L=4 um")]

    for ax, (fam, name, dshort, dlong, lshort, llong) in zip(axes, panels):
        rows = []
        for p in data["params"]:
            s = p["recs"].get(dshort, {}).get("d_rrms_pm10")
            l = p["recs"].get(dlong, {}).get("d_rrms_pm10")
            if s is None or l is None or s <= 0:
                continue
            ratio = s / l if l and l > 0 else np.inf
            if s > 1e-4 and ratio > 5:
                rows.append((p["param"], s, l, ratio, p["in15"]))
        rows.sort(key=lambda r: r[1])
        rows = rows[-9:]

        for i, (param, s, l, ratio, in15) in enumerate(rows):
            lo = max(l, FLOOR * 2.2)
            ax.plot([lo, s], [i, i], color=t[fam], lw=2.0, alpha=0.45,
                    zorder=2, solid_capstyle="round")
            ax.plot(lo, i, "o", ms=8, mfc=t["surface"], mec=t[fam], mew=1.8,
                    zorder=4)
            ax.plot(s, i, "o", ms=8, mfc=t[fam], mec=t[fam], mew=1.8, zorder=4)
            # A ratio against a denominator that is itself numerically zero is
            # spurious precision -- report the collapse, not an 11-digit number.
            txt = "→ ~0" if (l is None or l <= 1e-9 or ratio >= 1e6) \
                else f"×{ratio:,.0f}"
            ax.text(s * 1.7, i, txt, va="center", fontsize=8.6,
                    color=t["ink2"])

        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9.5)
        for lbl, r in zip(ax.get_yticklabels(), rows):
            if r[4]:
                lbl.set_fontweight("bold")
                lbl.set_color(t["ink"])
        ax.set_xscale("log")
        ax.set_xlim(FLOOR, 30)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(f"{name}   {lshort}  vs  {llong}", fontsize=11.5,
                     color=t["ink"], loc="left", pad=8)
        ax.set_xlabel("max |ΔRRMS| under a ±10% perturbation  (log)",
                      fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="-", ms=8, lw=2, color=t["base"],
               mfc=t["base"], mec=t["base"], label="short channel (filled)"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=t["surface"],
               mec=t["base"], mew=1.8, label="long channel (hollow)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.005))
    _titles(fig, "Short-channel parameters are dead on long-channel devices",
            "The same parameter, measured on a short and a long device of the "
            "same family. Screening on one device does not generalise.",
            t, y=0.975, ys=0.925)
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    return fig


# --- figure 3: vsat --------------------------------------------------------
def fig_vsat(data: dict, t: dict):
    survey = data["survey"]
    tags = sorted(survey, key=lambda k: (survey[k]["dev_type"] != "nmos", k))
    vals = [survey[k]["values"]["vsat"] for k in tags]
    fams = [survey[k]["dev_type"] for k in tags]

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4),
                             gridspec_kw={"height_ratios": [1.25, 1]})

    ax = axes[0]
    x = np.arange(len(tags))
    for i, (v, fam) in enumerate(zip(vals, fams)):
        neg = v < 0
        ax.bar(i, abs(v), width=0.70, color=t["surface"] if neg else t[fam],
               edgecolor=t[fam], lw=1.8, hatch="///" if neg else None,
               zorder=3)
    ax.axhline(C_LIGHT, color=t["ink2"], ls="--", lw=1.4, zorder=4)
    ax.text(0.4, C_LIGHT * 1.7, "speed of light,  3.0e8 m/s", ha="left",
            va="bottom", fontsize=9, color=t["ink2"])
    # Non-physical = faster than light, or negative. Counted over the 18
    # devices actually studied, not over every model block in the card.
    n_np = sum(1 for v, f in zip(vals, fams)
               if f == "pmos" and (abs(v) > C_LIGHT or v < 0))
    n_p = sum(1 for f in fams if f == "pmos")
    # Above the panel: the bars fill the plot area, so an inside legend collides.
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=10, color=t["base"],
               label="published value positive"),
        Line2D([], [], marker="s", ls="", ms=10, mfc=t["surface"],
               mec=t["base"], mew=1.8, label="published value negative"),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.005), fontsize=9,
        labelcolor=t["ink2"], ncol=2)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e10)
    ax.set_xticks(x, [k.replace("_", " ").replace("nmos", "nMOS")
                      .replace("pmos", "pMOS") for k in tags],
                  rotation=40, ha="right", fontsize=8.2)
    ax.set_ylabel("published |vsat|   (m/s, log)", fontsize=10)
    ax.set_title(f"Published vsat on all 18 studied devices — "
                 f"{n_np} of {n_p} pMOS devices are non-physical "
                 f"(faster than light, or negative)",
                 fontsize=11.5, color=t["ink"], loc="left", pad=8)
    ax.grid(axis="x", visible=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    ax = axes[1]
    # Explicit label placement: two pMOS points sit close together at the
    # bottom right, so shared offsets collide.
    OFFSETS = {"nmos_L0p15_W1p6": (0, 22, "center"),
               "nmos_L1_W3": (0, 22, "center"),
               "pmos_L0p35_W1p6": (14, 18, "left"),
               "pmos_L4_W7": (-14, -32, "right")}
    recs = {p["param"]: p for p in data["params"]}["vsat"]["recs"]
    for dtag, lbl, fam, short in DEVICES:
        r = recs.get(dtag)
        if not r:
            continue
        raw = r["published"]
        pub, s = max(abs(raw), 1.2), max(r["d_rrms_pm10"], 2e-8)
        ax.plot(pub, s, "o", ms=13, zorder=4,
                mfc=t[fam] if short else t["surface"], mec=t[fam], mew=2.0)
        why = "physical" if (0 < raw < C_LIGHT) else (
            "NEGATIVE" if raw < 0 else "FASTER THAN LIGHT")
        dx, dy, ha = OFFSETS[dtag]
        ax.annotate(f"{lbl}\n{why}", (pub, s), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9, color=t["ink2"])
    ax.axvspan(C_LIGHT, 1e10, color=t["band"], zorder=0)
    ax.axvline(C_LIGHT, color=t["ink2"], ls="--", lw=1.4, zorder=2)
    ax.text(C_LIGHT * 1.6, 3e-2, "faster than light", fontsize=9,
            color=t["ink2"], ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e10)
    ax.set_ylim(1e-8, 1.0)
    ax.text(1.6, 3e-2, "negative,\ndrawn at |vsat|", fontsize=9,
            color=t["ink2"], ha="left", va="top")
    ax.set_xlabel("published |vsat|   (m/s, log)", fontsize=10)
    ax.set_ylabel("max |ΔRRMS|  (log)", fontsize=10)
    ax.set_title("Where vsat is non-physical, perturbing it changes nothing — "
                 "the dimension is wasted", fontsize=11.5, color=t["ink"],
                 loc="left", pad=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    _titles(fig, "vsat: a tuned parameter that does nothing on most pMOS devices",
            "An enormous VSAT pushes velocity saturation out of the operating "
            "range, so the fitter has effectively switched the effect off.",
            t, y=0.985, ys=0.958)
    fig.tight_layout(rect=[0, 0.01, 1, 0.925])
    return fig


# --- figure 4: box-locked but active ---------------------------------------
def fig_box_locked(data: dict, t: dict):
    rows = []
    for p in data["params"]:
        best = None
        for dtag, lbl, fam, short in DEVICES:
            r = p["recs"].get(dtag)
            if not r or r.get("published") != 0:
                continue
            m = r.get("maxrel_large")
            if m is not None and (best is None or m > best[0]):
                best = (m, lbl, fam, r.get("test_value"))
        if best and best[0] > 1e-3:
            rows.append((p["param"], *best, p["in15"]))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(11.2, 0.62 * len(rows) + 3.0))
    for i, (param, m, lbl, fam, tv, in15) in enumerate(rows):
        ax.barh(i, m * 100, height=0.62, color=t[fam], edgecolor=t[fam],
                lw=1.8, alpha=1.0 if in15 else 0.82, zorder=3)
        ax.text(m * 100 + 0.7, i, f"{m*100:.1f}%   ({lbl})", va="center",
                fontsize=9.2, color=t["ink2"])

    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11)
    for lbl_, r in zip(ax.get_yticklabels(), rows):
        if r[5]:
            lbl_.set_fontweight("bold")
            lbl_.set_color(t["ink"])
    ax.set_xlim(0, max(r[1] for r in rows) * 100 * 1.42)
    ax.set_xlabel("max change in drain current when perturbed away from zero "
                  "(% of peak |I|)", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)

    handles = [
        Line2D([], [], marker="s", ls="", ms=10, color=t["nmos"],
               label="locked on an nMOS bin"),
        Line2D([], [], marker="s", ls="", ms=10, color=t["pmos"],
               label="locked on a pMOS bin"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9.5,
              labelcolor=t["ink2"])

    _titles(fig, "Live physics the ±10% box cannot reach",
            "These parameters are published at exactly zero, so a "
            "multiplicative box freezes them — yet each moves the current "
            "substantially.\nBold = already one of the 15 tuned parameters.",
            t, y=0.975, ys=0.925)
    fig.tight_layout(rect=[0, 0.01, 1, 0.88])
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="also write vector PDF")
    ap.add_argument("--modes", default="light,dark")
    args = ap.parse_args()

    data = load()
    FIGS.mkdir(exist_ok=True)
    builders = {
        "candidate_sensitivity": fig_sensitivity,
        "candidate_length_dependence": fig_length,
        "candidate_vsat": fig_vsat,
        "candidate_box_locked": fig_box_locked,
    }
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
