"""Per-device I-V detail: measured vs published card vs 7-param vs 15-param.

One figure per paper device, two panels: output characteristics (the fixed-VGS
family) on the left, transfer characteristics (the fixed-VDS family) on the
right. Unlike the 18-panel overview in ``plot_all_devices_iv.py``, this draws
**every** measured bias curve -- all 5 output and all 6 transfer sweeps -- so a
device can be inspected in full.

Series (four, each a real NGSpice re-simulation or the measurement itself;
never emulator output):

===================  ==========================  ===================
series               source                      mark
===================  ==========================  ===================
measured 77 K        vendored paper repo         open circles, no line
published card       paper's 77 K parameters     dashed
7-parameter          canonical ``emu_search+fd`` solid
15-parameter         ``params15`` surrogate+FD   dash-dot
===================  ==========================  ===================

Encoding. Colour carries **model identity**, not bias: within a panel the bias
curves are already stacked by physics (higher |V| -> higher |I|), so colouring
them would re-encode what y-position shows and spend the only free channel on
information already present. Each bias family is instead named by a direct
label at the right end of its curve, greedily de-collided in y. Series identity
is additionally carried by dash pattern and marker shape, so the panels survive
greyscale printing and colour-vision deficiency.

Palette: 7-param ``#2a78d6``, published ``#eb6834``, 15-param ``#4a3aa7``,
measured ``#3f3d39`` on surface ``#fcfcfb``. The three model hues are slots 1,
2 and 7 of the documented categorical palette and each passes all six checks
(lightness band, chroma floor, contrast >= 3:1). Validated all-pairs -- the
harder test, correct here because any two curves can sit adjacent -- at worst
normal-vision dE 16.3 (floor 15) and worst CVD dE 14.6 (target 8) under
protanopia and deuteranopia simulated with Machado-Oliveira-Fernandes 2009 at
severity 1.0. Measured is deliberately achromatic and below the chroma floor:
it is the reference the models are judged against rather than a competing
identity, and it is the only series drawn as bare markers. Preserve this
reasoning if the colours change.

The 15-parameter series is a **labeled experiment**, not the canonical
protocol: it tunes a different theta vector, so it is not comparable to the
paper's method and does not enter Table 4/6, headline RRMS, or the exported
card. The figures label it as such.

Usage::

    # all 18 devices
    PYTHONPATH=src .venv/bin/python scripts/plot_device_detail.py

    # one device, vector output
    PYTHONPATH=src .venv/bin/python scripts/plot_device_detail.py \\
        --device pmos_L2_W5 --pdf

The 7-parameter curves come from ``out/pdk_ml_emu`` if present, else the
re-simulated ``out/pdk_ml_emu_resim`` written by ``resim_fixed_series.py``.
No simulator is invoked here; every curve is read from a saved ``.npz``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))  # sibling helpers, kept in one place

from cryoml.config import FIGS_DIR, OUT_DIR  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402
from plot_all_devices_iv import baseline_inclusion, load_series  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#2b2a28", "#5c5a55", "#a3a099"
GRID, AXIS = "#eceae4", "#d5d3cb"

MEASURED_C = "#3f3d39"

# Fixed order; colour follows the entity, never its rank or its score.
SERIES = [
    ("paper", "published card", "#eb6834", (0, (5, 2.2)), "s"),
    ("emu7", "7-parameter (surrogate + FD)", "#2a78d6", "-", "^"),
    ("emu15", "15-parameter (surrogate + FD)", "#4a3aa7", (0, (6, 1.6, 1, 1.6)),
     "D"),
]

plt.rcParams.update({
    "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200, "figure.dpi": 200,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.alpha": 1.0, "axes.axisbelow": True,
})

PANELS = {
    "idvd": {"x": lambda c: c.Vd, "xlabel": "|V$_{DS}$| (V)",
             "bias": "V$_{GS}$", "title": "Output  (fixed V$_{GS}$ family)"},
    "idvg": {"x": lambda c: c.Vg, "xlabel": "|V$_{GS}$| (V)",
             "bias": "V$_{DS}$", "title": "Transfer  (fixed V$_{DS}$ family)"},
}


def bias_curves(curves, kind):
    """All curves of one sweep kind, weakest bias first."""
    pairs = [(i, c) for i, c in enumerate(curves) if c.kind == kind]
    return sorted(pairs, key=lambda p: abs(p[1].fixed))


def place_end_labels(ax, entries) -> None:
    """Label each bias family at its curve's right end, de-collided in y.

    Colour is spent on model identity, so the bias value has to be said in
    words. Labels are pushed apart greedily in axes coordinates because the
    low-bias curves bunch together near zero on most devices.
    """
    if not entries:
        return
    entries = sorted(entries, key=lambda e: e[0])
    min_gap = 0.052
    want = [min(max(ax.transLimits.transform((ax.get_xlim()[1], y))[1], 0.0),
                1.0) for y, _ in entries]

    placed: list[float] = []
    for y in want:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
    # The upward pass accumulates drift wherever curves converge (the
    # high-bias transfer sweeps all land within a few uA of each other), so
    # slide the whole stack back down if it has run off the top.
    overflow = placed[-1] - 1.0
    if overflow > 0:
        headroom = placed[0] - min(want[0], 0.0)
        placed = [y - min(overflow, max(headroom, 0.0)) for y in placed]

    for y, (_, text) in zip(placed, entries):
        ax.text(1.012, y, text, transform=ax.transAxes, ha="left",
                va="center", fontsize=7.6, color=MUTED, clip_on=False)


def draw_panel(ax, curves, sims, kind, present) -> None:
    cfg = PANELS[kind]
    labels = []
    for idx, curve in bias_curves(curves, kind):
        x = np.abs(cfg["x"](curve))
        order = np.argsort(x)
        x = x[order]
        meas = np.abs(np.asarray(curve.Id, dtype=float))[order] * 1e6
        every = max(1, len(x) // 16)

        ax.plot(x, meas, ls="none", marker="o", ms=4.0, markevery=every,
                mfc="none", mec=MEASURED_C, mew=1.0, zorder=5)
        for key, _lab, color, ls, marker in SERIES:
            if key not in present:
                continue
            y = np.abs(np.asarray(sims[key][idx], dtype=float))[order] * 1e6
            ax.plot(x, y, color=color, ls=ls, lw=1.5, zorder=4,
                    marker=marker, ms=3.4,
                    markevery=(every // 2 or 1, every),
                    mfc=SURFACE, mew=0.8)
        labels.append((meas[-1], f"{cfg['bias']}={abs(curve.fixed):.2f} V"))

    ax.set_title(cfg["title"], color=INK, fontsize=11, pad=6)
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("|I$_D$| (µA)")
    ax.margins(x=0.02)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.autoscale_view()
    place_end_labels(ax, labels)


def build_figure(dev, curves, sims, scores, present):
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4))
    fig.subplots_adjust(left=0.058, right=0.915, top=0.80, bottom=0.175,
                        wspace=0.42)
    for ax, kind in zip(axes, ("idvd", "idvg")):
        draw_panel(ax, curves, sims, kind, present)

    pol = "nMOS" if dev.dev_type == "nmos" else "pMOS"
    fig.suptitle(f"{pol}  L = {dev.L_um:g} µm,  W = {dev.W_um:g} µm  —  77 K",
                 x=0.058, ha="left", y=0.965, fontsize=15, color=INK)

    bits = [f"published {scores['paper']:.4f}"]
    bits += [f"{lab.split(' (')[0]} {scores[k]:.4f}"
             for k, lab, *_ in SERIES if k in present and k != "paper"]
    fig.text(0.058, 0.917,
             "frozen-inclusion RRMS — " + "  ·  ".join(bits),
             ha="left", fontsize=9.6, color=INK2)
    fig.text(0.058, 0.878,
             "every measured bias curve shown; colour = model, bias named at "
             "each curve's right end  ·  RRMS scored on all included curves  "
             "·  |V| and |I| plotted",
             ha="left", fontsize=8.6, color=MUTED)

    handles = [Line2D([], [], ls="none", marker="o", ms=7, mfc="none",
                      mec=MEASURED_C, mew=1.4, label="measured (77 K)")]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.9, marker=m, ms=5.5,
                       mfc=SURFACE, label=f"{lab} → NGSpice")
                for k, lab, c, ls, m in SERIES if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True, facecolor=SURFACE, edgecolor=AXIS,
               framealpha=1.0, handlelength=3.2, columnspacing=2.4,
               borderpad=0.7, fontsize=9.4, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.012))
    if "emu15" in present:
        fig.text(0.985, 0.017,
                 "15-parameter is a labeled experiment, not the canonical "
                 "protocol", ha="right", fontsize=7.6, color=MUTED)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emu7-dir", default=None,
                    help="7-parameter series dir (default: pdk_ml_emu, "
                         "falling back to pdk_ml_emu_resim)")
    ap.add_argument("--emu15-dir", default="out/pdk15_surrogate")
    ap.add_argument("--device", action="append",
                    help="device tag; repeatable. default: all 18")
    ap.add_argument("--out-dir", default=str(FIGS_DIR / "devices_detail"))
    ap.add_argument("--skip-missing", action="store_true",
                    help="plot without a series whose sims are absent")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    root = _HERE.parent
    if args.emu7_dir:
        emu7 = (root / args.emu7_dir).resolve()
    else:
        emu7 = OUT_DIR / "pdk_ml_emu"
        if not (emu7 / "sims_nmos_L0p15_W1p6.npz").exists():
            emu7 = OUT_DIR / "pdk_ml_emu_resim"
    emu15 = (root / args.emu15_dir).resolve()

    srcs = {"paper": (OUT_DIR / "pdk_baseline", None),
            "emu7": (emu7, "emu_search+fd"),
            "emu15": (emu15, "emu_search+fd")}
    present = []
    for key, (d, _) in srcs.items():
        if (d / "sims_nmos_L0p15_W1p6.npz").exists():
            present.append(key)
        elif args.skip_missing:
            print(f"warning: {key} series missing at {d} — omitted")
        else:
            raise SystemExit(
                f"missing {key} series at {d}.\n"
                "The 7-parameter sims are gitignored and absent on this "
                "checkout; rebuild them from the committed vectors with\n"
                "  PYTHONPATH=src .venv/bin/python "
                "scripts/resim_fixed_series.py\n"
                "or pass --skip-missing to plot without that line."
            )
    if "paper" not in present:
        raise SystemExit("the published-card baseline is required")

    wanted = set(args.device) if args.device else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for dev in PAPER_DEVICES:
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        if wanted and tag not in wanted:
            continue
        curves = load_device_curves(dev)
        n = len(curves)
        sims, scores = {}, {}
        for key in present:
            d, expect = srcs[key]
            if key == "paper":
                z = np.load(d / f"sims_{tag}.npz")
                sims[key] = [np.asarray(z[f"sim_{i}"]) for i in range(n)]
            else:
                sims[key], _ = load_series(dev, n, d, expect)
            scores[key] = score_device_new(
                dev.dev_type, dev.L_um, dev.W_um, curves, sims[key],
                include_tags=baseline_inclusion(tag))["rrms"]
        fig = build_figure(dev, curves, sims, scores, present)
        fig.savefig(out_dir / f"{tag}.png")
        if args.pdf:
            fig.savefig(out_dir / f"{tag}.pdf")
        plt.close(fig)
        written += 1
        print(f"  {tag:<22} " + "  ".join(
            f"{k} {scores[k]:.4f}" for k in present))

    if wanted and written != len(wanted):
        raise SystemExit(f"matched {written} of {len(wanted)} requested tags")
    print(f"\nwrote {written} figure(s) to "
          f"{out_dir.relative_to(root)}  [series: {', '.join(present)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
