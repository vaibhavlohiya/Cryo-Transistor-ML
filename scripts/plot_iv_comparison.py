#!/usr/bin/env python3
"""Qualitative I-V reconstruction comparison: measured data vs N methods.

The core is :func:`plot_iv_blocks`, which is repo-independent — it takes plain
numpy arrays, so it works with any measured/predicted I-V data. Each *block*
is one panel (a device, a bias condition, a parameter set — whatever you want
compared side by side) holding one measured curve and one predicted curve per
method.

Because two methods can easily agree to within a line width on a linear I-V
plot, every panel carries a residual strip underneath showing the error
normalised by the curve's mean |I| — the same normalisation the project's RRMS
uses, so the strip is a per-point decomposition of the headline number.

Colour, dash pattern AND marker all encode the method, so the panels survive
greyscale printing and colour-vision deficiency. The four line colours pass the
all-pairs CVD/contrast checks on a light surface (worst CVD dE 9.2, worst
normal-vision dE 16.3); no four-hue set clears them on a dark surface, so this
figure is light-mode only by design.

    # synthetic self-test — no repo data, no simulator needed
    python scripts/plot_iv_comparison.py --demo

    # real data: re-simulates each method's parameters in NGSpice (heavy)
    python scripts/plot_iv_comparison.py --from-repo
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"

# --- style -----------------------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MEASURED = "#0b0b0b"

# colour + dash + marker per method slot; all three carry identity.
METHOD_STYLES = [
    {"color": "#2a78d6", "ls": (0, (6, 2)), "marker": "s"},
    {"color": "#eb6834", "ls": (0, (1.6, 1.6)), "marker": "^"},
    {"color": "#1baf7a", "ls": (0, (7, 2, 1.5, 2)), "marker": "D"},
    {"color": "#4a3aa7", "ls": (0, (3, 1.4, 1, 1.4, 1, 1.4)), "marker": "v"},
]


@dataclass
class IVBlock:
    """One panel: a measured curve plus one predicted curve per method.

    ``x`` is the swept voltage; ``measured`` and every array in
    ``predictions`` must share its shape. Values may be signed (pMOS currents
    are negative) — the plot takes magnitudes.
    """

    title: str
    x: np.ndarray
    measured: np.ndarray
    predictions: dict[str, np.ndarray]
    x_label: str = "$V_{DS}$ (V)"
    y_label: str = "$|I_D|$ (µA)"
    note: str = ""                      # e.g. "V_GS = 1.85 V"
    metrics: dict[str, float] = field(default_factory=dict)   # method -> RRMS

    def __post_init__(self) -> None:
        self.x = np.abs(np.asarray(self.x, dtype=np.float64))
        self.measured = np.asarray(self.measured, dtype=np.float64)
        n = self.x.size
        if self.measured.size != n:
            raise ValueError(f"{self.title}: measured has {self.measured.size} "
                             f"points, x has {n}")
        for name, pred in self.predictions.items():
            pred = np.asarray(pred, dtype=np.float64)
            if pred.size != n:
                raise ValueError(f"{self.title}/{name}: {pred.size} points, "
                                 f"x has {n}")
            self.predictions[name] = pred


def _apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 0.9,
        "axes.grid": True, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "figure.dpi": 120, "legend.frameon": False,
    })


def plot_iv_blocks(
    blocks: list[IVBlock],
    *,
    ncols: int = 2,
    residuals: bool = True,
    log_y: bool = False,
    scale: float = 1e6,
    marker_every: int = 6,
    metrics_loc: str = "lower right",
    suptitle: str = "",
    subtitle: str = "",
    figsize: tuple[float, float] | None = None,
):
    """Render one panel per block; returns the Matplotlib Figure.

    scale        multiplies currents for display (1e6 -> µA, 1e3 -> mA, 1 -> A)
    residuals    add the normalised-error strip beneath each panel
    log_y        log current axis (use for subthreshold/transfer curves)
    marker_every plot a method marker every Nth point, so lines stay readable
    """
    _apply_style()
    if not blocks:
        raise ValueError("no blocks to plot")
    methods = list(blocks[0].predictions)
    if len(methods) > len(METHOD_STYLES):
        raise ValueError(f"{len(methods)} methods exceeds the "
                         f"{len(METHOD_STYLES)} validated style slots; facet "
                         f"into separate figures instead of adding hues")
    style = dict(zip(methods, METHOD_STYLES))

    nrows = int(np.ceil(len(blocks) / ncols))
    figsize = figsize or (6.4 * ncols, (4.6 if residuals else 3.7) * nrows)
    fig = plt.figure(figsize=figsize)
    # Outer grid separates blocks; the inner pair (curve + residual strip)
    # stays tightly coupled, so a strip's x-label never lands on the next
    # block's title.
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.34, wspace=0.24,
                  top=0.885 if suptitle else 0.965, bottom=0.085,
                  left=0.072, right=0.985)

    for i, block in enumerate(blocks):
        r, c = divmod(i, ncols)
        if residuals:
            inner = gs[r, c].subgridspec(2, 1, height_ratios=[3.1, 1.0],
                                         hspace=0.07)
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1], sharex=ax)
        else:
            ax, axr = fig.add_subplot(gs[r, c]), None

        meas = np.abs(block.measured) * scale
        denom = np.mean(np.abs(block.measured))          # RRMS normalisation
        if denom <= 0 or not np.isfinite(denom):
            denom = np.nan

        # ground truth: solid faint spine + open circles, drawn on top
        ax.plot(block.x, meas, "-", color=MEASURED, lw=1.0, alpha=0.35,
                zorder=5)
        ax.plot(block.x, meas, "o", mfc="none", mec=MEASURED, mew=1.1,
                ms=5.2, ls="none", zorder=6, label="Measured")

        for name, pred in block.predictions.items():
            st = style[name]
            ax.plot(block.x, np.abs(pred) * scale, color=st["color"],
                    ls=st["ls"], lw=1.9, marker=st["marker"], ms=4.6,
                    markevery=marker_every, markerfacecolor=SURFACE,
                    markeredgewidth=1.1, label=name, zorder=4)
            if axr is not None:
                axr.plot(block.x, (pred - block.measured) / denom * 100,
                         color=st["color"], ls=st["ls"], lw=1.6,
                         marker=st["marker"], ms=4.0, markevery=marker_every,
                         markerfacecolor=SURFACE, markeredgewidth=1.0)

        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(block.y_label, fontsize=10)
        title = block.title + (f"      {block.note}" if block.note else "")
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=7,
                     fontweight="bold")
        if block.metrics:
            # Default lower right: the quadrant a saturating output curve
            # leaves empty. Move it for log-scale transfer curves, which
            # occupy that corner instead.
            va, ha = metrics_loc.split()
            ax.text(0.985 if ha == "right" else 0.025,
                    0.03 if va == "lower" else 0.975,
                    "\n".join(f"{k}:  {v:.3f}"
                              for k, v in block.metrics.items()),
                    transform=ax.transAxes, ha=ha,
                    va="bottom" if va == "lower" else "top",
                    fontsize=7.8, color=MUTED, linespacing=1.5)
        if axr is None:
            ax.set_xlabel(block.x_label, fontsize=10)
        else:
            ax.tick_params(labelbottom=False)
            axr.axhline(0, color=AXIS, lw=1.0, zorder=2)
            axr.set_xlabel(block.x_label, fontsize=10)
            axr.set_ylabel("error\n(% of mean $|I|$)", fontsize=8.5)
            axr.tick_params(labelsize=8.5)
            lim = np.nanmax([np.nanmax(np.abs((p - block.measured) / denom))
                             for p in block.predictions.values()]) * 100
            lim = max(float(lim) * 1.25, 1.0)
            axr.set_ylim(-lim, lim)

    handles = [Line2D([], [], color=MEASURED, marker="o", mfc="none", mew=1.1,
                      ms=6, ls="-", alpha=0.6, label="Measured (ground truth)")]
    handles += [Line2D([], [], color=style[m]["color"], ls=style[m]["ls"],
                       marker=style[m]["marker"], markerfacecolor=SURFACE,
                       lw=1.9, ms=5.4, label=m) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.004), handlelength=2.8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14.5, color=INK, x=0.011, ha="left",
                     y=0.988, fontweight="bold")
    if subtitle:
        fig.text(0.011, 0.955, subtitle, fontsize=9.5, color=INK2, ha="left",
                 va="top")
    return fig


# --- synthetic self-test ----------------------------------------------------
def demo_blocks() -> list[IVBlock]:
    """Physically-shaped output curves with per-method parameter error.

    Doubles as the worked example of the input format: build the arrays any
    way you like, hand them over as numpy.
    """
    rng = np.random.default_rng(0)
    geoms = [("nMOS  L=0.15 µm, W=1.6 µm", 1.8, 1.0),
             ("nMOS  L=1.0 µm, W=3.0 µm", 1.2, 0.55),
             ("pMOS  L=0.35 µm, W=1.6 µm", 1.5, -0.62),
             ("pMOS  L=2.0 µm, W=5.0 µm", 0.9, -0.38)]
    # multiplicative (vth, mobility, channel-length-modulation) error per method
    method_err = {
        "Direct MLP": (1.045, 0.95, 1.30),
        "7-param surrogate + FD": (1.015, 0.99, 1.12),
        "15-param surrogate + FD": (1.004, 0.998, 1.03),
        "Foundation emulator + FD": (1.020, 1.010, 0.88),
    }

    def ideal(vds, vgs, vth, k, lam):
        vov = max(vgs - vth, 1e-9)
        vdsat = vov
        lin = k * (vov * vds - 0.5 * vds ** 2)
        sat = k * 0.5 * vov ** 2 * (1 + lam * (vds - vdsat))
        return np.where(vds < vdsat, lin, sat)

    blocks = []
    for title, gain, vth in geoms:
        vds = np.linspace(0.02, 1.85, 60)
        vgs = 1.85 * np.sign(vth)
        truth = ideal(vds, abs(vgs), abs(vth), gain * 1e-4, 0.09)
        truth = truth * (1 + rng.normal(0, 0.006, vds.size))   # measurement noise
        preds, metrics = {}, {}
        for name, (dvth, dmu, dlam) in method_err.items():
            p = ideal(vds, abs(vgs), abs(vth) * dvth, gain * 1e-4 * dmu,
                      0.09 * dlam)
            preds[name] = p
            metrics[name] = float(np.sqrt(np.mean((p - truth) ** 2))
                                  / np.mean(np.abs(truth)))
        blocks.append(IVBlock(title=title, x=vds, measured=truth,
                              predictions=preds, note=f"$V_{{GS}}$ = {vgs:+.2f} V",
                              metrics=metrics))
    return blocks


# --- real repo data ---------------------------------------------------------
def repo_blocks(device_tags: list[str], curve_kind: str = "idvd"):
    """Measured curves + a real NGSpice re-simulation per method.

    Never plots emulator output: each method's *parameter vector* is written
    into the 77 K card and simulated, which is the only thing the project
    permits in an I-V figure.
    """
    import json
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cryoml.data_io import load_device_curves          # noqa: E402
    from cryoml.devices import PAPER_DEVICES               # noqa: E402
    from cryoml.metrics import tag_metric_curves           # noqa: E402
    from cryoml.spice_pdk import simulate_pdk              # noqa: E402
    from cryoml.utils import device_tag                    # noqa: E402

    p15_dir = ROOT / "out" / "pdk15_surrogate"
    blocks = []
    for tag in device_tags:
        dev = next(d for d in PAPER_DEVICES
                   if device_tag(d.dev_type, d.L_um, d.W_um) == tag)
        rec = json.loads((p15_dir / f"ml_{tag}.json").read_text())
        bin_index = int(rec["bin_index"])
        curves = load_device_curves(dev)

        # strongest bias of the requested family — where methods differ most
        tagged = tag_metric_curves(curves)
        picks = sorted([(bias, c) for (kind, bias), c in tagged.items()
                        if kind == curve_kind], key=lambda t: abs(t[0]))
        bias, curve = picks[-1]
        idx = next(i for i, c in enumerate(curves) if c is curve)

        wanted = {"Published card": rec["params_by_method"]["published"],
                  "15-param surrogate, raw":
                      rec["params_by_method"]["emu_search"],
                  "15-param surrogate + FD":
                      rec["params_by_method"]["emu_search+fd"]}
        preds = {}
        for name, params in wanted.items():
            sims = simulate_pdk(dev.dev_type, dev.L_um, dev.W_um, curves,
                                params=params, bin_index=bin_index)
            preds[name] = np.asarray(sims[idx])[:len(curve.Id)]

        x = curve.Vd if curve_kind == "idvd" else curve.Vg
        fixed = "V_{GS}" if curve_kind == "idvd" else "V_{DS}"
        blocks.append(IVBlock(
            title=f"{dev.dev_type.replace('mos', 'MOS')}  "
                  f"L={dev.L_um:g} µm, W={dev.W_um:g} µm",
            x=np.asarray(x)[:len(curve.Id)], measured=np.asarray(curve.Id),
            predictions=preds, note=f"${fixed}$ = {bias:+.2f} V",
            x_label="$|V_{DS}|$ (V)" if curve_kind == "idvd"
                    else "$|V_{GS}|$ (V)"))
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="synthetic self-test; needs no repo data or simulator")
    ap.add_argument("--from-repo", action="store_true",
                    help="real measured data + NGSpice re-simulation (heavy: "
                         "do not run beside another NGSpice job)")
    ap.add_argument("--devices",
                    default="nmos_L0p15_W1p6,nmos_L1_W3,pmos_L0p35_W1p6,"
                            "pmos_L2_W5")
    ap.add_argument("--kind", default="idvd", choices=("idvd", "idvg"))
    ap.add_argument("--log-y", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.from_repo:
        blocks = repo_blocks(args.devices.split(","), args.kind)
        stem = args.out or f"iv_method_comparison_{args.kind}"
        sub = ("Measured points against real NGSpice re-simulations of each "
               "method's extracted parameters — never emulator output.\n"
               "Lower strip: per-point error normalised by the curve's mean "
               "|I|, the same normalisation the RRMS metric uses.")
    else:
        blocks = demo_blocks()
        stem = args.out or "iv_method_comparison_demo"
        sub = ("SYNTHETIC self-test data — verifies the plotting path and "
               "shows the expected input format.\n"
               "Lower strip: per-point error normalised by the curve's mean "
               "|I|.")

    fig = plot_iv_blocks(
        blocks, ncols=2, residuals=True, log_y=args.log_y,
        suptitle="I-V reconstruction quality by extraction method",
        subtitle=sub)
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIGS / f"{stem}.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=SURFACE)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
