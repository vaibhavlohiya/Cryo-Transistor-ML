"""Faceted I-V small multiples: measured vs published card vs one fixed method.

One panel per paper device, all 18 on a page, families grouped: the nMOS block
(2 rows x 4) sits above the pMOS block (2 rows x 5), so a panel's neighbours are
always its own polarity. Two pages are produced, one per curve kind, because
output (Id-Vd) and transfer (Id-Vg) families have different shapes and sharing a
panel between them hides both.

Readability rules this module follows, and why:

* **Three series, never more.** Measured points, the paper's published-card
  NGSpice curve, and exactly one fixed ML method. Every curve drawn is a real
  NGSpice re-simulation of a parameter vector -- never emulator output -- per
  the repository reporting policy.
* **Bias is encoded by position, not colour.** Within a panel the bias curves
  are physically stacked (higher |V| -> higher |I|), so spending the colour
  channel on bias would re-encode what the y-position already shows. Colour is
  therefore free to carry the thing the figure is actually about: which series
  a curve belongs to. Only ``--n-biases`` curves per panel are drawn (default
  3, evenly spread) because 11 x 3 lines per panel is unreadable at this size.
* **Composite encoding.** Series identity is carried by colour *and* dash
  pattern *and* marker shape simultaneously, so panels survive greyscale
  printing and colour-vision deficiency.
* **Per-panel autoscale.** Device currents span orders of magnitude (L=0.15 um
  against L=100 um), so a shared y-axis would flatten most panels into a line.
  Each panel scales to its own data and prints its own uA range; compare curve
  *shape* and series *separation* across panels, not absolute heights.
* **The legend lives outside the data.** It occupies its own full-width strip
  under the last panel row, so it can never overlap a curve.

Palette: measured ``#6f6c66``, published card ``#eb6834``, fixed method
``#2a78d6`` on surface ``#fcfcfb``. The two model hues are slots 1 and 2 of the
documented categorical palette. Validated all-pairs (the harder test, required
for small multiples, where any two panels can sit side by side): worst
normal-vision dE 17.8 (floor 15), worst CVD dE 10.9 under protanopia and
deuteranopia simulated with Machado-Oliveira-Fernandes 2009 at severity 1.0
(target 8). The measured grey is deliberately below the chroma floor: measured
data is the reference the models are judged against, not a competing identity,
and it is additionally distinguished by being the only series drawn as bare
markers with no connecting line. Preserve this reasoning if the colours change.

Usage::

    # default: published card vs the params15 surrogate+FD series
    PYTHONPATH=src .venv/bin/python scripts/plot_all_devices_iv.py

    # point at any other fixed series (e.g. the canonical 7-param emu_search+fd
    # once out/pdk_ml_emu has been regenerated)
    PYTHONPATH=src .venv/bin/python scripts/plot_all_devices_iv.py \\
        --series-dir out/pdk_ml_emu --series-label "surrogate + FD (7-param)" \\
        --expect-method emu_search+fd --out-stem iv_all18_emu_fd

No simulator is invoked: every curve is read from a committed/on-disk ``.npz``
of previously simulated currents.
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
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import FIGS_DIR, OUT_DIR  # noqa: E402
from cryoml.data_io import Curve, load_device_curves  # noqa: E402
from cryoml.devices import Device, PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

# ------------------------------------------------------------------ style
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#2b2a28", "#5c5a55", "#a3a099"
GRID, AXIS = "#eceae4", "#d5d3cb"

MEASURED_C = "#6f6c66"
PAPER_C = "#eb6834"
FIXED_C = "#2a78d6"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "figure.dpi": 200,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
})

KINDS = {
    "idvd": {
        "x": lambda c: c.Vd,
        "xlabel": "|V$_{DS}$| (V)",
        "bias": "V$_{GS}$",
        "title": "Output characteristics (I$_D$-V$_{DS}$)",
        "n_avail": 5,
    },
    "idvg": {
        "x": lambda c: c.Vg,
        "xlabel": "|V$_{GS}$| (V)",
        "bias": "V$_{DS}$",
        "title": "Transfer characteristics (I$_D$-V$_{GS}$)",
        "n_avail": 6,
    },
}


# ------------------------------------------------------------------ loading
def load_series(dev: Device, n_curves: int, series_dir: Path,
                expect_method: str | None) -> tuple[list[np.ndarray], dict]:
    """Load one fixed series' simulated currents, with an identity guard.

    Fails loudly rather than substituting another method: a quietly mislabelled
    series is exactly the failure the reporting policy exists to prevent.
    """
    tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
    npz = series_dir / f"sims_{tag}.npz"
    rec_path = series_dir / f"ml_{tag}.json"
    if not npz.exists():
        raise SystemExit(
            f"missing {npz}.\nThis series' per-device sims are not on disk. "
            "They are gitignored and are regenerated by the extraction "
            "stages; see docs/HANDOFF.md before launching one."
        )
    record = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    got = record.get("best_method") or record.get("method")
    if expect_method and got and got != expect_method:
        raise SystemExit(
            f"{tag}: expected fixed method {expect_method!r}, got {got!r}. "
            "Refusing to plot a different method under this label."
        )
    saved = np.load(npz)
    return [np.asarray(saved[f"sim_{i}"]) for i in range(n_curves)], record


def baseline_inclusion(tag: str, _cache={}) -> set[str]:
    """The published card's included-curve set -- the frozen scoring basis."""
    if not _cache:
        _cache["blob"] = json.load(
            open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))
    return {name for name, item in _cache["blob"]["per_curve"][tag].items()
            if item.get("included")}


def select_biases(curves: list[Curve], kind: str, n: int,
                  drop_weakest: bool = True) -> list[tuple[int, Curve]]:
    """Evenly spread bias selection over the *sorted* bias ladder.

    ``load_device_curves`` returns curves in file order, not bias order (a
    device's idvd set can arrive as 1.11, 1.48, 0.37, 1.85, 0.74), so sorting
    by |bias| first is what makes "evenly spread" and "weakest" mean anything.

    The weakest bias of each kind is dropped by default: the lowest V_GS
    output curve is near device-off, and the 0.01 V transfer curve sits deep
    in the linear region, so once a panel is autoscaled to its strongest curve
    both are a flat line on the axis. They still carry their full weight in
    the RRMS printed on the panel, which is scored on the frozen inclusion set
    -- every included curve -- not on the subset drawn here.
    """
    pairs = [(i, c) for i, c in enumerate(curves) if c.kind == kind]
    if not pairs:
        return []
    pairs.sort(key=lambda p: abs(p[1].fixed))
    pool = pairs[1:] if (drop_weakest and len(pairs) > n) else pairs
    if len(pool) <= n:
        return pool
    sel = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return [pool[i] for i in sel]


# ------------------------------------------------------------------ drawing
def place_note(ax, pts: np.ndarray, text: str) -> None:
    """Put an annotation in whichever corner holds the fewest plotted points.

    A fixed corner cannot work across 18 autoscaled panels: output curves that
    saturate within a fraction of a volt fill the upper left, while transfer
    curves leave it empty. Choosing per panel is the only way the label never
    lands on a curve.
    """
    if len(pts) == 0:
        return
    frac = ax.transLimits.transform(pts)
    corners = {
        ("left", "top"): (0.02, 0.97, frac[:, 0] < 0.50, frac[:, 1] > 0.72),
        ("right", "top"): (0.98, 0.97, frac[:, 0] > 0.50, frac[:, 1] > 0.72),
        ("left", "bottom"): (0.02, 0.03, frac[:, 0] < 0.50, frac[:, 1] < 0.28),
        ("right", "bottom"): (0.98, 0.03, frac[:, 0] > 0.50,
                              frac[:, 1] < 0.28),
    }
    (ha, va), (x, y, _, _) = min(
        corners.items(), key=lambda kv: int((kv[1][2] & kv[1][3]).sum()))
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=7.2, color=INK2, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc=SURFACE, ec=AXIS, lw=0.6))


def draw_panel(ax, dev: Device, curves, sims: dict, kind: str, n_biases: int,
               scores: dict, series_label: str,
               drop_weakest: bool = True) -> None:
    cfg = KINDS[kind]
    pairs = select_biases(curves, kind, n_biases, drop_weakest)
    if not pairs:
        ax.set_visible(False)
        return

    drawn: list[np.ndarray] = []
    for idx, curve in pairs:
        x = np.abs(cfg["x"](curve))
        order = np.argsort(x)
        x = x[order]
        meas = np.abs(np.asarray(curve.Id, dtype=float))[order] * 1e6

        # Measured: bare markers, no connecting line -- it is the reference,
        # and points read as data while lines read as model.
        every = max(1, len(x) // 13)
        ax.plot(x, meas, ls="none", marker="o", ms=3.4, markevery=every,
                mfc="none", mec=MEASURED_C, mew=0.9, zorder=3)
        drawn.append(np.column_stack([x, meas]))

        for key, color, ls, marker in (
            ("paper", PAPER_C, (0, (5, 2.2)), "s"),
            ("fixed", FIXED_C, "-", "^"),
        ):
            y = np.abs(np.asarray(sims[key][idx], dtype=float))[order] * 1e6
            ax.plot(x, y, color=color, ls=ls, lw=1.35, zorder=4,
                    marker=marker, ms=3.0, markevery=(every // 2 or 1, every),
                    mfc=SURFACE, mew=0.8)
            drawn.append(np.column_stack([x, y]))

    pol = "nMOS" if dev.dev_type == "nmos" else "pMOS"
    ax.set_title(f"{pol}  L={dev.L_um:g} µm, W={dev.W_um:g} µm",
                 color=INK, fontsize=8.8, pad=4)
    ax.set_xlabel(cfg["xlabel"], labelpad=1.5)
    ax.set_ylabel("|I$_D$| (µA)", labelpad=1.5)
    ax.margins(x=0.02)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Selective direct label: the one number this panel exists to move.
    ax.autoscale_view()
    place_note(ax, np.vstack(drawn) if drawn else np.empty((0, 2)),
               f"RRMS {scores['paper']:.3f} → {scores['fixed']:.3f}")


def build_figure(kind: str, devs_data: list, series_label: str,
                 n_biases: int, subtitle: str,
                 drop_weakest: bool = True) -> plt.Figure:
    cfg = KINDS[kind]
    nmos = [d for d in devs_data if d[0].dev_type == "nmos"]
    pmos = [d for d in devs_data if d[0].dev_type == "pmos"]

    # 20 columns is lcm(4, 5): the nMOS block runs 4 panels per row (span 5)
    # and the pMOS block 5 per row (span 4), so both blocks stay flush.
    fig = plt.figure(figsize=(17.5, 15.2))
    gs = GridSpec(5, 20, figure=fig,
                  height_ratios=[1, 1, 1, 1, 0.20],
                  hspace=0.44, wspace=1.9,
                  left=0.045, right=0.985, top=0.908, bottom=0.045)

    for n, (dev, curves, sims, scores) in enumerate(nmos):
        r, c = divmod(n, 4)
        ax = fig.add_subplot(gs[r, c * 5:(c + 1) * 5])
        draw_panel(ax, dev, curves, sims, kind, n_biases, scores,
                   series_label, drop_weakest)

    for n, (dev, curves, sims, scores) in enumerate(pmos):
        r, c = divmod(n, 5)
        ax = fig.add_subplot(gs[2 + r, c * 4:(c + 1) * 4])
        draw_panel(ax, dev, curves, sims, kind, n_biases, scores,
                   series_label, drop_weakest)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6, mfc="none",
               mec=MEASURED_C, mew=1.3, label="measured (77 K)"),
        Line2D([], [], color=PAPER_C, ls=(0, (5, 2.2)), lw=1.7, marker="s",
               ms=5, mfc=SURFACE, label="published card → NGSpice"),
        Line2D([], [], color=FIXED_C, ls="-", lw=1.7, marker="^", ms=5,
               mfc=SURFACE, label=f"{series_label} → NGSpice"),
    ]
    lax = fig.add_subplot(gs[4, :])
    lax.axis("off")
    lax.legend(handles=handles, loc="center", ncol=3, frameon=True,
               facecolor=SURFACE, edgecolor=AXIS, framealpha=1.0,
               handlelength=3.4, columnspacing=3.0, borderpad=0.8,
               fontsize=9.5, labelcolor=INK)

    fig.suptitle(f"{cfg['title']} — all 18 paper devices, 77 K",
                 x=0.045, ha="left", y=0.982, fontsize=15, color=INK)
    fig.text(0.045, 0.958, subtitle, ha="left", fontsize=9.2, color=INK2)
    fig.text(0.045, 0.940,
             f"{n_biases} of {cfg['n_avail']} {cfg['bias']} biases drawn per "
             f"panel{', weakest omitted' if drop_weakest else ''}, increasing "
             "upward — but RRMS is scored on every included curve, not just "
             "those drawn  ·  each panel autoscaled to its own current range "
             " ·  |V| and |I| plotted",
             ha="left", fontsize=8.6, color=MUTED)
    return fig


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--series-dir", default="out/pdk15_surrogate",
                    help="directory of the fixed method's sims_<tag>.npz")
    ap.add_argument("--series-label", default="params15 surrogate + FD",
                    help="legend label for the fixed method")
    ap.add_argument("--expect-method", default="emu_search+fd",
                    help="method id the series records must report")
    ap.add_argument("--out-stem", default="iv_all18_params15_fd")
    ap.add_argument("--kind", choices=["idvd", "idvg", "both"], default="both")
    ap.add_argument("--n-biases", type=int, default=3,
                    help="bias curves drawn per panel (default 3)")
    ap.add_argument("--keep-weakest", action="store_true",
                    help="also draw the weakest bias of each kind, which "
                         "is near device-off and reads as a flat line")
    ap.add_argument("--pdf", action="store_true", help="also write vector PDF")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    series_dir = (root / args.series_dir).resolve()
    paper_dir = OUT_DIR / "pdk_baseline"
    if not paper_dir.exists():
        raise SystemExit(f"missing published-card baseline at {paper_dir}")

    devs_data, param_sets = [], set()
    for dev in PAPER_DEVICES:
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        curves = load_device_curves(dev)
        n = len(curves)
        paper_sims = [np.asarray(np.load(paper_dir / f"sims_{tag}.npz")[
            f"sim_{i}"]) for i in range(n)]
        fixed_sims, record = load_series(dev, n, series_dir,
                                         args.expect_method)
        param_sets.add(record.get("param_set", "params7"))
        include = baseline_inclusion(tag)
        sims = {"paper": paper_sims, "fixed": fixed_sims}
        scores = {k: score_device_new(dev.dev_type, dev.L_um, dev.W_um,
                                      curves, s, include_tags=include)["rrms"]
                  for k, s in sims.items()}
        devs_data.append((dev, curves, sims, scores))

    if len(devs_data) != 18:
        raise SystemExit(f"expected 18 devices, built {len(devs_data)}")

    paper_mean = float(np.mean([d[3]["paper"] for d in devs_data]))
    fixed_mean = float(np.mean([d[3]["fixed"] for d in devs_data]))
    wins = sum(d[3]["fixed"] < d[3]["paper"] for d in devs_data)
    pset = "/".join(sorted(param_sets))
    subtitle = (f"fixed method: {args.series_label} ({pset}), "
                f"frozen-inclusion all-device RRMS {paper_mean:.4f} → "
                f"{fixed_mean:.4f}, {wins}/18 devices improved")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    kinds = ["idvd", "idvg"] if args.kind == "both" else [args.kind]
    written = []
    for kind in kinds:
        fig = build_figure(kind, devs_data, args.series_label,
                           args.n_biases, subtitle,
                           not args.keep_weakest)
        stem = f"{args.out_stem}_{kind}"
        png = FIGS_DIR / f"{stem}.png"
        fig.savefig(png)
        written.append(png)
        if args.pdf:
            pdf = FIGS_DIR / f"{stem}.pdf"
            fig.savefig(pdf)
            written.append(pdf)
        plt.close(fig)

    print(subtitle)
    for p in written:
        print(f"wrote {p.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
