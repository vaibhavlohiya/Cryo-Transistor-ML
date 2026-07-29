"""BSIM4 parameter extraction against the confirmed NGSpice-41 chain.

Theta layout is ``ACTIVE_PARAMS`` (order fixed), selected by the
``CRYOML_PARAM_SET`` environment variable: the canonical 7-parameter set
``vth0, u0, nfactor, vsat, delta, rdsw, eta0`` (default, all published
series), or the experimental 15-parameter superset ``params15``.

The current pipeline uses a linear +/-10% box around each bin's published
effective parameter values, matching the confirmed upstream's Latin-hypercube
multiplier perturbation. The legacy broad transform remains available only for
historical experiments.

The primary objective is the port of the confirmed upstream's ``rrmsCalc.py``
with measured-current cleaning/trimming and curve inclusion frozen to the
published-card baseline. The older all-curve objective remains for continuity.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .data_io import Curve
from .metrics import (curve_trim, device_rrms, score_device_new,
                      tag_metric_curves)
from .spice_pdk import (LIB_FILE, PDK77K_DIR, _CORNER_BASENAME, _MODEL,
                        _NGSPICE_BIN, _SUBCKT, _force_bin, _instance_line,
                        ensure_pdk77k, simulate_pdk)
from .utils import get_logger

logger = get_logger("cryoml.pdk_extract")

PARAMS7 = ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0")
# Extended experimental set: the paper's 7 plus output-conductance
# (pclm, pdiblc1, pdiblc2), saturation-knee shape (ags), vertical-field
# mobility shape (ua, ub), subthreshold offset (voff), and gate-bias-
# dependent S/D resistance (prwg). Zero-published parameters are frozen by
# the ±10% box: pdiblc1 is active only on nMOS bins, prwg only on pMOS.
PARAMS15 = PARAMS7 + ("pclm", "pdiblc1", "pdiblc2", "ags",
                      "ua", "ub", "voff", "prwg")
PARAM_SETS = {"params7": PARAMS7, "params15": PARAMS15}

# The active tuned-parameter set is resolved once at import from
# CRYOML_PARAM_SET so that multiprocessing spawn workers (which re-import
# this module in a fresh interpreter) agree with the parent process. Unset
# means the canonical 7-parameter protocol; every published series uses it.
ACTIVE_PARAM_SET = os.environ.get("CRYOML_PARAM_SET", "params7")
if ACTIVE_PARAM_SET not in PARAM_SETS:
    raise RuntimeError(f"CRYOML_PARAM_SET={ACTIVE_PARAM_SET!r} unknown; "
                       f"expected one of {sorted(PARAM_SETS)}")
ACTIVE_PARAMS = PARAM_SETS[ACTIVE_PARAM_SET]

PENALTY = 10.0


@dataclass
class FlatCurves:
    slices: list[tuple[int, int]]
    kept: list[int]
    denominators: np.ndarray
    n_total_curves: int


def flatten_paper_curves(curves: list[Curve]) -> FlatCurves:
    """Build the paper-exact optimization layout.

    All-zero curves have RRMS 0.0 in the companion notebook, independent of
    simulation output, so they contribute no optimization residual.
    """
    slices, kept, denominators = [], [], []
    cur = 0
    for i, curve in enumerate(curves):
        meas = np.asarray(curve.Id, dtype=np.float64)
        n = len(meas)
        denominator = float(np.mean(np.abs(meas))) if n else 0.0
        if n == 0 or denominator <= 0 or not np.isfinite(denominator):
            continue
        slices.append((cur, cur + n))
        kept.append(i)
        denominators.append(denominator)
        cur += n
    return FlatCurves(
        slices=slices,
        kept=kept,
        denominators=np.asarray(denominators, dtype=np.float64),
        n_total_curves=len(curves),
    )


# ---------------------------------------------------------------------------
# Published per-bin parameter readback (authoritative, via showmod)
# ---------------------------------------------------------------------------
def read_bin_params(dev_type: str, L_um: float, W_um: float,
                    bin_index: int) -> dict[str, float]:
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    base = (PDK77K_DIR / _CORNER_BASENAME[dev_type]).read_text(errors="replace")
    text = _force_bin(base, dev_type, bin_index)
    with tempfile.TemporaryDirectory(prefix="cryoml_showmod_") as td_s:
        td = Path(td_s)
        cp = td / _CORNER_BASENAME[dev_type]
        cp.write_text(text)
        lib = LIB_FILE.read_text().replace(
            (PDK77K_DIR / _CORNER_BASENAME[dev_type]).as_posix(), cp.as_posix())
        lp = td / "lib.spice"
        lp.write_text(lib)
        sign = -1 if dev_type == "pmos" else 1
        deck = td / "d.sp"
        deck.write_text(
            f'* showmod readback\n.options scale=1.0\n.lib "{lp.as_posix()}" tt_77k\n'
            ".options temp=-196.15\n"
            + _instance_line(dev_type, L_um, W_um) +
            f"VG ng 0 DC {1.0 * sign}\nVD nd 0 DC {1.0 * sign}\n.op\n"
            ".control\nrun\n"
            f"showmod m.xm1.m{_SUBCKT[dev_type]} : {' '.join(ACTIVE_PARAMS)}\n"
            "quit\n.endc\n.end\n")
        out = subprocess.run([_NGSPICE_BIN, "-b", str(deck)],
                             capture_output=True, text=True, timeout=30).stdout
    vals: dict[str, float] = {}
    for p in ACTIVE_PARAMS:
        m = re.search(rf"^\s*{p}\s+([0-9eE.+\-]+)\s*$", out, re.MULTILINE)
        if m:
            vals[p] = float(m.group(1))
    missing = [p for p in ACTIVE_PARAMS if p not in vals]
    if missing:
        raise RuntimeError(f"showmod readback missing {missing} for "
                           f"{dev_type} bin {bin_index}: {out[-500:]}")
    return vals


# ---------------------------------------------------------------------------
# Bounded theta box around the published bin values
# ---------------------------------------------------------------------------
@dataclass
class ThetaBox:
    """Sigmoid-bounded z <-> physical transform, per device bin."""
    dev_type: str
    bin_index: int
    published: dict[str, float]
    lo_t: np.ndarray = field(init=False)   # transform-space lower bounds
    hi_t: np.ndarray = field(init=False)
    is_log: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # absolute fallback boxes for log params whose published value is
        # zero/negative (some bins ship e.g. vsat=-2.6e4 or delta=-0.012;
        # ngspice clamps them internally, but log bounds need positives)
        fallback = {
            "u0": (1e-3, 0.5),
            "nfactor": (0.5, 120.0),
            "vsat": (1e4, 5e9),
            "delta": (1e-4, 0.5),
            "rdsw": (0.05, 1e5),
        }
        lo, hi, isl = [], [], []
        for p in PARAMS7:
            v = float(self.published[p])
            if p == "vth0":
                lo.append(v - 0.6); hi.append(v + 0.6); isl.append(False)
                continue
            if p == "eta0":
                span = max(0.3, 2.0 * abs(v))
                lo.append(v - span); hi.append(v + span); isl.append(False)
                continue
            if v <= 0 or not np.isfinite(v):
                l, h = fallback[p]
            elif p == "vsat":
                l = min(0.2 * v, 3.0e4); h = max(5.0 * v, 3.0e6)
            elif p == "rdsw":
                l = max(0.05, 0.02 * v); h = max(20.0 * v, 100.0)
            elif p == "delta":
                l = 0.05 * v; h = 40.0 * v
            else:  # u0, nfactor: positive, log x[0.1, 10]
                l = 0.1 * v; h = 10.0 * v
            lo.append(np.log(l)); hi.append(np.log(h)); isl.append(True)
        self.lo_t = np.array(lo, dtype=np.float64)
        self.hi_t = np.array(hi, dtype=np.float64)
        self.is_log = np.array(isl, dtype=bool)

    def z_to_params(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=np.float64)
        s = 1.0 / (1.0 + np.exp(-z))
        t = self.lo_t + (self.hi_t - self.lo_t) * s
        x = np.where(self.is_log, np.exp(t), t)
        return {p: float(x[i]) for i, p in enumerate(PARAMS7)}

    def params_to_z(self, params: dict[str, float]) -> np.ndarray:
        x = np.array([float(params[p]) for p in PARAMS7], dtype=np.float64)
        t = np.where(self.is_log, np.log(np.maximum(x, 1e-300)), x)
        frac = (t - self.lo_t) / (self.hi_t - self.lo_t)
        frac = np.clip(frac, 1e-6, 1 - 1e-6)
        return np.clip(np.log(frac / (1 - frac)), -5.5, 5.5)

    @property
    def z_published(self) -> np.ndarray:
        return self.params_to_z(self.published)


def theta_box_for(dev_type: str, L_um: float, W_um: float,
                  bin_index: int) -> ThetaBox:
    pub = read_bin_params(dev_type, L_um, W_um, bin_index)
    return ThetaBox(dev_type=dev_type, bin_index=bin_index, published=pub)


@dataclass
class LhcBox:
    """±10 % box around the published bin values — the confirmed-setup
    Monte-Carlo box (nomSweep_latinHypercube.py). Sigmoid z <-> physical,
    linear per param (the range is too narrow to need log spacing), sign-safe
    for negative published values. z=0 is exactly the published card."""
    dev_type: str
    bin_index: int
    published: dict[str, float]
    frac: float = 0.10
    lo: np.ndarray = field(init=False)
    hi: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        v = np.array([float(self.published[p]) for p in ACTIVE_PARAMS],
                     dtype=np.float64)
        b1, b2 = (1.0 - self.frac) * v, (1.0 + self.frac) * v
        self.lo = np.minimum(b1, b2)
        self.hi = np.maximum(b1, b2)
        degenerate = self.hi - self.lo < 1e-30
        self.lo[degenerate] -= 1e-30
        self.hi[degenerate] += 1e-30

    def z_to_params(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=np.float64)
        s = 1.0 / (1.0 + np.exp(-z))
        x = self.lo + (self.hi - self.lo) * s
        return {p: float(x[i]) for i, p in enumerate(ACTIVE_PARAMS)}

    def params_to_z(self, params: dict[str, float]) -> np.ndarray:
        x = np.array([float(params[p]) for p in ACTIVE_PARAMS],
                     dtype=np.float64)
        frac = (x - self.lo) / (self.hi - self.lo)
        frac = np.clip(frac, 1e-6, 1 - 1e-6)
        return np.clip(np.log(frac / (1 - frac)), -13.9, 13.9)

    @property
    def z_published(self) -> np.ndarray:
        return self.params_to_z(self.published)


def make_box(dev_type: str, L_um: float, W_um: float, bin_index: int,
             mode: str = "lhc10"):
    """Box factory: "lhc10" = confirmed-setup ±10 % LHC box (default),
    "wide" = the legacy broad sigmoid box."""
    if mode == "wide":
        if ACTIVE_PARAMS != PARAMS7:
            raise ValueError("the legacy wide box supports only the "
                             "canonical 7-parameter set")
        return theta_box_for(dev_type, L_um, W_um, bin_index)
    if mode == "lhc10":
        pub = read_bin_params(dev_type, L_um, W_um, bin_index)
        return LhcBox(dev_type=dev_type, bin_index=bin_index, published=pub)
    raise ValueError(f"unknown box mode {mode!r}")


# ---------------------------------------------------------------------------
# Metric evaluation + residual against the PDK backend
# ---------------------------------------------------------------------------
def eval_params(dev_type: str, L_um: float, W_um: float, bin_index: int,
                curves: list[Curve], params: dict[str, float]):
    sims = simulate_pdk(dev_type, L_um, W_um, curves, params=params,
                        bin_index=bin_index)
    meas = [c.Id for c in curves]
    return device_rrms(sims, meas), sims


# ---------------------------------------------------------------------------
# Confirmed-setup metric: optimization layout, evaluation, residuals
# ---------------------------------------------------------------------------
@dataclass
class NewMetricLayout:
    """Per-curve preprocessing for the confirmed-setup objective: curve
    index into the device curve list, trim start, cleaned measured array,
    trimmed denominator, tag string. The inclusion set is FROZEN (typically
    to the published-card baseline's included curves) so candidates can't
    game the sim-dependent exclusion rules."""
    entries: list[tuple[int, int, np.ndarray, float, str]]
    include_tags: set[str]


def new_metric_layout(dev_type: str, L_um: float, W_um: float,
                      curves: list[Curve],
                      include_tags: set[str]) -> NewMetricLayout:
    tagged = tag_metric_curves(curves)
    index = {id(c): i for i, c in enumerate(curves)}
    entries = []
    for (kind, bias), c in tagged.items():
        tag = f"{kind}@{bias:g}"
        if tag not in include_tags:
            continue
        start, cleaned, den = curve_trim(dev_type, L_um, W_um, kind,
                                         np.asarray(c.Id, dtype=np.float64))
        if den > 0 and np.isfinite(den):
            entries.append((index[id(c)], start, cleaned, den, tag))
    return NewMetricLayout(entries=entries, include_tags=set(include_tags))


def eval_params_new(dev_type: str, L_um: float, W_um: float, bin_index: int,
                    curves: list[Curve], params: dict[str, float],
                    include_tags: set[str]):
    """Simulate and score with the confirmed-setup metric. Returns
    (fixed-inclusion score, official dynamic-inclusion score, sims)."""
    sims = simulate_pdk(dev_type, L_um, W_um, curves, params=params,
                        bin_index=bin_index)
    fixed = score_device_new(dev_type, L_um, W_um, curves, sims,
                             include_tags=include_tags)
    official = score_device_new(dev_type, L_um, W_um, curves, sims)
    return fixed, official, sims


def residual_fn_new(z: np.ndarray, box, dev_type: str, L_um: float,
                    W_um: float, bin_index: int, curves: list[Curve],
                    layout: NewMetricLayout) -> np.ndarray:
    """Least-squares residuals whose sum of squares equals the
    fixed-inclusion confirmed-setup device RRMS."""
    params = box.z_to_params(z)
    sims = simulate_pdk(dev_type, L_um, W_um, curves, params=params,
                        bin_index=bin_index)
    n = max(len(layout.entries), 1)
    out = np.empty(n, dtype=np.float64)
    for oi, (ci, start, cleaned, den, _tag) in enumerate(layout.entries):
        s = np.asarray(sims[ci], dtype=np.float64)
        m = cleaned
        k = min(len(s), len(m))
        s, m = s[start:k], m[start:k]
        rrms = float(np.sqrt(np.mean((s - m) ** 2)) / den) if len(m) else PENALTY
        out[oi] = (np.sqrt(rrms / n)
                   if np.isfinite(rrms) and rrms >= 0 else PENALTY)
    if not layout.entries:
        out[0] = 0.0
    return out


def residual_fn(z: np.ndarray, box: ThetaBox, dev_type: str, L_um: float,
                W_um: float, bin_index: int, curves: list[Curve],
                flat: FlatCurves) -> np.ndarray:
    params = box.z_to_params(z)
    sims = simulate_pdk(dev_type, L_um, W_um, curves, params=params,
                        bin_index=bin_index)
    out = np.empty(max(len(flat.kept), 1), dtype=np.float64)
    for oi, ((a, b), ci, denominator) in enumerate(
        zip(flat.slices, flat.kept, flat.denominators)
    ):
        n = b - a
        s = np.asarray(sims[ci], dtype=np.float64)[:n]
        mvals = np.asarray(curves[ci].Id, dtype=np.float64)[:n]
        rrms = float(np.sqrt(np.mean((s - mvals) ** 2)) / denominator)
        out[oi] = (
            np.sqrt(rrms / max(flat.n_total_curves, 1))
            if np.isfinite(rrms) and rrms >= 0
            else PENALTY
        )
    if not flat.kept:
        out[0] = 0.0
    return out


# ---------------------------------------------------------------------------
# Multistart finite-difference extraction (per device)
# ---------------------------------------------------------------------------
@dataclass
class PdkFitResult:
    dev_type: str
    L_um: float
    W_um: float
    bin_index: int
    params: dict[str, float]
    rrms: float
    start_rrms: float
    method: str
    n_starts: int
    runtime_seconds: float
    sims: list = field(default_factory=list)


def fd_extract_device(
    dev_type: str,
    L_um: float,
    W_um: float,
    bin_index: int,
    curves: list[Curve],
    n_starts: int = 6,
    sigma_z: float = 1.0,
    max_nfev: int = 150,
    seed: int = 0,
    extra_start_zs: list[np.ndarray] | None = None,
    method_label: str = "pdk_fd_multistart",
) -> PdkFitResult:
    t0 = time.time()
    box = theta_box_for(dev_type, L_um, W_um, bin_index)
    flat = flatten_paper_curves(curves)
    start_metrics, _ = eval_params(dev_type, L_um, W_um, bin_index, curves,
                                   box.published)
    best = {"params": box.published, "m": start_metrics, "which": "published"}

    rng = np.random.default_rng(seed)
    z0 = box.z_published
    starts = [z0]
    if extra_start_zs:
        starts += [np.asarray(z, dtype=np.float64) for z in extra_start_zs]
    while len(starts) < n_starts:
        starts.append(z0 + rng.normal(0, sigma_z, size=len(PARAMS7)))

    f = lambda z: residual_fn(z, box, dev_type, L_um, W_um, bin_index, curves, flat)
    for si, zs in enumerate(starts):
        try:
            sol = least_squares(f, zs, method="trf", jac="2-point",
                                diff_step=2e-2, max_nfev=max_nfev)
        except Exception as e:  # noqa: BLE001
            logger.debug("start %d failed: %s", si, e)
            continue
        params = box.z_to_params(np.asarray(sol.x, dtype=np.float64))
        m, _ = eval_params(dev_type, L_um, W_um, bin_index, curves, params)
        if np.isfinite(m["rrms"]) and m["rrms"] < best["m"]["rrms"]:
            best = {"params": params, "m": m, "which": f"start{si}"}

    final_m, sims = eval_params(dev_type, L_um, W_um, bin_index, curves,
                                best["params"])
    return PdkFitResult(
        dev_type=dev_type, L_um=L_um, W_um=W_um, bin_index=bin_index,
        params=best["params"],
        rrms=float(final_m["rrms"]),
        start_rrms=float(start_metrics["rrms"]),
        method=method_label, n_starts=len(starts),
        runtime_seconds=time.time() - t0, sims=sims,
    )
