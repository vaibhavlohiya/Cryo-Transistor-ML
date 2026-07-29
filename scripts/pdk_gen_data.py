#!/usr/bin/env python3
"""Generate synthetic NGSpice training data per device (PDK chain).

The default confirmed-setup mode generates a 10,000-point Latin hypercube in
the +/-10% physical box around each native bin's published parameter vector,
then simulates all 11 measured-bias metric curves in NGSpice-41. The legacy
wide z-space sampler remains available with ``--box wide``.

Output: data/processed/pdk_synth/<tag>.npz with
  Z      (N, n_params)  z-space theta samples
  THETA  (N, n_params)  physical params (ACTIVE_PARAMS order; the
                        ``param_names`` array records the exact layout)
  IDS    (N, P)  simulated currents at every kept measured point
  ok     (N,)    all-finite mask
plus the static curve layout (point voltages, slices, measured Id) needed
to reconstruct curves.

  python scripts/pdk_gen_data.py --num-samples 10000 --workers 10 --box lhc10
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("pdk_gen_data")

OUT_SYNTH = PROCESSED_DIR / "pdk_synth"


def gen_for_device(job):
    spec, num_samples, seed, extra_center, append, box_mode, out_synth = job
    import time
    from cryoml.pdk_extract import ACTIVE_PARAMS, make_box
    from cryoml.spice_pdk import simulate_pdk

    d, bin_index = spec
    n_par = len(ACTIVE_PARAMS)
    t0 = time.time()
    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    curves = load_device_curves(d)
    box = make_box(d.dev_type, d.L_um, d.W_um, bin_index, box_mode)
    z0 = box.z_published

    rng = np.random.default_rng(seed)
    if box_mode == "lhc10":
        # Confirmed-setup sampling: Latin hypercube over the ±10 % box
        # around the published bin values (nomSweep_latinHypercube.py),
        # with row 0 pinned to the published card itself.
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_par, seed=seed)
        U = sampler.random(n=num_samples)
        THETA = box.lo + U * (box.hi - box.lo)
        THETA[0] = np.array([box.published[p] for p in ACTIVE_PARAMS])
        Z = np.stack([
            box.params_to_z({p: t[i] for i, p in enumerate(ACTIVE_PARAMS)})
            for t in THETA])
    elif append and extra_center is not None:
        # Active densification: most samples in the current winner's basin,
        # the rest uniform, appended to the existing dataset.
        zc = box.params_to_z(extra_center)
        n_tight = num_samples // 2
        n_mid = num_samples // 4
        n_box = num_samples - n_tight - n_mid
        Z = np.concatenate([
            zc + rng.normal(0, 0.25, size=(n_tight, n_par)),
            zc + rng.normal(0, 0.6, size=(n_mid, n_par)),
            rng.uniform(-3.5, 3.5, size=(n_box, n_par)),
        ], axis=0)
        Z[0] = zc
    elif extra_center is not None:
        # Concentrate a quarter of the budget around a known-good basin
        # (e.g. the FD-control winner) while keeping global coverage.
        zc = box.params_to_z(extra_center)
        n_tight = num_samples // 4
        n_center = num_samples // 4
        n_wide = num_samples // 4
        n_box = num_samples - n_tight - n_center - n_wide
        Z = np.concatenate([
            z0 + rng.normal(0, 0.5, size=(n_tight, n_par)),
            zc + rng.normal(0, 0.5, size=(n_center, n_par)),
            z0 + rng.normal(0, 1.2, size=(n_wide, n_par)),
            rng.uniform(-3.5, 3.5, size=(n_box, n_par)),
        ], axis=0)
        Z[1] = zc  # always include the extra center itself
    else:
        n_tight = num_samples // 3
        n_wide = num_samples // 3
        n_box = num_samples - n_tight - n_wide
        Z = np.concatenate([
            z0 + rng.normal(0, 0.5, size=(n_tight, n_par)),
            z0 + rng.normal(0, 1.2, size=(n_wide, n_par)),
            rng.uniform(-3.5, 3.5, size=(n_box, n_par)),
        ], axis=0)
    if box_mode != "lhc10":
        if not (append and extra_center is not None):
            Z[0] = z0  # always include the published point
        THETA = np.zeros((len(Z), n_par), dtype=np.float64)
        for i, z in enumerate(Z):
            params = box.z_to_params(z)
            THETA[i] = [params[p] for p in ACTIVE_PARAMS]

    P = sum(len(c.Id) for c in curves)
    IDS = np.full((len(THETA), P), np.nan, dtype=np.float64)
    for i, t in enumerate(THETA):
        params = {p: float(t[j]) for j, p in enumerate(ACTIVE_PARAMS)}
        sims = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves, params=params,
                            bin_index=bin_index)
        IDS[i] = np.concatenate([np.asarray(s, dtype=np.float64)[:len(c.Id)]
                                 for s, c in zip(sims, curves)])
    ok = np.isfinite(IDS).all(axis=1)

    # static layout
    Vg = np.concatenate([np.asarray(c.Vg, dtype=np.float64)[:len(c.Id)] for c in curves])
    Vd = np.concatenate([np.asarray(c.Vd, dtype=np.float64)[:len(c.Id)] for c in curves])
    meas = np.concatenate([np.asarray(c.Id, dtype=np.float64) for c in curves])
    slices, cur = [], 0
    kinds, fixeds = [], []
    for c in curves:
        slices.append((cur, cur + len(c.Id)))
        cur += len(c.Id)
        kinds.append(c.kind)
        fixeds.append(c.fixed)

    out_synth.mkdir(parents=True, exist_ok=True)
    out_path = out_synth / f"{tag}.npz"
    if append and out_path.exists():
        old = np.load(out_path, allow_pickle=True)
        if old["IDS"].shape[1] == IDS.shape[1]:
            Z = np.concatenate([old["Z"].astype(np.float64), Z])
            THETA = np.concatenate([old["THETA"], THETA])
            IDS = np.concatenate([old["IDS"], IDS])
            ok = np.concatenate([old["ok"], ok])
        else:
            logger.warning("%s: layout changed, not appending", tag)
    np.savez_compressed(
        out_path,
        Z=Z.astype(np.float32), THETA=THETA.astype(np.float64),
        IDS=IDS.astype(np.float64), ok=ok,
        Vg=Vg, Vd=Vd, meas=meas,
        slices=np.array(slices, dtype=np.int64),
        kinds=np.array(kinds), fixeds=np.array(fixeds, dtype=np.float64),
        bin_index=bin_index, box_mode=np.array(box_mode),
        published=np.array([box.published[p] for p in ACTIVE_PARAMS],
                           dtype=np.float64),
        param_names=np.array(ACTIVE_PARAMS),
    )
    return tag, int(ok.sum()), len(Z), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--centers-from", default=None,
                    help="directory of <prefix>_<tag>.json records whose "
                         "'params' (or params_by_method[best_method]) become "
                         "extra sampling centers (e.g. out/pdk_fd)")
    ap.add_argument("--append", action="store_true",
                    help="densify around the centers and append to the "
                         "existing dataset instead of replacing it")
    ap.add_argument("--box", default="lhc10", choices=("lhc10", "wide"),
                    help="sampling box: 'lhc10' = confirmed-setup Latin "
                         "hypercube ±10%% around the published bin values "
                         "(default), 'wide' = legacy broad z-space sampling")
    ap.add_argument("--param-set", default="params7",
                    choices=("params7", "params15"),
                    help="tuned-parameter set; params7 is the canonical "
                         "protocol, params15 the labeled experiment")
    ap.add_argument("--out-dir", default=None,
                    help="output dataset directory (default: pdk_synth for "
                         "params7, pdk_synth_<set> otherwise)")
    args = ap.parse_args()
    if args.box == "lhc10" and (args.centers_from or args.append):
        ap.error("--centers-from/--append are legacy wide-box options; the "
                 "confirmed lhc10 dataset must remain an unbiased Latin "
                 "hypercube around the published vector")

    # Resolve the parameter set before any cryoml.pdk_extract import; spawn
    # workers inherit the environment and therefore agree with the parent.
    import os
    os.environ["CRYOML_PARAM_SET"] = args.param_set
    if args.out_dir:
        out_synth = Path(args.out_dir)
    elif args.param_set == "params7":
        out_synth = OUT_SYNTH
    else:
        out_synth = PROCESSED_DIR / f"pdk_synth_{args.param_set}"

    ensure_dirs()
    bins = {r["device"]: int(r["bin_index"]) for r in json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]}
    centers = {}
    if args.centers_from:
        for path in Path(args.centers_from).glob("*.json"):
            rec = json.loads(path.read_text())
            if not isinstance(rec, dict) or "device" not in rec:
                continue
            if "params" in rec:
                centers[rec["device"]] = rec["params"]
            elif "params_by_method" in rec and "best_method" in rec:
                centers[rec["device"]] = rec["params_by_method"][rec["best_method"]]
    devices = parse_device_list(args.devices)
    jobs = []
    for i, d in enumerate(devices):
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        jobs.append(((d, bins[tag]), args.num_samples, args.seed + 1000 * i,
                     centers.get(tag), args.append, args.box, out_synth))

    ctx = get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for tag, nok, n, dt in pool.imap_unordered(gen_for_device, jobs):
            logger.info("%-22s %d/%d ok  (%.0fs)", tag, nok, n, dt)
    print(f"wrote {out_synth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
