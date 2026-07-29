#!/usr/bin/env python3
"""Surrogate-driven BSIM4 extraction against the PDK NGSpice chain.

One emulator is trained per device using only synthetic NGSpice data from
``data/processed/pdk_synth/<tag>.npz``:

* **Emulator** E(z) -> signed_log Id at every kept measured-bias point.
  The MLP makes the parameter-to-I-V model differentiable, so thousands of
  parameter vectors can be optimized against the measured I-V curves without
  putting NGSpice in the search loop.

The best surrogate-search candidates are validated in real NGSpice in the
native geometry bin, then optionally FD-polished on the confirmed-setup
rrmsCalc objective. Curve inclusion is frozen to the published-card
baseline's included set. The search box follows the synth dataset's box_mode
(±10 % LHC by default).

  python scripts/pdk_ml_extract.py --device mps
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.utils import device_tag, get_logger, set_seed  # noqa: E402

logger = get_logger("pdk_ml_extract")

OUT_ML = OUT_DIR / "pdk_surrogate_final"
SYNTH = PROCESSED_DIR / "pdk_synth"
I_REF = 1e-9


def slog(x):
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x) / I_REF)
    return torch.sign(x) * torch.log1p(torch.abs(x) / I_REF)


def current_from_slog(y):
    return torch.sign(y) * I_REF * torch.expm1(torch.abs(y))


def mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def train_net(net, X, Y, device, epochs=1500, lr=1e-3, batch=None, wd=1e-5,
              val_frac=0.15, patience=200):
    """Full-batch (or minibatch) Adam training with early stopping."""
    n = len(X)
    idx = torch.randperm(n)
    n_val = max(1, int(n * val_frac))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_v, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        if batch is None:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt), Yt)
            loss.backward()
            opt.step()
        else:
            perm = torch.randperm(len(Xt))
            for s in range(0, len(Xt), batch):
                b = perm[s:s + batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(net(Xv), Yv))
            if v < best_v - 1e-6:
                best_v, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 10
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return best_v


def baseline_include_tags(tag: str) -> set[str]:
    """Curve-inclusion set frozen to the published-card baseline run, so no
    candidate can game the sim-dependent exclusion rules of the metric."""
    per_curve = json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["per_curve"][tag]
    return {t for t, v in per_curve.items() if v.get("included")}


def extract_device(d, tdev, n_adam_starts=512, adam_steps=400, n_validate=8,
                   n_polish=3, max_nfev=80, seed=0,
                   emu_sizes=(256, 256, 256), emu_save_path=None,
                   synth_dir=None):
    from cryoml.metrics import clean_current
    from cryoml.pdk_extract import (ACTIVE_PARAM_SET, ACTIVE_PARAMS, LhcBox,
                                    ThetaBox, eval_params_new,
                                    new_metric_layout, residual_fn_new)
    from scipy.optimize import least_squares

    tag = device_tag(d.dev_type, d.L_um, d.W_um)
    t0 = time.time()
    set_seed(seed)
    data = np.load((synth_dir or SYNTH) / f"{tag}.npz", allow_pickle=True)
    IDS, ok = data["IDS"], data["ok"]
    meas, slices = data["meas"], data["slices"]
    bin_index = int(data["bin_index"])
    n_par = len(ACTIVE_PARAMS)
    # The dataset must have been generated with the same theta layout; fail
    # rather than silently reinterpreting columns.
    if "param_names" in data:
        stored = tuple(str(p) for p in data["param_names"])
        if stored != ACTIVE_PARAMS:
            raise RuntimeError(f"{tag}: dataset param_names {stored} do not "
                               f"match active set {ACTIVE_PARAMS}")
    elif n_par != len(data["published"]):
        raise RuntimeError(f"{tag}: dataset has {len(data['published'])} "
                           f"parameters; active set expects {n_par}")
    published = {p: float(v)
                 for p, v in zip(ACTIVE_PARAMS, data["published"])}
    box_mode = str(data["box_mode"]) if "box_mode" in data else "wide"
    if box_mode == "lhc10":
        box = LhcBox(dev_type=d.dev_type, bin_index=bin_index,
                     published=published)
        z_clamp = 8.0     # covers >99.9 % of the ±10 % box in logit space
    else:
        box = ThetaBox(dev_type=d.dev_type, bin_index=bin_index,
                       published=published)
        z_clamp = 3.5
    # recompute z from physical theta with the CURRENT box so the dataset
    # stays valid even when bound definitions evolve
    THETA = data["THETA"].astype(np.float64)
    Z = np.stack([box.params_to_z({p: t[i]
                                   for i, p in enumerate(ACTIVE_PARAMS)})
                  for t in THETA])
    curves = load_device_curves(d)
    include_tags = baseline_include_tags(tag)
    layout = new_metric_layout(d.dev_type, d.L_um, d.W_um, curves,
                               include_tags)

    # Emulator I/O covers every nonzero curve on its full grid; the search
    # objective below slices the confirmed-setup metric's trimmed spans.
    kept_mask = np.zeros(len(meas), dtype=bool)
    kept_span = {}                     # curve index -> (a, b) in kept coords
    meas_clean_parts = []
    kept_offset = 0
    for ci, (a, b) in enumerate(slices):
        m = np.asarray(meas[a:b], dtype=np.float64)
        den = np.mean(np.abs(m))
        if den > 0 and np.isfinite(den):
            kept_mask[a:b] = True
            kept_span[ci] = (kept_offset, kept_offset + b - a)
            meas_clean_parts.append(clean_current(m))
            kept_offset += b - a
    meas_clean_flat = np.concatenate(meas_clean_parts)
    # confirmed-setup objective spans in kept coordinates
    search_layout = []
    for ci, start, _cleaned, den, _t in layout.entries:
        A, B = kept_span[ci]
        search_layout.append((A + start, B, den))

    Zok, Yok = Z[ok], IDS[ok][:, kept_mask]
    Y_slog = slog(Yok)

    dev = torch.device(tdev)
    Zt = torch.tensor(Zok, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y_slog, dtype=torch.float32, device=dev)
    P = Yt.shape[1]

    # ---------------- emulator E(z) -> slog Id ----------------
    emu = mlp([n_par, *emu_sizes, P]).to(dev)
    emu_val = train_net(emu, Zt, Yt, dev, epochs=2000, lr=1e-3,
                        batch=4096 if len(Zt) > 4000 else None)
    if emu_save_path is not None:
        torch.save({
            "state": emu.state_dict(), "emu_sizes": list(emu_sizes), "P": P,
            "curve_layout": search_layout, "n_curves": len(search_layout),
            "meas_kept": meas_clean_flat, "emu_val": float(emu_val),
        }, emu_save_path)

    rng = np.random.default_rng(seed + 7)

    # ------ Adam multistart search on the emulator (confirmed metric) ------
    meas_t = torch.tensor(meas_clean_flat, dtype=torch.float32, device=dev)
    S = n_adam_starts
    if S < 3:
        raise ValueError("n_adam_starts must be at least 3")
    z0 = box.z_published
    n_fixed = 1
    if isinstance(box, LhcBox):
        # uniform coverage of the ±10 % box: z = logit(u), u ~ U(0.001, 0.999)
        u = rng.uniform(1e-3, 1 - 1e-3, size=(S - S // 2 - n_fixed, n_par))
        box_starts = np.log(u / (1 - u))
        local = z0 + rng.normal(0, 2.0, size=(S // 2, n_par))
    else:
        box_starts = rng.uniform(-3.0, 3.0,
                                 size=(S - S // 2 - n_fixed, n_par))
        local = z0 + rng.normal(0, 0.7, size=(S // 2, n_par))
    starts = np.concatenate([
        z0[None, :], local, box_starts,
    ], axis=0)
    zv = torch.tensor(starts, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([zv], lr=0.05)
    best_loss = np.full(len(starts), np.inf)
    best_z = starts.copy()
    for _ in range(adam_steps):
        opt.zero_grad()
        pred = current_from_slog(emu(zv))
        curve_rrms = []
        for a, b, denominator in search_layout:
            rmse = torch.sqrt(torch.mean(
                (pred[:, a:b] - meas_t[a:b].unsqueeze(0)) ** 2, dim=1
            ))
            curve_rrms.append(rmse / denominator)
        loss_per = torch.stack(curve_rrms, dim=1).sum(dim=1) / len(search_layout)
        loss_per.sum().backward()
        opt.step()
        with torch.no_grad():
            zv.clamp_(-z_clamp, z_clamp)  # stay inside the training range
        lv = loss_per.detach().cpu().numpy()
        improved = lv < best_loss
        if improved.any():
            zc = zv.detach().cpu().numpy().astype(np.float64)
            best_loss[improved] = lv[improved]
            best_z[improved] = zc[improved]
    order = np.argsort(best_loss)

    # ---------------- NGSpice validation of candidates ----------------
    # Candidates are ranked/selected on the FIXED-inclusion confirmed-setup
    # score; the official dynamic-inclusion score is recorded alongside.
    results = {}
    start_fixed, start_official, _ = eval_params_new(
        d.dev_type, d.L_um, d.W_um, bin_index, curves, box.published,
        include_tags)
    results["published"] = {"params": box.published,
                            "rrms": float(start_fixed["rrms"]),
                            "rrms_official": float(start_official["rrms"]),
                            "sigma": float(start_official["sigma"])}

    def validate(z):
        params = box.z_to_params(np.asarray(z, dtype=np.float64))
        fixed, official, _ = eval_params_new(
            d.dev_type, d.L_um, d.W_um, bin_index, curves, params,
            include_tags)
        m = dict(fixed)
        m["rrms_official"] = float(official["rrms"])
        m["sigma_official"] = float(official["sigma"])
        return params, m

    cand = []
    # top emulator candidates (deduplicated in z by distance)
    seen = []
    for i in order:
        z = best_z[i]
        if any(np.linalg.norm(z - s) < 0.5 for s in seen):
            continue
        seen.append(z)
        cand.append(("emu_search", z))
        if len(seen) >= n_validate:
            break
    validated = []
    for lbl, z in cand:
        params, m = validate(z)
        rr = m["rrms"] if np.isfinite(m["rrms"]) else np.inf
        validated.append((rr, lbl, z, params, m))
    validated.sort(key=lambda t: t[0])

    best_emu = next((v for v in validated if v[1] == "emu_search"), None)
    if best_emu:
        results["emu_search"] = {"params": best_emu[3],
                                 "rrms": float(best_emu[4]["rrms"]),
                                 "rrms_official": best_emu[4]["rrms_official"]}

    # ---------------- FD polish of the best few candidates ----------------
    f = lambda z: residual_fn_new(z, box, d.dev_type, d.L_um, d.W_um,
                                  bin_index, curves, layout)
    # if every ML candidate failed in NGSpice, polish from the published
    # start instead so the method always returns something sane
    if not any(np.isfinite(v[0]) for v in validated):
        sf = dict(start_fixed)
        sf["rrms_official"] = float(start_official["rrms"])
        validated.insert(0, (start_fixed["rrms"], "emu_search",
                             box.z_published, box.published, sf))
    polished = []
    fd_attempts = []
    emu_validated = [v for v in validated if v[1] == "emu_search"]
    for rank, (rr, lbl, z, params, m) in enumerate(
            emu_validated[:n_polish]):
        fd_t0 = time.time()
        try:
            sol = least_squares(f, z, method="trf", jac="2-point",
                                diff_step=2e-2, max_nfev=max_nfev)
            p2, m2 = validate(sol.x)
            polished.append((m2["rrms"], lbl + "+fd", p2, m2))
            accepted = (np.isfinite(m2["rrms"]) and m2["rrms"] <= rr)
            fd_attempts.append({
                "candidate_rank": rank,
                "is_raw_winner_pair": rank == 0,
                "start_rrms": float(rr),
                "endpoint_rrms": float(m2["rrms"]),
                "endpoint_accepted_for_pair": bool(accepted),
                "start_params": params,
                "endpoint_params": p2,
                "paired_params": p2 if accepted else params,
                "paired_rrms": float(m2["rrms"] if accepted else rr),
                "nfev": int(sol.nfev),
                "njev": int(sol.njev) if sol.njev is not None else None,
                "success": bool(sol.success),
                "status": int(sol.status),
                "runtime_s": round(time.time() - fd_t0, 1),
            })
        except Exception as exc:  # noqa: BLE001
            fd_attempts.append({
                "candidate_rank": rank,
                "is_raw_winner_pair": rank == 0,
                "start_rrms": float(rr),
                "endpoint_rrms": None,
                "endpoint_accepted_for_pair": False,
                "start_params": params,
                "endpoint_params": None,
                "paired_params": params,
                "paired_rrms": float(rr),
                "nfev": 0, "njev": None, "success": False,
                "status": None, "runtime_s": round(time.time() - fd_t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
    for rr, lbl, p2, m2 in polished:
        cur = results.get(lbl)
        if cur is None or (np.isfinite(rr) and rr < cur["rrms"]):
            results[lbl] = {"params": p2, "rrms": float(m2["rrms"]),
                            "rrms_official": m2.get("rrms_official")}
    # A polish stage is non-regressing by definition. Preserve the attempted
    # endpoints above, but materialize the raw vector if every FD attempt is
    # worse or fails.
    if "emu_search" in results:
        raw = results["emu_search"]
        fd = results.get("emu_search+fd")
        if (fd is None or not np.isfinite(fd["rrms"])
                or fd["rrms"] > raw["rrms"]):
            results["emu_search+fd"] = dict(raw)

    # Final pick is the best paper-exact NGSpice score.
    ml_keys = [k for k in results if k != "published"
               and np.isfinite(results[k]["rrms"])]
    if not ml_keys:
        ml_keys = ["published"]
    best_key = min(ml_keys, key=lambda k: results[k]["rrms"])
    from cryoml.spice_pdk import simulate_pdk
    best_params = results[best_key]["params"]
    sims = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves,
                        params=best_params, bin_index=bin_index)

    rec = {
        "device": tag, "dev_type": d.dev_type, "L_um": d.L_um, "W_um": d.W_um,
        "bin_index": bin_index, "paper_reported": d.paper_rrms,
        "param_set": ACTIVE_PARAM_SET, "param_names": list(ACTIVE_PARAMS),
        "box_mode": box_mode, "include_tags": sorted(include_tags),
        "selection_policy": "fixed surrogate search, with FD as an ablation",
        "emulator_val_mse": float(emu_val),
        "fd_attempts": fd_attempts,
        "production_config": {
            "seed": int(seed), "emulator_hidden": list(emu_sizes),
            "n_adam_starts": int(n_adam_starts),
            "adam_steps": int(adam_steps),
            "n_ngspice_validate": int(n_validate),
            "n_fd_polish": int(n_polish),
            "fd_max_nfev": int(max_nfev),
            "fd_diff_step": 2e-2,
        },
        "n_synth": int(ok.sum()),
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "params"}
                    for k, v in results.items()},
        "params_by_method": {k: v["params"] for k, v in results.items()},
        "best_method": best_key,
        "rrms": results[best_key]["rrms"],
        "rrms_official": results[best_key].get("rrms_official"),
        "start_rrms": results["published"]["rrms"],
        "runtime_s": round(time.time() - t0, 1),
    }
    return rec, sims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None)
    ap.add_argument("--device", default="mps", help="torch device")
    ap.add_argument("--n-adam-starts", type=int, default=512)
    ap.add_argument("--adam-steps", type=int, default=400)
    ap.add_argument("--n-validate", type=int, default=8)
    ap.add_argument("--n-polish", type=int, default=3)
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emu-arch", default="256,256,256",
                    help="comma-separated emulator hidden sizes")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--param-set", default="params7",
                    choices=("params7", "params15"),
                    help="tuned-parameter set; params7 is the canonical "
                         "protocol, params15 the labeled experiment")
    ap.add_argument("--synth-dir", default=None,
                    help="synthetic dataset directory (default: pdk_synth "
                         "for params7, pdk_synth_<set> otherwise)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip devices with an existing finite result")
    args = ap.parse_args()

    # Resolve before the lazy cryoml.pdk_extract import inside extract_device.
    import os
    os.environ["CRYOML_PARAM_SET"] = args.param_set
    if args.synth_dir:
        synth_dir = Path(args.synth_dir)
    elif args.param_set == "params7":
        synth_dir = SYNTH
    else:
        synth_dir = PROCESSED_DIR / f"pdk_synth_{args.param_set}"
    if args.param_set != "params7" and not args.out_dir:
        ap.error("--param-set experiments must name an explicit --out-dir "
                 "so canonical series are never overwritten")

    ensure_dirs()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_ML
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = parse_device_list(args.devices)

    emu_sizes = tuple(int(s) for s in args.emu_arch.split(","))

    rows = []
    for d in devices:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        result_path = out_dir / f"ml_{tag}.json"
        if args.resume and result_path.exists():
            old = json.load(open(result_path))
            if np.isfinite(old.get("rrms", np.nan)):
                logger.info("%-22s already complete; skipping", tag)
                continue
        rec, sims = extract_device(
            d, args.device, n_adam_starts=args.n_adam_starts,
            adam_steps=args.adam_steps, n_validate=args.n_validate,
            n_polish=args.n_polish, max_nfev=args.max_nfev, seed=args.seed,
            emu_sizes=emu_sizes,
            emu_save_path=out_dir / f"emu_{tag}.pt",
            synth_dir=synth_dir)
        rows.append(rec)
        json.dump(rec, open(out_dir / f"ml_{rec['device']}.json", "w"), indent=2)
        np.savez(out_dir / f"sims_{rec['device']}.npz",
                 **{f"sim_{i}": np.asarray(s) for i, s in enumerate(sims)})
        logger.info("%-22s %.3f->%.3f [%s] %ss",
                    rec["device"], rec["start_rrms"], rec["rrms"],
                    rec["best_method"], rec["runtime_s"])

    # Always aggregate every completed result in the output directory. This
    # keeps subset/resume runs from replacing the full experiment summary.
    from cryoml.metrics import device_rrms, family_totals, score_device_new
    rows = []
    new_by_tag = {}
    for path in sorted(out_dir.glob("ml_*.json")):
        rec = json.load(open(path))
        if np.isfinite(rec.get("rrms", np.nan)):
            d = next(d for d in PAPER_DEVICES
                     if device_tag(d.dev_type, d.L_um, d.W_um) == rec["device"])
            curves = load_device_curves(d)
            saved = np.load(out_dir / f"sims_{rec['device']}.npz")
            sims = [np.asarray(saved[f"sim_{i}"]) for i in range(len(curves))]
            include = set(rec.get("include_tags") or
                          baseline_include_tags(rec["device"]))
            fixed = score_device_new(d.dev_type, d.L_um, d.W_um, curves, sims,
                                     include_tags=include)
            official = score_device_new(d.dev_type, d.L_um, d.W_um, curves,
                                        sims)
            rec["rrms"] = float(fixed["rrms"])
            rec["rrms_official"] = float(official["rrms"])
            rec["sigma_official"] = float(official["sigma"])
            rec["n_curves_official"] = int(official["n_curves"])
            meas = [c.Id for c in curves]
            rec["legacy_rrms"] = float(device_rrms(sims, meas)["rrms"])
            json.dump(rec, open(path, "w"), indent=2)
            rows.append(rec)
            new_by_tag[rec["device"]] = official
    if not rows:
        raise RuntimeError(f"no finite ML results found in {out_dir}")

    scores = np.array([r["rrms"] for r in rows])
    baseline = {r["device"]: r for r in json.load(
        open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))["devices"]}
    base = np.array([baseline[r["device"]]["rrms"] for r in rows])
    summary = {
        "n_devices": len(rows),
        "metric": "confirmed-setup rrmsCalc port (fixed baseline inclusion)",
        "mean_rrms": float(np.nanmean(scores)),
        "baseline_mean_rrms": float(np.nanmean(base)),
        "wins_vs_baseline": int(np.sum(scores < base)),
        **{f"official_{k}": v for k, v in family_totals(new_by_tag).items()},
    }
    # per-method means
    method_keys = sorted({k for r in rows for k in r["methods"]
                          if k != "published"})
    for key in method_keys:
        vals = [r["methods"][key]["rrms"] for r in rows if key in r["methods"]]
        if vals:
            summary[f"mean_{key}"] = float(np.nanmean(vals))
            summary[f"n_{key}"] = len(vals)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "bin", "paper_reported", "start_rrms", "rrms",
                    "best_method"])
        for r in rows:
            w.writerow([r["device"], r["bin_index"], r["paper_reported"],
                        f"{r['start_rrms']:.4f}", f"{r['rrms']:.4f}",
                        r["best_method"]])
    print("\n=== PDK ML EXTRACTION ===")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
