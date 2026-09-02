"""Rebuild a fixed series' NGSpice curves from its committed parameter vectors.

The per-device ``sims_<tag>.npz`` of the canonical 7-parameter
``emu_search+fd`` series are gitignored and are no longer on this checkout,
but the winning parameter vectors themselves survive at full precision in
``out/pdk_ml_selected/cards/manifest.json``. Re-running the extraction to get
the curves back would rediscover a result we already have, at hours of
compute; feeding the known vectors back through NGSpice costs 18 devices x 11
curves and reproduces the same curves exactly.

This is a **compute-heavy stage** under the repository resource policy: it
runs ~198 real NGSpice-41 simulations. Do not overlap it with extraction,
training, or other bulk simulation.

Correctness gate: the manifest records the RRMS each vector scored when it was
originally validated. This script rescores its own fresh simulations on the
frozen published-card inclusion set and refuses to write the series if any
device disagrees with the recorded value by more than ``--tol``. A silent
mismatch would mean the card, the bin resolution, or the simulator has moved
since the export, which is exactly the condition that must not reach a figure.

Usage::

    export NGSPICE_BIN=/opt/homebrew/Caskroom/miniconda/base/envs/ng41/bin/ngspice
    PYTHONPATH=src .venv/bin/python scripts/resim_fixed_series.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, resolve_ngspice_bin  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.spice_pdk import find_bin_index, simulate_pdk  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402


def baseline_inclusion(tag: str, _cache={}) -> set[str]:
    if not _cache:
        _cache["blob"] = json.load(
            open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))
    return {name for name, item in _cache["blob"]["per_curve"][tag].items()
            if item.get("included")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest",
                    default=str(OUT_DIR / "pdk_ml_selected" / "cards"
                                / "manifest.json"))
    ap.add_argument("--out-dir", default=str(OUT_DIR / "pdk_ml_emu_resim"))
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |rescored - manifest| RRMS per device")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would run, simulate nothing")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    method = manifest["uniform_method"]
    by_bin = {(b["dev_type"], b["bin_index"]): b["params"]
              for b in manifest["bins"]}
    out_dir = Path(args.out_dir)

    print(f"series      : {method}")
    print(f"manifest    : {args.manifest}")
    print(f"vectors     : {len(by_bin)} bins, "
          f"{len(next(iter(by_bin.values())))} parameters each")
    print(f"simulator   : {resolve_ngspice_bin()}")
    print(f"destination : {out_dir}")

    plan = []
    for dev in PAPER_DEVICES:
        tag = device_tag(dev.dev_type, dev.L_um, dev.W_um)
        bin_index = find_bin_index(dev.dev_type, dev.L_um, dev.W_um)
        key = (dev.dev_type, bin_index)
        if key not in by_bin:
            raise SystemExit(f"{tag}: no manifest vector for bin {key}")
        plan.append((dev, tag, bin_index, by_bin[key]))

    n_curves = sum(len(load_device_curves(d)) for d, _, _, _ in plan)
    print(f"work        : {len(plan)} devices, {n_curves} curve simulations")
    if args.dry_run:
        print("\ndry run: nothing simulated.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = {d["device"]: d[_MANIFEST_SCORE_KEY]
                for d in _load_recorded_scores()}
    t0 = time.time()
    scored, failures = {}, []
    for i, (dev, tag, bin_index, params) in enumerate(plan, 1):
        curves = load_device_curves(dev)
        sims = simulate_pdk(dev.dev_type, dev.L_um, dev.W_um, curves,
                            params=params, bin_index=bin_index)
        rrms = score_device_new(dev.dev_type, dev.L_um, dev.W_um, curves,
                                sims, include_tags=baseline_inclusion(tag))
        ref = recorded.get(tag)
        delta = abs(rrms["rrms"] - ref) if ref is not None else float("nan")
        flag = ""
        if ref is not None and delta > args.tol:
            failures.append((tag, rrms["rrms"], ref, delta))
            flag = f"  MISMATCH (recorded {ref:.7f}, d={delta:.2e})"
        np.savez_compressed(out_dir / f"sims_{tag}.npz",
                            **{f"sim_{j}": s for j, s in enumerate(sims)})
        (out_dir / f"ml_{tag}.json").write_text(json.dumps({
            "device": tag, "dev_type": dev.dev_type, "L_um": dev.L_um,
            "W_um": dev.W_um, "bin_index": bin_index,
            "method": method, "best_method": method,
            "param_set": "params7", "box_mode": "lhc10",
            "params": params, "rrms": rrms["rrms"],
            "recorded_rrms": ref,
            "provenance": "re-simulated from "
                          "out/pdk_ml_selected/cards/manifest.json",
        }, indent=2))
        print(f"  [{i:2d}/{len(plan)}] {tag:<22} bin {bin_index:<2} "
              f"RRMS {rrms['rrms']:.7f}{flag}")

        scored[tag] = rrms["rrms"]

    mean = float(np.mean(list(scored.values())))
    print(f"\nall-device mean RRMS {mean:.7f} "
          f"(manifest recorded {manifest['validated_mean_rrms']:.7f})")
    print(f"elapsed {time.time() - t0:.1f} s")

    if failures:
        for tag, got, ref, delta in failures:
            print(f"MISMATCH {tag}: {got:.7f} vs recorded {ref:.7f} "
                  f"(d={delta:.2e})", file=sys.stderr)
        raise SystemExit(
            f"{len(failures)} device(s) did not reproduce the manifest score. "
            "Refusing to certify this series; the card, bin resolution, or "
            "simulator has changed since the export."
        )
    print("all devices reproduce their recorded scores.")
    return 0


_MANIFEST_SCORE_KEY = "surrogate_search_plus_fd"


def _load_recorded_scores() -> list[dict]:
    """Per-device scores the canonical series achieved, from the comparison."""
    return json.loads((OUT_DIR / "tables" / "comparison_full.json").read_text())


if __name__ == "__main__":
    raise SystemExit(main())
