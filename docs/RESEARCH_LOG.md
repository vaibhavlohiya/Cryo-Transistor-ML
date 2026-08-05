# Research log

Current goal: reproduce arXiv:2604.21625v1 in the authors' confirmed open
setup and extract its seven parameters through a surrogate search, with a
direct one-pass parameter MLP as the fixed comparison.
Current headline metrics are the confirmed `rrmsCalc.py` family-combined RRMS
and the all-device mean (lower is better). Entries dated June 2026 below are
historical experiments under an incompatible old card/metric/binning setup.

## Current experiment revision (completed 2026-07-15)

The user replaced the former inverse-network comparison with a direct MLP
forward pass: 301 sampled currents in linear and signed-log form (602 inputs)
map directly to the seven normalized parameters. This method uses one
inference pass, real-NGSpice validation, and no search or FD polish.

The surrogate extractor was rerun without the old inverse-network initializer.
Primary reporting contains paper cards, direct MLP, surrogate raw, and
surrogate + FD, each fixed across all 18 devices. Later fixed exploratory rows
add the foundation emulator and high-voltage guard without changing the
canonical export.

A new independent FD experiment starts at the published parameter vector and
compares published -> FD alone with surrogate raw -> surrogate + FD. It records
RRMS gains, `nfev`, runtime, and all seven physical parameter movements.

The fresh four-device scaling pilot completed all 56 unique cells. Emulator validation
MSE showed power exponents 0.793 with training-set size and 0.996 with network
weights. Search-start count had essentially zero exponent. Raw real-NGSpice
RRMS improved modestly with data and capacity, but FD-polished RRMS stayed
near 0.184 across all three axes on the four-device testbed. This is evidence
that local real-simulator polish, rather than more starts, dominates the final
pilot result. This is not the requested final scaling result: the same grid
was initially planned to cover all 18 transistors (252 cells total). The final
decision and completed all-device result are recorded below. The clean
surrogate-only production extraction then started as the sole heavy job (PID
31818, `out/pdk_surrogate_final`).

The clean surrogate-only extraction subsequently completed all 18 devices in
2,420.2 seconds total (median 94.3 seconds/device). Fixed raw surrogate search
scored nMOS `0.0827710`, pMOS `0.3580934`, combined `0.2204322`, all-device
`0.2357279`, with 17/18 wins over the published cards. The fixed surrogate+FD
series scored nMOS `0.0788212`, pMOS `0.3491185`, combined `0.2139699`,
all-device `0.2289864`, with 18/18 wins. These replace the nearly identical but
methodologically contaminated mixed-run surrogate values below. The all-18
scaling extension then started as the sole heavy job.

### Scaling stop decision (2026-07-14 16:02 EDT)

The completed all-18 data axis was sufficient to answer the scientific
question. From 375 to 6,000 samples/transistor, held-out emulator signed-log
MSE improved `0.0049619 -> 0.0005243` (about 9.5x), while raw real-NGSpice
RRMS changed only `0.2497 -> 0.2398` and FD-polished real-NGSpice RRMS was
flat/non-monotonic: `0.2280, 0.2270, 0.2273, 0.2280, 0.2294`. The best final
mean occurred at only 750 samples. With user approval, the remaining all-18
10,000/capacity/search cells were stopped. Final scaling uses the five complete
data configurations for all 18 devices (90 cells), arithmetic means, and faded
individual traces. Partial rows remain in `out/scaling/results.csv` for audit
but are excluded by `make_scaling_fig.py`. Capacity/search remain the labeled
four-device pilot.

### Completed fixed comparisons and FD study (2026-07-15)

| fixed method | nMOS | pMOS | combined | all-device | wins vs cards |
|---|---:|---:|---:|---:|---:|
| published cards | 0.1198250 | 0.3992356 | 0.2595303 | 0.2750531 | - |
| direct MLP, one pass/no FD | 0.1283710 | 0.3873572 | 0.2578641 | 0.2722522 | 11/18 |
| clean surrogate raw | 0.0827710 | 0.3580934 | 0.2204322 | 0.2357279 | 17/18 |
| clean surrogate + FD | 0.0788212 | 0.3491185 | 0.2139699 | 0.2289864 | 18/18 |
| foundation + FD (exploratory) | 0.0793719 | 0.3434197 | 0.2113958 | 0.2260651 | 18/18 |
| high-voltage guarded (diagnostic) | 0.0806091 | 0.3544207 | 0.2175149 | 0.2327267 | 18/18 |

The direct MLP is the requested ordinary I-V-to-parameter forward inference,
not the retired inverse-network initializer. It makes one prediction, then its
parameters are re-simulated in real NGSpice. The clean surrogate uses only its
fixed 2,048-start differentiable emulator search, real-NGSpice candidate
validation, and optional FD.

FD was confirmed to optimize against measured data through fresh NGSpice
parameter perturbations. Published-start FD was unchanged (`0.2750531 ->
0.2750533`, 0/18 wins, `16.2 s`). The exact same raw surrogate winners improved
`0.2357279 -> 0.2296494` on all 18; production top-five FD reached `0.2289864`.
The nonzero published-start Jacobian plus unchanged endpoint establishes that
the trust-region method needs a better basin, rather than FD being skipped.

At the user's request, the fixed direct MLP was subsequently given the same
measured-data FD treatment as a separate paired ablation. Each device used only
its one-pass MLP prediction as the start; every residual evaluation generated
fresh NGSpice curves, and no per-device initializer selection was allowed. It
improved nMOS `0.1283710 -> 0.0876711`, pMOS `0.3873572 -> 0.3489002`, combined
`0.2578641 -> 0.2182856`, and all-device `0.2722522 -> 0.2327984`, with 18/18
accepted improvements. The all-device run took `373.5 s`. The raw MLP remains
the primary one-pass comparison; MLP+FD appears only in the FD study and the
paired FD-improvement slide. Artifacts are `out/pdk_direct_mlp_fd`,
`out/tables/direct_mlp_fd_study.{md,json}`, and
`slides/plots/fd_improvement_comparison.png`.

### Foundation emulator (2026-07-15)

One conditional parameter-to-I-V network was trained once across all 18 known
geometries and reused for each fixed inverse search. Its validation MSE was
`0.000242765`, close to the `0.000214598` mean of the 18 separate emulators.
Foundation raw scored `0.2269044`; +FD scored `0.2260651`. Cache, training, and
18-device search/validation/FD times were `14.5 s`, `189.5 s`, and `996.7 s`,
or `1200.7 s` total versus `2420.2 s` for separate emulators. This is promising
reuse on the trained geometry set, but the random within-geometry validation
split does not prove unseen-L/W generalization. It remains excluded from the
uniform surrogate+FD card export.

### pMOS L=2/W=5 and high-voltage tradeoff (2026-07-15)

The foundation card improves the pMOS L=2/W=5 device mean from paper `0.1907`
to `0.1758`, but worsens high-output `idvd@1.85` from `0.0109` to `0.0470`.
Allowing a different best 10k-LHC card for each voltage produces a nondeployable
included-curve mean `0.1078` and high-output `0.0046`, proving a cross-bias
compact-model compromise. No single sampled candidate improved every included
curve in the seven-parameter +/-10% box.

The guarded study keeps the paper RRMS exactly unchanged. It selects the best
official-RRMS LHC candidate subject to strongest included output/transfer
limits of `max(1.5 * paper curve RRMS, paper curve RRMS + 0.005)`. Every device
had feasible candidates. It scored `0.2327267` all-device and 18/18 wins. For
pMOS L=2/W=5 it traded some device mean (`0.1861`) to restore high-output RRMS
to `0.0153`. It is a selection-policy diagnostic, not a new metric or export.

Training a hybrid signed-log plus normalized-linear emulator could improve
strong-current candidate ranking, but cannot create BSIM degrees of freedom
absent from the chosen seven parameters. Widening the box or adding parameters
would be a separately labeled extension to the reproduction protocol.

### Final verification (2026-07-15)

Twelve unit tests and `compileall` passed. Fresh simulator verification passed
all 198 curves at the pinned ngspice-41 binary and reproduced the documented
`1.0000001e-11 A` absolute and `0.0037594` peak-relative maxima. Setup-data
validation and `git diff --check` passed. Structured artifact checks confirmed
18 current `lhc10` records for direct, surrogate raw, surrogate+FD, foundation,
and guarded series; 18 FD records; exactly 90 accepted scaling cells; and an
18-bin uniform `emu_search+fd` manifest with zero saved-score difference.
Final plots were inspected at original resolution and contain the required
measured, paper-card, and real-NGSpice predicted-parameter curves.

## Superseded mixed-run result (2026-07-14 earlier)

The following result is retained as experiment history. Its surrogate search
included an inverse-network initializer and its inverse rows are no longer
part of current reporting. The clean replacement values are recorded above.

Pinned inputs: `CryoPDK_Skywater130nm_ML@39b1e518`, conda-forge ngspice-41,
updated pFET card, native bins, all 18 Table-6 devices, 10,000 LHC samples per
device in the published +/-10% parameter box, and a faithful port of the
upstream `rrmsCalc.py` scorer. Primary method scores freeze the baseline's
included curves so candidates cannot game simulated-current exclusions.

| fixed method | nMOS | pMOS | combined | all-device | wins vs cards |
|---|---:|---:|---:|---:|---:|
| published cards | 0.1198 | 0.3992 | 0.2595 | 0.2751 | - |
| surrogate search, raw | 0.0828 | 0.3581 | 0.2205 | 0.2358 | 16/18 |
| **surrogate search + FD** | **0.0788** | **0.3491** | **0.2139** | **0.2290** | **18/18** |
| inverse MLP, raw | 0.1670 | 0.6534 | 0.4102 | 0.4372 | 4/18 |
| inverse MLP + FD | 0.0853 | 0.3532 | 0.2192 | 0.2341 | 17/18 |

No row selects a method per device. The extraction runner's diagnostic
best-of chose surrogate+FD 13 times and inverse+FD 5 times, but that diagnostic
is excluded from all headline tables, plots, and export. The selected library
uses the globally best polished fixed method, surrogate+FD, for every device
and native bin.

### FD-polish ablation

- Surrogate inverse search: `0.2357561 -> 0.2289597` all-device mean,
  reduction `0.0067964`.
- One-shot inverse MLP: `0.4372150 -> 0.2341032`, reduction `0.2031118`.

The inverse network is a useful initializer but is poor without real-simulator
polish. Surrogate search is already strong before polish and gains modestly.
The dedicated paired table and plot are `out/tables/fd_polish_ablation.md`
and `figs/fd_polish_ablation.png`.

### Reproduction checks

- All 198 local sweeps match the pinned upstream saved sweeps; worst absolute
  difference `1.0000001e-11 A`. The worst peak-relative difference is
  `0.0037594` on a tiny-current curve (`7.10543e-15 A` absolute).
- Confirmed published-card baseline is `0.2595303` family-combined and
  `0.2750531` all-device, close to the paper's `0.2629125` and `0.2787778`.
- Four fixed-method directories contain all 18 current `lhc10` records.
- Canonical card manifest reports `uniform_method=emu_search+fd`, 18/18 wins,
  and zero difference when re-simulated from the written library.
- A fresh schema-2 confirmed-setup scaling study was started on July 14; see
  `docs/HANDOFF.md` for live status and resume commands.

## Historical June scoreboard

| date | method | mean RRMS | wins vs paper cards | notes |
|---|---|---:|---:|---|
| — | paper reported (HSPICE/Mystic) | 0.279 | — | not comparable: different simulator |
| 2026-06-09 | paper cards in NGSpice (baseline) | 0.6919 | — | published params, native bins; ngspice-46 |
| 2026-06-16 | paper cards in NGSpice-41 (exact ogzamour recipe) | 0.7434 | — | conda-forge ngspice=41; WORSE than -46, not closer to paper |
| 2026-06-09 | ML deployable shared-bin cards | **0.5438** | 17/18 | emulator search + FD polish + joint shared-bin polish |
| 2026-06-10 | fd control (paper-method retry) | 0.5010 | 18/18 | per-device cards; multistart FD least squares |
| 2026-06-10 | **cma control** | **0.4991** | 18/18 | per-device cards; CMA-ES + FD polish — STRONGEST CLASSICAL, the bar ML must beat |
| 2026-06-10 | fd control, deployable | 0.5497 | | one card per model bin (joint polish) |
| 2026-06-10 | cma control, deployable | 0.5445 | | ML v1's 0.5438 already edges this by 0.0007 |
| 2026-06-10 | ml v2, per-device | 0.4917 | 18/18 | beats cma 0.4991 & fd 0.5010; 5 strict wins, 13 ties, 0 losses |
| 2026-06-10 | ml v2, standalone (emu_search+fd, no warm starts) | 0.4938 | | pure-ML pipeline alone also beats both classical controls |
| 2026-06-10 | ml v2, deployable | 0.5368 | 17/18 | beats cma-deploy & fd-deploy |
| 2026-06-10 | cma control, 8500 evals (budget-matched) | 0.4981 | 18/18 | 3.5x budget buys 0.001 → classical plateau |
| 2026-06-10 | cma control deployable (8.5k) | 0.5411 | 17/18 | rebuilt; bin-2 joint 0.684 |
| 2026-06-10 | ml v3 active-BO, per-device | 0.4914 | 18/18 | 3 more strict wins (nmos_L20 0.148, pmos_L0p5_W0p64 0.469, pmos_L8_W1p6 0.566); standalone active_bo+fd 0.4928 |
| 2026-06-10 | **ml FINAL (best of v2/v3), per-device** | **0.4911** | 18/18 | vs strongest classical 0.4981 (-1.4%); 3 strict wins, 15 ties, 0 losses after folding in CMA-8.5k warm starts |
| 2026-06-10 | **ml FINAL, deployable** | **0.5363** | 17/18 | vs strongest classical deployable 0.5411 (-0.9%); out/pdk_ml_final, cards exported |

### 2026-06-10 — ML v2 results (goal met)
Strict per-device wins by the scaled emulator search over the best
classical result: nmos_L20_W0p64 0.150 (vs 0.194), pmos_L0p5_W0p64 0.472
(vs 0.477), pmos_L4_W7 0.609 (vs 0.641), pmos_L8_W1p6 0.569 (vs 0.593),
pmos_L8_W0p84 0.564 (vs 0.565). Joint emulator search improved the bin-2
shared card to 0.690 pair-mean (controls: 0.709/0.726); bin-10 pair stays
~0.876 for every method (real deployability cost). Run config: 6k
samples/device (quarter FD-centered), emu 512x4, inverse [1024,512],
2048 starts x 600 steps, 14 validations, 5 polishes nfev 120, warm starts
from both controls. Wall-clock ~62 min for 18 devices on MPS + 12-core
NGSpice.

Per-device detail: `out/tables/comparison.md` (incl. per-family means),
`out/tables/table6.md`. Family split of the final result (per-device /
deployable): nMOS — classical 0.378/0.378, ml 0.373/0.373; pMOS —
classical 0.594/0.671, ml 0.586/0.667. The deployment cost is entirely
pMOS (both shared bins are pMOS).

## History

### 2026-06-16 — Tried the EXACT ogzamour recipe (ngspice-41): does NOT match the paper
The corrected repo (`ogzamour/CryoSkywater130nm_CorrectedForNgspice`) pins
`conda install -c conda-forge ngspice=41`; this project had been using
ngspice-46 and recorded "ngspice 41 not needed". Tested the pinned version
directly: installed conda-forge `ngspice=41` (osx-64 build via Rosetta, the
same binary the recipe produces) and re-ran the 18-device published-card
baseline.

- **ngspice-41 mean RRMS = 0.7434**, vs ngspice-46 **0.6919**, vs paper
  **0.2788**. ngspice-41 is *worse*, not closer to the paper. The
  HSPICE/Mystic → ngspice gap is **not** a simulator-version artifact.
- Cross-version isolation (`scripts/compare_ng_versions.py`,
  `out/tables/ng41_vs_ng46.json`): on **13/18 devices both versions select
  the same native bin and the per-device RRMS is byte-identical (max diff
  0.0e+00)** — the cryo BSIM4 evaluation is version-independent. All 8 nMOS
  match exactly.
- The only differences are the **5 overlapping-box pMOS devices** where the
  two versions tie-break native bin selection differently (e.g.
  pmos_L0p35_W0p55: -46 → bin 10 = 0.931, -41 → bin 5 = 3.070; pmos_L8_W5:
  -46 → bin 2 = 1.458, -41 → bin 1 = 0.852). This reconfirms that "native
  bin selection" is itself ambiguous for the overlapping pMOS boxes (the
  reason we force per-device best bin elsewhere) — it is now shown to be
  *version-dependent*, not just bin-overlap-dependent.
- `verify_simulator.py` under ngspice-41 reproduces the corrected repo's
  saved reference sweeps to the same 3.6e-7 A as ngspice-46 — i.e. both our
  ngspice builds reproduce ogzamour's saved ngspice-41 output identically.
- Conclusion: stay on ngspice-46. The paper's 0.279 remains unreachable
  through any open NGSpice (41 or 46); method-vs-method inside the identical
  chain remains the only fair comparison. ngspice-41 saved to
  `/Users/anrunw/cryo-ng41/mm/envs/ng41/bin/ngspice`; ng46 baseline preserved
  in `out/pdk_baseline`, ng41 baseline in `out/pdk_baseline_ng41`.

### 2026-06-16 — Floor reconciliation: the paper gap is the METRIC, not the sim
Recomputed every method both all-curve and with a device-off floor (curves
whose mean|I| < frac*device-peak counted as 0, applied identically to all),
from each method's saved sims (`scripts/floored_comparison.py`,
`out/tables/floored_comparison.json`):

| method | all-curve | floor 1% | floor 2% | floor 5% |
|---|---:|---:|---:|---:|
| published baseline | 0.692 | 0.181 | 0.166 | 0.141 |
| fd control | 0.501 | 0.098 | 0.090 | 0.078 |
| cma-8500 control | 0.498 | 0.102 | 0.094 | 0.082 |
| ml_final (search) | 0.491 | 0.098 | 0.090 | 0.078 |
| tandem seed0 | 0.494 | 0.104 | 0.096 | 0.083 |
| paper reported | 0.279 | — | — | — |

- Floored, even the published baseline (0.14-0.18) drops BELOW the paper's
  0.279. The all-curve 0.692 is inflated by near-off curves: across all 18
  devices' near-off curves (9,672 pts) **41% are hard 0, plus SMU glitch
  spikes up to 51 uA** that no BSIM4 card can fit. So the "0.69 vs 0.28" gap
  is the all-curve weighting of unfittable noise-floor curves, not HSPICE-vs-
  NGSpice and not ngspice-41-vs-46. (Consistent with the long-known best-bin
  median-floor ~0.282.) Near-off curves are padded to 0 by the instrument,
  NOT to a high sentinel; what blows up is RRMS = rmse/mean|I| (tiny denom).
- The ML win is ROBUST to the metric: under both all-curve and floored,
  ml_final stays at/tied-for best (0.090 floored), still ~2x better than the
  baseline. The advantage is not an artifact of all-curve weighting.
- Caveat: "count near-off as 0" can be gamed (a device with many off curves
  scores low for free), so it is reported ALONGSIDE all-curve, never instead,
  and applied identically to every method.

### 2026-06-16 — Direct IV->params amortized extractor (no-surrogate + surrogate)
New `scripts/pdk_forward_direct.py`: a one-shot curve->7-param network (the
classic amortized inverse / iPREFER-style extractor), trained two ways and
applied once to the measured curve, every prediction NGSpice-validated.
Loss = pure parameter distance (z-MSE); `--recon-weight>0` adds a frozen-
emulator curve-reconstruction term. Early stopping + z-clamp to the inner box
(off-manifold measured inputs otherwise extrapolate to pathological corners,
e.g. 905x current). Compared in `scripts/forward_compare.py`
(`out/tables/forward_compare.json`):

| metric (median/18) | no-surrogate | with-surrogate |
|---|---:|---:|
| synthetic in-dist reconstruction | 0.190 | 0.134 |
| measured one-shot | 1.39 | 0.768 |
| measured one-shot (mean) | 52.8 | 3.95 |
| measured + FD polish | 0.578 | 0.545 |
| measured + FD polish (mean) | 0.579 | 0.516 |

- **In-distribution (synthetic) reconstruction is excellent** (median ~0.13-
  0.19; 14/18 devices < 0.3) — reproduces the literature result. The 4 high
  values (nmos_L0p15, nmos_L8, pmos_L0p35_W5, pmos_L4_W7) are the low-current
  near-threshold devices where RRMS is ill-conditioned, not model failures.
- **Adding the surrogate reconstruction loss helps on every aggregate metric**
  and rescues the catastrophic measured blow-ups (nmos_L1_W1p6 905->0.87,
  pmos_L0p35_W1p6 15.3->4.3) by anchoring predictions to curve-space.
- **But pure one-shot is not usable for extraction on measured cryo data**
  (median 0.77 even with surrogate vs the search pipeline's 0.491): the
  measured curves sit off the clean synthetic manifold (glitch spikes,
  quantization, real leakage outside the 7 params). FD polish from the
  with-surrogate start reaches 0.516 mean, near the controls but not beating
  NGSpice-in-the-loop search. This is exactly why the winning pipeline keeps
  NGSpice in the loop rather than amortized inference.

### 2026-06-16 — Tandem training continued (seeds 3-5): converged to the floor
Ran 3 more confirmatory tandem seeds (identical config, only --seed differs).
6-seed per-device ensemble (min over NGSpice-validated cards) = 0.4921 (was
0.4928 at 3 seeds); best-of-all (tandem ensemble + ml_final) = 0.4910 ≈ the
0.4911 ml_final. The extra seeds tightened but did not break the data/model
floor — confirms the four method families converge ~0.49.
(`scripts/tandem_ensemble.py`, `out/tables/tandem_ensemble.json`.)

### 2026-06-09 — NGSpice binning blocker solved
The long-standing "ngspice can't bin the cryo corner files" failure was a
scale-units issue: corner files set `.option scale=1.0u`, so X-instances
must pass bare micron numbers (`l=0.15 w=1.6`, no `u` suffix). Working
backend: `src/cryoml/spice_pdk.py` (ngspice-46, Volare sky130
`a918dc7c...`). pMOS bin boxes overlap (12 bins / 10 devices), so auto-bin
can pick a bin fit to a different device.

### 2026-06-09 — Baseline validated, old claims audited
Published cards through the PDK chain with per-device *best* bin: 0.2823
mean median-floor RRMS vs paper-reported 0.279 (within 1.2% → benchmark copy
is good). Exact reproduction impossible for pmos 0.35/1.6, nmos 20/0.64,
nmos 100/100 (~2x off even best-bin) — genuine HSPICE-vs-NGSpice gap. An
older "0.221 beats 0.279" claim was retracted: cross-simulator comparison +
over-aggressive curve filtering. Current fairness rules (native bins, all-curve
metric, same simulator both sides) date from this audit.

### 2026-06-09 — ML pipeline v1 (current best: 0.5438)
Per device: ~1,500 NGSpice samples in the native bin → MLP emulator
(z → signed-log curves) + inverse MLP (curves → z) → multistart Adam search
through the frozen emulator (|z| ≤ 3.5) → top candidates re-scored in real
NGSpice → FD least-squares polish → devices sharing a model bin jointly
polished into one card. Stage means over 18 devices:

| stage | mean RRMS |
|---|---:|
| paper cards | 0.6919 |
| inverse MLP raw | 390.19 (unusable alone) |
| inverse MLP + FD polish | 0.6159 |
| emulator search | 0.6011 |
| emulator search + FD polish | 0.5020 (per-device best, not deployable) |
| deployable shared-bin cards | **0.5438** |

Worst devices after v1: pmos_L0p5_W0p42 1.181 (regressed vs 1.098),
nmos_L0p25_W1p6 0.913, pmos_L2_W5 0.963, pmos_L0p35_W1p6 0.754,
pmos_L8_W5 0.739.

### 2026-06-10 — Figures, README, slides
`scripts/make_figs.py` reproduces paper Figs. 2/4/5 + Table 6 with
measured vs paper-cards vs ML overlays; `scripts/make_slides.py` builds
`slides/cryo-ml-77k.pptx` (14 slides, non-ML audience).

### 2026-06-10 — Classical controls launched (paper-method retry)
`pdk_fd_extract.py --workers 8` running: 6-start FD least squares per
device, max_nfev=150, warm start at published params — the closest analogue
of the paper's extraction process inside our simulator. CMA-ES control
queued behind it.

### 2026-06-10 — Diagnostics: where the error actually lives
Per-curve RRMS breakdown of ML v1 sims (all 18 devices):

- The device score is dominated by the **lowest-|Vg| output curves**
  (`idvd @ |Vg|=0.37`, near/below threshold): e.g. nmos_L0p25_W1p6 — that
  single curve scores 9.61 and contributes 0.87 of the device's 0.913;
  pmos_L2_W5 — two such curves are 77% of the error.
- **Emulator val-MSE does not correlate with final RRMS** (best emulator
  0.026 on a 0.576 device; worst 0.92 on a 0.058 device) → surrogate
  capacity is not the current bottleneck; the objective landscape is.
- Killer curves split three ways:
  1. *Spike-corrupted off-curves* (median 0, isolated 6 µA codes):
     irreducible, RRMS ≈ 1/sqrt(spike fraction) for any card.
  2. *Quantization floors*: measured currents quantized at ~6 µA (some
     ranges) — one sweep reads 0 where another reads 6 µA at the same bias.
  3. *Real high-Vd leakage floors* (nmos_L100_W100: flat ~70–83 µA for
     Vg=0.3–0.6 at Vd=1.85, self-consistent across sweep families;
     also nmos_L20_W0p64, pmos_L0p35_W0p55, pmos_L0p5 pair): GIDL-like,
     controlled by BSIM4 params outside the allowed 7, but partially
     mimicable via large ETA0 (DIBL) within bounds. **This is real,
     fittable headroom that no method has exploited yet** — every card
     (published, FD, ML v1) sits 3–4 decades below these floors.
- Parameter pinning at box edges is rare (vsat/rdsw on 4 devices) — bounds
  are mostly not the constraint.

### 2026-06-10 — FD control interim (13/18 devices)
FD (paper-method retry) is nearly matching ML v1 per-device: ML ahead by
0.002–0.05 on most devices, FD ahead on nmos_L0p19_W7 (0.429 vs 0.430) and
pmos_L2_W5 (0.945 vs 0.963). Implication: ML v1's edge over the classical
method is thin; the planned scaling experiments must target the leakage-floor
basins (item 3 above) where global search can win.

### 2026-06-10 — Pipeline audit + scaling prep (ML v2 design)
Audit of v1: emulator MLP [7,256,256,256,P~2046], inverse [P,512,256,7],
512 Adam starts x 400 steps, 8 NGSpice validations, 3 FD polishes (nfev 80),
~1.5k synth samples/device.

Probes:
- ETA0 (DIBL) is numerically inert for long channels (BSIM4 cosh(L/lt)
  term) → nmos_L100_W100's high-Vd leakage floor is unfittable within the
  7 params; bounded headroom ~0.03 there. Short-channel pMOS DIBL is live.
- pmos_L2_W5 over-conducts at Vg=-0.74 (sim nA vs meas ~0) → needs VTH0
  shift, opposite direction from the leakage devices.
- pmos_L0p5_W0p42 per-device ML was already 0.525 (beats FD 0.526) but the
  shared-bin joint polish wrecked it to 1.181; joint optimum for the
  (0p35_W0p55, 0p42) bin-10 pair sits ~0.875 pair-mean even with a new
  joint emulator search → genuine deployability cost, not a search bug.
- Emulator arch A/B (held-out val MSE): 512x4 ≈ 3-4x better than v1 256x3
  on 2/3 testbed devices; capacity chosen for v2: emu 512x4, inverse
  [1024,512].

v2 pipeline upgrades implemented in `pdk_ml_extract.py`:
1. `--emu-arch/--inv-arch` flags; minibatch path for >4k samples.
2. `--starts-from out/pdk_fd[,out/pdk_cma]`: control winners become warm
   starts AND directly-validated candidates → ML ≥ controls per device by
   construction.
3. Trained emulators saved per device; new **joint emulator gradient
   search** on the summed objective now seeds the shared-bin joint polish.
4. Training data v2 (`pdk_gen_data.py --centers-from out/pdk_fd`): 6000
   samples/device, quarter tight around published, quarter around the FD
   winner, quarter wide, quarter uniform box. (~5 min/device, 10 workers.)
5. `make_fd_deploy.py`: FD control under the same shared-bin deployment
   constraint, so deployable-vs-deployable is method-fair.

### 2026-06-10 — Data race found & fixed: 4 devices re-run
Timestamp cross-check showed the v2 extraction overtook the slower second
wave of data generation: pmos_L0p35_W0p55, pmos_L0p5_W0p42, pmos_L0p5_W0p64
and pmos_L8_W1p6 trained on the OLD 1.5k data (results still valid — all
NGSpice-scored — but without the v2 data benefit). Also: the FD-centered
sampling has a much lower ok-rate on these devices (67-95% vs 100%),
because FD winners near box edges produce more failing sims. Re-ran those
4 devices on the full 6k data (out/pdk_ml2 backed up to out/pdk_ml2_bak;
joint shared-bin cards redone — bin-2 group restored from backup if the
redo regresses).

### 2026-06-10 — Budget-matched CMA: classical method has plateaued
CMA-ES with 8,500 evals/device (≈ ML's total simulation budget, 3.5x the
regular control): mean **0.4981** vs 0.4991 at 2,400 evals. Tripling the
classical budget buys 0.001 — the ML margin cannot be attributed to
simulation budget. NOTE: this run replaced out/pdk_cma (the 2,400-eval
per-device records); the 0.4991 summary survives here and in git-less
history only. cma_deploy rebuilt from the 8.5k-eval results.

### 2026-06-10 — New goal: best possible ML vs control + scaling laws
New method implemented (`pdk_ml_active.py`, ML v3 = **ensemble active-BO**):
K=3 emulator ensemble, pessimistic acquisition (mean+std; half the starts
optimistic mean−std for disagreement-driven exploration), 4 active rounds
of search → NGSpice-validate top 8 → append TRUE evals to training set →
fine-tune ensemble. Targets the diagnosed v2 bottleneck (surrogate
exploitation error). Warm starts: fd, cma, ml2 winners (validated round 0
→ incumbent never worse than best known). Round-3 data appended first:
+2k samples/device densified around ML v2 winners (pool now 8k).

Also queued for rigor/science:
- Budget-matched CMA control (8,500 evals/device ≈ ML's total sim budget)
  so the ML win can't be attributed to budget alone.
- Scaling-law study (`scaling_study.py` → `make_scaling_fig.py`): data
  {375..6000} / capacity {64x3..1024x4} / search {128..8192 starts} sweeps
  on 4 testbed devices, standalone pipeline, fixed round-2 pool.
- Figures: log-scale I-V panels removed per user request (fig2 twin axis,
  fig4 row 2, appendix); slides to be refreshed at the end.

### 2026-06-10 — Scaling laws (out/scaling/results.csv, figs/scaling_laws.png)
52-cell sweep, standalone pipeline, 4 testbed devices, single seed:
- **Training data is the dominant axis**: final polished RRMS geo-mean
  0.66 (n=375) → 0.30 (n=6000), with catastrophic failures below ~1k
  samples on some devices (nmos_L20: 1.28 at n=375 vs 0.15 at full data);
  saturates by ~3-6k.
- **Capacity**: emulator val MSE follows a clean power law, ~params^-0.25
  (0.114 at 142k params → 0.025 at 5.3M); final RRMS improves modestly all
  the way to 1024x4 (0.49 → 0.41 on the testbed) — some headroom may
  remain beyond 512x4.
- **Search budget is saturated**: surrogate search loss flat from 128 →
  8192 starts (α≈0.02); consistent with v2→v3 gains coming from data and
  acquisition design, not more search.
- Final extraction error decouples from surrogate quality once the
  surrogate is good enough — the measurement/model floor takes over.

## Literature notes (2026-06-10)

ML-driven compact-model extraction is an active area; our contribution is
the cryogenic + open-simulator-verified setting, not the NN machinery:

- iPREFER (arXiv:2404.07827): feature-based NN parameter extractor for
  BSIM-CMG — amortized inverse like our inverse MLP.
- "A single neural network global I-V and C-V parameter extractor for
  BSIM-CMG" (Solid-State Electronics, 2024): one NN predicting params from
  curves across geometries; motivates a future multi-device inverse net.
- "Scientific machine learning for generic compact model parameter
  extraction" (Eng. Appl. AI, 2025): random-forest + ANN inverse modeling.
- "ML-Based Standard Compact Model Binning Parameter Extraction" (Adv.
  Intelligent Systems, 2026): ML for bin-level extraction — adjacent to our
  shared-bin joint card problem.
- None target cryogenic cards or score across two simulators; the
  simulator-verified (NGSpice-in-the-loop) candidate validation + classical
  controls protocol appears to be our differentiator.

## Experiment queue (ML scaling)

- [x] Audit architectures + budgets; A/B emulator capacity (512x4 wins)
- [x] More synthetic training data (1.5k → 6k, FD-centered quarter)
- [ ] ML v2 full run (out/pdk_ml2): emu 512x4, 2048 starts x 600 steps,
      14 validations, 5 polishes nfev 120, warm starts from FD **and CMA**
      (relaunched after CMA finished at 0.4991)
- [x] CMA-ES control (out/pdk_cma): **0.4991**, 18/18 vs published; beats
      FD control on 11/18 devices; notable basins ML v1 missed:
      pmos_L8_W5 0.648, nmos_L20_W0p64 0.194, nmos_L0p15 0.038
- [ ] FD-deployable control (out/pdk_fd_deploy)
- [ ] If v2 insufficient: emulator ensembles, round-3 active sampling
      around v2 winners, stronger inverse (curve-embedding arch)

## 15-parameter extraction experiment (2026-08-05)

First run of the configurable-theta mechanism added on the `parameters`
branch (`CRYOML_PARAM_SET`, commit 17dd813). Same confirmed setup, box,
metric, freeze rules, and production search recipe as the canonical
campaign; only the theta vector changes, from the paper's 7 to 15 BSIM4
parameters: + `pclm, pdiblc1, pdiblc2, ags, ua, ub, voff, prwg`.

- A showmod survey of all 18 bins drove the final list: `pvag` (originally
  proposed) is zero-published on every bin and hence dead inside the ±10%
  multiplicative box; `ags` took its slot. `pdiblc1`/`prwg` are
  zero-published on the pMOS/nMOS family respectively and stay pinned
  there, so each device searches 14 live dimensions.
- Patched-published z=0 cards reproduce the bare published card to ~1e-6
  relative current on both families, so box centers are trustworthy.
- Data: 10,000-sample LHC per device, all 11 curves, real ngspice-41
  (`data/processed/pdk_synth_params15/`, ~2.3 GB, regenerable).
- Results (frozen-inclusion): **nMOS 0.0746, pMOS 0.3260, combined
  0.2003, all-device 0.2143, 18/18 wins vs the published card** — better
  than the canonical 7-param surrogate+FD (0.2140/0.2290) and the
  exploratory foundation+FD (0.2114/0.2261) on every aggregate.
- Raw 15-D surrogate search (0.2699 all-device) is worse than raw 7-D
  (0.2357): same sample budget, harder search space. FD polish recovers
  it (0.2699 → 0.2143) — the surrogate only needs to find the basin.
- 67/270 winning values (25%) peg a ±10% box edge (`delta` 11, `ub` 9,
  `ags` 8, `ua` 8 devices): the data wants several parameters outside
  ±10% of published. Follow-up candidate: selectively wider box.
- Status: labeled experiment per protocol. Not the canonical export; no
  Table 4/6 or figure changes. Full report:
  `out/tables/params15_study.{md,json,csv}`.
