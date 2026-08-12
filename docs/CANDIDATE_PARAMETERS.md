# Candidate BSIM4 parameters beyond the current 15

Design note, 2026-08-12. **Status: analysis + measured screen.** No protocol,
metric, or card change. Companion to `docs/MODEL_EXPANSION.md`, which covers
the general method; this note answers the concrete question — *which*
parameters, and in what order.

**Revision note.** An earlier draft of this note ranked candidates from BSIM4
equation structure alone. Steps A and B below have now been run, and the
measurement **overturned several of those predictions**, including the
headline recommendation. Section 2 is the measured result and supersedes the
predicted ranking; §2.5 records what was wrong and why, because the failure
modes are instructive.

## Data behind this note

- `out/tables/candidate_showmod_survey.json` — Step A: resolved values for 61
  parameters on all 18 native bins.
- `out/tables/candidate_sensitivity.{json,csv}` — Step B: 244 one-at-a-time
  perturbation records over 4 devices.
- `scripts/make_candidate_figs.py` → `figs/candidate_*.png` (light and dark):
  `candidate_sensitivity` (§2), `candidate_length_dependence` (§0.6),
  `candidate_vsat` (§0.2), `candidate_box_locked` (§0.5).
- Simulator: conda-forge ngspice-41, `temp=-196.15`, native bins, frozen
  published-card curve inclusion. Runtime 26 s per device pair.

Two sensitivity numbers per parameter per device:

- **`d_rrms_pm10`** — max |ΔRRMS| over a ±10% multiplicative perturbation, i.e.
  sensitivity *within your existing box*. This is the ranking quantity.
- **`maxrel_large`** — max |ΔI| / max|I| under a large perturbation (3× the
  published value, or an absolute magnitude when published is zero). This
  separates "inert" from "frozen at zero by the box."

## 0. Findings that reshape the question

### 0.1 `showmod` exposes everything — there is no readback gate

All 61 candidates were echoed on all 18 bins, so `read_bin_params()` will not
raise for any parameter considered here. Expression-valued entries
(`dvt0={2.4422*dvt0_nom}`, the `MC_MM_SWITCH` forms on `vth0`/`nfactor`/
`voff`/`toxe`) resolve correctly because the readback runs NGSpice rather than
parsing text. **Any text-based screen will mislabel those four as zero** — they
are not.

### 0.2 The published card is non-physical in `vsat`, and the measurement
confirms the dimension is wasted

`showmod`-resolved `vsat` for the 18 studied devices (m/s). An earlier draft
listed these in card-file order, which is **not** bin order, so values were
misassigned to devices; these are the survey-authoritative numbers:

```
nMOS   L0.15/W1.6  1.378e5    L0.19/W7    2.309e5    L0.25/W1.6  1.522e5
       L1/W1.6     3.816e4    L1/W3      -3.379      L8/W1.6     3.965e4
       L20/W0.64   2.360e4    L100/W100   6.538e3
pMOS   L0.35/W0.55 -2.586e4   L0.35/W1.6  1.028e9    L0.35/W5    3.091e9
       L0.5/W0.42  -4.914e3   L0.5/W0.64  1.490e9    L2/W5       6.704e6
       L4/W7        7.393e8   L8/W0.84    6.106e8    L8/W1.6     5.451e8
       L8/W5        7.425e8
```

**Nine of the ten pMOS devices are non-physical**: seven carry `vsat` above the
speed of light (up to 3.1×10⁹ m/s, ten times c) and two are negative. One nMOS
device (L=1/W=3) is negative. Only `pmos_L2_W5` stays below the speed of light
with a positive value, and at 6.7×10⁶ m/s even that is ~70× a plausible
saturation velocity. All three probed non-physical devices measure inert:

| device | published `vsat` | `d_rrms_pm10` |
|---|---:|---:|
| nmos_L0p15_W1p6 | 1.378e5 — physical | **2.69e−02** |
| nmos_L1_W3 | −3.379 — non-physical | 8.45e−07 |
| pmos_L0p35_W1p6 | 1.028e9 — non-physical | 1.84e−07 |
| pmos_L4_W7 | 7.393e8 — non-physical | 5.07e−07 |

Where `vsat` is physical and the channel is short, it is a strong knob
(2.7e−02, comparable to `u0`). Where it is non-physical, perturbing it does
essentially nothing — an enormous `VSAT` has pushed velocity saturation out of
the operating range, so ±10% around 7×10⁸ m/s changes no current.

**`vsat` is therefore a dead slot in your 7- and 15-parameter vectors on nine
of the ten pMOS devices**, and pMOS is where your residual concentrates
(0.3260 vs nMOS 0.0746). This is the single most actionable finding in the
note: it costs nothing to test, and it is a candidate explanation for a large,
persistent, family-specific gap.

→ `figs/candidate_vsat.png`

### 0.3 Zero body bias — mostly confirmed, with one important exception

The deck ties source and bulk to ground (`Xm1 nd ng 0 0`, only `VG`/`VD`
sources), so **V_bs ≡ 0** everywhere, with `mobmod=0` and `rdsmod=0`.

Measured max `d_rrms_pm10` across all four devices for the body-bias family:

| parameter | max `d_rrms_pm10` | verdict |
|---|---:|---|
| `etab` | 8.07e−08 | negligible — confirmed |
| `dvt2` | 9.97e−09 | negligible — confirmed |
| `cdscb` | 8.40e−10 | negligible — confirmed |
| `prwb` | 3.76e−09 | negligible — confirmed |
| `pdiblcb` | 8.81e−11 | negligible — confirmed |
| `uc` | 2.81e−08 | negligible — confirmed |
| `dwb` | 9.96e−09 | negligible — confirmed |
| `keta` | 2.54e−07 | negligible — confirmed |
| `k2` | **3.25e−03** | **moderate — prediction wrong** |
| `k3b` | 3.38e−05 | weak — prediction wrong |
| `k1` | **6.08e−01** | **STRONG — prediction wrong** |

Eight of eleven confirmed. But `k1` is not merely non-inert, it is the
**third most sensitive parameter measured anywhere in this study**. The
reasoning error: `K1` does not only appear in the Vth body term
`K1(√(Φs−Vbs) − √Φs)`, which does vanish at `Vbs = 0`. It also enters the
bulk-charge factor `Abulk` through `K1ox/(2√Φs)`, and `Abulk` multiplies `Vds`
throughout the triode and saturation expressions. So `K1` shapes the entire
output characteristic independently of body bias. `K2` and `K3B` have smaller
secondary paths of the same kind.

**`k1` is now a top candidate.** The rest of the body-bias block stays
excluded, and body-bias sweeps remain the cheapest measurement that would
unlock it.

### 0.4 Parameters gated off by model switches are exactly inert

Nine parameters produce **bit-identical** current under a large perturbation on
every device tested:

| parameter(s) | reason |
|---|---|
| `rsw`, `rdw` | unused when `rdsmod=0` (internal `Rds` from `RDSW`/`PRWG`) |
| **`eu`** | **unused when `mobmod=0`** — `EU` is a mobMod=2 parameter |
| `bgidl`, `cgidl`, `egidl`, `agidl` | GIDL block off; forcing `agidl=1e−9` still gives zero |
| `pvag`, `pdits`, `pditsd`, `alpha1` | published null/zero and no path to `Ids` |

`eu` was in the previous draft's Tier 1. It is inert for a structural reason I
missed: with `mobmod=0` the mobility denominator has no `EU` exponent at all.
This is the same class of error as `rsw`/`rdw`, which I did catch — the lesson
is to check the model switches for every candidate, not just the suspicious
ones.

### 0.5 Five parameters are physically active but locked out by the box

This is the strongest argument in the note for the box redesign. These have
published value **0** on at least one family, so `LhcBox` freezes them
(`frac × 0 = 0`), yet an absolute perturbation moves the current substantially:

| parameter | device | test value | `maxrel_large` |
|---|---|---:|---:|
| `b0` | nmos_L0p15_W1p6 | 5.42e−06 | **0.327** |
| `a0` | nmos_L0p15_W1p6 | 6.588 | **0.244** |
| `minv` | pmos_L0p35_W1p6 | 0.5 | **0.238** |
| `a1` | nmos_L1_W3 | 0.1 | **0.225** |
| `voffl` | nmos_L0p15_W1p6 | 1e−09 | **0.169** |
| `pdiblc1` | pmos_L0p35_W1p6 | 0.969 | **0.105** |
| `prwg` | nmos_L1_W3 | 0.052 | **0.023** |

→ `figs/candidate_box_locked.png`

A 10–33% current change from a parameter your search cannot reach. `pdiblc1`
is the pointed case: it is in your existing 15, live on nMOS, and **published
zero on every pMOS bin** — so on the family where your residual is worst, one
of your fifteen dimensions is frozen while being demonstrably worth 10% of the
current. An additive fallback for zero-published values recovers this without
adding a single new parameter.

### 0.6 Sensitivity is strongly channel-length dependent

The first Step-B run used only long-channel devices (nMOS L=1 µm, pMOS
L=4 µm), which made every short-channel parameter look inert. Re-running on
short-channel devices changed the picture completely:

| parameter | nmos L=0.15 | nmos L=1 | ratio |
|---|---:|---:|---:|
| `dsub` | **1.32e−01** | 2.30e−12 | ~10¹¹ |
| `eta0` | **1.24e−01** | 0.00 | ∞ |
| `dvt1` | **7.84e−02** | 3.18e−04 | 247× |
| `dvt0` | **2.89e−02** | 6.52e−05 | 443× |
| `pclm` | **8.64e−02** | 4.95e−04 | 175× |

The SCE/DIBL terms enter through `exp(−DROUT·Leff/2ltw)`-type factors that
underflow to exactly zero at long channel. → `figs/candidate_length_dependence.png`

**Any screen must be run per
device**, and a single "global" candidate ranking is not meaningful for this
device set — you have L spanning 0.15 µm to 100 µm.

## 1. Can you train beyond 15?

Yes. params15 already proves the mechanism (all-device 0.2143 vs 0.2290,
18/18 wins vs published). The measured screen supports going further, but
redirects where the headroom is:

- **Roughly 20 parameters clear a `d_rrms_pm10` > 1e−3 bar** on at least one
  device, versus 15 in use — and several currently-used slots (`vsat` on most
  pMOS bins, `pdiblc1` on all pMOS bins) are inert or frozen, so your
  *effective* dimension is already below 15.
- The binding constraint is not parameter count but **box shape** (§0.5) and
  **per-device applicability** (§0.6).

My estimate stands at **25–30 well-identified parameters**, but with the
correction that this must be a *per-device* count with a device-specific
active set, not one global list.

## 2. The measured candidate list

Ranked by max `d_rrms_pm10` across the four probed devices. **Sensitivity is
necessary, not sufficient** — a highly sensitive parameter that is collinear
with one already in the set adds nothing. §3 gives the collinearity map.

→ `figs/candidate_sensitivity.png`

### 2.1 Tier 1 — strong, and not already in the set

| parameter | best `d_rrms_pm10` | where it acts | note |
|---|---:|---|---|
| `k1` | **6.08e−01** (nmos L=0.15) | body charge → `Abulk`, all regions | strongest new candidate; see §0.3 |
| `dsub` | **1.32e−01** (nmos L=0.15) | subthreshold DIBL | short-channel only |
| `dvt1` | **1.31e−01** (pmos L=0.35) | SCE V_th roll-off | short-channel only |
| `a2` | **4.38e−02** (nmos L=0.15) | V_dsat knee | nMOS only (pMOS ~1e−7) |
| `b0` | **4.18e−02** (pmos L=0.35) | narrow-width bulk charge | needs additive box on nMOS |
| `dvt0` | **2.89e−02** (nmos L=0.15) | SCE V_th roll-off | pick one of `dvt0`/`dvt1` |
| `a0` | **2.36e−02** (pmos L=4) | bulk charge | zero on one nMOS bin |
| `lint` | **1.31e−02** (nmos L=0.15) | effective length offset | see collinearity warning |
| `wr` | **9.42e−02** (pmos L=4) | `Rds` width exponent | see collinearity warning |

### 2.2 Tier 2 — moderate

`k2` (3.25e−03), `k3` (1.59e−03), `wint` (1.48e−03). All real but an order of
magnitude below Tier 1.

### 2.3 Restore before adding — free wins inside the existing 15

1. **Re-center `vsat`** on the bins where it is non-physical (§0.2). Recovers a
   dimension you are already paying for, on the family that needs it most.
2. **Additive box fallback for `pdiblc1`** on pMOS (§0.5) — worth ~10% of the
   current on short-channel pMOS, currently frozen.
3. **Additive fallback for `prwg`** on nMOS, same reason.

These three cost no new parameters and address the family with the worst
residual. Do them before expanding.

### 2.4 Excluded, with measured justification

| parameter(s) | max `d_rrms_pm10` | reason |
|---|---:|---|
| `eu`, `rsw`, `rdw`, `pvag`, `pdits`, `pditsd`, `alpha1`, `agidl`, `bgidl`, `cgidl`, `egidl` | **0.00** | inert — model-switch gated or no path to `Ids` (§0.4) |
| `keta`, `etab`, `dvt2`, `cdscb`, `prwb`, `pdiblcb`, `uc`, `dwb` | < 3e−07 | body-bias only, `V_bs ≡ 0` (§0.3) |
| `drout` | 1.68e−05 | **weak even at L=0.15 µm** — see §2.5 |
| `pscbe1`, `pscbe2` | < 1.5e−08 | negligible |
| `cdsc`, `cdscd`, `cit` | < 1.6e−05 | negligible; also collinear with `nfactor` |
| `alpha0`, `beta0` | < 1.4e−05 | drive substrate current, not `Ids` |
| `w0`, `k3b`, `dwg` | < 5.3e−04 | weak |
| `toxe`, `ndep`, `xj` | not probed | process constants, degenerate with `vth0` — `MODEL_EXPANSION.md` §3.2 |

### 2.5 Predictions the measurement overturned

Recorded because the failure modes generalize:

| prediction | measured | why the prediction failed |
|---|---|---|
| `drout` — "highest expected value; if only one parameter is added, make it this one" | 1.68e−05, weak on **all four** devices | I reasoned from which *mechanism* it belongs to (output conductance, where params15 gained) without checking whether this card's `drout` actually moves current. It sits inside a saturating exponential; `pdiblc2` already covers the same effect. |
| `eu` — Tier 1 | exactly inert | `mobmod=0` does not use `EU`. Switch-gating must be checked per parameter. |
| `pscbe2` — Tier 1 | 3.1e−10 | mechanism present in the model, but negligible at these biases. |
| `k1`, `k2`, `k3b` — "structurally unidentifiable" | `k1` **6.08e−01** | equation reasoning was incomplete: `K1` enters `Abulk`, not just the Vth body term. |
| `cdscd`, `cit` — Tier 2 | < 1.6e−05 | real mechanisms, negligible magnitude. |
| `wr`, `lint` — Tier 3 "degenerate" | 9.4e−02, 1.3e−02 | sensitivity is high; the degeneracy claim was about *identifiability* and remains untested — see §3. |

The pattern: **mechanism membership does not predict sensitivity.** Every one
of these errors would have been caught by exactly the screen that has now been
run, which is the argument for running it before each expansion rather than
reasoning from the parameter's role.

## 3. Collinearity — the untested half

Sensitivity screening is done; **identifiability screening is not**. High
`d_rrms_pm10` with high collinearity is a trap: the parameter moves the curves
but along a direction another parameter already covers, which is what produces
flat optimizer directions and multistart disagreement.

Blocks to test before freeing more than one member:

- **Threshold** — `vth0` *(in set)*, `k1`, `dvt0`, `dvt1`. All four are now
  high-sensitivity, and all four shift `V_th`. This is the highest-risk block:
  expect strong collinearity, and free at most two.
- **Bulk charge / V_dsat** — `ags` *(in set)*, `a0`, `a1`, `a2`, `b0`, `k1`.
  `k1` appears here as well as in the threshold block, so it couples the two.
- **Series resistance / geometry** — `rdsw` *(in set)*, `wr`, `lint`, `wint`.
  For a *single* device, `(1e6·Weff)^WR` is a constant multiplier, so `wr` is
  expected to be near-exactly degenerate with `rdsw`; `lint`/`wint` shift
  `Leff`/`Weff` and trade against `u0`/`rdsw`. Their high sensitivity does not
  contradict this — it is what degeneracy looks like before you test for it.
- **Subthreshold** — `nfactor`, `voff` *(in set)*, `minv`, `voffl`, `cdsc`,
  `cit`. `minv` and `voffl` are the box-locked pair worth unlocking; the rest
  are negligible anyway.

The screen: build the normalized sensitivity matrix from the FD Jacobian
`fd_polish` already computes, and reject subsets with collinearity index
`γ = 1/σ_min(S̃) > 10–20` (Brun, Reichert & Künsch 2001).

## 4. Recommended sequencing

1. **Fix the box first** — additive fallback for zero-published values, per
   parameter ranges, log spacing. §0.5 shows this unlocks `pdiblc1`, `prwg`,
   `a0`, `b0`, `a1`, `minv`, `voffl` with no new dimensions.
2. **Re-center `vsat`** where non-physical (§0.2) and re-score pMOS.
3. **Run the collinearity screen** (§3) on the Tier 1 set. Do not add
   `k1` + `dvt0` + `dvt1` together before this.
4. **Expand per device, not globally** — short-channel devices get
   `dsub`/`dvt0`/`dvt1`; long-channel devices get none of them and should
   spend their budget on `k1`, `a0`, `a2`, mobility.
5. **Re-run the screen after each expansion.** It costs 26 s.

## 5. Reproducing

```bash
export NGSPICE_BIN=/opt/homebrew/Caskroom/miniconda/base/envs/ng41/bin/ngspice
export PYTHONPATH=src
# Step A: showmod survey, 18 bins, ~2 s
# Step B: one-at-a-time sensitivity, 2 devices per invocation, ~26 s
.venv/bin/python <probe> long    # nmos_L1_W3, pmos_L4_W7
.venv/bin/python <probe> short   # nmos_L0p15_W1p6, pmos_L0p35_W1p6
```

The probe script lives in the session scratchpad and is **not** committed —
it is a diagnostic, not a pipeline stage. Promote it to `scripts/` if the
screen becomes a standing part of the expansion workflow.

## Open questions

1. **Are body-bias sweeps measurable?** Still the highest-leverage question.
   It would unlock eight confirmed-inert parameters and, more importantly,
   let `k1`/`k2` be identified against the body effect rather than only
   through `Abulk`.
2. **Is the pMOS `vsat` anomaly known upstream?** §0.2 is now measured, not
   inferred: the dimension is wasted on 8 of 12 pMOS bins.
3. **Per-device or joint-geometry fitting?** `wr`, `lint`, `wint`, `k3`, `w0`
   are sensitive but expected-degenerate per device; a joint multi-geometry
   fit is where they would become identifiable.
