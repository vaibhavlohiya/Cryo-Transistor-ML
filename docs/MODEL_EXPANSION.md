# Expanding the parameter domain

Design note, 2026-08-12. **Status: analysis only.** Nothing here changes the
canonical protocol, the metric, or the exported card. Every proposal below
would be a labeled experiment under the rules in `CLAUDE.md`.

Covers three questions: scaling the tuned parameter count past 15, dropping
the isothermal 77 K assumption, and mixing low-level process parameters with
high-level electrical figures of merit.

## Assumptions

Three questions were left open when this note was commissioned; I proceeded on
these assumptions and flag where the answer would change.

1. **Multi-temperature measurement is not yet committed.** Section 2 is
   written as a feasibility analysis with a staged path, plus a measurement
   spec should acquisition become possible.
2. **Target parameter count is "significantly higher," unspecified.** I treat
   ~25–30 as the next rung and mark explicitly where advice diverges above
   ~40.
3. **Repo-specific.** Recommendations name concrete code and artifacts.

## 0. Three findings that constrain everything below

**The 77 K card is a retargeted isothermal card, not a temperature model.**
In both corner files, `tnom=-196.15` and the entire BSIM4 temperature
machinery is zeroed: `kt1`, `kt2`, `kt1l`, `at`, `ute`, `ua1`, `ub1`, `uc1`,
`prt`, `xtis` — and every `l*`/`w*`/`p*` binning variant of each. Temperature
scaling is switched off and the reference temperature moved to 77 K. This was
a sound engineering decision (see §2.2), but it means the card has no
temperature behavior to extend — §2 is about turning on machinery that is
currently entirely disabled.

**The simulator plumbing for arbitrary temperature already exists.**
`simulate_pdk(..., temp_K: float = 77.0)` and `_batch_deck` already emit
`.options temp={temp_K - 273.15:.4f}`. Nothing hardcodes 77 K but the default.

**Measured data is 77 K only.** 22 device folders in `cryo_data`, no
temperature axis. `KT1`/`UTE`/`PRT` are not identifiable from isothermal data
at any parameter count — §2 is data-blocked until that changes.

## 1. Scaling the tuned parameter count past 15

### 1.1 Your own results already identify the constraint

Three distinct failure modes usually get collapsed into "dimensionality." The
params15 study separates them cleanly:

| observation | what it actually means |
|---|---|
| 67/270 winning values (25%) peg a ±10% box edge | **box misspecification** — the optimum is outside the feasible set |
| raw 15-D (0.2699) worse than raw 7-D (0.2357) at equal budget | **search / surrogate sample-efficiency** wall |
| 10k → 100k samples leaves final RRMS flat | **not** sample volume |

Only the middle one is dimensionality in the usual sense. The first is a
modeling error that gets *worse* with more parameters, and it is almost
certainly the largest single source of loss right now. Fix it before adding
dimensions.

### 1.2 The box is the wrong shape, and it is inherited

`LhcBox` is a ±10% multiplicative box around the published bin values,
inherited from the upstream `nomSweep_latinHypercube.py`. That box was
designed as a **manufacturing-variation envelope** — a plausible spread of a
process around nominal. You are using it as an **extraction search domain**.
Those are different objects, and conflating them is the root cause of the
edge-pegging.

Three concrete defects, all of which amplify with parameter count:

- **Zero-published parameters are permanently dead.** `frac * 0 == 0`, so
  `lo == hi` and the sigmoid maps every `z` to the same point. You already hit
  this with `pvag` (dropped) and work around it for `pdiblc1`/`prwg` by
  accepting 14 live dimensions instead of 15. At 30 parameters this is not a
  workaround, it is a wall: many BSIM4 parameters are legitimately zero by
  default (`kt2`, `prwb`, `keta`, most `l*/w*/p*` terms). **Fix: additive
  fallback** — when `|published| < eps`, use an absolute range from the
  parameter's physical scale rather than a multiplicative one.
- **Linear spacing on parameters that span decades.** `LhcBox` comments that
  "the range is too narrow to need log spacing," which is true at ±10% but
  stops being true the moment you widen. `ua` (~-9.6e-11), `ub` (~1.3e-18),
  `vsat` (~1e5) differ by 29 orders of magnitude; a widened linear box on
  `ub` is numerically hopeless. **Fix: log-spacing for positive-definite
  parameters, signed-log for the rest.**
- **One global `frac` for physically unlike parameters.** `delta` (a smoothing
  constant, weakly constrained) and `vth0` (directly observable, tightly
  constrained) get the same ±10%. The edge-pegging histogram —
  `delta` 11 devices, `ub` 9, `ags` 8, `ua` 8 — is exactly the set of shape
  parameters whose physical plausible range is much wider than 10%. **Fix:
  per-parameter ranges from physical bounds, not a single fraction.**

The pegging statistic is your evidence that the last one matters: those four
parameters are telling you the box is too tight for them specifically, not
that the fit wants to run away.

### 1.3 Group by mechanism, because BSIM4 parameters are not exchangeable

Adding parameters at random is what produces the convergence failures the
question anticipates. BSIM4 parameters partition into groups that each
dominate a specific bias region, and identifiability within a group is far
worse than across groups:

| group | parameters | constrained by |
|---|---|---|
| subthreshold | `voff`, `nfactor`, `cdsc`, `cdscd`, `cdscb`, `minv` | low-Vg IdVg, log scale |
| short-channel Vth / DIBL | `dvt0`, `dvt1`, `dvt2`, `eta0`, `etab`, `dsub` | Vth vs L, IdVg at multiple Vd |
| mobility | `u0`, `ua`, `ub`, `uc`, `eu` | linear-region IdVg, moderate–strong inversion |
| velocity saturation | `vsat`, `a0`, `a1`, `a2`, `ags`, `keta` | IdVd knee |
| output conductance | `pclm`, `pdiblc1`, `pdiblc2`, `pdiblcb`, `drout`, `pvag` | IdVd saturation slope |
| series resistance | `rdsw`, `prwg`, `prwb`, `wr` | high-Vg linear region |

Your params15 additions were well chosen against this table — they added
output conductance (`pclm`, `pdiblc1`, `pdiblc2`), mobility shape (`ua`,
`ub`), subthreshold offset (`voff`), saturation knee (`ags`), and series
resistance (`prwg`). That is one or two per group rather than a deep dive into
one group, which is why it converged at all. **Keep that discipline when
scaling**: breadth across groups before depth within a group, because
within-group parameters are the collinear ones.

Note what this table implies about your 11 metric curves: you have 5 IdVd and
6 IdVg curves per device. Some groups (output conductance) are constrained by
only a handful of curves. There is a real information ceiling here that no
optimizer fixes.

### 1.4 Screen for identifiability before adding, using machinery you have

`fd_polish` already calls `least_squares(method="trf", jac="2-point",
diff_step=2e-2)`. That finite-difference Jacobian is the identifiability
diagnostic — it is currently computed, used for a step, and thrown away.

Proposed screening protocol, cheap because it reuses the existing FD path:

1. At the published card, compute the sensitivity matrix `S` of the residual
   vector w.r.t. each candidate parameter (one FD sweep, `n_params` NGSpice
   evaluations per device).
2. Normalize columns to unit scale — `S̃_j = S_j · Δθ_j / ||r||` — so
   parameters with wildly different units compare.
3. **Sensitivity screen**: drop any parameter whose column norm is below the
   simulator's own numerical noise floor. Those are unfittable regardless of
   algorithm, and each one you keep adds a flat direction the optimizer will
   wander along.
4. **Collinearity screen**: compute the collinearity index
   `γ_K = 1 / σ_min(S̃_K)` over candidate subsets `K` (Brun, Reichert &
   Künsch, 2001). Rule of thumb: `γ > 10–20` means the subset is practically
   unidentifiable. Use it to choose *which* members of a group to free.
5. Rank by SVD: the right singular vectors of `S̃` tell you which *combinations*
   are determined. If two parameters only ever appear in one combination, free
   one and pin the other.

This converts "how many parameters can we add?" from a guess into a measured
quantity, per device and per bias region. My expectation is that it will
justify roughly 25–35 well-identified parameters on this curve set, and reject
several you might otherwise have added.

Beyond local screening, **Morris elementary-effects** screening is the right
next tier (global, ~`(D+1)·r` evaluations, r≈10–20 trajectories), and
**Sobol** indices only if you need variance attribution for a paper — Sobol is
expensive and you likely do not need it to make decisions.

### 1.5 Regularization, not clipping, is what prevents non-physical minima

A hard box is a constraint whose active set *is* the pathology you observe:
25% of parameters sitting on the boundary means the optimizer is being held in
place by the constraint rather than by the data. Widening the box alone will
convert some of that into genuine runaway.

The standard fix is a soft prior instead of a hard wall:

```
minimize  RRMS(θ)² + λ · || (θ − θ_published) / s ||²
```

with per-parameter scale `s` from the physically plausible spread. This is MAP
estimation with a Gaussian prior centered on the published card, and it has
three properties you want: it permits excursions beyond ±10% when the data
demands them, it penalizes them proportionally, and `λ` gives you a single
dial trading fit against physicality that you can report as a curve rather
than a binary choice. Cross-validate `λ` on held-out curves.

Additional guards worth adopting as D grows:

- **Multistart disagreement as a diagnostic.** You already run 2,048 Adam
  starts. Record the spread of the top-k *parameter vectors*, not just their
  losses. Tight loss with wide parameter spread is the signature of
  unidentifiability, and it is free to measure.
- **Physical-plausibility assertions** rather than box edges: mobility
  positive, `vsat` within a decade of thermal velocity limits, subthreshold
  swing above the band-tail floor. These reject non-physical minima that sit
  comfortably inside any box.

### 1.6 Algorithms

Your current stack — multistart Adam through a frozen surrogate → NGSpice
validation of top candidates → TRF finite-difference polish — is sound, and
the division of labor it implies is the correct one for this problem: **the
surrogate finds the basin, FD finds the bottom.** That is exactly what the
params15 numbers show (raw 15-D worse than raw 7-D, FD recovering it to
better-than-7-D), and it is the single most important structural fact for
scaling. It means surrogate quality needs to be good enough for *basin
identification only*, which is a much weaker requirement than global accuracy
and scales far better with D.

Adjustments as D grows:

- **Surrogate sample complexity is the thing that degrades**, not the
  optimizer. Your 100k probe showed the *raw* stage improving substantially
  (mean 0.295 → 0.157 from 30k to 100k on the probe pair) while post-FD stayed
  flat — more data buys basin-finding, which FD was already providing. Above
  ~25 parameters, expect raw-stage degradation to be the first visible symptom.
- **CMA-ES** deserves promotion from control to first-class option. Your
  existing CMA control (0.4991, beating the FD control on 11/18 devices, and
  finding basins ML v1 missed) is evidence it explores differently. It is
  well-suited to 20–40 dimensions with ill-conditioned, non-separable
  landscapes, which is precisely this problem. Cost is its serial nature.
- **Keep TRF**, but reconsider `diff_step=2e-2` at higher D — FD Jacobian cost
  is `O(D)` per iteration, so a 30-parameter polish is 2× the simulator cost
  of 15 at the same `max_nfev`. Budget accordingly, or move to a
  Broyden-updated Jacobian to amortize.
- **Staged group-wise initialization then joint polish.** The classical
  industry extraction flow fits groups sequentially in bias regions where each
  dominates (subthreshold → mobility → velocity saturation → output
  conductance → resistance), then releases everything for a final joint
  refinement. This is strictly better conditioned than starting a 30-parameter
  joint fit cold, and it maps naturally onto your group table.

### 1.7 The next gain is in acquisition, not volume

Your probe result is unusually clean evidence: 10× the data, flat final RRMS.
Uniform LHC in a fixed box has saturated. What has *not* been tried at scale
is sequential design — allocating samples where they change the answer:

- resample around the incumbent optimum after each campaign (you did a
  FD-centered quarter in earlier scaling work; make it the default and
  iterate);
- weight sampling by the screened sensitivity from §1.4, so flat directions
  stop consuming budget;
- active learning on surrogate disagreement (ensemble variance) rather than on
  parameter-space coverage.

At 30 parameters, uniform LHC coverage is hopeless on principle — the box
volume is inaccessible at any budget you can afford — so this stops being an
optimization and becomes a requirement.

### 1.8 Recommended ladder

1. **Redesign the box** (additive fallback for zeros, log spacing,
   per-parameter ranges) and **swap hard clipping for a soft prior**. No new
   data. Re-run params15 unchanged otherwise; the pegging fraction is the
   success metric. I expect this to be the largest single improvement
   available.
2. **Run the identifiability screen** (§1.4) and let it select the next tier.
3. **Scale to ~25–30**, breadth-first across mechanism groups, with staged
   group-wise initialization.
4. **Reassess above ~40.** Advice genuinely diverges there: with that many
   free parameters on 11 curves per device you are into hierarchical /
   binning-aware fitting (sharing parameters across geometries via the
   `l*/w*/p*` binning terms rather than fitting each bin independently), which
   is a different problem statement and closer to what a foundry does. Your
   foundation-emulator work is the natural precedent.

## 2. Dropping the isothermal assumption

### 2.1 The two blockers, stated first

Neither is an argument against doing this — both are things to resolve before
any modeling effort is meaningful.

**You have no multi-temperature data.** Temperature parameters are identified
purely from how curves *move* with T. With one temperature there is nothing to
fit; any value of `KT1` can be absorbed into `VTH0`, any `UTE` into `U0`. This
is exact degeneracy, not poor conditioning.

**Your card has no temperature model to extend.** All of `kt1, kt2, kt1l, at,
ute, ua1, ub1, uc1, prt, xtis` are zero with `tnom=-196.15`.

### 2.2 Why the upstream card was built that way

Worth being explicit, because it explains the difficulty. BSIM4's temperature
equations are shifts about a reference:

| quantity | BSIM4 temperature law | parameters |
|---|---|---|
| threshold voltage | `ΔVth = (KT1 + KT1L/Leff + KT2·Vbseff)·(T/TNOM − 1)` | `KT1`, `KT1L`, `KT2` |
| mobility | `U0(T) = U0·(T/TNOM)^UTE` | `UTE` |
| mobility degradation | `UA(T) = UA + UA1·(T/TNOM − 1)`, same for `UB`, `UC` | `UA1`, `UB1`, `UC1` |
| velocity saturation | `VSAT(T) = VSAT − AT·(T/TNOM − 1)` | `AT` |
| series resistance | `RDSW(T) = RDSW + PRT·(T/TNOM − 1)` | `PRT` |
| junction leakage | Arrhenius in `XTIS`/`XTID`, ideality `NJS`/`NJD` | `XTIS`, `NJS`, … |

These are smooth low-order fits calibrated over roughly 220–400 K. Below
~100 K several of them fail structurally rather than numerically:

- **Subthreshold slope.** BSIM4 gives `SS = n·(kT/q)·ln 10`, which → 0 as
  T → 0. Measured devices saturate near 10–20 mV/dec below ~50 K because
  band-tail states and interface traps dominate. No choice of `NFACTOR`,
  `VOFF`, or `CDSC*` reproduces a floor, because the `kT/q` prefactor is
  hardcoded physics. This is the single best-known BSIM4 cryo failure.
- **Threshold voltage.** The linear-in-T `KT1` law cannot capture the
  saturating upturn seen from 77 K down; incomplete dopant ionization makes
  the effective body doping itself temperature-dependent.
- **Mobility.** The `(T/TNOM)^UTE` power law assumes phonon-limited transport.
  At cryogenic temperatures phonon scattering freezes out and Coulomb and
  surface-roughness scattering take over, so mobility flattens or turns over —
  a single exponent cannot span both regimes.

So retargeting `TNOM` to 77 K and zeroing the temperature terms is not
laziness; it is the standard and correct move when you need an accurate card
at *one* cryogenic temperature. The cost is exactly what you are now asking
about: the card is valid only at 77 K.

### 2.3 What variable temperature does to the parameter count

The base additions are modest — roughly 10–15 core parameters
(`KT1, KT1L, KT2, UTE, UA1, UB1, UC1, AT, PRT`, plus junction terms if you
model leakage). Three multipliers make it much larger than that:

- **Geometry binning.** Your card carries `l*/w*/p*` variants of every
  temperature parameter (`lkt1, wkt1, pkt1`, …). A fully binned temperature
  model is ~4× the base count, i.e. 40–60 card entries. You need not tune all
  of them, but the binning is where the count actually explodes.
- **Data volume.** `N_temperatures × 18 devices × 11 curves`. At 6
  temperatures that is ~1,200 curves, and your synthetic LHC datasets multiply
  by the same factor — a 10k-sample dataset per device per temperature at
  current cost is already ~2.3 GB × N_T.
- **Structural change in the fit.** This is the important one. Temperature
  parameters are *shared across temperature by construction* — that is what
  makes them meaningful. So the problem stops being 18 independent per-device
  fits and becomes a joint fit with parameters tied across a temperature axis.
  Your per-device optimization loop does not express this. The foundation
  emulator (one model conditioned across devices) is the closest existing
  precedent in the repo, and its architecture is roughly what a
  temperature-conditioned surrogate would need.

### 2.4 A staged path that reuses what you have

**Stage A — per-temperature isothermal cards.** Run the *existing, unchanged*
pipeline at each measured temperature, exploiting the `temp_K` argument that
is already plumbed. Output: an independent 7- or 15-parameter card per
(device, temperature). This requires no new modeling, no new optimizer, and no
decision about temperature equations. It is also independently publishable and
is what most cryo-CMOS papers actually report.

**Stage B — inspect θ(T) trajectories.** Plot each extracted parameter against
temperature. This is the decision-making step, and it is nearly free once
Stage A exists. Smooth monotonic trajectories mean the standard BSIM4 laws
have a chance; kinks, non-monotonicity, or saturation mean they do not.

**Stage C — choose the temperature model** on that evidence:
 - *standard BSIM4 equations* if trajectories are smooth over your range;
 - *piecewise / multi-TNOM cards* with a handoff temperature — pragmatic,
   widely used, and honest about the physics changing regime;
 - *interpolation between per-temperature cards* — most accurate, least
   elegant, and not portable to a foundry flow;
 - *modified temperature equations* (e.g. an effective-temperature floor
   `T_eff = √(T² + T_0²)` in the subthreshold prefactor) — this is the active
   research direction for cryo compact models and would be a genuine
   contribution, but it means patching model equations, not just parameters,
   and therefore a Verilog-A or simulator-level change rather than a card
   change.

**Stage D — joint fit** across the temperature axis, only once C is decided.

Stage A is the right immediate deliverable, and it degrades gracefully: even
if you never reach Stage D, per-temperature cards are a real result.

### 2.5 Measurement spec, if acquisition becomes possible

Should multi-temperature measurement come on the table: sample more densely
where the curvature is, i.e. at the low end — something like
**4, 20, 40, 77, 150, 225, 300 K**. Seven temperatures resolves the
subthreshold-floor onset and the mobility turnover, both of which sit below
100 K; uniform spacing to 300 K would waste points in the region where BSIM4
already works. Keep the same 11 metric curves and the same devices so the
existing scoring path applies unchanged.

## 3. Mixing low-level parameters with high-level electrical features

### 3.1 The question conflates three different roles

Untangling these resolves most of it:

1. **Free parameters** — what the optimizer varies (currently the 7 or 15).
2. **Model inputs/features** — what the surrogate or direct MLP consumes.
3. **Objective terms** — what defines a good fit.

`TOXE`, `NDEP`, `XJ` are candidates for role 1. `Vth`, `Ion/Ioff`, `gm` are
*derived quantities* — they are not parameters at all and belong in roles 2
and 3. My answer differs sharply between them.

### 3.2 TOXE, NDEP, XJ as free parameters: mostly no

Three reasons, in order of severity.

**Degeneracy.** Long-channel threshold voltage depends on doping and oxide
through
`Vth ≈ V_FB + 2φ_B + γ√(2φ_B − V_bs)` with `γ = √(2qε_si·NDEP)/C_ox` and
`C_ox = ε_ox/TOXE`. `VTH0`, `NDEP`, and `TOXE` therefore trade off against
each other almost exactly on DC I-V. Freeing all three creates a nearly flat
manifold — precisely the pathology §1.4's collinearity screen is designed to
reject, and it will reject this trio.

**You have DC I-V only.** `TOXE` is properly constrained by C-V measurement,
where `C_ox` is directly observable. Without C-V it is weakly identifiable in
principle, not just in practice. Same for `XJ`, which is best constrained by
short-channel behavior across a geometry series and is largely absorbed by
`RDSW` and the SCE parameters on I-V alone.

**They are process constants, and transferability is the point.** The foundry
fixes `TOXE`/`NDEP`/`XJ`; they do not change at 77 K, because cooling does not
move oxide thickness. A card that "fits better" by moving `TOXE` has stopped
being a physical model of the process and become a curve fit wearing BSIM4's
clothes. It will not extrapolate to geometries or bias conditions outside the
fitting set, and it undermines the claim that the extracted card is a
*cryogenic model* rather than an interpolant.

**The legitimate exception.** Incomplete ionization at cryogenic temperatures
genuinely changes the *effective* active doping relative to the metallurgical
value. Freeing `NDEP` is defensible on that basis — but as a physically
motivated, tightly priored adjustment (§1.5), reported as such, never as a
free ±10% dimension. If you do it, freeze `TOXE` and `XJ` so the degeneracy
has only one active direction.

### 3.3 Figures of merit as objective terms: yes, and this is the strong idea

Here the answer flips, and it targets a real weakness of the current setup.

Your metric is `curve_rrms_new = RMSE / mean|I_meas|`, per curve, on **linear**
current, averaged over the 11 included curves. The per-curve normalization
equalizes *across* curves — a subthreshold-heavy curve counts as much as a
strong-inversion one. But *within* a curve, the squared error is on linear
current, so points near `I_on` dominate the sum and points three decades down
contribute essentially nothing.

Two consequences worth taking seriously:

- **Subthreshold behavior is nearly unconstrained** within each curve. `Ioff`,
  `SS`, and `VOFF`/`NFACTOR` shape can be badly wrong at negligible metric
  cost. For a cryogenic model this is unfortunate, since subthreshold slope is
  one of the headline cryo-CMOS quantities.
- **Output conductance is nearly invisible.** `g_ds = ∂I_d/∂V_ds` is a
  *derivative*; an RMSE on `I` barely sees it. This is a plausible mechanism
  for why params15's added output-conductance freedom (`pclm`, `pdiblc1`,
  `pdiblc2`) helped the pMOS family so much — it gave the optimizer a knob for
  something the metric only weakly rewards, and pMOS is where your residual
  concentrates (0.3260 vs nMOS 0.0746).

So adding FoM terms to the **training objective** is well motivated:

```
L = RRMS² + w_log·|Δ log I|² + w_gm·|Δ g_m|² + w_gds·|Δ g_ds|²
      + w_ss·|ΔSS|² + w_vth·|ΔVth|² + w_ioff·|Δ log Ioff|²
```

The log-current and `g_ds` terms are the two I would add first — they address
the two blind spots above directly, and both are computable from curves you
already simulate, with no new NGSpice cost.

**Protocol collision, and how to stay inside it.** `CLAUDE.md` says "Do not
redefine RRMS," and that rule is right. The resolution: FoM terms are a
*training/search objective only*. Frozen-inclusion RRMS remains the sole
reported score, and any such run is a labeled experiment. The interesting
result would be an objective that is not RRMS producing a *better* RRMS,
because it conditions the search better — that is a legitimate and reportable
finding. If instead it improves subthreshold at a cost in RRMS, that is a
trade you report honestly rather than bury.

### 3.4 Architecture consequences

- **Multi-head surrogate.** Predict the 11 curves *and* the derived FoMs from
  the same trunk. FoMs are smooth, low-dimensional functions of θ and are
  easier to learn than full curves, so the auxiliary head acts as a
  regularizer on the shared representation. Cheap to add to the existing
  emulator.
- **Direct MLP inputs.** You already feed linear + signed-log current (602
  inputs). Appending derived features — `g_m`, `g_ds`, extracted `Vth`,
  `log Ioff`, `SS` — is a small change with a plausible payoff, because these
  are exactly the quantities a human extraction engineer reads off the curves.
  It also partly addresses the direct MLP's known weakness relative to the
  surrogate path.
- **Loss weighting is the practical difficulty.** The terms have different
  units and scales. Fixed physically motivated weights (normalize each term by
  its measurement uncertainty) are the defensible choice; learned uncertainty
  weighting is an alternative but adds a moving part that will complicate
  reporting.
- **Do not let FoMs replace curves.** Fitting only to `Vth`/`Ion`/`Ioff`
  reduces 11 curves to a handful of numbers and throws away most of the
  information. FoMs are a *reweighting* of the objective, not a substitute
  target.

### 3.5 Expected effect on accuracy

Better conditioning (FoMs are more identifiable than individual current
points), better balance across bias regions, and specifically better
subthreshold and output-conductance behavior. Headline RRMS may move either
way — you are explicitly optimizing something else — which is why this must
run as a labeled paired experiment against the fixed series, with both the
RRMS and the FoM residuals reported.

## 4. Recommended sequencing

Ordered by expected value per unit of effort and risk:

1. **Box redesign + soft prior** (§1.2, §1.5). No new data, no new physics,
   directly targets your measured 25% edge-pegging. Highest confidence.
2. **Log-current and `g_ds` objective terms** (§3.3). No new data, no new
   simulation cost, addresses a demonstrated metric blind spot that plausibly
   explains part of the pMOS gap.
3. **Identifiability screening** (§1.4), then scale to ~25–30 parameters
   breadth-first across mechanism groups.
4. **Sequential/active acquisition** to replace uniform LHC (§1.7) — becomes
   mandatory rather than optional above ~25 parameters.
5. **Temperature**, Stage A first (§2.4), and only once multi-temperature
   measurement exists.

Items 1–2 are worth doing before any parameter-count increase: both make the
existing 15-parameter result better, and both make the higher-dimensional
problem better posed rather than merely larger.

## Open questions

These would change specific recommendations above:

1. Is multi-temperature measurement actually achievable, and over what range?
   If the floor is 77 K rather than 4 K, §2 becomes much easier — the worst
   BSIM4 breakdowns are below ~50 K.
2. What parameter count are you actually targeting? Above ~40 the
   recommendation shifts toward hierarchical binning-aware fitting.
3. Is C-V measurement available? It would change the answer on `TOXE` in §3.2
   from "no" to "yes, and it would improve the DC fit too."

## References

- Brun, Reichert & Künsch (2001), "Practical identifiability analysis of large
  environmental simulation models" — collinearity index and the `γ > 10–20`
  rule used in §1.4.
- BSIM4.8.1 manual, ch. 11 (temperature dependence) — the equations in §2.2.
- Beckers, Jazaeri & Enz, "Cryogenic MOSFET modeling" — subthreshold-slope
  saturation and the effective-temperature approaches referenced in §2.5.
- `docs/RESEARCH_LOG.md` (2026-08-05 entry) — the params15 results this note
  reasons from.
