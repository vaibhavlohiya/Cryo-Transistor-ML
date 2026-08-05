# 15-parameter surrogate extraction (params15 study)

**Status: labeled experiment.** The canonical exported card remains the
7-parameter `emu_search+fd` series; nothing here enters the canonical
export, Table 4, or Table 6.

## Setup

Identical to the confirmed 7-parameter protocol — pinned upstream
`39b1e518`, conda-forge ngspice-41, native bins, 10,000-point Latin
hypercube per device in the published vector's +/-10% box, frozen
published-card curve inclusion, faithful `rrmsCalc.py` scoring — except the
theta vector is extended from 7 to 15 BSIM4 parameters
(`CRYOML_PARAM_SET=params15`):

- paper's 7: `vth0, u0, nfactor, vsat, delta, rdsw, eta0`
- added 8: `pclm`, `pdiblc1`, `pdiblc2`, `ags`, `ua`, `ub`, `voff`, `prwg`

`ags` replaced the initially proposed `pvag`, whose published value is zero
(or 1e-12) on all 18 bins and therefore permanently frozen inside a
+/-10% multiplicative box. `pdiblc1` is zero-published on every pMOS bin
and `prwg` on every nMOS bin, so each stays pinned on that family and each
device effectively searches 14 live dimensions.

Production search recipe (unchanged from the 7-parameter campaign):
512x4 GELU emulator, 2,048 Adam starts x 600 steps, 14 NGSpice-validated
candidates, FD polish of the top 5 with `max_nfev` 120.

## Aggregate results

Frozen-inclusion RRMS; lower is better. `combined` = (nMOS mean + pMOS
mean)/2; `all-device` = mean over the 18 devices.

| fixed method | nMOS | pMOS | combined | all-device |
|---|---:|---:|---:|---:|
| paper parameters (published card) | 0.1198 | 0.3992 | 0.2595 | 0.2751 |
| 7-param surrogate raw | 0.0828 | 0.3581 | 0.2204 | 0.2357 |
| 7-param surrogate + FD (canonical) | 0.0788 | 0.3491 | 0.2140 | 0.2290 |
| 7-param foundation + FD (exploratory) | 0.0794 | 0.3434 | 0.2114 | 0.2261 |
| 15-param surrogate raw | 0.0832 | 0.4192 | 0.2512 | 0.2699 |
| **15-param surrogate + FD** | **0.0746** | **0.3260** | **0.2003** | **0.2143** |

The 15-parameter surrogate+FD wins against the published card on
**18/18 devices** and improves the canonical 7-parameter result on all
four aggregates (all-device 0.2290 ->
0.2143, -6.4%).
The gain concentrates in the pMOS family
(0.3491 -> 0.3260), consistent with
the added output-conductance/mobility/subthreshold freedom targeting the
documented cross-bias compromise.

## Per-device results

| device | paper card | 15p raw | 15p + FD |
|---|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.0595 | 0.0346 | 0.0319 |
| nmos_L0p19_W7 | 0.0967 | 0.0372 | 0.0368 |
| nmos_L0p25_W1p6 | 0.0983 | 0.0439 | 0.0435 |
| nmos_L100_W100 | 0.1403 | 0.1349 | 0.1325 |
| nmos_L1_W1p6 | 0.1283 | 0.0642 | 0.0626 |
| nmos_L1_W3 | 0.1599 | 0.1527 | 0.1003 |
| nmos_L20_W0p64 | 0.1283 | 0.1176 | 0.1099 |
| nmos_L8_W1p6 | 0.1474 | 0.0807 | 0.0795 |
| pmos_L0p35_W0p55 | 0.7270 | 0.7611 | 0.6195 |
| pmos_L0p35_W1p6 | 0.3776 | 0.3998 | 0.2882 |
| pmos_L0p35_W5 | 0.3291 | 0.4355 | 0.2907 |
| pmos_L0p5_W0p42 | 0.4560 | 0.3433 | 0.3187 |
| pmos_L0p5_W0p64 | 0.3498 | 0.3175 | 0.2975 |
| pmos_L2_W5 | 0.1907 | 0.1904 | 0.1622 |
| pmos_L4_W7 | 0.2702 | 0.3903 | 0.1993 |
| pmos_L8_W0p84 | 0.4340 | 0.3845 | 0.3773 |
| pmos_L8_W1p6 | 0.5433 | 0.5509 | 0.4689 |
| pmos_L8_W5 | 0.3146 | 0.4184 | 0.2381 |

## Observations

1. **FD polish carries more weight in 15-D.** The raw 15-parameter
   surrogate search (0.2699 all-device) is *worse* than
   the raw 7-parameter search (0.2357) —
   with the same 10,000 samples the emulator search degrades in the higher
   dimension — but FD recovers far more
   (0.2699 -> 0.2143) than in 7-D
   (0.2357 ->
   0.2290). The surrogate's role reduces to
   basin-finding; numerical polish does the precision work.
2. **The +/-10% box is binding.** 67 of 270 winning parameter values
   (~25%) sit at a box edge —
   `delta` on 11, `ub` on 9, `ags` on 8, `ua` on 8
   devices. The data pulls these parameters beyond +/-10% of the published
   values: a caveat for physical interpretation, and headroom for a
   wider-box follow-up.
3. `pmos_L2_W5`, the documented cross-bias-compromise device, improves
   0.1907 -> 0.1622 without any per-voltage selection.

## Caveats

- Identifiability: with 11 DC curves, several of the 15 parameters act as
  effective fit knobs (see box-edge pegging); extracted values should not
  be read as physical device parameters.
- Per-curve high-voltage behavior was not audited here; the 7-parameter
  high-voltage-guard diagnostic has no 15-parameter counterpart yet.
- Total extraction runtime 4.3 h across resumed
  segments on Apple Silicon with Rosetta ngspice-41; per-device runtimes in
  the JSON are not comparable to the original campaign's timing because of
  interruptions and machine-load differences.

## Artifacts

- `out/pdk15_surrogate/ml_<tag>.json` — per-device records
  (`param_set: params15` embedded), `sims_<tag>.npz`, `emu_<tag>.pt`
  (untracked, regenerable).
- `data/processed/pdk_synth_params15/` — 18 x 10,000-sample datasets
  (untracked, ~2.3 GB).
- `out/tables/params15_study.{md,json,csv}` — this report.

Reproduce: `pdk_gen_data.py --param-set params15 --num-samples 10000`,
then `pdk_ml_extract.py --param-set params15 --emu-arch 512,512,512,512
--n-adam-starts 2048 --adam-steps 600 --n-validate 14 --n-polish 5
--max-nfev 120 --out-dir out/pdk15_surrogate`.
