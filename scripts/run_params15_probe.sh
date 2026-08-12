#!/bin/zsh
# 2-device data-scaling probe for the params15 experiment.
# For nmos_L1_W3 and pmos_L4_W7 (largest raw->FD recovery at 10k), generate
# 30k and 100k LHC datasets and run the unchanged production extraction on
# each, so raw-surrogate and +FD RRMS can be compared against the existing
# 10k results in out/pdk15_surrogate.
set -euo pipefail
cd "$(dirname "$0")/.."

export NGSPICE_BIN="${NGSPICE_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/ng41/bin/ngspice}"
export PYTHONPATH=src
DEVICES="nmos:1:3,pmos:4:7"

for N in 30000 100000; do
  SYNTH="data/processed/pdk_synth_params15_n${N}"
  OUT="out/pdk15_probe_n${N}"
  if [ ! -f "$SYNTH/pmos_L4_W7.npz" ] || [ ! -f "$SYNTH/nmos_L1_W3.npz" ]; then
    echo "=== [$(date)] generating $N samples/device -> $SYNTH ==="
    .venv/bin/python scripts/pdk_gen_data.py --param-set params15 \
      --num-samples "$N" --workers 2 --devices "$DEVICES" --out-dir "$SYNTH"
  fi
  echo "=== [$(date)] extracting from $SYNTH -> $OUT ==="
  .venv/bin/python scripts/pdk_ml_extract.py --param-set params15 \
    --device mps --emu-arch 512,512,512,512 \
    --n-adam-starts 2048 --adam-steps 600 --n-validate 14 \
    --n-polish 5 --max-nfev 120 --devices "$DEVICES" \
    --synth-dir "$SYNTH" --out-dir "$OUT" --resume
done
echo "=== [$(date)] PROBE COMPLETE ==="
