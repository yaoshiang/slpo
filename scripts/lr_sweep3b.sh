#!/bin/bash
# LR sweep round 3b: re-run the 4 LRs that failed in sweep3 due to CUDA crash.
# lr=1e-8 already completed in sweep3; this picks up from lr=3e-8.
# Run in tmux: conda activate slpo && bash scripts/lr_sweep3b.sh 2>&1 | tee /tmp/lr_sweep3b.log

set -e
cd "$(dirname "$0")/third_party/dpo"

ARCHIVE="./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt"
COMMON="model=pythia28 datasets=[hh] loss.name=slpo +loss.alpha=0.5
  gradient_accumulation_steps=1 batch_size=4 eval_batch_size=16
  trainer=BasicTrainer eval_every=99999 do_first_eval=false
  n_examples=8000 n_epochs=null debug=true warmup_steps=150
  max_grad_norm=10
  model.archive=$ARCHIVE"

run() {
  local LR=$1
  local NAME="lr_sweep3_lr${LR}"
  echo ""
  echo "======================================================================="
  echo "  lr=${LR}  max_grad_norm=10  n_examples=8000  exp=${NAME}"
  echo "======================================================================="
  python train.py $COMMON lr=$LR exp_name=$NAME
  echo "--- DONE: ${NAME} ---"
}

for LR in 3e-8 1e-7 3e-7 1e-6; do
  run $LR
done

echo ""
echo "======================================================================="
echo "  ALL RUNS COMPLETE"
echo "======================================================================="
