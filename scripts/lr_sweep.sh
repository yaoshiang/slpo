#!/bin/bash
# LR x max_grad_norm sweep for SLPO.
# Run in tmux: conda activate slpo && bash scripts/lr_sweep.sh 2>&1 | tee /tmp/lr_sweep.log
#
# Strategy: sweep lr x max_grad_norm grid.
# Current grad_norm before clipping is ~500-1350, so:
#   gnorm=10   -> ~50-135x clipping (current baseline)
#   gnorm=100  -> ~5-14x clipping
#   gnorm=1000 -> effectively no clipping
# Effective parameter step ~= lr * min(gnorm, actual_norm)
# We want the sweet spot: fastest convergence without explosion.
#
# 21 runs x 1600 examples ~= 8-10 hrs total.

set -e
cd "$(dirname "$0")/third_party/dpo"

ARCHIVE="./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt"
COMMON="model=pythia28 datasets=[hh] loss.name=slpo +loss.alpha=0.5
  gradient_accumulation_steps=1 batch_size=4 eval_batch_size=16
  trainer=BasicTrainer eval_every=99999 do_first_eval=false
  n_examples=1600 n_epochs=null debug=true warmup_steps=50
  model.archive=$ARCHIVE"

run() {
  local LR=$1
  local GNORM=$2
  local NAME="lr_sweep_lr${LR}_gnorm${GNORM}"
  echo ""
  echo "======================================================================="
  echo "  lr=${LR}  max_grad_norm=${GNORM}  exp=${NAME}"
  echo "======================================================================="
  python train.py $COMMON lr=$LR max_grad_norm=$GNORM exp_name=$NAME
  echo "--- DONE: ${NAME} ---"
}

# Grid: 7 lr values x 3 gnorm values = 21 runs
# lr spans 4 decades around current baseline (1e-7)
# gnorm spans heavy-clip -> no-clip

for LR in 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5; do
  for GNORM in 10 100 1000; do
    run $LR $GNORM
  done
done

echo ""
echo "======================================================================="
echo "  ALL RUNS COMPLETE"
echo "======================================================================="
