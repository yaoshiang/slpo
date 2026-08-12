#!/bin/bash
# LR sweep round 3: MSE loss (option D) instead of Huber.
# Run in tmux: conda activate slpo && bash scripts/lr_sweep3.sh 2>&1 | tee /tmp/lr_sweep3.log
#
# Motivation: sweep2 showed gap_w worsens at all LRs with Huber (option C).
# Huber clips both loss_w and loss_l gradients to the same coefficient (delta=1),
# giving loss_w only a 2x advantage. MSE restores gradient proportional to gap
# magnitude, giving loss_w a ~66x advantage when gap_w~=-33 vs gap_l~=0.5.
# Grad clipping (max_grad_norm=10) already handles large-gradient robustness.
#
# Same LRs as sweep2 for direct comparison.
# 5 runs x 8000 examples ~= 10-12 hrs.

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

for LR in 1e-8 3e-8 1e-7 3e-7 1e-6; do
  run $LR
done

echo ""
echo "======================================================================="
echo "  ALL RUNS COMPLETE"
echo "======================================================================="
