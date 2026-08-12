#!/bin/bash
# LR sweep round 5: RMSprop + Huber, loss_w ONLY (option B).
# Run in tmux: conda activate slpo && bash scripts/lr_sweep5.sh 2>&1 | tee /tmp/lr_sweep5.log
#
# Motivation: sweeps 2-4 show gap_w worsening at all LRs regardless of optimizer
# (RMSprop vs AdamW) or loss function (Huber vs MSE). Hypothesis: loss_l is
# actively interfering with loss_w because gap_l > 0 causes loss_l to push model
# logprobs down for rejected sequences, and chosen/rejected share language
# patterns so this also drags p(w) down. Removing loss_l isolates the gap_w
# convergence signal and tests whether interference is the root cause.
#
# If gap_w converges with loss_w only, the fix is to use loss_w alone (or with
# a gated/delayed loss_l that only activates once gap_w is near zero).
#
# Same LRs as previous sweeps for direct comparison.
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
  local NAME="lr_sweep5_lr${LR}"
  echo ""
  echo "======================================================================="
  echo "  lr=${LR}  max_grad_norm=10  loss=loss_w_only  n_examples=8000  exp=${NAME}"
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
