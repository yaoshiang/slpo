#!/bin/bash
# LR sweep round 4: AdamW optimizer with Huber loss (option C).
# Run in tmux: conda activate slpo && bash scripts/lr_sweep4.sh 2>&1 | tee /tmp/lr_sweep4.log
#
# Motivation: sweep2 (RMSprop + Huber) and sweep3 (RMSprop + MSE) both fail to
# converge gap_w. RMSprop normalizes gradient magnitudes, reducing the optimizer
# to sign-gradient descent. AdamW adds momentum (bias-corrected first moment)
# which averages gradient directions across steps, potentially giving a more
# coherent signal on gap_w. AdamW also adds weight decay for regularization.
# Defaults: betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01.
#
# Same LRs as sweep2/sweep3 for direct comparison.
# 5 runs x 8000 examples ~= 10-12 hrs.

set -e
cd "$(dirname "$0")/third_party/dpo"

ARCHIVE="./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt"
COMMON="model=pythia28 datasets=[hh] loss.name=slpo +loss.alpha=0.5
  gradient_accumulation_steps=1 batch_size=4 eval_batch_size=16
  trainer=BasicTrainer eval_every=99999 do_first_eval=false
  n_examples=8000 n_epochs=null debug=true warmup_steps=150
  max_grad_norm=10 optimizer=AdamW
  model.archive=$ARCHIVE"

run() {
  local LR=$1
  local NAME="lr_sweep4_lr${LR}"
  echo ""
  echo "======================================================================="
  echo "  lr=${LR}  max_grad_norm=10  optimizer=AdamW  n_examples=8000  exp=${NAME}"
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
