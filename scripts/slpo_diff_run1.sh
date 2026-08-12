#!/bin/bash
# slpo_diff_run1: SLPO-diff formulation, full dataset, 3 epochs, lr=1e-7.
# SLPO-diff: Huber regression on the margin (Δω = logit(π_w) - logit(π_l))
# toward a target margin (Δt = logit(τ_w) - logit(τ_l)) derived from the
# α mass-transfer. Fixes the unreachable target_w problem in original SLPO.
#
# Run: conda activate slpo && bash scripts/slpo_diff_run1.sh 2>&1 | tee /tmp/slpo_diff_run1.log

set -e
cd "$(dirname "$0")/third_party/dpo"

ARCHIVE="./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt"

echo ""
echo "======================================================================="
echo "  slpo_diff_run1: full dataset, 3 epochs, lr=1e-7, SLPO-diff (option E)"
echo "======================================================================="

python train.py \
  model=pythia28 \
  datasets=[hh] \
  loss.name=slpo \
  +loss.alpha=0.5 \
  gradient_accumulation_steps=1 \
  batch_size=4 \
  eval_batch_size=16 \
  trainer=BasicTrainer \
  eval_every=10000 \
  do_first_eval=true \
  n_examples=null \
  n_epochs=3 \
  debug=true \
  warmup_steps=150 \
  max_grad_norm=10 \
  lr=1e-7 \
  model.archive=$ARCHIVE \
  exp_name=slpo_diff_run1

echo "--- DONE: slpo_diff_run1 ---"
