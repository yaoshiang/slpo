#!/bin/bash
# dpo_baseline: Full dataset, 3 epochs, lr=1e-7, DPO (beta=0.1).
# Matches long_run1 in all hyperparameters except loss function.
# Purpose: validity check — if DPO moves the chosen/rejected margin on this
# dataset/model, SLPO should be able to as well; if DPO can't, the mush
# problem is in the data, not the loss.
#
# Run: conda activate slpo && bash scripts/dpo_baseline.sh 2>&1 | tee /tmp/dpo_baseline.log

set -e
cd "$(dirname "$0")/third_party/dpo"

ARCHIVE="./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt"

echo ""
echo "======================================================================="
echo "  dpo_baseline: full dataset, 3 epochs, lr=1e-7, beta=0.1"
echo "======================================================================="

python train.py \
  model=pythia28 \
  datasets=[hh] \
  loss=dpo \
  +loss.beta=0.1 \
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
  exp_name=dpo_baseline

echo "--- DONE: dpo_baseline ---"
