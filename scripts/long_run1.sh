#!/bin/bash
# long_run1: Full dataset, 3 epochs, lr=1e-7, RMSprop + Huber, loss_w + loss_l.
# Run in tmux: conda activate slpo && bash scripts/long_run1.sh 2>&1 | tee /tmp/long_run1.log
#
# Hypothesis (June 2026): 8000-example sweeps were too short to see real signal.
# The goal is for the GAP (model_w_logodds - model_l_logodds) to widen over
# training even if both absolute logprobs decrease (DPO displacement). With the
# full 160k-example HH dataset and multiple epochs, the model should have enough
# signal to learn the W/L distinction.
#
# Per-token logodds normalization is active (chosen_lengths / rejected_lengths
# passed through slpo_loss), preventing outlier long-sequence batches from
# dominating the gradient.
#
# lr=1e-7 chosen as best compromise from lr sweeps 2-5 (least gap_w degradation
# at 8k examples; at full scale with multiple epochs the signal should dominate).
#
# Estimated runtime: ~3 days (160k examples/epoch × 3 epochs at ~1 ex/sec).

set -e
cd "$(dirname "$0")/third_party/dpo"

ARCHIVE="./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt"

echo ""
echo "======================================================================="
echo "  long_run1: full dataset, 3 epochs, lr=1e-7, RMSprop+Huber, loss_w+loss_l"
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
  exp_name=long_run1

echo "--- DONE: long_run1 ---"
