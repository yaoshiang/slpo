#!/bin/bash
# LR sweep round 2: longer runs with eval to see convergence trajectories.
# Run in tmux: conda activate slpo && bash scripts/lr_sweep2.sh 2>&1 | tee /tmp/lr_sweep2.log
#
# Round 1 showed: lr >= 1e-5 diverges, lr >= 3e-6 is worse, lr <= 1e-6 all look
# the same at 1600 examples. Need 8000 examples to see the trajectories diverge.
#
# Fixing gnorm=10 (standard), sweeping lr only.
# eval_every=1600 gives 5 eval checkpoints per run.
#
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
  local NAME="lr_sweep2_lr${LR}"
  echo ""
  echo "======================================================================="
  echo "  lr=${LR}  max_grad_norm=10  n_examples=8000  exp=${NAME}"
  echo "======================================================================="
  python train.py $COMMON lr=$LR exp_name=$NAME
  echo "--- DONE: ${NAME} ---"
}

# 5 runs: one decade below baseline, baseline, one above, two further above
# (1e-6 was borderline in sweep1, 3e-6 was clearly worse)
for LR in 1e-8 3e-8 1e-7 3e-7 1e-6; do
  run $LR
done

echo ""
echo "======================================================================="
echo "  ALL RUNS COMPLETE"
echo "======================================================================="
