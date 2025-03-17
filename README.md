# Welcome to the official repo for Supervised Learning Preference Optimization

I developed the thinking of SLPO after looking into DPO and it's amazing
reformulation of RLHF into a Maximum Likelihood Estimation problem. But
I didn't like the use of RL concepts like a reward, or the Bradley-Terry
model, so I sought to further reduce the problem into an even more
familiar construct of Crossentropy loss of a softmax activated model. There
was a limit to how far I could take this, so I made some adjustments
to derive a pure CCE based loss. 

The paper will be put on TechRxiv shortly and then submitted to a 
conference or journal. The implementation is vectorized within sequences,
but when packing sequences into a full context window, and batching,
those are not yet vectorized.

This code is GPL, which was chosing because it is infecting. The code and
paper are copyright 2025 Yaoshiang Ho. Contact me for commercial
uses or for other licenses.

## Organization

The `paper/` directory includes the raw latex, viz.ipynb for visualizations,
and the PDF.

The `src/slpo` directory contains the slpo function itself, as well
as some data handling utilities. 

The `test` directory contains multiple unit and ML tests. Run pytest. Goal
of this repo is zero warnings and zero errors. 

The `src/experiments` directory contains code to repro experiments.

## SLPO

Like most RLHF and RLHF-inspired algorithms, the inputs are:

* x: This is the prompt.
* y_w: This is the chosen (or winning) completion.
* y_l: This is the rejected (or losing) completion.

Like DPO, SLPO also requires values from a reference model, namely,
* prob(y_w | x): Probability of the chosen/winning completion. This is 
  typically a very small number, so we store it as a logprob.
* prob(y_l | x): Probability of the rejected/losing completion. This is 
  also stored as a logprob.

## Data prep

The current implementation only handled padded sequences. Packing is not supported.

## Experiments

`controlled_sentiment_generation.py` attempts to reproduce and extend the
first of the three approaches used by the DPO paper.

Fine Tune GPT2:
1. Load IMDB dataset.
2. Load GPT2 pretrained.
3. Run supervised fine-tuning (SFT) GPT2 on IMDB train, positive only. This
  is the reference model. 
4. Save the model to disk. 

Generate preference dataset:
1. Load reference model. 
2. Prompt using first two to eight tokens with low temperature, yielding 12 completions.
3. Use groundtruth sentiment model to rank completions, 
  using the same model as DPO: `siebert/sentiment-roberta-large-english`.
4. Create six pairs of completions, with more positive value as `y_w`. 
5. Save dataset to disk with probs. These probs become the soft targets for SLPO.

SLPO alignment training:
1. Load completions, with probs.
2. Use data_util's `GPT2PackedDataset` to perform alignment using SLPO.
3. Save the language model to disk.

Evaluation:
1. Load language model.
2. Prompt with two to eight tokens from IMDB (disjoint from SFT set).
3. Generate y_pred.
4. Evaluate y_pred using groundtruth sentiment model.
5. Evaluate frontier by measuring KL-Divergence (TBD: seq or token level).
6. Evaluate win-rate 
