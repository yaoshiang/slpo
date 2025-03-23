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

`experiment/sentiment/` attempts to perform controlled sentiment analysis,
the same task in the original DPO paper. Instead of their approach, we 
use an existing sentiment dataset, Stanford Sentiment Treebank 2 (SST-2), and
use a sentiment classifier created for it, namely, `textattack/roberta-base-SST-2`
which we demonstrate achieves 92% accuracy. 

To reproduce, run:

```sh
$ cd src
$ python -m experiment.sentiment.eval
$ python -m experiment.sentiment.analyze

Accuracy: 0.92
```

It also generates a bar chart confusion matrix. ![bar chart confusion matrix](src/calibration_curve.png). 
