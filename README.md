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

## Installation

Install requirements:

``` sh
pip install -r requirements.txt
```

Run pytest to validate your setup:
``` sh
$ pytest
============================================== test session starts ==============================================
platform linux -- Python 3.12.1, pytest-8.3.5, pluggy-1.5.0
rootdir: /workspaces/rlhf-mle
configfile: pyproject.toml
plugins: anyio-4.9.0
collected 33 items                                                                                              

test/functional/test_functional.py ....                                                                   [ 12%]
test/functional/test_microsoft_phi.py .                                                                   [ 15%]
test/ml/test_ml_slpo.py ..............                                                                    [ 57%]
test/unit/test_memorization_model.py .....                                                                [ 72%]
test/unit/test_slpo.py .........                                                                          [100%]

========================================= 33 passed in 74.99s (0:01:14) ======================================
```

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

```sh
$ cd src
$ python -m experiment.sentiment.eval_textattack
$ python -m experiment.sentiment.analyze_textattack

Accuracy: 0.92
```

It also generates a bar chart confusion matrix. ![bar chart confusion matrix](src/calibration_curve.png). 

Now that we have an evaluation of the textattack classifer and have identified a Bayes Error of 8%, we 
can evaluate controlled sentiment generation on un-aligned Phi2 and Phi2 aligned with DPO and SLPO. 

```sh
$ cd src
$ python -m experiment.sentiment.eval_phi2
$ python -m experiment.sentiment.analyze_phi2

Accuracy: 0.??
```
