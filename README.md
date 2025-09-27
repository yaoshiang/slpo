# Welcome to the official repo for Supervised Learning Preference Optimization

I developed the thinking of SLPO after looking into DPO and it's amazing
reformulation of RLHF into a Maximum Likelihood Estimation problem. But
I didn't like the use of RL concepts like a reward, or the Bradley-Terry
model, so I sought to further reduce the problem into an even more
familiar construct of Crossentropy loss of a softmax activated model. I ended
with an interpretation less about Bradley-Terry, about more about a 
crossentropy loss with a scaling factor that essentially clips the 
CE loss from adjusting the logits very much. This is no surprise since
RLHF with PPO depends on some version of clipping. 

I realized I could make minor adjustment to further eliminate the clipping...
but at the cost of a much more complex management of the space of possible
token sequences. In fact, this complex management is why I think RL was
used initially, but I hope to prove that RL concepts are 
unnecessary with the ability to group intracticably large numbers of
sequences using the y_bar technique I describe. 

The [preprint](https://www.techrxiv.org/users/888076/articles/1267902-supervised-learning-preference-optimization-rethinking-rlhf-and-dpo-as-supervised-learning)
 is available now. The implementation is vectorized within sequences,
but when packing sequences into a full context window, and batching,
those are not yet vectorized.

This code is GPL, which was chosen because it is infecting. The code and
paper are copyright 2025 Yaoshiang Ho. Contact me for commercial
uses or for other licenses.

## Installation

This project has two sets of dependencies. The core dependencies are in 
`requirements.txt`, and the dependencies for the vendored 
`direct-preference-optimization` library are in `requirements-dpo.txt`.

To install all dependencies, run:

```bash
pip install -r requirements.txt
pip install -r requirements-dpo.txt
```

## Organization

The `paper/` directory includes the raw latex, viz.ipynb for visualizations,
and the PDF.

The `src/slpo` directory contains the slpo function itself, as well
as some data handling utilities. 

The `test` directory contains multiple unit and ML tests. Run pytest. Goal
of this repo is zero warnings and zero errors. 

The `src/experiments` directory contains code to repro experiments.

The `src/third_party/dpo` directory contains a copy of https://github.com/eric-mitchell/direct-preference-optimization

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

## The original DPO implementation

`src/third_party/dpo/` is a clone of
https://github.com/eric-mitchell/direct-preference-optimization. The
only changes done are to get a repo from 2022 on PyTorch 2.0 to work 
in 2025, and recording the commit hash the clone was made from.

The vendored repo is particularly useful to reproduce the experiments described in the DPO paper. 

### Test 

Test that the dpo directory works as intended.

Make sure you are authenticated with HuggingFace:

```sh
hf auth login
```

Run
```sh
python -u src/third_party/dpo/train.py model=pythia28 datasets=[hh] \
exp_name=anthropic_dpo_pythia28_cpu_test \
gradient_accumulation_steps=2 \
batch_size=2 \
eval_batch_size=2 \
n_examples=4 \
trainer=BasicTrainer \
model.archive=.cache
```

### SFT and DPO

SFT Run. Pythia 2.8B base is trained with SFT on Anthropic HH. During SFT
training, preference_dataset.py will load only the chosen (or winning)
responses as the dataset. So SFT training *also* implements the Preferred-FT
algorithm. 

The number of epochs is not specified. However, we ran for multiple epochs to ablate the effect of SFT/Preferred-FT training. Conditioning the model to follow the instruction-following template is very quick. Marginal improvement o
the 
```sh
nohup \
  python -u train.py \
  model=pythia28 \
  datasets=[hh] \
  loss=sft \
  exp_name=anthropic_dpo_pythia28 \
  gradient_accumulation_steps=8 \
  batch_size=64 \
  eval_batch_size=32 \
  trainer=BasicTrainer
```

DPO Run. The paper says they took a snapshot every 20,000 examples, but the 
chart indicates they evaluate every 300 steps. With a global batch size of 
64, 300 steps is 19_200 examples. 

```sh
nohup \
  python -u train.py \
  model=pythia28 \
  datasets=[hh] \
  loss=dpo \
  loss.beta=0.1 \
  exp_name=anthropic_dpo_pythia28 \
  gradient_accumulation_steps=16 \
  batch_size=64 \
  eval_batch_size=16 \
  trainer=BasicTrainer \
\  model.archive=.cache/yho_google_com/anthropic_dpo_pythia28_2025-06-29_22-10-00_490115/LATEST/policy.pt
```