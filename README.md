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
 is available now.

This code is GPL, which was chosen because it is infecting. The code and
paper are copyright 2025 Yaoshiang Ho. Although I am currently a 
Staff Software Engineer working at Google, this work is outside my
employment at Google. Contact me for commercial uses or for other licenses.

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

The `test` directory contains tests. Run them by running
pytest. 

The `scripts/third_party/dpo` directory contains a copy of https://github.com/eric-mitchell/direct-preference-optimization

The `src/eval` directory contains LLM-as-a-judge scripts.

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

## The original DPO implementation

`scripts/third_party/dpo/` is a clone of
https://github.com/eric-mitchell/direct-preference-optimization. The
only changes done are to get a repo from 2022 on PyTorch 2.0 to work 
in 2025, and recording the commit hash the clone was made from.

The vendored repo is used to reproduce the DPO experiments.

### SFT and DPO

SFT Run. Pythia 2.8B base is trained with SFT on Anthropic HH. During SFT
training, preference_dataset.py will load only the chosen (or winning)
responses as the dataset. So SFT training *also* implements the Preferred-FT
algorithm. 

The number of epochs is not specified in the paper. However, we ran for 
multiple epochs to ablate the effect of SFT/Preferred-FT training. The eval 
loss bottomed at step 497_232, with a loss of 121.05481. This corresponds to 
roughly 3 epochs. We use that for subsequent DPO runs. 


```sh
pushd scripts/third_party/dpo
nohup \
  python -u train.py \
  model=pythia28 \
  datasets=[hh] \
  loss=sft \
  exp_name=pythia28_sft_anthropic_hh \
  gradient_accumulation_steps=8 \
  batch_size=64 \
  eval_batch_size=32 \
  n_epochs=10 \
  trainer=BasicTrainer 
```

For the DPO run, once again, it is not specified how many epochs or what
hyperparameter to use for beta. We sweep the beta values between 0.1, 0.3, and 
0.5, based on the author's description of those being "reasonable places
to start" in their repo. We sweep for 10 epochs. 

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
  n_epochs=10 \
  model.archive=./.cache/yaoshiang/pythia28_sft_anthropic_HH__2025-10-21_16-48-21_233387/step-479232/policy.pt
```