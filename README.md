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

H/T to the authors of the DPO paper, a repo is available with their
implementation and use of the Anthropic HH dataset. Since KL divergence is 
explicitely not a goal, I chose not to replicate that evaluation. The only
evaluation I used is the one I am most interested in: preference optimization
of an LLM for question answering. Although there are more recent research
LLMs available, I chose to stick with the DPO repo's use of Pythia for 
ease of implementation. I ran DPO, and evaluated multiple checkpoints using
the LLM-as-judge technique they describe. 