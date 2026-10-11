---
layout: post
title: "Muon vs AdamW: what an optimiser does, and what Muon changes"
tags: [optimisers, Muon, AdamW, deep learning, LLM training, visual explainer]
---

Yesterday my teammate [Shaurya](https://x.com/ShauryaSharthak) posted [a short note on X](https://x.com/ShauryaSharthak/status/2108981684171682137) asking why more people aren't using Muon for speech recognition, since in his runs it was mostly better than AdamW. He had trained the same speech recognition model twice, once with each optimiser. He swept the learning rate only for AdamW and then reused that same value for Muon, which should have tilted things in AdamW's favour. Muon still came out ahead, with a word error rate of 20.64 against 21.11, and it won on 23 of his 29 test sets.

I knew Muon by name. Kimi K2 and the recent DeepSeek models were trained with it. But if you had asked me what it does differently from AdamW, I could not have told you. So I sat down with the papers this weekend, and this post is what I came away with. It starts from what an optimiser is for, goes through Adam and AdamW, and then gets to Muon.

## What an optimiser is for

A neural network is a long list of numbers called weights. Training is a loop. You push a batch of examples through the network, measure how wrong its outputs are as a single number called the loss, and then run backpropagation to get a gradient for every weight. The gradient tells you which way that weight should move to make the loss go up, so you move it the other way.

How far to move it is the optimiser's job. The simplest answer is plain gradient descent: subtract the gradient times a small constant called the learning rate. When the gradient is computed on a random batch, this is called SGD.

The trouble is that the loss surface of a real network is a bad shape to walk down. Some directions are steep and others are almost flat. A learning rate small enough to be safe in the steep directions makes painfully slow progress in the flat ones. Pick a larger one and you bounce off the walls. Every optimiser since SGD is a different way of dealing with that.

The first fix is **momentum**, which goes back to Boris Polyak in 1964. Instead of stepping along today's gradient, you step along a running sum of recent gradients. Pushes that keep flipping direction cancel out, and pushes that keep pointing the same way add up. It is still inside every optimiser in this post, Muon included.

## Adam: a step size for every number

[Adam](https://arxiv.org/abs/1412.6980), from Diederik Kingma and Jimmy Ba in 2014, keeps two running averages for each weight. One is momentum, the average gradient. The other is the average of the *squared* gradient, which tells you how big that weight's gradients usually are. Adam divides the first by the square root of the second.

That division is what made Adam so popular. A weight with huge gradients gets its step divided by something huge, and a weight with tiny gradients gets divided by something tiny. When a gradient is steady, the ratio comes out close to plus or minus one, so every weight moves by roughly the learning rate no matter how big its gradient was. When a gradient keeps flipping sign, the average cancels out while the squared average stays large, and the step shrinks. You get an automatic step size for each of the billions of numbers in the model, and it works well with very little tuning.

There are two costs. Adam stores two extra numbers per weight, so a 7B model carries 14 billion numbers of optimiser state on top of its weights. And it treats every number on its own. Adam has no idea that the 4,096 × 4,096 numbers in one attention projection belong to a single matrix.

## AdamW: weight decay, done properly

Weight decay pulls every weight a little towards zero on each step, which keeps weights from growing without bound and helps the model generalise. With SGD you can get it by adding a small multiple of the weight to its gradient. Ilya Loshchilov and Frank Hutter pointed out in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (2017) that this goes wrong with Adam. The decay term gets added to the gradient and then divided by Adam's running size estimate along with everything else. So weights with large gradients get almost no decay, and quiet weights get a lot.

Their fix, AdamW, is a one-line change. Adam only sees the real gradient, and the decay is applied straight to the weight afterwards, so every weight shrinks by the same fraction. AdamW became the default for training Transformers. Most large language models of the last five years were trained with it, usually with weight decay 0.1.

## Muon: treat the weight matrix as a matrix

Muon was written up by [Keller Jordan](https://kellerjordan.github.io/posts/muon/) in December 2024, after it had been setting records in the NanoGPT and CIFAR-10 training speedruns since October. The name stands for MomentUm Orthogonalized by Newton–Schulz, and that is a fair summary of the algorithm. It keeps momentum, the same as SGD. But before applying the momentum to a weight matrix, it replaces it with the nearest *orthogonal* matrix.

It took me a while to see why that helps, so here is the picture that made it click for me. A weight matrix maps input vectors to output vectors. Feed it every input of length one, and the outputs trace out an ellipse. The lengths of the ellipse's axes are the matrix's singular values. Gradient updates for Transformer layers tend to be very lopsided. One or two directions dominate and the rest are close to zero, so the ellipse is long and thin.

<figure style="margin:28px 0;padding:18px;background:#fffdf8;border:1px solid #d7cfbf;border-radius:10px;overflow-x:auto;">
<svg viewBox="0 0 760 230" role="img" aria-labelledby="muon-ell-t muon-ell-d" style="display:block;width:100%;min-width:560px;height:auto;">
<title id="muon-ell-t">What orthogonalising does to an update</title>
<desc id="muon-ell-d">A unit circle of inputs is mapped by the momentum matrix to a long thin ellipse with axes 1.0 and 0.08. Muon's orthogonalised matrix maps the same circle to a circle of radius one, keeping the same axis directions.</desc>
<defs><marker id="me-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#20221f"/></marker></defs>
<g font-family="system-ui,sans-serif" fill="#20221f" font-size="12">
<circle cx="110" cy="110" r="65" fill="#ebe7f6" stroke="#66539a" stroke-width="1.6"/>
<text x="110" y="205" text-anchor="middle" font-weight="700">inputs of length 1</text>
<g fill="none" stroke="#20221f" stroke-width="1.5" marker-end="url(#me-ar)"><path d="M195 110H270"/><path d="M490 110H565"/></g>
<text x="232" y="98" text-anchor="middle" font-size="11" fill="#65645d">momentum</text><text x="527" y="98" text-anchor="middle" font-size="11" fill="#65645d">Muon</text>
<g transform="translate(380 110) rotate(-28)"><ellipse rx="85" ry="7" fill="#e4ebf7" stroke="#2456a6" stroke-width="1.6"/></g>
<text x="380" y="190" text-anchor="middle" font-size="11" fill="#65645d">axes 1.0 and 0.08</text>
<text x="380" y="207" text-anchor="middle" font-weight="700">one direction dominates</text>
<circle cx="650" cy="110" r="65" fill="#f8e4d8" stroke="#a74723" stroke-width="1.6"/>
<text x="650" y="205" text-anchor="middle" font-weight="700">every direction gets a full step</text>
</g></svg>
<figcaption style="margin-top:10px;color:#5b5b55;font-size:14px;">Orthogonalising keeps the directions of the update and sets every singular value to one. The weak direction, which momentum alone would have moved by 8% as much as the strong one, now gets an equal share.</figcaption>
</figure>

Muon squashes that ellipse back into a circle. The directions stay where they were, but every one of them now gets a full-sized step. Keller Jordan's explanation is that the small directions still matter for learning, and in an ordinary update they get drowned out by the big ones.

The contrast with Adam was the part I found most useful. Adam normalises each *number*, so roughly every entry in the matrix moves by the same amount. Muon normalises the *matrix*, so every direction it acts on moves by the same amount. Jeremy Bernstein and Laker Newhouse frame these as the same idea, steepest descent, measured with two different rulers, in [Old Optimizer, New Norm](https://arxiv.org/abs/2409.20325). For a layer whose job is to map vectors to vectors, the matrix ruler is the more natural one.

Computing the nearest orthogonal matrix exactly needs a singular value decomposition, and that is far too slow to do for every layer at every step. Muon gets close enough with a Newton–Schulz iteration instead: five rounds of a small polynomial, built only from matrix multiplications, which runs happily in bfloat16 on a GPU. Each round pushes small singular values up by about 3.4 times and leaves the ones near 1 roughly alone. After five rounds they all sit somewhere between about 0.7 and 1.2. That is not exactly 1, but it turns out to be good enough, and the extra compute is under 1% for typical language model training.

One detail that surprised me is that Muon does not replace AdamW. It only makes sense for the 2D weight matrices inside the network, such as the attention projections and the feed-forward layers. The embedding table, the output head, and the small vectors like norm gains and biases all stay on AdamW. So a Muon setup really runs two optimisers side by side. PyTorch has shipped [`torch.optim.Muon`](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html) since version 2.9, and it refuses anything that is not 2D.

## Does it hold up at scale?

Speedrun wins on small models do not always survive contact with large ones, so the next question was whether Muon would hold up. Moonshot AI tested this in [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (February 2025). They needed two changes. One was adding AdamW-style weight decay. The other was scaling each update so its overall size matches what AdamW would produce, which lets you reuse AdamW's learning rate and decay settings. With those in place, Muon reached the same loss with about 52% of AdamW's compute. Moonshot then trained [Kimi K2](https://arxiv.org/abs/2507.20534), a trillion-parameter model, on 15.5 trillion tokens with a Muon variant called MuonClip and reported zero loss spikes. DeepSeek's V4 models use Muon for their weight matrices and AdamW for everything else.

Shaurya's run adds a data point from a very different domain. His gains were biggest on phone audio, noisy far-field recordings and meeting audio. At equal training time, 28.8 minutes, Muon still won, 20.78 against 21.11, even though it fits fewer steps into that time. The exception was the AMI headset set, where AdamW won in both of his runs. And in a shorter one-epoch run, the two optimisers tied. So I would read this as one promising result rather than a settled one. It is still more interesting than most of what I read about optimisers, because the learning rate was tuned for the other side.

## The visual explainer

I built a step-by-step page to go with this post. It has an interactive race between SGD, momentum and Adam in a narrow valley, a figure showing how Adam sizes the steps for three different weights, the AdamW weight decay comparison, and a slider that runs Muon's real Newton–Schulz polynomial on a set of singular values so you can watch them move into the band. It ends with the parameter split, the evidence so far and the open questions.

<p style="margin:21px 0;">
  <a href="/explainers/muon-vs-adamw/" target="_blank" rel="noopener"
     style="display:inline-block;background:#a74723;color:#fff;padding:14px 18px;border-radius:7px;font-weight:600;text-decoration:none;border:0;font-size:16px;">
    Open the Muon vs AdamW explainer &rarr;
  </a>
  <br>
  <small style="color:#5b5b55;">Opens in a new tab. The page is self-contained and uses no external JavaScript or web fonts.</small>
</p>

## Papers and posts

- Keller Jordan et al., [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/), December 2024, with code at [KellerJordan/Muon](https://github.com/KellerJordan/Muon).
- Jeremy Bernstein and Laker Newhouse, [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325), 2024, and Bernstein's [Deriving Muon](https://jeremybernste.in/writing/deriving-muon).
- Jingyuan Liu et al. (Moonshot AI), [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982), 2025.
- Kimi Team, [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534), 2025.
- Diederik Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.
- Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017.
- Vineet Gupta, Tomer Koren and Yoram Singer, [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568), 2018. Muon is closely related to Shampoo with its accumulation turned off.
- Shaurya Sharthak, [Muon vs AdamW on ASR](https://x.com/ShauryaSharthak/status/2108981684171682137), October 2026.
