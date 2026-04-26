---
layout: post
title: "Prompt Engineering as Hyperparameter Search: How DSPy Uses Optuna"
tags: [DSPy, Optuna, LLMs, prompt optimization, TPE, MIPROv2]
---

Most people who pick up DSPy stop at `BootstrapFewShot`. It works, it runs fast, it spits out a compiled program with reasonable few-shot demos, and the diminishing-returns instinct kicks in. Why keep tuning?

Because two of DSPy's other optimizers, `BootstrapFewShotWithOptuna` and `MIPROv2`, quietly do something more interesting. They reframe prompt engineering as something that should feel familiar to anyone who has ever tuned an XGBoost model: black-box hyperparameter optimization.

Once you see it that way, a lot of folklore around prompts starts looking like an unprincipled grid search you've been running in your head.

## What is actually being optimized

When DSPy "compiles" a program, the parameters it tunes aren't model weights. They're the artifacts that shape the prompt sent to the LM:

- Few-shot demonstrations per predictor
- Instructions (the natural-language directive at the top of a prompt)
- The combination of those across multiple predictors in a pipeline

The search space here is combinatorial and discrete. If you have 16 candidate demos and want to pick 4 per predictor across 3 predictors, you're already staring at millions of configurations. Grid search is hopeless. Random search burns LM calls for not much in return. You want something smarter.

## Optuna and TPE in one paragraph

Optuna is a hyperparameter optimization framework. Its default sampler is TPE, the <span class="hovernote" tabindex="0"><a href="https://arxiv.org/abs/2304.11127">Tree-structured Parzen Estimator</a><span class="note">See Watanabe (2023), <em>Tree-Structured Parzen Estimator: Understanding Its Algorithm Components</em> for a careful walkthrough of TPE.</span></span>. Unlike Gaussian-process Bayesian optimization, which fits a single surrogate model over the objective, TPE fits two densities: `l(x)` over trials that scored well and `g(x)` over trials that scored poorly. It then samples new candidates where the ratio `l(x) / g(x)` is highest. The intuition is just: spend trials in regions that look like past winners, not regions that look like past losers. It's cheap, it scales to high-dimensional categorical spaces, and it tolerates noisy objectives. All useful properties when your "objective" is an LM-graded metric that wobbles a bit between runs.

## BootstrapFewShotWithOptuna: the simpler case

The mechanic is straightforward:

1. Bootstrap N candidate few-shot demonstrations by running the unoptimized program over the trainset and keeping traces that satisfy the metric.
2. For each predictor in the program, expose demo selection as a categorical Optuna parameter.
3. Each Optuna trial samples one combination, runs the program against a validation set, and returns a score.
4. TPE updates its belief over which combinations work and proposes the next trial.
5. After `num_candidate_programs` trials, return the program that scored highest.

```python
from dspy.teleprompt import BootstrapFewShotWithOptuna

tp = BootstrapFewShotWithOptuna(
    metric=my_metric,
    max_bootstrapped_demos=4,
    num_candidate_programs=30,  # Optuna trials
)
compiled = tp.compile(program, trainset=train, valset=val)
```

Why does TPE beat random search here? Because demos interact. A demonstration that's perfect for predictor A may quietly poison predictor B's context window with off-distribution patterns. The reward surface is non-separable. TPE picks up on those interactions across trials without you having to model the joint distribution explicitly.

## MIPROv2: the richer case

MIPROv2 (Multi-prompt Instruction Proposal Optimizer, v2) takes the next step. It optimizes both instructions and demos jointly. The pipeline:

1. Summarize the dataset and inspect the program's code structure.
2. Use an LM to propose candidate instructions grounded in that context.
3. Bootstrap candidate demonstrations, same as before.
4. Hand the joint space (`instruction_candidates × demo_candidates` per predictor) to Optuna.
5. Run Optuna trials, each one evaluating a full configuration against the validation set.

If you peek inside `dspy/teleprompt/mipro_optimizer_v2.py`, you'll see the relevant bit:

```python
sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
study = optuna.create_study(direction="maximize", sampler=sampler)
```

Two details there are worth pausing on.

First, `multivariate=True`. Standard TPE models each parameter independently. Multivariate TPE models correlations between parameters. For MIPROv2 that matters a lot: the instruction you choose changes which demos work best with it, and the other way round. Independent sampling would miss those interactions. Multivariate captures them.

Second, the candidate instructions are LM-proposed, not human-written. So MIPROv2 is essentially using an LM to expand the search space, then using TPE to navigate it. That's a tidy decomposition. Generative breadth from the LM, sample-efficient search from Optuna.

(Optuna is an optional dependency for both optimizers. Install with `pip install dspy[optuna]`.)

## Where this earns its compute cost

Worth being honest about the tradeoffs:

- **Each trial costs real money.** A trial is a full program eval against the valset. With 30 trials and a 100-example valset, you're at 3,000 LM calls before counting any internal calls the program itself makes.
- **TPE needs warmup.** It usually takes 20 to 50 trials to convincingly beat random search. Below that, the priors don't have enough signal to be confident about anything.
- **Categorical only.** No continuous knobs in the loop.
- **Diminishing returns on small spaces.** If your demo pool is small, `BootstrapFewShotWithRandomSearch` often ties, and at lower cost.

Reach for the Optuna-based optimizers when:

- Your validation set is cheap to score (or your metric is fast).
- Your program has multiple predictors with non-trivial interactions.
- You're willing to spend compute upfront for a one-time compiled artifact you'll reuse.

Skip them when you're in early exploration, prototyping a single predictor, or working with a metric so noisy that TPE can't get a clean signal out of it.

## The reframe

The thing to take away here isn't "use MIPROv2." It's the conceptual shift:

> Prompt engineering is black-box optimization over a discrete combinatorial space, scored by a metric you define.

Once you accept that framing, the entire toolbox of classical hyperparameter optimization becomes available: TPE, CMA-ES, Hyperband, multi-objective Pareto search. DSPy happens to wire up Optuna because it's a clean fit, but nothing stops you from writing your own optimizer over the same primitives. The framework gives you `compile(program, trainset, valset, metric)`. What runs inside that call is up to you.

That's the actual unlock.
