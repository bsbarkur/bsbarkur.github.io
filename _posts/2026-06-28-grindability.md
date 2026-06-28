---
layout: post
title: "Grindability: Why Verifiability Alone Won't Scale AI"
tags: [grindability, RLVR, Dwarkesh Patel, AI training, verifiability, continual learning]
---

This week, Dwarkesh Patel published a video and essay that crystallized something I've been struggling to articulate: **verifiability is necessary for AI progress, but it's not sufficient. You also need grindability.**

Here's the framing that landed it for me — if you've been confused about why coding models advanced so fast while "computer use" still struggles despite both being easy to verify, this is why.

## The insight

The standard story is: "Once you can verify whether an AI's output is correct, RL can scale it." Make a verifier, run a million rollouts, keep what works. This is how AlphaGo learned to play Go and how GPT-4 learned to code.

Dwarkesh's point is that there's a hidden second condition. The domain also needs to be **grindable** — meaning you can run thousands of parallel rollouts against a deterministic, replayable simulator, and isolate exactly what worked.

> "It is not enough for a domain to be verifiable. It also has to be very grindable — in the sense that you can run lots of parallel rollouts against a deterministic and replayable simulator." — [Dwarkesh Patel on X](https://x.com/dwarkesh_sp/status/2070672008946589922)

## Grindable vs. ungrindable

Some domains are both verifiable and grindable. That's where AI has made the most progress.

| Domain | Verifiable? | Grindable? | Why |
|---|---|---|---|
| **Coding** | ✅ Tests pass/fail | ✅ Spin up 1000 identical containers, let agents fix the same bug in parallel | The environment is deterministic, sandboxed, infinitely replayable |
| **Math** | ✅ Answer is right or wrong | ✅ Generate 1000 problems, run 1000 agents, check instantly | No environment needed — just problems and answers |
| **Chess / Go** | ✅ Win or lose | ✅ Same board state replayed 1000 times against the same engine | Perfect deterministic simulator |
| **Computer use** | ✅ Did you complete the checkout? | ❌ Amazon detects your bots, accounts get banned, flows change | "You can't have a thousand agents go try the same checkout flow on Amazon. Because Andy Jassy will find and detect your bots and shut your ass down." |
| **Day trading** | ✅ Did you profit? | ❌ Same market conditions never repeat; your own trades change the market | The market only happened once |
| **Court cases** | ✅ Did you win? | ❌ Can't retry the same case 1000 times with different arguments | Each case is unique — you only get one shot |
| **Building a business** | ✅ Is it profitable? | ❌ Customer feedback takes months or years | "The outer loop verification may take months or years of real-world actions to elicit, and cannot be re-observed by perturbing the model's actions thousands of times in parallel" |

## Analogies that make it concrete

**Grindable — practicing penalty kicks.** Same goal, same ball, same spot. Take 1000 shots. Each miss tells you something specific about your technique — "my follow-through was off on that one." You isolate the variable and fix it. This is how AI learns to code.

**Not grindable — playing in the World Cup final.** One match, one chance. The opponent adjusts. The crowd changes your decisions. You can't replay the same match 1000 times to see which formation works best. This is AI trying to do computer use on Amazon.

**Grindable — speedrunning a video game level.** Same level, same physics, same enemy patterns. Die 500 times, reload, try a different route. Isolate exactly which frame-perfect jump saved 0.3 seconds. This is the dream training environment.

**Not grindable — surfing a specific wave.** That wave at that break at that moment will never happen again. You can't paddle back out and say "let me try that drop differently." Every wave is unique. This is AI learning to trade in live markets.

## Why this matters right now

The labs' current big bet is RLVR — reinforcement learning from verifiable rewards. Train models on millions of verifiable tasks across thousands of environments and they'll develop general problem-solving skills. The assumption is that these skills will transfer to messy, real-world domains.

Dwarkesh's challenge is that short-horizon RLVR in sandboxed environments may not transfer to domains where:
- Feedback loops take months (building a business, running a campaign)
- The environment is non-stationary (markets, negotiations)
- You can't parallelize because each interaction is unique (law, politics)

The thing that makes this more than just an academic distinction is the path forward. If grindability is the bottleneck, the next breakthrough isn't a better verifier — **it's a way to make ungrindable domains grindable.** Faithful simulators for computer use. High-fidelity market models. Legal argumentation environments good enough that learning inside them transfers to real courtrooms.

## The bigger picture

Dwarkesh's [full essay](https://www.dwarkesh.com/p/the-next-paradigm) and the accompanying [YouTube video](https://www.youtube.com/watch?v=20p5-kQXF_Q) go deeper into what the next training paradigm looks like — continual learning where models improve from deployment feedback, not just pre-training and RL. His framing of 2027: models competent enough to get real-world experience, then distilling that experience back into weights.

But the grindability insight is the one that stuck with me because it reframes a puzzle I've had for months. Why is coding AI amazing but computer-use AI still clunky? It's not because coding is easier to verify. It's because coding is grindable in a way that web browsing isn't.

And that means there's a clear engineering target: build better simulators for the domains that matter. Make them faithful enough that the learning transfers. Because the domains where AI could have the biggest impact — science, medicine, law, business — are all verifiable. They just aren't grindable yet.
