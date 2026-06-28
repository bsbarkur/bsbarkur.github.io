---
layout: post
title: "Grindability: Why Verifiability Alone Won't Scale AI"
tags: [grindability, RLVR, Dwarkesh Patel, AI training, verifiability, continual learning]
---

This week, Dwarkesh Patel published a video and essay that crystallized something I have been struggling to put into words. **Verifiability is necessary for AI progress, but it's not enough. You also need grindability.**

If you have been confused about why coding models got amazing so fast while computer use still kind of sucks, even though both are easy to check for correctness, this is why.

## The insight

The conventional wisdom goes something like this: once you can verify whether an AI's output is correct, reinforcement learning can scale it. Build a verifier, run a million rollouts, keep what works. This is how AlphaGo learned to demolish the world's best Go players. It is how GPT-4 learned to write code that compiles.

Dwarkesh points out that there is a hidden second condition hiding in plain sight. The domain doesn't just need to be verifiable. It also needs to be what he calls grindable. You need to be able to spin up thousands of parallel attempts against a deterministic environment that you can replay identically, so you can isolate exactly what the model did that actually worked.

> "It is not enough for a domain to be verifiable. It also has to be very grindable — in the sense that you can run lots of parallel rollouts against a deterministic and replayable simulator."

## The split

Some domains are both. That is where AI has made the most progress.

| Domain | Verifiable? | Grindable? | Why |
|---|---|---|---|
| **Coding** | ✅ Tests pass/fail | ✅ Spin up 1000 identical containers, let agents fix the same bug in parallel | The environment is deterministic, sandboxed, infinitely replayable |
| **Math** | ✅ Answer is right or wrong | ✅ Generate 1000 problems, run 1000 agents, check instantly | No environment needed, just problems and answers |
| **Chess / Go** | ✅ Win or lose | ✅ Same board state replayed 1000 times against the same engine | Perfect deterministic simulator |
| **Computer use** | ✅ Did you complete the checkout? | ❌ Amazon detects your bots, accounts get banned, flows change | "You can't have a thousand agents go try the same checkout flow on Amazon. Andy Jassy will find and shut your ass down." |
| **Day trading** | ✅ Did you profit? | ❌ Same market conditions never repeat, your own trades change the market | The market only happened once |
| **Court cases** | ✅ Did you win? | ❌ Can't retry the same case 1000 times with different arguments | Each case is unique, you only get one shot |
| **Building a business** | ✅ Is it profitable? | ❌ Customer feedback takes months or years | "The outer loop verification may take months or years of real world actions to elicit, and cannot be re observed by perturbing the model's actions thousands of times in parallel" |

Coding is the poster child. Tests tell you if the solution works. You can spin up a thousand identical Docker containers, each with a repo that has a missing feature, and let a thousand agents attack the same problem in parallel. The environment costs nothing to replicate. You can do this all day.

Math works the same way. Generate a thousand problems, run a thousand agents, check the answers instantly. No environment needed at all.

Chess and Go are the original examples. The same board state can be replayed a thousand times against the same engine. AlphaGo Zero trained by playing 44 million games against itself, each one a perfect simulation.

Now look at domains where AI has moved slower.

Computer use. You cannot have a thousand agents try the same checkout flow on Amazon. As Dwarkesh put it, Andy Jassy will detect your bots and shut your ass down. The environment fights back. Accounts get banned. Flows change. You get one shot per account.

Day trading. Same problem in a different disguise. You can verify whether a strategy made money. But the same market conditions never repeat. Your own trades change the market. The only way to run that Tuesday again is if you built a time machine.

Court cases. Win or lose is perfectly verifiable. But you cannot retry the same case a thousand times with different arguments to see which one works. The judge and jury only happen once.

Building a business. The feedback loop takes months or years. You cannot perturb the model's decisions and re observe the outcome thousands of times in parallel to isolate what worked. The market moves on.

## Analogies that make it concrete

Here is the way I think about it.

Practicing penalty kicks is grindable. Same goal, same ball, same spot. You take a thousand shots. Each miss tells you something specific. My follow through was off on that one. You isolate the variable and fix it. This is how AI learns to code.

Playing in a World Cup final is not grindable. One match, one chance. The opponent adjusts to what you do. The crowd changes your decisions. You cannot replay the same match a thousand times to figure out which formation works best. This is AI trying to navigate Amazon's checkout flow.

Speedrunning a video game level is grindable. Same level, same physics, same enemy patterns. You die five hundred times, reload, try a different route. You isolate exactly which frame perfect jump saved you 0.3 seconds. This is the dream training environment for RL.

Surfing a specific wave is not grindable. That wave at that break at that moment will never happen again. You cannot paddle back out and say let me try that drop differently. Every wave is unique. This is AI learning to trade in live markets.

## Why this matters right now

The labs are betting big on RLVR, reinforcement learning from verifiable rewards. The thesis is that training models on millions of verifiable tasks across thousands of environments will produce general problem solving skills that transfer to messy real world domains.

Dwarkesh's challenge is worth sitting with. Short horizon RLVR in sandboxed environments might not transfer to domains where feedback loops take months, the environment keeps moving under you, or you cannot parallelize because every interaction is unique.

The reason this matters beyond being a fun mental model is that it points to a clear engineering target. If grindability is the bottleneck, the next breakthrough is not a better verifier. It is a way to make ungrindable domains grindable. Faithful simulators for computer use. High fidelity market models. Legal argumentation environments realistic enough that learning inside them transfers to real courtrooms.

## The bigger picture

Dwarkesh's [full essay](https://www.dwarkesh.com/p/the-next-paradigm) and the [YouTube video](https://www.youtube.com/watch?v=20p5-kQXF_Q) go deeper into what the next training paradigm looks like, continual learning where models improve from deployment feedback instead of just pre training and RL. His 2027 picture is one where models are competent enough to get real world experience and then distill that experience back into their weights.

But the grindability idea is the one that stuck with me because it reframes a puzzle I have had for months. Why is coding AI amazing but computer use still clunky? It is not because coding is easier to check. It is because coding is grindable in a way that web browsing is not.

And that means there is something concrete to build. Better simulators for the domains that matter. Make them faithful enough that the learning transfers. Because the domains where AI could have the biggest impact, science, medicine, law, business, are all verifiable. They just are not grindable yet.
