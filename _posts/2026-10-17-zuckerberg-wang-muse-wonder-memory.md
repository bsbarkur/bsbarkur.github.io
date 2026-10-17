---
layout: post
title: "Muse Wonder, memory, and Meta's $100 billion bet"
date: 2026-10-17 08:00:00 +0530
tags: [AI agents, Meta, Muse, memory, OpenClaw, visual explainer]
---

This week [John Collison](https://x.com/collision) had [Mark Zuckerberg](https://x.com/finkd) and [Alexandr Wang](https://x.com/alexandr_wang) on [Cheeky Pint](https://cheekypint.substack.com) for a little over two hours. The occasion was the launch of Muse Wonder ([Meta's research post](https://research.meta.ai/blog/muse-wonder), [announcement](https://about.fb.com/news/2026/10/introducing-muse-wonder-models/)), the first model from [Meta Superintelligence Labs](https://en.wikipedia.org/wiki/Meta_Superintelligence_Labs) that Meta says is at the frontier rather than chasing it. I watched the whole thing and pulled the captions so I could check quotes against what was actually said. What I was most curious about was memory, because that is the part of agents that breaks first for me.

## What they launched

Muse Wonder is a new and much bigger pre-train than [Muse Spark](https://about.fb.com/news/2026/04/introducing-muse-spark/), the model behind the Muse app so far. Internally Spark was Avocado and Wonder was Watermelon. There are two versions, Wonder and Wonder Full. In the Muse app it starts on paid plans because it costs more to serve, and free users stay on Spark for now. Developers get it through the Meta Model API and [OpenRouter](https://openrouter.ai), and [Muse Code](https://developer.meta.com/ai/resources/blog/build-with-muse-code/) users get it too.

Alex listed four things they built in. Active memory is the first. Active focus is their name for staying on a long task and finding another route when something fails. Domain expertise covers legal, finance, tax and health questions, and they claim state of the art on several finance and legal evals. Coding is the fourth, where they say it leads on [DeepSWE](https://deepswe.datacurve.ai/) and [SWE-Marathon](https://arxiv.org/abs/2606.07682). Coding is not the point of the product, though. Alex said coders "will be a minority of all of our users".

## Memory is the real pitch

The most useful ten minutes for me were about memory. John asked how memory is different from context. Alex's answer was that it is about the experience over time:

> "How well does it remember things that you mentioned to it over days, weeks, or months, and then leverage those in the process of doing future tasks?"

Then he named [OpenClaw](https://github.com/openclaw/openclaw) directly. He said they are fans and use it, but "it ultimately does this compaction where it has like the long-term memory and the working short-term memory or whatever", and that once people use it for a while "they find the memory pretty buggy." In OpenClaw the compaction lives in the harness, the code wrapped around a general model. Meta says it moved that job into the model. Mark said the compaction is something "the model does", and John checked the point back: "It's not a harness, it's in the model itself." Alex confirmed it. They described the result as both "the model and a system for the model to use", which they call active memory, and they measure it with an internal eval called Memory Bench.

<figure style="margin:28px auto;text-align:center;max-width:100%;">
  <a href="/explainers/zuckerberg-100b-bet/#memory" target="_blank" rel="noopener"><img src="/explainers/zuckerberg-100b-bet/img/openclaw-vs-muse-memory.png" alt="Diagram comparing OpenClaw's harness-level memory compaction with Muse Wonder's active memory trained into the model" style="display:block;margin:0 auto;width:100%;height:auto;border-radius:8px;border:1px solid #ddd;"></a>
  <figcaption style="margin-top:8px;color:#5b5b55;font-size:14px;text-align:center;">OpenClaw keeps memory in the harness around the model. Meta says Muse Wonder's memory is trained into the model itself. From the <a href="/explainers/zuckerberg-100b-bet/#memory" target="_blank" rel="noopener">explainer</a>.</figcaption>
</figure>

They didn't show numbers for Memory Bench, and Alex said himself that "there's still plenty of room to improve." But the claim is specific enough to test. If memory really is trained into the model, it should hold up after weeks of use in a way harness compaction doesn't. That is the thing we can try checking first, perhaps.

There is a second part to the memory story that I hadn't thought about. Memory is what lets an agent be discreet. Mark's example was booking a restaurant when you are pregnant or have an allergy. The agent needs to know that about you and find a suitable place without telling the restaurant your medical details. A coding agent never needs that skill, because everyone in a repo shares the same context.

## "The model is the product"

John pushed the idea that AI products converge quickly on one obvious shape, first chat and now the persistent agent with a computer. Mark and Alex disagreed. Their view is that the differences are real but sit inside the model: how proactive it is, when it speaks up, how it talks to you. Mark said that when [Nat Friedman](https://x.com/natfriedman) thinks about making the product feel better, "he's talking about post-training the model". The work outside the model goes mostly into privacy and security, like giving each agent its own VM instead of shared compute.

This is also why Muse has no model picker and no effort picker. Alex was blunt about how the rest of the industry names models: "the naming of some of the models is like so egregious."

## The bet

The title comes from Mark describing the end of last year, when they had to commit "like $100 billion, $200 billion" before seeing whether the models worked. He put rough numbers on it. Running a one gigawatt cluster for a year costs on the order of $15 billion, and building one costs maybe $50 or $60 billion up front. [Prometheus in Ohio](https://techcrunch.com/2025/07/14/mark-zuckerberg-says-meta-is-building-a-5gw-ai-data-center) is past a gigawatt, and [Hyperion in Louisiana](https://techcrunch.com/2025/07/14/mark-zuckerberg-says-meta-is-building-a-5gw-ai-data-center) will scale to five. Alex told the story of Meta's CFO, Susan Li, taking him aside at the Hawaii offsite to say that if the model was bad it would be "really, really, really bad." Mark's reasoning for why it was less risky than it looked was that "it's not just that we have to fail, AI overall has to fail", because someone else would rent the compute.

## Agents that buy things

The commerce section was the most concrete. Mark signed up for [Instacart](https://www.instacart.com) so his Muse could order baking ingredients for weekends with his three-year-old daughter, and now he orders once or twice a week instead of twice a year. His question was whether that account is "really for me, or is it for the Muse?" If agents hold their own accounts, commerce grows instead of just moving between channels. The path they described starts with the agent using websites through the browser, then moves to direct APIs with partners, and ends with every business running its own agents. John said [Stripe's agentic commerce](https://stripe.com/blog/everything-we-announced-at-sessions-2026) numbers only started "going vertical" in September.

## Why it won't be one winner

Alex gave three reasons no lab has run away with the lead. Every lab has stretches of weak execution. [Distillation](https://en.wikipedia.org/wiki/Knowledge_distillation) keeps working, so small models catch up fast. And people choose where to work and who to buy from. Mark added a safety argument. He thinks a world with many models is safer than "a couple of good models that are only in a couple companies' hands."

## How MSL was built

The last forty minutes were about the lab itself. Mark said "we were not happy with the trajectory that we were on with Llama 4" ([the Llama 4 launch post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)), started calling people for advice in April 2025, and ended up with Alex. The research team is under 200 people. They decided on a full reboot while recruiting, because the best people wanted to start fresh. There are no non-technical managers and no deadlines. Alex told a story about one of his first meetings, where a lead from the old effort argued that consumers didn't need a reasoning model, and his reaction was "Oh, no, I've joined the wrong company."

## What I am taking away

I don't think the interesting claim here is the benchmark position. Everyone says they are at the frontier. The interesting claim is that memory, discretion and persistence can be trained into the model and not bolted on with a harness, and that this is where personal agents will be won. I run my own agents with harness-level memory, and the failures they describe are the ones I see. If Wonder actually fixes that, it matters more than any leaderboard.

## The visual explainer

I turned the episode into a long visual page. It has the twelve main points, a memory diagram, a comparison of chat assistants, OpenClaw, Muse on Spark, Muse on Wonder and the other labs, a walkthrough of how they say frontier models are built, and the trends they point to. Every quote is copied exactly from the captions and links to its timestamp. The speaker labels are my best guess from the flow of the conversation, so treat short interjections with some care.

<p style="margin:21px 0;">
  <a href="/explainers/zuckerberg-100b-bet/" target="_blank" rel="noopener"
     style="display:inline-block;background:#c2410c;color:#fff;padding:14px 18px;border-radius:7px;font-weight:600;text-decoration:none;border:0;font-size:16px;">
    Open the Muse Wonder explainer &rarr;
  </a>
  <br>
  <small style="color:#5b5b55;">Opens in a new tab. Every timestamp links back to the original video.</small>
</p>

If you have two hours, watch the [full conversation on YouTube](https://www.youtube.com/watch?v=XfdIhd1g6EI). The podcast also publishes an [official transcript on Substack](https://cheekypint.substack.com/p/muse-wonder-zuckerberg-and-wangs).
