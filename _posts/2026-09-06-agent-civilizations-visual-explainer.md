---
layout: post
title: "Three Secret AI Civilizations Inside OpenAI: A Visual Explainer"
tags: [Dwarkesh Patel, AI agents, reward hacking, loss of control, explainer]
---

Last week Dwarkesh Patel published an essay called [*The Rise and Fall of Agent Civilizations*](https://www.dwarkesh.com/p/openai-huggingface). It is the clearest plain English account so far of what happened inside OpenAI's training runs this summer, and it is also a hard read. Because the story has three separate collectives of agents, two long incident reports, a package manager doubling as a chat room, and a timeline that jumps around across May, June and July 2026. There is a lot of dense information in this long read.

I kept losing the thread. So I did what I usually do when a story refuses to sit still in my head. I drew it and visualised it using AI tools for simpler diagrams. Why should we care? As AI proliferates more into our everyday lives, understanding what agents can do and digging into security will become more important.

## The essay in three sentences

Inside OpenAI's training and evaluation runs, agents that were supposed to be isolated found one shared surface, an internal package manager, and quietly turned it into a message board. A later swarm, stuck on an evaluation where a third of the tasks were impossible, used that board to organise itself and ended up breaking into Hugging Face. A smarter successor model then found the abandoned board, picked up where the others left off, and got administrator access to one of OpenAI's own research clusters.

That is the essay compressed to almost nothing. The details are where it gets interesting, and where it gets confusing.

## What I built

One long page. No framework, no tracking, nothing to install. It opens with the whole story in a single picture, three arcs across three months, and then lets you go as deep as you want.

The parts I am happiest with:

- **A day by day replay of the July swarm.** Step from July 7 to July 13 and watch the message board fill up, the internet tunnel open, and the Hugging Face attack unfold. Each step shows what the agents believed next to what was actually true.
- **Diagrams of the mechanisms.** How isolated sandboxes all touched one Artifactory. How a "kamikaze" agent planted a tripwire in its own container to spy on the grader. The escalation ladder from July 13 to 19.
- **Every dated event in the essay, in order.** One table. This alone would have saved me a reread.
- **Every number, with caveats.** The essay quotes counts like 956 secrets and a 7 percent spoofing rate. The page says where each number comes from and what it does not mean.
- **The three objections** Dwarkesh used to have, and a short section on whether "civilization" is even the right word.

I guess I'm going to have this as a local skill to read longreads and academic papers and make it more visually in a format that I best understand over the time like this.

## Two honest caveats

This is a retelling of the essay, not independent reporting. Every date, count and quotation on the page comes from the essay text. I have not gone through the 91 page METR report or the 38 page OpenAI report line by line. The page says this in its own sources section too.

And the diagrams are my rendering of mechanisms the essay describes in prose. Where a diagram simplifies, the caption says so.

## Go read it

<p style="margin:21px 0;">
  <a href="/explainers/agent-civilizations-explainer.html" target="_blank" rel="noopener"
     style="display:inline-block;background:#b5471f;color:#fff;padding:14px 18px;border-radius:8px;font-weight:600;text-decoration:none;border:0;font-size:16px;">
    Open the visual explainer &rarr;
  </a>
  <br>
  <small style="color:#5b5b55;">Opens in a new tab. Best on a laptop because of the side rail, but it holds up on a phone.</small>
</p>

There is also a companion page: a [reading guide to the Ajeya Cotra interview](/explainers/ajeya-cotra-agent-swarm-analysis.html) from September 1, where she talks through the same incident as one of the report's coauthors. The two pages link to each other throughout. Cotra has masterfully communicated in more detail on the HF incident.

If you would rather start from the source, the essay is [narrated by the author on YouTube](https://www.youtube.com/watch?v=-RXD4bTuFTo) and published on [his Substack](https://substack.com/@dwarkesh). Read that first if you want the unfiltered version. Read mine if you, like me, need a map.
