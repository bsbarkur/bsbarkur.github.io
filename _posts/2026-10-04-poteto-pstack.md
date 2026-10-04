---
layout: post
title: "The work behind 2,500 pull requests a month"
tags: [AI agents, software engineering, pstack, visual explainer]
---

When I first saw the claim that [Lauren Tan](https://x.com/poteto) (known online as *poteto*) is shipping 2,500 pull requests a month, my initial reaction was naturally a bit skeptical. 2,500 PRs sounds like pure noise or AI-generated slop.

So I sat down and watched her recent conversation with [Matt Pocock](https://x.com/mattpocockuk) to see how she actually pulls it off. It turns out the reality is far more interesting than a headline number. That total isn't 2,500 new product features; a substantial chunk of it is continuous maintenance, refactoring, and codebase gardening.

What caught my attention wasn't the raw volume, but *how* she built the environment around her coding agents so she could trust them to merge without reading every single diff first.

## Don't add more cooks: rearrange the kitchen

Most discussions about scaling AI agents focus on getting smarter LLM models or spinning up massive agent swarms. Lauren uses a different mental model: a Michelin-starred kitchen.

If an order backlogs in a restaurant, simply cramming twenty extra cooks into a cramped kitchen won't get meals out faster; it creates total chaos. Cooks need dedicated prep stations, standardized tools, clear workflows, and an executive chef who checks what leaves the pass.

<figure style="margin:24px 0;">
  <img src="/explainers/poteto-pstack/img/0760-kitchen.jpg" alt="Frame from the conversation with Lauren Tan and Matt Pocock discussing the Michelin kitchen analogy" style="width:100%;height:auto;border-radius:8px;border:1px solid #ddd;">
  <figcaption style="margin-top:8px;color:#5b5b55;font-size:14px;">Lauren Tan talking to Matt Pocock about setting up stations and environment guardrails for agents (<a href="https://www.youtube.com/watch?v=MN9dGgmLyso&t=760s" target="_blank" rel="noopener">12:40</a>).</figcaption>
</figure>

In her setup, the software engineer doesn't disappear; they step back into the role of the kitchen architect. You stop writing every line of code by hand and focus on designing the workspace so agents can execute cleanly.

## From "human bridge" to automated verification

When Lauren started using Cursor's agent window, she found herself stuck acting as a manual bridge. The agent would modify code, but it couldn't see what actually happened when the app ran. So she had to inspect DevTools, copy-paste error traces, check flame graphs, and feed the diagnostic data back into the prompt window.

The breakthrough came when she gave the agents their own eyes and ears through what she calls **verification**.

<figure style="margin:24px 0;">
  <img src="/explainers/poteto-pstack/img/1010-verification.jpg" alt="Frame from the conversation discussing verification and DevTools integration" style="width:100%;height:auto;border-radius:8px;border:1px solid #ddd;">
  <figcaption style="margin-top:8px;color:#5b5b55;font-size:14px;">Closing the return path: letting the agent inspect runtime traces and verify its own work (<a href="https://www.youtube.com/watch?v=MN9dGgmLyso&t=1010s" target="_blank" rel="noopener">16:50</a>).</figcaption>
</figure>

She built a lightweight CLI tool around Playwright and the Chrome DevTools Protocol. Now, instead of asking a human to test the UI, the agent launches the app, interacts with elements, reads console logs, inspects network requests, and checks if the result matches the goal.

She summarized her core philosophy in one great question:

> "How do I make the easy thing the right thing?"

Instead of writing a 10-page prompt document asking agents not to make a specific mistake, she bakes guardrails directly into the codebase. In *Dune*, her internal Electron framework, features live in predictable directories, registries discover them automatically, and strict linter rules reject anti-patterns before code gets committed.

## Codebase gardening over raw output

This is where the 2,500 PR figure starts making sense. A lot of her agents aren't writing features at all; they're doing background maintenance.

For example, she runs agents that continuously scan React code for subtle bugs or deprecated patterns. But instead of blindly creating 20 tiny pull requests that spam the repository, the agents append their observations to a shared document. Every few days, Lauren reviews the list, identifies the root cause, and resolves it with a single linter rule or structural refactor.

When agents do open PRs, she doesn't read all of them upfront. She lets verified background tasks merge overnight and uses post-merge daily sampling to spot shortcuts or recurring mistakes.

PR count is a vanity metric on its own; it doesn't tell you the risk, value, or defect rate of the code. But when paired with strong automated verification, it represents a steady beat of low-risk maintenance that keeps technical debt near zero.

## Where autopilot stops

Lauren is refreshingly pragmatic about where this approach works, and where it hits a wall.

<figure style="margin:24px 0;">
  <img src="/explainers/poteto-pstack/img/3420-limits.jpg" alt="Frame from the conversation discussing the limits of verifiability in high-stakes domains" style="width:100%;height:auto;border-radius:8px;border:1px solid #ddd;">
  <figcaption style="margin-top:8px;color:#5b5b55;font-size:14px;">Lauren discussing high-stakes domains where verification is hard and actions are irreversible (<a href="https://www.youtube.com/watch?v=MN9dGgmLyso&t=3420s" target="_blank" rel="noopener">57:00</a>).</figcaption>
</figure>

When Matt asked how this strategy applies to high-stakes fields like healthcare, legal, or financial software, she didn't pretend there's a silver bullet. Automatic verification works best when the feedback loop is fast and deterministic. A passing test suite reduces uncertainty, but it can't make an irreversible real-world mistake reversible.

Her takeaway is honest: installing `pstack` or setting up agents won't instantly transform a team. The speed comes from years of invested effort into test suites, tooling, and clean architecture.

## The visual explainer

To map out all the details of her setup, I turned the conversation into a 17-section visual explainer with process diagrams, timestamped quotes, real video frames, and source notes.

<p style="margin:21px 0;">
  <a href="/explainers/poteto-pstack/" target="_blank" rel="noopener"
     style="display:inline-block;background:#6741d9;color:#fff;padding:14px 18px;border-radius:7px;font-weight:600;text-decoration:none;border:0;font-size:16px;">
    Open the step-by-step explainer &rarr;
  </a>
  <br>
  <small style="color:#5b5b55;">Opens in a new tab. Every timestamp links back to the original video.</small>
</p>

If you have an hour, I highly recommend watching the [full conversation on YouTube](https://www.youtube.com/watch?v=MN9dGgmLyso). It's one of the best technical deep-dives into what real-world agentic software engineering actually looks like.
