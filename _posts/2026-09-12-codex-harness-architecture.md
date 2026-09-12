---
layout: post
title: "Inside the Codex Harness: A Visual Architecture Guide"
tags: [Codex, AI agents, software architecture, developer tools, visual explainer]
---

A coding agent looks simple from the chat window. You ask for a change, it reads a few files, runs commands, and gives you an answer. The interesting part is everything around the model that makes those actions controlled, observable, and recoverable.

I wanted a map of that surrounding system. So I read through the open source Codex repository and drew the main boundaries: where client requests enter, how a turn gets its context, how tool calls cross policy gates, what gets saved, and how one agent can delegate bounded work to another.

## The idea that made the architecture click for me

The model chooses the next move. The harness makes the move real.

A model can write a shell command as text, but that text has no effect by itself. The harness has to recognize a structured tool request, check it against the active sandbox and approval policy, execute it, capture the result, and feed that result into the next model call.

That creates a loop:

1. Assemble instructions, history, tools, configuration, and the current request.
2. Ask the model what to do next.
3. Evaluate any requested action under the active policy.
4. Run the tool and capture its output.
5. Add the result to the session record.
6. Continue the loop or finish the turn.

<figure style="margin:28px 0;padding:18px;background:#fffdf8;border:1px solid #d7cfbf;border-radius:10px;overflow-x:auto;">
<svg viewBox="0 0 760 230" role="img" aria-labelledby="post-loop-title post-loop-desc" style="display:block;width:100%;min-width:620px;height:auto;color:#20221f;">
<title id="post-loop-title">The Codex model and tool loop</title>
<desc id="post-loop-desc">Context passes to the model, through policy and a tool, then the result returns to context for another decision.</desc>
<defs><marker id="post-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#2456a6"/></marker></defs>
<g font-family="system-ui,sans-serif" fill="#20221f">
<g fill="#fffdf8" stroke="#d7cfbf" stroke-width="1.5"><rect x="25" y="55" width="125" height="68" rx="9"/><rect x="218" y="55" width="125" height="68" rx="9"/><rect x="410" y="55" width="125" height="68" rx="9"/><rect x="610" y="55" width="125" height="68" rx="9"/></g>
<g font-size="12" font-weight="700" text-anchor="middle"><text x="87" y="94">Context</text><text x="280" y="94">Model</text><text x="472" y="94">Policy + tool</text><text x="672" y="94">Result</text></g>
<g stroke="#2456a6" stroke-width="2" fill="none" marker-end="url(#post-arrow)"><path d="M150 89H218"/><path d="M343 89H410"/><path d="M535 89H610"/><path d="M672 123V154C672 206 87 206 87 154V123"/></g>
<text x="379" y="182" text-anchor="middle" font-size="11" fill="#386853">evidence changes what the model knows next</text>
</g></svg>
<figcaption style="margin-top:10px;color:#65645d;font-size:14px;line-height:1.5;"><strong style="color:#20221f;">The core rhythm.</strong> A tool result becomes evidence for the next model request. The detailed explainer turns this into a seven-step interactive replay.</figcaption>
</figure>

The visual explainer lets you step through this loop one stage at a time. It also separates concepts that are easy to blur together, such as a sandbox and an approval policy, or an instruction in `AGENTS.md` and a model field passed by an orchestrator.

## What the guide covers

The page moves from a plain English orientation into the implementation boundaries:

- The client surfaces and protocol layer around the core session loop
- The inputs used to construct a turn context
- A step-through diagram of the model and tool loop
- The separate roles of sandboxing, approvals, and command policy
- Rollout records, resume, fork, steering, cancellation, and compaction
- A planner and worker pattern using explicit model and reasoning fields
- When to use an `AGENTS.md` file, skill, plugin, MCP server, connector, or hook

Each figure states what it simplifies. The diagrams show conceptual flow and responsibility rather than pretending to be a complete Rust call graph.

## The boundaries that mattered most

The current [terminal interface](https://github.com/openai/codex/blob/944d6fd1ba4baab69dbedd205282dc72ec20abb5/codex-rs/tui/src/lib.rs) and headless command are clients of App Server infrastructure. That boundary can exist inside one operating-system process. An external application can use the App Server protocol, while the [TypeScript SDK](https://github.com/openai/codex/blob/944d6fd1ba4baab69dbedd205282dc72ec20abb5/sdk/typescript/src/exec.ts#L92) takes another route by launching `codex exec` and exchanging JSONL events. The model connection is separate again: model requests use Responses API transport, not the client-facing application protocol. The repeated sampling and tool cycle lives in the [turn loop](https://github.com/openai/codex/blob/944d6fd1ba4baab69dbedd205282dc72ec20abb5/codex-rs/core/src/session/turn.rs).

The storage picture also needed more than one box. The active session and model context live in memory. In the [local persistence path](https://github.com/openai/codex/blob/944d6fd1ba4baab69dbedd205282dc72ec20abb5/codex-rs/thread-store/src/local/mod.rs), JSONL rollouts remain the canonical replay record, while SQLite supports metadata queries and projected history. The working project is a third system. Resuming a conversation can restore its history, but it cannot undo a file write or retract an effect in an external service.

Permissions have a similar split. Approval policy decides whether an action may proceed. A sandbox constrains what an executing process can access. The [inspected backends](https://github.com/openai/codex/blob/944d6fd1ba4baab69dbedd205282dc72ec20abb5/codex-rs/core/README.md) vary by platform: macOS uses Seatbelt, Linux supports bubblewrap and a legacy Landlock path, and Windows has its own implementations. MCP-backed services have their own tool contracts and authorization boundaries, so a local shell sandbox is not a universal wrapper around every external action.

## A concrete orchestration example

This workspace uses a root agent for research, planning, and review, then delegates code changes to a bounded implementation worker. The written policy helps every agent understand that split. The orchestration call still has to set the worker's actual `model` and `reasoning_effort` fields.

That distinction matters. Prose can say which model should do the job. Only the runtime assignment makes that choice concrete.

The page includes a source-checked three-file example: project configuration for Astra at high reasoning effort, an implementer role fixed to Sol at medium effort, and an `AGENTS.md` policy that tells the root to delegate every edit. There is also a downloadable bundle. The TOML was parsed and checked against the repository schema, but I did not run a fresh authenticated installation test. Model and multi-agent access still depend on the installed client, effective settings, and account.

The stronger reading of “always delegate” needs one further step. A prose instruction does not remove the root agent's write tools. If this split is a hard product requirement, the host application should enforce it by controlling which identity receives write-capable tools and by validating child model settings at the orchestration boundary.

## Read the explainer

<p style="margin:21px 0;">
  <a href="/explainers/codex-harness-architecture.html" target="_blank" rel="noopener"
     style="display:inline-block;background:#a74723;color:#fff;padding:14px 18px;border-radius:7px;font-weight:600;text-decoration:none;border:0;font-size:16px;">
    Open the visual architecture guide &rarr;
  </a>
  <br>
  <small style="color:#5b5b55;">Opens in a new tab. The page is self-contained, responsive, printable, and uses no external JavaScript or web fonts.</small>
</p>

The guide is based on the checked-out repository snapshot dated September 12, 2026. Codex changes quickly, so use the source links at the end of the page when you need exact details for the inspected snapshot.

Related reading: [From the Codex Harness to Cloud Agents: Using the Agents API](/2026/09/12/agents-api-codex-harness.html) follows these boundaries into managed and self-hosted Agents API sessions.
