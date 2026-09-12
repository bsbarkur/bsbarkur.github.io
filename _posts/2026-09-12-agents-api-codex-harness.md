---
layout: post
title: "From the Codex Harness to Cloud Agents: Using the Agents API"
date: 2026-09-12 09:14:00 +0530
tags: [Codex, Agents API, AI agents, software architecture, developer tools]
---

In the [Codex architecture post](/2026/09/12/codex-harness-architecture.html), I mapped the runtime around the model: the loop, tools, permissions, context, and conversation state. OpenAI's [Agents API announcement](https://openai.com/index/introducing-the-agents-api/), published on September 10, 2026, makes that architecture available through a managed service.

The connection is direct: the Agents API runs a Codex harness for your application. OpenAI operates that runtime; you supply the task and integrations, and choose where code executes. This post follows that boundary from an API request to a repository in your own environment, then revisits the Astra planner and Sol implementer example from the original guide. [Agents API overview](https://developers.openai.com/api/docs/guides/agents-api/overview)

## Where the existing architecture fits

There are two deployment choices worth keeping separate. You can operate Codex yourself and integrate through its CLI, SDK, or App Server. Alternatively, you can use the Agents API and let OpenAI operate the harness. The public repository explains the foundation, but it does not establish that the hosted service runs the exact commit inspected in the earlier article.

Our source snapshot already contains the executor machinery for remote environments. Its `codex-exec-server` crate handles process and filesystem operations and documents remote registration and relay requirements. That supports the architectural connection; compatibility with a live hosted service still depends on the installed executor and service. [Pinned executor source](https://github.com/openai/codex/blob/944d6fd1ba4baab69dbedd205282dc72ec20abb5/codex-rs/exec-server/README.md)

<figure style="margin:28px 0;padding:18px;background:#fffdf8;border:1px solid #d7cfbf;border-radius:10px;">
<div class="agents-api-figure-title" style="margin-bottom:12px;color:#a74723;font:700 12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.07em;text-transform:uppercase;">Figure 1 · Who owns what in a self-hosted session</div>
<div class="agents-api-figure-scroll" style="max-width:100%;overflow-x:auto;padding:2px 0 8px;">
<svg viewBox="0 0 840 390" role="img" aria-labelledby="agents-ownership-title agents-ownership-desc" style="display:block;width:100%;min-width:620px;height:auto;color:#20221f;">
<title id="agents-ownership-title">Application, managed harness, and self-hosted environment ownership</title>
<desc id="agents-ownership-desc">The application sends tasks to the managed Agents API and receives events. An executor in the application's isolated environment opens an outbound connection to the service. The managed harness then exchanges commands and results with that executor.</desc>
<defs>
<marker id="agents-own-blue-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#2456a6"/></marker>
<marker id="agents-own-copper-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#a74723"/></marker>
<marker id="agents-own-green-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#386853"/></marker>
</defs>
<g font-family="system-ui,sans-serif" fill="#20221f">
<g stroke-width="1.5">
<rect x="20" y="54" width="190" height="226" rx="10" fill="#f8e4d8" stroke="#a74723"/>
<rect x="290" y="32" width="260" height="270" rx="10" fill="#e4ebf7" stroke="#2456a6"/>
<rect x="630" y="54" width="190" height="226" rx="10" fill="#deeee6" stroke="#386853"/>
<rect x="316" y="103" width="208" height="150" rx="8" fill="#fffdf8" stroke="#d7cfbf"/>
</g>
<g text-anchor="middle">
<text x="115" y="86" font-size="12" font-weight="700" fill="#a74723">YOUR APPLICATION</text>
<text x="115" y="122" font-size="16" font-weight="700">Backend</text>
<text x="115" y="150" font-size="12">submits tasks</text>
<text x="115" y="171" font-size="12">consumes events</text>
<text x="115" y="192" font-size="12">handles function tools</text>
<text x="420" y="65" font-size="12" font-weight="700" fill="#2456a6">OPENAI SERVICE</text>
<text x="420" y="91" font-size="16" font-weight="700">Agents API</text>
<text x="420" y="132" font-size="13" font-weight="700">Managed Codex harness</text>
<text x="420" y="164" font-size="12">model and tool loop</text>
<text x="420" y="187" font-size="12">context</text>
<text x="420" y="210" font-size="12">session</text>
<text x="725" y="80" font-size="12" font-weight="700" fill="#386853">YOUR ISOLATED</text>
<text x="725" y="96" font-size="12" font-weight="700" fill="#386853">ENVIRONMENT</text>
<text x="725" y="126" font-size="15" font-weight="700">codex exec-server</text>
<text x="725" y="158" font-size="12">repository</text>
<text x="725" y="180" font-size="12">dependencies</text>
</g>
<path d="M210 112H290" fill="none" stroke="#a74723" stroke-width="2.5" marker-end="url(#agents-own-copper-arrow)"/>
<text x="250" y="101" text-anchor="middle" font-size="11" fill="#a74723">task</text>
<path d="M290 156H210" fill="none" stroke="#2456a6" stroke-width="2.5" marker-end="url(#agents-own-blue-arrow)"/>
<text x="250" y="176" text-anchor="middle" font-size="11" fill="#2456a6">events</text>
<path d="M630 202H550" fill="none" stroke="#386853" stroke-width="2.5" stroke-dasharray="7 6" marker-end="url(#agents-own-green-arrow)"/>
<text x="590" y="188" text-anchor="middle" font-size="10.5" fill="#386853">executor opens</text>
<path d="M550 237H630" fill="none" stroke="#2456a6" stroke-width="2.5" marker-end="url(#agents-own-blue-arrow)"/>
<path d="M630 262H550" fill="none" stroke="#2456a6" stroke-width="2.5" marker-end="url(#agents-own-blue-arrow)"/>
<text x="590" y="229" text-anchor="middle" font-size="10.5" fill="#2456a6">commands</text>
<text x="590" y="280" text-anchor="middle" font-size="10.5" fill="#2456a6">results</text>
<rect x="20" y="330" width="800" height="40" rx="7" fill="#f6f1e8" stroke="#d7cfbf"/>
<text x="420" y="355" text-anchor="middle" font-size="12" font-weight="700">self_hosted deployment · application owns environment lifecycle</text>
</g>
</svg>
</div>
<figcaption style="margin-top:10px;color:#65645d;font-size:14px;line-height:1.5;"><strong style="color:#20221f;">The harness remains managed even when execution happens on your infrastructure.</strong> Connection direction and command direction are different: the executor opens the channel, then the harness sends work over it. Sources: <a href="https://developers.openai.com/api/docs/guides/agents-api/architecture">architecture</a> and <a href="https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted">self-hosted environments</a>.</figcaption>
</figure>

The [architecture documentation](https://developers.openai.com/api/docs/guides/agents-api/architecture) separates the application server, harness, and environment. The application submits work, consumes events, and handles custom function tools. The environment supplies compute and files when a task needs them.

`codex app-server` and `codex exec-server` have different jobs. App Server exposes Codex conversation and application operations. The execution server supplies process and filesystem capabilities to a harness. The Agents API uses its own HTTP resources and event types, so the previous guide's App Server messages should be mapped to the new interface deliberately. [App Server documentation](https://learn.chatgpt.com/docs/app-server)

## Choose where the agent works

| Environment | What it provides | A reasonable use |
| --- | --- | --- |
| `none` | A managed session with configured service tools, without a workspace or built-in Bash or apply-patch execution. | Reviewing supplied text or querying remote tools. |
| `openai_hosted` | An OpenAI-managed Linux workspace, configurable files, packages, and setup commands. | Producing a report or testing a small supplied project. |
| `self_hosted` | Your compute and files, connected through an executor. | Working against a prepared repository or private dependencies. |

These are execution choices. OpenAI operates the harness in all three. Remote MCP tools can be called from the service; application-defined functions need your application to run the function and return its result. [Architecture](https://developers.openai.com/api/docs/guides/agents-api/architecture)

For the hosted option, packages and supplied files are prepared before setup commands. A setup failure prevents the agent from starting. Environment templates reuse configuration; they do not preserve a live workspace. [OpenAI-hosted sandboxes](https://developers.openai.com/api/docs/guides/agents-api/environments/openai-hosted)

## Connect a repository through the executor

Consider the illustrative CSV export bug from the original guide: a name containing a comma causes incorrect columns. Assume an application has prepared an isolated checkout at `/workspace`, installed its test dependencies, and obtained a bounded implementation brief. The example paths below stand in for that application's repository.

First, create the session from your backend. The documented prerequisites are an application key with `api.agents.read`, `api.agents.write`, and `api.responses.write`, plus an OpenAI SDK version that exposes `beta.agents`. The SDK supplies the beta header; direct HTTP requests require `OpenAI-Beta: agents=v1`. Keep the application key outside the execution environment. [Quickstart](https://developers.openai.com/api/docs/guides/agents-api/quickstart)

Prepare the checkout, dependencies, and executor before creating the session because queued initial input waits up to five minutes for the environment to connect. [Sandbox lifecycle](https://developers.openai.com/api/docs/guides/agents-api/environments/lifecycle)

```javascript
import OpenAI from "openai";

const client = new OpenAI();

// Illustrative paths. A prepared checkout and dependencies must exist at /workspace.
const session = await client.beta.agents.sessions.create({
  agent: {
    model: "gpt-5.6-sol",
    reasoning: { effort: "medium" },
    multi_agent: { enabled: false },
    instructions:
      "Change only the assigned source and test files; save the requested patch and report separately. Run relevant tests and report the evidence.",
  },
  environment: {
    type: "self_hosted",
    workspace_directory: "/workspace",
  },
  input: [
    {
      role: "user",
      content: [
        {
          type: "input_text",
          text:
            "Fix CSV quoting in illustrative src/exportCsv.js and test/exportCsv.test.js. Preserve the exported API. Cover commas, quotes, and newlines. Write the patch and report to /workspace/outputs. Do not publish or merge.",
        },
      ],
    },
  ],
});

if (session.environment?.type !== "self_hosted") {
  throw new Error("Expected a self-hosted environment");
}

console.log({
  sessionId: session.id,
  environmentId: session.environment.id,
  remoteUrl: session.environment.remote_url,
});
```

The session holds configuration and conversation state. Save the session ID and environment details, then open the session's event stream before starting the executor. The initial input is already queued and can wait for the environment connection. Creating the session does not prove that a checkout exists, that an executor is connected, or that a patch has been produced. [Sessions](https://developers.openai.com/api/docs/guides/agents-api/sessions)

The prepared environment should already contain the executor version recommended by the current guide. At the time of writing, that guide uses `@openai/codex@alpha`. Supply a restricted environment key as `CODEX_API_KEY`. After creating the session and opening its event stream, connect using the returned environment ID and remote URL. [Self-hosted setup](https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted)

```bash
# CODEX_API_KEY is supplied through environment secret injection.
# Preparation, before session creation:
npm install -g @openai/codex@alpha
# Start after session creation and after opening the event stream:
codex exec-server \
  --remote "<session.environment.remote_url>" \
  --environment-id "<session.environment.id>"
```

The prepared `/workspace` checkout is a precondition; these commands do not create it.

The executor registers and makes outbound connections to the service, then receives execution requests over the connection. The remote URL must be used unchanged. Each self-hosted session receives its own environment ID and needs its own executor. [Connection contract](https://developers.openai.com/api/docs/guides/agents-api/environments/self-hosted)

Keep the executor running while work is pending. Your application should save the session-to-compute mapping and coordinate startup and shutdown with incoming requests. Deleting a session does not stop self-hosted compute. Replacing compute while retaining an environment ID does not restore the old files. [Sandbox lifecycle](https://developers.openai.com/api/docs/guides/agents-api/environments/lifecycle)

## Watch the outcome and collect the patch

Subscribe to the event stream before sending subsequent input. Follow the root turn's completion, failure, or cancellation, and inspect its saved messages and tool results. An idle session or a closed stream does not prove that the task succeeded. Streams do not replay missed events. Reconnect and buffer incoming events, retrieve the session and saved items, then merge buffered updates by item ID. [Events and items](https://developers.openai.com/api/docs/guides/agents-api/sessions/events)

File retrieval depends on the environment. For `self_hosted`, collect the patch and report using your provider's file API or mounted filesystem. Writing into `/workspace/outputs` does not publish self-hosted files through the Artifacts API. In `openai_hosted`, files under that directory are published as immutable artifacts when a turn completes. Those copies can outlive the sandbox. [Files and artifacts](https://developers.openai.com/api/docs/guides/agents-api/environments/files)

For this maintenance workflow, my application would retrieve the diff, run its acceptance checks, and present a reviewable change. Creating a pull request or merging it would be a separate application action with its own authority. This is a proposed workflow, not a claim that the example has completed a real repair.

## Carry the Astra/Sol policy across carefully

The earlier guide uses Astra at high reasoning effort to plan and review, with Sol at medium effort for code changes. A local `AGENTS.md` instruction expresses that policy, while local role configuration and orchestration fields select models.

The Agents API supports built-in delegation through `agent.multi_agent.enabled` and `max_concurrent_subagents`. Its harness supplies the coordination tools. Subagents have their own context but share the session's environment; they inherit configured MCP access and web-search settings. The current guide says subagents do not support application function tools. [Multi-agent guide](https://developers.openai.com/api/docs/guides/agents-api/multi-agent)

There is a subtle detail in the reference: a `create_subagent_call` item can record requested `model` and `reasoning_effort` values. That is evidence of per-spawn requests, not a documented policy table that locks an implementer role to Sol. The session's `multi_agent` configuration exposes enablement and concurrency controls. I would not treat the local TOML role map as automatically portable, or treat an observed request as proof that a policy was enforced. [Agents reference](https://developers.openai.com/api/reference/resources/beta/subresources/agents)

For an application that must enforce the split, I would make the application own the handoff:

1. Create an Astra/high planning session using supplied repository evidence or explicitly read-only tools.
2. Validate its brief, then create a separate Sol/medium implementation session with an isolated working checkout.
3. Retrieve the diff and test evidence, then submit them to Astra for review.
4. Create another bounded Sol task if corrections are required.

<figure style="margin:28px 0;padding:18px;background:#fffdf8;border:1px solid #d7cfbf;border-radius:10px;">
<div class="agents-api-figure-title" style="margin-bottom:12px;color:#a74723;font:700 12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.07em;text-transform:uppercase;">Figure 2 · A proposed application-controlled handoff</div>
<div class="agents-api-figure-scroll" style="max-width:100%;overflow-x:auto;padding:2px 0 8px;">
<svg viewBox="0 0 840 440" role="img" aria-labelledby="agents-handoff-title agents-handoff-desc" style="display:block;width:100%;min-width:620px;height:auto;color:#20221f;">
<title id="agents-handoff-title">A proposed application-controlled Astra to Sol handoff</title>
<desc id="agents-handoff-desc">An application controller creates separate Astra planning, Sol implementation, and Astra review sessions. It transfers a validated brief to Sol and returns the diff and test evidence to Astra. Only the Sol session has a write-capable source-control environment.</desc>
<defs>
<marker id="agents-route-blue-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#2456a6"/></marker>
<marker id="agents-route-copper-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#a74723"/></marker>
<marker id="agents-route-green-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto" overflow="visible"><path d="M0 0L10 5L0 10z" fill="#386853"/></marker>
</defs>
<g font-family="system-ui,sans-serif" fill="#20221f">
<rect x="30" y="24" width="780" height="72" rx="10" fill="#f8e4d8" stroke="#a74723" stroke-width="1.5"/>
<text x="420" y="53" text-anchor="middle" font-size="15" font-weight="700">Application controller</text>
<text x="420" y="76" text-anchor="middle" font-size="11.5">sets session models · coordinates handoff · checks results</text>
<path d="M125 96V132" fill="none" stroke="#a74723" stroke-width="2.5" marker-end="url(#agents-route-copper-arrow)"/>
<path d="M420 96V132" fill="none" stroke="#a74723" stroke-width="2.5" marker-end="url(#agents-route-copper-arrow)"/>
<path d="M715 96V132" fill="none" stroke="#a74723" stroke-width="2.5" marker-end="url(#agents-route-copper-arrow)"/>
<g stroke-width="1.5">
<rect x="30" y="132" width="190" height="105" rx="9" fill="#e4ebf7" stroke="#2456a6"/>
<rect x="325" y="132" width="190" height="105" rx="9" fill="#deeee6" stroke="#386853"/>
<rect x="620" y="132" width="190" height="105" rx="9" fill="#e4ebf7" stroke="#2456a6"/>
<rect x="30" y="302" width="190" height="65" rx="8" fill="#f6f1e8" stroke="#d7cfbf"/>
<rect x="325" y="286" width="190" height="98" rx="8" fill="#fffdf8" stroke="#386853"/>
<rect x="620" y="302" width="190" height="65" rx="8" fill="#f6f1e8" stroke="#d7cfbf"/>
</g>
<g text-anchor="middle">
<text x="125" y="166" font-size="14" font-weight="700">Astra / high</text>
<text x="125" y="191" font-size="12">planning session</text>
<text x="125" y="214" font-size="11" fill="#65645d">supplied evidence</text>
<text x="420" y="166" font-size="14" font-weight="700">Sol / medium</text>
<text x="420" y="191" font-size="12">implementation session</text>
<text x="420" y="214" font-size="11" fill="#65645d">bounded code task</text>
<text x="715" y="166" font-size="14" font-weight="700">Astra / high</text>
<text x="715" y="191" font-size="12">review session</text>
<text x="715" y="214" font-size="11" fill="#65645d">supplied evidence</text>
<text x="125" y="327" font-size="11.5" font-weight="700">Read-only evidence</text>
<text x="125" y="348" font-size="10.5">repository facts and task context</text>
<text x="420" y="316" font-size="11.5" font-weight="700">Isolated working checkout</text>
<text x="420" y="340" font-size="10.5">source control · write-capable tools</text>
<text x="420" y="361" font-size="10.5">tests and patch evidence</text>
<text x="715" y="327" font-size="11.5" font-weight="700">Read-only evidence</text>
<text x="715" y="348" font-size="10.5">diff and test results</text>
</g>
<path d="M220 184H325" fill="none" stroke="#2456a6" stroke-width="2.5" marker-end="url(#agents-route-blue-arrow)"/>
<text x="272" y="172" text-anchor="middle" font-size="10.5" fill="#2456a6">validated brief</text>
<path d="M515 184H620" fill="none" stroke="#2456a6" stroke-width="2.5" marker-end="url(#agents-route-blue-arrow)"/>
<text x="568" y="172" text-anchor="middle" font-size="10.5" fill="#2456a6">diff + evidence</text>
<path d="M125 302V237" fill="none" stroke="#2456a6" stroke-width="2" marker-end="url(#agents-route-blue-arrow)"/>
<path d="M420 237V286" fill="none" stroke="#386853" stroke-width="2" marker-end="url(#agents-route-green-arrow)"/>
<path d="M715 302V237" fill="none" stroke="#2456a6" stroke-width="2" marker-end="url(#agents-route-blue-arrow)"/>
<text x="420" y="420" text-anchor="middle" font-size="11.5" font-weight="700" fill="#65645d">Separate API sessions · workspace provisioned by the application</text>
</g>
</svg>
</div>
<figcaption style="margin-top:10px;color:#65645d;font-size:14px;line-height:1.5;"><strong style="color:#20221f;">Separate API sessions, not a claimed built-in role policy.</strong> The application provisions workspaces and transfers evidence.</figcaption>
</figure>

The API accepts agent-level `model` and `reasoning.effort`; the design above uses those fields on separate sessions. Disable built-in delegation in these sessions when routing must stay under the application's control. Give the planner only the access it needs, and validate the returned configuration and work. The session-creation example above represents the worker step. This design still requires an application controller and integration testing. [Agent configuration](https://developers.openai.com/api/docs/guides/agents-api/configuration), [configuration reference](https://developers.openai.com/api/reference/resources/beta/subresources/agents/methods/create)

## Reuse instructions and tools, then test the boundary

Skills and plugins are a practical way to carry project knowledge into the environment. For self-hosted plugins, the guide documents registering each plugin root through `environment.capability_directories`. Existing sessions do not reload changed plugin tools; create a new session when testing a changed package. Treat local configuration keys and API fields as separate contracts. [Plugins](https://developers.openai.com/api/docs/guides/agents-api/tools/plugins)

Execution access deserves the same care as in the original harness guide. Agent-generated code can use the files, credentials, and network exposed to its environment. Separate workloads and keep broad application credentials outside the sandbox. The executor key is deliberately narrower. [Sandbox security](https://developers.openai.com/api/docs/guides/agents-api/environments/security)

The deployment choice changes who operates the harness. You still define the task, supply useful context, choose access, and decide whether the result is good enough to accept.

Read the [original Codex architecture post](/2026/09/12/codex-harness-architecture.html) or explore the [full visual architecture guide](/explainers/codex-harness-architecture.html) for the runtime behind these API concepts.

*Documentation checked September 12, 2026, during the public beta. Examples were checked against documentation and have not been run against an authenticated Agents API session. The application-controlled planner/worker flow is an architectural proposal. Account access and compatible SDK and executor versions must be verified before use.*
