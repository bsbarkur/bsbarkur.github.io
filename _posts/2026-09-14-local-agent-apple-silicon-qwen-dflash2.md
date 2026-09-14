---
layout: post
title: "Running a local coding agent on a MacBook with Qwen3.8-27B and DFlash 2"
date: 2026-09-14 11:30:00 +0530
tags: [local LLM, Apple Silicon, llama.cpp, speculative decoding, DFlash, Qwen, AI agents]
---

I spent this morning getting a 27-billion-parameter model to run as a coding agent on my laptop, with no API key and nothing leaving the machine. A 1.1 GB helper model called a drafter made the larger model generate faster. Inco AI released it as [DFlash 2](https://inco.ai/blog/dflash2/). This post covers the setup I ran on an M4 Max with 64 GB of memory, the numbers I measured, and a 126-line Python agent that used it to fix two bugs.

Every command here was run on 14 September 2026. The numbers are from one machine, on battery, with a lot of other things open, so treat them as one data point.

## How DFlash 2 works

A language model produces one token per forward pass, and on a Mac each pass means reading the model's weights from memory. For a 19 GB model, that memory traffic is a major part of the cost. Speculative decoding adds a small second model that guesses the next several tokens cheaply, then has the larger model check all of the guesses in one pass. Checking eight positions can require about the same memory traffic as generating one. If most guesses are right, the model produces several tokens for the cost of one verification step. DFlash guesses a whole block of tokens at once instead of one at a time. DFlash 2 adds a selector that picks a coherent path through the top candidates at each position. Speculative decoding is designed to preserve the target model's output distribution while changing how quickly it generates. I wrote a longer [explainer on the first DFlash](https://latentsig.com/insights/dflash-muse-spark-1-2/) for the Latentsig site in August if you want more detail on the architecture.

Since then, [llama.cpp merged DFlash 2 support](https://github.com/ggml-org/llama.cpp/pull/27342) on 27 August, and the 0.4.0 release that Homebrew ships includes it. The required support is now available through `brew upgrade`.

## Requirements

- A Mac with Apple Silicon and at least 32 GB of unified memory. I used an M4 Max with 64 GB. The weights alone are 19 GB, so a 16 GB machine cannot run this configuration.
- About 21 GB of free disk.
- Homebrew and Python 3. The download took about fifteen minutes on my connection; everything else took less than five.

The memory budget, taken from the server's own load log, came to roughly 23 GB on the GPU:

| Component | Metal buffer |
| :--- | ---: |
| Qwen3.8-27B weights, Q4_K_M, 64 layers | 18,084 MiB |
| DFlash 2 drafter weights, Q4_K_M, 5 layers | 1,080 MiB |
| Target KV cache at 32k context | 2,048 MiB |
| Drafter KV cache | 50 MiB |
| Compute buffers: target 257, drafter 1,468 | 1,725 MiB |

The process sat at 24 GB resident once warm. The KV cache is small for a 27B model because Qwen3.8 is a hybrid. Only every fourth layer is ordinary attention. The other 48 are Gated DeltaNet layers, a linear-attention design whose state does not grow with context.

## Install llama.cpp

```sh
brew install llama.cpp        # or brew upgrade llama.cpp
llama-server --version        # version: 0.4.0 (build 10809, commit 5266f24da) or later
llama-server --list-devices   # should list MTL0: Apple M4 Max (or your chip)
```

Two things to check. The build number must be 10809 or higher, because anything older predates the DFlash 2 merge. And the device list must show an `MTL0` line. If you only see `BLAS: Accelerate`, the GPU is not being used and everything will be about ten times slower.

If you prefer building from source, the Metal build is the standard one:

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j
```

Older guides tell you to fetch the PR branch first. That is no longer needed.

## Download the two models

The target is the ggml-org conversion of Qwen3.8-27B. The drafter is Inco's DFlash 2 checkpoint as GGUF. The drafter cannot generate text by itself; it reads hidden states from five specific layers of the target and proposes tokens for the target to verify, which is also why it only works with this exact target.

```sh
mkdir -p models && cd models
curl -L -C - -o Qwen3.8-27B-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf
curl -L -C - -o Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/resolve/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

`-C -` makes the 19 GB download resumable. The drafter repo was updated on 29 August with a rotary-embedding fix, so make sure you have the current file; the `z-lab` mirror has identical bytes. You can also let llama-server fetch both with `-hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`, but I like seeing the files.

## Start the server

Return to the directory that contains `models` before starting the server.

```sh
llama-server \
  -m  models/Qwen3.8-27B-Q4_K_M.gguf \
  -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  -c 32768 \
  --jinja \
  --reasoning-format auto \
  -np 1 \
  --alias qwen3.8-27b \
  --host 127.0.0.1 --port 8080
```

What the flags do:

- `-md` is the drafter. `--spec-type draft-dflash` is the same value the first DFlash used; the server reads the GGUF metadata and turns on the DFlash 2 selector and convolution code when it finds them. There is no separate `dflash2` type.
- `--spec-draft-n-max 7` is how many tokens the drafter proposes per verification step. Inco recommends 7. The llama.cpp reviewers found 4 slightly better at long contexts. I measured both below.
- `-c 32768` is the context. Qwen3.8 allows 262k, but a single agent rarely needs it and the full-attention KV cache grows by about 2 GiB per 32k tokens.
- `--jinja` applies the model's own chat template. Without it, Qwen's tool-calling format is not used and the agent loop never sees a tool call, just text that looks like one. This is the flag I would forget.
- `--reasoning-format auto` puts the model's thinking into a separate `reasoning_content` field instead of the answer.
- `-np 1` configures one parallel slot for these measurements.

The server began listening in under three seconds. The weights are memory-mapped and paged in lazily, so the first request still pays the loading cost. Run it once with `-v` to see the load details. The lines to look for are `ggml_metal_init: found device: Apple M4 Max`, `load_tensors: offloaded 65/65 layers to GPU` for the target and `6/6` for the drafter, and finally `listening on http://127.0.0.1:8080`. One warning is normal and can be ignored: `dflash requires ctx_other to be set (this warning is normal during memory fitting)` appears while the server measures memory before wiring the drafter to the target.

The drafter's metadata describes DFlash 2 as a 5-layer, 1.9-billion-parameter non-causal transformer with a block size of 8, a selector that keeps the top 16 candidates per position, and a 2-tap grouped convolution. It reads hidden states from target layers 6, 20, 34, 48 and 62.

I wrapped all of this in a small `serve.sh` so that `SPEC=0 ./serve.sh` runs without the drafter and `DRAFT_N=4 ./serve.sh` changes the block size. It is at the end of the post.

## Benchmark results

I sent three prompts: a Python parsing task, a two-heap median explanation, and a word problem. I used 512-token answers, with thinking off and Qwen's recommended instruct sampling of temperature 0.7, top-p 0.8, and top-k 20. I ran each configuration six times. llama-server returns per-request timings that include how many drafted tokens the target accepted, so the acceptance numbers are the server's, not mine.

| Configuration | Median decode | Range | Tokens per verification step | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| No drafter | 8.5 tok/s | 8.1 to 8.8 | 1.00 | 1.0× |
| DFlash 2, block 7 | 12.7 tok/s | 3.5 to 16.9 | 4.82 | 1.5× |
| DFlash 2, block 4 | 10.6 tok/s | 9.2 to 12.0 | 3.83 | 1.25× |

With block 7 the target accepted 4.8 tokens per verification step on average, and the fastest run, the Python parsing task, held 16.9 tok/s, twice the baseline. Code and the maths problem accepted best. The prose explanation accepted worst. That is the same ordering the paper reports.

The gain is much smaller than the acceptance suggests. If a verification step cost the same as one plain decode step, 4.8 tokens per step would suggest a similar speedup. I got 1.5×, which means one DFlash step costs about three plain steps on this machine. Qwen3.8's Gated DeltaNet layers do recurrent work per token, so checking 8 positions may cost more than it does for a plain attention stack. The drafter's own pass and selector also add work when the baseline is only 8.5 tok/s. I did not isolate these costs, so this explanation remains a hypothesis. The llama.cpp PR reports 1.81× on an M5 Pro for the same model and quantization, which is in the same range. Inco reports 2.7× to 3.4× on NVIDIA hardware, but my runs do not establish the reason for that difference.

Block 7 stalled and block 4 did not. Two of the six block-7 runs dropped to 3.5 and 6.4 tok/s with acceptance unchanged, while all six no-drafter runs on the same machine a few minutes later stayed within 0.7 tok/s of each other. These runs point to the speculative path, but six runs are not enough to establish the cause. The drafter's graph has a small CPU-side component, visible in the load log as a CPU compute buffer. My guess is contention with the other things I had running at the CPU to GPU handoff each step, but I did not confirm that. Block 4 had no stalls and a higher acceptance rate, 72% against 55%, at the cost of fewer tokens per step. For an agent I would rather have a steady 10.6 than a 12.7 that sometimes freezes for two seconds, so block 4 is what I am running.

One more caveat on the absolute numbers. The machine was on battery for the whole session, dropping from 64% to 52% during the benchmark, and it had 12 GB of swap in use from other apps. Silicon Score lists 16.6 tok/s for the sibling Qwen3.6-27B at the same quantization on an M4 Max through Ollama, about double my baseline. I expect higher figures on mains power with a quieter machine. The relative speedups are more useful here than the absolute rates.

## A minimal agent loop

An agent, in the sense I mean here, is a loop. Send the conversation plus a list of tools to the model. If the reply contains tool calls, run them, append the results, and go round again. If it contains plain text, stop. The model does the planning. The loop does the bookkeeping and controls how the tools are exposed.

I wrote it in standard-library Python, 126 lines, no framework, because the part worth understanding is about forty lines and a framework would hide it. The request is an ordinary OpenAI-style chat completion, which llama-server implements:

```python
def chat(messages):
    body = {
        "model": MODEL, "messages": messages, "tools": TOOL_SPECS, "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": THINK},
        "temperature": 1.0 if THINK else 0.7, "top_p": 0.95 if THINK else 0.8, "top_k": 20,
        "presence_penalty": 0.0 if THINK else 1.5,
    }
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)
```

Two fields are specific to this setup. `chat_template_kwargs.enable_thinking` switches Qwen3.8's thinking mode per request. It is on by default, and I turn it off for the short coding task below. The sampling parameters follow the Qwen model card. Thinking mode uses 1.0 / 0.95, while instruct mode uses 0.7 / 0.80 with a presence penalty of 1.5. Mixing them up produces either repetition or drift.

The tools are four functions. The path check confines file operations to one directory, while `run` only starts in that directory:

```python
def _safe(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError(f"{path} is outside the workspace")
    return p

def list_files() -> str: ...
def read_file(path: str) -> str: ...
def write_file(path: str, content: str) -> str: ...
def run(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, cwd=WORKSPACE, capture_output=True, text=True, timeout=60)
    out = (r.stdout + r.stderr).strip()
    return f"exit {r.returncode}\n{out}" if out else f"exit {r.returncode}"
```

`run` is what makes a local agent useful and also what makes it dangerous. It executes shell commands with the workspace as the working directory and a 60-second timeout. That is a sandbox by convention, not by enforcement: `cd ..` still works. For anything beyond a demo, run this inside a container or restrict `run` to an allowlist.

The loop:

```python
for step in range(1, MAX_STEPS + 1):
    resp = chat(messages)
    msg = resp["choices"][0]["message"]
    messages.append(msg)
    calls = msg.get("tool_calls") or []
    if not calls:
        print(msg["content"]); break
    for call in calls:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"] or "{}")
        try:
            result = TOOLS[name](**args)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result[:MAX_OUTPUT]})
```

Errors go back to the model as tool output instead of being raised, because a model that reads "error: FileNotFoundError" usually recovers and a crashed loop never does. Tool output is cut at 4,000 characters so one chatty command cannot flood the context. Twenty steps is the cap.

## Fixing FizzBuzz

The workspace holds a FizzBuzz implementation with two bugs I planted: an off-by-one range and a branch order that makes `FizzBuzz` unreachable. Its test fails because of them.

```sh
python3 agent.py "Fix the bug in fizzbuzz.py so test_fizzbuzz.py passes, then run the test to prove it."
```

The run, lightly trimmed. Each step is one round trip to the server; the per-step line shows generated tokens, decode speed, and how many drafted tokens were accepted.

```text
--- step 1 · 14 tok @ 8.7 tok/s · draft accepted 11/21
[tool] list_files({})
       fizzbuzz.py
       test_fizzbuzz.py

--- step 2 · 57 tok @ 13.6 tok/s · draft accepted 49/56
[tool] read_file({'path': 'fizzbuzz.py'})
[tool] read_file({'path': 'test_fizzbuzz.py'})

--- step 3 · 269 tok @ 12.8 tok/s · draft accepted 226/301
[tool] write_file({'path': 'fizzbuzz.py', 'content': '"""Print FizzBuzz for 1..n. …'})
       wrote 511 chars to fizzbuzz.py

--- step 4 · 31 tok @ 12.3 tok/s · draft accepted 26/35
[tool] run({'cmd': 'python test_fizzbuzz.py'})
       exit 0
       ok

--- step 5 · 136 tok @ 9.0 tok/s · draft accepted 106/210

Fixed. Two bugs in `fizzbuzz.py`:
1. Off-by-one: `range(1, n)` stopped at `n-1`, so it never reached 15. Changed to `range(1, n + 1)`.
2. Unreachable branch: the `i % 15 == 0` check came after the `% 3` and `% 5` checks,
   so multiples of 15 were always caught by `Fizz` first. Moved it to the top.

=== 5 steps · 507 generated tokens · 11.5 tok/s avg · 54s wall · DFlash acceptance 67%
```

The run took fifty-four seconds and five model calls. The agent found both bugs, made a minimal diff, and ran the test successfully. It asked for both files in one step, which the loop handles without special casing. Step 3, where it wrote the file, was the largest generation and one of the best accepted. The target accepted 226 of 301 drafted tokens while the model rewrote a file it had just read. Step 5, the prose summary, had the lowest acceptance, which matches the benchmark.

I ran it again with thinking on. It found the same two bugs and produced the same fix, but took 7 steps, 988 generated tokens and 76 seconds against 5 steps, 507 tokens and 54 seconds. Acceptance fell from 67% to 56%. At step 4, it tried to run the test with `cd /home/luca84/tbench/2025-06-05/227 && python test_fizzbuzz.py`. That path does not exist on my machine, and the run provides no evidence for where the model got it. The shell returned exit 1, the model read the error, dropped the `cd`, and carried on. This is why the loop feeds tool errors back as text. It is also why `run` should not have access to files you would mind losing.

On this short, well-specified task, thinking added time and tokens without changing the fix. A harder task may behave differently.

## Limits and tradeoffs

- Context is memory. The KV cache for both models sits in the same unified memory as the weights. If the server fails to start or macOS starts swapping, lower `-c` first.
- Thinking mode changes the sampling settings. Switch both together.
- Acceptance falls on prose and at high temperature. Ask this server for a short story at temperature 1.2 and the drafter will be rejected often enough that you pay its cost for little gain. The server timings tell you when this is happening.
- These measurements used one parallel slot, set with `-np 1`. Concurrent requests need separate measurement, and continuous batching may work better than a drafter.
- The drafter is tied to this model. It reads hidden states from Qwen3.8-27B's layers and will not help a different size. Q4_K_M worked; I have not tried heavier quantisation of the target.
- Vision is a separate path. Qwen3.8-27B can take images through its `mmproj` file, but the PR discussion reports drafter acceptance collapsing on image inputs with current GGUFs. Keep DFlash 2 for text.

## Using the server with other tools

The server above is a general local inference endpoint. Coding assistants, evaluation harnesses, and batch jobs that support the OpenAI chat API can point at `http://127.0.0.1:8080/v1`.

On this machine, the 27B model ran as a local coding agent with no per-token API charge and no prompts sent to an external service. The 1 GB drafter improved median generation speed by 1.5× and reached 2× on the Python task in these runs.

## Appendix: serve.sh

```sh
#!/usr/bin/env bash
# Serve Qwen3.8-27B with DFlash 2 speculative decoding on llama-server (Apple Silicon / Metal).
#
#   ./serve.sh                 # DFlash 2 on (default)
#   SPEC=0 ./serve.sh          # plain autoregressive decoding, for A/B comparison
#   CTX=65536 DRAFT_N=4 ./serve.sh --verbose   # extra args pass straight through to llama-server
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${TARGET:-$DIR/models/Qwen3.8-27B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-$DIR/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"
SPEC="${SPEC:-1}"
DRAFT_N="${DRAFT_N:-7}"

need=("$TARGET"); [[ "$SPEC" == "1" ]] && need+=("$DRAFT")
for f in "${need[@]}"; do
  [[ -f "$f" ]] || { echo "missing $f - run ./download.sh" >&2; exit 1; }
done

args=(
  -m "$TARGET"
  -c "$CTX"
  --host 127.0.0.1 --port "$PORT"
  --alias qwen3.8-27b
  --jinja
  --reasoning-format auto
  -np 1
)
if [[ "$SPEC" == "1" ]]; then
  args+=( -md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "$DRAFT_N" )
fi
exec llama-server "${args[@]}" "$@"
```

## Sources

- Inco AI, [DFlash 2](https://inco.ai/blog/dflash2/), August 2026. Drafter: [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF), updated 29 August 2026.
- ggml-org, [llama.cpp PR #27342: spec: add DFlash2 support](https://github.com/ggml-org/llama.cpp/pull/27342), merged 27 August 2026. The M5 Pro figure is from the PR discussion.
- ggml-org, [Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), converted 14 August 2026.
- Qwen team, [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B), sampling parameters and context length.
- Silicon Score, [Qwen3.6-27B on Apple Silicon](https://siliconscore.com/models/qwen3-6-27b/), M4 Max reference figure, April 2026.
- Jian Chen, Yesheng Liang, and Zhijian Liu, [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), ICML 2026.
