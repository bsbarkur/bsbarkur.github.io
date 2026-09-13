---
layout: post
title: "One Paper, Nine Years: How a Translation Model Started the AI Boom"
tags: [transformers, LLM, AI history, DeepSeek, visual explainer]
---

In June 2017, eight researchers at Google posted a paper about machine translation. It proposed a network with no recurrence and no convolutions, trained it for three and a half days on eight GPUs, and beat every published translation system on the standard English to German benchmark. The title was a joke on a Beatles song: *Attention Is All You Need*.

Nine years later, nearly everything people call "AI" descends from the diagram in that paper. ChatGPT, Claude, Gemini, LLaMA and the DeepSeek models all use the same basic machine with parts swapped out. I wanted to understand exactly which parts, so I am writing a visual series that goes through the changes one model at a time. This post is the short version of the history, and an introduction to that series.

## The problem the paper solved

Before 2017, the best translation systems were recurrent networks. They read a sentence one word at a time, carrying a hidden state forward, and wrote the translation the same way. Attention already existed as a helper for these systems: Bahdanau, Cho and Bengio had shown in 2014 that letting the decoder look back at every source word improved translations.

The bottleneck was the recurrence itself. Each step depended on the previous one, so a long sentence could not be processed in parallel. Training was slow, and long-range dependencies degraded as they passed through many steps.

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin removed the recurrence and kept only the attention. Every token looks at every other token in one parallel step. Order comes from a fixed positional signal added to the inputs. The result was faster to train, easier to scale, and better.

That is really why it took over. A model that parallelises well can be made bigger, and over the next few years bigger turned out to be the thing that mattered.

## What happened next

<figure style="margin:28px 0;padding:18px;background:#fffdf8;border:1px solid #d7cfbf;border-radius:10px;overflow-x:auto;">
<svg viewBox="0 0 760 270" role="img" aria-labelledby="boom-title boom-desc" style="display:block;width:100%;min-width:620px;height:auto;color:#20221f;">
<title id="boom-title">Nine years on one timeline</title>
<desc id="boom-desc">A timeline from 2017 to 2026. Above the line: the paper, BERT and GPT-1, GPT-3, ChatGPT, DeepSeek-R1, DeepSeek V4.1 Flash. Below the line: the architectural edits, from encoder-decoder to decoder-only, to rotary positions and mixture of experts, to latent attention and multi-token prediction, to sparse and compressed attention.</desc>
<g font-family="system-ui,sans-serif" fill="#20221f">
<line x1="30" y1="135" x2="730" y2="135" stroke="#d7cfbf" stroke-width="2"/>
<text x="30" y="34" font-size="10" font-weight="700" fill="#a74723" letter-spacing="1">THE BOOM</text>
<text x="30" y="250" font-size="10" font-weight="700" fill="#2456a6" letter-spacing="1">THE MACHINE</text>
<g stroke="#a74723" stroke-width="1.5"><line x1="60" y1="135" x2="60" y2="96"/><line x1="170" y1="135" x2="170" y2="96"/><line x1="300" y1="135" x2="300" y2="96"/><line x1="440" y1="135" x2="440" y2="96"/><line x1="580" y1="135" x2="580" y2="96"/><line x1="700" y1="135" x2="700" y2="96"/></g>
<g fill="#a74723"><circle cx="60" cy="135" r="6"/><circle cx="170" cy="135" r="6"/><circle cx="300" cy="135" r="6"/><circle cx="440" cy="135" r="6"/><circle cx="580" cy="135" r="6"/><circle cx="700" cy="135" r="6"/></g>
<g font-size="11" text-anchor="middle">
<text x="60" y="62" font-weight="700">Jun 2017</text><text x="60" y="78">the paper</text><text x="60" y="92" font-size="9.5" fill="#65645d">8 GPUs, 3.5 days</text>
<text x="170" y="62" font-weight="700">2018</text><text x="170" y="78">GPT-1 and BERT</text><text x="170" y="92" font-size="9.5" fill="#65645d">each keeps one half</text>
<text x="300" y="62" font-weight="700">2020</text><text x="300" y="78">GPT-3, 175B</text><text x="300" y="92" font-size="9.5" fill="#65645d">scaling laws hold</text>
<text x="440" y="62" font-weight="700">Nov 2022</text><text x="440" y="78">ChatGPT</text><text x="440" y="92" font-size="9.5" fill="#65645d">100M users in 2 months</text>
<text x="580" y="62" font-weight="700">Jan 2025</text><text x="580" y="78">DeepSeek-R1</text><text x="580" y="92" font-size="9.5" fill="#65645d">open weights, low cost</text>
<text x="700" y="62" font-weight="700">Sep 2026</text><text x="700" y="78">V4.1 Flash</text><text x="700" y="92" font-size="9.5" fill="#65645d">1M context, 890 B/token</text>
</g>
<g stroke="#2456a6" stroke-width="1.5"><line x1="60" y1="135" x2="60" y2="172"/><line x1="200" y1="135" x2="200" y2="172"/><line x1="350" y1="135" x2="350" y2="172"/><line x1="500" y1="135" x2="500" y2="172"/><line x1="640" y1="135" x2="640" y2="172"/></g>
<g fill="#fffdf8" stroke="#2456a6" stroke-width="2"><circle cx="60" cy="135" r="6" fill="#a74723" stroke="none"/><circle cx="200" cy="135" r="6"/><circle cx="350" cy="135" r="6"/><circle cx="500" cy="135" r="6"/><circle cx="640" cy="135" r="6"/></g>
<g font-size="11" text-anchor="middle">
<text x="60" y="190" font-weight="700">encoder + decoder</text><text x="60" y="205" font-size="9.5" fill="#65645d">6 + 6 layers, 65M</text>
<text x="200" y="190" font-weight="700">decoder only</text><text x="200" y="205" font-size="9.5" fill="#65645d">pre-norm, learned positions</text>
<text x="350" y="190" font-weight="700">RoPE, GQA, MoE</text><text x="350" y="205" font-size="9.5" fill="#65645d">LLaMA, Mixtral</text>
<text x="500" y="190" font-weight="700">latent KV, MTP</text><text x="500" y="205" font-size="9.5" fill="#65645d">DeepSeek-V2, V3</text>
<text x="640" y="190" font-weight="700">sparse, compressed</text><text x="640" y="205" font-size="9.5" fill="#65645d">V3.2, V4, V4.1 Flash</text>
</g>
</g></svg>
<figcaption style="margin-top:10px;color:#65645d;font-size:14px;line-height:1.5;"><strong style="color:#20221f;">Two rows, same nine years.</strong> Above the line are the things that made the news. Below the line are the changes inside the model that made them possible. The series is about the bottom row.</figcaption>
</figure>

The paper's diagram had two halves, an encoder that reads and a decoder that writes. In 2018 each half became a product line. OpenAI's GPT-1 kept the decoder and trained it to predict the next word. Google's BERT kept the encoder and trained it to fill in blanks. Both showed that pretraining on raw text, then adapting to a task, beat training from scratch.

After that, the main thing that changed was size. GPT-2 in 2019 had 1.5 billion parameters. GPT-3 in 2020 had 175 billion and could perform tasks from a few examples in the prompt, with no fine-tuning at all. The same year, a paper from OpenAI showed that loss falls predictably as compute, data and parameters grow, which gave labs a reason to keep spending. Transformers also crossed into vision with ViT and into protein structure prediction with AlphaFold 2.

Then, on November 30, 2022, OpenAI put a chat interface in front of a fine-tuned GPT-3.5 and called it ChatGPT. It reached a hundred million users in about two months. That was when most people first noticed, but the architecture underneath had barely changed in five years.

The most recent stretch has been about cost and open weights. Meta released LLaMA's weights in 2023, and the open ecosystem adopted its recipe of RMSNorm, SwiGLU and rotary positions. Mixtral showed that a mixture of experts could match a much larger dense model. DeepSeek, a lab in Hangzhou, then spent 2024 and 2025 attacking the two costs that scale worst, the key-value cache and the number of tokens that attention has to touch. Its R1 model in January 2025 matched frontier reasoning performance with open weights and a published training cost that briefly erased around six hundred billion dollars of Nvidia's market value in a day. Its V4.1 Flash, released this month, runs a million-token context with a cache of under a kilobyte per token.

## What has not changed

Put the 2017 diagram next to any of these models and you can still match up the parts. Two ideas in particular have never been replaced. The first is the residual stream, where each block adds its result onto a running vector rather than replacing it. The second is softmax attention over query and key scores, the mechanism by which one token reads another.

Everything else has been swapped at least once. Layer normalisation moved before the block, then became RMSNorm. Sinusoidal positions became learned positions, then rotary ones. The feed-forward block gained a gate, then became a router in front of hundreds of experts. Attention heads learned to share keys and values, then to compress them into a small latent, then to skip most tokens entirely. V4.1 Flash even brings back a form of the encoder and decoder split, although both halves are now causal.

I wanted a way to see those changes rather than read about them. Peter Gostev built a striking [three-dimensional comparison](https://transformer-architecture.petergostev.chatgpt.site/) of the 2017 model against V4.1 Flash, but it jumps straight from the first frame to the last. The series I am publishing fills in the steps between.

## The series

Each chapter takes one architecture, redraws the same block diagram, and highlights only the components that changed. Every chapter has hand-drawn SVG figures, at least one interactive figure, and a fact strip with numbers taken from the primary paper or model card. The planned chapters:

1. The original Transformer (2017)
2. GPT-2 and the decoder-only stack (2019)
3. LLaMA: RMSNorm, SwiGLU, RoPE and grouped-query attention (2023)
4. Mixtral and the sparse mixture of experts (2023)
5. DeepSeek-V2: multi-head latent attention (2024)
6. DeepSeek-V3: load balancing and multi-token prediction (2024)
7. DeepSeek-V3.2: sparse attention with a lightning indexer (2025)
8. DeepSeek-V4: compressed attention and hyper-connections (2026)
9. DeepSeek V4.1 Flash: the causal encoder-decoder (2026)

Chapter 1 is up. It goes through the original model's parts, follows one token through an encoder layer, and works through how self-attention computes its weights, with queries, keys, values and the eight heads. It ends with a table of what each later chapter replaces.

<p style="margin:21px 0;">
  <a href="/explainers/transformer-evolution/01-original-transformer.html" target="_blank" rel="noopener"
     style="display:inline-block;background:#a74723;color:#fff;padding:14px 18px;border-radius:7px;font-weight:600;text-decoration:none;border:0;font-size:16px;">
    Open chapter 1: the original Transformer &rarr;
  </a>
  <br>
  <small style="color:#5b5b55;">Opens in a new tab. The page is self-contained, responsive, and uses no external JavaScript or web fonts.</small>
</p>

Later chapters will be linked here as they are published.

## Sources

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017. Training details in Section 5, results in Table 2.
- Bahdanau, Cho and Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2014.
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018. Devlin et al., [BERT](https://arxiv.org/abs/1810.04805), 2018.
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020. Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
- DeepSeek-AI, [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and [release note](https://api-docs.deepseek.com/news/news260910/), September 2026.
