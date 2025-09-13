# Llama 2: Open Foundation and Fine-Tuned Chat Models

In this blog, I capture the notes on paper session on Meta's Llama 2, conducted by the [paper reading community](https://hasgeek.com/fifthelephant/call-for-papers/) under the aegis of fifth elephant community orchestrated by Hasgeek. Sachin and Anjineyulu presented this paper recently and it was a very interesting discussion and introduction to salient and high level important points in the paper by Meta. I capture them here below.

## Paper link
[Llama2 Paper by Meta](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)


## High level architecture of the model training pipeline followed in LLAMA2 model training
![High level architecture of llama2 training pipeline](https://github.com/bsbarkur/bsbarkur.github.io/assets/106684/21797517-1c11-48bd-b215-10dc0d152219)

## Why continual pre-training is hard?
Continual pre-trained models are difficult to train because they must learn to perform well on a wide range of tasks, often without access to a lot of data for each task. 

This can make it difficult to find a set of hyper-parameters that work well for all tasks. Additionally, continual pre-trained models must be able to learn new tasks without forgetting what they have already learned. This is a difficult problem, as it requires the model to be able to distinguish between different tasks and to update its knowledge in a way that does not interfere with its performance on previous tasks. 

Here are some specific challenges associated with continual pre-trained models: 
* **Data scarcity:** Continual pre-trained models often have access to limited data for each task. This can make it difficult to learn to perform well on a wide range of tasks. 
* **Hyper-parameter tuning:** Finding a set of hyper-parameters that work well for all tasks can be difficult. This is because different tasks may require different settings in order to achieve good performance.
* **Catastrophic forgetting:** Continual pre-trained models must be able to learn new tasks without forgetting what they have already learned. This is a difficult problem, as it requires the model to be able to distinguish between different tasks and to update its knowledge in a way that does not interfere with its performance on previous tasks. 
* **Negative transfer:** Continual pre-trained models may experience negative transfer, where learning a new task can hurt performance on previous tasks. This can be caused by the model learning to focus on features that are specific to the new task, at the cost of features that are important for previous tasks. 

## **Mixture of Experts**
Mixture of experts is a way to achieve lower inference latency but with more parameters. More details at [HF blog on MOE](https://huggingface.co/blog/moe)


## **Responsible AI:**
Pre-trained dataset had documents filtered for PII. So it is easy to fine-tune on LLAMA2 base model without worrying on hateful content.


## **Supervised fine-tuning [SFT]**
Flan dataset by Google used.
Manually annotated 27450 instruction and response pairs.

27k instruction and response pairs are sufficient for fine-tuning task, basically. This is for English or any other multilingual task ? This has to be considered and evaluated.

## Comparing of pre-trained hyperparameters and the SFT hyperparameters.

* The cosine learning rate in SFT is reduced by an order of one magnitude. This is likely because in SFT, we want to change the style of information and we don't want to add any new information. Hence lower learning rate is picked.
* The weight decay remains the same
* Sequence length also remains the same.

In pre-training, we ask the model to learn the next token.
In SFT we are asking the model to learn the response tokens.
We don't care on instruction, we backpropagate for our loss on the prompt.

## **RLHF data collection**
A binary comparison of this response vs other responses was done.
They used four degrees of comparison
* Significantly better
* Better
* Slightly better
* Unsure


## Reward model
The reward model was used on fine-tuning RLHF for weeks, till they were confident of improvements.

The objective for the ranking for the reward model

If the reward is low, the negative value will come, and vice versa.

Margin term gives them more granular control over how they can control the function.

1 epoch of training was done, so that it won't overfit. DPO sometimes has this issue of overfitting.

In the reward model, the learning rate is further reduced by an order of one magnitude.


## RLHF: RL training

- In RLHF, the agent (our fine-tuned instruct LLM) in its environment (Context Window) takes one action (of generating text) from all available actions in the action space (the entire vocabulary of tokens/words in the LLM).

### Rejection sampling
- Close to SFT
- given the prompt to the model, generate 10 samples with 10 different temperatures, ask the reward model which of these samples had max reward, then fine-tune on that particular prompt response pair

### PPO - proximal policy optimization
* We make our policy get maximum amount of reward
* Two models were trained for saftey and helpfulness
* If safety was less than 0.15 they won't look at helpfulness
	* They take the safety model output and then just say it is 
* If the safety score is above a certain threshold of 0.15, they determine the response to be safe and optimize for the helpfulness score.


AdamW is used as optimizer because it takes care of Weight decay in a nicer way than the ADAM standard optimizer


## Context distillation
In this stage, you set the context using system prompt for the model, such as "You are safe and responsible assistant" and fine-tune on those responses.

## Ghost attention:
This is exactly like context distillation for dialogue setting, where a synthetic instruction is added before all dialogues and then fine-tuned.

## Interesting findings
Temperature rescaling. Higher temperatures give creative generations, and lower give factual generations.

Model understands the time. For eg: if you set a system prompt as specific to a date like 1940, and post a question after that date, it might say i don't know about it.

Emergent tooling - function calling. Able to do zero shot function calling.

## Difference from llama 1 to llama2

We adopt most of the pretraining setting and model architecture from Llama 1. We use the standard transformer architecture (Vaswani et al., 2017), apply pre-normalization using RMSNorm (Zhang and Sennrich, 2019), use the SwiGLU activation function (Shazeer, 2020), and rotary positional embeddings (RoPE, Su et al. 2022). The primary architectural differences from Llama 1 include increased context length and grouped-query attention (GQA)

## Grouped Query Attention GQA
![gqa](https://github.com/bsbarkur/bsbarkur.github.io/assets/106684/224abbbc-4010-4b6b-8976-1c1112b4667e)


[This video](https://www.youtube.com/watch?v=o68RRGxAtDo
 ) explains how Grouped Query Attention works.


![Self Attention head Blocks in transformer](https://github.com/bsbarkur/bsbarkur.github.io/assets/106684/9d7bdfad-cd8f-4d68-8eca-6154baed1d9d)


So the attention score in the above diagram is calculated this way.

* Query and Key matrixes are multiplied.
* Scaling on the dot product of it happens using d_k and taking square root of d_k
* A mask can also be applied (optional)
* Finally on this scaled Q x K^T product, we apply soft max
* This is multiplied by value matrix to get an attention score matrix

In multi-head attention, we have `h` heads, as shown in the middle. Each head produces a scaled dot-product attention as described earlier. It is concatenated and fed into a linear layer.

## **Challenge of Multi-head attention**

The crux of the issue lies in the memory overhead. 
Each decoding step in autoregressive models like Transformers requires loading decoder weights along with all attention keys and values. 
This process is not only computationally intensive but also memory bandwidth-intensive. As model sizes grow, this overhead also increases, making scaling up an increasingly arduous task.

The below figure shows how a Grouped-attention scenario works.
![multi attention](https://github.com/bsbarkur/bsbarkur.github.io/assets/106684/4ad9cdaf-13da-4d51-98ad-3e1e18878ffa)
