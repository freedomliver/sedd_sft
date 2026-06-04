# SEDD 监督微调任务文件

## 任务目标

基于 SEDD（Score Entropy Discrete Diffusion）论文和官方实现，完成一个离散扩散语言模型的监督微调版本。目标是在 s1K-1.1 数学题数据集上进行条件化监督微调，同时避免基础模型能力崩坏或生成乱码。

## 推荐主线方案

采用 `sedd-medium` 作为基础模型，使用 LoRA 进行参数高效微调。训练目标不使用普通 causal language modeling 的 next-token cross entropy，而是继续使用 SEDD 的 Score Entropy / DWDSE 损失，并改造成 response-only 条件化损失。

核心约束：

1. prompt/question 区域保持 clean，不加噪，不计算 loss。
2. answer/solution 区域参与前向离散扩散，并计算 Score Entropy / DWDSE loss。
3. padding 区域不加噪，不计算 loss。
4. 采样阶段 prompt 始终 clamp，只对 answer 区域进行反向扩散生成。
5. 首版训练只更新 LoRA 参数，冻结 embedding、output layer 和时间条件相关模块。

## 任务拆解

### 1. 环境与代码基线

- 拉取 SEDD 官方代码。
- 能够加载 `louaaron/sedd-medium`。
- 确认 `load_model` 返回 `model, graph, noise`。
- 跑通官方 unconditional 或 conditional sampling demo。
- 保存 base model 的采样输出作为对照。

### 2. 数据处理

- 下载 `simplescaling/s1K-1.1`。
- 使用字段：第一版建议用 `question` + `solution`。
- 构造样本格式：

```text
Question:
{question}

Answer:
{solution}<|endoftext|>
```

- 构造三个 mask：
  - `prompt_mask`：Question 和 Answer 前缀区域。
  - `answer_mask`：solution 和 eos 区域。
  - `pad_mask`：非 padding 区域。
- 自行切分数据集，例如 800 train / 100 validation / 100 test。

### 3. 实现 response-only SEDD loss

基于官方 `losses.py` 修改：

- 采样连续扩散时间 `t`。
- 通过 `noise(t)` 得到 `sigma, dsigma`。
- 使用 `graph.sample_transition(batch, sigma[:, None])` 对 clean tokens 加噪。
- 只把 answer 区域替换成 noisy tokens。
- prompt 和 padding 区域保持原 token。
- 用 `mutils.get_score_fn(model, train=True, sampling=False)` 得到 log-score。
- 调用 `graph.score_entropy(...)`。
- 用 `answer_mask & pad_mask` 过滤 loss。
- 按 answer token 数归一化。

### 4. 接入 LoRA

首版推荐 target modules：

```text
blocks.*.attn_qkv
blocks.*.attn_out
blocks.*.mlp.0
blocks.*.mlp.2
```

推荐超参数：

```text
r = 8 或 16
alpha = 16 或 32
dropout = 0.05
learning rate = 3e-5 ~ 1e-4
warmup = 50 ~ 100 steps
grad clip = 1.0
effective batch size = 16 ~ 64
epochs = 3 ~ 10
```

首版冻结：

```text
embedding
output_layer.linear
sigma_map
adaLN_modulation
其他非 LoRA 参数
```

### 5. 训练实验

至少实现四组实验：

| 实验 | 训练方式 | Loss | 目的 |
|---|---|---|---|
| Base | 不训练 | 无 | 作为基线 |
| Full FT | 全参 | 全序列 SEDD loss | 展示灾难性遗忘/乱码风险 |
| LoRA-SFT | LoRA | answer-only DWDSE | 主实验 |
| LoRA-SFT+Replay | LoRA | answer-only DWDSE + replay DWDSE | 验证能力保持 |

### 6. 条件采样实现

推理流程：

```text
prompt = "Question:
...

Answer:
"
x[prompt_positions] = prompt token ids
x[answer_positions] = MASK token id
for t = T ... 0:
    run reverse diffusion update
    clamp x[prompt_positions] back to prompt token ids
decode answer_positions until eos
```

采样参数建议：

```text
steps = 128
predictor = analytic 或 euler
noise_removal = True
max_answer_len = 512 或 768
```

### 7. 评估指标

#### 7.1 Loss 评估

- validation/test response-only DWDSE。
- 每个样本可 Monte Carlo 采样多个 timestep，降低方差。

#### 7.2 数学生成评估

- Exact Match。
- boxed answer match。
- pass@1。
- pass@4 / pass@8。
- 平均生成长度。
- 无效输出率：空输出、乱码、重复、无 eos。

#### 7.3 原能力保持评估

- WikiText/OpenWebText 上的 SEDD loss 或 pseudo-NLL。
- 无条件 sample 可读性。
- 通用 prompt 的 conditional generation 可读性。

### 8. Debug checklist

- LoRA 初始化为 0、不训练时，输出应接近 base。
- 只有 LoRA 参数 `requires_grad=True`。
- prompt token 在 forward corruption 后保持不变。
- padding token 不参与 loss。
- answer token 有非零 loss。
- 没有使用 causal CE：不要写 `shift_logits` / `shift_labels`。
- 采样时每一步都 clamp prompt。
- 学习率过大时优先降 LR，而不是改 loss 形式。

### 9. 最终交付物

建议面试作业最终提交：

1. 训练代码：包含 data collator、masked SEDD loss、LoRA 注入、训练循环。
2. 推理代码：prompt-clamped conditional sampler。
3. 实验表格：Base / Full FT / LoRA / LoRA+Replay。
4. 采样样例：展示数学题输入和生成解答。
5. 消融分析：LoRA rank、是否 response-only、是否 replay、采样步数。
6. Debug 分析：解释为什么普通 SFT 或全参训练会乱码。

---

# 原始对话整理内容

是的，**我建议你用 LoRA 做 SEDD 的监督微调**；但你现在“生成完全乱码”的根因很可能不只是“没有 LoRA”，而是把 SEDD 当成了普通自回归 LM 来训，或者在小数据上用全参、高学习率、全 token loss 破坏了原模型的 score/ratio 结构。

核心思路是：**不要用 next-token cross entropy；继续用 SEDD 的 Score Entropy / DWDSE 损失，只是把它改成“条件化、response-only”的监督微调损失。**

---

## 1. 先明确：SEDD SFT 不是普通 causal LM SFT

SEDD 的模型输出不是“下一个 token 的 logits”，而是用于反向离散扩散的 **score ratio**，也就是近似类似：

\[
s_\theta(x_t,t)_y \approx \frac{p_{t|0}(y|x_0)}{p_{t|0}(x_t|x_0)}
\]

论文里说 SEDD 是用数据分布的 ratio 来参数化反向离散扩散过程，并用 score entropy 来学习这些 ratio。([arxiv.org](https://arxiv.org/pdf/2310.16834)) 官方代码也不是只加载一个 model，而是 `load_model("louaaron/sedd-medium")` 同时返回 `model, graph, noise`，因为 graph 和 noise 是 loss/sampling 的一部分。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion))

所以你做 SFT 时，目标应该是：

> 给定 prompt/question，把 answer/solution 区域视为需要生成的区域；只对 answer 区域做离散扩散扰动和 Score Entropy 损失；prompt 区域在训练和采样时都保持 clamped。

这对应的是 **conditional denoising SFT**，而不是 causal LM 的“预测下一个词”。

---

## 2. 为什么你全参训练后会乱码

你这个现象很典型，主要可能有几类原因。

第一，**数据太小**。s1K-1.1 只有 default/train 的 1k rows，字段包括 `question`、`solution`、`deepseek_thinking_trajectory`、`deepseek_attempt` 等。([huggingface.co](https://huggingface.co/datasets/simplescaling/s1K-1.1)) 用 1000 条数学题全参更新 300M+ 量级模型，很容易把原本的通用 score 分布破坏掉。

第二，**训练目标可能错了**。如果你直接把 SEDD 的输出当成 autoregressive logits 做 CE，基本一定会坏。SEDD 的官方 loss 是：采样 \(t\)，通过 noise 得到 \(\sigma,\sigma'\)，用 graph 把 clean batch 扰动成 \(x_t\)，模型输出 log-score，然后调用 `graph.score_entropy(...)`，最后乘以 `dsigma` 并在序列维求和。官方 `losses.py` 里的流程就是这个结构。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/losses.py))

第三，**loss 位置可能错了**。如果你对 prompt、answer、padding 全部算 loss，模型会把容量浪费在“复现题目文本”和“复现 padding/eos 模式”上；对 1k 数据来说，这会非常快地过拟合。

第四，**学习率和可训练参数太大**。论文原训练使用大 batch、长训练、EMA、预训练语料规模；SEDD medium 在论文里约有 320M non-embedding parameters，而且训练时用了 8×A100 80GB 或 16×A100 40GB 级别资源。([arxiv.org](https://arxiv.org/pdf/2310.16834)) 你的小规模 SFT 不应该沿用预训练式全参设置。

---

## 3. 应该用什么损失：response-only DWDSE / Score Entropy

论文中的 score entropy 定义本质上是让模型学正的 ratio，并且 \(K(a)=a(\log a-1)\) 保证非负；作者还强调它是 non-negative、symmetric、convex，并且把 cross entropy 推广到正值 score。([arxiv.org](https://arxiv.org/pdf/2310.16834)) 在 SEDD 训练里，用的是 diffusion-weighted denoising score entropy，算法是：采样 \(x_0\)、采样 \(t\)、由 \(x_0\) 构造 \(x_t\)，再计算 DWDSE 并反传。([arxiv.org](https://arxiv.org/pdf/2310.16834))

你要改成下面这个监督微调目标：

\[
\mathcal{L}_{\text{SFT-SEDD}}
=
\mathbb{E}_{x_0,t,x_t}
\left[
\frac{
\sum_i m_i \cdot \sigma'(t) \cdot
\mathcal{L}_{SE}^{(i)}
\left(
s_\theta(x_t,t), x_t, x_0
\right)
}{
\sum_i m_i
}
\right]
\]

其中：

\[
m_i =
\begin{cases}
1, & \text{第 } i \text{ 个 token 属于 answer/solution 区域}\\
0, & \text{第 } i \text{ 个 token 属于 prompt 或 padding}
\end{cases}
\]

也就是说：

- prompt/question 区域：**不加噪、不算 loss**。
- answer/solution 区域：**加噪、算 Score Entropy loss**。
- padding 区域：**不加噪、不算 loss**。
- 采样时：prompt 区域始终 clamp，answer 区域从 mask/base state 开始反向扩散生成。

对 SEDD medium 默认的 absorbing graph 来说，前向过程大致就是 token 以概率 \(1-e^{-\sigma(t)}\) 变成 MASK。官方 `Absorbing.sample_transition` 也是按这个概率把 token 替换成 absorbing/mask token；对应的 `score_entropy` 只在 mask 位置产生有效 entropy。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/graph_lib.py))

---

## 4. 损失函数实现骨架

你可以基于官方 `losses.py` 改一个 masked loss。核心不是重写数学公式，而是复用官方 `graph.score_entropy`，只加上 `answer_mask`。

```python
import torch
from model import utils as mutils

def get_sft_loss_fn(noise, graph, train=True, sampling_eps=1e-3):
    """
    batch:       [B, L] token ids, clean x0
    answer_mask: [B, L] bool or 0/1, only answer/solution positions are 1
    pad_mask:    [B, L] bool or 0/1, valid tokens are 1; optional
    """
    def loss_fn(model, batch, answer_mask, pad_mask=None):
        device = batch.device
        B, L = batch.shape

        # 1. sample continuous diffusion time
        t = (1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps
        sigma, dsigma = noise(t)

        # 2. sample forward corruption
        noisy_all = graph.sample_transition(batch, sigma[:, None])

        # 3. only corrupt answer region; keep prompt/pad clean
        if pad_mask is None:
            valid_mask = torch.ones_like(answer_mask, dtype=torch.bool)
        else:
            valid_mask = pad_mask.bool()

        loss_mask = answer_mask.bool() & valid_mask
        perturbed = torch.where(loss_mask, noisy_all, batch)

        # 4. model predicts log-score, not next-token logits
        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed, sigma)

        # 5. official per-token score entropy
        per_tok_loss = graph.score_entropy(
            log_score,
            sigma[:, None],
            perturbed,
            batch,
        )  # [B, L]

        # 6. response-only weighting and normalization
        per_tok_loss = dsigma[:, None] * per_tok_loss * loss_mask.float()

        denom = loss_mask.float().sum(dim=-1).clamp_min(1.0)
        loss_per_example = per_tok_loss.sum(dim=-1) / denom

        return loss_per_example.mean()

    return loss_fn
```

重点是这三行：

```python
loss_mask = answer_mask.bool() & valid_mask
perturbed = torch.where(loss_mask, noisy_all, batch)
per_tok_loss = per_tok_loss * loss_mask.float()
```

这就是把 SEDD 的无条件训练目标改成 SFT 目标的关键。

---

## 5. LoRA 应该加在哪里

SEDD medium 的配置是 hidden size 1024、24 blocks、16 heads、length 1024、`scale_by_sigma=True`。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/configs/model/medium.yaml)) 模型 block 里有这些典型线性层：`attn_qkv`、`attn_out`、MLP 的两个 Linear 层 `mlp[0]` 和 `mlp[2]`。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/transformer.py))

建议的 LoRA target：

```text
blocks.*.attn_qkv
blocks.*.attn_out
blocks.*.mlp.0
blocks.*.mlp.2
```

初始版本建议：

```text
r = 8 或 16
alpha = 16 或 32
dropout = 0.05
LoRA lr = 3e-5 ~ 1e-4
warmup = 50 ~ 100 steps
grad_clip = 1.0
bf16 / fp16 mixed precision
freeze embedding
freeze output_layer.linear，至少第一版先冻结
```

为什么先冻结 embedding 和 output layer？因为 SEDD 的输出层直接影响所有 token ratio 的尺度，小数据全参动它很容易让 score 分布崩掉。第一版只让 attention/MLP 适配数学问答风格，通常更稳。

如果发现模型能保持通顺但不会学到数学格式，再做第二版：

```text
额外给 output_layer.linear 加低 rank LoRA，例如 r=4 或 8
学习率减半
训练步数更少
```

不要优先动 `sigma_map`、`adaLN_modulation`。这些层和噪声时间条件强相关，小数据微调时动它们容易破坏不同噪声等级下的校准。

---

## 6. 数据格式怎么做

s1K-1.1 的 `solution` 字段最长约 6.9k 字符，`deepseek_thinking_trajectory` 可以到 80k 字符量级。([huggingface.co](https://huggingface.co/datasets/simplescaling/s1K-1.1)) SEDD medium 的上下文长度是 1024 tokens，所以第一版不要直接塞完整 DeepSeek reasoning trace。

推荐格式：

```text
Question:
{question}

Answer:
{solution}<|endoftext|>
```

然后构造三个 mask：

```text
prompt_mask = Question + Answer: 之前的 token
answer_mask = solution + eos 的 token
pad_mask    = 非 padding token
```

第一版建议用 `solution`，不要用完整 `deepseek_thinking_trajectory`。等流程跑通后，再考虑用：

```text
{deepseek_thinking_trajectory}

Final Answer:
{deepseek_attempt}
```

但要做严格截断，例如 prompt 最多 256 tokens，answer 最多 768 tokens，并且确保最后答案没有被截掉。

由于 Hugging Face 页面显示这个数据集只有 `train` split、1k rows，你需要自己切分，例如：

```text
800 train / 100 validation / 100 test
```

最好按 `source_type` 或 `cot_type` 做近似分层，避免 test 全是某一类题。数据集页面显示字段里确实有 `cot_type` 和 `source_type`。([huggingface.co](https://huggingface.co/datasets/simplescaling/s1K-1.1))

---

## 7. 训练流程

完整流程可以这样做。

**Step 1：加载模型**

```python
model, graph, noise = load_model("louaaron/sedd-medium")
```

官方说明里 pretrained small/medium 都通过 `load_model` 加载，而且返回 graph 和 noise。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)) 不要自己临时造一个 graph/noise，除非你能保证和 checkpoint 完全一致。

**Step 2：tokenize**

用和 SEDD/GPT-2 一致的 tokenizer。`tokens: 50257` 是官方 config 里的 vocab size；absorbing graph 会额外加一个 mask/absorbing state。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/configs/config.yaml))

**Step 3：构造 batch**

每个 batch 至少包含：

```python
{
    "input_ids": batch,          # clean x0
    "answer_mask": answer_mask,  # only solution/eos
    "pad_mask": pad_mask,
}
```

**Step 4：只训练 LoRA 参数**

检查：

```python
for name, p in model.named_parameters():
    if "lora_" not in name:
        p.requires_grad = False
```

或者你用自己的 LoRA wrapper 精确替换 `attn_qkv/attn_out/mlp.0/mlp.2`。

**Step 5：训练**

建议第一版配置：

```text
epochs: 3 ~ 10
effective batch size: 16 ~ 64
lr: 5e-5
warmup: 100 steps
optimizer: AdamW
weight_decay: 0 或 0.01
grad_clip: 1.0
eval every: 50 ~ 100 steps
early stopping: validation response-DWDSE 不再下降即停
```

如果你只有一张显卡，SEDD medium 的输出 `[B, L, V]` 会很吃显存。可以先把 sequence length 改到 512 做 demo，或者用 sedd-small 跑通流程，再切回 sedd-medium。

**Step 6：可选 replay 防遗忘**

如果你还想保留原能力，不要只做 s1K。加一个小 replay loss：

\[
\mathcal{L}
=
\mathcal{L}_{s1K}
+
\lambda \mathcal{L}_{replay}
\]

其中 `replay` 可以是少量 OpenWebText/WikiText 文本，使用原始 SEDD loss。建议：

```text
λ = 0.05 ~ 0.2
每个 batch 里 70% s1K，30% replay
```

如果没有 replay 数据，至少用 LoRA、低学习率、early stopping，别全参训。

---

## 8. 采样 / 推理怎么做

论文的条件采样算法是：先从 base distribution 初始化 \(x_T\)，把 prompt 空间 \(\Omega\) 的 token 设置为给定 prompt；每一步反向采样时，只有非 prompt 位置更新，prompt 位置保持不变。([arxiv.org](https://arxiv.org/pdf/2310.16834)) 官方采样代码也是从 `graph.sample_limit(...)` 初始化，然后按 timesteps 迭代 predictor，最后可做 denoise step。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/sampling.py))

推理时你要这样：

```text
prompt = "Question:\n...\n\nAnswer:\n"
x[prompt_positions] = prompt token ids
x[answer_positions] = MASK token id
for t = T ... 0:
    run reverse diffusion update
    clamp x[prompt_positions] back to prompt token ids
decode answer_positions until eos
```

如果用官方 `get_pc_sampler`，可以传一个 `proj_fun`：

```python
def clamp_prompt(x):
    x[:, :prompt_len] = prompt_ids[None, :]
    return x
```

每一步 predictor 前后都 clamp prompt。官方代码里 `get_pc_sampler(..., proj_fun=lambda x: x)` 已经有 projector 入口，并在循环中调用。([github.com](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/sampling.py))

采样参数第一版建议：

```text
predictor: analytic 或 euler 都可先试
steps: 128
noise_removal: True
max_answer_len: 512 或 768
```

论文附录里提到 unconditional generation 里 Tweedie denoiser 通常比 Euler sampling 略好。([arxiv.org](https://arxiv.org/pdf/2310.16834)) 所以如果官方实现能跑 analytic/Tweedie，建议和 Euler 都做 ablation。

---

## 9. 评估怎么设计

你这个大作业至少要有三类评估。

### 9.1 Held-out response DWDSE

在 test split 上计算 response-only DWDSE：

```text
base SEDD
full finetune
LoRA SFT
LoRA SFT + replay
```

注意对每个样本最好 Monte Carlo 多采几次 \(t\)，例如 5 或 10 次，降低方差。论文也说明 likelihood evaluation 会用 Monte Carlo timesteps。([arxiv.org](https://arxiv.org/pdf/2310.16834))

### 9.2 数学答案生成质量

对 test 的 question 做 conditional sampling，生成 answer。

指标：

```text
Exact Match / boxed answer match
pass@1
pass@4 或 pass@8
平均生成长度
无效输出率：空输出、乱码、重复、无 eos
```

如果题目是 proof 类型，不适合 exact match，就用人工/LLM judge 做子集评估；但数值题、AIME 类题可以做 `\boxed{}` 或最后数字抽取。

### 9.3 原能力保持

你的问题里已经遇到“原能力丧失”，所以这个评估很重要。可以做：

```text
WikiText/OpenWebText response-free DWDSE 或 pseudo-NLL
无条件 sample 的可读性
通用 prompt 的 conditional generation 可读性
```

对比：

```text
Base SEDD
LoRA SFT
Full FT
```

如果 LoRA SFT 在 s1K test 上提升，同时 WikiText/OpenWebText loss 只小幅变差，就说明方案站得住。

---

## 10. 最小可交付实验设计

面试大作业里，我建议你做 4 组实验：

| 实验 | 训练方式 | Loss | 预期 |
|---|---|---|---|
| Base | 不训练 | 无 | 作为生成和 loss baseline |
| Full FT | 全参 | 全序列 SEDD loss | 很可能过拟合/乱码，用来说明问题 |
| LoRA-SFT | LoRA | answer-only DWDSE | 主实验 |
| LoRA-SFT+Replay | LoRA | answer-only DWDSE + replay DWDSE | 保留原能力最好 |

重点不是跑出 SOTA，而是展示你理解了：

```text
SEDD 的 forward diffusion
SEDD 的 reverse sampling
Score Entropy 为什么不是普通 CE
如何把无条件 diffusion LM 改成 supervised conditional fine-tuning
为什么小数据全参会灾难性遗忘
LoRA + response-only DWDSE 如何缓解
```

---

## 11. Debug checklist

你可以按这个顺序排查，通常能很快定位乱码原因。

**第一，确认 LoRA 关闭时输出等于 base。**

加载 base model，加 LoRA 但 LoRA 初始化为 0，不训练，采样结果应该和 base 基本一致。

**第二，确认只有 LoRA 有梯度。**

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

应该远小于 390M/320M 量级。

**第三，确认 prompt 不被加噪。**

打印一条 batch：

```text
clean prompt tokens == perturbed prompt tokens
answer tokens 部分变成 mask
pad tokens 不参与 loss
```

**第四，确认 loss mask 正确。**

```python
assert per_tok_loss[prompt_mask].sum() == 0
assert per_tok_loss[pad_positions].sum() == 0
assert per_tok_loss[answer_mask].sum() > 0
```

**第五，确认没有用 causal CE。**

SEDD 训练里不应该有：

```python
shift_logits = logits[:, :-1]
shift_labels = input_ids[:, 1:]
cross_entropy(...)
```

除非你额外训练了一个自回归 head；否则这会破坏模型。

**第六，确认采样时 clamp prompt。**

如果训练时 prompt 是 clean 的，但采样时 prompt 也被扩散/更新，模型会失配，输出容易崩。

---

## 结论

你的方向应该是：

```text
sedd-medium
+ LoRA on attention/MLP
+ prompt-clamped conditional diffusion
+ answer-only Score Entropy / DWDSE loss
+ small LR / early stopping
+ optional replay loss
+ held-out DWDSE + math answer pass@k + retention eval
```

不要把 SEDD 当成普通 GPT 做 SFT。**LoRA 是必要的稳定手段之一，但真正关键的是：损失函数仍然必须是 SEDD 的 Score Entropy，只是在 token 维度上做 response-only mask，并在训练/采样两端都保持 prompt clamped。**
