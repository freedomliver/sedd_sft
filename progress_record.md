# SEDD 监督微调项目记录

记录日期：2026-06-01

项目路径：`/Users/jinc_air/Documents/resume/maril/sedd_sft`

## 1. 项目目标

本项目基于 SEDD（Score Entropy Discrete Diffusion）的官方 PyTorch 实现，尝试在 `simplescaling/s1K-1.1` 数学题数据集上完成监督微调，使离散扩散语言模型能够在给定题目 prompt 后生成答案。

核心目标不是把 SEDD 当作普通自回归语言模型做 next-token cross entropy，而是在保留原始 Score Entropy / DWDSE 训练形式的前提下，将任务改造成条件化 denoising：

```text
Question:
{question}

Answer:
{answer}<|endoftext|>
```

训练时 prompt 区域保持 clean，answer 区域参与离散扩散加噪和 response-only DWDSE，padding 区域不参与 loss；推理时 prompt 始终 clamp，只对 answer 区域做反向扩散采样。

## 2. 当前代码状态

当前工作区有未提交改动。主要新增或修改文件如下：

- `README.md`：补充 SEDD LoRA-SFT 使用说明。
- `SEDD监督微调任务文件.md`：整理任务目标、推荐方案、实验设计和调试 checklist。
- `INTERVIEW_PROGRESS.md`：记录理论说明、实验计划、远程运行日志、失败分析和后续方向。
- `data_sft.py`：重写 SFT 数据加载、格式化、mask 构造和 split 逻辑。
- `losses.py`：新增 response-only DWDSE loss，并保留兼容入口；新增 all-mask CE 辅助诊断 loss。
- `lora.py`：新增本地 LoRA 注入、冻结、保存和加载逻辑。
- `train_sft.py`：新增 LoRA response-only SFT 训练主流程。
- `infer_sft.py`：新增 prompt-clamped 条件生成入口。
- `eval_correct.py`：新增 boxed-answer 评估、pass@k、无效输出统计和可选 response-only loss 评估。
- `sampling.py`：支持 `init_x`、prompt clamp、answer 从 graph limit distribution 初始化，并在 predictor update 前后和 denoise 后强制 clamp。
- `sft_utils.py`：抽出模型加载、LoRA 加载、EOS 截断解码、boxed 提取和答案归一化工具。
- `run_train_eval.sh`：远程训练/评估脚本，支持 repo-location agnostic 路径和可配置数据 split。
- `run_overfit16_sft.sh`：16 样本过拟合诊断脚本，包含 DWDSE + all-mask CE 两阶段流程。
- `eval_sft.py`、`infer_v9.py`：降级为兼容 wrapper，分别转调 `eval_correct.py` 和 `infer_sft.py`。

轻量检查结果：

```bash
python -m py_compile data_sft.py losses.py lora.py train_sft.py infer_sft.py eval_correct.py sampling.py sft_utils.py test_data_format.py
```

该语法检查已通过。

## 3. 技术路线

### 3.1 数据格式与 mask

`data_sft.py` 当前负责：

- 加载本地 JSON 或 Hugging Face 数据集 `simplescaling/s1K-1.1`。
- 按 `train/valid/test` 切分数据。
- 构造统一 prompt：`Question:\n...\n\nAnswer:\n`。
- 生成 `prompt_mask`、`answer_mask`、`pad_mask`。
- 支持 `answer_field` 选择：
  - `solution`
  - `deepseek_attempt`
  - `final_boxed` / `boxed` / `boxed_answer`
- 支持 `answer_truncate=head|tail`，用于避免长 reasoning 目标把最后 boxed answer 截断掉。
- 支持 `boxed_prompt=True`，即 prompt 直接以 `\boxed{` 结束，只让模型生成 box 内内容。

当前推荐的短答案目标是 `answer_field=final_boxed`。该模式从 `deepseek_attempt` 或 `solution` 中提取最后一个 boxed answer，并构造短目标，降低模型学习长链式推理格式的难度。

### 3.2 Response-only DWDSE

`losses.py` 中的 `get_response_only_sft_loss_fn` 实现了当前主 loss：

- 对 batch 采样扩散时间 `t`。
- 使用 `noise(t)` 得到 `sigma, dsigma`。
- 用 `graph.sample_transition` 对 clean token 加噪。
- 只在 answer/window 区域使用 noisy token。
- prompt 和 padding 保持 clean。
- 调用 SEDD 原有 score function 和 `graph.score_entropy`。
- 用 `answer_mask & pad_mask` 过滤 token loss。
- 按每条样本的 answer token 数归一化。

兼容入口 `get_sft_loss_fn` 会把旧代码里的 `condition_len` 转成 answer mask，避免旧诊断脚本仍然走到错误的全序列或 boxed-only loss。

### 3.3 LoRA 微调

`lora.py` 的实现策略：

- 默认冻结所有 base model 参数。
- 对指定 `nn.Linear` 注入 `LoRALinear`。
- 默认 target suffix：

```text
attn_qkv, attn_out, mlp.0, mlp.2
```

- 对 MLP target 做额外限制，只匹配 `blocks.*.mlp.0/2`，避免误包 `sigma_map.mlp.0/2`。
- 保存 LoRA-only checkpoint，包含 LoRA 权重、LoRA config、optimizer state 和训练参数。

当前主训练默认更偏向调试短答案 overfit：

- `answer_field=final_boxed`
- `boxed_prompt=True`
- `max_answer_len=32`
- `lora_r=32`
- `lora_alpha=64`
- `lora_dropout=0.0`

远程主实验中也跑过更保守配置：

- `lora_r=8`
- `lora_alpha=16`
- `lora_dropout=0.05`
- `max_answer_len=512`

### 3.4 条件采样

`sampling.py` 当前的 prompt-clamped 采样逻辑：

- 将 prompt token 拼接到 answer 初始化状态前面。
- answer 区域用 `graph.sample_limit(B, answer_len)` 初始化；对 absorbing graph 来说，这对应 mask/absorbing token。
- 在每个 predictor update 前 clamp prompt。
- predictor update 后再次 clamp prompt。
- denoise 前后也 clamp prompt。
- 支持 `sampling_mode=sample|argmax`。

这解决了早期 sampler 只在 update 前和最终返回时 clamp，导致中间状态 prompt 可能 drift 的问题。

### 3.5 推理与评估

`infer_sft.py` 用于少量样例生成，输出 JSONL，字段包括：

- question
- prompt
- generated
- generated_suffix
- pred_boxed
- target_answer
- target_boxed
- match
- exact
- eos_present

`eval_correct.py` 用于批量评估，指标包括：

- `pass@1_boxed_match`
- `pass@k_boxed_match`
- `exact_target_match`
- `no_boxed_generations`
- `empty_generations`
- `no_eos_generations`
- `prompt_truncated`
- `avg_gen_tokens`
- 可选 `response_only_dwdse`

## 4. 调试过程总览

这一轮工作不是一次性把参数调到成功，而是按下面顺序逐层排除问题：

1. 先确认全参训练和普通 CE 思路为什么失败。
2. 把训练改成 SEDD 原生的 response-only DWDSE。
3. 加 LoRA，减少小样本对 base model 的破坏。
4. 检查训练目标和评估目标是否一致。
5. 检查采样阶段 prompt 是否每一步都被 clamp。
6. 做 16 样本过拟合，确认训练目标能不能记住训练集。
7. 把样本从 16 扩到 64，观察是否还能过拟合，以及 holdout 是否有非零准确率。
8. 去掉辅助 CE，单独验证纯 DWDSE 是否能下降、是否能带来生成准确率。

最终到目前为止的状态是：64 样本训练集可以明显过拟合，但 holdout 仍没有稳定超过 `10%`，说明当前方法还没有形成真正泛化，只是证明了 pipeline 和小样本记忆能力。

## 5. 详细实验过程

### 5.1 Full FT 失败基线

早期 full-parameter / boxed-only 训练得到过不可读的混合符号和英文碎片，boxed-match 为 `0/200`。

当时的问题判断：

1. s1K 规模太小，约 1k 条数学题不足以稳定全参更新 300M 级别模型。
2. 全参训练容易破坏 SEDD 预训练出来的 score-ratio 几何结构。
3. boxed-only 或全序列目标没有明确区分 prompt、answer、padding。
4. 如果直接把 SEDD 当成自回归 LM 做 next-token CE，训练目标和模型输出语义不匹配。

这个实验保留下来作为失败基线：它说明问题不能只靠“继续训更久”解决，必须重构 SFT 目标。

### 5.2 第一次 LoRA response-only DWDSE

第一版 LoRA-SFT 改成只训练 LoRA 参数，并使用 response-only DWDSE。

远端配置：

```text
base: pretrained/sedd-medium
data: data/s1K_train_599.json
loss: response-only DWDSE
train target: answer_field=solution
batch_size: 12
grad_accum: 4
effective batch: 48
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lr: 5e-5
max_steps: 1000
max_answer_len: 512
```

validation response-DWDSE 曾经下降：

| step | valid response-DWDSE |
|---:|---:|
| 100 | 3.7506 |
| 200 | 3.2296 |
| 400 | 2.9564 |
| 900 | 2.8677 |
| 1000 | 4.3327 |

生成评估结果：

| checkpoint | held-out boxed match | no boxed | no EOS |
|---|---:|---:|---:|
| `lora_final.pt` | 0/79 | 100% | 89.9% |
| `checkpoint_step_750.pt` | 0/79 | 100% | 84.8% |

结论：LoRA + response-only DWDSE 可以优化 scalar loss，但没有转化成 boxed-answer 生成能力。下一步必须查目标格式、采样逻辑和答案窗口。

### 5.3 Sampler clamp 审计

用户指出 SEDD 条件采样必须满足：

```text
prompt 不加噪、不算 loss
answer 加噪算 Score Entropy
采样时 prompt 始终 clamp
answer 区域从 mask/base state 开始反向生成
每一步 reverse update 后都重新执行 x[:, :prompt_len] = prompt_ids
```

检查后发现早期 sampler 虽然是 prompt-clamped，但只在 predictor update 前和最终返回时 clamp，没有在每个 reverse update 后立刻恢复 prompt。

已修复：

1. `get_pc_sampler` 支持显式 `init_x`。
2. prompt token 直接写入初始状态。
3. answer 区域使用 `graph.sample_limit(B, answer_len)` 初始化。对 absorbing graph 来说，这就是 mask/absorbing token。
4. predictor update 前 clamp prompt。
5. predictor update 后再次 clamp prompt。
6. denoise 前后也 clamp prompt。

严格 clamp 后重新评估早期 LoRA checkpoint，结果仍是 `0/79`，`100%` no boxed，`89.9%` no EOS。

结论：prompt drift 确实是一个实现 bug，但不是之前 LoRA 失败的唯一原因。

### 5.4 Target-format 审计

继续检查后发现训练目标和评估目标并不一致。

第一版 LoRA 用的是：

```text
answer_field=solution
```

但评估指标只抽取生成文本里的 `\boxed{...}`。在 `data/s1K_train_599.json` 的 train split 中：

| 字段 | 含 boxed 的比例 |
|---|---:|
| `solution` | 109/399 |
| `deepseek_attempt` | 399/399 |

也就是说，大部分训练目标没有 boxed，但评估要求 boxed，这会天然导致 `no_boxed=100%`。

直接切到 `deepseek_attempt` 也不够，因为 long reasoning 目标很长。如果用 head truncation，最后的 boxed answer 很容易被截掉。检查后确认：

1. `deepseek_attempt` 的最后 boxed answer 在尾部。
2. 128/256 token 的 head truncation 基本保不住最后 boxed。
3. tail truncation 能保住 boxed，但训练目标仍然是长 reasoning 尾部，不适合小样本 sanity check。

因此改成短答案目标：

```text
answer_field=final_boxed
boxed_prompt=True
answer_prefix=Answer:
max_length=1024
min_answer_len=32
max_answer_len=32
```

实际格式变成：

```text
Question:
{question}

Answer:
\boxed{
```

prompt 里已经包含 `\boxed{`，answer 区域只生成 box 内答案和 EOS。评估时再把生成 suffix 包回 `\boxed{...}` 做答案匹配。这样做的目的不是改变数学任务，而是先把格式难度降到最低，验证模型能不能条件化生成短答案。

### 5.5 上下文长度和截断

曾经担心 `max_length=512` 或更小会导致长题 prompt 在训练和评估中不一致。现在统一改成：

```text
max_length=1024
min_answer_len=32
max_answer_len=32
```

这里的含义是：

1. SEDD 网络输入长度固定为 `1024`。
2. prompt 最多占 `1024 - min_answer_len = 992` 个 token。
3. answer window 固定留至少 32 个 token。
4. 如果题目太长，prompt 会被截到 992 token。
5. 当前评估会统计 `prompt_truncated`，已有 overfit/holdout 评估中都是 `0/64` 或 `0/16`，没有发现题目截断。

所以当前失败不是因为训练 prompt 和 eval prompt 被截成了不同内容。

### 5.6 le32 短答案数据筛选

为了避免 answer 太长，后续从 599 条正确样本中筛选了 box 内答案 token 长度不超过 32 的样本。

远端数据检查结果：

| 数据文件 | 样本数 | box 内答案 GPT-2 token <=32 | >32 | <=64 | >64 |
|---|---:|---:|---:|---:|---:|
| `data/s1K_train_599.json` | 599 | 586 | 13 | 597 | 2 |
| `data/s1K_train_answer_inner_le32.json` | 586 | 586 | 0 | 586 | 0 |

这里的长度是去掉 `\boxed{}` 外壳后的 box 内答案 token 长度，不是字符串字符数。后续 16/64 样本实验都从 `data/s1K_train_answer_inner_le32.json` 抽样。

### 5.7 LoRA 参数范围

当前 LoRA wrapper 默认 target 是：

```text
attn_qkv, attn_out, mlp.0, mlp.2
```

同时做了匹配限制，只包 transformer block 内部的：

```text
blocks.*.attn_qkv
blocks.*.attn_out
blocks.*.mlp.0
blocks.*.mlp.2
```

这个限制是为了解决早期 suffix matching 误包 `sigma_map.mlp.0/2` 的问题。当前默认训练参数偏向小样本 overfit：

```text
lora_r=32
lora_alpha=64
lora_dropout=0.0
batch_size=16
grad_accum=1
lr=5e-5
all_mask_ce_weight=0.0
```

训练时只有 LoRA 参数更新，base model 参数冻结。

### 5.8 all-mask CE 辅助诊断

在 DWDSE loss 能下降但采样不出答案之后，加过一个辅助诊断项：all-mask answer CE。

它的作用是直接训练“prompt + 全 mask answer window”这个采样起点：

1. prompt 保持 clean。
2. answer window 输入全部替换成 `graph.absorb/mask`。
3. 只在真实 answer token 位置算 CE。
4. CE 后处理里修了 pad/EOS 边界，避免把 answer 结束后的 EOS padding 也当成目标反复训练。

两阶段脚本默认是：

```text
Stage 1:
  all_mask_ce_weight=30
  lr=3e-4
  steps=300

Stage 2:
  all_mask_ce_weight=1
  lr=5e-5
  steps=1000
```

这个 CE 不是最终想保留的主方法，只是为了判断问题是不是出在采样起点分布。后来用户明确要求不要继续做显式 token 权重或结构 token 权重实验，因此相关 `all_mask_ce_mode=structure/eos` 方向已经撤回，没有保留为默认行为。当前 `train_sft.py` 的默认 `all_mask_ce_weight=0.0`。

### 5.9 pad/EOS 边界修复

GPT-2 tokenizer 没有独立 pad token，代码里用 `eos_token_id` 作为 padding 值。如果 CE 或其他 token-level loss 覆盖整个固定 answer window，就会出现一个问题：

```text
真实答案: answer + EOS
窗口后部: EOS EOS EOS ...  这些其实只是 padding
```

如果把窗口后部也训练成 EOS，模型会学到大量“生成 EOS padding”的伪目标。修复后的逻辑是：

```text
DWDSE loss mask = answer_mask & pad_mask
CE target mask = 真实 answer token + 真实 EOS
post-EOS padding = 不算 loss
```

这个修复和“显式提高某些结构 token 权重”不是一回事；它只是排除 padding 假标签。

## 6. 过拟合验证过程

### 6.1 早期 16 样本 DWDSE sanity check

第一轮短答案 overfit 使用：

```text
answer_field=final_boxed
answer_prefix=Final Answer:
max_answer_len=64
max_steps=500
loss=response-only DWDSE
```

训练 loss 从约 `6.86` 降到 `0.43`，说明 scalar DWDSE 能被优化。但在同一 16 个 train prompt 上生成仍是 `0/16` boxed 输出，而且 EOS 很差。

这个结果说明：只看 DWDSE loss 降低不够，必须直接在 train split 上做生成评估，确认 reverse sampling 是否真的能复原答案。

### 6.2 16 样本纯 DWDSE, CE=0

后来按用户要求去掉 CE，从头训练 16 样本，只用 DWDSE：

```text
data: data/s1K_train_answer_inner_le32.json
seed: 10
train_size: 16
valid_size: 16
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 3e-4
batch_size: 16
grad_accum: 1
lora_r: 32
lora_alpha: 64
lora_dropout: 0
max_steps: 800
```

训练 DWDSE 曲线：

| step | train DWDSE |
|---:|---:|
| 10 | 6.7477 |
| 100 | 1.0340 |
| 200 | 0.4265 |
| 400 | 0.2293 |
| 600 | 0.1179 |
| 720 | 0.0550 |
| 800 | 0.1439 |

生成评估：

| split | argmax64 accuracy | no_eos |
|---|---:|---:|
| train16 | 15/16 = 93.8% | 0/16 |
| valid16 | 0/16 = 0.0% | 0/16 |

结论：纯 DWDSE 在 16 个样本上可以强过拟合，说明 loss 路径、LoRA 更新、prompt clamp 和短答案格式在 train split 上是可用的。但它没有泛化到 valid。

### 6.3 64 样本低 CE 实验

把样本数从 16 扩到 64，并保留较低 CE 辅助，配置大致为：

```text
seed: 0
train_size: 64
valid_size: 0 during training
data: data/s1K_train_answer_inner_le32.json
max_length: 1024
max_answer_len: 32
boxed_prompt: True
lora_r: 32
lora_alpha: 64
lora_dropout: 0

Stage 1:
  steps=300
  lr=3e-4
  all_mask_ce_weight=30

Stage 2:
  steps=1000
  lr=5e-5
  all_mask_ce_weight=1
```

评估结果：

| split/mode | accuracy | no_eos |
|---|---:|---:|
| train64 argmax64 | 46/64 = 71.9% | 6/64 |
| train64 sample4 | pass@4 48/64 = 75.0% | 32/256 |
| holdout64 argmax64 | 0/64 = 0.0% | 未作为主结论 |
| holdout64 sample4 | pass@4 2/64 = 3.1% | 未作为主结论 |

结论：64 样本训练集可以被部分记忆，sample4 可以略微提高 train/holdout 命中，但 holdout 仍远低于目标。

### 6.4 64 样本强 CE 实验

为了确认是不是 CE 太弱，又做过强 CE 版本：

```text
seed: 10
train_size: 64
valid_size: 64
all_mask_ce_weight: 100
max_steps: 1200
```

结果：

| split/mode | accuracy | no_eos |
|---|---:|---:|
| train64 argmax64 | 55/64 = 85.9% | 1/64 |
| valid64 argmax64 | 0/64 = 0.0% | 未作为主结论 |
| valid64 sample4 | pass@4 2/64 = 3.1% | 未作为主结论 |

结论：强 CE 可以提高 train 记忆和 EOS 稳定性，但 holdout 没有同步起来。继续提高 CE 大概率只会增强记忆，不会解决泛化。

### 6.5 64 样本纯 DWDSE, CE=0

为了验证“完全使用 DWDSE 能不能自己下降”，又做了 64 样本 CE=0 训练：

```text
seed: 10
train_size: 64
valid_size: 64
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 3e-4
batch_size: 16
grad_accum: 1
lora_r: 32
lora_alpha: 64
lora_dropout: 0
max_steps: 1600
save_freq: 0
```

训练 DWDSE 明显下降：

| step | train DWDSE |
|---:|---:|
| 10 | 6.7732 |
| 100 | 1.8790 |
| 200 | 0.9624 |
| 300 | 0.8017 |
| 400 | 0.5266 |
| 600 | 0.4420 |
| 800 | 0.2515 |
| 1200 | 0.1332 |
| 1500 | 0.1152 |
| 1600 | 0.1133 |

训练过程中的 valid loss 波动较大：

| step | valid DWDSE |
|---:|---:|
| 100 | 3.5436 |
| 200 | 4.8090 |
| 300 | 4.9253 |
| 400 | 3.6479 |
| 600 | 4.5729 |
| 800 | 3.9772 |
| 1200 | 4.4051 |
| 1600 | 3.1613 |

最终生成评估：

| split | argmax64 accuracy | no_eos |
|---|---:|---:|
| train64 | 48/64 = 75.0% | 1/64 |
| valid64 | 0/64 = 0.0% | 3/64 |

结论：CE=0 也能在 64 个训练样本上形成明显记忆，但 valid 生成仍不稳定。

### 6.6 64 样本 checkpoint 曲线

为了确认过拟合发生在哪个 step，重新跑了一轮保存 checkpoint 的 CE=0 曲线实验：

```text
out_dir: /root/autodl-tmp/sedd_outputs_archive/lora_overfit64_seed10_dwdse_only_curve
seed: 10
train_size: 64
valid_size: 64
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 3e-4
batch_size: 16
grad_accum: 1
lora_r: 32
lora_alpha: 64
lora_dropout: 0
save_freq: 100
eval steps: 100, 200, 400, 600, 800
eval sampler: argmax, steps=64
```

参数地图如下：

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.8543 | 3.4770 | 5/64 = 7.8% | 1/64 = 1.6% | 1/64 | 2/64 |
| 200 | 0.7574 | 4.7430 | 31/64 = 48.4% | 0/64 = 0.0% | 10/64 | 11/64 |
| 400 | 0.5755 | 4.6791 | 39/64 = 60.9% | 0/64 = 0.0% | 7/64 | 4/64 |
| 600 | 0.3880 | 3.5177 | 48/64 = 75.0% | 2/64 = 3.1% | 7/64 | 1/64 |
| 800 | 0.3246 | 4.8251 | 48/64 = 75.0% | 2/64 = 3.1% | 4/64 | 0/64 |

这个表的判断：

1. train loss 持续下降，train acc 从 `5/64` 上升到 `48/64`，说明过拟合大约从 step 200 开始明显发生，step 600 左右进入平台。
2. valid loss 没有单调下降，step 600 短暂较低，但 step 800 又升高。
3. valid acc 有非零点，但只有 `1/64` 到 `2/64`，没有形成稳定趋势，也没有达到 `>10%`。
4. no_eos 在 valid 上后期变好，说明边界格式有所改善，但答案正确率没有同步改善。

## 7. 当前可复现命令

### 7.1 16/64 样本 overfit 脚本

```bash
./run_overfit16_sft.sh
```

脚本当前默认：

```text
DATA_JSON=data/s1K_train_answer_inner_le32.json
SEED=0
TRAIN_SIZE=16
VALID_SIZE=0
ANSWER_PREFIX=Answer:
MAX_LENGTH=1024
MIN_ANSWER_LEN=32
MAX_ANSWER_LEN=32
BOXED_PROMPT=1
LORA_TARGETS=attn_qkv,attn_out,mlp.0,mlp.2
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.0
BATCH_SIZE=16
GRAD_ACCUM=1
ALL_MASK_CE_ANSWER_WEIGHT=4.0
ALL_MASK_CE_WEIGHT_STAGE1=30.0
ALL_MASK_CE_WEIGHT_STAGE2=1.0
STAGE1_LR=3e-4
STAGE2_LR=5e-5
```

如果要跑纯 DWDSE，需要显式传：

```bash
ALL_MASK_CE_WEIGHT_STAGE1=0 \
ALL_MASK_CE_WEIGHT_STAGE2=0 \
STAGE1_LR=3e-4 \
STAGE2_LR=3e-4 \
TRAIN_SIZE=64 \
VALID_SIZE=64 \
SEED=10 \
./run_overfit16_sft.sh
```

实际远端 64 样本曲线实验是直接调用 `train_sft.py`，并设置 `--save_freq 100` 保存中间 checkpoint。

### 7.2 当前训练主入口默认参数

`train_sft.py` 当前默认更适合短答案实验：

```text
max_length=1024
max_answer_len=32
min_answer_len=32
answer_field=final_boxed
answer_prefix=Answer:
boxed_prompt=True
batch_size=16
grad_accum=1
lr=5e-5
all_mask_ce_weight=0.0
lora_r=32
lora_alpha=64
lora_dropout=0.0
lora_targets=attn_qkv,attn_out,mlp.0,mlp.2
```

### 7.3 生成评估命令模板

```bash
python -u eval_correct.py \
  --pretrained pretrained/sedd-medium \
  --lora_ckpt outputs/lora_xxx/lora_final.pt \
  --data_json data/s1K_train_answer_inner_le32.json \
  --split valid \
  --train_size 64 \
  --valid_size 64 \
  --test_size 0 \
  --seed 10 \
  --limit 64 \
  --num_samples 1 \
  --steps 64 \
  --predictor analytic \
  --sampling_mode argmax \
  --max_length 1024 \
  --min_answer_len 32 \
  --max_answer_len 32 \
  --answer_field final_boxed \
  --answer_prefix Answer: \
  --answer_leading_newline \
  --boxed_prompt \
  --offline
```

如果 argmax 不是满分，曾经也用过：

```text
num_samples=4
steps=128
sampling_mode=sample
```

但 sample4 目前只把 holdout 提到过 `2/64 = 3.1%`，没有解决核心问题。

## 8. 当前结论

已经确认：

1. SEDD SFT 不能直接套 causal CE，当前主 loss 是 response-only DWDSE。
2. prompt、answer、padding mask 已明确分离。
3. 训练时 prompt 不加噪、不算 loss；answer 加噪算 DWDSE；padding 不算 loss。
4. 采样时 prompt 每一步都 clamp，answer 从 absorbing/mask state 开始生成。
5. `final_boxed + boxed_prompt` 修复了训练目标和 boxed-only 评估不一致的问题。
6. 1024 上下文下当前 16/64 样本实验没有 prompt truncation。
7. 纯 DWDSE 可以在 16 样本上达到 `15/16` train acc，在 64 样本上达到 `48/64` train acc。
8. 64 样本 holdout 目前最高只看到 `2/64 = 3.1%`，没有达到 `>10%`。

尚未解决：

1. valid loss 没有稳定下降，说明现在主要是记忆训练集，不是学习可泛化的条件生成。
2. valid acc 的非零结果很少，可能是题目/答案分布偶然命中。
3. DWDSE 的 scalar loss 和最终 reverse-sampling accuracy 之间仍有明显错位。
4. 强 CE 能提升 train acc，但没有明显提升 holdout。

## 9. 后续优化方向

短期不建议继续盲目增加 step。更合理的下一步是从以下方向选一个做小实验：

1. 固定 64 样本，做 per-sample train/valid loss 和生成结果对齐，确认哪些题 loss 低但生成错。
2. 查看 answer 第一个 token、数字 token、EOS token 的 top-k score/logit，比较 base 和 LoRA 是否真的把目标 token 排上去了。
3. 做更干净的 train/valid 划分，例如按题源或答案长度分桶，避免 64 样本 holdout 方差过大。
4. 在 CE=0 主线下尝试较低学习率或更长 warmup，观察 valid loss 是否更平滑。
5. 重新评估是否需要 output projection 附近的 LoRA，但这要单独归因，不和 loss 改动混在一起。
6. 如果目标是面试展示，可以把当前结果表述为“成功复现 SEDD 条件化 SFT pipeline，并证明小样本可过拟合，但泛化仍未达标”，不要把它包装成已经完成的 math SFT。

## 10. 面试叙事建议

建议按下面逻辑讲：

1. 先说明 SEDD 和 causal LM 不同，不能直接用 next-token CE。
2. 展示 full FT 失败，说明小数据全参更新会破坏预训练 score-ratio。
3. 说明重构后的正确训练流程：prompt clean，answer noised，answer-only DWDSE，padding masked。
4. 说明 sampler bug：prompt 必须每一步 update 后重新 clamp，并且已经修复。
5. 说明 target-format bug：之前训练 `solution`，评估 boxed，目标不一致；后来改成 `final_boxed + boxed_prompt`。
6. 展示 16 样本纯 DWDSE 可以 `15/16` 过拟合，证明训练 pipeline 有效。
7. 展示 64 样本曲线：train acc 上升，valid loss 不稳定，valid acc 最高 `2/64`。
8. 最后给出真实结论：当前还没有泛化成功，但已经形成了可复现、可解释的调试链路，下一步要分析 DWDSE loss 与 sampling accuracy 的错位。

## 11. 样本规模与泛化曲线实验记录

### 11.1 阶段目标

当前进入“样本规模与泛化曲线”阶段。目标是先固定主要训练逻辑不变，逐步增加训练样本数，观察 valid loss 和 valid accuracy 是否随训练集规模扩大而改善。

实验优先级：

1. 先跑样本规模曲线：`128 -> 256 -> 全部 le32 样本`。
2. 每个规模记录 `train loss + valid loss + train acc + valid acc`。
3. 如果大样本 valid 仍不好，再单独调整超参数，不把多个改动混在一轮里。
4. 超参数搜索顺序：`lr -> LoRA rank/alpha -> LoRA dropout -> weight decay -> max_steps/early stopping`。

样本规模命名后续按“总样本量”理解，不再简单把 `train_size` 当作规模名。当前已经启动的第一轮仍保留为 `train=128, valid=128` 参照；下一轮 256 总量建议使用：

```text
total=256
train_size=176
valid_size=40
test_size=40
```

全量 le32 当前有 586 条，建议使用：

```text
total=586
train_size=458
valid_size=64
test_size=64
```

其中 valid 用于 checkpoint 曲线和 early stopping 判断，test 只在选定 checkpoint 后做一次最终确认，避免反复根据 test 调参。

### 11.2 128 样本起始实验

这是样本规模阶段的第一组实验，按用户给定初始位置启动。

```text
run_name: scale128_seed10_lr1e4_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale128_seed10_lr1e4_ga2_dwdse_only
remote_train_log: logs/scale_grid/scale128_seed10_lr1e4_ga2_dwdse_only_train.log
remote_post_eval_log: logs/scale_grid/scale128_seed10_lr1e4_ga2_dwdse_only_post_eval.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
train_size: 128
valid_size: 128
test_size: 0
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 1e-4
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampler_eval: argmax, steps=64
checkpoint_eval_steps: 400,800,1200,1600
```

运行状态：

```text
2026-06-01 23:08 Asia/Shanghai: 远端训练已启动，PID=39802。
已挂自动 post-eval，训练结束后会评估 400/800/1200/1600 的 train/valid。
step 100: train loss=2.8067, valid_loss=3.3002。
step 200: train loss=2.0932, valid_loss=2.4579。
step 300: train loss=1.9213, valid_loss=3.4167。
step 400: train loss=1.9476, valid_loss=3.5596，已保存 checkpoint_step_400.pt。
step 500: train loss=1.1634, valid_loss=4.4259。
step 600: train loss=0.8962, valid_loss=3.7862。
step 700: train loss=0.9552, valid_loss=3.3392。
step 800: train loss=0.8153, valid_loss=4.8969，已保存 checkpoint_step_800.pt。
step 900: train loss=1.0392, valid_loss=4.1200。
step 1000: train loss=0.7862, valid_loss=3.3045。
step 1100: train loss=0.6552, valid_loss=4.0281。
step 1200: train loss=0.5210, valid_loss=3.7561，已保存 checkpoint_step_1200.pt。
```

结果表待训练和自动评估完成后补齐：

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 1.9476 | 3.5596 | 2/128 = 1.6% | 0/128 = 0.0% | 96/128 | 102/128 |
| 800 | 0.8153 | 4.8969 | 6/128 = 4.7% | 0/128 = 0.0% | 80/128 | 86/128 |
| 1200 | 0.5210 | 3.7561 | 12/128 = 9.4% | 0/128 = 0.0% | 67/128 | 77/128 |
| 1600 | 0.7182 | 3.6573 | 9/128 = 7.0% | 0/128 = 0.0% | 97/128 | 100/128 |

观察：

1. train acc 随训练先上升，step 1200 达到最高 `12/128 = 9.4%`，但 step 1600 回落到 `9/128 = 7.0%`。
2. valid acc 四个 checkpoint 全部是 `0/128`，没有看到泛化提升。
3. train loss 持续下降到 step 1200，但 valid loss 在 `3.56-4.90` 之间波动，没有稳定下降。
4. no_eos 比 64 样本实验更差，step 1600 valid no_eos 达到 `100/128`。
5. 输出失败模式主要是 `\dfrac/rac/texttext/反斜杠` 重复，说明这组 `lr=1e-4, grad_accum=2` 没有稳定学到短答案生成。

临时结论：当前 128 训练样本实验没有提升 valid；甚至 train 过拟合程度也弱于 64 样本纯 DWDSE 的 `48/64`。下一轮不宜直接照这个学习率扩到更大规模。恢复实验时优先考虑在 256 总量划分上降低 lr，例如先试 `lr=5e-5` 或 `3e-5`，并保留 `400/800/1200/1600` 曲线；同时按总量划分为 `train=176, valid=40, test=40`。

### 11.3 256 总量实验，降低 lr 到 5e-5

128 训练样本使用 `lr=1e-4` 时，valid acc 全部为 0，train acc 最高也只有 `12/128`。因此 256 总量实验不继续沿用 `1e-4`，先降低到 `5e-5`，同时按总量 256 划分为 `176/40/40`。

```text
run_name: scale256t176v40_seed10_lr5e5_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale256t176v40_seed10_lr5e5_ga2_dwdse_only
remote_train_log: logs/scale_grid/scale256t176v40_seed10_lr5e5_ga2_dwdse_only_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
total_used: 256
train_size: 176
valid_size: 40
test_size: 40
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 5e-5
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampler_eval: argmax, steps=64
checkpoint_eval_steps: 400,800,1200,1600
```

运行状态：

```text
2026-06-02: 远端训练已启动，PID=44917。
step 100: train loss=3.7689, valid_loss=3.1097。
step 200: train loss=7.5921, valid_loss=4.5224。
step 300: train loss=3.6904, valid_loss=3.3721。
step 400: train loss=2.2086, valid_loss=3.2477，已保存 checkpoint_step_400.pt。
step 500: train loss=1.9980, valid_loss=4.3627。
step 600: train loss=2.6453, valid_loss=3.2416。
step 700: train loss=2.1480, valid_loss=3.2177。
step 800: train loss=3.3183, valid_loss=2.5021，已保存 checkpoint_step_800.pt。
step 900: train loss=1.7448, valid_loss=3.2149。
step 1000: train loss=1.7581, valid_loss=3.4709。
step 1100: train loss=1.3670, valid_loss=1.6835。
step 1200: train loss=1.3185, valid_loss=2.5169，已保存 checkpoint_step_1200.pt。
step 1300: train loss=1.3853, valid_loss=2.2497。
step 1400: train loss=1.2110, valid_loss=3.0984。
step 1500: train loss=1.2027, valid_loss=2.7067。
step 1600: train loss=1.1922, valid_loss=3.0117，已保存 checkpoint_step_1600.pt。
```

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 2.2086 | 3.2477 | 0/176 = 0.0% | 0/40 = 0.0% | 163/176 | 36/40 |
| 800 | 3.3183 | 2.5021 | 0/176 = 0.0% | 0/40 = 0.0% | 166/176 | 37/40 |
| 1200 | 1.3185 | 2.5169 | 0/176 = 0.0% | 0/40 = 0.0% | 167/176 | 38/40 |
| 1600 | 1.1922 | 3.0117 | 1/176 = 0.6% | 0/40 = 0.0% | 169/176 | 38/40 |

评估补充：

- 评估命令使用 `argmax, steps=64, num_samples=1`。
- 远端长评估会被执行层约两分钟终止，所以给 `eval_correct.py` 增加了 `--start_idx`，默认仍为 0；只用于把长 train eval 拆片，不改变默认行为。
- step 1600 train 使用 `24/12` 条小分片评估后合并，合并后 JSONL 行数为 `176`。

观察：

1. 256 总量降低到 `lr=5e-5` 后，valid loss 的确比 128/lr1e-4 好看，step 1100 甚至出现 `1.6835`，但保存 checkpoint 的生成准确率仍然没有转化。
2. train generation 也没有过拟合成功：step 1600 只有 `1/176`，明显弱于 128/lr1e-4 的 `12/128`，更弱于 64 样本纯 DWDSE 的 `48/64`。
3. no_eos 仍在 `90%+`，说明采样大部分没有自然结束。当前主要瓶颈不是 boxed 格式，因为 boxed prompt 模式下 `no_boxed` 基本为 0，而是 answer window 内生成内容和 EOS 边界仍然不稳定。
4. 结论：单纯从 128 增到 256，并把 lr 降到 `5e-5`，没有提升 valid accuracy。更大的 full le32 仍需要跑曲线，但如果 valid 继续为 0，下一步应回到超参数顺序，优先重新扫 lr，而不是继续加训练步数。

### 11.4 full le32 初始曲线

按全量 le32 的建议划分启动了第一轮 full-scale 训练。当前只确认到 checkpoint step 400；训练 stdout 日志没有在 `logs/a800_full_le32` 中保留下来，后续需要从新 resume 日志继续记录 loss。

```text
run_name: full_le32_t458v64_seed10_lr5e5_b54_dwdse_only
remote_ckpt_dir: /root/autodl-tmp/a800_ckpts/full_le32_t458v64_seed10_lr5e5_b54_dwdse_only
data: data/s1K_train_answer_inner_le32.json
seed: 10
total_used: 586
train_size: 458
valid_size: 64
test_size: 64
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 5e-5
warmup_steps: 100
batch_size: 54
grad_accum: 1
effective_batch: 54
max_steps: 4000
save_freq: 100
eval_freq: 100
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampler_eval: argmax, steps=64
```

已评估的 valid 结果：

| step | train loss | valid loss | valid acc | valid no_eos | prompt_truncated |
|---:|---:|---:|---:|---:|---:|
| 100 | 未恢复 | 未恢复 | 0/64 = 0.0% | 52/64 | 0/64 |
| 300 | 未恢复 | 未恢复 | 0/64 = 0.0% | 53/64 | 0/64 |
| 400 | 未恢复 | 未恢复 | 0/64 = 0.0% | 56/64 | 0/64 |

初步观察：full le32 在早期 step 100/300/400 仍然是 `0/64`，输出以反斜杠、`texttext`、重复数字和未闭合结构为主。它还不能证明 full scale 无效，因为只跑到 step 400，但目前没有看到早期 valid 非零信号。

### 11.5 full le32 可比配置，batch16/grad_accum2

上面的 `b54` run 是早期尝试，后来在 RTX 5090 32GB 上从 step 400 resume 时 OOM：

```text
torch.OutOfMemoryError: Tried to allocate 324.00 MiB
GPU total: 31.37 GiB
batch_size: 54
```

因此重新启动一个和 128/256 更可比的 full le32 run：仍使用 `lr=5e-5, CE=0, LoRA r32/alpha64/dropout0`，但把 batch 设置回 `batch_size=16, grad_accum=2`，effective batch 仍是 32。

第一次启动时漏传 `--epochs 100000`，触发 `train_sft.py` 默认 `epochs=5`，只跑到 step 70 后正常保存 `lora_final.pt`。这不是训练崩溃。之后从 step 70 的 `lora_final.pt` resume，并补上 `--epochs 100000`，继续到 step 1600。

```text
run_name: scale586t458v64_seed10_lr5e5_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale586t458v64_seed10_lr5e5_ga2_dwdse_only
remote_train_log_stage0: logs/scale_grid/scale586t458v64_seed10_lr5e5_ga2_dwdse_only_train.log
remote_train_log_resume: logs/scale_grid/scale586t458v64_seed10_lr5e5_ga2_dwdse_only_resume70_to1600_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
total_used: 586
train_size: 458
valid_size: 64
test_size: 64
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 5e-5
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
eval_batches: 10
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampler_eval: argmax, steps=64
checkpoint_eval_steps: 400,800,1200,1600
```

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 100 | 3.8204 | 4.1968 |
| 200 | 3.4764 | 3.8408 |
| 300 | 2.5987 | 2.8030 |
| 400 | 2.5226 | 3.0660 |
| 500 | 2.3881 | 2.0486 |
| 600 | 3.0118 | 3.9252 |
| 700 | 2.1248 | 2.5616 |
| 800 | 2.8379 | 3.0130 |
| 900 | 2.5795 | 3.2294 |
| 1000 | 3.2010 | 2.9714 |
| 1100 | 2.3620 | 4.3300 |
| 1200 | 1.5881 | 3.5855 |
| 1300 | 2.1799 | 3.5203 |
| 1400 | 2.5401 | 2.3841 |
| 1500 | 2.2732 | 2.8392 |
| 1600 | 2.1587 | 2.5831 |

生成评估结果：

| step | train eval | train acc | valid acc | train no_eos | valid no_eos |
|---:|---|---:|---:|---:|---:|
| 400 | train first 64 | 0/64 = 0.0% | 0/64 = 0.0% | 56/64 | 53/64 |
| 800 | train first 64 | 0/64 = 0.0% | 0/64 = 0.0% | 62/64 | 59/64 |
| 1200 | train first 64 | 0/64 = 0.0% | 0/64 = 0.0% | 59/64 | 56/64 |
| 1600 | train first 64 | 0/64 = 0.0% | 0/64 = 0.0% | 62/64 | 58/64 |

观察：

1. full le32 的 valid loss 有下降信号，最低保存外 step 是 `500: 2.0486`，保存点里最好是 `1600: 2.5831`。但这些 loss 信号仍没有转化为生成准确率。
2. valid accuracy 四个保存点全部是 `0/64`，没有看到增大样本量带来的泛化提升。
3. train first 64 也全部是 `0/64`，说明这个配置下连训练集前 64 条都没有明显记忆，不是单纯 holdout 太难。
4. 与 128/lr1e-4 相比，full/lr5e-5 的 scalar loss 更平滑，但 generation 过拟合更弱。当前证据更支持“纯 DWDSE + 当前采样/训练超参不能稳定转成短答案生成”，而不是“样本量不够”。
5. 远端长评估会被执行层终止，因此 full valid/train64 评估都用 `--start_idx` 分片完成；每个完整 JSONL 已合并到 `logs/scale_grid/eval_scale586t458v64_seed10_lr5e5_ga2_dwdse_only_step{step}_{split}_argmax64.jsonl` 或 `..._train64_argmax64.jsonl`。

阶段性结论：`128 -> 256 -> full le32` 这条样本规模曲线没有带来 valid accuracy 改善。继续扩大样本本身不是当前最有效方向，下一步应进入第二优先级的超参数搜索，首先单独扫 `lr`。

### 11.6 lr 搜索第一轮：128 样本从 1e-4 提到 3e-4

样本规模曲线显示 `lr=5e-5` 在 256/full 上 train generation 都很弱，而 128 初始 `lr=1e-4` 虽然 valid 为 0，但 train acc 至少有非零信号。因此第一轮 lr 单变量实验选择把 128 样本学习率提高到 `3e-4`，其他参数保持 128 初始实验一致。

```text
run_name: scale128_seed10_lr3e4_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale128_seed10_lr3e4_ga2_dwdse_only
remote_train_log: logs/scale_grid/scale128_seed10_lr3e4_ga2_dwdse_only_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
train_size: 128
valid_size: 128
test_size: 0
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 3e-4
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampler_eval: argmax, steps=64
checkpoint_eval_steps: 400,800,1200,1600
```

结果：

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 1.6210 | 3.4638 | 17/128 = 13.3% | 0/128 = 0.0% | 77/128 | 84/128 |
| 800 | 0.4348 | 5.0422 | 71/128 = 55.5% | 4/128 = 3.1% | 24/128 | 28/128 |
| 1200 | 0.2161 | 3.7589 | 97/128 = 75.8% | 4/128 = 3.1% | 5/128 | 4/128 |
| 1600 | 0.2271 | 3.8031 | 100/128 = 78.1% | 7/128 = 5.5% | 9/128 | 5/128 |

对比 128/lr1e-4：

| run | best train acc | best valid acc | best valid no_eos |
|---|---:|---:|---:|
| `lr=1e-4` | 12/128 = 9.4% | 0/128 = 0.0% | 77/128 |
| `lr=3e-4` | 100/128 = 78.1% | 7/128 = 5.5% | 4/128 |

观察：

1. `lr=3e-4` 明显恢复了训练集生成能力，train acc 从 `12/128` 提高到 `100/128`。
2. valid acc 首次在 128 样本规模出现稳定非零，step 800/1200/1600 分别为 `4/128, 4/128, 7/128`。
3. no_eos 大幅改善，step 1200 valid no_eos 只有 `4/128`，step 1600 为 `5/128`。这说明此前很多失败不是 boxed prompt 本身，而是 lr 太低时 answer denoising 没有学到可结束的生成轨迹。
4. valid loss 并不可靠地预测生成准确率：step 800 valid_loss 最差为 `5.0422`，但 valid acc 已经非零；step 1600 valid_loss `3.8031`，valid acc 最高。
5. 当前最直接的下一步是把 `lr=3e-4` 扩到 256 总量，检查非零 valid 是否随样本量增加而提升。如果 256 valid 仍只有低个位数，再在 `2e-4/3e-4/5e-4` 间继续细扫。

### 11.7 lr=3e-4 扩到 256 总量

在 128/lr3e-4 出现非零 valid 后，把同样 lr 扩到 256 总量划分，保持其余参数和 256/lr5e-5 一致。

```text
run_name: scale256t176v40_seed10_lr3e4_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale256t176v40_seed10_lr3e4_ga2_dwdse_only
remote_train_log: logs/scale_grid/scale256t176v40_seed10_lr3e4_ga2_dwdse_only_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
total_used: 256
train_size: 176
valid_size: 40
test_size: 40
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 3e-4
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
eval_batches: 10
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampler_eval: argmax, steps=64
checkpoint_eval_steps: 400,800,1200,1600
```

结果：

| step | train loss | valid loss | train eval | train acc | valid acc | train no_eos | valid no_eos |
|---:|---:|---:|---|---:|---:|---:|---:|
| 400 | 0.9409 | 3.4174 | train first 64 | 21/64 = 32.8% | 2/40 = 5.0% | 26/64 | 15/40 |
| 800 | 0.7534 | 2.7869 | train first 64 | 32/64 = 50.0% | 1/40 = 2.5% | 23/64 | 13/40 |
| 1200 | 0.3598 | 2.4850 | train first 64 | 30/64 = 46.9% | 1/40 = 2.5% | 10/64 | 6/40 |
| 1600 | 0.3322 | 3.3795 | train first 64 | 36/64 = 56.2% | 1/40 = 2.5% | 21/64 | 14/40 |

补充观察：

1. 相比 256/lr5e-5 的 valid 全 0，`lr=3e-4` 把 valid 拉到了非零：最好 `2/40 = 5.0%`。
2. 但扩大到 256 后，valid 比例没有超过 128/lr3e-4 的 `7/128 = 5.5%`。样本更多没有自动带来更好泛化。
3. train first 64 acc 最高 `36/64 = 56.2%`，低于 128/lr3e-4 的 train full `100/128 = 78.1%`，说明 256 在同样步数下训练记忆也更弱。
4. no_eos 比低 lr 大幅改善，但仍比 128/lr3e-4 更差，尤其 step400/800/1600。
5. valid loss 最好出现在未保存 checkpoint 的 step 1100：`valid_loss=1.9358`。当前 save_freq=400 导致无法评估这个点，后续若继续这个方向，建议把 `save_freq` 改成 100 仅用于曲线观察。

当前 lr 搜索结论：`lr=3e-4` 是目前最有希望的方向，因为它让生成准确率从全 0 变成非零，并显著改善 EOS。但 256 的 valid 仍只有低个位数。下一步优先建议继续在 lr 上细扫，例如：

```text
128 or 256:
  lr = 2e-4
  lr = 5e-4
  save_freq = 100  # 只为观察 900/1000/1100/1200 等中间点，不改变训练目标
```

### 11.8 lr=2e-4 细扫，以及 full-context eval 修正

在 128/lr3e-4 出现训练集过拟合和少量 valid 非零后，继续做 lr 细扫，把学习率放在 `1e-4` 与 `3e-4` 中间：`2e-4`。为了观察中间 checkpoint，把 `save_freq` 改成 100，其余训练目标和 LoRA 参数保持不变。

```text
run_name: scale128_seed10_lr2e4_ga2_save100_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale128_seed10_lr2e4_ga2_save100_dwdse_only
remote_train_log: logs/scale_grid/scale128_seed10_lr2e4_ga2_save100_dwdse_only_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
train_size: 128
valid_size: 128
test_size: 0
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 2e-4
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 100
eval_freq: 100
eval_batches: 10
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
sampling_eval: argmax, steps=64, num_samples=1
```

运行备注：

1. 第一次启动在 step 60 附近中断，远端 GPU 表现为 `1 MiB, 100%` 且无训练进程。执行一次小 CUDA matmul 后 GPU util 恢复，再从头重启。
2. 重启后训练正常完成到 step 1600，保存了 `checkpoint_step_100.pt` 到 `checkpoint_step_1600.pt` 以及 `lora_final.pt`。
3. 训练峰值显存约 21.4GB；训练结束后 GPU 空闲。

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 100 | 2.3927 | 3.0384 |
| 200 | 1.6189 | 2.5769 |
| 300 | 1.2093 | 3.5091 |
| 400 | 1.3814 | 3.3852 |
| 500 | 0.7176 | 4.4447 |
| 600 | 0.8794 | 3.7978 |
| 700 | 0.7256 | 3.3923 |
| 800 | 0.5546 | 5.0887 |
| 900 | 0.5513 | 4.1180 |
| 1000 | 0.5399 | 3.2894 |
| 1100 | 0.3674 | 3.6977 |
| 1200 | 0.3419 | 3.7041 |
| 1300 | 0.2902 | 4.4934 |
| 1400 | 0.2734 | 4.0292 |
| 1500 | 0.3256 | 3.8504 |
| 1600 | 0.3776 | 3.7420 |

评估逻辑修正：

旧的 `eval_correct.py` 是逐题构造 `prompt + answer_window` 的短序列，生成长度约为 `prompt_len + 32`。但训练时 `train_sft.py` 的输入始终是 `max_length=1024`：`prompt + answer + EOS pad`，loss 只算 answer 区域，prompt 和 pad 区域不该参与反向生成。这个差异会造成 train/eval 上下文边界不一致。

因此新增了 `eval_correct_batched.py`，只作为评估脚本，不改变训练目标。它的采样逻辑是：

```text
x shape: [batch, 1024]
prompt 区域: clamp 为原 prompt token
answer 区域: graph.sample_limit / absorb mask 初始化，反向生成
answer 后区域: clamp 为 EOS pad
每一步 reverse update 后重新 clamp prompt 和 EOS pad
```

这个逻辑更贴近训练时的 1024 上下文，也能 batched eval。当前使用 `eval_batch_size=16`，推理显存约 23.6GB。

重要：本节的生成评估使用 full-context batched eval；11.6 和 11.7 中记录的 `lr=3e-4` 数字仍是旧短序列 eval，不能和本节直接做严格横向比较。后续需要用同一套 full-context eval 重新评估 `lr=3e-4` 的 128/256 checkpoint。

full-context 生成评估结果：

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1.6189 | 2.5769 | 22/128 = 17.2% | 1/128 = 0.8% | 0/128 | 0/128 | 0/128 |
| 400 | 1.3814 | 3.3852 | 78/128 = 60.9% | 4/128 = 3.1% | 0/128 | 1/128 | 0/128 |
| 800 | 0.5546 | 5.0887 | 100/128 = 78.1% | 6/128 = 4.7% | 0/128 | 0/128 | 0/128 |
| 1200 | 0.3419 | 3.7041 | 104/128 = 81.2% | 6/128 = 4.7% | 0/128 | 0/128 | 0/128 |
| 1600 | 0.3776 | 3.7420 | 104/128 = 81.2% | 6/128 = 4.7% | 2/128 | 2/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr2e4_ga2_save100_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

观察：

1. `lr=2e-4` 在 full-context eval 下明确发生训练集过拟合：train acc 从 step 200 的 `22/128` 上升到 step 1200/1600 的 `104/128`。
2. valid acc 最高只有 `6/128 = 4.7%`，从 step 800 开始基本平台化。它说明模型学会了条件生成格式和部分短答案，但泛化仍很弱。
3. no_eos 问题几乎消失，说明旧评估中的大量 no_eos 至少有一部分来自上下文边界不一致，而不是训练目标完全无法学 EOS。
4. prompt truncation 全部为 0，当前 `max_length=1024, min_answer_len=32` 对这组 128/128 split 没有题目截断。
5. valid loss 不预测生成准确率：step 200 的 valid loss 最低 `2.5769`，但 valid acc 只有 `1/128`；step 800 的 valid loss 最高 `5.0887`，valid acc 反而是 `6/128`。
6. 失败样本常见两类：长 LaTeX/文本答案会生成正确前缀但拖出额外尾巴，导致 normalized match 失败；valid 上大量数字答案会回忆训练集中常见数字，表现为记忆而非推理。

当前结论：

1. 需要把“采样时 full 1024 context，prompt 和 EOS pad 每步 clamp”作为后续标准评估方式。
2. 不能再用旧短序列 eval 的 no_eos/accuracy 直接判断训练好坏。
3. `lr=2e-4` 已经足够让 128 训练集强过拟合，但 valid 仍只有 `4.7%`；后续优化重点仍是泛化，而不是继续证明 train 能记忆。
4. 下一步的严格比较应先重评 `lr=3e-4` 的 128 和 256 checkpoint，使用同一个 `eval_correct_batched.py`。如果 `lr=3e-4` full-context 仍明显优于 `2e-4`，再继续试 `5e-4`；如果二者接近，则优先扩大到 256/全量并观察 full-context valid 曲线。

### 11.9 128/lr3e-4 使用 full-context eval 重评

为了和 11.8 的 `lr=2e-4` 公平比较，重新评估 `scale128_seed10_lr3e4_ga2_dwdse_only` 的 128 样本 checkpoint。训练 checkpoint 不变，只更换评估脚本为 `eval_correct_batched.py`：

```text
run_name: scale128_seed10_lr3e4_ga2_dwdse_only
eval_script: eval_correct_batched.py
eval_context: full 1024
clamp: prompt + answer 后 EOS pad 每步 clamp
answer_init: graph.sample_limit / absorb mask
eval_batch_size: 16
sampler_eval: argmax, steps=64, num_samples=1
checkpoint_eval_steps: 400,800,1200,1600
```

full-context 重评结果：

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 1.6210 | 3.4638 | 70/128 = 54.7% | 3/128 = 2.3% | 1/128 | 0/128 | 0/128 |
| 800 | 0.4348 | 5.0422 | 101/128 = 78.9% | 5/128 = 3.9% | 1/128 | 1/128 | 0/128 |
| 1200 | 0.2161 | 3.7589 | 112/128 = 87.5% | 4/128 = 3.1% | 0/128 | 0/128 | 0/128 |
| 1600 | 0.2271 | 3.8031 | 111/128 = 86.7% | 10/128 = 7.8% | 1/128 | 0/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr3e4_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

和旧短序列 eval 的差异：

| step | old train acc | full-context train acc | old valid acc | full-context valid acc |
|---:|---:|---:|---:|---:|
| 400 | 17/128 = 13.3% | 70/128 = 54.7% | 0/128 = 0.0% | 3/128 = 2.3% |
| 800 | 71/128 = 55.5% | 101/128 = 78.9% | 4/128 = 3.1% | 5/128 = 3.9% |
| 1200 | 97/128 = 75.8% | 112/128 = 87.5% | 4/128 = 3.1% | 4/128 = 3.1% |
| 1600 | 100/128 = 78.1% | 111/128 = 86.7% | 7/128 = 5.5% | 10/128 = 7.8% |

同口径比较 `lr=2e-4` vs `lr=3e-4`：

| run | best train acc | best valid acc | best valid step | valid no_eos at best |
|---|---:|---:|---:|---:|
| `lr=2e-4` | 104/128 = 81.2% | 6/128 = 4.7% | 800/1200/1600 | 0/128 or 2/128 |
| `lr=3e-4` | 112/128 = 87.5% | 10/128 = 7.8% | 1600 | 0/128 |

观察：

1. full-context eval 让 `lr=3e-4` 的训练集生成准确率显著升高，特别是 step 400 从 `17/128` 变成 `70/128`。这说明旧短序列 eval 确实低估了模型能力。
2. valid 也从旧评估的最好 `7/128` 提高到 `10/128`，但仍是低个位数比例，泛化问题没有解决。
3. 在同一套 full-context eval 下，`lr=3e-4` 优于 `lr=2e-4`：train 更强，valid 也更高。
4. no_eos 基本不再是主因。当前主要失败是答案语义错误、复制训练集中常见短答案，以及长公式/文本答案生成额外尾巴。

当前更新后的 lr 结论：

1. 后续 128 样本 lr 搜索里，`3e-4` 暂时优于 `2e-4`。
2. 如果继续扫 lr，可以尝试 `5e-4`，但需要监控是否过快记忆、valid 下降或长答案尾巴更严重。
3. 在开启新训练前，更应该先把已有的 256/lr3e-4 checkpoint 也用 full-context eval 重评，因为旧 256 结果同样使用短序列 eval，可能低估了 train/valid。

### 11.10 256/lr3e-4 使用 full-context eval 重评

继续重评已有的 `scale256t176v40_seed10_lr3e4_ga2_dwdse_only`。训练 checkpoint 不变，只把评估改为 full-context batched eval。

为避免远端长 SSH 被中断，本轮采用和旧 256 记录可比的口径：

```text
train eval: train first 64
valid eval: valid full 40
eval_script: eval_correct_batched.py
eval_context: full 1024
clamp: prompt + answer 后 EOS pad 每步 clamp
eval_batch_size: 16
sampler_eval: argmax, steps=64, num_samples=1
```

full-context 重评结果：

| step | train loss | valid loss | train eval | train acc | valid acc | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 400 | 0.9409 | 3.4174 | train first 64 | 33/64 = 51.6% | 1/40 = 2.5% | 6/64 | 1/40 | 0 |
| 800 | 0.7534 | 2.7869 | train first 64 | 49/64 = 76.6% | 3/40 = 7.5% | 0/64 | 0/40 | 0 |
| 1200 | 0.3598 | 2.4850 | train first 64 | 44/64 = 68.8% | 2/40 = 5.0% | 1/64 | 0/40 | 0 |
| 1600 | 0.3322 | 3.3795 | train first 64 | 55/64 = 85.9% | 2/40 = 5.0% | 0/64 | 0/40 | 0 |

输出文件：

```text
logs/scale_grid/eval_scale256t176v40_seed10_lr3e4_ga2_dwdse_only_step{step}_train64_batched_fullctx_argmax64.jsonl
logs/scale_grid/eval_scale256t176v40_seed10_lr3e4_ga2_dwdse_only_step{step}_valid_batched_fullctx_argmax64.jsonl
```

和旧短序列 eval 的差异：

| step | old train64 acc | full-context train64 acc | old valid acc | full-context valid acc |
|---:|---:|---:|---:|---:|
| 400 | 21/64 = 32.8% | 33/64 = 51.6% | 2/40 = 5.0% | 1/40 = 2.5% |
| 800 | 32/64 = 50.0% | 49/64 = 76.6% | 1/40 = 2.5% | 3/40 = 7.5% |
| 1200 | 30/64 = 46.9% | 44/64 = 68.8% | 1/40 = 2.5% | 2/40 = 5.0% |
| 1600 | 36/64 = 56.2% | 55/64 = 85.9% | 1/40 = 2.5% | 2/40 = 5.0% |

观察：

1. 和 128 一样，full-context eval 明显提高 train accuracy，说明短序列 eval 也低估了 256 的训练集记忆能力。
2. valid 最好是 `3/40 = 7.5%`，比旧 eval 的最好 `2/40 = 5.0%` 略高，但样本少，差异不应过度解释。
3. 256/lr3e-4 的 valid 比例和 128/lr3e-4 的 `10/128 = 7.8%` 基本持平，没有看到扩大到 256 后泛化比例提升。
4. no_eos 几乎消失，进一步确认 EOS 问题主要来自旧 eval 上下文不一致。
5. 256 的 train64 在 step 1600 达到 `55/64 = 85.9%`，接近 128/lr3e-4 的 train full `111/128 = 86.7%`。这说明 256 在同样 step 下也能记忆相当一部分训练样本。

更新后的样本规模结论：

1. 修正 eval 后，`128/lr3e-4` 和 `256/lr3e-4` 都能发生明显训练集过拟合。
2. 泛化仍然没有随样本量从 128 到 256 明显提升：128 valid best `7.8%`，256 valid best `7.5%`。
3. 因此当前瓶颈不是“模型完全学不会 answer 生成”，而是“在 DWDSE-only + LoRA 当前设置下，answer 格式/短答案记忆学会了，但 holdout 泛化很弱”。
4. 下一步如果继续样本规模线，应在 full-context eval 下直接跑 full le32 的可比 lr（优先 `3e-4` 或先短跑 `5e-4` 128 判断是否更好），而不是依赖旧 full/lr5e-5 的短序列 eval。

### 11.11 full le32/lr3e-4 可比训练与 full-context eval

在 128/256 的 `lr=3e-4` full-context 重评后，补齐 full le32 同口径实验。目的：判断把训练量扩到 le32 全量后，valid accuracy 是否明显超过 128/256 的 7-8% 水平。

```text
run_name: scale586t458v64_seed10_lr3e4_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale586t458v64_seed10_lr3e4_ga2_dwdse_only
remote_train_log: logs/scale_grid/scale586t458v64_seed10_lr3e4_ga2_dwdse_only_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
total_used: 586
train_size: 458
valid_size: 64
test_size: 64
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 3e-4
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
eval_batches: 10
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
dtype: bfloat16
offline: True
```

训练状态：

1. 训练正常完成到 step 1600，保存 `checkpoint_step_400/800/1200/1600.pt` 和 `lora_final.pt`。
2. 训练显存约 21.4GB；结束后 GPU 空闲。
3. 数据盘在评估后约 50% 使用率，系统盘约 13%，无明显磁盘压力。

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 100 | 3.3056 | 3.5254 |
| 200 | 2.5898 | 2.2116 |
| 300 | 2.5564 | 3.5293 |
| 400 | 2.5688 | 3.5289 |
| 500 | 1.3946 | 3.9062 |
| 600 | 1.1782 | 3.5402 |
| 700 | 1.1797 | 2.5776 |
| 800 | 1.5233 | 3.0571 |
| 900 | 1.1735 | 5.0794 |
| 1000 | 1.0562 | 2.8145 |
| 1100 | 1.0388 | 2.4488 |
| 1200 | 1.0642 | 3.0529 |
| 1300 | 0.5629 | 3.7364 |
| 1400 | 0.6781 | 2.6654 |
| 1500 | 0.6541 | 3.4449 |
| 1600 | 0.6943 | 3.3356 |

valid loss 最低出现在未保存 checkpoint 的 step 200：`2.2116`。保存点里 step 800/1200 都约 `3.05`，step 1600 为 `3.3356`。和前面实验一样，valid loss 仍不可靠预测生成准确率。

full-context 生成评估：

```text
eval_script: eval_correct_batched.py
eval_context: full 1024
clamp: prompt + answer 后 EOS pad 每步 clamp
answer_init: graph.sample_limit / absorb mask
eval_batch_size: 16
sampler_eval: argmax, steps=64, num_samples=1
train eval: train first 64
valid eval: valid full 64
```

结果：

| step | train loss | valid loss | train eval | train acc | valid acc | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 400 | 2.5688 | 3.5289 | train first 64 | 10/64 = 15.6% | 2/64 = 3.1% | 34/64 | 36/64 | 0 |
| 800 | 1.5233 | 3.0571 | train first 64 | 41/64 = 64.1% | 4/64 = 6.2% | 2/64 | 0/64 | 0 |
| 1200 | 1.0642 | 3.0529 | train first 64 | 40/64 = 62.5% | 4/64 = 6.2% | 0/64 | 0/64 | 0 |
| 1600 | 0.6943 | 3.3356 | train first 64 | 48/64 = 75.0% | 6/64 = 9.4% | 1/64 | 0/64 | 0 |

输出文件：

```text
logs/scale_grid/eval_scale586t458v64_seed10_lr3e4_ga2_dwdse_only_step{step}_train64_batched_fullctx_argmax64.jsonl
logs/scale_grid/eval_scale586t458v64_seed10_lr3e4_ga2_dwdse_only_step{step}_valid_batched_fullctx_argmax64.jsonl
```

和 128/256/lr3e-4 的同口径比较：

| scale/run | train eval | best train acc | valid eval | best valid acc |
|---|---|---:|---|---:|
| 128/lr3e-4 | train full 128 | 112/128 = 87.5% | valid full 128 | 10/128 = 7.8% |
| 256/lr3e-4 | train first 64 | 55/64 = 85.9% | valid full 40 | 3/40 = 7.5% |
| full le32/lr3e-4 | train first 64 | 48/64 = 75.0% | valid full 64 | 6/64 = 9.4% |

观察：

1. full le32 的 valid best `6/64 = 9.4%`，比 128/256 的 `7-8%` 略高，但不是质变。
2. full le32 的 train64 记忆程度低于 128/256：best `48/64 = 75.0%`，说明同样 1600 steps 下，训练样本更多会稀释每个样本的记忆强度。
3. step 400 仍有大量 no_eos，valid no_eos `36/64`；到 step 800 后 no_eos 基本消失。full scale 需要更长步数才能形成稳定 EOS/短答案生成。
4. valid accuracy 在 step 1600 最高，而 valid loss 在 step 1600 不是最低；仍然不能用 scalar valid loss 直接选生成 checkpoint。
5. 当前证据支持：增加样本量到 full le32 有轻微 valid 提升，但幅度很小，仍然处在个位数准确率。

更新后的样本规模结论：

1. 在同一 `lr=3e-4, CE=0, LoRA r32/alpha64/dropout0, full-context eval` 口径下：
   - 128: best valid `7.8%`
   - 256: best valid `7.5%`
   - full le32: best valid `9.4%`
2. 样本量增加没有带来明显泛化跃迁；最多只能说 full le32 略高。
3. full le32 训练集记忆低于小样本，说明如果继续 full scale，可能需要更多 steps 或更合适的 schedule，但仅加 steps 可能进一步过拟合常见答案。
4. 下一阶段应该进入第二优先级的超参数搜索，而不是继续只扩大样本量。当前最合理的下一步是：
   - 先在 128 或 full le32 上试 `lr=5e-4`，确认更高 lr 是否能提高 valid 或只是加速记忆；
   - 然后再考虑 LoRA rank/alpha 正则化、dropout、weight decay。

### 11.12 lr 搜索：128 样本 lr=5e-4

样本规模线显示 full le32 只带来很小 valid 提升，因此进入第二优先级的 lr 搜索。为了判断更高学习率是否能继续提升泛化，先在 128 样本上把 `lr=3e-4` 提高到 `5e-4`，其余参数保持不变。

```text
run_name: scale128_seed10_lr5e4_ga2_dwdse_only
remote_out_dir: /root/autodl-tmp/sedd_outputs_archive/scale_grid/scale128_seed10_lr5e4_ga2_dwdse_only
remote_train_log: logs/scale_grid/scale128_seed10_lr5e4_ga2_dwdse_only_train.log
data: data/s1K_train_answer_inner_le32.json
seed: 10
train_size: 128
valid_size: 128
test_size: 0
loss: response-only DWDSE
all_mask_ce_weight: 0
lr: 5e-4
warmup_steps: 100
batch_size: 16
grad_accum: 2
effective_batch: 32
max_steps: 1600
save_freq: 400
eval_freq: 100
eval_batches: 10
lora_r: 32
lora_alpha: 64
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
boxed_prompt: True
```

训练备注：

1. 训练正常完成到 step 1600，保存 `checkpoint_step_400/800/1200/1600.pt` 和 `lora_final.pt`。
2. 早期 step 60 出现一次 DWDSE spike：`12.8986`，随后恢复，没有中断。
3. 后期 train loss 很低，step 1600 为 `0.1607`，明显强于 `lr=3e-4` 的 step 1600 `0.2271`。

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 100 | 2.1560 | 3.0981 |
| 200 | 0.9232 | 2.5563 |
| 300 | 0.7279 | 3.8442 |
| 400 | 1.1116 | 3.3864 |
| 500 | 0.4604 | 4.3547 |
| 600 | 0.4919 | 4.0317 |
| 700 | 0.5674 | 3.4748 |
| 800 | 0.4162 | 4.2605 |
| 900 | 0.2206 | 4.2396 |
| 1000 | 0.2965 | 3.2878 |
| 1100 | 0.2753 | 3.6772 |
| 1200 | 0.1760 | 3.8704 |
| 1300 | 0.1257 | 4.5497 |
| 1400 | 0.1423 | 4.0144 |
| 1500 | 0.1669 | 3.8761 |
| 1600 | 0.1607 | 3.8167 |

valid loss 最低为 step 200：`2.5563`；保存点里最好是 step 400：`3.3864`。后续 valid loss 大多在 `3.8-4.5` 区间。

full-context 生成评估：

| step | train loss | valid loss | train acc | valid acc | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 1.1116 | 3.3864 | 74/128 = 57.8% | 5/128 = 3.9% | 0/128 | 0/128 | 0/128 |
| 800 | 0.4162 | 4.2605 | 109/128 = 85.2% | 6/128 = 4.7% | 0/128 | 0/128 | 0/128 |
| 1200 | 0.1760 | 3.8704 | 113/128 = 88.3% | 4/128 = 3.1% | 0/128 | 0/128 | 0/128 |
| 1600 | 0.1607 | 3.8167 | 115/128 = 89.8% | 4/128 = 3.1% | 0/128 | 0/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr5e4_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

lr 同口径比较：

| lr | best train acc | best valid acc | best valid step | comment |
|---:|---:|---:|---:|---|
| 2e-4 | 104/128 = 81.2% | 6/128 = 4.7% | 800/1200/1600 | 训练较慢，valid 低 |
| 3e-4 | 112/128 = 87.5% | 10/128 = 7.8% | 1600 | 当前最好 |
| 5e-4 | 115/128 = 89.8% | 6/128 = 4.7% | 800 | 更强记忆，但 valid 下降 |

观察：

1. `5e-4` 确实进一步增强训练集记忆，train best 到 `115/128 = 89.8%`。
2. valid 没有提升，最好只有 `6/128 = 4.7%`，低于 `3e-4` 的 `10/128 = 7.8%`。
3. no_eos 全部为 0，说明这个阶段 EOS 不是限制因素。
4. 当前结果更像是学习率过高导致更快记忆/过拟合，而不是提升泛化。

更新后的 lr 结论：

1. 在当前 LoRA r32/alpha64/dropout0/weight_decay0 下，`lr=3e-4` 是目前最好的学习率。
2. `5e-4` 不建议继续扩到 256/full，除非配合更强正则化。
3. 下一步进入 LoRA rank/alpha 正则化：保持 `lr=3e-4`，先降低 LoRA 容量，例如 `r=16, alpha=32` 或 `r=16, alpha=16`，看能否牺牲少量 train 记忆换 valid 提升。

### 11.13 LoRA rank/alpha 搜索：128 样本 r16/alpha32

目的：

1. 保持当前最好学习率 `lr=3e-4` 不变。
2. 将 LoRA 容量从 `r32/alpha64` 降到 `r16/alpha32`，保持 `alpha/r=2` 不变。
3. 验证降低可训练参数量是否能减少训练集记忆、提升 unseen valid accuracy。

run：

```text
scale128_seed10_lr3e4_r16a32_ga2_dwdse_only
```

关键参数：

```text
data_json: data/s1K_train_answer_inner_le32.json
train_size: 128
valid_size: 128
test_size: 0
seed: 10
batch_size: 16
grad_accum: 2
effective batch: 32
lr: 3e-4
warmup_steps: 100
max_steps: 1600
eval_freq: 100
save_freq: 400
eval_batches: 10
all_mask_ce_weight: 0.0
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
answer_leading_newline: true
boxed_prompt: true
dtype: bfloat16
offline: true
```

训练日志确认：

```text
LoRA modules: 96 matched (attn_qkv, attn_out, mlp.0, mlp.2)
Trainable params: 6,291,456/430,768,466 (1.4605%)
Using response-only DWDSE
all_mask_ce=0.0000
```

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 400 | 0.8353 | 3.6871 |
| 800 | 0.5740 | 4.6599 |
| 1200 | 0.4157 | 3.6686 |
| 1600 | 0.2587 | 3.7060 |

full-context 生成评估，主指标使用 `exact_target_match`：

| step | train loss | valid loss | train exact | valid exact | train boxed | valid boxed | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.8353 | 3.6871 | 74/128 = 57.8% | 4/128 = 3.1% | 75/128 = 58.6% | 4/128 = 3.1% | 0/128 | 0/128 | 0/128 |
| 800 | 0.5740 | 4.6599 | 97/128 = 75.8% | 3/128 = 2.3% | 97/128 = 75.8% | 3/128 = 2.3% | 0/128 | 0/128 | 0/128 |
| 1200 | 0.4157 | 3.6686 | 106/128 = 82.8% | 4/128 = 3.1% | 106/128 = 82.8% | 4/128 = 3.1% | 1/128 | 0/128 | 0/128 |
| 1600 | 0.2587 | 3.7060 | 111/128 = 86.7% | 5/128 = 3.9% | 112/128 = 87.5% | 6/128 = 4.7% | 1/128 | 0/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr3e4_r16a32_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

和 `r32/alpha64, lr=3e-4` 对比：

| config | trainable params | best train exact | best valid exact | best valid step | comment |
|---|---:|---:|---:|---:|---|
| r32/alpha64 | 12.58M | 112/128 = 87.5% | 10/128 = 7.8% | 1600 | 当前 128 样本最好 |
| r16/alpha32 | 6.29M | 111/128 = 86.7% | 5/128 = 3.9% | 1600 | 降低容量，但 valid 下降 |

观察：

1. `r16/alpha32` 在 step800 前确实压低了训练集记忆，例如 step800 train exact 为 `97/128`，低于 r32 的 `101/128`。
2. 到 step1600，`r16/alpha32` 的训练集拟合几乎追上 r32，train exact 达到 `111/128`。
3. valid 没有受益，strict best 只有 `5/128 = 3.9%`；即使按 boxed pass，也只有 `6/128 = 4.7%`。
4. no_eos 不是主要问题：valid 全部 `0/128`，train 只有后期 `1/128`。
5. prompt trunc 始终为 `0/128`，本轮没有题目截断导致的 train/eval prompt 不一致。

当前 rank/alpha 结论：

1. 单纯把 rank 从 32 降到 16，并保持 `alpha/r=2`，没有提升泛化。
2. r16 的容量仍足够记忆训练集，最终 train exact 接近 r32，但 valid 更差。
3. 下一步如果继续 rank/alpha，应尝试更强的 adapter 缩放正则化，例如 `r16/alpha16`，即把 `alpha/r` 从 2 降到 1；如果仍不提升，再进入 LoRA dropout 或 weight decay。

### 11.14 LoRA rank/alpha 搜索：128 样本 r16/alpha16

目的：

1. 继续保持样本、学习率、batch、训练步数全部不变。
2. 在 `r16` 的基础上，将 `alpha` 从 32 降到 16，把 `alpha/r` 从 2 降到 1。
3. 验证更强的 adapter 缩放正则化是否能降低训练集记忆并提升 valid。

run：

```text
scale128_seed10_lr3e4_r16a16_ga2_dwdse_only
```

关键参数：

```text
train_size: 128
valid_size: 128
test_size: 0
seed: 10
batch_size: 16
grad_accum: 2
effective batch: 32
lr: 3e-4
warmup_steps: 100
max_steps: 1600
eval_freq: 100
save_freq: 400
eval_batches: 10
all_mask_ce_weight: 0.0
lora_r: 16
lora_alpha: 16
lora_dropout: 0.0
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
answer_leading_newline: true
boxed_prompt: true
dtype: bfloat16
offline: true
```

训练日志确认：

```text
LoRA modules: 96 matched (attn_qkv, attn_out, mlp.0, mlp.2)
Trainable params: 6,291,456/430,768,466 (1.4605%)
Using response-only DWDSE
all_mask_ce=0.0000
```

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 400 | 0.9862 | 3.6853 |
| 800 | 0.9910 | 5.2482 |
| 1200 | 0.5074 | 3.6698 |
| 1600 | 0.4294 | 3.8219 |

full-context 生成评估，主指标使用 `exact_target_match`：

| step | train loss | valid loss | train exact | valid exact | train boxed | valid boxed | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.9862 | 3.6853 | 74/128 = 57.8% | 2/128 = 1.6% | 74/128 = 57.8% | 2/128 = 1.6% | 0/128 | 1/128 | 0/128 |
| 800 | 0.9910 | 5.2482 | 89/128 = 69.5% | 4/128 = 3.1% | 89/128 = 69.5% | 4/128 = 3.1% | 0/128 | 0/128 | 0/128 |
| 1200 | 0.5074 | 3.6698 | 96/128 = 75.0% | 4/128 = 3.1% | 96/128 = 75.0% | 4/128 = 3.1% | 1/128 | 2/128 | 0/128 |
| 1600 | 0.4294 | 3.8219 | 106/128 = 82.8% | 4/128 = 3.1% | 106/128 = 82.8% | 4/128 = 3.1% | 0/128 | 2/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr3e4_r16a16_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

rank/alpha 同口径比较：

| config | alpha/r | trainable params | best train exact | best valid exact | valid no_eos | comment |
|---|---:|---:|---:|---:|---:|---|
| r32/alpha64 | 2 | 12.58M | 112/128 = 87.5% | 10/128 = 7.8% | 0/128 | 当前 128 样本最好 |
| r16/alpha32 | 2 | 6.29M | 111/128 = 86.7% | 5/128 = 3.9% | 0/128 | 降低 rank，无收益 |
| r16/alpha16 | 1 | 6.29M | 106/128 = 82.8% | 4/128 = 3.1% | up to 2/128 | 更强缩放，valid 更差 |

观察：

1. `r16/alpha16` 明确降低了训练集拟合速度和最终 train acc：step1600 train exact 为 `106/128`，低于 r16/alpha32 的 `111/128`。
2. valid 没有改善，最好只有 `4/128 = 3.1%`，低于 r16/alpha32，也低于 r32/alpha64。
3. 更强 adapter 缩放还带来少量 no_eos，valid 在 step1200/1600 有 `2/128`。
4. prompt trunc 仍然为 0，输入长度不是本轮差异来源。

当前 rank/alpha 分支结论：

1. `r16/alpha32` 和 `r16/alpha16` 都没有提升 unseen valid accuracy。
2. 降低 rank/alpha 能压低训练集记忆，但没有转化为泛化；因此目前不是简单容量过大导致的问题。
3. 当前 128 样本最好配置仍是 `r32/alpha64, lr=3e-4, dropout=0, weight_decay=0`，best strict valid `10/128 = 7.8%`。
4. 下一步按原超参搜索顺序进入 LoRA dropout：建议先保持 best base config 不变，只加 `lora_dropout=0.05`，看是否能在保持 train 拟合的同时提升 valid；如果无效，再试 `weight_decay`。

### 11.15 LoRA dropout 搜索：128 样本 dropout=0.05

目的：

1. 回到当前 128 样本最好配置 `r32/alpha64, lr=3e-4`。
2. 只改一个参数：`lora_dropout=0.05`。
3. 验证 dropout 是否能在不明显破坏 train 拟合的情况下提升 unseen valid accuracy。

run：

```text
scale128_seed10_lr3e4_r32a64_do005_ga2_dwdse_only
```

关键参数：

```text
train_size: 128
valid_size: 128
test_size: 0
seed: 10
batch_size: 16
grad_accum: 2
effective batch: 32
lr: 3e-4
warmup_steps: 100
max_steps: 1600
eval_freq: 100
save_freq: 400
eval_batches: 10
all_mask_ce_weight: 0.0
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
weight_decay: 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_field: final_boxed
answer_prefix: Answer:
answer_leading_newline: true
boxed_prompt: true
dtype: bfloat16
offline: true
```

训练日志确认：

```text
LoRA modules: 96 matched (attn_qkv, attn_out, mlp.0, mlp.2)
Trainable params: 12,582,912/437,059,922 (2.8790%)
Using response-only DWDSE
all_mask_ce=0.0000
```

训练 loss / valid loss：

| step | train loss | valid loss |
|---:|---:|---:|
| 400 | 0.8096 | 4.5248 |
| 800 | 0.4378 | 4.2163 |
| 1200 | 0.3876 | 4.6673 |
| 1600 | 0.2530 | 5.1617 |

full-context 生成评估，主指标使用 `exact_target_match`：

| step | train exact | valid exact | train boxed | valid boxed | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 77/128 = 60.2% | 3/128 = 2.3% | 77/128 = 60.2% | 3/128 = 2.3% | 4/128 | 5/128 | 0/128 |
| 800 | 82/128 = 64.1% | 6/128 = 4.7% | 82/128 = 64.1% | 6/128 = 4.7% | 4/128 | 4/128 | 0/128 |
| 1200 | 100/128 = 78.1% | 4/128 = 3.1% | 100/128 = 78.1% | 4/128 = 3.1% | 4/128 | 0/128 | 0/128 |
| 1600 | 102/128 = 79.7% | 5/128 = 3.9% | 103/128 = 80.5% | 5/128 = 3.9% | 2/128 | 0/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr3e4_r32a64_do005_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

和当前 best baseline 对比：

| config | dropout | best train exact | best valid exact | best valid step | no_eos behavior | comment |
|---|---:|---:|---:|---:|---|---|
| r32/alpha64, lr=3e-4 | 0.00 | 112/128 = 87.5% | 10/128 = 7.8% | 1600 | 基本为 0 | 当前最好 |
| r32/alpha64, lr=3e-4 | 0.05 | 102/128 = 79.7% | 6/128 = 4.7% | 800 | early/mid checkpoint 有 4-5 条 no_eos | dropout 压低 train，但 valid 没提升 |

观察：

1. `dropout=0.05` 明显压低了训练集拟合，step1600 train exact 只有 `102/128`，低于无 dropout 的 `112/128`。
2. valid 没有提升，最好是 step800 的 `6/128 = 4.7%`，低于无 dropout baseline 的 `10/128 = 7.8%`。
3. dropout 还带来更多 no_eos，尤其 step400/800 的 train/valid 都有 `4-5/128` no_eos；这说明它对生成终止边界有负面影响。
4. prompt trunc 仍然全部为 0，输入长度不是本轮差异来源。

当前 dropout 结论：

1. `lora_dropout=0.05` 不是有效方向：它降低 train 拟合，但没有提升 unseen valid。
2. 如果继续 dropout 分支，可以试更小的 `dropout=0.01`，但优先级不高，因为 `0.05` 已经引入 no_eos 问题。
3. 按原搜索顺序，下一步更适合进入 `weight_decay`，保持 best baseline 的 `r32/alpha64, lr=3e-4, dropout=0`，只加小权重衰减，例如 `weight_decay=0.01` 或 `0.001`。

暂停点：

用户要求当前服务器命令结束后暂停，等待提供第二台服务器后再继续双服务器并行实验。因此本节先记录生成评估结果，暂不启动下一轮训练。

### 11.16 Weight decay 搜索：128 样本 wd=0.001 / wd=0.01

目的：

1. 在当前 best baseline 上只改 `weight_decay`，检查能否减少 128 样本过拟合并提升 unseen valid。
2. 使用两台 5090 实例并行跑：
   - 实例 1：`weight_decay=0.001`
   - 实例 2：`weight_decay=0.01`

固定配置：

```text
data_json: data/s1K_train_answer_inner_le32.json
train_size / valid_size / test_size: 128 / 128 / 0
seed: 10
answer_field: final_boxed
max_length: 1024
min_answer_len / max_answer_len: 32 / 32
answer_prefix: Answer:
answer_leading_newline: true
boxed_prompt: true
loss: response-only DWDSE
all_mask_ce_weight: 0.0
lora_r / lora_alpha / lora_dropout: 32 / 64 / 0.0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
batch_size: 16
grad_accum: 2
effective batch: 32
lr: 3e-4
warmup_steps: 100
max_steps: 1600
save_freq: 400
eval_freq: 100
eval_batches: 10
dtype: bfloat16
offline: true
```

Run 名：

```text
scale128_seed10_lr3e4_r32a64_wd001_ga2_dwdse_only
scale128_seed10_lr3e4_r32a64_wd01_ga2_dwdse_only
```

训练 loss / valid loss：

| wd | step | train loss | valid loss |
|---:|---:|---:|---:|
| 0.001 | 400 | 1.2665 | 3.4917 |
| 0.001 | 800 | 0.5590 | 4.7377 |
| 0.001 | 1200 | 0.2567 | 3.5672 |
| 0.001 | 1600 | 0.2166 | 3.8148 |
| 0.01 | 400 | 0.9793 | 3.4908 |
| 0.01 | 800 | 0.5121 | 4.8669 |
| 0.01 | 1200 | 0.2525 | 3.7499 |
| 0.01 | 1600 | 0.2345 | 3.8281 |

full-context 生成评估，主指标使用 `exact_target_match`：

| wd | step | train exact | valid exact | train boxed | valid boxed | train no_eos | valid no_eos | prompt trunc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 400 | 76/128 = 59.4% | 5/128 = 3.9% | 76/128 = 59.4% | 5/128 = 3.9% | 0/128 | 1/128 | 0/128 |
| 0.001 | 800 | 103/128 = 80.5% | 6/128 = 4.7% | 103/128 = 80.5% | 6/128 = 4.7% | 2/128 | 1/128 | 0/128 |
| 0.001 | 1200 | 111/128 = 86.7% | 4/128 = 3.1% | 111/128 = 86.7% | 4/128 = 3.1% | 0/128 | 0/128 | 0/128 |
| 0.001 | 1600 | 112/128 = 87.5% | 5/128 = 3.9% | 113/128 = 88.3% | 5/128 = 3.9% | 1/128 | 0/128 | 0/128 |
| 0.01 | 400 | 77/128 = 60.2% | 5/128 = 3.9% | 77/128 = 60.2% | 5/128 = 3.9% | 9/128 | 6/128 | 0/128 |
| 0.01 | 800 | 96/128 = 75.0% | 4/128 = 3.1% | 96/128 = 75.0% | 4/128 = 3.1% | 2/128 | 1/128 | 0/128 |
| 0.01 | 1200 | 106/128 = 82.8% | 5/128 = 3.9% | 107/128 = 83.6% | 5/128 = 3.9% | 0/128 | 0/128 | 0/128 |
| 0.01 | 1600 | 114/128 = 89.1% | 7/128 = 5.5% | 114/128 = 89.1% | 7/128 = 5.5% | 1/128 | 0/128 | 0/128 |

输出文件：

```text
logs/scale_grid/eval_scale128_seed10_lr3e4_r32a64_wd001_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
logs/scale_grid/eval_scale128_seed10_lr3e4_r32a64_wd01_ga2_dwdse_only_step{step}_{split}_batched_fullctx_argmax64.jsonl
```

和当前 no-WD baseline 对比：

| config | weight_decay | best train exact | best valid exact | best valid step | comment |
|---|---:|---:|---:|---:|---|
| r32/alpha64, lr=3e-4 | 0.0 | 112/128 = 87.5% | 10/128 = 7.8% | 1600 | 当前最好 |
| r32/alpha64, lr=3e-4 | 0.001 | 112/128 = 87.5% | 6/128 = 4.7% | 800 | 没有提升 valid |
| r32/alpha64, lr=3e-4 | 0.01 | 114/128 = 89.1% | 7/128 = 5.5% | 1600 | 比 wd=0.001 略高，但仍低于 baseline |

观察：

1. `weight_decay=0.001` 基本没有改善泛化，best valid exact 只有 `6/128 = 4.7%`。
2. `weight_decay=0.01` 在 step1600 valid exact 达到 `7/128 = 5.5%`，但仍低于无 WD baseline 的 `10/128 = 7.8%`。
3. `weight_decay=0.01` 的 step400 no_eos 明显更高：train `9/128`、valid `6/128`，早期生成终止更不稳定。
4. 两个 WD 分支的 train exact 都能继续上涨，说明 WD 没有真正阻止记忆化；valid exact 没有超过 baseline。

当前 WD 结论：

1. `weight_decay=0.001` 和 `0.01` 都不是更好的 checkpoint 配置。
2. 目前 checkpoint 版本仍建议使用 no-WD baseline：`lr=3e-4, r=32, alpha=64, dropout=0, weight_decay=0, all_mask_ce_weight=0`。
3. 如果要做正式 checkpoint，建议把精力放回数据规模配置，而不是继续扩大 WD/dropout 搜索。

### 11.17 样本规模 / 超参曲线本地汇总脚本

为了把已经完成的 full-context 评估结果固定成可复用图表，新增本地分析脚本：

```text
analysis/plot_scale_grid.py
```

脚本数据来源：

1. `CURRENT_PROJECT_RECORD.md` 中已经验证过的 full-context eval 结果。
2. 当前包含：
   - 样本规模曲线：128 / 256 total / full le32，统一使用 `lr=3e-4, r32/alpha64, dropout0, weight_decay0`。
   - 128 样本 lr 搜索：`2e-4, 3e-4, 5e-4`。
   - rank/alpha：`r16/alpha32`, `r16/alpha16`。
   - dropout：`dropout=0.05`。
   - weight decay：`0.001`, `0.01`。

生成命令：

```bash
python analysis/plot_scale_grid.py
```

输出文件：

```text
analysis/outputs/scale_grid_results.csv
analysis/outputs/scale_grid_best_by_run.csv
analysis/outputs/scale_grid_summary.md
analysis/outputs/scale_valid_curve.png
analysis/outputs/scale_valid_heatmap.png
analysis/outputs/hyperparam_valid_heatmap.png
analysis/outputs/best_valid_by_run.png
analysis/outputs/latest_train_vs_valid.png
```

当前曲线结论和 11.9-11.16 一致：

1. 128 baseline best valid：`10/128 = 7.8%` at step1600。
2. 256 total best valid：`3/40 = 7.5%` at step800。
3. full le32 best valid：`6/64 = 9.4%` at step1600。
4. 样本量从 128 -> 256 -> full le32 只有小幅提升，没有泛化跃迁。
5. lr / rank-alpha / dropout / weight decay 分支都没有超过 128 no-WD baseline。

已补：

1. `max_steps / early stopping` 分支已完成，详见 11.18：
   - `scale128_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200`
   - `scale586t458v64_seed10_lr3e4_ga2_dwdse_only_resume1600_to3200`
2. 2000/2400/2800/3200 结果已经追加到 `analysis/plot_scale_grid.py`，并已重新生成 CSV/PNG。

### 11.18 Max steps / early stopping：1600 -> 3200

目的：

1. 检查当前 best baseline 延长训练后，valid exact 是否继续提升。
2. 分别续跑：
   - 128 baseline：从 `scale128_seed10_lr3e4_ga2_dwdse_only/checkpoint_step_1600.pt` 续到 3200。
   - full le32 baseline：从 `scale586t458v64_seed10_lr3e4_ga2_dwdse_only/checkpoint_step_1600.pt` 续到 3200。

结果：

| run | step | train exact | valid exact | no_eos valid | comment |
|---|---:|---:|---:|---:|---|
| 128 resume | 2000 | 116/128 = 90.6% | 7/128 = 5.5% | 0/128 | 低于原 step1600 的 10/128 |
| 128 resume | 2400 | 116/128 = 90.6% | 5/128 = 3.9% | 0/128 | 继续下降 |
| 128 resume | 2800 | 119/128 = 93.0% | 4/128 = 3.1% | 0/128 | train 继续涨，valid 下降 |
| 128 resume | 3200 | 120/128 = 93.8% | 5/128 = 3.9% | 0/128 | 无恢复 |
| full le32 resume | 2000 | 48/64 = 75.0% | 6/64 = 9.4% | 0/64 | 持平原 step1600 |
| full le32 resume | 2400 | 51/64 = 79.7% | 6/64 = 9.4% | 0/64 | 持平 |
| full le32 resume | 2800 | 50/64 = 78.1% | 5/64 = 7.8% | 0/64 | 下降 |
| full le32 resume | 3200 | 55/64 = 85.9% | 6/64 = 9.4% | 0/64 | 回到持平，无提升 |

训练 loss / valid loss：

| run | step | train loss | valid loss |
|---|---:|---:|---:|
| 128 resume | 2000 | 0.2090 | 3.9452 |
| 128 resume | 2400 | 0.1463 | 5.1491 |
| 128 resume | 2800 | 0.0725 | 4.0788 |
| 128 resume | 3200 | 0.0942 | 4.1557 |
| full le32 resume | 2000 | 0.5133 | 4.0583 |
| full le32 resume | 2400 | 0.4339 | 2.9297 |
| full le32 resume | 2800 | 0.5847 | 3.2654 |
| full le32 resume | 3200 | 0.4093 | 3.5352 |

结论：

1. 对 128 baseline，延长训练显著提高 train exact：`111/128` at step1600 -> `120/128` at step3200，但 valid 从 `10/128` 降到 `5/128`。因此 128 最好 checkpoint 仍是 step1600。
2. 对 full le32，延长训练只让 train64 从 `48/64` 提高到 `55/64`，valid 始终没有超过原 step1600 的 `6/64 = 9.4%`。
3. `max_steps / early stopping` 分支没有带来泛化提升；更长训练主要增加记忆，不提升 valid。
4. 本轮 LoRA SFT 小网格已覆盖：样本规模、lr、rank/alpha、dropout、weight_decay、max_steps。当前最好实用配置仍是 `lr=3e-4, r32/alpha64, dropout0, weight_decay0, step1600`。

### 11.19 本轮样本规模与超参小网格最终总结

本轮目标：

1. 先做样本规模曲线：`128 -> 256 -> full le32`。
2. 如果泛化提升不明显，再逐项搜索超参：`lr -> LoRA rank/alpha -> dropout -> weight_decay -> max_steps / early stopping`。
3. 主指标统一使用 full-context batched eval 的 `exact_target_match`，采样为 `argmax, steps=64, num_samples=1`。

最终最好结果：

| category | best config | best valid exact | best step | comment |
|---|---|---:|---:|---|
| 128 baseline | `lr=3e-4, r32/a64, dropout0, wd0` | 10/128 = 7.8% | 1600 | 当前 128 最好 |
| 256 total | `lr=3e-4, r32/a64, dropout0, wd0` | 3/40 = 7.5% | 800 | 与 128 基本持平 |
| full le32 | `lr=3e-4, r32/a64, dropout0, wd0` | 6/64 = 9.4% | 1600/2000/2400/3200 | 略高，但无质变 |
| lr search | `3e-4` | 10/128 = 7.8% | 1600 | `2e-4`/`5e-4` 都不如 |
| rank/alpha | `r32/a64` | 10/128 = 7.8% | 1600 | 降 rank/alpha 无收益 |
| dropout | `dropout=0` | 10/128 = 7.8% | 1600 | `dropout=0.05` 降 train，valid 不升 |
| weight decay | `wd=0` | 10/128 = 7.8% | 1600 | `0.001`/`0.01` 不如 baseline |
| max_steps | `step1600` for 128 | 10/128 = 7.8% | 1600 | 延长到 3200 只增加记忆 |

最终结论：

1. 样本量从 128 增到 full le32 有轻微 valid 提升，但没有形成明显泛化跃迁。
2. 当前 LoRA SFT 小网格没有找到比 baseline 更好的泛化配置。
3. 最实用 checkpoint 配置仍是：

```text
data: data/s1K_train_answer_inner_le32.json
answer_field: final_boxed
max_length: 1024
min_answer_len: 32
max_answer_len: 32
answer_prefix: Answer:
answer_leading_newline: true
boxed_prompt: true
loss: response-only DWDSE
all_mask_ce_weight: 0
batch_size: 16
grad_accum: 2
lr: 3e-4
warmup_steps: 100
lora_r: 32
lora_alpha: 64
lora_dropout: 0
weight_decay: 0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_steps: 1600
save_freq: 400
dtype: bfloat16
offline: true
```

后续建议：

1. 不建议继续在当前 LoRA 的 `lr/rank/dropout/weight_decay/max_steps` 上做细粒度扩展，收益已经很低。
2. 如果要继续提升泛化，更值得转向训练目标、数据格式、答案表达、采样策略或评估策略，而不是继续扩大这组小超参。
3. 如果要做报告或 checkpoint 版本，建议优先报告 `128/256/full le32` 样本规模曲线和本轮超参搜索的负结果。

### 12. 599 条压缩 reasoning 样本与新框架训练启动

时间：2026-06-02

压缩数据最终状态：

```text
base_dir: data/compression_pilot/deepseek_q220_r512_parts_v3
input: data/s1K_train_599.json
merged_train: data/compression_pilot/deepseek_q220_r512_parts_v3/merged_train.json
total_samples: 599
candidate_rows: 599
ok_rows: 599
bad_rows: 0
missing_rows: 0
failed_indices_count: 0
train_rows: 599
question_tokens_min_max_avg: 14 219 79.7
reasoning_tokens_min_max_avg: 105 512 310.2
error_counts: {}
```

最后 13 条边角样本未继续依赖 API，改为按原题数学逻辑手工压缩并写入：

```text
manual_pass_1: 120,135,142,159,162,171,218,250,341,429,439,521,564
```

正式训练已在云端启动，使用完整 599 条新压缩样本：

```text
run_name: compressed_reasoning_599_dwdseonly_seed10_r32a64_b8ga4_step1600
out_dir: /root/autodl-tmp/sedd_outputs/lora_compressed_reasoning_599_dwdseonly_seed10_r32a64_b8ga4_step1600
log: /root/sedd/logs/compressed_reasoning/nohup_compressed_reasoning_599_dwdseonly_seed10_r32a64_b8ga4_step1600.log
pid_file: /root/sedd/logs/compressed_reasoning/compressed_reasoning_599_dwdseonly_seed10_r32a64_b8ga4_step1600.pid
```

关键训练参数：

```text
data_json: data/compression_pilot/deepseek_q220_r512_parts_v3/merged_train.json
answer_field: solution
max_length: 1024
max_answer_len: 512
boxed_prompt: false
answer_leading_newline: true
loss: response-only DWDSE
supervise_answer_window_eos: false
all_mask_ce_weight: 0
batch_size: 8
grad_accum: 4
lr: 3e-4
warmup_steps: 50
lora_r: 32
lora_alpha: 64
lora_dropout: 0
lora_targets: attn_qkv,attn_out,mlp.0,mlp.2
max_steps: 1600
save_freq: 400
```

启动后日志确认：

```text
epoch=0 step=10 loss=3.6876 dwdse=3.6876 all_mask_ce=0.0000
epoch=3 step=70 loss=2.9856 dwdse=2.9856 all_mask_ce=0.0000
```
