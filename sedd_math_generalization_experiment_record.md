# SEDD 数学泛化实验记录

更新时间：2026-06-04

本文记录当前 SEDD fixed-block 数学训练/强化学习路线的主要实验、结果和结论。重点是区分“训练集过拟合是否成立”和“valid/test 泛化是否提升”。

## 1. 当前固定格式

### simple400 短窗口

- 模型：`pretrained/sedd-small`
- 数据：`/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_400_simple_seed10_train320_valid40_test40.json`
- 固定块：
  - `max_length=160`
  - question: `[0,48)`
  - reasoning: `[48,144)`
  - answer: `[144,160)`
- 关键 checkpoint：
  - `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple400_small_shortwin_q48_r96_a16_fa4pad025_seed10_train320_valid40_test40_r32a64_b32ga4_step10000/checkpoint_step_6500.pt`

### manual900 中等窗口

- 模型：`pretrained/sedd-small`
- 数据：`/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_900_simple400_medium500_seed10_train720_valid90_test90.json`
- 固定块：
  - `max_length=256`
  - question: `[0,48)`
  - reasoning: `[64,192)`，长度 128
  - answer: `[213,229)`，长度 16
- 关键 checkpoint：
  - `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_manual900_pos256_q48_rs64_r128_as213_a16_activeonly_freshmid5_fa8pad025_seed10_train720_valid90_test90_r32a64_b160ga1_resume3k_to15k/checkpoint_step_13000.pt`

## 2. 重要基线结果

### 2.1 simple400 SFT 过拟合

| 模型/评估 | train | valid | test | 结论 |
|---|---:|---:|---:|---|
| step6500 argmax64 | 未完整记录 | 8/40 = 20.0% | 5/40 = 12.5% | 训练集外泛化弱 |
| step6500 train sampleK8 | pass@1 283/320 = 88.4%; oracle@8 320/320 = 100% | - | - | train 基本过拟合 |
| step6500 sampleK32 | - | 15/40 = 37.5% | 14/40 = 35.0% | 采样 oracle 明显高于 argmax |

结论：

- 当前 fixed-block answer 评估可以正常工作。
- 小样本 train 能过拟合，但 valid/test 泛化很差。
- `oracle@K` 明显高于 argmax，说明模型分布里有正确路径，但默认采样/argmax 不能稳定取到。

### 2.2 manual900 SFT baseline

| 模型 | valid argmax | test argmax | valid K16 oracle | test K16 oracle | 结论 |
|---|---:|---:|---:|---:|---|
| base13k | 14/90 = 15.6% | 15/90 = 16.7% | 33/90 = 36.7% | 31/90 = 34.4% | 当前 900 的主要基线 |

补充：

- train720 K16 采样显示 train 端几乎已经记住：
  - oracle 约 713/720
  - 全 16 个 sample 都正确：102 题
  - 0 个正确 sample：7 题
  - mixed：611 题
- 这说明泛化问题不是“train 学不动”，而是 train/test 分布外的计算能力没有形成。

## 3. 训练框架与 t 分布实验

### 3.1 多 t：`num_t_per_sample=4`

目的：同一 batch 内对每个样本取多个 t，降低 t 采样方差。

关键观察：

- 相同步数下，`t_num=4` 约等于 4 倍 forward/backward 计算量。
- 按“同 step”看不公平；按“相同 forward 预算”看，效果不如 base 方式。

结论：

- `num_t_per_sample=4` 不是优先路线。
- 它提高了计算量，但没有带来足够收益。

### 3.2 fresh fixed t-grid / mid5

目的：控制 t 分布，而不是每次完全随机采样。

术语说明：

- `fresh fixed t-grid per batch`：每个 batch 仍然新鲜加噪，不缓存 noisy 样本；但 t 从固定 grid/分桶中取，控制 t 覆盖。
- `answer 从固定 answer block 提取`：不是从生成文本里搜索 `\boxed{}`，而是直接读取固定 answer 区间的 token，和 gold answer 对比。

观察：

- `fresh mid5` 比复杂 replay buffer 更稳，且每 step 仍基本是单 t 采样。
- t 控制对训练稳定性有帮助。

结论：

- 固化 SFT 方式时，优先使用 `fresh mid5` 一类的固定 t 分布。
- replay buffer 的主要价值不是“省一点加噪计算”，而是可控 t 分布；但工程复杂度更高。

## 4. fixed-block SFT 权重实验

### 4.1 active-only vs full-window

问题：只在 q/r/a active 区域训练，还是全窗口包括 pad/空白位置一起训练？

观察：

- `active-only` 没有明显优于全窗口。
- 全窗口训练在部分设置中反而更容易复现已有结果。

结论：

- 空白/固定区域不能简单认为“没有训练价值”；它可能帮助模型稳定固定布局。

### 4.2 answer/reasoning/pad 权重

实验中使用过类似：

- `fa4/pad0.25`
- `fa8/pad0.25`

其中：

- `fa` 是 final answer 区域权重。
- `pad` 是非主要监督区域/空白固定区域的低权重约束，不是 padding token 本身的 CE。

结论：

- `fa8/pad0.25` 是 900 SFT 使用过的较好参数之一。
- 但即使 answer 权重提高，valid/test 仍然平台化，说明单纯加权不能解决泛化。

## 5. 生成/推理诊断

### 5.1 gold reasoning clamp

实验：

- 固定 question。
- 将 gold reasoning clamp 住。
- 只生成 answer block。

结果：

- simple400 valid: 40/40
- simple400 test: 40/40

结论：

- answer block 本身不是主要瓶颈。
- 主要瓶颈是 reasoning 生成质量；只要 reasoning 正确，answer 能被模型稳定输出。

### 5.2 staged decode

实验：

- 分阶段生成 reasoning 和 answer。

结果：

- 没有明显提升。

结论：

- 简单拆分采样阶段不能解决 reasoning 的计算错误。

### 5.3 base model 输出

观察：

- 原始 small/medium base 在这些固定数学任务上基本不能直接正确作答。
- small base 输出常见问题：
  - 数字混乱
  - 公式格式破损
  - reasoning 模板像样但计算错误

结论：

- 当前能力主要来自 SFT 记忆/局部模式，不是 base SEDD 已有数学推理能力。

## 6. RL / GRPO 路线

### 6.1 RL 脚本状态

脚本：`train_fixed_blocks_reinforce.py`

已经实现：

- 从 SFT LoRA checkpoint 继续训练。
- 对每道题采样多条 trajectory。
- verifier reward：
  - exact answer reward
  - dense reward：answer + trace/formula/format 等弱信号
- group-relative advantage。
- 只对 mixed reward group 做 backward。
- logprob replay microbatch，避免 q4K16 显存爆炸。
- 日志字段：
  - `reward_std`
  - `exact_rate`
  - `adv_abs`
  - `mixed_groups`
  - `selected_traj`

注意：

- 不使用 CE loss。
- 之前 repair CE 类实验仅用于诊断，不作为 diffusion 训练主路线。

### 6.2 simple400 RL

| 实验 | 结果 | 结论 |
|---|---|---|
| 从 simple400 step6500 做 sparse/dense REINFORCE | valid/test 没有稳定提升 | 训练集太容易，全对/全错组太多，relative advantage 信号弱 |
| hard subset/dense RL | 外部指标没有明显改善 | 单纯 RL 更新不足以产生泛化推理 |

### 6.3 manual900 全 train720 GRPO

| 实验 | valid argmax | test argmax | valid K16 | test K16 | 结论 |
|---|---:|---:|---:|---:|---|
| base13k | 14/90 | 15/90 | 33/90 | 31/90 | 基线 |
| q4K8 trace2 mb4 lr5e-6 u60 | 15/90 | 15/90 | 33/90 | 33/90 | 小幅/噪声级变化 |
| q4K16 trace2 mb4 lr1e-5 u40 | 15/90 | 15/90 | 33/90 | 31/90 | 基本无变化 |
| dense q4K16 trace2 mb4 lr1e-5 u40 | 15/90 | 15/90 | 34/90 | 32/90 | 当前全 train RL 中最好的轻微提升 |

结论：

- q4K16 已能稳定运行，显存约 20GB。
- dense reward 能产生更稳定训练信号。
- 但短程 RL 对 valid/test 改善很小，只有噪声级别。

## 7. winner replay 路线

### 7.1 simple400 winner replay

数据：

- `/root/sedd/data/simple_manual_synthetic/simple400_step6500_train320_winner_replay_k8.json`

模型：

- `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_simple400_winnerreplay_k8_from6500_lr1e4_to7500/lora_final.pt`

结果：

- valid argmax: 8/40
- test argmax: 6/40
- valid oracle@32: 19/40
- test oracle@32: 15/40

结论：

- oracle 略有提升。
- argmax 没有明显提升。

### 7.2 manual900 mixed winner replay

从 base13k 对 train720 采样 K16：

- mixed 题：611
- 构造 one-winner replay 数据：
  - `/root/sedd/data/simple_manual_synthetic/manual900_base13k_train720_mixed_winner_replay_k16_one.json`

训练：

- `/root/autodl-tmp/sedd_outputs/lora_manual900_winnerreplay_mixed611_from13k_lr1e4_to14000_v2/lora_final.pt`

结果：

- argmax valid: 16/90
- argmax test: 15/90
- K16 valid: 31/90
- K16 test: 28/90

结论：

- argmax valid 略升，但 sample 分布变差。
- 自生成 winner reasoning 经常“答案对但推理坏”，会污染 reasoning 训练。
- winner replay 不能直接当作高质量 SFT 数据。

## 8. trace 改写路线

### 8.1 canonical symbolic trace

数据：

- `/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_900_canonical_expr_trace_seed10_train720_valid90_test90.json`

格式：

- 类似 `T:... E:... F:...`

统计：

- reasoning mean 40.6 tokens
- p95 66 tokens
- 原始 reasoning mean 72.4, p95 104

训练：

- `/root/autodl-tmp/sedd_outputs/lora_manual900_canonical_expr_from13k_lr1e4_to14000/lora_final.pt`

结果：

- argmax valid: 10/90
- argmax test: 13/90
- K16 valid: 30/90
- K16 test: 29/90

现象：

- 训练 loss 能降到约 0.20。
- 生成退化为 `;;;;`、`FFF`、`cdcdcdot` 一类符号串。

结论：

- 过度符号化不适合 GPT2 tokenizer + 当前 SEDD 采样。

### 8.2 natural short trace v1

数据：

- `/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_900_natural_trace_v1_seed10_train720_valid90_test90.json`

格式：

```text
Method: <topic>.
Steps:
1. Identify the requested quantity.
2. Apply the standard formula or operation for this problem type.
3. The computed final value is <answer>.
```

统计：

- reasoning max 52 tokens
- p95 46 tokens
- mean 43.2 tokens

训练：

- `/root/autodl-tmp/sedd_outputs/lora_manual900_natural_trace_v1_from13k_lr1e4_to14000/lora_final.pt`

结果：

- argmax valid: 12/90
- argmax test: 13/90
- K16 valid: 31/90
- K16 test: 32/90

现象：

- loss 降到约 0.073。
- 生成学会了语言 shell，但没有学会计算。

结论：

- 泛化不能靠更短、更模板化的 natural trace 自动得到。
- 如果 trace 太通用，模型会学习“像推理的壳”，而不是可执行计算。

## 9. 最新实验：manual900 hard subset dense GRPO

### 9.1 动机

全 train720 中很多题对 winner 模型来说已经全对或全错，导致同一 group 内 reward 全一样，脚本会跳过更新。为提高 RL 训练效率，抽取 base13k 在 train K16 中只有少量正确样本的题。

### 9.2 hard subset 构造

来源：

- 原始数据：
  - `/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_900_simple400_medium500_seed10_train720_valid90_test90.json`
- base13k train K16 采样文件：
  - `/root/sedd/outputs/rl_oracle/manual900_base13k_train240_sampleK16_for_winner_replay.jsonl`
  - `/root/sedd/outputs/rl_oracle/manual900_base13k_train240_240_sampleK16_for_winner_replay.jsonl`
  - `/root/sedd/outputs/rl_oracle/manual900_base13k_train480_240_sampleK16_for_winner_replay.jsonl`

规则：

- 取 `1 <= correct_count <= 8` 的 train 题。
- 共 164 题。

correct_count 分布：

| correct_count | 题数 |
|---:|---:|
| 1 | 15 |
| 2 | 19 |
| 3 | 23 |
| 4 | 25 |
| 5 | 20 |
| 6 | 19 |
| 7 | 21 |
| 8 | 22 |

保存：

- `/root/sedd/data/simple_manual_synthetic/manual900_train720_hard_cc1_8_from_base13k_k16.json`

### 9.3 训练配置

输出目录：

- `/root/autodl-tmp/sedd_outputs/rl_fixed_blocks_manual900_hardcc1_8_dense_from13k_u100_lr1e5_q4k16_trace2_mb4`

关键参数：

- resume:
  - manual900 base13k checkpoint
- `question_batch_size=4`
- `num_samples=16`
- `trace_steps=2`
- `logprob_microbatch_size=4`
- `max_updates=100`
- `lr=1e-5`
- `reward_mode=dense`
- `answer_reward=1.0`
- `trace_reward=0.5`
- `format_reward=0.05`
- `dtype=bfloat16`

训练日志摘要：

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| reward_mean | 0.7432 | 0.413 | 1.018 |
| reward_std | 0.4979 | 0.277 | 0.578 |
| exact_rate | 0.3105 | 0.062 | 0.562 |
| adv_abs | 0.3870 | 0.130 | 0.535 |

其他：

- 100/100 updates 都是 `mixed_groups=4/4`
- 100/100 updates 都是 `selected_traj=64/64`
- 显存约 20.2GB
- 训练用时约 16.3 分钟

结论：

- hard subset 明显解决了“reward 全一样导致跳过更新”的训练效率问题。
- 但是外部 valid/test 没有明显改善。

### 9.4 评估结果

与 base13k 对比：

| 模型 | valid argmax | test argmax | valid K16 oracle | test K16 oracle | 结论 |
|---|---:|---:|---:|---:|---|
| base13k | 14/90 | 15/90 | 33/90 | 31/90 | 原始 900 SFT 基线 |
| hard dense GRPO update50 | 15/90 | 15/90 | 32/90 | 28/90 | argmax valid +1，但 K16 下降 |
| hard dense GRPO update100/final | 15/90 | 15/90 | 32/90 | 27/90 | test K16 继续下降 |

生成现象：

- 正确样本仍多是模型已能套对的短模式。
- 错误样本 reasoning 表面结构完整，但计算步骤常错：
  - 斜率/比例数字错
  - 组合数公式项错
  - 概率分母/有利情况错
  - 中点/LCM/判别式等公式形式接近但代入错误

本轮结论：

- hard-subset RL 让训练信号变强，但没有提升泛化。
- 只在 hard train 子集上做 dense GRPO，可能把分布推向 train-hard 的局部模式，反而压缩 valid/test 采样多样性。
- 当前 reward 更像“选出已存在的正确样本”，还不能把错误 reasoning 修成真正可泛化的推理路径。

## 10. 当前总体结论

### 已确认

1. fixed-block answer 提取逻辑可用，不需要搜索 `\boxed{}`。
2. 小样本/训练集过拟合成立。
3. answer block 本身不是瓶颈；gold reasoning clamp 后 answer 可 100%。
4. bottleneck 是 reasoning 的计算可靠性。
5. RL/GRPO 工程链路已经跑通：
   - 多轨迹采样
   - verifier reward
   - relative advantage
   - microbatch logprob replay
   - hard subset sampling
6. dense reward 和 hard subset 能显著提升训练更新效率。

### 仍未解决

1. valid/test 泛化没有实质提升。
2. 自生成正确答案轨迹的 reasoning 质量不可靠。
3. symbolic trace 和 generic natural trace 都会学坏：
   - 一个变成符号退化
   - 一个变成模板壳
4. 当前 reward 对“中间推理是否数学正确”的约束太弱。
5. group-relative RL 更擅长放大已有正确路径，不擅长创造新的可泛化计算能力。

## 11. 下一步可能方向

### 方向 A：改 reward，做可验证中间步骤

优先级最高。

思路：

- 从样本生成程序中抽出每类题的 symbolic verifier。
- 不只验证 final answer，而是验证中间变量：
  - linear equation: 移项值、除法值、check
  - probability: total、favorable、fraction
  - combination: n、k、binomial value
  - geometry: formula、代入项、中间平方/根号
- reward 改成 per-type dense reward。

预期：

- 让 reward 真正约束 reasoning 计算，而不是只看最后答案。

### 方向 B：把采样路径当时间序列，做 step-level credit assignment

思路：

- 对同一问题采样多条 denoising path。
- 不只看最终文本，而是记录不同 t/time 的中间 token 状态。
- 用 verifier reward 训练能走向正确答案的路径。
- 后续可加入 value model / TD，把稀疏 final reward 变成稠密时间序列 reward。

风险：

- 工程量明显大。
- 当前只有 final verifier，credit assignment 仍弱。

### 方向 C：hard + mixed curriculum，而不是只训 hard

本轮 hard-only 使信号变强但 K16 下降。下一步可以：

- batch 中混合：
  - easy/all-train 题保持分布
  - hard-mixed 题提供强 RL 信号
- 例如每个 RL batch：
  - 2 hard questions
  - 2 random train questions
- 避免 policy 只朝 hard subset 局部偏移。

### 方向 D：加入 KL/reference 约束

当前 GRPO 没有显式 KL 约束。可以考虑：

- 以 base13k 为 reference。
- reward 优化同时限制偏离。
- 目标是保留原采样分布中的 oracle 能力，避免 K16 下降。

### 方向 E：更结构化的数据，而不是更短模板

已有结果显示：

- 太符号化会退化。
- 太模板化会学壳。

更可行的是“自然语言 + 明确可验证槽位”：

```text
Method: ...
Given: ...
Compute: ...
Intermediate: ...
Answer: ...
```

每个槽位都能被 verifier 检查，而不是泛泛模板。

## 12. 当前最佳 checkpoint/数据路径

### 数据

- manual900:
  - `/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_900_simple400_medium500_seed10_train720_valid90_test90.json`
- hard cc1-8:
  - `/root/sedd/data/simple_manual_synthetic/manual900_train720_hard_cc1_8_from_base13k_k16.json`

### SFT baseline

- manual900 base13k:
  - `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_manual900_pos256_q48_rs64_r128_as213_a16_activeonly_freshmid5_fa8pad025_seed10_train720_valid90_test90_r32a64_b160ga1_resume3k_to15k/checkpoint_step_13000.pt`

### 当前 best-ish RL

- 全 train dense GRPO u40：
  - `/root/autodl-tmp/sedd_outputs/rl_fixed_blocks_manual900_grpo_dense_from13k_u40_lr1e5_q4k16_trace2_mb4/lora_final.pt`
  - valid/test K16: 34/90, 32/90
- hard cc1-8 dense GRPO u100：
  - `/root/autodl-tmp/sedd_outputs/rl_fixed_blocks_manual900_hardcc1_8_dense_from13k_u100_lr1e5_q4k16_trace2_mb4/lora_final.pt`
  - valid/test K16: 32/90, 27/90

当前如果要继续做泛化实验，建议从 manual900 base13k 或全 train dense GRPO u40 出发，不建议从 hard-only u100 出发。
