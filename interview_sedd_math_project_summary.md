# SEDD 数学推理项目面试总结

日期：2026-06-08

## 1. 30 秒项目概述

这个项目尝试把 SEDD 这类离散扩散语言模型用于简单数学题的推理生成。核心问题不是让模型只输出 final answer，而是让模型在固定的 question / reasoning / answer 区间内生成可验证的推理过程，并观察它是否能从训练样本过拟合进一步发展出 valid/test 泛化能力。

目前结论是：fixed-block 训练和评估链路已经跑通，模型可以在小训练集上明显过拟合；但泛化能力弱。强化学习路线也已经实现了多轨迹采样、verifier reward、group-relative advantage 和 logprob microbatch replay，但目前只带来噪声级提升，还没有真正解决数学 reasoning 泛化。

## 2. 我做了什么

### 训练格式设计

把每个样本拆成固定区间：

- question block：问题输入。
- reasoning block：中间推理。
- answer block：最终答案。

这样评估时不再依赖从文本中搜索 `\boxed{}`，而是直接从固定 answer block 提取答案进行对比。

使用过两套主要窗口：

| 数据 | max length | question | reasoning | answer |
|---|---:|---:|---:|---:|
| simple400 | 160 | 48 | 96 | 16 |
| manual900 | 256 | 48 | 128 | 16 |

### 训练优化尝试

尝试过：

- response/fixed-block DWDSE 训练。
- answer / reasoning / pad 区域加权。
- active-only 与 full-window 训练对比。
- 固定 t-grid / fresh mid5 的时间步采样控制。
- `num_t_per_sample=4` 多 t 训练。
- winner replay。
- symbolic trace / natural trace 改写。
- 基于 verifier reward 的 GRPO / REINFORCE。
- hard subset 采样，提高 RL batch 内 reward 方差。

### RL 工程实现

实现了一个 fixed-block trajectory RL 脚本：

- 从 SFT LoRA checkpoint 继续训练。
- 每道题采样 K 条 trajectory。
- 根据 final answer 和弱 reasoning signal 计算 reward。
- 使用 group-relative advantage。
- 只在 group 内 reward 有差异时反传。
- 使用 logprob microbatch replay，避免 q4K16 显存爆炸。
- 记录 `reward_std`、`exact_rate`、`adv_abs`、`mixed_groups` 等训练诊断指标。

## 3. 关键结果

### 3.1 simple400：训练集能过拟合，但泛化弱

| 设置 | train | valid | test |
|---|---:|---:|---:|
| SFT step6500 argmax | - | 8/40 = 20.0% | 5/40 = 12.5% |
| SFT step6500 train sampleK8 | pass@1 283/320 = 88.4%; oracle@8 320/320 = 100% | - | - |
| SFT step6500 sampleK32 | - | 15/40 = 37.5% | 14/40 = 35.0% |

结论：

- 模型能记住训练集。
- 采样 oracle 明显高于 argmax，说明模型分布里存在正确路径。
- 但 valid/test argmax 很低，泛化推理没有形成。

### 3.2 gold reasoning clamp：证明瓶颈在 reasoning

实验方式：

- 固定 question。
- 把 gold reasoning clamp 住。
- 只让模型生成 answer block。

结果：

| 设置 | valid | test |
|---|---:|---:|
| gold reasoning + generated answer | 40/40 | 40/40 |

结论：

- answer block 生成不是主要瓶颈。
- 如果 reasoning 正确，模型可以稳定给出正确 answer。
- 真正问题是模型生成的 reasoning 数字和中间步骤经常错误。

### 3.3 manual900：训练集几乎记住，valid/test 仍低

manual900 使用 720/90/90 划分。

| 模型 | valid argmax | test argmax | valid K16 oracle | test K16 oracle |
|---|---:|---:|---:|---:|
| SFT base13k | 14/90 = 15.6% | 15/90 = 16.7% | 33/90 = 36.7% | 31/90 = 34.4% |

train720 的 K16 采样：

- oracle 约 713/720。
- 102 题 16 个 sample 全对。
- 611 题是 mixed group。
- 只有 7 题完全没有正确 sample。

结论：

- 训练集并不是学不动，而是强烈过拟合。
- valid/test 泛化仍然很弱。

### 3.4 全 train720 GRPO：只有噪声级提升

| 实验 | valid argmax | test argmax | valid K16 | test K16 |
|---|---:|---:|---:|---:|
| SFT base13k | 14/90 | 15/90 | 33/90 | 31/90 |
| q4K8 trace2 mb4 lr5e-6 u60 | 15/90 | 15/90 | 33/90 | 33/90 |
| q4K16 trace2 mb4 lr1e-5 u40 | 15/90 | 15/90 | 33/90 | 31/90 |
| dense q4K16 trace2 mb4 lr1e-5 u40 | 15/90 | 15/90 | 34/90 | 32/90 |

结论：

- RL 链路能稳定运行。
- dense reward 有一点帮助，但提升很小。
- 当前 reward 还不足以让模型学出可泛化计算能力。

### 3.5 hard subset GRPO：训练信号变强，但泛化没提升

构造方式：

- 从 manual900 train720 中选 base13k K16 采样正确数为 1 到 8 的题。
- 共 164 题。
- 这些题最适合 relative advantage，因为同一个 group 内既有对也有错。

训练配置：

- q4K16。
- dense reward。
- 100 updates。
- lr=1e-5。
- 显存约 20.2GB。

训练诊断：

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| reward_mean | 0.7432 | 0.413 | 1.018 |
| reward_std | 0.4979 | 0.277 | 0.578 |
| exact_rate | 0.3105 | 0.062 | 0.562 |
| adv_abs | 0.3870 | 0.130 | 0.535 |

100/100 updates 都是 `mixed_groups=4/4`，说明 hard subset 有效解决了 reward 全一样导致跳过更新的问题。

但评估结果：

| 模型 | valid argmax | test argmax | valid K16 | test K16 |
|---|---:|---:|---:|---:|
| SFT base13k | 14/90 | 15/90 | 33/90 | 31/90 |
| hard dense GRPO update50 | 15/90 | 15/90 | 32/90 | 28/90 |
| hard dense GRPO update100 | 15/90 | 15/90 | 32/90 | 27/90 |

结论：

- hard subset 提高了训练效率。
- 但 hard-only RL 没有提升泛化，甚至降低了 K16 oracle。
- 说明当前 RL 更像是在放大已有局部正确路径，而不是产生新的推理能力。

## 4. 失败实验与排除结论

### 多 t 训练

`num_t_per_sample=4` 能降低单 batch 内 t 采样方差，但相同步数下计算量约 4 倍。按相同计算预算看，没有明显优于 baseline。

结论：不是当前优先路线。

### winner replay

把模型采样中 answer 正确的轨迹拿回来继续 SFT。

结果：

- simple400 oracle 略有提升，但 argmax 没明显提升。
- manual900 argmax valid 略升，但 K16 分布变差。

问题：

- 自生成 winner 里有大量“答案对但 reasoning 坏”的样本。
- 继续 SFT 会污染 reasoning。

结论：不能直接把 self-generated winner 当高质量推理数据。

### trace 改写

做过两类：

- symbolic canonical trace。
- natural short trace。

结果：

| 数据改写 | valid argmax | test argmax | valid K16 | test K16 | 现象 |
|---|---:|---:|---:|---:|---|
| symbolic trace | 10/90 | 13/90 | 30/90 | 29/90 | 符号串退化 |
| natural short trace | 12/90 | 13/90 | 31/90 | 32/90 | 学到模板壳，不会计算 |

结论：

- trace 不能只是变短。
- 太符号化会破坏 tokenizer/采样。
- 太模板化会让模型学语言形式，而不是计算过程。

## 5. 当前核心判断

### 已经解决的部分

- fixed-block 数据格式和评估方式可用。
- 小样本训练可以过拟合。
- answer block 直接评估比搜索 `\boxed{}` 更稳定。
- SFT、采样评估、oracle pass@K、RL 训练链路都已经跑通。
- 多轨迹 RL 的显存问题已经通过 microbatch replay 解决。

### 尚未解决的部分

- valid/test 泛化没有实质提升。
- reasoning 里的中间计算经常错。
- 当前 reward 对中间步骤约束不够。
- group-relative RL 能选择已有正确路径，但不能有效创造新的正确推理路径。

一句话总结：

> 这个项目证明了 SEDD 可以被 fixed-block SFT 到训练集过拟合，也证明了 answer 生成不是瓶颈；真正瓶颈是 reasoning 过程的可验证计算能力。目前 RL 工程链路可运行，但 final-answer reward 太稀疏，尚不能稳定提升泛化。


## 6. 展示的关键数字

| 结论 | 数字 |
|---|---:|
| simple400 train sampleK8 oracle | 320/320 = 100% |
| simple400 valid/test argmax | 8/40, 5/40 |
| gold reasoning clamp valid/test | 40/40, 40/40 |
| manual900 base13k valid/test argmax | 14/90, 15/90 |
| manual900 base13k valid/test K16 | 33/90, 31/90 |
| manual900 train720 K16 oracle | 约 713/720 |
| best dense GRPO valid/test K16 | 34/90, 32/90 |
| hard subset GRPO mixed updates | 100/100 |
| hard subset GRPO final valid/test K16 | 32/90, 27/90 |

## 7. 当前文件和结果路径

本地完整实验记录：

- `sedd_math_generalization_experiment_record.md`

本地面试总结：

- `interview_sedd_math_project_summary.md`

远程关键数据：

- `/root/sedd/data/simple_manual_synthetic/s1K_manual_synthetic_900_simple400_medium500_seed10_train720_valid90_test90.json`
- `/root/sedd/data/simple_manual_synthetic/manual900_train720_hard_cc1_8_from_base13k_k16.json`

远程关键 checkpoint：

- SFT base13k:
  - `/root/autodl-tmp/sedd_outputs/lora_fixed_blocks_manual900_pos256_q48_rs64_r128_as213_a16_activeonly_freshmid5_fa8pad025_seed10_train720_valid90_test90_r32a64_b160ga1_resume3k_to15k/checkpoint_step_13000.pt`
- hard dense GRPO:
  - `/root/autodl-tmp/sedd_outputs/rl_fixed_blocks_manual900_hardcc1_8_dense_from13k_u100_lr1e5_q4k16_trace2_mb4/lora_final.pt`
