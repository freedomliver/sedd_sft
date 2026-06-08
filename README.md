# SEDD-SFT 数学推理探索

本仓库基于 [Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834) 的开源实现，围绕“能否把离散扩散语言模型 SEDD 通过 SFT / LoRA / RL 训练成具备简单数学推理能力的模型”做了一轮快速研究和工程验证。

当前结论比较明确：**fixed-block SFT 可以让模型在训练集上过拟合，并稳定输出 question / reasoning / answer 格式；但模型还没有形成可靠的数学推理泛化能力。** 后续需要把 reward 从 final answer 扩展到可验证的中间推理步骤。

## 项目目标

探索 SEDD 这类离散扩散语言模型在数学推理任务上的训练方式：

- 适配数学题数据格式。
- 尝试全参数训练、LoRA-SFT、固定区间 answer 评估。
- 引入 reasoning 过程，让模型不仅输出答案，也输出推理步骤。
- 参考 DeepSeek-R1 的思路，把扩散采样过程看成序列决策过程，尝试 RL / GRPO 提升推理路径。

## 5 天工作总览

### Day 1-2：阅读论文、全参数训练和数据格式适配

前两天主要做三件事：

- 阅读 SEDD 论文和代码结构，理解 DWDSE / score entropy / absorbing graph 等训练逻辑。
- 尝试把数学问答样本适配到原始训练框架中。
- 尝试全参数训练，希望模型能直接拟合小规模数学样本。

结果：

- 全参数训练没有实现稳定过拟合。
- 生成语言序列紊乱，格式不稳定，常出现重复、断裂、无意义符号。
- 这一路径在当前数据规模和算力条件下不可行。

结论：

> 亲身验证：直接全参数调整 SEDD 做数学推理 SFT，不是当前最有效路径。

### Day 3：转向 LoRA-SFT，先解决过拟合

第三天开始尝试 LoRA 做监督微调：

- 冻结 base SEDD，只训练 LoRA adapter。
- 将样本拆成“没有正确答案”和“有正确答案”的子集。
- 优先在有正确答案的样本上做过拟合测试。
- 实现 response-only / fixed-block 的 DWDSE 训练和评估。

结果：

- 在有正确答案的小样本上基本实现了训练集过拟合。
- 但 valid/test 泛化能力很差。
- 模型更像是在记忆训练集，而不是学会数学推理。

关键观察：

- 训练集 sample oracle 可以很高。
- valid/test argmax 仍然很低。
- `oracle@K` 明显高于 argmax，说明采样分布中偶尔存在正确路径，但模型不能稳定选中。

### Day 4：简单数学样本 + reasoning 监督

第四天沿着 LoRA-SFT 路线继续推进：

- 使用自己生成的简单数学样本做监督训练。
- 尝试把 reasoning 过程加入训练。
- 将样本拆为固定 block：
  - question block
  - reasoning block
  - answer block
- 推理评估时不再搜索 `\boxed{}`，而是直接从固定 answer block 提取答案。

结果：

- fixed-block 格式可以稳定训练。
- 模型可以生成比较稳定的 Method / Steps / Answer 外形。
- 训练集仍然可以过拟合。
- 但泛化问题依旧存在。

一个重要诊断：

- 如果 clamp 住 gold reasoning，只让模型生成 answer，valid/test 可以达到 100%。
- 说明 answer block 不是瓶颈。
- 真正瓶颈是 reasoning 过程中的数学计算。

结论：

> 模型已经能学会“输出格式”，但还没有学会“可靠计算”。

### Day 5：尝试 RL / GRPO 推理路径

第五天参考 DeepSeek-R1 的方向，尝试把扩散模型的 denoising path 看成序列决策过程：

- 每道题采样多条 trajectory。
- 使用 verifier reward 判断答案是否正确。
- 使用 group-relative advantage / GRPO 更新模型。
- 只在同一问题的多条采样结果中 reward 有差异时反传。
- 使用 logprob microbatch replay 控制显存。
- 尝试 hard subset，让 batch 内更容易出现正确/错误混合，从而提供 RL 信号。

结果：

- RL / GRPO 工程链路已经跑通。
- 可以稳定采样、打分、计算 advantage、反传。
- hard subset 明显提高了训练效率，减少 reward 全一样导致跳过 update 的情况。
- 但目前 valid/test 泛化提升不明显，甚至部分 K16 oracle 下降。

当前生成状态：

- 已经会生成 `Method` / `Steps` 的外形。
- answer block 基本都有 EOS，不是早期那种完全不收尾。
- 但数学值大量错误，reasoning 内部也常常自相矛盾。
- 有些样本仍出现格式退化，例如 `\frac{11}{}{}}`、`iv`、重复词、断裂公式。

## 典型生成样例

### 答案碰巧正确，但 reasoning 错误

```text
Q: Let f(x)=2(x-5)^2+1670. Find f(5).
gold: 1670
pred: 1670

reasoning:
Method: Substitute into vertex form.
1. $x-5=1$ when $x=5$.
2. $f(5)=2\cdot(1)^2+1670=1670$
```

这里 final answer 正确，但 reasoning 错了：当 `x=5` 时，`x-5=0`，不是 1。答案只是碰巧对。

另一个类似例子：

```text
Q: Let f(x)=1(x-18)^2+1586. Find f(18).
gold: 1586
pred: 1586

reasoning:
Method: Substitute into vertex form.
1. $x-18=-1$ when $x=18$.
2. $f(18)=1\cdot(-1)^2+1586=1586$
```

这里 `x=18` 时 `x-18=0`，reasoning 同样错误。

### 失败样本

```text
Q: A right triangle has legs 114 and 152. Find its perimeter.
gold: 456
pred: 480

reasoning:
Method: Use the Pythagorean theorem.
1. The hypotenuse is sqrt(114^2+152^2)=252.
2. Perimeter is 114+152+216=480
```

问题：

- 正确斜边应为 190。
- reasoning 中第一步写 252，第二步又用 216。
- 推理内部自相矛盾。

```text
Q: x ≡ 8 mod 19, x ≡ 4 mod 23. Find x+2.
gold: 29
pred: 126

reasoning:
Method: Use the Chinese remainder theorem.
1. The moduli are coprime, so one class exists modulo 437.
2. The least positive representative satisfying both congruences is 8.
3. Therefore x+2=126
```

问题：

- 第 2 步和第 3 步不一致。
- 如果 `x=8`，则 `x+2=10`，不可能是 126。

```text
Q: Two balls drawn without replacement, probability both red.
gold: \frac{925}{4082}
pred: \frac{11}{}{}}

reasoning:
Method: Multiply conditional probabilities.
1. First red probability is 75/126.
2. Then red probability is 74/130.
3. The product reduces to \frac{11}{}{}}
```

问题：

- 条件概率分母错误。
- answer 格式退化。

## 关键实验结论

### 1. 全参数训练路径失败

直接全参数训练无法稳定拟合小样本，语言序列容易紊乱。当前更合理的路线是 LoRA-SFT。

### 2. LoRA-SFT 可以过拟合，但不能泛化

在 simple400 / manual900 等数据上，训练集可以达到很高 oracle，但 valid/test 仍然较低。这说明模型主要学到训练集模式，而不是抽象数学规则。

### 3. answer block 不是主要瓶颈

gold reasoning clamp 后，answer 生成可以接近 100%。这说明模型能从正确 reasoning 中提取答案，问题主要出在 reasoning 生成过程。

### 4. 当前 RL reward 太稀疏

final answer reward 可以选出某些正确轨迹，但不能保证 reasoning 正确。很多轨迹是“答案对、过程错”，winner replay 会污染模型。

### 5. 需要可验证中间步骤 reward

下一步更有价值的方向不是继续堆训练步数，而是为每类数学题构造 verifier：

- 一元一次方程：移项、除法、代回检查。
- 概率：total、favorable、条件概率、约分。
- 组合数：`n`、`k`、组合公式和结果。
- 几何：公式选择、中间平方、根号、周长/面积。
- 同余/CRT：模数互素性、候选解验证、最终偏移。

## 主要结果表

| 实验 | train | valid | test | 结论 |
|---|---:|---:|---:|---|
| simple400 LoRA-SFT argmax | - | 8/40 | 5/40 | 泛化弱 |
| simple400 train sampleK8 | oracle 320/320 | - | - | 训练集可过拟合 |
| simple400 gold reasoning clamp | - | 40/40 | 40/40 | answer 不是瓶颈 |
| manual900 SFT base13k argmax | - | 14/90 | 15/90 | 泛化弱 |
| manual900 SFT base13k K16 | train oracle 约 713/720 | 33/90 | 31/90 | 采样中有正确路径 |
| dense GRPO q4K16 | - | 34/90 | 32/90 | 只有噪声级提升 |
| hard subset dense GRPO | mixed update 100/100 | 32/90 | 27/90 | 训练信号强，但泛化未提升 |

## 当前代码内容

新增/重点文件：

- `data_sft.py`：SFT 数据格式化、fixed-block 样本构造。
- `train_sft.py`：LoRA-SFT 训练。
- `losses.py`：response-only / fixed-block DWDSE 训练目标。
- `lora.py`：LoRA adapter 注入。
- `eval_fixed_blocks.py`：固定 answer block 评估。
- `eval_fixed_blocks_oracle_passk.py`：sampleK / oracle pass@K 评估。
- `train_fixed_blocks_reinforce.py`：fixed-block RL / GRPO 探索。
- `eval_fixed_blocks_staged.py`：分阶段生成诊断。
- `train_fixed_blocks_repair.py`：repair 路线诊断脚本。
- `sedd_math_generalization_experiment_record.md`：完整实验记录。
- `interview_sedd_math_project_summary.md`：面试版总结。
- `rl_diffusion_math_plan.md`：RL 路线计划。

## 当前状态

这个项目目前不是一个“已经解决数学推理”的结果，而是一次较完整的失败分析和路线筛选：

- 已经证明全参数训练不可行。
- 已经证明 LoRA-SFT 可以稳定过拟合。
- 已经证明 fixed-block answer 评估更可靠。
- 已经证明 bottleneck 是 reasoning，不是 answer。
- 已经跑通 RL / GRPO 工程链路。
- 目前还没有实现可靠泛化推理。

## 下一步方向

优先级最高的是构造类型化 verifier，把 reward 从 final answer 扩展到中间推理步骤。

可行路线：

1. 从样本生成脚本中保留结构化中间变量。
2. 每类题实现独立 verifier。
3. 对 reasoning block 做中间步骤 reward。
4. 在 RL 中混合 hard subset 和 random train，避免 hard-only 分布偏移。
5. 加入 reference / KL 约束，避免 RL 后采样分布退化。
6. 长期尝试把 denoising path 当时间序列，用 value model / TD learning 做 credit assignment。

## 原始项目

本仓库基于 SEDD 官方实现：

```bibtex
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

原始实现致谢：

- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
- [plaid](https://github.com/igul222/plaid)
- [DiT](https://github.com/facebookresearch/DiT)
