# PerturbCoT 项目进度更新

> 答案抽取审计更新：已新增确定性的 v2 抽取器和无 gold 的盲审流程。历史
> 450 条输出中有 162 条需要复核；复核完成前，v2 准确率和监控指标仅为
> `provisional`，本文件下方的第一版数字仍是保留用于追溯的旧口径。

**更新时间：** 2026-11-21
**当前实验：** Qwen2.5-Math-1.5B，50 道 GSM8K 原题，200 个语义保持变体

## 总体状态

第一版小规模实验已完成范围定义、题目准备、四类变体生成、450 次模型推理、答案与 CoT 监控器，以及基于已有独立采样的自适应验证离线重放。干预结果仍为 provisional；正式案例集和最终项目报告尚未完成。

| 步骤 | 状态 | 当前结论 |
| --- | --- | --- |
| 1. 研究目标和范围 | 已完成 | 三个研究问题、四类答案保持扰动和 50 题试运行范围已固定 |
| 2. 题目和变体 | 试运行数据完成 | 50 原题、200 变体通过自动检查和 Codex 复核；独立人工复核待完成 |
| 3. 模型运行 | 已完成 | 450/450 次生成成功，原始输出与成本记录齐全 |
| 4. 两个监控器 | 第一版代码完成 | 答案监控有效但误报高；显式方程 CoT 监控精确但覆盖不足；人工核验待完成 |
| 5. 统一验证步骤 | 离线重放完成 | 触发后使用 r1-r4 与 r0 的五样本唯一众数；并列或不可用时保留 r0 |
| 6. 公平对照实验 | provisional 完成 | 已实现无验证、始终验证、随机预算、四扰动和顺序风险策略；dev 冻结阈值 |
| 7. 效果与案例评估 | provisional 完成 | 已生成干预准确率、净修正、token、bootstrap、顺序敏感性和成本曲线 |
| 8. 最终交付物 | 部分完成 | 自适应实验代码和图表已完成；答案盲审、正式案例集和最终报告待完成 |

## 1. 确定研究目标和第一版范围

**状态：已完成。**

已固定三个研究问题：扰动对答案与推理的影响、CoT 检查相对答案检查的增量价值，以及触发验证后的准确率与成本。第一版只使用释义改写、实体改名、格式变化和添加无关信息，不包含数值扰动。

**输出：**

- [`experiment_scope.md`](experiment_scope.md)：一页实验范围、研究问题、数据规模、监控与验证口径。

## 2. 准备题目并生成经过检查的变体

**状态：试运行数据完成，独立人工复核待完成。**

从固定版本的 GSM8K test 集中按 SHA-256 排名确定性抽取 50 道原题，划分为 10 道 dev 和 40 道 test。同一原题的全部变体保持在同一集合。共生成 200 个变体，每类 50 个。运行时文件不含标准答案。

所有变体通过结构检查和 Codex 语义复核；`independent_human_review_status` 仍为 `pending`，因此不能将当前状态描述成独立人工核验完成。

**输出：**

- [`perturbcot_data_prep/`](perturbcot_data_prep/)：原题抽取脚本、50 道原题、dev/test 划分、离线金标准和验证报告。
- [`perturbcot_variants/`](perturbcot_variants/)：变体生成代码、一致性规格、200 条配对变体、运行时无答案数据、审核记录和测试。
- [`perturbcot_variants/data/offline_eval/paired_variants.jsonl`](perturbcot_variants/data/offline_eval/paired_variants.jsonl)：完整配对数据。
- [`perturbcot_variants/data/runtime/variants.jsonl`](perturbcot_variants/data/runtime/variants.jsonl)：运行时变体。
- [`step2_summary.md`](step2_summary.md)：第二步完成说明和限制。

## 3. 运行模型并保存完整实验记录

**状态：已完成。**

使用 Qwen2.5-Math-1.5B：50 道原题各生成 5 次，200 个变体各生成 1 次，共 450 次。固定 `temperature=0.7`、`top_p=0.95`、`top_k=50`、`max_new_tokens=512` 和逐条随机种子。各版本独立运行，不共享历史回答。

450 条记录全部为 `status=ok`，无重复或缺失；449 条成功抽取答案，20 条达到长度上限并保留。运行时记录不含标准答案。总成本为 46,022 输入 token、111,725 输出 token，逐条耗时之和为 4,805.65 秒。

**输出：**

- [`perturbcot_model_run/`](perturbcot_model_run/)：可恢复、可分片的运行脚本与配置。
- [`perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl`](perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl)：450 条完整原始生成记录。
- [`perturbcot_qwen25_math_15b_results/offline_evaluation/`](perturbcot_qwen25_math_15b_results/offline_evaluation/)：逐条离线正确性、按题汇总和总体指标。
- [`perturbcot_qwen25_math_15b_results.zip`](perturbcot_qwen25_math_15b_results.zip)：第三步完整压缩包。

### 3.1 本地备份是否足以重跑

**结论：足以重新执行同一实验流程并得到可直接比较的结果。** 本地已经保存重跑所需的模型运行脚本、固定生成配置、50 道运行时原题、200 个运行时变体、随机种子生成规则、答案抽取逻辑和最终结果。标准答案继续与运行时输入隔离，只在离线评估阶段使用。

在相同模型快照、Python/torch/transformers/CUDA 版本和 GPU 类型下，固定种子应产生高度可比的结果。但本实验使用随机采样，CUDA 算子和依赖版本也可能引入非确定性，因此复现目标是相同实验配置、样本规模和统计结论，**不能承诺 450 条文本逐字符完全相同**。

ICRN 原始运行环境记录如下：

- Python `3.13.11`
- PyTorch `2.6.0+cu118`
- Transformers `5.7.0`
- CUDA `11.8`
- GPU `NVIDIA A100-SXM4-80GB MIG 2g.20gb`
- 模型 `Qwen/Qwen2.5-Math-1.5B`
- 模型快照 `/home/haod6/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`

本地备份不包含远端全部 `.pid`、失败尝试目录和临时日志。这些文件不参与实验定义，也不是重新运行所必需。最终分片清单和关键重试日志已经保存在 `merged-results/`。

### 3.2 关键文件分别做什么

#### `perturbcot_model_run/`：可以重新运行的输入与程序

| 文件 | 用途 |
| --- | --- |
| `run_qwen.py` | 主运行程序。构造固定 450 条调度，加载本地 Qwen 模型，逐条生成 CoT，抽取答案并记录 token、耗时、截断和错误。支持断点续跑与分片。 |
| `config.json` | 固定模型路径、提示词、采样参数、重复次数、期望原题数、变体数和总生成数。 |
| `input/originals.jsonl` | 50 道运行时原题，不含标准答案。每道原题会独立生成 5 次。 |
| `input/variants.jsonl` | 200 个运行时变体，不含标准答案。四种扰动各 50 个，每个生成 1 次。 |
| `test_run_qwen.py` | 检查答案抽取、调度规模和关键运行逻辑。正式运行前执行。 |
| `evaluate_outputs.py` | 离线评估脚本。生成结束后才读取 `gold.jsonl`，计算准确率、稳定性、投票结果、监控基线和成本。 |
| `README.md` | 第三步的简要运行说明。 |
| `perturbcot_model_run_bundle.zip` | 上述运行程序、配置和输入的压缩备份。 |

#### `perturbcot_qwen25_math_15b_results/merged-results/`：最终权威运行记录

| 文件 | 用途 |
| --- | --- |
| `generations.jsonl` | 最重要的输出。包含全部 450 条提示词、原始 CoT、提取答案、种子、token、耗时、截断和运行状态。 |
| `schedule.jsonl` | 450 条固定执行顺序、版本类型、重复编号和逐条随机种子。 |
| `originals.jsonl`、`variants.jsonl` | 实际提交给模型的无标准答案输入副本。 |
| `config.json` | 实际运行时使用的固定配置副本。 |
| `raw_summary.json` | 不读取标准答案的运行完整性、答案抽取、token 和耗时汇总。 |
| `shard0_manifest.json` 至 `shard3_manifest.json` | 各 GPU 分片的软件、硬件、输入哈希、记录数和完成状态。 |
| `run3.log`、`shard1_retry.log`、`shard2c.log`、`shard3c.log` | 关键运行和错误重试日志。 |
| `run_qwen.py` | 与实际结果一起保存的主程序快照，防止以后源代码变化。 |

`perturbcot_qwen25_math_15b_results/offline_evaluation/` 是下载结果后连接标准答案生成的评估输出：

- `evaluated_generations.jsonl`：450 条带离线正确性标签的记录；
- `group_summary.jsonl`：按 50 道原题汇总重复生成和变体；
- `metrics.json`：机器可读的准确率、稳定性和成本指标；
- `run_summary.md`：人可读的结果摘要。

### 3.3 ICRN 上如何重新运行

先把 `perturbcot_model_run/` 放到 ICRN，并确认 `config.json` 中的模型快照路径仍然存在。进入目录：

```bash
cd ~/521project/perturbcotremote20261015/outputs/perturbcot_model_run
```

先检查代码并运行两条 smoke test：

```bash
/home/haod6/.conda/envs/dlcv/bin/python test_run_qwen.py

/home/haod6/.conda/envs/dlcv/bin/python run_qwen.py \
  --config config.json \
  --originals input/originals.jsonl \
  --variants input/variants.jsonl \
  --output-dir smoke-results-new \
  --limit 2
```

完整运行：

```bash
/home/haod6/.conda/envs/dlcv/bin/python run_qwen.py \
  --config config.json \
  --originals input/originals.jsonl \
  --variants input/variants.jsonl \
  --output-dir results-new
```

脚本会逐条追加写入并强制落盘。如果运行中断，重新执行同一条命令即可；已有 `run_id` 会被跳过。

需要分片时，450 条调度可以按 `113 + 113 + 112 + 112` 划分。每个命令必须使用不同输出目录，可以顺序执行，也可以分配到不同 GPU 会话：

```bash
/home/haod6/.conda/envs/dlcv/bin/python run_qwen.py --output-dir rerun-shard0 --start 0   --limit 113
/home/haod6/.conda/envs/dlcv/bin/python run_qwen.py --output-dir rerun-shard1 --start 113 --limit 113
/home/haod6/.conda/envs/dlcv/bin/python run_qwen.py --output-dir rerun-shard2 --start 226 --limit 112
/home/haod6/.conda/envs/dlcv/bin/python run_qwen.py --output-dir rerun-shard3 --start 338 --limit 112
```

不要把四个命令同时放到同一张 GPU 上运行。分片完成后应按 `schedule_index` 合并，并按 `run_id` 去重；合并结果必须满足 450 条记录、450 个唯一 `run_id`，且没有缺失调度项。当前权威合并结果已经保存在 `merged-results/generations.jsonl`，一般不需要重新合并。

### 3.4 如何检查运行结果

在输出目录中执行：

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path("results-new/generations.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]
print("records:", len(rows))
print("unique_run_ids:", len({row["run_id"] for row in rows}))
print("status:", Counter(row["status"] for row in rows))
print("answer_extraction_success:", sum(bool(row.get("answer_extraction_success")) for row in rows))
print("truncated:", sum(bool(row.get("truncated")) for row in rows))
print("input_tokens:", sum(row.get("input_tokens", 0) for row in rows))
print("output_tokens:", sum(row.get("output_tokens", 0) for row in rows))
PY
```

完整运行的最低验收条件：

- `records = 450`；
- `unique_run_ids = 450`；
- 所有记录都有明确状态，失败记录不能删除；
- 50 个 `original_initial`、200 个 `original_repeat`、200 个 `variant_initial`；
- 四种扰动各 50 条。

### 3.5 如何运行离线评估

离线评估必须在模型运行完成后执行，且 `gold.jsonl` 不能提供给运行时模型。以当前本地目录结构为例：

```bash
cd outputs/perturbcot_model_run

python evaluate_outputs.py \
  --generations ../perturbcot_qwen25_math_15b_results/merged-results/generations.jsonl \
  --gold ../perturbcot_data_prep/data/offline_eval/gold.jsonl \
  --output-dir ../perturbcot_qwen25_math_15b_results/offline_evaluation
```

这会重新生成逐条评估、按原题汇总、`metrics.json` 和 `run_summary.md`。

### 3.6 当前权威结果和校验值

当前 `merged-results/generations.jsonl` 的结果为：

- 450/450 条生成成功，450 个唯一 `run_id`；
- 449/450 条成功抽取最终答案；
- 20 条达到 `max_new_tokens=512`，保留为截断记录；
- 输入 46,022 token，输出 111,725 token，总计 157,747 token；
- 原题首次准确率 25/50，即 50.0%；
- 原题五次多数投票准确率 33/50，即 66.0%；
- 200 个变体准确率 117/200，即 58.5%。

关键文件 SHA-256：

```text
71144764aa2d0192f2b134fde74dcf2560cdd6e7f8d824de713098c5f0228a03  run_qwen.py
6fc7470862c443aff00dbbb04641f6d02f18bb582636d955ea22295c43c824f8  config.json
f2c0eb9187b4084e1f7e3075fb69c35fe23d1d249aa8040b6f5c132c532b0366  input/originals.jsonl
5f7b04eb0defe992785f621dafd78e9c43bcb28414ca6bb4c439470d55d203ce  input/variants.jsonl
f7ac121bf08b2020fa21a6125cbef167f98f3bd2536b2408675b0cd01c669766  merged-results/generations.jsonl
```

如果重跑前文件哈希不同，应先解释文件为何变化，再比较新旧实验。完整校验清单见 [`perturbcot_qwen25_math_15b_results/checksums.sha256`](perturbcot_qwen25_math_15b_results/checksums.sha256)。

## 4. 实现两个监控器

**状态：第一版代码与机器评估完成，独立人工核验待完成。**

监控器 A 比较原题首次答案和四个变体答案。监控器 B 在相同输出上抽取显式数值方程，分别记录原题算术错误、共享操作数上的结果冲突、变体算术错误和无关数字使用。不同顺序、舍入差异和结果相同的等价写法不报警；没有可靠关系时输出 `unknown`。

运行时监控不读取标准答案。离线评估阶段才将报警与原题首次答案的正确性连接。

**输出：**

- [`perturbcot_monitors/monitor_answer.py`](perturbcot_monitors/monitor_answer.py)：监控器 A。
- [`perturbcot_monitors/monitor_cot.py`](perturbcot_monitors/monitor_cot.py)：CoT 关系抽取与监控器 B。
- [`perturbcot_monitors/monitor_spec.md`](perturbcot_monitors/monitor_spec.md)：规则、等价处理和限制。
- [`perturbcot_monitors/results/runtime/monitor_results.jsonl`](perturbcot_monitors/results/runtime/monitor_results.jsonl)：不含金标准的 50 组监控结果和具体理由。
- [`perturbcot_monitors/results/runtime/cot_relations.jsonl`](perturbcot_monitors/results/runtime/cot_relations.jsonl)：抽取出的方程和关系。
- [`perturbcot_monitors/results/offline_evaluation/monitor_metrics.json`](perturbcot_monitors/results/offline_evaluation/monitor_metrics.json)：全部、dev 和 test 指标。
- [`perturbcot_monitors/results/offline_evaluation/human_audit.csv`](perturbcot_monitors/results/offline_evaluation/human_audit.csv)：30 题分层人工核验表，当前均为 `pending_human_review`。

## 5. 实现统一的额外验证步骤

**状态：离线重放完成，结果 provisional。**

自适应实验把已有的四次独立原题采样 `r1-r4` 作为验证包，与初始 `r0` 组成五样本唯一众数。存在唯一最高频答案时更新；并列、有效票不足或无法解析时保留初始答案。风险构建与验证均不读取标准答案。

**输出：** [`perturbcot_adaptive/`](perturbcot_adaptive/) 包含协议、实现、测试、gold-free 特征和 provisional 评估结果。

## 6. 设置公平的对照实验

**状态：provisional 完成。**

已实现无验证、始终验证、成本匹配随机验证、四扰动任一分歧、四扰动完整风险，以及顺序风险提前停止。透明风险分数组合答案熵、原答案支持度、CoT 方程异常和抽取不确定性；阈值只在 10 道 dev 上选择，40 道 test 只评估。

**已有输出：**

- [`perturbcot_qwen25_math_15b_results/offline_evaluation/metrics.json`](perturbcot_qwen25_math_15b_results/offline_evaluation/metrics.json)：基础和重复投票结果。
- [`perturbcot_monitors/results/offline_evaluation/monitor_metrics.json`](perturbcot_monitors/results/offline_evaluation/monitor_metrics.json)：答案与 CoT 检测对比。
- [`perturbcot_adaptive/provisional_evaluation/metrics.json`](perturbcot_adaptive/provisional_evaluation/metrics.json)：自适应策略、成本、消融、bootstrap 和顺序敏感性结果。
- [`perturbcot_adaptive/provisional_evaluation/accuracy_cost_curve.svg`](perturbcot_adaptive/provisional_evaluation/accuracy_cost_curve.svg)：准确率与额外 token 曲线。

## 7. 评估效果并解释案例

**状态：机器评估 provisional 完成，人工案例核验待完成。**

已统计原题初始准确率、重复投票准确率、扰动准确率、稳定性、监控混淆矩阵、置信区间、token 和耗时。已自动列出答案监控成功、答案分歧但初始答案正确、CoT 新增有效信息、答案一致但仍错误四类案例集合。后两类在本批样本中为空。

已完成验证前后准确率、错题修正、正确改错、触发成本和结果图；经过人工确认的案例叙述仍待完成。

**已有输出：**

- [`perturbcot_monitors/results/offline_evaluation/evaluated_monitor_results.jsonl`](perturbcot_monitors/results/offline_evaluation/evaluated_monitor_results.jsonl)
- [`perturbcot_monitors/results/offline_evaluation/monitor_summary.md`](perturbcot_monitors/results/offline_evaluation/monitor_summary.md)
- [`perturbcot_monitors/results/offline_evaluation/human_audit.csv`](perturbcot_monitors/results/offline_evaluation/human_audit.csv)

## 8. 整理最终交付物

**状态：部分完成。**

现有交付物包括实验范围、变体规格、配对数据、生成代码、450 条原始输出、运行配置、监控代码、自适应验证重放、完整机器对照和结果图。尚缺完成盲审后的正式指标、正式案例集、端到端演示和最终项目报告。

## 当前分析 Summary

### 模型与扰动稳定性

- 原题首次准确率为 25/50，即 50.0%，95% Wilson 区间为 36.6%–63.4%。
- 原题五次多数投票准确率为 33/50，即 66.0%，说明重复采样能纠正部分随机错误。
- 只有 16/50 个原题组在五次重复采样中答案完全稳定。
- 200 个变体中只有 89 个答案与原题首次答案相同，匹配率为 44.5%。输入扰动造成的输出变化很大，但这部分变化也混合了模型自身的采样波动。
- 变体准确率分别为：实体改名 54%、格式变化 52%、无关信息 62%、释义改写 66%。50 题样本的区间较宽，不能把这些排序解释成稳定的扰动难度结论。

### 监控器结果

- 答案监控对 37/50 题报警，抓到全部 25 个初始错误：TP=25、FP=12、FN=0、TN=13，召回率 100%，精确率 67.6%，正确题误报率 48%。
- CoT 独立信号对 4/50 题报警，这 4 题的原始答案都错误，因此机器离线标签下精确率为 100%。但它只抓到 25 个错误中的 4 个，总体召回率为 16%；20/50 题因缺少可比较的显式关系而为 `unknown`。
- CoT 的 4 个报警全部已经被答案监控发现，新增真阳性为 0，新增误报也为 0。答案＋CoT 的报警集合因此与答案监控相同。
- 组合监控把 4 个答案一致且原始答案正确的题组从 `consistent` 降为 `unknown`，覆盖率从 100% 降到 92%，没有提高错误检测。

### 当前能回答与不能回答的问题

RQ1 已有初步答案：Qwen2.5-Math-1.5B 在这些输入扰动和重复采样下都表现出较强不稳定性。答案不一致是高召回但高误报的风险信号。

RQ2 的第一版结果不支持 CoT 有增量优势。显式方程检查能给出准确、可解释的错误理由，例如“`80 - 220 + 23 = 83`，实际为 `-117`”，但覆盖率太低，而且没有发现答案监控漏掉的错误。这不是最终否定结论，因为当前解析器无法理解大量纯自然语言关系，独立人工审核也未完成。

RQ3 已有 provisional 答案：test 初始准确率为 23/40；始终验证和完整风险顺序策略均达到 30/40。完整风险策略使用 58,119 个额外 token，略高于始终验证的 57,014；答案-only 顺序消融同样达到 30/40，只使用 54,645 token。当前数据支持顺序提前停止的成本价值，但不支持 CoT/抽取风险分量带来额外收益。

### 建议的下一步

1. 完成 162 条答案抽取盲审，并使用 `--require-reviewed` 重跑全部正式指标。
2. 完成监控器人工审核表，重点检查 CoT 报警、`unknown` 和答案监控误报。
3. 在不查看 test 的前提下决定是否调整 CoT 表示；若调整，当前 50 题继续作为探索集，不覆盖已有结论。
4. 若需要确认性结论，再追加未查看的新题组；当前结果只按 exploratory pilot 报告。
