# DACI / PPINf 论文实验与代码运行说明

本文档对应论文 `MobiHoc_2026_PPINf.pdf`，并说明当前仓库中各实验的入口、运行方式、输出数据和论文图表的对应关系。

## 1. 先理解实验框架

当前仓库实现的是一个**可复现的 Python 数值仿真器**，而不是直接驱动论文中 8 台 Jetson 设备的部署程序。它根据配置中的设备算力、显存、交换开销、网络参数及漂移模型，逐窗口模拟协同 LLM 推理，并比较 DACI 与基线的端到端性能。

运行链路如下：

```text
experiments/<实验脚本>.sh
  -> run.py（读取 configs，解析命令行覆盖项）
  -> src/simulator.py::run_trace（对每个 seed 模拟一条漂移轨迹）
  -> src/schemes/schemes.py（DACI / SDA / RT / FM / OR 的决策）
  -> src/metrics.py（汇总并落盘）
  -> outputs/<实验名>/<run_id>/summary.csv 和 traces/*.jsonl
```

核心入口和职责：

| 代码位置                     | 作用                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `run.py`                   | 通用 CLI 入口；创建运行目录、执行多个 seed、写入元数据与汇总结果。                 |
| `src/simulator.py`         | 模拟工作负载、温度和网络漂移，产生 token/window 级结果。                           |
| `src/schemes/schemes.py`   | 实现 DACI、SDA、RT、FM、Oracle (OR) 以及 DACI 的消融模式。                         |
| `src/metrics.py`           | 计算 TTLT、TTFT、P99 TPOT、Ovhd、`#Reconf`，并写 CSV/JSONL。                     |
| `configs/*.json`           | 设备、模型、漂移和算法默认参数。                                                   |
| `experiments/aggregate.py` | 将多个运行目录汇总为作图用 CSV；预测器分析使用 `exp4_sensitivity/aggregate.py`。 |

推荐从仓库根目录（`DACI-Drift-Aware-Collaborative-LLM-Inference`）运行以下命令。脚本为 Bash，Windows 请使用 WSL、Git Bash 或 Linux/macOS shell。

## 2. 论文实验与仓库对应关系

### 2.1 每个实验在验证什么

以下按论文 §5.1--§5.5 的实验描述和结果讨论归纳，说明的是论文提出这些实验的目的，而不只是脚本执行了哪些参数扫描。

| 论文实验                                      | 实验在做什么：论文要验证什么                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §5.1 Experiment Setting                      | 建立后续比较共享的条件：论文以 8 个异构 Jetson 节点、三种模型规模和包含 thermal、workload、network 的隐藏真实漂移轨迹，模拟一个延迟敏感的长请求。控制器只能看到带噪声的观测，因此该设置检验的是 DACI 是否能在非平稳边缘环境中工作，而不是静态机器上的一次性切分。论文将各方法放在相同条件下，并说明常规结果平均自 30 条随机 drift traces。                                                               |
| §5.2 Overall Performance Comparison，Table 3 | 在 4B、8B、14B 三种模型上比较 SDA、RT、FM、DACI，验证论文的核心假设：模型越大，在线改变 stage placement 所需的权重加载和 KV 迁移成本越高。DACI 在请求开始时冻结 placement、运行时只移动相邻 stage 的 layer boundary；论文要证明这种结构性地避免 placement migration 的策略，会比静态方案和迁移型方案同时有更低的总请求延迟（TTLT）与重配置开销，而且优势随模型规模扩大。                                 |
| §5.3 Ablation Study，Figure 4                | 逐一移除 DACI 的关键设计，验证性能收益来自哪些组件。`w/o Freeze` 允许每窗口重新优化 placement，检查冻结 placement 是否确实避免昂贵迁移；`w/o Predictor` 用当前 drift 的持久性假设替代多步预测，检查前瞻性预测是否改善决策；`w/o Lazy-Switching` 每个窗口都提交候选边界，检查成本门控是否能阻止无收益的连续 handoff。论文结论是：三者缺一不可，尤其无 lazy switching 时开销会因频繁提交而急剧放大。 |
| §5.4 Sensitivity Analysis，Figure 3          | 扫描三个 MPC 超参数，检验 DACI 是否依赖脆弱的调参。窗口长度 `W` 是“太频繁地响应噪声”和“太慢地响应已发生 drift”的权衡；预测上限 `H_max` 是预测信息增益与远期预测方差的权衡；鲁棒 slack `lambda` 决定候选重配置需要有多大收益才值得支付成本。论文用这些曲线说明各参数在一段宽范围内都保持接近最优。                                                                                              |
| §5.5 Long-Horizon Efficiency，Figure 5       | 改变预计输出长度 `G_hat`，仅比较 DACI 和静态 SDA，验证 DACI 的优势是否会随请求变长而扩大。论文的推理是：初始部署成本只在开始支付一次，长请求有更多 decoding windows 来摊销这一成本，也会积累更多可由 boundary adaptation 吸收的 drift。因此短请求可能没有明显收益，而长请求应呈现越来越大的 TTLT 改进。                                                                                                |
| 预测器误差诊断（仓库补充，非论文主图）        | `predictor_accuracy.sh` 直接对照模拟器隐藏的真实漂移 `phi_true` 与预测值 `phi_hat_horizon`，量化 RMSE。这是对 §5.3 中“预测器改善控制质量”的代码级证据补充；当前 PDF 并未将预测 RMSE 作为独立图表报告。                                                                                                                                                                                          |
| 漂移机制分解（仓库扩展，非论文主图）          | `exp3_regime` 将 calibrated、thermal、workload、network 情形拆开，目的在于定位不同 drift 来源下各策略的表现，并导出一条逐窗口代表性轨迹。它与论文的三类漂移系统模型一致，但当前论文的 §5.2--§5.5 未将其作为单独主图或表，因此不能将该实验数据直接当成论文已报告的结果。                                                                                                                              |

### 2.2 论文图表、代码位置与运行方式

| 论文位置与目的                                                       | 代码位置                                                                                              | 默认比较 / 参数                                                                                                                                                                                                   | 运行方式                                                                                                                                                                       | 主要输出                                                                                                                                                                          |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §5.1 Experimental Setup：模型、设备参数和默认漂移设置               | `configs/models.json`、`configs/devices.json`、`configs/drift.json`、`configs/algo.json`      | Gemma3-4B、LLaMA、Qwen3-14B；默认 `W=20`、`H_max=8` 等                                                                                                                                                        | 配置会被下面所有脚本自动读取                                                                                                                                                   | 每次运行中的 `config_snapshot.json`，用于记录实际配置快照。                                                                                                                     |
| §5.2 Overall Performance，**Table 3**                         | `experiments/exp1_overall/run.sh`                                                                   | SDA、RT、FM、DACI；三个模型；default drift<br /><br />Table 3：完整的 DACI 与四类基线比较，实际表中为 SDA、RT、FM、DACI。                                                                                         | `bash experiments/exp1_overall/run.sh`                                                                                                                                       | `outputs/exp1_overall_small`、`_medium`、`_large` 下每个方案的 `summary.csv`。每行是一个 seed，包含 `TTLT_s`、`TTFT_s`、`P99_TPOT_s`、`Ovhd_s`、`n_reconfigs`。 |
| §5.3 Ablation Study，**Figure 4**（Qwen3-14B）                | `experiments/exp2_ablation/run.sh`；实现见 `src/schemes/schemes.py`                               | Full DACI、no-freeze (FM)、no-lazy、no-predictor<br />Figure 4：以完整 DACI 为参照的消融比较。`w/o Freeze` 本质上接近 FM，另外两项是 DACI 的 no-lazy、no-predictor 变体；不是引入外部 baseline 的完整横向比较。 | `bash experiments/exp2_ablation/run.sh`                                                                                                                                      | `outputs/exp2_ablation/<variant>/summary.csv`，用于 TTLT、重配置次数、开销、P99 TPOT 四个面板。                                                                                 |
| §5.4 Sensitivity Analysis，Figure 3                                 | `experiments/exp4_sensitivity/W_sweep.sh`、`H_sweep.sh`、`lambda_sweep.sh`                      | 分别扫描窗口长度 `W`、最大预测 horizon `H_max`、鲁棒 slack `lambda`<br />只运行 DACI，扫描 `W`、`H_max`、`lambda`，考察超参数敏感性。                                                                 | 依次运行三个脚本；或 `bash experiments/exp4_sensitivity/run_all.sh`                                                                                                          | `outputs/exp4_sensitivity/W_sweep`、`H_sweep`、`lambda_sweep` 的 `summary.csv`；再由聚合脚本整理为曲线数据。                                                              |
| §5.4（预测精度的补充诊断）                                          | `experiments/exp4_sensitivity/predictor_accuracy.sh`、`experiments/exp4_sensitivity/aggregate.py` | DACI，full token/window log                                                                                                                                                                                       | `bash experiments/exp4_sensitivity/predictor_accuracy.sh`，再运行 `python experiments/exp4_sensitivity/aggregate.py predictor outputs/exp4_sensitivity/predictor_accuracy` | `traces/*_seed*.jsonl` 中的 `phi_true` 和 `phi_hat_horizon`；聚合得到预测 RMSE。                                                                                            |
| §5.5 Horizon / request-length 效果，Figure 5                        | `experiments/exp5_scalability/G_sweep.sh`                                                           | Qwen3-14B；DACI vs SDA；`G_hat={5000,10000,15000,20000,40000}`<br />DACI 与 SDA 的二元比较，验证请求长度变长时 DACI 相对静态方案的优势是否扩大                                                                  | `bash experiments/exp5_scalability/G_sweep.sh`                                                                                                                               | `outputs/exp5_scalability/G_sweep/G_<G>_<scheme>/summary.csv`，以不同生成长度下的 TTLT 或相对差距作图。                                                                         |
| 仓库扩展：不同漂移机制下的鲁棒性（主论文当前版本未单独给出对应主图） | `experiments/exp3_regime/run.sh`                                                                    | `R1_calibrated`、`R2_thermal`、`R3_workload`、`R4_network` x SDA/RT/FM/DACI/OR                                                                                                                            | `bash experiments/exp3_regime/run.sh`                                                                                                                                        | `outputs/exp3_regime/<regime>_<scheme>/summary.csv`；另有 `representative_R2/traces/` 的逐窗口和逐 token 日志。                                                               |

## 3. 每个论文数据是如何生成的

### Table 3：总体性能

`exp1_overall/run.sh` 对每个模型和每个策略调用一次 `run.py`。`run.py` 为 `n_traces` 个连续 seed 各生成一条漂移轨迹；每一条轨迹模拟完整请求生成过程。`src/metrics.py` 将该轨迹的请求级指标写为一行 `summary.csv`。

论文表格中的一个单元格应由同一模型、同一策略对应 `summary.csv` 的多行做均值/分位数汇总得到。论文 §5.1.5 表述为“每个报告数值平均 30 条 drift traces”；因此要用于论文级统计时，建议显式执行：

```bash
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp1_overall/run.sh
```

### Figure 4：DACI 消融

消融本质上是向 `run.py` 传入 `--ablation`，再由 `DACIScheme` 改变控制逻辑：

| 图中变体     | 命令参数 / 实现                                                          |
| ------------ | ------------------------------------------------------------------------ |
| Full DACI    | `--schemes DACI --ablation none`                                       |
| No freeze    | 使用 `--schemes FM`，每个窗口联合优化切分点。                          |
| No lazy      | `--schemes DACI --ablation no_lazy`，不再延迟/冻结边界提交。           |
| No predictor | `--schemes DACI --ablation no_predictor`，使用 persistence predictor。 |

当前脚本中只有 `no_predictor` 未被注释；要得到完整 Figure 4，需在 `VARIANTS` 中恢复其余三项后运行：

```bash
N_TRACES=30 bash experiments/exp2_ablation/run.sh
```

### Figure 3：敏感性曲线

三个扫描脚本都固定其他配置，仅用 `run.py` 的 CLI 覆盖一个参数：

| 脚本                | 覆盖参数           | 当前脚本扫描值     |
| ------------------- | ------------------ | ------------------ |
| `W_sweep.sh`      | `--W_tokens`     | 5, 10, 20, 50, 100 |
| `H_sweep.sh`      | `--H_max`        | 6, 10, 14          |
| `lambda_sweep.sh` | `--lambda_slack` | 5, 10              |

这些脚本默认 `N_TRACES=1`，只适合快速检查。生成论文级曲线时应提高 trace 数，例如：

```bash
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp4_sensitivity/W_sweep.sh
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp4_sensitivity/H_sweep.sh
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp4_sensitivity/lambda_sweep.sh
```

### Figure 5：生成长度 / horizon 下的差距

`G_sweep.sh` 通过 `--G_hat` 改变每条请求的目标生成 token 数。它对每个 `G_hat` 分别运行 DACI 和 SDA，读取两方 `summary.csv` 的 TTLT 后计算差值或百分比改进，即可得到 Figure 5 的趋势。论文的横轴若使用“horizon”表述，需要在作图阶段明确它对应此脚本的 `G_hat`，而非 DACI 的 `H_max`。

## 4. 输出目录和数据格式

每个 `run.py` 调用都会在 `--output_dir/<run_id>/` 下创建：

```text
<run_id>/
  config_snapshot.json       # 本次实际生效的完整配置
  experiment_meta.json       # run id、策略、seed 范围、时间等
  summary.csv                # 每条 trace 一行的请求级指标
  traces/
    <scheme>_seed<N>.jsonl   # 窗口级决策与性能日志
    <scheme>_seed<N>_devices.jsonl  # 仅 log_level=full
    <scheme>_seed<N>_tokens.jsonl   # 仅 log_level=full
```

`summary.csv` 是重建论文表和主图的首选数据源。`traces/*.jsonl` 用于解释某次重配置、绘制逐窗口曲线或计算预测误差；其中窗口日志包含 `phi_true`、`phi_hat_horizon`、`accepted`、`omega` 等字段。

## 5. 复现前必须处理的差异

当前分支不能无修改地保证复现 PDF 中的数值；以下差异应在出图前固定并记录：

1. `exp1_overall/run.sh` 使用已不存在的模型键 `llama-3.2-8b`，而 `configs/models.json` 中的有效键为 `llama-3-8b`。不修正将导致 medium 模型运行失败。
2. 论文称结果平均 30 条轨迹，但现有脚本常用 1、5 或 20 条；必须通过 `N_TRACES=30` 统一。
3. `exp2_ablation/run.sh` 当前只运行 `no_predictor`，其余三项被注释，无法直接生成完整 Figure 4。
4. 顶层 `experiments/run_all.sh` 引用了缺失的 `delta_sweep.sh`、`N_sweep.sh`、`L_sweep.sh`，且将 Exp1 错误汇总为不存在的 `outputs/exp1_overall`；不要把它作为一键复现入口。
5. `configs/algo.json` 的 `lambda_slack=4.0` 与论文正文中关于默认 slack 的表述并不一致（论文内也存在 `lambda=0` 与 `lambda=1` 两种说法）。复现实验前应选定一个默认值，并把最终值保存在每次输出的 `config_snapshot.json`。
6. `R2_thermal` 在 `configs/drift.json` 中同时启用了 workload 和 thermal 漂移，因此它不是严格的“thermal only”。

此外，`results/m5a_fixes/README.md` 与 `analyze_30trace.py` 记录了后续的 30-seed 重新统计和若干正确性修复；若目标是评价当前代码的可信结论，应优先参考该目录下的 regenerated Table 3，而不是只比较 PDF 中印刷的原始数值。

## 6. 建议的可审计运行顺序

```bash
# 0. 在仓库根目录；先验证单个短实验
python run.py --config_dir configs --output_dir outputs/smoke --run_id DACI \
  --schemes DACI --n_traces 1 --seed_start 42 --regime default \
  --model_name qwen3-14b --log_level summary_only

# 1. 修正脚本中的模型键和消融 VARIANTS 后，运行论文主实验
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp1_overall/run.sh
N_TRACES=30 PARALLEL_JOBS=4 bash experiments/exp2_ablation/run.sh
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp4_sensitivity/W_sweep.sh
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp4_sensitivity/H_sweep.sh
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp4_sensitivity/lambda_sweep.sh
N_TRACES=30 PARALLEL_JOBS=5 bash experiments/exp5_scalability/G_sweep.sh

# 2. 对应目录分别做聚合；参数为实验类型和输出目录
python experiments/aggregate.py ablation outputs/exp2_ablation
python experiments/aggregate.py sensitivity outputs/exp4_sensitivity
python experiments/aggregate.py scalability outputs/exp5_scalability
```

Exp1 因按模型拆分输出目录，应分别读取/汇总 `outputs/exp1_overall_small`、`outputs/exp1_overall_medium` 和 `outputs/exp1_overall_large`。所有最终作图表应同时保存源 `summary.csv`、聚合 CSV、绘图脚本和本次的 `config_snapshot.json`，以便审稿或后续 baseline 扩展时精确追溯。
