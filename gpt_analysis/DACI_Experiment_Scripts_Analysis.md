# DACI 仓库实验脚本与执行流程分析

> 仓库：[hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference)  
> 检查版本：`main` 分支 commit `2328018c8d8a54e37514baccb8134d9fa0e61201`

仓库中提供了较完整的实验脚本，但当前存在脚本与 README 不同步的问题，顶层 `run_all.sh` 不能直接无错误地跑完。

这些实验本质上是 **Python 数值模拟**，没有真的在 8 台 Jetson 上部署并执行 LLM。

## 1. 实验的基本执行流程

所有实验最终都调用 [`run.py`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/run.py)：

1. 从 `configs/*.json` 加载模型、设备、漂移和 DACI 参数。
2. 根据命令行参数覆盖模型、请求长度、窗口长度等配置。
3. 构造模拟的异构集群，默认是：
   - 1 台 Orin AGX
   - 3 台 Orin NX
   - 4 台 Orin Nano
4. 对每个 scheme 和 seed 独立运行一个 trace。
5. 初始部署阶段计算 placement `a`、partition boundaries `b`、startup 和 prefill 延迟。
6. Decode 被划分成长度为 `W` 的控制窗口，默认参数是：
   - Prompt：512 tokens
   - 输出：15,000 tokens
   - `W = 20` tokens
   - 共 750 个控制窗口
7. 每个窗口模拟：
   - background workload
   - 温度变化
   - 网络状态变化
   - 带噪声的观测
   - controller 预测与重新配置决策
   - 权重/KV cache 迁移开销
   - 当前窗口 TPOT
8. 最后统计 TTLT、TTFT、P99 TPOT、迁移开销和重配置次数。

相同 seed 会给不同方法提供相同的外生随机过程，因此可以进行配对比较。

## 2. 比较的方法

实现位于 [`src/schemes/schemes.py`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/src/schemes/schemes.py)：

- `SDA`：只做初始部署，运行时不改变 partition。
- `RT`：当前 TPOT 超过初始值的 1.2 倍时触发，默认冷却 50 个窗口，然后重新搜索 placement 和 partition。
- `FM`：每个窗口都联合搜索 placement 和 partition。
- `DACI`：预测未来漂移，冻结 placement，只优化 partition boundary，并使用 adaptive horizon 和 lazy switching。
- `OR`：使用预先生成的未来 ground truth，作为近似 Oracle。它忽略了自发热反馈，因此并非严格的全局最优解。

## 3. 现有实验脚本

| 实验 | 脚本 | 当前脚本实际执行的内容 |
|---|---|---|
| Overall comparison | [`exp1_overall/run.sh`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/experiments/exp1_overall/run.sh) | 3 个模型 × SDA/RT/FM/DACI × 5 seeds |
| Ablation | [`exp2_ablation/run.sh`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/experiments/exp2_ablation/run.sh) | 当前只启用了 `no_predictor`，5 seeds |
| Drift regimes | [`exp3_regime/run.sh`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/experiments/exp3_regime/run.sh) | 4 regimes × 5 methods × 20 seeds，外加 R2 representative trace |
| Sensitivity | [`exp4_sensitivity/`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/tree/main/experiments/exp4_sensitivity) | 扫描 `W`、`H_max`、`lambda_slack`，并评估 predictor RMSE |
| Scalability | [`exp5_scalability/G_sweep.sh`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/experiments/exp5_scalability/G_sweep.sh) | 扫描输出长度，比较 DACI 与 SDA |
| 全部实验入口 | [`experiments/run_all.sh`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/experiments/run_all.sh) | 设计上依次运行全部实验，但当前不能完整执行 |
| 结果汇总 | [`experiments/aggregate.py`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/experiments/aggregate.py) | 将各实验的 `summary.csv` 汇总成绘图 CSV |

## 4. Exp1：总体性能比较

当前脚本实际运行：

- 模型：
  - `gemma3-4b`
  - `llama-3.2-8b`
  - `qwen3-14b`
- 方法：`SDA RT FM DACI`
- seeds：42–46，共 5 个
- regime：`default`，即 workload、thermal、network 同时变化
- 输出目录：
  - `outputs/exp1_overall_small/`
  - `outputs/exp1_overall_medium/`
  - `outputs/exp1_overall_large/`

当前存在以下问题：

1. README 说运行 5 个方法、20 traces，但脚本只有 4 个方法、5 traces。
2. 脚本使用 `llama-3.2-8b`，最新 `models.json` 中模型已经改名为 `llama-3-8b`，因此 medium model 会报 `KeyError`。
3. README 提到 Oracle，但当前 `SCHEMES` 数组没有 `OR`。

## 5. Exp2：消融实验

代码支持以下消融版本：

- `no_predictor`
- `no_lazy`
- `no_bottleneck`
- `no_adaptive_H`

README 设计的是 Full DACI 加四个消融版本。但是当前 `run.sh` 中其他 variant 都被注释了，只会执行：

```bash
no_predictor:DACI:no_predictor
```

因此，直接运行当前脚本无法得到完整的消融实验结果。

## 6. Exp3：不同漂移类型

实际运行：

- R1：没有 workload、thermal、network drift
- R2：thermal 和 workload 都开启，network 关闭
- R3：只有 workload drift
- R4：只有 network drift
- 每个 regime 比较 SDA、RT、FM、DACI、OR
- 每个组合运行 20 seeds

README 把 R2 称为 “thermal only”，但实际 `drift.json` 中为：

```json
"workload_active": true,
"thermal_active": true
```

所以当前 R2 不是纯 thermal drift。

另外，representative R2 部分没有提前建立 `outputs/exp3_regime/representative_R2`，日志重定向时可能因为目录不存在而失败。

## 7. Exp4：敏感性分析和 Predictor Accuracy

当前参数如下：

- `W_sweep.sh`：`W ∈ {5, 10, 20, 50, 100}`
- `H_sweep.sh`：`H_max ∈ {6, 10, 14}`
- `lambda_sweep.sh`：`λ ∈ {5, 10}`
- `predictor_accuracy.sh`：
  - 保存完整的 `phi_true`
  - 保存各预测 horizon 的 `phi_hat`
  - 计算 Kalman+AR(1) 与 persistence baseline 的 RMSE

虽然脚本注释说每个点运行 5 traces，但这些脚本当前默认都是：

```bash
N_TRACES=1
```

可以通过环境变量覆盖：

```bash
N_TRACES=20 PARALLEL_JOBS=5 \
bash experiments/exp4_sensitivity/run_all.sh
```

## 8. Exp5：Scalability

README 设计了三个方向：

- 集群规模 `N_sweep`
- 模型深度 `L_sweep`
- 输出长度 `G_sweep`

但是仓库中只有 `G_sweep.sh`。当前实际扫描：

```text
G = 5000, 10000, 15000, 20000, 40000
```

它比较 `DACI` 和 `SDA`，每个点默认运行 5 traces。

README 和 `config_override.json` 写的是 `{256, 1024, 2048, 4096}`，与当前脚本不一致。

## 9. 输出文件

每个实验子任务会生成：

```text
outputs/<experiment>/<run_id>/
├── config_snapshot.json
├── experiment_meta.json
├── summary.csv
└── traces/
    └── <scheme>_seed<N>.jsonl
```

`summary.csv` 包括：

- TTLT mean/std
- TTFT mean/std
- P99 TPOT mean/std
- reconfiguration overhead
- number of reconfigurations

使用 `--log_level full` 时还会生成：

- `*_devices.jsonl`：每秒温度、负载和内存状态
- `*_tokens.jsonl`：每个 token 的 TPOT 和各 stage 延迟

仓库目前没有真正的 Matplotlib、Seaborn 或其他绘图脚本，只负责生成 CSV；论文中的柱状图和折线图还需要另外绘制。

## 10. 当前最重要的脚本问题

顶层命令：

```bash
bash experiments/run_all.sh
```

目前不能直接完整运行，原因包括：

1. Exp1 使用了不存在的 `llama-3.2-8b` 模型名。
2. `run_all.sh` 引用了不存在的 `delta_sweep.sh`。
3. `run_all.sh` 引用了不存在的 `N_sweep.sh`。
4. `run_all.sh` 引用了不存在的 `L_sweep.sh`。
5. Exp1 输出到三个 model-specific 目录，但总汇总脚本查找的是 `outputs/exp1_overall`。
6. `config_override.json` 只是说明文件，程序从未读取它。
7. Controller 自身的 DP/搜索时间默认不计入 TTLT，只有迁移开销被计入。虽然配置中提供了 Jetson 上测得的 controller cost，但 `enabled=false`。

## 11. 额外的 30-seed 修正实验

`results/m5a_fixes/` 中还有一组独立的 30-seed 修正实验，以及 [`analyze_30trace.py`](https://github.com/hzhou10cs/DACI-Drift-Aware-Collaborative-LLM-Inference/blob/main/analyze_30trace.py)。

这组实验用于比较：

- `BEFORE`：原始模拟器
- `AFTER`：修复 `H_swap`、baseline placement search、设备内存等问题之后
- `AFTERC`：在上述修复基础上使用实测网络参数

它没有接入 `experiments/run_all.sh`。已有结果显示，修复 baseline 后 FM 和 RT 明显变强，DACI 相对 FM 的原有性能优势大幅缩小；当前更有依据的结论是 DACI 能以较低的重配置开销获得接近 FM 的延迟。

