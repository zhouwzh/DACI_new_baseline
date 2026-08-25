# 新 Baseline 接入暂存区

本目录是对 DACI 论文实验的增量扩展，刻意不修改仓库既有的 `run.py`、`src/` 或 `experiments/`。所有代码仅在本目录的 runner 进程中动态注册新 scheme，因此可以先审查、测试并决定是否迁入正式代码树。

当前交付包含：

| Baseline | 接入状态 | 方法 |
| --- | --- | --- |
| DynaPipe | 可运行，但不是原始论文系统的严格复现 | 语义等价模拟适配。保留其 layer redistribution 与稳定窗口语义；DACI 的单请求默认触发原始代码的 `decode < 5` guard，因此默认产物是静态 pipeline fallback。 |
| FlexPipe | 不支持直接接入 §5.2/§5.5 | 仅提供外部结果适配器。FlexPipe 是 serverless、多请求、弹性资源和 pipeline-granularity 系统，给定链接还是 trace dataset，不是可调用的实现仓库。 |
| Seesaw | 不支持直接接入 §5.2/§5.5 | 仅提供外部结果适配器。Seesaw 面向离线 throughput，且需在 prefill/decode 间重分片 tensor/pipeline parallelism；DACI 当前成本模型不表示这些状态。 |

先阅读：

1. [BASELINE_INTEGRATION.md](docs/BASELINE_INTEGRATION.md)：每个 baseline 的代码、数据、脚本和论文图表映射。
2. [COMPATIBILITY_AND_LIMITATIONS.md](docs/COMPATIBILITY_AND_LIMITATIONS.md)：论文/代码差异和可比性边界。
3. [RUN_FROM_ZERO.md](docs/RUN_FROM_ZERO.md)：从零运行、聚合与日后迁入正式目录的步骤。

快速 smoke test：

```bash
cd DACI-Drift-Aware-Collaborative-LLM-Inference
python -m unittest gpt_new_baseline_code.tests.test_dynapipe_adapter
```

运行 DynaPipe 的 §5.2 扩展：

```bash
N_TRACES=30 bash gpt_new_baseline_code/scripts/run_exp1_overall_with_new_baselines.sh
```

运行 DynaPipe 的 §5.5 扩展：

```bash
N_TRACES=30 bash gpt_new_baseline_code/scripts/run_exp5_g_sweep_with_new_baselines.sh
```

上述命令产生 DACI 兼容的 `summary.csv` 与 trace JSONL，但不要把默认 DynaPipe 行解释为在 Jetson 单请求任务上复现了原始 DynaPipe 的论文优势。原因和最接近的可复现实验见限制说明。
