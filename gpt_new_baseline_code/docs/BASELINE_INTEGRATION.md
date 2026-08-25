# 新 Baseline 接入说明

## 目标与映射

本交付针对 DACI 论文 `MobiHoc_2026_PPINf.pdf` 的两个需要和外部方法比较的实验：

| DACI 论文实验 | 目的 | 现有代码入口 | 新代码入口 | 标准输出 |
| --- | --- | --- | --- | --- |
| §5.2 Overall Performance / Table 3 | 在三种模型规模和 default drift 下比较 DACI 与静态、反应式、迁移型方法的 TTLT、P99 TPOT、重配置开销。 | `experiments/exp1_overall/run.sh` | `scripts/run_exp1_overall_with_new_baselines.sh` | `outputs/exp1_overall_small|medium|large/<baseline>/summary.csv` |
| §5.5 Long-Horizon Efficiency / Figure 5 | 改变 `G_hat`，比较 DACI 与对照方法，观察请求变长时 TTLT 差距是否扩大。 | `experiments/exp5_scalability/G_sweep.sh` | `scripts/run_exp5_g_sweep_with_new_baselines.sh` | `outputs/exp5_scalability/G_sweep/G_<G>_<baseline>/summary.csv` |

所有脚本通过 `run_new_baselines.py` 调用 DACI 的 `Config`、`build_cluster`、`run_trace` 和 `src.metrics`，所以落盘格式和现有实验兼容：

```text
outputs/<experiment>/<run_id>/
  config_snapshot.json
  experiment_meta.json
  summary.csv
  traces/
    <scheme>_seed<N>.jsonl
    <scheme>_seed<N>_devices.jsonl   # --log_level full 时
    <scheme>_seed<N>_tokens.jsonl    # --log_level full 时
```

`summary.csv` 是 Table 3 与 Figure 5 后处理的主数据。JSONL 保留每个窗口的边界、placement、是否提交、重配置开销、真实 drift 和预测相关字段，便于检查过程。

## DynaPipe

### 原论文与原始代码语义

输入材料：`gpt_analysis/new_baseline/DynaPipe_Dynamic_Layer_R.pdf` 和 `https://github.com/xhx1022/DynaPipe`，检查的 upstream revision 是 `69f69ad`。

DynaPipe 的论文目标是解决批处理 pipeline serving 中末阶段 logits/sampling 所引入的 bubble。它：

- 基于执行时间预测，识别末 stage 的 sampling 开销；
- 将少量 layer 从末 stage 向上游 stage 均分，以平衡 pipeline stage 时间；
- 预加载可能迁移的权重，并异步迁移 KV cache；
- 使用稳定窗口抑制频繁重分配。

原始实现的 `gllm/worker.py` 中还包含关键 guard：当没有 prefill 且活跃 decode sequence 少于 5 条时，会重置调整计数，不触发 layer adjustment。DynaPipe 的论文实验是四张 A100-PCIe GPU、ShareGPT/Azure-Conv 多请求服务；并非 DACI 的单个长请求、8 个异构 Jetson 仿真。

### 本次接入方式

采用**语义等价的模拟适配**，不是调用原始 gLLM/A100 实现。新增的 `baseline_adapters/dynapipe.py`：

- 固定运行期 worker/placement，只改变 contiguous layer boundary；
- 用 DACI 当前时刻观测与 `C_stage`/`D_stage` 构造 pipeline bottleneck 评分；
- 将至多 `S-1` 个层从最后 stage 分发给上游 stage；
- 保留 25-window 稳定门；
- 默认 `active_decode_requests=1`，`minimum_active_decode_requests=5`，因此忠实保留原始 guard 并不做调整。

这意味着默认 §5.2 和 §5.5 运行会产生可读、可聚合的 `DynaPipe` 行，但它代表“DynaPipe 在单请求条件下不满足其启动条件时的静态 pipeline fallback”，不是 DynaPipe 论文的原始多请求性能。这个保守行为优于在 DACI 单请求 cost model 中虚构 batch bubble、sampling 与异步迁移的收益。

如需研究性地观察 layer redistribution 控制逻辑，可显式开启：

```bash
python gpt_new_baseline_code/run_new_baselines.py \
  --config_dir configs \
  --output_dir outputs/exploratory_dynapipe \
  --run_id qwen_batch8 \
  --schemes DynaPipe \
  --n_traces 1 \
  --model_name qwen3-14b \
  --G_hat 1000 \
  --dynapipe-active-decode-requests 8 \
  --allow-exploratory-batch-mode \
  --log_level full
```

这个模式会在 `experiment_meta.json` 中标记为 `exploratory_batch_semantic_adapter`，不能用于当前 DACI 论文的 Table 3 或 Figure 5 结论。DACI 的 token-time cost 仍是单请求串行模型，不能表示 DynaPipe 的 batch pipeline bubble 和吞吐。

### 新增文件与作用

| 文件 | 作用 | 获批后建议正式位置 |
| --- | --- | --- |
| `baseline_adapters/dynapipe.py` | DynaPipe 语义适配 controller。 | `src/schemes/dynapipe.py` |
| `baseline_adapters/registry.py` | 进程内注册和可比性元数据。 | `src/schemes/schemes.py` 的 registry 附近，或单独 `src/schemes/registry.py` |
| `run_new_baselines.py` | 不改 `run.py` 的兼容 runner，写入 DACI 标准输出。 | 合并后可把注册逻辑加入 `run.py`；保留为 `tools/run_new_baselines.py` 也可。 |
| `scripts/run_exp1_overall_with_new_baselines.sh` | 产生 Table 3 中 DynaPipe 的三个模型目录。 | `experiments/exp1_overall/run_new_baselines.sh` |
| `scripts/run_exp5_g_sweep_with_new_baselines.sh` | 产生 Figure 5 的五个 `G_hat` 目录。 | `experiments/exp5_scalability/run_new_baselines.sh` |
| `scripts/aggregate_table3_extension.py` | 合并可用 Table 3 结果为 long-form CSV。 | `experiments/aggregate.py` 的子命令或 `experiments/exp1_overall/` |
| `scripts/aggregate_figure5_extension.py` | 合并 G sweep 并计算 DACI 相对对照方案的 TTLT 差异。 | `experiments/exp5_scalability/` |

### 运行和数据解释

运行 Table 3 扩展：

```bash
N_TRACES=30 bash gpt_new_baseline_code/scripts/run_exp1_overall_with_new_baselines.sh
```

输出路径为：

```text
outputs/exp1_overall_small/DynaPipe/
outputs/exp1_overall_medium/DynaPipe/
outputs/exp1_overall_large/DynaPipe/
outputs/exp1_overall_table3_with_new_baselines.csv
```

运行 Figure 5 扩展：

```bash
N_TRACES=30 bash gpt_new_baseline_code/scripts/run_exp5_g_sweep_with_new_baselines.sh
```

输出路径为：

```text
outputs/exp5_scalability/G_sweep/G_5000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_10000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_15000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_20000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_40000_DynaPipe/
outputs/exp5_scalability/G_sweep/figure5_with_new_baselines.csv
```

`aggregate_table3_extension.py` 将现有 SDA/RT/FM/DACI 和新增目录中已经存在的 `summary.csv` 聚合为 long-form 表。`aggregate_figure5_extension.py` 同样读取已有 DACI/SDA 与新增目录；若 DACI 结果存在，它会增加 `daci_relative_ttlt_change_pct`。两个聚合器不会重跑或改写原有实验数据。

## FlexPipe

### 可行性判定

FlexPipe 论文研究 serverless GPU cluster 中的多请求 LLM serving：根据请求到达波动和资源碎片，在细/粗粒度 pipeline、data-parallel replica 与 GPU 资源分配之间进行 inflight refactoring。其评测基于 82 GPU Kubernetes 集群和吞吐/资源效率指标。

给定的 GitHub 链接 `alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI` 是论文所用 workload trace 的数据仓库，不是 FlexPipe 可执行实现。更重要的是，DACI 当前模型没有请求 arrival process、队列、batch、弹性 GPU 获取/释放、data parallelism 或 resource fragmentation allocator。因此不能在不改变研究问题的前提下，把 FlexPipe 接入 §5.2 和 §5.5。

结论：**不生成 FlexPipe 的 DACI 模拟器 scheme，也不生成 Table 3/Figure 5 数值。** 最接近且可复现的替代实验是：在 FlexPipe 原始系统或其作者提供的实现上，定义新的多请求/吞吐实验，并将 DACI 作为必须重新实现的对照；这应被报告为新增实验而非当前论文 Table 3/Figure 5 的扩展。

## Seesaw

### 可行性判定

Seesaw 论文研究离线、throughput-oriented LLM inference。它在 prefill 与 decode 之间切换 pipeline/tensor parallelism，借助 CPU tiered KV buffer 与 transition-minimizing schedule 降低 model re-sharding 开销。其局部 artifact 为 `gpt_analysis/new_baseline/seesaw-artifact/cgen`，要求 Docker、CUDA 12.1、NVIDIA GPU、HuggingFace Llama 权重；artifact 已知限制是仅支持 Llama family，且仅支持从较大 PP 过渡到较小 PP。

DACI 的 Table 3 包括 Gemma/Qwen，并测量单请求 TTLT；Figure 5 则研究单请求 drift 下的延迟摊销。Seesaw 的高吞吐 metric、batch scheduler、TP/PP 状态、CPU KV buffer 均不在 DACI 模拟器状态中。因此原样接入会改变 benchmark 的目标，不能给出有科学意义的同口径结果。

结论：**不生成 Seesaw 的 DACI simulator scheme，也不生成 Table 3/Figure 5 数值。** 最接近的可复现实验是在原 artifact 上，针对 Llama 运行其 throughput benchmark；若后续设计了统一的 batch/throughput 实验，可使用下节的外部结果适配器落盘，但不能把它和当前单请求 TTLT 数字直接混在一起。

## 外部结果适配器

`external_result_adapter.py` 用于“调用原 baseline 仓库 + 结果适配器”的场景。它不调用 FlexPipe 或 Seesaw，也不验证科学可比性；它只把经过人工确认、已经对齐单位的每条 trace 写成 DACI 标准目录。

输入 JSONL 每行必须至少有：

```json
{
  "seed": 42,
  "TTLT_s": 12.34,
  "TTFT_s": 0.56,
  "TPOT_series_s": [0.01, 0.01],
  "overhead_s": 0.2,
  "n_reconfigs": 1
}
```

可选字段为 `scheme`、`windows`、`device_seconds`、`tokens`。在导入前必须制作 attestation 文件，明确模型、硬件、请求定义、seed、计时边界和单位都如何与目标实验匹配。命令示例：

```bash
python gpt_new_baseline_code/external_result_adapter.py \
  --baseline Seesaw \
  --input-jsonl /path/to/aligned_seesaw_traces.jsonl \
  --attestation /path/to/measurement_attestation.md \
  --output-dir outputs/externally_measured \
  --run-id Seesaw_matched
```

这只保证输出文件格式兼容，**不自动使 Seesaw/FlexPipe 结果成为 DACI Table 3/Figure 5 的合法数据**。
