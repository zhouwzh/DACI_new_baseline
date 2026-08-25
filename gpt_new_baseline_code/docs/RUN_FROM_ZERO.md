# 从零运行新 Baseline 实验

## 1. 前提

在仓库根目录 `DACI-Drift-Aware-Collaborative-LLM-Inference` 下执行。需要一个能运行现有 DACI 模拟器的 Python 环境（至少 `numpy`），以及 Bash 环境（WSL、Git Bash 或 Linux/macOS shell）来运行两个批处理脚本。

本交付不安装 DynaPipe 的 gLLM/CUDA 依赖。DynaPipe 的默认接入是 DACI 内的语义适配器；FlexPipe 和 Seesaw 不会由本交付自动启动。

## 2. 先做 smoke test

```bash
python -m unittest gpt_new_baseline_code.tests.test_dynapipe_adapter
```

测试做两件事：

1. 以 1 条 Gemma trace 运行 DynaPipe 默认单请求 fallback，并检查标准输出文件和 full JSONL。
2. 确认试图用 batch 语义运行 DynaPipe 却没有明确 opt-in 时会失败，避免误把 exploratory 结果放进论文图表。

也可手动验证单个运行：

```bash
python gpt_new_baseline_code/run_new_baselines.py \
  --config_dir configs \
  --output_dir outputs/smoke_new_baseline \
  --run_id DynaPipe \
  --schemes DynaPipe \
  --n_traces 1 \
  --seed_start 42 \
  --model_name qwen3-14b \
  --G_hat 200 \
  --log_level full

python gpt_new_baseline_code/scripts/validate_output_contract.py \
  outputs/smoke_new_baseline/DynaPipe
```

## 3. §5.2 / Table 3 扩展

```bash
N_TRACES=30 SEED_START=42 LOG_LEVEL=summary_only \
  bash gpt_new_baseline_code/scripts/run_exp1_overall_with_new_baselines.sh
```

该脚本修正了原 Exp1 中的有效模型键，使用：

```text
gemma3-4b   -> outputs/exp1_overall_small/DynaPipe/
llama-3-8b -> outputs/exp1_overall_medium/DynaPipe/
qwen3-14b  -> outputs/exp1_overall_large/DynaPipe/
```

每个目录中：

- `config_snapshot.json` 保存合并后的 DACI 配置和 `new_baselines.dynapipe` 参数；
- `experiment_meta.json` 保存 adapter mode、upstream revision、可比性警告和 seed 范围；
- `summary.csv` 为一个 DynaPipe 聚合行，包含均值、标准差和 `n_traces`；
- `traces/DynaPipe_seed<N>.jsonl` 保存每条 trace 的窗口日志；
- `DynaPipe.new_baseline.log` 是 shell 运行日志，位于父实验目录。

脚本最后生成 `outputs/exp1_overall_table3_with_new_baselines.csv`。它是 long-form CSV，不是论文排版表；作图/排版应只读取已验证的可比行。

如果已有 DACI/SDA/RT/FM 原始输出不在默认 `outputs/`，可设置：

```bash
OUTPUT_ROOT=/absolute/path/to/outputs N_TRACES=30 \
  bash gpt_new_baseline_code/scripts/run_exp1_overall_with_new_baselines.sh
```

## 4. §5.5 / Figure 5 扩展

先确保已有 DACI 与 SDA 的 G sweep 数据也以相同 `G_hat`、模型、seed 和 trace 数运行，否则相对差值不可比较。然后执行：

```bash
N_TRACES=30 SEED_START=42 MODEL_NAME=qwen3-14b LOG_LEVEL=summary_only \
  bash gpt_new_baseline_code/scripts/run_exp5_g_sweep_with_new_baselines.sh
```

它写入五个目录：

```text
outputs/exp5_scalability/G_sweep/G_5000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_10000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_15000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_20000_DynaPipe/
outputs/exp5_scalability/G_sweep/G_40000_DynaPipe/
```

最后生成 `outputs/exp5_scalability/G_sweep/figure5_with_new_baselines.csv`。其中每行保留原始 TTLT、P99 TPOT、开销和 `n_traces`；若同一目录已有 DACI 行，会计算：

```text
daci_relative_ttlt_change_pct = 100 * (baseline_TTLT - DACI_TTLT) / baseline_TTLT
```

正值表示 DACI 的 TTLT 更低。这个 CSV 是 Figure 5 扩展的后处理输入，而不是直接的论文图。

## 5. 使用外部结果适配器

当未来为 FlexPipe 或 Seesaw 建立了真正匹配的实验环境时，先准备一份每行一条 trace 的 JSONL 和一份 attestation 文档。然后：

```bash
python gpt_new_baseline_code/external_result_adapter.py \
  --baseline <FlexPipe-or-Seesaw> \
  --input-jsonl /absolute/path/aligned_traces.jsonl \
  --attestation /absolute/path/measurement_attestation.md \
  --output-dir outputs/external_baselines \
  --run-id <run_id> \
  --log-level full
```

接着运行：

```bash
python gpt_new_baseline_code/scripts/validate_output_contract.py \
  outputs/external_baselines/<run_id>
```

只有当 attestation 证明 metric、模型、硬件、请求定义和随机重复方式都与目标实验相符时，才能将这类数据放进共同的 Table/Figure。当前提供的 FlexPipe 与 Seesaw 材料不满足这一条件。

## 6. 获批后的迁入顺序

当前所有文件故意留在 `gpt_new_baseline_code`。若决定正式接入，建议按以下顺序迁移，并在每一步重新跑 smoke test：

1. 将 `baseline_adapters/dynapipe.py` 移到 `src/schemes/dynapipe.py`。
2. 在 `src/schemes/schemes.py` 导入并注册 `DynaPipeScheme`，让正式 `run.py --schemes DynaPipe` 可用。
3. 将两个 Bash 脚本移到各自的 `experiments/exp1_overall/`、`experiments/exp5_scalability/` 目录，并决定是否合并到原脚本。不要修改 `experiments/run_all.sh`，除非同时修复其已知失效引用。
4. 将两个 aggregation 脚本移到对应实验目录或扩展 `experiments/aggregate.py`。
5. 把 docs 中的限制说明保留到 `gpt_analysis/` 或正式实验 README，避免未来把 DynaPipe exploratory 结果误用为论文复现。
6. 只有在新增了多请求 batch/pipeline cost model 后，才考虑把 FlexPipe 或 Seesaw 变成 `src/schemes` 中的可比较方案。
