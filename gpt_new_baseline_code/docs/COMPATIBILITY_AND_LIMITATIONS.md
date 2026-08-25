# 兼容性、不一致与限制

## 结论优先

当前 DACI 仓库是数值模拟器，论文 §5 的目标是异构 edge cluster 上单个长请求在 runtime drift 下的延迟控制。DynaPipe、FlexPipe 和 Seesaw 的原始问题设定均与此不完全相同。因而本交付将“输出格式兼容”和“论文结论可比”严格分开：前者可以实现，后者只有在相同工作负载和状态模型下才成立。

| 方法 | 原始目标 | 与 DACI §5.2/§5.5 的关系 | 交付决策 |
| --- | --- | --- | --- |
| DynaPipe | 多请求 pipeline serving 中，消除尾 sampling 的 stage bubble。 | 有 layer runtime redistribution 这一局部语义重合，但原始收益依赖 batch concurrency；DACI 是单请求。 | 语义等价模拟适配，默认忠实执行原始 `<5 decode` guard，产出静态 fallback。exploratory batch 模式不可用于当前论文结论。 |
| FlexPipe | Serverless 多请求、资源碎片、GPU 弹性扩缩与 pipeline granularity。 | 缺 arrival/batch/resource allocator；给定链接不含其可执行系统。 | 不接入 simulator。仅提供外部结果格式适配器。 |
| Seesaw | 离线 throughput，通过 prefill/decode 间 TP/PP re-sharding 提升吞吐。 | metric、状态和硬件/模型范围都不匹配 DACI 单请求 TTLT。 | 不接入 simulator。仅提供外部结果格式适配器。 |

## DynaPipe 适配的必要近似

1. **共同的初始部署。** DynaPipe 原论文在同质 A100 pipeline 上工作，并未解决 DACI 的异构 Jetson placement 选择；适配器复用 DACI simulator 的初始 deployment 来构建可行 stage/placement。这个 shared harness choice 已写入 `experiment_meta.json`。
2. **稳定窗口。** 默认 `25` 来自 DynaPipe 论文 §4.1 的实验设置；它可通过 `--dynapipe-stability-windows` 修改，并会保存在配置快照中。
3. **sampling profile。** release 中 Qwen-14B profile 的 sampling predictor 常数会被记录，但 DACI 现有模型中没有等价 sampling cost 标定。因此在默认单请求 guard 下，它不会改变 DACI 计时；在 exploratory batch 模式中它只用于 controller 的候选边界评分。
4. **KV/weight 迁移。** DynaPipe 原系统通过预加载和异步 KV migration 使迁移与计算重叠。DACI 的 `run_trace` 使用其统一 `Omega_reconfig` 计费，不能表达这种 overlap。本交付没有伪造单独的迁移折扣。
5. **pipeline bubble。** DynaPipe 优化的是多 microbatch pipeline 的 bottleneck 时间；DACI `T_decode_window` 为单请求串行 stage cost。因此 exploratory 输出只演示控制选择，不能用作延迟性能的公平比较。

## DACI 论文与当前代码的已有不一致

这些不是本次新增 baseline 引入的，但会影响任何扩展实验的解释：

| 项目 | 论文 / 说明 | 当前代码 | 影响 |
| --- | --- | --- | --- |
| LLaMA 模型名 | 论文文字一处写 LLaMA-3.2-8B，Table 3 是 Llama-3-8B。 | `configs/models.json` 的有效 key 是 `llama-3-8b`，但原 `exp1_overall/run.sh` 使用过时 `llama-3.2-8b`。 | 原 Exp1 medium run 会失败；新增脚本使用有效 key。 |
| trace 数 | 论文 §5.1.5 说默认每个报告数值平均 30 条 drift traces。 | Exp1 默认 5、Exp2 默认 5、Exp3 为 20，若干 sensitivity 脚本为 1。 | 新脚本默认 `N_TRACES=30`；历史输出不能自动视为论文统计。 |
| `lambda` | §5.1.4 写 `lambda=0`；§5.4 又称使用 `lambda=1`。 | `configs/algo.json` 为 `lambda_slack=4.0`。 | 每次运行应保留 `config_snapshot.json`；不要把脚本默认直接说成论文默认。 |
| Exp1 基线集合 | 脚本注释说五种方案。 | 实际数组通常为 SDA/RT/FM/DACI，未含 OR。 | Table 3 扩展以真实存在的目录为准。 |
| Exp2 消融 | 论文有 Full、w/o Freeze、w/o Lazy、w/o Predictor。 | 当前脚本仅启用 `no_predictor`，其他被注释。 | 不能直接声称已复现完整 Figure 4。 |
| 顶层运行器 | `experiments/run_all.sh` 宣称运行所有实验。 | 引用缺失的 `delta_sweep.sh`、`N_sweep.sh`、`L_sweep.sh`，并错误聚合 Exp1 路径。 | 不应作为本次或未来 baseline 的一键入口。 |
| R2 名称 | 注释称 thermal。 | `configs/drift.json` 的 `R2_thermal` 同时启用 workload 与 thermal。 | Regime 结果需按实际 config snapshot 解释。 |

## 如何解释新增数据

- 标准格式只表示文件可被现有 CSV/JSONL 工具读取，不表示方法语义、硬件或工作负载可比。
- DynaPipe 默认结果可用于验证扩展管线、目录、聚合与可审计 metadata；不能宣称为原 DynaPipe 在 DACI 任务上的性能比较。
- 若论文要加入 DynaPipe/FlexPipe/Seesaw 的正式数字，应设计新的共同 benchmark，至少定义多请求 arrival、batch scheduler、吞吐/SLO 指标、TP/PP 状态和相同硬件接口，然后同时重新评估 DACI。
- `external_result_adapter.py` 只在存在人工 attestation 时导入结果。它不会把不一致的指标自动转换成科学可比的 TTLT。
