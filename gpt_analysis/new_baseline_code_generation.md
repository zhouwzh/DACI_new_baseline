你现在要把一个或多个新的 baseline 集成到 DACI 仓库中，并让它们能够参与 DACI 当前论文实验。

背景材料：

- 目标论文：MobiHoc_2026_PPINf.pdf
- 目标仓库：DACI\DACI-Drift-Aware-Collaborative-LLM-Inference
- 仓库主入口：
  - run.py：统一实验入口
  - src/simulator.py：单条 trace 的模拟执行
  - src/schemes/schemes.py：SDA / RT / FM / DACI / OR
  - src/metrics.py：summary.csv、trace jsonl 输出
  - experiments/：论文实验脚本
- 你需要实现的涉及其他baselin的论文实验映射：
  - Table 3 / §5.2：experiments/exp1_overall/run.sh
  - Figure 5 / §5.5：experiments/exp5_scalability/G_sweep.sh
- 注意：当前 repo 和论文存在不一致，不能盲信脚本或 README；必须同时读论文和代码，先建立“论文实验项 -> 代码脚本 -> 输出数据文件”的映射，再动手改。

我会提供给你：

- 一个或多个新 baseline 的 paper
- 对应 baseline 的代码仓库
- 这些baseline说明会保存在 DACI\DACI-Drift-Aware-Collaborative-LLM-Inference\gpt_analysis\new_baseline\new_baseline.md，其中包括每个baseline的名称，paper pdf 的相对路径， code GitHub repo的link或者代码路径。

你的任务：

1. 先完整阅读 DACI 论文和 DACI 仓库，确认当前实验流程、输出格式、已有 baseline 的实现方式，以及 repo/paper 不一致之处。
2. 再阅读新 baseline 的 paper 和代码，判断它应该如何被接入 DACI 实验框架,能够运行DACI中5.2和5.5中的实验，产生类似可可以用于制作Table3和Figure5的实验数据，。
3. 直接产出可运行代码，不要只给高层方案。
4. 保持 DACI 现有输出格式兼容：
   - outputs/`<experiment>`/<run_id>/config_snapshot.json
   - outputs/`<experiment>`/<run_id>/experiment_meta.json
   - outputs/`<experiment>`/<run_id>/summary.csv
   - outputs/`<experiment>`/<run_id>/traces/*.jsonl
   - 如有需要，支持 full log 的 *_devices.jsonl 和 *_tokens.jsonl
5. 生成一个 markdown 文件为每个新增 baseline 说明：
   - 新增或修改了哪些文件
   - 每个文件的作用
   - 这些文件应放在仓库的哪个位置
   - 如何运行新的实验脚本
   - 会生成什么数据文件
   - 这些数据分别对应当前论文里的哪个表/图/实验
   - 数据如何存储、如何聚合、如何后处理
6. 另外生成一个 markdown 文件说明：
   - 与原论文/原仓库不一致之处
   - 任何必要的假设、近似或限制
7. 如果新 baseline 不能原样接入 DACI 模拟器，要明确说明是：
   - 通过“语义等价的模拟适配”接入
   - 还是通过“调用原 baseline 仓库 + 结果适配器”接入
     并说明这样做会影响哪些实验解释。
8. 如果某个论文实验不能被该 baseline 合法支持，不要硬做；请明确指出缺口，并给出最接近且可复现的替代实验。
9. 最终交付必须包含：
   - 修改好的代码
   - 可运行的脚本
   - 写好的说明文件
   - 一个清晰的“如何从零跑出新 baseline 实验数据”的步骤说明

额外要求：

- 不要默认当前 experiments/run_all.sh 是正确的。
- 不要默认 README 与代码一致。
- 优先保持 baseline 原始语义，不要为了接入而把它改成 DACI。
- 输出必须以“当前 paper 的实验复现/扩展”为目标，而不是只做单独 benchmark。
- 不要改变已有的代码和脚本，新生成的代码统一放在DACI\DACI-Drift-Aware-Collaborative-LLM-Inference\gpt_new_baseline_code中，后续批准之后，在根据你生的md文件中的指示，移动到合适位置。
