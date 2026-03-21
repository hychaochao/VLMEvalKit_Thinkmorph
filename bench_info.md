### Benchmark Table

| No. | Benchmark            | Category   | Importance | 当前实现进度 | 原始仓库支持情况 | 官方VLMEvalKit实现 | 仓库对应名称/说明 | 官方实现/代码 |
|-----|----------------------|------------|------------|--------------|------------------|--------------------|-------------------|---------------|
| 1   | MathCanvas-in (3k)   | Math       | 0          | ✓ | 暂不支持（仅 `MathCanvas-Bench`） | 暂不支持（仅 `MathCanvas-Bench`） | 当前仓库新增兼容名称 `MathCanvas-in`、`MathCanvas-in (3k)`，实际映射到官方公开的 `MathCanvas-Bench` 3K benchmark | [MathCanvas](https://github.com/shiwk24/MathCanvas) |
| 2   | MathVision           | Math       | 0          | ✓ | 支持 | 支持 | `MathVision`、`MathVision_MINI` | [MATH-V / MathVision](https://github.com/mathllm/MATH-V) |
| 3   | MathVista            | Math       | 0          | ✓ | 暂不支持（仅 `MathVista_MINI`） | 暂不支持（仅 `MathVista_MINI`） | 当前仓库新增完整 `MathVista` 本地官方数据接入，同时保留 `MathVista_MINI` | [MathVista](https://github.com/lupantech/MathVista) |
| 4   | VSP / VSP-in         | Spatial    | 0          |  | 暂不支持（仅 `VSP_maze_task_main_original`） | 暂不支持 | 当前仓库新增 `VSP`、`VSP-in` 命名兼容，但底层仍对应官方已公开的 `VSP_maze_task_main_original` maze split；官方仓库暂未公开完整 benchmark 数据 | [VisualPlanning](https://github.com/yix8/VisualPlanning) |
| 5   | MindCube             | Spatial    | 0          | ✓ | 暂不支持 | 支持 | 当前仓库新增 `MindCube`、`MindCubeBench`、`MindCubeBench_raw_qa`、`MindCubeBench_tiny_raw_qa` | [MindCube](https://github.com/mll-lab-nu/MindCube) |
| 6   | MMSI                 | Spatial    | 0          | ✓ | 暂不支持 | 支持 | 当前仓库新增 `MMSI`、`MMSIBench`、`MMSIBench_wo_circular`；`MMSI` 默认映射官方 non-circular 评测集 | [MMSI-Bench](https://github.com/OpenRobotLab/MMSI-Bench) |
| 7   | EMMA                 | General    | 0          | ✓ | 支持 | 支持 | `EMMA`、`EMMA_COT` | [EMMA](https://github.com/EMMA-Bench/EMMA) |
| 8   | V*StarBench          | Perception | 0          | ✓ | 支持 | 支持 | `VStarBench` | [V* / VStarBench](https://github.com/penghao-wu/vstar) |
| 9   | HRBench              | Perception | 0          | ✓ | 支持 | 支持 | `HRBench4K`、`HRBench8K` | [HR-Bench](https://github.com/DreamMr/HR-Bench) |
| 10  | CVBench              | Perception | 0          | ✓ | 支持 | 支持 | `CV-Bench-2D`、`CV-Bench-3D` | [Cambrian / CV-Bench](https://github.com/cambrian-mllm/cambrian) |
| 11  | MMVP                 | Perception | 0          | ✓ | 支持 | 支持 | `MMVP` | [MMVP](https://github.com/tsb0601/MMVP) |
| 12  | BLINK                | Perception | 0          | ✓ | 支持 | 支持 | `BLINK`、`BLINK_circular`，另含 `BLINK_Jigsaw` | [BLINK](https://github.com/zeyofu/BLINK_Benchmark) |
| 13  | MMMU                 | General    | 0          | ✓ | 支持 | 支持 | `MMMU_DEV_VAL`、`MMMU_TEST`，另含 `MMMU_Pro_*` | [MMMU](https://github.com/MMMU-Benchmark/MMMU) |
| 14  | ScienceQA            | General    | 0          | ✓ | 支持 | 支持 | `ScienceQA_VAL`、`ScienceQA_TEST` | [ScienceQA](https://github.com/lupantech/ScienceQA) |
| 15  | ARC-AGI              | General    | 0          | ✓ | 暂不支持 | 暂不支持 | 当前仓库新增 `ARC-AGI`：基于官方 `ARC-AGI-2` public evaluation JSON 渲染任务图，要求模型输出 JSON grid，并做 exact-match task-level eval | [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) |

### 本地已跑结果汇总

当前已检查的结果目录：`results/Qwen2.5-VL-7B-Instruct/T20260321_Gbfc358a1`

| Benchmark | 本地结果 | 结果文件 | 备注 |
|-----------|----------|----------|------|
| MathCanvas-in (3k) | 已生成预测文件，暂未见自动汇总分数 | `Qwen2.5-VL-7B-Instruct_MathCanvas-in.xlsx` | 表头仅含 `question/answer/prediction`，当前没有配套 `acc.csv/json` |
| MathVision | 已生成预测文件，暂未见自动汇总分数 | `Qwen2.5-VL-7B-Instruct_MathVision.xlsx` | 当前没有配套 `acc.csv/json` |
| VSP | 已生成预测文件，暂未见自动汇总分数 | `Qwen2.5-VL-7B-Instruct_VSP.xlsx` | 当前没有配套 `acc.csv/json` |
| VSP-in | 已生成预测文件，暂未见自动汇总分数 | `Qwen2.5-VL-7B-Instruct_VSP-in.xlsx` | 当前没有配套 `acc.csv/json` |
| EMMA | 已生成预测文件，暂未见自动汇总分数 | `Qwen2.5-VL-7B-Instruct_EMMA.xlsx` | 当前没有配套 `acc.csv/json` |
| V*StarBench | `Overall = 76.44%` | `Qwen2.5-VL-7B-Instruct_VStarBench_acc.csv` | 子项：`direct_attributes = 77.39%`，`relative_position = 75.00%` |
| CVBench | `2D = 74.90%`，`3D = 73.58%` | `Qwen2.5-VL-7B-Instruct_CV-Bench-2D_acc.csv`，`Qwen2.5-VL-7B-Instruct_CV-Bench-3D_acc.csv` | 2D 子项：`COCO = 81.86%`，`ADE20K = 67.93%`；3D 子项：`Depth = 70.33%`，`Distance = 76.83%` |
| MMVP | `Overall = 77.33%` | `Qwen2.5-VL-7B-Instruct_MMVP_acc.csv` | 已有自动汇总 |
| ScienceQA | `VAL = 71.58%`，`TEST = 72.78%` | `Qwen2.5-VL-7B-Instruct_ScienceQA_VAL_acc.csv`，`Qwen2.5-VL-7B-Instruct_ScienceQA_TEST_acc.csv` | 已有自动汇总 |
| ARC-AGI | `overall_task_accuracy = 0.00%`（`0 / 120` tasks） | `Qwen2.5-VL-7B-Instruct_ARC-AGI_score.json` | 任务级 exact-match 评测 |

### Qwen2.5-VL-7B 公开结果对比

| Benchmark | 本地结果 | 公开/官方结果 | 差值 | 来源 | 备注 |
|-----------|----------|---------------|------|------|------|
| V*StarBench | `76.44%` | `76.44%` | `0.00` | [ThinkMorph README](https://github.com/ThinkMorph/ThinkMorph) | 公开表中的 `VStar` 与当前本地 `VStarBench` 数值完全一致 |
| MMVP | `77.33%` | `77.33%` | `0.00` | [ThinkMorph README](https://github.com/ThinkMorph/ThinkMorph) | 数值完全一致 |
| CVBench | `2D = 74.90%`，`3D = 73.58%` | `CV-Bench = 75.20%` | `约 -0.96` | [ThinkMorph README](https://github.com/ThinkMorph/ThinkMorph) | 公开表给的是单一 `CV-Bench` 总分；本地当前输出是 `2D/3D` split。这里的差值按 `(74.90 + 73.58) / 2 = 74.24` 做简单近似，仅供参考 |
| ScienceQA | `TEST = 72.78%`，`VAL = 71.58%` | `IMG Score = 88.6` | `不直接可比` | [Wizwand ScienceQA 榜单](https://www.wizwand.com/task/science-question-answering) | 该公开页面给的是第三方聚合的 `ScienceQA / IMG Score`，不是当前 VLMEvalKit 结果文件里的 `Overall` 指标，口径不一致 |
| ARC-AGI | `0.00%` | `未找到可靠公开结果` | `-` | - | 检索公开网页时，暂未找到可复核的 `Qwen2.5-VL-7B` ARC-AGI 分数 |

### 结果解读

- 当前已经自动产出汇总分数的 benchmark 里，`MMVP (77.33%)` 和 `V*StarBench (76.44%)` 是表现最好的两项。
- `CVBench` 当前拆成 `2D = 74.90%` 与 `3D = 73.58%` 两个 split；如果只做粗略平均，大约是 `74.24%`，与公开 ThinkMorph 表里的 `75.20` 基本接近。
- `ScienceQA` 当前 `TEST = 72.78%`、`VAL = 71.58%`，说明模型在常规图文科学问答上是能稳定工作的，但相比官方/公开聚合榜单里的高分结果，仍有明显差距。
- `MathCanvas-in`、`MathVision`、`VSP`、`VSP-in`、`EMMA` 目前已经生成预测文件，但当前结果目录下还没有自动评测后生成的 `acc.csv/json`，所以暂时只能算“已跑完预测，未完成汇总打分”。
- 对于“官网结果”这件事，要区分来源：`Qwen` 官方模型卡公开了 `MMMU / MathVista / MathVision` 等通用 benchmark 分数，但没有公开 `V*StarBench / MMVP / CV-Bench / ScienceQA / ARC-AGI` 的完整同口径结果；因此上表里这些重合项主要参考的是 `ThinkMorph` 公开结果表，而不是 `Qwen` 官方模型卡。

### ARC-AGI 说明

- `ARC-AGI` 测的是抽象规则归纳与组合泛化。每个任务会给出若干组 `train input -> train output` 网格示例，以及一个或多个 `test input`；模型需要自己归纳变换规则，并输出对应的 `test output` 网格。
- 当前仓库里的实现见 `vlmeval/dataset/arc_agi.py`。它会把每个任务渲染成一张图，把训练对和测试输入都画出来，然后要求模型“只返回 JSON”，输出格式必须是 `[[...]]` 或 `{\"test_1\": [[...]], ...}` 这种 grid 结构。
- 评分口径非常严格：不是多选题 accuracy，也不是 token-level 相似度，而是 `task-level exact match`。也就是一个任务里所有测试网格都要和标准答案完全一致，才记 `1` 分，否则就是 `0` 分。
- 本次结果里 `overall_task_accuracy = 0.00%`，对应 `0 / 120` 个 public evaluation tasks 全部 miss。这不是评测脚本挂了，因为结果文件里 `120` 个任务都有预测，且其中 `111` 个任务的输出还能被成功解析成合法 grid/json；只是解析后的结果没有任何一个任务与标准答案完全一致。
- 因此，这个 `0 分` 更像是“当前 prompt + 直接单次生成 JSON grid”的解题方式对 ARC-AGI 基本无效，而不是简单的格式错误。格式问题确实存在一部分，但不是主因。
- 这个结果是偏低的，但在 ARC-AGI 这种 benchmark 上并不反常。它本来就是故意设计来测系统性泛化和组合规则发现的；对没有专门 search / program synthesis / self-refine 流程的通用 7B VLM，直接 `0 / 120` 是完全可能出现的。
