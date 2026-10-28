# results/ 目录说明（2026-10-27 起）

**当前有效的只有 `dc_*` 这一套（新协议，每题固定 100 道，分母恒为 100）。**
下面的"历史留档"一节说明哪些文件被删了、为什么。

| 内容 | 文件 / 目录 |
|---|---|
| 新协议单次运行结果（每次 100 题） | `dc_<条件>_<模型>[_musique].json` |
| 模型判定结果 | 同名 + `_judged.json` |
| **"我们"方法的正式结果** | 三个早期模型是 `dc_ours_noleak_<模型>_eval.json`（+`_judged`），**不是** `dc_ours_<模型>.json`。前者更晚、更正式，`summarize_dc.py` 也是优先读它；后者是更早的一轮，留作对比，别再"清理"前者 |
| 新协议自动汇总（脚本 `experiments/summarize_dc.py` 生成） | `dc_summary.md`（HotpotQA）/ `dc_summary_musique.md`（MuSiQue） |
| **消融的毒文本 + 评测**（10 个设置 × 4 模型 × 2 数据集） | `dc_ours_<设置>_<模型>[_musique].json`（毒文本）+ 同名 `_eval.json` / `_eval_judged.json`；设置在 `vol2/vol4/vol6/embed_hybrid/mono/nodiv/greedy/trig_semantic/topk4/topk16` |
| 剂量臂的来源 | `vol2/4/6` **不再单独生成**，是从 `dc_ours_noleak_<模型>[_musique].json`（主表 8 条）按取样顺序取前 k 条，见 `experiments/derive_dose.py` |
| 检索窗口两行的"攻击前"对照（同窗口 clean） | `dc_clean_clean_<模型>[_musique]_topk{4,16}.json`（+ `_judged`） |
| 消融汇总（`experiments/summarize_ablation_dc.py` 生成） | `ablation_summary.md`（含跨模型矩阵一节） |
| **跨模型迁移**（甲写毒、乙受害；12 个换人组合 × 2 数据集） | `dc_ours_xsm_<攻击者>_to_<受害者>[_musique]_eval.json`（+`_judged`）；汇总 `cross_model_summary.md`，脚本 `experiments/summarize_cross_model_dc.py` |
| 离线重算的触发清单 | `dc_trig_{semantic,keyword}_{hotpot,musique}_qids.json` |
| 新协议论文表（人工整理，含翻转/说不出构成） | `dc_final_report.md` |
| 基础环境必须共享的实测依据（三臂对照） | `benign_ablation_gpt-oss-20b.json` |

条件 `dc_<条件>_...` 的取值：`clean`（`dc_clean_clean_*`）、`ours`、
`ours_xsm`（跨模型，必须写明 `--attacker` 和 `--poison-file`）、`poisonedrag`、
`agentpoison`、`corruptrag_as`、`corruptrag_ak`。`_musique` 后缀 = MuSiQue 数据集，
没有后缀的 = HotpotQA。

## 历史留档

**旧协议的结果已全部删除（2026-10-27，确认不可用后）。** 现在只剩两类历史文件：

| 内容 | 文件 | 说明 |
|---|---|---|
| 基础环境必须共享的实测依据 | `benign_ablation_gpt-oss-20b.json` | 三臂对照（只放毒 30/38 → 加正常写回 21/38），说明这一层必须固定成共享底座，**必须留** |
| 复现性测试 | `determinism/` | llama 同输入两遍 clean 60 vs 56、gpt-oss 93/100 一致；剔除 llama 的证据 |

删掉的是：旧协议（60 题）的全部 `08_longtail*` 主表与 11 个消融设置、4 个基线的 60 题版、
跨模型矩阵、旧汇总 md，以及 llama-3.1-8b 的全部结果（34 个文件）。
理由：分母不同（27/31/38/25 互不可比）、各方法的基础环境不一致（旧流程只有我们的方法有正常写回）、
qwen3-8b 下线 + llama 剔除 —— 协议和模型都换了，这些数字进不了任何表格。
结论本身仍完整记录在 `research/monitor/experiment_ledger.md` 与
`research/issues/known_issues.md`，不依赖被删的文件。

汇总进度与下一步见 `research/monitor/PROGRESS.md`。

## 注意

- 单次结果文件在写回时是**全量覆盖**（同名 `dc_<条件>_<模型>.json` 会被本次运行覆盖），
  每次运行同时会在 `runs/<run_id>/` 留一份，`runs/` 才是不会丢的历史。
- 基线毒文本与受害者模型无关，可以复制复用；但复制来的文件里**带着上一个模型的答案**，
  必须先剥掉 `answers` / `summary` 再当新模型的输入用（2026-10-27 出过一次错）。
