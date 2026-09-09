# 代码审查报告（2026-09-09，消融/跨模型运行期间）

- **审查方式**：通读可执行代码（`src/agentic_rag/**`、`experiments/*.py`、`experiments/*.sh`、`kidnaprag/ReAct/ReAct/attack_react.py` 及检索适配层），并用结果 JSON 独立核验代码疑点是否已落到数据上（KB 块数、replay 计数、clean 分母来源、qid↔idx 映射、字段口径）。
- **不重复** `research/issues/known_issues.md` 已关闭/已记录的条目（A1/A2/B2/C1–C4/D1–D4 等）。
- 严重度沿用 known_issues.md 的约定：**[高]** 影响结论有效性；**[中]** 口径/表述与数据不符或需披露；**[低]** 工程隐患、显示级。
- 文中标识符（`topk4`/`topk16`、`cluster`、`doc-consolidator` 等）保持代码原名；行文用 README 术语表的措辞（受害者检索窗口、poison dose、poison text style）。

---

## 摘要表

| # | 严重度 | 主题 | 核心文件 | 是否需重跑 |
|---|--------|------|----------|------------|
| R1 | 中 | 检索窗口臂（topk4/topk16）的 clean 分母是 k=8 主表基线，且 meta 未记录 top-k 覆盖 | `run_ablation_v2.sh`、`longtail_attack.py` | 可不重跑（补表注）；可选补跑分窗口 clean |
| R2 | 中 | `ablation_summary.md` 的 "clean-EM" 列实为 substring-correct（v2 口径），非官方 EM | `summarize_ablation.py` | 否，改列名/加注 |
| R3 | 中 | eval-after（跨模型）结果字段名是 v1 语义：`n_correct_official`/`em_official` 实存 substring-correct | `longtail_attack.py` | 否，改名或加注 |
| R4 | 中(低) | `load_clean_baseline` 静默回退 headline（可能拿错模型的 clean 分母） | `longtail_attack.py:167-184` | 否，加校验 |
| R5 | 低 | `react_baselines.py` 仅在非 clean 方法前清毒；`--methods` 顺序改变时 clean 会跑在带毒 KB 上 | `react_baselines.py:116-121` | 否 |
| R6 | 低 | `--use-shared-targets` 用 set 迭代决定目标顺序，run 间不可逐位复现 | `longtail_attack.py:375` | 否 |
| R7 | 低 | co-retrieval 仪表固定 k=8，topk 臂的 `true_in_top8` 不是该臂真实窗口状态 | `longtail_attack.py:456,646` | 否 |
| R8 | 低 | 注释/文档遗留："victim = GLM-4-9B"、"restore the default GLM server" | `unified_eval.py`、`react_baselines.py`、两个 driver | 否 |
| R9 | 低 | 剂量表主表行显示 "xlam-2-8b / None" | `summarize_ablation.py:166` | 否 |

**总体结论：主表、dose/style/trigger 臂、跨模型矩阵、react 基线的核心数据链路可信，未发现使结论失效的 bug。** 需要处理的集中在 R1（口径决策）与 R2/R3（命名会误导引用）。

---

## A. 实验有效性 / 口径

### R1 [中] 检索窗口臂（topk4/topk16）的 clean 分母是 k=8 的复用基线

**现象**：`run_ablation_v2.sh:100-108` 的 `run_arm` 对**所有**臂统一传 `--clean-from <主表文件>`。已核验 `results/08_longtail_xlam-2-8b_topk4.json` 与 `..._topk16.json` 的 `clean.answers` 与主表 `08_longtail.json` **逐字节一致**——即这两个臂的 flip 定义实际是：

> "k=8 检索下 clean 答对（substring 口径）→ k=4/16 检索下答错"

**影响**：
- **topk4**：一部分 "flip" 可能只是 4 窗口本身检索不到正确段落（窗口退化），并非毒物翻转知识 → 该臂 flip 率被**高估**。
- **topk16**：方向相反，16 窗口 clean 可能更好，分母未计入 → 效应可能被**低估**。
- 另外**结果文件 meta 完全没有记录 top-k 覆盖**（meta 里只有 `volume: 8`，没有任何 top_k 字段），事后审计无法从文件本身看出该臂 victim 检索窗口被改过。

**说明**：若这是有意设计——"固定分母集合（k=8 clean-correct 集），只变 after 侧窗口"，使 topk4/8/16 三行分子分母可比——是可辩护的协议；vol2/4/6 与 style/trigger 各臂复用 clean 是**有效**的（同模型、同 k=8、纯净 KB），只有 topk 臂的 clean 条件与 after 条件不同。

**建议**（三选一）：
1. 最低成本：`summarize_ablation.py` 检索窗口表加表注，明确"clean 分母为该骨干 k=8 主表基线，flip 含窗口退化成分"。
2. 补跑每窗口 clean（每臂 ~60 题，1/3 臂成本），让 flip 分母与 after 同窗口。
3. 至少把 `--top-k` 写入结果 meta（`longtail_attack.py` meta 段补 `kb_top_k` 字段），保证可审计。

---

## B. 口径命名 / 显示与数据不符

### R2 [中] `ablation_summary.md` 的 "clean-EM" 列实为 substring-correct

**现象**：`summarize_ablation.py:85-89` 用 `unified.correct`（substring）算 `clean_em`，`:100-101` 的 `after_em` 同理；表头（`:143-145`）却写 "clean-EM"，文档头写 "flip(knowledge) = clean-EM-correct"。

**事实核对**：xlam 主表行显示 clean-EM 27/60，而该 run 的官方 EM（`clean.em_official`）是 **6/60**。数字本身正是 v2 协议要的 flip 分母（**正确**），但列名会让人按官方 EM 引用，相差 4.5 倍。

**现状**：README 主表用手写 "clean correct"（措辞正确），风险在自动生成的 `results/ablation_summary.md` 被直接抄进论文/附录。

**建议**：列名改为 "clean-correct (substring)" 或表注注明口径；文档头同步改。`raw_all_runs` 的 EM0/EM1 列同理。

### R3 [中] eval-after（跨模型）结果字段名是 v1 语义

**现象**：`longtail_attack.py:517-521` 定义 `_em = unified.correct`（substring），`:544` 存为 `"n_correct_official": n_clean`，`:551` 存为 `"em_official": after_em`——两处存的都是 **substring-correct 计数**，不是官方 EM。

**下游核对**：`summarize_ablation.build_cross_model`（`:268`）取 `n_correct_official` 当 flip 分母——**数值口径恰好正确**（v2 flip 分母就是 substring-correct），已产出的 6 个 pair 数值全部一致。但字段名是 v1 语义，后续任何按名字消费这些字段的分析都会读错。

**建议**：改名为 `n_correct_substring` / `after_correct_substring`（下游同步），或在文件头/README 注明这两个字段在 eval-after 文件中的实际口径。改名只对**新跑**生效，历史 6 个 pair 文件建议在分析脚本侧按口径读。

---

## C. 工程 / 潜在隐患

### R4 [中(低)] `load_clean_baseline` 静默回退 headline

`longtail_attack.py:167-184`：victim 主表文件 `results/08_longtail_<model>.json` 缺失时，回退到 headline `08_longtail.json`（xlam-2-8b 的 clean）。victim 是 xlam 时回退**恰好正确**；但若未来新增 victim 而其主表未跑，`eval-after` 会**静默用错模型的 clean 分母**（不报错；`meta.clean_source` 可事后审计，但 run 本身无告警）。

**已核验**：当前 6 个已产出 pair 的 `meta.clean_source` 全部指向 victim 自己的主表文件，没有触发回退。driver（`run_cross_model.sh:103` / split 版 `:119`）只检查 **attacker** 主表存在，不检查 victim 主表。

**建议**：`load_clean_baseline` 内加一行校验（headline 分支要求 victim == xlam-2-8b，否则 `SystemExit`）；或 split/cross driver 在 eval-after 前检查 victim 主表文件存在。

### R5 [低] react_baselines 清毒顺序依赖 methods 列表顺序

`react_baselines.py:116-121`：循环内只在**非 clean** 方法前 `delete_poison()`；clean 前的清毒只有脚本入口的一次（`:115`）。默认 `METHODS` 顺序 clean 在前 → 安全。若有人以 `--methods naive,clean` 之类顺序重跑，clean 会跑在上一方法遗留的毒上——正是 2026-09-09 已修过的"clean 基线被毒污染"问题的另一半路径。

**建议**：clean 分支前也 `store.delete_poison()` 一次（幂等，成本为零）。

### R6 [低] `--use-shared-targets` 目标顺序由 set 迭代决定

`longtail_attack.py:375`：`targets = [by_qid[q] for q in shared_qids]`，`shared_qids` 是 set，Python 字符串哈希每进程随机 → 目标处理顺序、`target_records` JSON 键序逐 run 不同。

**影响**：不影响任何指标（60 题全量处理、请求相互独立、victim 采样温度 0），仅日志顺序与严格逐位复现受影响。

**建议**：改为按 corpus 顺序取（`[by_qid[q] for q in questions if q.qid in shared_qids]` 后过滤）或 `sorted(shared_qids)`，一行修复，对后续 run 生效。

---

## D. 显示 / 注释级

### R7 [低] co-retrieval 仪表固定 k=8

`longtail_attack.py:456,646`：`co_retrieval(..., k=8)` 硬编码。topk4/topk16 臂存下的 `true_in_top8`/`consensus_cov` 既不是 k=4 也不是 k=16 窗口的真实状态。纯诊断列，不进主指标；展示时注意口径即可（topk16 下毒块可能进 16 窗口但 8 窗口看不到）。

### R8 [低] 注释/文档遗留（v2 xlam-2-8b 时代未更新）

- `unified_eval.py:43-45`：注释 "baselines re-run on GLM-4-9B"（实际 xlam-2-8b）。
- `react_baselines.py:8` docstring："victim model = GLM-4-9B"（同上；`:32` `MODEL_NAME` 是对的）。
- `run_main_table_v2.sh:7,164`、`run_cross_model.sh:141`："restore the default GLM server"（实际 xlam-2-8b）。

### R9 [低] 剂量表主表行显示 "xlam-2-8b / None"

`summarize_ablation.py:166`：`label = f"{m} / {arm}"`，arm=None 的主表行直接打印 "None"。应为 "cluster (volume 8, 主表)" 之类。

---

## 已核验无问题的点（本次审查的"阴性结果"）

1. **指标口径三处一致**：`longtail_attack.py`（run 内）、`unified_eval.py`、`summarize_ablation.py` 的 flip/collapse/ASR 定义与分母完全一致——flip = clean-substring-correct → 非空错答；collapse 单列不计 flip；ASR 一律对"实际注入"的 per-target wrong（`store` 元数据 / frozen records / 共享文件三路同源）。
2. **跨模型 replay 无损**：`data/chroma` 与 `data/chroma_xsm141` 均 67053 块（副本完整）；已产出 pair 的 `_inj` 文件 n_poison（472/472/460/472）与 attacker 主表 `poison_writes` 逐一对上；6 个 eval-after 的 `clean_source` 全部正确指向 victim 主表，无一触发 R4 回退。
3. **react 侧协议映射**：`hotpotqa_qid_to_idx.json` 的 60 个 idx 在 kidnaprag `hotpot_dev_v1_simplified.json` 上取出的题面与共享 `hotpotqa.json` **60/60 一致**（该文件无 qid 字段，映射纯按位置，但内容验证无误）；react clean n=60、EM=13、empty=20，与 README 披露的 "15 substring-correct / 20 empty" 自洽。react 与 langgraph victim 同 KB、同 top-8（`E5WikiEnv(retriever, k=8)`）。
4. **变体隔离与协议断言**：变体间 `delete_poison` + `delete_writer("doc-consolidator")`、eval-after 保留毒不清理、共享 wrong 缺失即 assert、共享目标缺失即 SystemExit，均按协议执行。
5. **触发臂**：`trig_always`（p=1.0）fired 60/60；主表 keyword fired 59/60（known issue A2，已关闭非问题）。`SemanticTrigger` 对 task 文本（含模板前后缀）与裸问题嵌入的阈值行为一致于臂定义。
6. **并发安全**：8-worker ask 各自独立 compiled graph（无共享可变状态，`ChatOpenAI` 线程安全）；payload 全部逐 target 显式传参，`last_selection_stats` 仅在串行 install 相位内读写；dose/style/trigger 臂的 refill+MMR 保证请求体积（`refill_capped` 如实落盘）。
7. **结果落盘**：`save_results` 的 run 目录 + 根镜像机制在两个并行 split 实例下无名字冲突（run_id、结果名按 pair 唯一）。
8. **主表 meta**：headline `08_longtail.json` 确为 v2 xlam-2-8b run（2026-09-08 06:25:42，`use_shared_targets: true`）。
9. **统一对比表新鲜度**：`13_unified_comparison.json`（06:41）在 react 基线（06:40）之后重新生成，README 引用数字与之一致。

---

## 运行期间注意事项（写于两个 split 实例运行时）

- **本地 split 实例当前使用共享 `data/chroma`**：其跑完之前，不要并行启动任何也写 `data/chroma` 的任务（重跑 react baselines、重跑主表/消融臂、re-inject），否则彼此 `delete_poison`/`delete_writer` 会互相清掉对方的毒或写入。
- 两个 split 实例一 local 一 remote 的组合是安全的；**不要同时跑两个 local 实例**（都会 `pkill -f "vllm serve"`，互相杀对方的 server）。remote 实例不碰 server 进程，可与此刻任何占用本地 GPU 的 driver 并行（KB 用独立的 `chroma_xsm141`）。
- `data/` 下另有 `chroma_gpt-oss-20b`、`chroma_gpt-oss-20b_b` 两个旧目录及各 KB 内的 `hotpot_tmp` 集合，当前无 driver 引用，属遗留产物，可另行归档清理。

---

## 建议行动顺序

1. **（措辞级，写稿前必做）** R2 / R3：改列名与字段名或在生成处加口径注；`raw_all_runs` 同步。
2. **（决策）** R1：topk 臂口径——加表注（最小）或补分窗口 clean（更干净）；并把 `--top-k` 写进 meta（`longtail_attack.py` meta 补 `kb_top_k`，对后续 run 生效）。
3. **（一行加固）** R4 headline 回退校验；R5 clean 前清毒；R6 目标顺序确定化。
4. **（顺手）** R7 表注、R8 注释更新、R9 label 修复。
5. 本清单可并入 `research/issues/known_issues.md`（编号 R1–R9），或保持独立文件按日期归档。
