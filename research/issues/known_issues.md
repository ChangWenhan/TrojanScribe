# AgenticRAG 代码/实验已知问题清单

- 审计日期：2026-09-07（主表 5 骨干 × 2 数据集 + 消融 B/C/D 完成后）
- 审计方式：通读代码 + vLLM 0.15.1 注册表核对 + 逐模型 serving/运行日志 + 用 `unified.normalize` 从结果 JSON 独立重算全部指标
- 严重度定义：
  - **[高]** 影响某个实验结论的有效性或可解释性，需决策（重跑或改表述）
  - **[中]** 文档/表述与数据不符，或需主动向审稿人披露的事实
  - **[低]** 口径显示、工程遗留、潜在隐患，不改变结论但建议清理

---

## 摘要表

| # | 严重度 | 主题 | 核心文件 | 状态 |
|---|--------|------|----------|------|
| A1 | 高 | 方向 C：mono/near_dup 体积未控制（请求 8 块实写 ~3.6/4.7） | `payload.py` | 建议重跑或降级结论 |
| A2 | 中 | 1/60 目标因自动关键词机制系统性不触发 | `trigger.py` | 关闭（非问题） |
| A3 | 中 | phi-4-mini "证据行"归因应为工具协议解析失败而非纯模型能力 | `vllm_phi-4-mini.log` | 已解决（删行） |
| B1 | 高 | README "identical poison volume"（75% vs 45%）建立于未控体积 | `README.md:66-68` | 与 A1 同源 |
| B2 | 中 | README MuSiQue insight "true_in_top8 ≤ 1" 与数据矛盾 | `README.md:93` | 已修复 |
| C1 | 低 | qwen3-4b musique 附录 ASR 分母显示 39/60，主表 39/59 | `summarize_ablation.py` | 已修复 |
| C2 | 低 | co-retrieval 日志把 true_in_top8 硬编码打印 /60（59 目标亦然） | `08_longtail.py:425` | 已修复 |
| C3 | 低 | ask_targets 答案记录不存 trace / n_tool_calls，无法事后审计 victim 是否调用工具 | `08_longtail.py:86-87` | 已修复 |
| C4 | 低 | headline `08_longtail.json` 为旧格式，meta 缺 model/trigger_kind 等字段 | `results/08_longtail.json` | 关闭（接受现状） |
| D1 | 低 | embed_hybrid 候选上限 `n_candidates=6` 结构性 < 8 | `payload.py:150/293` | 与 A1 同源 |
| D2 | 低 | 并发采样与串行去重语义略异 | `payload.py:195-227` | 已修复 |
| D3 | 低 | 两套 n_candidates 配置（kb=12 未用 / attack 默认 6）易混 | `default.yaml`、`attack.py:45` | 已修复 |
| D4 | 低 | 旧机制日志 `results/logs/08_*.log` 与新 `results/runs/<run_id>/` 并存 | `results/logs/` | 已修复 |

---

## A. 实验有效性 / 可解释性

### A1 [高] 方向 C：mono / near_dup 体积未控制

**现象**：方向 C 声称"全部 v8"，但实际每目标写入块数相差近一倍：

| 臂 | 请求块 | 实际块/目标 | 总毒块 | flip | ASR |
|----|--------|------------|--------|------|-----|
| full (cluster) | 8 | ~7.37 | 435 | 15/20 (75%) | 37/60 |
| nodiv | 8 | ~8.00 | 472 | 16/20 (80%) | 36/60 |
| greedy | 8 | ~7.44 | 439 | 14/20 (70%) | 34/60 |
| **mono (authority)** | **8** | **~3.61** | **206/57 目标** | 9/20 (45%) | 20/60 |
| **near_dup (embed_hybrid)** | **8** | **~4.69** | **277/59 目标** | 9/20 (45%) | 26/60 |

**根因**：
- `src/agentic_rag/agents/poisoned/payload.py`：`generate_cluster()` 内 `cands = self._cluster_candidates(p, n=16 if archetypes else 8)`（mono 只生成 16 候选），经 `_filter_candidate` + pairwise 去重门后每目标仅通过 ~3.6 块。
- 同文件 `embed_hybrid` 分支（`if variant == "embed_hybrid"`）走 `self.n_candidates` 生成候选，而 `PayloadGenerator.__init__` 默认 `n_candidates=6`（`payload.py:150`）→ 结构上永远凑不满 8。
- 代码注释（"16 个即可与 v8 相当"）与实际通过率不符。

**影响**："单声音/近重复塌方到 45%"既可能是**缺少多样性**，也可能只是**体积只有一半**——两者混杂，无法归因。nodiv(80%)/greedy(70%) vs full(75%) 是真正同体积、结论干净；问题只在两个"去掉多样性"的臂。

**建议**（二选一）：
1. **重跑**（推荐，约 1 小时/臂）：把 mono 候选数提到 48-64、embed_hybrid 的 n_candidates 提到 ≥16，让两者实写 ≥7 块再比。若 mono 仍 ~45-55% → "多声音多样性是承重"结论坐实；若升到 70%+ → 诚实的结论是"多样性主要解决生成容量而非攻击力"。
2. **降级表述**：方向 C 主结论收敛为"单一声音在去重门下无法撑起方法所需体积（容量论证）"；mono 不再表述为与 v8 同体积对照，mono/near_dup 的 45% 明确标注"实写 ~3.6/4.7 块/目标"。

**涉及文件**：`payload.py`、`results/08_longtail_mono.json`、`results/08_longtail.json`（embed_hybrid variant）、`research/ablation/plan.md`（方向 C insight）、`README.md`、`results/ablation_summary.md`

---

### A2 [中] 1/60 目标因自动关键词机制系统性不触发

**现象**：所有 keyword 触发的主表/消融行 fired 都是 **59/60**。未触发 qid `5a7fa2ad5542994857a76792`（"…a rule that is expressed algebraically…" / 黄金比例相关书）题面**不含任何专有名词 token**，`_auto_keywords` 无法为它产出关键词。

**决策（2026-09-07）**：关闭，非问题。fired 59/60 如实展示；该 qid 在各骨干上
clean-EM 均为 False，不进 flip/ASR 分母，不影响任何比率。不再要求额外披露。
（README 既有 Known limitation 行保留原样，不扩展、不删除。）

---

### A3 [中] phi-4-mini "证据行"归因需精确到"工具协议解析失败"

**现象**：phi-4-mini clean-EM 1/60（HotpotQA）与 0/59（MuSiQue），44/60、54/59 的 clean 答案为**空字符串**。

**根因（补充证据）**：`results/logs/vllm_phi-4-mini.log` 全程出现 **308 次** `phi4mini_tool_parser` 解析失败（`Failed to parse function calls from model output: Expecting ',' delimiter`），发生在带工具的多轮 ReAct 请求上——phi-4-mini 即使配了自定义 `chat_template_tools.jinja`，多轮工具历史中仍频繁产出非法 function JSON。**writer 侧同样走工具却成功写 472 块并 fired 59/60**，说明不是"模型完全不会用工具"，而是 victim 多轮协议的不遵从/解析脆性。

**决策（2026-09-07）**：phi-4-mini **弃用并全量清理**（用户决定，不再使用该模型）：
- 主表删行（4 骨干），正文一句话排除说明；
- 结果文件 2 个、日志 5 个、服务适配模板 `LLMs/Phi-4-mini-instruct/chat_template_tools.jinja`
  全部删除；附录 raw 行随结果文件消失；
- README / ablation_summary 的排除说明已去掉全部数字（1/60、0/59、308 次等），
  指向 ledger 作审计记录；
- `experiment_ledger.md`、`plan.md`、本清单保留 phi 运行史与决策过程（审计依据，
  不删）。

---

## B. 文档/表述与数据不符

### B1 [高] README "identical poison volume" 声明建立在未控体积上

**位置**：`README.md:66-68` —— "cluster doubles the knowledge flip rate (75.0% vs 45.0%) at identical poison volume"。

**事实**：cluster 实写 435 块，embed_hybrid(near_dup) 实写 277 块 —— **不是同体积**。这与 A1 是同一问题在 README 核心声明处的体现，是本项目最需要先处理的表述。

**建议**：与 A1 一起解决。若重跑 near_dup 到 ~8 块：若仍 ~45% → 该声明可恢复；若升到 60%+ → 删除"同体积"说法，改为诚实的容量论证。**在该声明修正前，不要在论文或对外材料引用这句话。**

**涉及文件**：`README.md:66-68`

---

### B2 [中] README MuSiQue insight 与主表数据矛盾

**位置**：`README.md:92-93` —— "On MuSiQue the true paragraph is displaced out of top-8 on essentially all targets (true_in_top8 ≤ 1)"。

**事实**：主表（`results/ablation_summary.md` MuSiQue 节）gpt-oss-20b 行 true_in_top8 = **5/59**，并非全部 ≤1。

**已修复（2026-09-07）**：改为按骨干列数 "qwen3-4b 0, qwen3-8b 1, llama-3.1-8b 0;
gpt-oss-20b is the outlier at 5/59"（phi-4-mini 行已删，不再列出）。

---

## C. 口径 / 显示级

### C1 [低] qwen3-4b musique 附录 ASR 分母 39/60 vs 主表 39/59

**位置**：`experiments/summarize_ablation.py` —— 主表用 frozen-59 restrict（命中 39/59），附录 `raw_all_runs()` 无 restrict 显示 39/60。命中数一致，仅分母口径不同，易被误读为数据不一致。

**建议**：附录行或表注注明 restrict 差异，或附录也标注 frozen-59 分母。

**涉及文件**：`summarize_ablation.py`（`summarize` / `raw_all_runs`）、`results/ablation_summary.md`

---

### C2 [低] co-retrieval 日志硬编码打印 /60

**位置**：`experiments/08_longtail.py:425` —— `f"true_in_top8={true_found}/60 ..."` 在 MuSiQue（59 目标）运行中也打印 /60。JSON 内数据正确，仅日志显示误导。

**建议**：改成 `f"/{len(targets)}"`。

---

### C3 [低] ask_targets 不记录 trace / n_tool_calls

**位置**：`experiments/08_longtail.py:86-87` —— `record()` 只存 `{"pred", "correct"}`，而 `common.answer_questions` 会存 trace。导致事后无法验证 victim 是否真的调用了 `kb_search`（phi-4-mini 的分析只能靠空答案率推断，无法直接看工具调用）。

**建议**：`record` 里补存 `res["n_tool_calls"]`、`res["trace"]`、`res.get("error")`。此改动会改变结果 JSON schema，只对**新跑**生效。

---

### C4 [低] headline `08_longtail.json` 为旧格式，meta 缺字段

**位置**：`results/08_longtail.json` —— 2026-09-06 早期运行，meta 无 `model`/`trigger_kind`/`use_shared_targets`/`target_records` 等新字段（`08_longtail.py:350-367` 的 meta 是后来加的），附录 raw 行中该行 model 列显示为空。

**决策（2026-09-07）**：关闭，接受现状。影响核实仅限此一个文件（现存 18 个结果
文件中唯一 meta 缺失者）；主表按固定路径读取不受影响，per-target `poison_writes`
记录完整，ASR 口径不依赖该 meta。不修改已采信的数据文件，不另行注释；headline
重跑时自然补齐。旧骨干（qwen2.5-3b/7b、llama-3.2-3b）结果文件被删为用户有意
决策（撤出主表，README.md:126 明示），非此条目范围。

---

## D. 工程 / 潜在隐患

### D1 [低] embed_hybrid 候选上限 `n_candidates=6`（与 A1 同源）

`payload.py:150` 默认 `n_candidates=6`；`attack.py:45` 从 attack_cfg 取 `n_candidates` 但 `default.yaml` 的 `attack` 段未定义该项 → 恒为 6。**embed_hybrid / keyword_dense / paraphrase 等一切经 `_gen_i(..., self.n_candidates)` 的变体都 ≤6 候选**，选择上限本身也限制了体积上限。建议与 A1 一并处理。

### D2 [低] 并发采样分支与串行去重语义略异

`payload.py:195-227` —— 串行：`while len<texts n and tries<n*4`，每个失败/重复都消耗一次 try；并发：oversample 2x/轮 + seen set 去重，同样 ≤4n 次尝试。分布同为 temp=0.9 独立请求，但"恰好返回的候选数"在边界条件下可能略异。**不影响本批实验**（qwen3-4b 主表用 8 workers 后 flip 数与历史一致），但复现说明里应写明并发版本仅加速、不改变分布。

### D3 [低] 两套 n_candidates 配置易混

`configs/default.yaml`：`kb.n_candidates=12` 与 `eval.n_candidates=12` 均存在，但攻击路径实际用的是 `attack_cfg.get("n_candidates", 6)`（`attack.py:45`，且 `attack` 段无此项）。三个来源三个值，建议统一到 `attack.n_candidates` 并删除未用的。

### D4 [低] 旧日志与 run 目录并存（过渡期）

`results/logs/08_*.log`（旧机制产物）与新机制 `results/runs/<run_id>/08_*.log` 并存；`results/logs/` 仍保留 vllm_*.log、toolsmoke_*.log 等启动/服务日志。旧 08_*.log 无 run_id 对应关系，仅作历史参考，可整批移入一个 `results/logs/legacy/` 或删除。

---

## 建议行动顺序

1. **（写稿前必做）** B2 修正 README MuSiQue true_in_top8 断言 —— 5 分钟。
   （已完成 2026-09-07，见 B2 正文。）
2. **（写稿前必做）** A2 / A3 在表注与正文补披露（fired 59/60 原因、phi parser 归因）—— 措辞级。
   （均已关闭：A2 用户判定非问题；A3 用户决定删除 phi-4-mini 主表行并改正文排除说明。）
3. **（决策）** A1+B1：决定重跑 mono/near_dup 至真实 8 块，还是降级 README/plan.md 的"同体积"与方向 C 表述。重跑约 1-2 小时（qwen3-4b 已恢复在 :8000）。
4. **（顺手）** C2 打印改 `len(targets)`；C3 补存 trace/n_tool_calls（对后续 run 生效）。
5. **（可选）** C1 附录注明 restrict 口径；D1/D3 统一 n_candidates；D4 归并旧日志。
   （C1/C2/D3/D4/C3 已完成，见附录"近期已修复"；C4 已关闭——接受现状，
   待 headline 重跑时自然补齐。）

---

## 附：近期已修复（不再列入待办）

- **C2 日志分母硬编码**（2026-09-07）：`experiments/08_longtail.py:425` 的
  `true_in_top8=…/60` 改为 `…/{len(targets)}`，MuSiQue（59 目标）运行不再显示误导分母。
- **C1 附录 musique 分母口径**（2026-09-07）：`summarize_ablation.py` 的
  `raw_all_runs()` 现对 `*_musique` 文件同样裁剪到冻结 59 目标（标题加注），
  附录与主表 ASR 分母一致（39/59），已重新生成 `results/ablation_summary.md`。
- **D4 旧日志归并**（2026-09-07）：无 run_id 对应的全部 `results/logs/08_*.log`
  （21 个）移入 `results/logs/legacy/`；服务/驱动日志（vllm_*.log、toolsmoke_*.log、
  *_driver.log 等）保留原位供审计引用。注意：新 run 的日志仍由驱动脚本写到
  `results/logs/`（`results/runs/` 当前为空）——run_id 机制尚未实际落地，见 D4 正文。
- **D3 n_candidates 配置统一**（2026-09-07）：删除死键 `kb.n_candidates` 与
  整段无消费者的 `eval` 段（grep 全库确认零引用）；`attack` 段新增显式
  `attack.n_candidates: 6`（含候选语义注释）。行为零变化——`load_config`
  验证取值仍为 6，attack.py 的 `.get("n_candidates", 6)` 兜底保留。
- **D2 并发采样语义文档化**（2026-09-07）：`payload.py` `_gen_i` 并发分支注释
  扩写（并发仅加速、分布与串行一致、break 时未结算请求不占 4n 上限的边界
  说明）；README Environment 节 concurrency 条目同步补注。
- **C3 ask_targets 补存审计字段**（2026-09-07）：`08_longtail.py` `record()` 每条
  答案补存 `n_tool_calls` / `trace` /（异常时）`error`，clean 与 after 两轮同生效。
  只对**新跑**结果 JSON 生效（schema 增量，下游按 `.get`/固定键读取无影响）；
  历史 phi-4-mini 运行仍只能靠日志推断工具遵从。

- **结果归档机制**（2026-09-07）：`results/archive/` 已废弃删除；`save_results`（`experiments/common.py`）现写入 `results/runs/<run_id>/`（run_id = `<运行名>_<YYYYmmdd_HHMMSS>`），根目录保留最新镜像供下游读取；`run_qwen_arms.sh`/`run_main_table_v2.sh` 每次运行生成唯一 run 目录并把日志放进同目录。
- **模型适配**（审计通过）：qwen3-8b think 关闭、llama-3.1 多工具模板、gpt-oss 32768 ctx + 64 seq、phi-4-mini 自定义模板、internlm3 排除，均经 vLLM 注册表 + serving 日志 + 运行零失败验证。
- **指标口径**（审计通过）：主表 5×2 全部数字经独立重算与报告一致；flip/collapse/ASR 定义正确。
