# 消融实验计划（frozen 2026-09-06）

在修复后的主协议（60 共享目标、干净 KB、统一评分 src/agentic_rag/eval/unified.py、
cluster v8 headline、embed_hybrid v8 对照）基础上规划四个消融方向。
所有 langgraph 侧运行走 experiments/08_longtail.py；react 基线不重跑。
结果均按 08 结果 JSON 的统一字段汇总（experiments/summarize_ablation.py）。

## A. Backbone 消融（模型泛化）
- 固定：60 共享目标（--use-shared-targets，跨骨干同题同 wrong answer）、
  hotpotqa KB 66,581、top-8、cluster v8、keyword 触发、benign_rounds=3。
- 骨干（本地 LLMs/，全部 ≤7B，tool-call 预检通过）：
  1. qwen3-4b（Qwen3-4B-Instruct-2507，headline 已有结果）
  2. qwen2.5-7b（Qwen2.5-7B-Instruct，hermes）
  3. llama-3.2-3b（Llama-3.2-3b-instruction，llama3_json）
  4. qwen2.5-3b（Qwen2.5-3B-Instruct，hermes）
  （mistral-7b-v0.3 预检失败：HF checkpoint 的 [TOOL_CALLS] 纯文本不被
  vLLM 0.15 mistral parser 结构化，victim 需要结构化 tool_calls，故弃用。）
- 读数：clean EM（模型能力锚点）、flips_knowledge、ASR、poison_follow、
  true_in_top8。预期：更小/更弱的骨干 flip 与 follow 更高；强参数记忆
  （famous-entity stays）随家族变化。

## B. 数据集消融（迁移性）
- 固定：qwen3-4b、cluster v8、keyword 触发；变化：HotpotQA → MuSiQue
  （musique_ans_v1.0_dev，2-4 hop 组合式多跳，KB=17,629 块，
  data/chroma_musique）。
- 目标：MuSiQue 上重走 candidate→prior filter→60 目标（无共享文件，
  wrong answer 由 payload.wrong_answer 生成并逐题持久化——运行内
  ASR 口径仍然一致：打分串==注入串）。
- 读数：同 A；对照 headline（hotpotqa）。预期：多跳+更小 KB 下检索条件
  更易满足，TPR 不降；flip 依 KB 结构而变。

## C. 体积消融（diversity vs volume 曲线）
- 固定：qwen3-4b、hotpotqa、cluster、keyword 触发。
- 变化：volume 8（已有）→ volume 4（--volume 4）。
- 读数：flips_knowledge、consensus_cov、mean_distinct、true_in_top8。
  预期（AR3 冻结预测 A1）：v4 已获得大部分 flip 收益（v8/v4 ≈ 1.0-1.15）；
  若 v8 相对 v4 增益显著，说明体积本身（而非多样性）驱动 flip。

## D. 触发器消融（攻击面实用性）
- 固定：qwen3-4b、hotpotqa、cluster v8。
- 变化：keyword（已有）→ semantic（threshold 0.82，BGE 任务文本 vs 目标
  问题余弦）。
- 读数：n_targets_fired（触发覆盖率）、flips、ASR、以及"毒文写入量/触发
  精度"权衡。预期：semantic 覆盖率 ≥ keyword（修复 keyword 漏掉全小写
  问题的边界），端到端 flip 接近 keyword。

## 运行安排（单卡 4090 串行）
1. run_model_ablation.sh：qwen2.5-7b → llama-3.2-3b → qwen2.5-3b
   （每模型 ~65 min，结束后自动恢复 qwen3-4b 服务）。
2. qwen2.5-7b 补跑（首轮因 --corpus-path argparse 缺失失败，已修复）。
3. MuSiQue 数据集消融（~70 min，含 target 选择）。
4. volume v4（~55 min）。
5. trigger semantic（~55 min）。

## 结果归档
results/08_longtail_<name>.json + results/logs/ 各阶段日志；汇总表
results/ablation_summary.md（summarize_ablation.py 生成）；结论写入
experiment_ledger.md。

---

# ABLATION-V2 + MAIN TABLE UPGRADE (frozen 2026-09-06, before any run)

动机：主表升级为「多骨干 × 双数据集」，并把消融重构为彼此独立、每个方向都有
明确 insight 的三组。旧骨干消融（qwen2.5-7b/3b、llama-3.2-3b）保留归档于
results/archive/，不再进入主表。

## 主表协议（4 骨干 × 2 数据集）

| 骨干 | 版本 | 服务参数 |
|---|---|---|
| qwen3-4b (Qwen3-4B-Instruct-2507) | 2025-07 主模型 | 已有全部结果 |
| qwen3-8b | 2025-04 | hermes + --reasoning-parser qwen3（思考剥离） |
| gpt-oss-20b | 2025-08 | openai(harmony) 解析器，smoke 决定 flags |
| phi-4-mini (Phi-4-mini-instruct) | 2025-02 | phi4_mini_json 解析器；internlm3 因弱工具格式遵从 + 啰嗦思考前导被排除（证据见 ledger 2026-09-07） |

* HotpotQA 行：--use-shared-targets（共享 60 目标 + 共享 wrongs，与 baselines 同协议）。
* MuSiQue 行：--target-records research/frozen/musique_targets_59.json（59 目标 +
  冻结 wrongs，跨模型零重选；qwen3-4b 已有行的汇总在分析期裁剪到同一 59 qid）。
* 每行指标：clean-EM(official)、flip(knowledge, 非空错误答案)、collapse、ASR、
  poison_follow、true_in_top8。insight：攻击是否随骨干家族/年代/规模泛化；
  协议依从性仍是主要调节变量；MuSiQue 多跳上是否完全迁移。

## 方向 B：投毒体积（volume = 每目标写入块数）

* 臂：2 / 4 / 6 / 8（cluster，HotpotQA，qwen3-4b，keyword 触发）。
* v4（results/08_longtail_cluster_v4.json）与 v8（results/08_longtail.json）复用；
  新跑 v2、v6。
* insight：剂量-反应曲线与饱和点；体积 vs 隐匿成本（总量/目标）。

## 方向 C：簇多样性（独立方向，全部 v8 / HotpotQA / qwen3-4b）

| 臂 | 机制 | 状态 |
|---|---|---|
| embed_hybrid（近重复） | 语义挤位经典基线，无多声音 | 已有（08_longtail.json） |
| cluster_mono | 单一 authority 原型（生成数↑保证同体积） | 新 |
| cluster_nodiv | 5 原型 + 轮转，去掉两两多样性门 | 新 |
| cluster_greedy | 无轮转，纯 query-cosine + 多样性门 | 新 |
| cluster（完整） | 5 原型 + 轮转 + 多样性门 | 已有 |

* insight：把"多声音"拆成 表面多样性(mono) / 原型覆盖(round-robin) /
  多样性约束(gate) 三个可分离组件，量化各自贡献。

## 方向 D：触发器

* 臂：keyword（已有）、semantic 0.82（已有）、always p=1.0（新，无过滤上界，
  用于回答"选择性是否必要、keyword 的选择性是否零成本"）。
* insight：触发保真度对 ASR/覆盖的影响；keyword 触发在自身目标查询上近乎
  必然命中，因此应逼近无过滤上界。

## 控制变量（所有新跑一致）

benign_rounds=3、共享/冻结目标与 wrongs、victim 提示词、KB、bge 嵌入、
16384 ctx 服务参数；每个 run 前清 KB（08 自清理）；增量保存沿用。

## 冻结时的预登记结论槽（跑完填写，禁止跑后改臂）

* B-volume insight：剂量-反应在 ~6 块/目标处饱和——flip 2/4/6/8 = 40/60/80/75%，
  ASR 45/51.7/66.7/61.7%（v6 与 v8 的差距 1 题，处于 n=20 噪声内）；true_in_top8
  单调下降 48→44→36→22，位移机制随体积增强；v6 是效率甜点，v8 为保守上界。
* C-diversity insight：承重组件是"生成层多声音"本身，而非选择机制。去掉两两
  多样性门（nodiv 80%）或原型轮转（greedy 70%）仍在完整版（75%）噪声带内（n=20，
  1 题=5pp）；彻底去掉声音多样性则塌方——near_dup 45%、mono(authority) 45%，
  且 mono 的 ASR 仅 33.3%（同预算对照：mono ~3.6 块/目标 vs v4 4.0 块/目标，
  v4 flip 60%/ASR 51.7% 全面占优）。附带发现：单声音池过不了去重门（57/60 目标
  仅 206 块），多样设计是维持体积的前提。top8 真段落存活：nodiv 10 < full 22 <
  mono 40 —— faq 泛滥型位移最强但收益饱和。
* D-trigger insight：keyword 触发达到无过滤上界（always p=1.0：fired 60/60，
  flip 80%、ASR 63.3%，与 keyword 75%/61.7% 同噪声带）——keyword 的选择性在
  自身目标查询上近乎必然命中，即选择性是"零成本"的；semantic 0.82 阈值损失
  ~10pp（65% flip / 58.3% ASR，漏触发所致）。触发器实现方式不是成功的关键路径，
  但语义阈值需要校准。
* 主表 backbone×dataset insight：攻击跨骨干家族与年代完全泛化——4 个可信骨干
  （Qwen3-4B-2507 / Qwen3-8B / gpt-oss-20b / Llama-3.1-8B）在 HotpotQA 上的
  knowledge-flip 率 71.9%–86.7%、MuSiQue 上 73.7%–100%，骨干越强并非越抗毒
  （8B 84% > 4B 75%；gpt-oss 20B 反而最低 71.9%，其强 clean-EM 反而提供更大
  翻转面 32/60）。ASR 是骨干敏感指标（51.7%–83.3%），与模型复述注入串的倾向
  相关。MuSiQue 多跳上 true_in_top8≈0（完全位移）且 flip 分母极小（6–19），
  说明攻击在多跳语料上直接夺取检索位。协议脆弱性是攻击的隐性门槛：phi-4-mini
  连 victim 工具协议都守不住（clean-EM 1/60、0/59），flip 无定义——攻击面以
  目标模型的 agent 协议遵从为前提。服务层事实：gpt-oss 需 32k 上下文 +
  max-num-seqs 64（sampler warmup OOM）；Llama-3.1 需多工具调用模板补丁
  （官方模板仅支持单调用历史）；Qwen3-8B 需请求侧关闭思考。
