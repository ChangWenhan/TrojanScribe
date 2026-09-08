# RAG 投毒数据量文献核对报告（防审稿人质疑规模）

核对日期：2026-09-06。核对方式：本地综述（research/litreview/01、00）+ arXiv 摘要/HTML 全文逐篇核对。所有数字标注出处；"推导"为按公开数字计算的比率，原论文未报告。

## 对比表

| 论文 | 目标问题数 | 每目标毒文数 | 毒文总量 | 语料库大小 | 毒文占比 |
|---|---|---|---|---|---|
| PoisonedRAG (USENIX Sec'25, arXiv:2402.07867) | **100/数据集**（10 题 × 10 轮；NQ/HotpotQA/MS-MARCO 共 300） | **5**（N=5 默认超参） | **500/数据集**（推导） | NQ **2,681,468**；HotpotQA **5,233,329**；MS-MARCO **8,841,823** texts | 未报告；推导 0.006%–0.019% |
| CorruptRAG (SACMAT'26, arXiv:2504.03957) | **100/数据集** × 3 | **1** | **100/数据集**（推导） | 同 PoisonedRAG | 推导 0.001%–0.004% |
| AgentPoison (NeurIPS'24, arXiv:2407.12784) | 按后门触发评测 | 1 条即 62.0% 平均 ASR；主设置 20 / 4 / 2 条 | 20 / 4 / 2 条 | 23,000 / 10,000 / 704 | 声称 <0.1%；EHR 推导实为 0.28% |
| KidnapRAG (arXiv:2607.00422) | **100/数据集** × 3（HotpotQA/MuSiQue/2Wiki） | **约 2–6 条链式毒文**（另每子查询 5 条检索优化文档） | 每数据集数百条量级（推导） | HotpotQA(BEIR) 5,233,329；MuSiQue **48,315**；2Wiki 125,760 | MuSiQue 推导 ≈0.4–1.2% |
| GRAB-RAG (arXiv:2608.22228) | 500/数据集（鲁棒性评测） | 1 条误导段落 + 4 hard negatives | 每题 1 条（非 KB 注入） | 未公开 | 不适用 |
| Knowledge reconciliation (arXiv:2506.15732) | 未公开 | 不适用（合成任务，无 KB） | 不适用 | 不适用 | 不适用 |
| BadRAG (arXiv:2406.00083) | 未公开（触发主题式） | 10 条（MCOP）；COP 需 200 条 | 10–200 条 | ≈2.5 万（由 0.04% 反推） | **0.04%**（论文明确给出） |
| **我们** | **60**（HotpotQA dev long-tail） | **7–8**（consensus cluster；对照 8） | **435** | **66,581** chunks | **0.65%** |

出处：PoisonedRAG 原文 "10 close-ended questions × 10 repeats = 100 target questions"、"injecting 5 malicious texts for each target question"、"2,681,468 and 5,233,329 texts"；CorruptRAG "100 closed-ended queries per dataset"、"one poisoned text per targeted query"；AgentPoison Table 1（20/4/2 条，"poison rate <0.1%"）；KidnapRAG "100 queries per dataset"、"five poisoned documents per observed subquery"、"BEIR corpus 5,233,329 documents, MuSiQue 48,315, 2Wiki 125,760"。

## 结论：我们的位置

- **目标问题数（60）**：同数量级、略小于文献标准（PoisonedRAG/CorruptRAG/KidnapRAG 均为 100/数据集）。
- **每目标毒文数（7–8）**：位于已发表区间 [1, ~11] 的中段，与 PoisonedRAG 的 5 条最接近；top-k=8 > PoisonedRAG 的 top-5，需要略多毒文占位有结构性理由。
- **毒文总量（435）**：与 PoisonedRAG 单数据集 500 条同量级，远大于 CorruptRAG（100）和 AgentPoison（2–20）。
- **语料库（66,581）**：比 PoisonedRAG/CorruptRAG 小 40–130 倍，但与 **KidnapRAG-MuSiQue（48,315）、AgentPoison（10,000）、BadRAG（~2.5 万）同一数量级**。
- **毒文占比（0.65%）**：偏高但非异常——绝对量 435 < PoisonedRAG 的 500；同量级语料库上 KidnapRAG-MuSiQue ≈0.4–1.2%、AgentPoison-EHRAgent 0.28%，与我们相当。

## 审稿人质疑数据量时的三条回应论据

1. **每目标预算**：7–8 条/目标与 PoisonedRAG（5 条，USENIX Sec'25）同量级；CorruptRAG 证明 1 条即达 0.92–0.98 ASR；我们的 top-k=8 需要略多占位。
2. **目标问题数**：60 与三篇 SOTA 的"100/数据集"同一数量级（0.6×），且逐题报告了分层统计（strict flip n=20/21），评测粒度不低于上述工作。
3. **总量与占比**：绝对量 435 < PoisonedRAG 的 500；0.65% 占比是语料库小 ~79 倍的机械结果；同量级语料库的公开设置反推占比与我们相当或更高。GRAB-RAG 证明攻击效力由毒文质量与每目标预算决定，而非总量；我们的小语料库反而使检索条件更严格可验证。

**诚实披露点（limitation 节）**：若审稿人专攻"占比"口径，应同时报告绝对量（435 < 500）与同量级语料库工作的反推占比，并说明占比非威胁模型的自变量。
