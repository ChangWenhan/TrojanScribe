# Collapse 归因报告：ReAct 基线的空答案是方法属性，不是 harness bug

日期：2026-09-06。分析脚本：/tmp/collapse_analysis.py（临时，已清理）。

## 分类结果（对 5 个 react 结果文件的每一行）

| method | A 步数耗尽(未发Finish) | B 主动空Finish | C 解析bug | C-mismatch | D error行 | 提取正常 |
|---|---|---|---|---|---|---|
| clean | 15 | 0 | **0** | 0 | 0 | 45 |
| naive | 38 | 0 | **0** | 0 | 0 | 22 |
| poisonedRAG | 20 | 0 | **0** | 0 | 0 | 40 |
| ours | 37 | 0 | **0** | 0 | 0 | 23 |
| topicattack | 24 | 0 | **0** | 0 | 12 | 24 |

## A 类细分

| method | empty 合计 | 完全卡死(9步无可解析动作) | 检索循环(有Search不收敛) | 长thought无action |
|---|---|---|---|---|
| clean | 15 | **6** | 9 | 14 |
| naive | 38 | **6** | 32 | 11 |
| poisonedRAG | 20 | **6** | 14 | 19 |
| ours | 37 | **6** | 31 | 37 |
| topicattack | 24 | **6** | 18 | 21 |

## 判定与证据

1. **不存在 harness 解析 bug**：没有任何一行出现"模型输出了 Finish[非空] 但 llm_answer 为空/不符"（C 类全 0）；也没有模型主动提交空 Finish（B 类全 0）。抽取链路（_clean_thought_act → finish[...]）被证明在 Finish 出现时工作正常。
2. **6 例"完全卡死"在所有方法中完全一致（各 6 个）**——这是 4B 模型在官方 ReAct 协议（9 步上限、max_tokens=100）上的确定性基线失败，与投毒无关。
3. **方法可归因的空答案 = 检索循环增量**：naive +23、ours(KidnapRAG) +22、poisonedRAG +5、topicattack +9（相对 clean 的 9 个）。其毒文内容（prompt-injection 式指令/冲突文本）驱动模型反复搜索或写超长 thought 直至步数耗尽——这正是这些方法"非定向 DoS"的机制证据，而非评测器故障。
4. topicattack 的 12 个 error 行（agent 运行时异常）已单独计数，不混入知识翻转。
5. 协议层面因素（9 步上限、max_tokens=100、stop tokens）均为 KidnapRAG 官方设置，保持不动以维持基线可比性。

**结论：collapse 是方法/协议属性，代码无需修复，react 基线结果无需重跑。**
论文写作建议：在 flip 指标定义处引用本分解（flips_knowledge vs flips_collapse），并给出"6 例共享卡死"作为模型基线失败的锚点。
