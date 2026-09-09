# Cross-model matrix F — poison texts by attacker A replayed into victim B

cluster v8, keyword trigger, 60 shared HotpotQA targets. flip = clean-correct -> NON-EMPTY wrong answer (denominator = victim's own main-table clean-correct); ASR = injected wrong-answer substring over all 60 targets; coll = collapse (empty/crashed).

| attacker \ victim | xlam-2-8b | qwen3-8b | gpt-oss-20b | llama-3.1-8b |
|---|---|---|---|---|
| xlam-2-8b | — (main table) | **28/31 (90%)**, ASR 85% | **28/38 (74%)**, ASR 67%, coll 1 | **23/25 (92%)**, ASR 78% |
| qwen3-8b | **22/27 (81%)**, ASR 88%, coll 2 | — (main table) | **26/38 (68%)**, ASR 65% | **22/25 (88%)**, ASR 73%, coll 1 |
| gpt-oss-20b | **24/27 (89%)**, ASR 85% | **29/31 (94%)**, ASR 92% | — (main table) | **23/25 (92%)**, ASR 77%, coll 1 |
| llama-3.1-8b | **21/27 (78%)**, ASR 65% | **25/31 (81%)**, ASR 72% | **24/38 (63%)**, ASR 55%, coll 1 | — (main table) |
