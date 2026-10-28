# Ablation on D_c (100 frozen questions per dataset)

Every number is a COUNT OUT OF 100 - the same 100 questions and the same denominator for every model, every setting and every method, so rows are directly comparable.

## HotpotQA (100 questions)

### questions the attack got to (flip + collapse) - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 96 | 77 | 88 | 75 |
| poison dose 2 per question | 72 | 59 | 68 | 62 |
| poison dose 4 per question | 85 | 69 | 73 | 73 |
| poison dose 6 per question | 92 | 76 | 79 | 78 |
| text style: embed_hybrid | 89 | 67 | 81 | 84 |
| text style: mono (a single writing style) | 81 | 67 | 66 | 72 |
| text style: nodiv (no diversity term when picking) | 93 | 78 | 81 | 80 |
| text style: greedy (no per-style anchor when picking) | 90 | 72 | 86 | 78 |
| trigger: semantic (fires on the matching questions only) | 96 | 78 | 88 | 77 |
| victim retrieval window k=4 | 84 | 83 | 84 | 72 |
| victim retrieval window k=16 | 84 | 77 | 85 | 76 |

### of which really flipped to the injected wrong answer - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 93 | 77 | 87 | 75 |
| poison dose 2 per question | 71 | 59 | 67 | 62 |
| poison dose 4 per question | 85 | 69 | 72 | 73 |
| poison dose 6 per question | 90 | 76 | 78 | 78 |
| text style: embed_hybrid | 89 | 67 | 80 | 84 |
| text style: mono (a single writing style) | 80 | 67 | 66 | 72 |
| text style: nodiv (no diversity term when picking) | 91 | 78 | 80 | 80 |
| text style: greedy (no per-style anchor when picking) | 89 | 72 | 85 | 78 |
| trigger: semantic (fires on the matching questions only) | 93 | 78 | 87 | 77 |
| victim retrieval window k=4 | 84 | 83 | 84 | 72 |
| victim retrieval window k=16 | 82 | 77 | 85 | 75 |

### HotpotQA: poison chunks actually written per question

Nominal dose is 2 / 4 / 6 / 8. A number below the nominal means the single-template arm could not produce that many **distinct** chunks for that question while still passing the generator's quality filter (states the injected wrong answer, length window, no leaked gold). Those questions still count in the 100; they just carry fewer chunks.

| setting | nominal | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|---|
| text style: embed_hybrid | 8 | 7.84 | 5.63 | 8 | 8 |
| text style: mono (a single writing style) | 8 | 7.92 | 7.77 | 7.92 | 8 |

### HotpotQA full ledger (out of 100)

| model | setting | got it | flipped | broke | still ok | unclear | already wrong before | strict ASR |
|---|---|---|---|---|---|---|---|---|
| xlam-2-8b | main | 96 | 93 | 3 | 4 | 0 | 0 | 93 |
| xlam-2-8b | vol2 | 72 | 71 | 1 | 27 | 1 | 0 | 71 |
| xlam-2-8b | vol4 | 85 | 85 | 0 | 13 | 2 | 0 | 85 |
| xlam-2-8b | vol6 | 92 | 90 | 2 | 8 | 0 | 0 | 90 |
| xlam-2-8b | embed_hybrid | 89 | 89 | 0 | 9 | 2 | 0 | 89 |
| xlam-2-8b | mono | 81 | 80 | 1 | 17 | 2 | 0 | 79 |
| xlam-2-8b | nodiv | 93 | 91 | 2 | 5 | 2 | 0 | 91 |
| xlam-2-8b | greedy | 90 | 89 | 1 | 7 | 3 | 0 | 89 |
| xlam-2-8b | trig_semantic | 96 | 93 | 3 | 3 | 1 | 0 | 93 |
| xlam-2-8b | topk4 | 84 | 84 | 0 | 12 | 1 | 3 | 87 |
| xlam-2-8b | topk16 | 84 | 82 | 2 | 7 | 1 | 8 | 87 |
| qwen3-4b-instruct | main | 77 | 77 | 0 | 22 | 0 | 1 | 78 |
| qwen3-4b-instruct | vol2 | 59 | 59 | 0 | 39 | 1 | 1 | 60 |
| qwen3-4b-instruct | vol4 | 69 | 69 | 0 | 30 | 0 | 1 | 70 |
| qwen3-4b-instruct | vol6 | 76 | 76 | 0 | 23 | 0 | 1 | 77 |
| qwen3-4b-instruct | embed_hybrid | 67 | 67 | 0 | 31 | 1 | 1 | 66 |
| qwen3-4b-instruct | mono | 67 | 67 | 0 | 32 | 0 | 1 | 68 |
| qwen3-4b-instruct | nodiv | 78 | 78 | 0 | 21 | 0 | 1 | 78 |
| qwen3-4b-instruct | greedy | 72 | 72 | 0 | 27 | 0 | 1 | 73 |
| qwen3-4b-instruct | trig_semantic | 78 | 78 | 0 | 21 | 0 | 1 | 79 |
| qwen3-4b-instruct | topk4 | 83 | 83 | 0 | 15 | 0 | 2 | 84 |
| qwen3-4b-instruct | topk16 | 77 | 77 | 0 | 21 | 0 | 2 | 79 |
| gpt-oss-20b | main | 88 | 87 | 1 | 11 | 0 | 1 | 88 |
| gpt-oss-20b | vol2 | 68 | 67 | 1 | 31 | 0 | 1 | 68 |
| gpt-oss-20b | vol4 | 73 | 72 | 1 | 26 | 0 | 1 | 72 |
| gpt-oss-20b | vol6 | 79 | 78 | 1 | 20 | 0 | 1 | 79 |
| gpt-oss-20b | embed_hybrid | 81 | 80 | 1 | 18 | 0 | 1 | 80 |
| gpt-oss-20b | mono | 66 | 66 | 0 | 33 | 0 | 1 | 66 |
| gpt-oss-20b | nodiv | 81 | 80 | 1 | 18 | 0 | 1 | 81 |
| gpt-oss-20b | greedy | 86 | 85 | 1 | 13 | 0 | 1 | 86 |
| gpt-oss-20b | trig_semantic | 88 | 87 | 1 | 11 | 0 | 1 | 88 |
| gpt-oss-20b | topk4 | 84 | 84 | 0 | 13 | 0 | 3 | 86 |
| gpt-oss-20b | topk16 | 85 | 85 | 0 | 12 | 0 | 3 | 86 |
| olmo-3-7b | main | 75 | 75 | 0 | 12 | 1 | 12 | 86 |
| olmo-3-7b | vol2 | 62 | 62 | 0 | 26 | 0 | 12 | 72 |
| olmo-3-7b | vol4 | 73 | 73 | 0 | 15 | 0 | 12 | 83 |
| olmo-3-7b | vol6 | 78 | 78 | 0 | 10 | 0 | 12 | 88 |
| olmo-3-7b | embed_hybrid | 84 | 84 | 0 | 4 | 0 | 12 | 95 |
| olmo-3-7b | mono | 72 | 72 | 0 | 15 | 1 | 12 | 81 |
| olmo-3-7b | nodiv | 80 | 80 | 0 | 7 | 1 | 12 | 91 |
| olmo-3-7b | greedy | 78 | 78 | 0 | 10 | 0 | 12 | 89 |
| olmo-3-7b | trig_semantic | 77 | 77 | 0 | 11 | 0 | 12 | 88 |
| olmo-3-7b | topk4 | 72 | 72 | 0 | 15 | 1 | 12 | 83 |
| olmo-3-7b | topk16 | 76 | 75 | 1 | 7 | 0 | 17 | 90 |

## MuSiQue (100 questions)

### questions the attack got to (flip + collapse) - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 84 | 73 | 78 | 50 |
| poison dose 2 per question | 71 | 55 | 52 | 40 |
| poison dose 4 per question | 78 | 68 | 62 | 47 |
| poison dose 6 per question | 80 | 70 | 69 | 46 |
| text style: embed_hybrid | 80 | 73 | 81 | 53 |
| text style: mono (a single writing style) | 73 | 61 | 61 | 48 |
| text style: nodiv (no diversity term when picking) | 78 | 71 | 73 | 54 |
| text style: greedy (no per-style anchor when picking) | 80 | 67 | 67 | 48 |
| trigger: semantic (fires on the matching questions only) | 83 | 71 | 68 | 51 |
| victim retrieval window k=4 | 76 | 71 | 70 | 48 |
| victim retrieval window k=16 | 80 | 65 | 67 | 53 |

### of which really flipped to the injected wrong answer - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 84 | 73 | 77 | 50 |
| poison dose 2 per question | 70 | 55 | 52 | 40 |
| poison dose 4 per question | 78 | 68 | 62 | 47 |
| poison dose 6 per question | 80 | 70 | 69 | 46 |
| text style: embed_hybrid | 79 | 72 | 81 | 53 |
| text style: mono (a single writing style) | 71 | 61 | 61 | 48 |
| text style: nodiv (no diversity term when picking) | 77 | 70 | 73 | 54 |
| text style: greedy (no per-style anchor when picking) | 80 | 67 | 67 | 48 |
| trigger: semantic (fires on the matching questions only) | 82 | 71 | 68 | 51 |
| victim retrieval window k=4 | 75 | 71 | 70 | 48 |
| victim retrieval window k=16 | 79 | 65 | 67 | 53 |

### MuSiQue: poison chunks actually written per question

Nominal dose is 2 / 4 / 6 / 8. A number below the nominal means the single-template arm could not produce that many **distinct** chunks for that question while still passing the generator's quality filter (states the injected wrong answer, length window, no leaked gold). Those questions still count in the 100; they just carry fewer chunks.

| setting | nominal | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|---|
| text style: embed_hybrid | 8 | 7.84 | 6.51 | 8 | 8 |
| text style: mono (a single writing style) | 8 | 6.92 | 6.97 | 7.65 | 7.79 |
| trigger: semantic (fires on the matching questions only) | 8 | 7.92 | 7.92 | 7.92 | 7.92 |

### MuSiQue full ledger (out of 100)

| model | setting | got it | flipped | broke | still ok | unclear | already wrong before | strict ASR |
|---|---|---|---|---|---|---|---|---|
| xlam-2-8b | main | 84 | 84 | 0 | 7 | 1 | 8 | 80 |
| xlam-2-8b | vol2 | 71 | 70 | 1 | 17 | 4 | 8 | 71 |
| xlam-2-8b | vol4 | 78 | 78 | 0 | 10 | 4 | 8 | 77 |
| xlam-2-8b | vol6 | 80 | 80 | 0 | 7 | 5 | 8 | 80 |
| xlam-2-8b | embed_hybrid | 80 | 79 | 1 | 9 | 3 | 8 | 78 |
| xlam-2-8b | mono | 73 | 71 | 2 | 17 | 2 | 8 | 74 |
| xlam-2-8b | nodiv | 78 | 77 | 1 | 10 | 4 | 8 | 77 |
| xlam-2-8b | greedy | 80 | 80 | 0 | 10 | 2 | 8 | 83 |
| xlam-2-8b | trig_semantic | 83 | 82 | 1 | 8 | 1 | 8 | 80 |
| xlam-2-8b | topk4 | 76 | 75 | 1 | 10 | 5 | 9 | 72 |
| xlam-2-8b | topk16 | 80 | 79 | 1 | 9 | 2 | 9 | 77 |
| qwen3-4b-instruct | main | 73 | 73 | 0 | 24 | 1 | 2 | 60 |
| qwen3-4b-instruct | vol2 | 55 | 55 | 0 | 38 | 5 | 2 | 50 |
| qwen3-4b-instruct | vol4 | 68 | 68 | 0 | 26 | 4 | 2 | 64 |
| qwen3-4b-instruct | vol6 | 70 | 70 | 0 | 21 | 7 | 2 | 62 |
| qwen3-4b-instruct | embed_hybrid | 73 | 72 | 1 | 18 | 7 | 2 | 67 |
| qwen3-4b-instruct | mono | 61 | 61 | 0 | 28 | 9 | 2 | 53 |
| qwen3-4b-instruct | nodiv | 71 | 70 | 1 | 22 | 5 | 2 | 61 |
| qwen3-4b-instruct | greedy | 67 | 67 | 0 | 24 | 7 | 2 | 59 |
| qwen3-4b-instruct | trig_semantic | 71 | 71 | 0 | 21 | 6 | 2 | 63 |
| qwen3-4b-instruct | topk4 | 71 | 71 | 0 | 11 | 9 | 9 | 67 |
| qwen3-4b-instruct | topk16 | 65 | 65 | 0 | 23 | 2 | 10 | 61 |
| gpt-oss-20b | main | 78 | 77 | 1 | 18 | 0 | 4 | 72 |
| gpt-oss-20b | vol2 | 52 | 52 | 0 | 40 | 4 | 4 | 51 |
| gpt-oss-20b | vol4 | 62 | 62 | 0 | 31 | 3 | 4 | 60 |
| gpt-oss-20b | vol6 | 69 | 69 | 0 | 22 | 5 | 4 | 67 |
| gpt-oss-20b | embed_hybrid | 81 | 81 | 0 | 11 | 4 | 4 | 74 |
| gpt-oss-20b | mono | 61 | 61 | 0 | 25 | 10 | 4 | 58 |
| gpt-oss-20b | nodiv | 73 | 73 | 0 | 18 | 5 | 4 | 68 |
| gpt-oss-20b | greedy | 67 | 67 | 0 | 22 | 7 | 4 | 65 |
| gpt-oss-20b | trig_semantic | 68 | 68 | 0 | 19 | 9 | 4 | 64 |
| gpt-oss-20b | topk4 | 70 | 70 | 0 | 17 | 9 | 4 | 66 |
| gpt-oss-20b | topk16 | 67 | 67 | 0 | 26 | 3 | 4 | 65 |
| olmo-3-7b | main | 50 | 50 | 0 | 4 | 7 | 39 | 71 |
| olmo-3-7b | vol2 | 40 | 40 | 0 | 16 | 5 | 39 | 50 |
| olmo-3-7b | vol4 | 47 | 47 | 0 | 8 | 6 | 39 | 57 |
| olmo-3-7b | vol6 | 46 | 46 | 0 | 7 | 8 | 39 | 66 |
| olmo-3-7b | embed_hybrid | 53 | 53 | 0 | 1 | 7 | 39 | 82 |
| olmo-3-7b | mono | 48 | 48 | 0 | 4 | 9 | 39 | 64 |
| olmo-3-7b | nodiv | 54 | 54 | 0 | 4 | 3 | 39 | 73 |
| olmo-3-7b | greedy | 48 | 48 | 0 | 4 | 9 | 39 | 71 |
| olmo-3-7b | trig_semantic | 51 | 51 | 0 | 5 | 5 | 39 | 71 |
| olmo-3-7b | topk4 | 48 | 48 | 0 | 9 | 6 | 37 | 62 |
| olmo-3-7b | topk16 | 53 | 53 | 0 | 2 | 3 | 42 | 61 |

## How to read this

- **main row** - the shipped method: 8 poison texts per question, cluster style, the trigger fires on everything, victim reads 8 passages.
  Numbers are the main table's own run, cited here for comparison.
- **poison dose** - how many fake passages go in per question (2 / 4 / 6 instead of the usual 8). Everything else is unchanged.
- **text style** - how the fake passages are worded and picked (embed_hybrid, mono, nodiv, greedy are the four ablated variants of the same generator).
- **trigger: semantic** - instead of firing on everything, the writer only writes when the incoming task is semantically close to one of the target questions. Questions that do not fire get no poison and are still part of the 100. Measured offline with the identical task text and trigger class, it fired on 100/100 (HotpotQA) and 99/100 (MuSiQue) - at this scale the trigger is not selective, which is exactly why this row lands next to the main row. The keyword design behaves the same way (99/100 on both datasets), so it is not run as its own row.
- **victim retrieval window** - how many passages the victim pulls from the knowledge base per question (4 / 16 instead of 8). Nothing about the attack changes. These two rows are scored against a CLEAN run made at the same window (snapshot only, no poison), so their 'already wrong before' column is what that window alone costs the model - a question the victim cannot answer with a smaller window is not counted as an attack win.
- **already wrong before** - questions the model answered wrongly with no attack at all. They stay in the denominator and are never credited to the attack.
- `-` means that run has not finished yet.

Generated 2026-10-28 16:03:37 by `experiments/summarize_ablation_dc.py` from the judged files in `results/` (judge: gpt-oss-20b, same judge as the main table).
