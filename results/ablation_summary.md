# Ablation on D_c (100 frozen questions per dataset)

Every number is a COUNT OUT OF 100 - the same 100 questions and the same denominator for every model, every setting and every method, so rows are directly comparable.

## HotpotQA (100 questions)

### questions the attack got to (flip + collapse) - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 96 | 77 | 88 | 75 |
| poison dose 2 per question | - | - | - | - |
| poison dose 4 per question | - | - | - | - |
| poison dose 6 per question | - | - | - | - |
| text style: embed_hybrid | - | - | - | - |
| text style: mono (a single writing style) | - | - | - | - |
| text style: nodiv (no diversity term when picking) | - | - | - | - |
| text style: greedy (pure question similarity) | - | - | - | - |
| trigger: semantic (fires on the matching questions only) | - | - | - | - |
| victim retrieval window k=4 | - | - | - | - |
| victim retrieval window k=16 | - | - | - | - |

### of which really flipped to the injected wrong answer - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 93 | 77 | 87 | 75 |
| poison dose 2 per question | - | - | - | - |
| poison dose 4 per question | - | - | - | - |
| poison dose 6 per question | - | - | - | - |
| text style: embed_hybrid | - | - | - | - |
| text style: mono (a single writing style) | - | - | - | - |
| text style: nodiv (no diversity term when picking) | - | - | - | - |
| text style: greedy (pure question similarity) | - | - | - | - |
| trigger: semantic (fires on the matching questions only) | - | - | - | - |
| victim retrieval window k=4 | - | - | - | - |
| victim retrieval window k=16 | - | - | - | - |

### HotpotQA full ledger (out of 100)

| model | setting | got it | flipped | broke | still ok | unclear | already wrong before | strict ASR |
|---|---|---|---|---|---|---|---|---|
| xlam-2-8b | main | 96 | 93 | 3 | 4 | 0 | 0 | 93 |
| xlam-2-8b | vol2 | - | - | - | - | - | - | - |
| xlam-2-8b | vol4 | - | - | - | - | - | - | - |
| xlam-2-8b | vol6 | - | - | - | - | - | - | - |
| xlam-2-8b | embed_hybrid | - | - | - | - | - | - | - |
| xlam-2-8b | mono | - | - | - | - | - | - | - |
| xlam-2-8b | nodiv | - | - | - | - | - | - | - |
| xlam-2-8b | greedy | - | - | - | - | - | - | - |
| xlam-2-8b | trig_semantic | - | - | - | - | - | - | - |
| xlam-2-8b | topk4 | - | - | - | - | - | - | - |
| xlam-2-8b | topk16 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | main | 77 | 77 | 0 | 22 | 0 | 1 | 78 |
| qwen3-4b-instruct | vol2 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | vol4 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | vol6 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | embed_hybrid | - | - | - | - | - | - | - |
| qwen3-4b-instruct | mono | - | - | - | - | - | - | - |
| qwen3-4b-instruct | nodiv | - | - | - | - | - | - | - |
| qwen3-4b-instruct | greedy | - | - | - | - | - | - | - |
| qwen3-4b-instruct | trig_semantic | - | - | - | - | - | - | - |
| qwen3-4b-instruct | topk4 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | topk16 | - | - | - | - | - | - | - |
| gpt-oss-20b | main | 88 | 87 | 1 | 11 | 0 | 1 | 88 |
| gpt-oss-20b | vol2 | - | - | - | - | - | - | - |
| gpt-oss-20b | vol4 | - | - | - | - | - | - | - |
| gpt-oss-20b | vol6 | - | - | - | - | - | - | - |
| gpt-oss-20b | embed_hybrid | - | - | - | - | - | - | - |
| gpt-oss-20b | mono | - | - | - | - | - | - | - |
| gpt-oss-20b | nodiv | - | - | - | - | - | - | - |
| gpt-oss-20b | greedy | - | - | - | - | - | - | - |
| gpt-oss-20b | trig_semantic | - | - | - | - | - | - | - |
| gpt-oss-20b | topk4 | - | - | - | - | - | - | - |
| gpt-oss-20b | topk16 | - | - | - | - | - | - | - |
| olmo-3-7b | main | 75 | 75 | 0 | 12 | 1 | 12 | 86 |
| olmo-3-7b | vol2 | - | - | - | - | - | - | - |
| olmo-3-7b | vol4 | - | - | - | - | - | - | - |
| olmo-3-7b | vol6 | - | - | - | - | - | - | - |
| olmo-3-7b | embed_hybrid | - | - | - | - | - | - | - |
| olmo-3-7b | mono | - | - | - | - | - | - | - |
| olmo-3-7b | nodiv | - | - | - | - | - | - | - |
| olmo-3-7b | greedy | - | - | - | - | - | - | - |
| olmo-3-7b | trig_semantic | - | - | - | - | - | - | - |
| olmo-3-7b | topk4 | - | - | - | - | - | - | - |
| olmo-3-7b | topk16 | - | - | - | - | - | - | - |

## MuSiQue (100 questions)

### questions the attack got to (flip + collapse) - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 84 | 73 | 78 | 50 |
| poison dose 2 per question | - | - | - | - |
| poison dose 4 per question | - | - | - | - |
| poison dose 6 per question | - | - | - | - |
| text style: embed_hybrid | - | - | - | - |
| text style: mono (a single writing style) | - | - | - | - |
| text style: nodiv (no diversity term when picking) | - | - | - | - |
| text style: greedy (pure question similarity) | - | - | - | - |
| trigger: semantic (fires on the matching questions only) | - | - | - | - |
| victim retrieval window k=4 | - | - | - | - |
| victim retrieval window k=16 | - | - | - | - |

### of which really flipped to the injected wrong answer - out of 100

| setting | xlam-2-8b | qwen3-4b-instruct | gpt-oss-20b | olmo-3-7b |
|---|---|---|---|---|
| main row: dose 8, cluster style, trigger always on, window k=8 | 84 | 73 | 77 | 50 |
| poison dose 2 per question | - | - | - | - |
| poison dose 4 per question | - | - | - | - |
| poison dose 6 per question | - | - | - | - |
| text style: embed_hybrid | - | - | - | - |
| text style: mono (a single writing style) | - | - | - | - |
| text style: nodiv (no diversity term when picking) | - | - | - | - |
| text style: greedy (pure question similarity) | - | - | - | - |
| trigger: semantic (fires on the matching questions only) | - | - | - | - |
| victim retrieval window k=4 | - | - | - | - |
| victim retrieval window k=16 | - | - | - | - |

### MuSiQue full ledger (out of 100)

| model | setting | got it | flipped | broke | still ok | unclear | already wrong before | strict ASR |
|---|---|---|---|---|---|---|---|---|
| xlam-2-8b | main | 84 | 84 | 0 | 7 | 1 | 8 | 80 |
| xlam-2-8b | vol2 | - | - | - | - | - | - | - |
| xlam-2-8b | vol4 | - | - | - | - | - | - | - |
| xlam-2-8b | vol6 | - | - | - | - | - | - | - |
| xlam-2-8b | embed_hybrid | - | - | - | - | - | - | - |
| xlam-2-8b | mono | - | - | - | - | - | - | - |
| xlam-2-8b | nodiv | - | - | - | - | - | - | - |
| xlam-2-8b | greedy | - | - | - | - | - | - | - |
| xlam-2-8b | trig_semantic | - | - | - | - | - | - | - |
| xlam-2-8b | topk4 | - | - | - | - | - | - | - |
| xlam-2-8b | topk16 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | main | 73 | 73 | 0 | 24 | 1 | 2 | 60 |
| qwen3-4b-instruct | vol2 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | vol4 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | vol6 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | embed_hybrid | - | - | - | - | - | - | - |
| qwen3-4b-instruct | mono | - | - | - | - | - | - | - |
| qwen3-4b-instruct | nodiv | - | - | - | - | - | - | - |
| qwen3-4b-instruct | greedy | - | - | - | - | - | - | - |
| qwen3-4b-instruct | trig_semantic | - | - | - | - | - | - | - |
| qwen3-4b-instruct | topk4 | - | - | - | - | - | - | - |
| qwen3-4b-instruct | topk16 | - | - | - | - | - | - | - |
| gpt-oss-20b | main | 78 | 77 | 1 | 18 | 0 | 4 | 72 |
| gpt-oss-20b | vol2 | - | - | - | - | - | - | - |
| gpt-oss-20b | vol4 | - | - | - | - | - | - | - |
| gpt-oss-20b | vol6 | - | - | - | - | - | - | - |
| gpt-oss-20b | embed_hybrid | - | - | - | - | - | - | - |
| gpt-oss-20b | mono | - | - | - | - | - | - | - |
| gpt-oss-20b | nodiv | - | - | - | - | - | - | - |
| gpt-oss-20b | greedy | - | - | - | - | - | - | - |
| gpt-oss-20b | trig_semantic | - | - | - | - | - | - | - |
| gpt-oss-20b | topk4 | - | - | - | - | - | - | - |
| gpt-oss-20b | topk16 | - | - | - | - | - | - | - |
| olmo-3-7b | main | 50 | 50 | 0 | 4 | 7 | 39 | 71 |
| olmo-3-7b | vol2 | - | - | - | - | - | - | - |
| olmo-3-7b | vol4 | - | - | - | - | - | - | - |
| olmo-3-7b | vol6 | - | - | - | - | - | - | - |
| olmo-3-7b | embed_hybrid | - | - | - | - | - | - | - |
| olmo-3-7b | mono | - | - | - | - | - | - | - |
| olmo-3-7b | nodiv | - | - | - | - | - | - | - |
| olmo-3-7b | greedy | - | - | - | - | - | - | - |
| olmo-3-7b | trig_semantic | - | - | - | - | - | - | - |
| olmo-3-7b | topk4 | - | - | - | - | - | - | - |
| olmo-3-7b | topk16 | - | - | - | - | - | - | - |

## How to read this

- **main row** - the shipped method: 8 poison texts per question, cluster style, the trigger fires on everything, victim reads 8 passages.
  Numbers are the main table's own run, cited here for comparison.
- **poison dose** - how many fake passages go in per question (2 / 4 / 6 instead of the usual 8). Everything else is unchanged.
- **text style** - how the fake passages are worded and picked (embed_hybrid, mono, nodiv, greedy are the four ablated variants of the same generator).
- **trigger: semantic** - instead of firing on everything, the writer only writes when the incoming task is semantically close to one of the target questions. Questions that do not fire get no poison and are still part of the 100. Measured offline with the identical task text and trigger class, it fired on 100/100 (HotpotQA) and 99/100 (MuSiQue) - at this scale the trigger is not selective, which is exactly why this row lands next to the main row. The keyword design behaves the same way (99/100 on both datasets), so it is not run as its own row.
- **victim retrieval window** - how many passages the victim pulls from the knowledge base per question (4 / 16 instead of 8). Nothing about the attack changes.
- **already wrong before** - questions the model answered wrongly with no attack at all. They stay in the denominator and are never credited to the attack.
- `-` means that run has not finished yet.

Generated 2026-10-28 00:56:41 by `experiments/summarize_ablation_dc.py` from the judged files in `results/` (judge: gpt-oss-20b, same judge as the main table).

## Missing pieces (these rows are not finished yet)

- missing run for `hotpot` / xlam-2-8b / vol2 (dc_ours_vol2_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / vol2 (dc_ours_vol2_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / vol2 (dc_ours_vol2_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / vol2 (dc_ours_vol2_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / vol4 (dc_ours_vol4_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / vol4 (dc_ours_vol4_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / vol4 (dc_ours_vol4_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / vol4 (dc_ours_vol4_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / vol6 (dc_ours_vol6_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / vol6 (dc_ours_vol6_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / vol6 (dc_ours_vol6_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / vol6 (dc_ours_vol6_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / embed_hybrid (dc_ours_embed_hybrid_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / embed_hybrid (dc_ours_embed_hybrid_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / embed_hybrid (dc_ours_embed_hybrid_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / embed_hybrid (dc_ours_embed_hybrid_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / mono (dc_ours_mono_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / mono (dc_ours_mono_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / mono (dc_ours_mono_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / mono (dc_ours_mono_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / nodiv (dc_ours_nodiv_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / nodiv (dc_ours_nodiv_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / nodiv (dc_ours_nodiv_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / nodiv (dc_ours_nodiv_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / greedy (dc_ours_greedy_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / greedy (dc_ours_greedy_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / greedy (dc_ours_greedy_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / greedy (dc_ours_greedy_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / trig_semantic (dc_ours_trig_semantic_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / trig_semantic (dc_ours_trig_semantic_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / trig_semantic (dc_ours_trig_semantic_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / trig_semantic (dc_ours_trig_semantic_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / topk4 (dc_ours_topk4_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / topk4 (dc_ours_topk4_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / topk4 (dc_ours_topk4_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / topk4 (dc_ours_topk4_olmo-3-7b_eval_judged.json)
- missing run for `hotpot` / xlam-2-8b / topk16 (dc_ours_topk16_xlam-2-8b_eval_judged.json)
- missing run for `hotpot` / qwen3-4b-instruct / topk16 (dc_ours_topk16_qwen3-4b-instruct_eval_judged.json)
- missing run for `hotpot` / gpt-oss-20b / topk16 (dc_ours_topk16_gpt-oss-20b_eval_judged.json)
- missing run for `hotpot` / olmo-3-7b / topk16 (dc_ours_topk16_olmo-3-7b_eval_judged.json)
- missing run for `musique` / xlam-2-8b / vol2 (dc_ours_vol2_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / vol2 (dc_ours_vol2_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / vol2 (dc_ours_vol2_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / vol2 (dc_ours_vol2_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / vol4 (dc_ours_vol4_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / vol4 (dc_ours_vol4_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / vol4 (dc_ours_vol4_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / vol4 (dc_ours_vol4_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / vol6 (dc_ours_vol6_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / vol6 (dc_ours_vol6_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / vol6 (dc_ours_vol6_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / vol6 (dc_ours_vol6_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / embed_hybrid (dc_ours_embed_hybrid_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / embed_hybrid (dc_ours_embed_hybrid_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / embed_hybrid (dc_ours_embed_hybrid_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / embed_hybrid (dc_ours_embed_hybrid_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / mono (dc_ours_mono_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / mono (dc_ours_mono_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / mono (dc_ours_mono_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / mono (dc_ours_mono_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / nodiv (dc_ours_nodiv_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / nodiv (dc_ours_nodiv_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / nodiv (dc_ours_nodiv_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / nodiv (dc_ours_nodiv_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / greedy (dc_ours_greedy_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / greedy (dc_ours_greedy_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / greedy (dc_ours_greedy_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / greedy (dc_ours_greedy_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / trig_semantic (dc_ours_trig_semantic_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / trig_semantic (dc_ours_trig_semantic_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / trig_semantic (dc_ours_trig_semantic_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / trig_semantic (dc_ours_trig_semantic_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / topk4 (dc_ours_topk4_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / topk4 (dc_ours_topk4_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / topk4 (dc_ours_topk4_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / topk4 (dc_ours_topk4_olmo-3-7b_musique_eval_judged.json)
- missing run for `musique` / xlam-2-8b / topk16 (dc_ours_topk16_xlam-2-8b_musique_eval_judged.json)
- missing run for `musique` / qwen3-4b-instruct / topk16 (dc_ours_topk16_qwen3-4b-instruct_musique_eval_judged.json)
- missing run for `musique` / gpt-oss-20b / topk16 (dc_ours_topk16_gpt-oss-20b_musique_eval_judged.json)
- missing run for `musique` / olmo-3-7b / topk16 (dc_ours_topk16_olmo-3-7b_musique_eval_judged.json)
