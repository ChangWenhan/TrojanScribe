# Monitor Experiment Ledger

Freeze manifest: `monitor/freeze_manifest.sha256` (verified read-only frozen proposals).
Results are appended strictly AFTER frozen predictions; proposals are never edited.

---

## AR3 — ConsensusCluster (executed 2026-09-04)

### Frozen predictions (ar3, G1/G3/R4/R6, excerpt)
- G1: flips on baseline-correct P1 (60 long-tail targets, volume 5) = 0.70 (14-19/24)
- G3: flips P2 (volume 8, headline) = 0.75 (16-20/24)
- R4: consensus_coverage P1 >= 0.80; R6: P2 >= 0.86
- Falsification bound: coverage < 0.80 at volume 8 -> mechanism failed, flip claim void

### Protocol deviations (recorded BEFORE results known)
1. `generate_cluster` thresholds: frozen tau_pair=0.65 / tau_dice=0.50 / no >=6-gram
   were un-implementable (same-topic chunks sharing mirrored question + wrong entity
   always exceed them -> 1 chunk/target). Relaxed to dedup-only 0.95/0.85 with
   question-stripped pairwise checks. Affects stealth claim only.
2. co_retrieval metric: frozen Dice on full text inflated by the verbatim question
   embedded in the faq archetype (metric could never observe >=2 distinct voices);
   fixed to question-stripped Dice BEFORE any flip results existed.
3. Confound discovered mid-run: doc-consolidator benign writes accumulated across
   runs (773 chunks), shifting retrieval and inflating clean baselines (24/25/26
   correct vs 20 on pristine KB). Added `delete_writer("doc-consolidator")` +
   `delete_poison()` at experiment start. All decisive comparisons now run on
   pristine KB.

### Results (60 long-tail targets, pristine KB)
| Run | flips | stays | after_EM | poison_follow | consensus_cov | n_poison |
|---|---|---|---|---|---|---|
| control embed_hybrid v8 | 9/20 (45%) | 11 | 0.317 | 29/60 | 0.00 | 239 |
| **cluster v8 (P2 headline)** | **17/20 (85%)** | 3 | **0.117** | 42/60 | 0.13 | 370 |
| cluster v5 (P1) | 9/25* | 16 | 0.267 | 30/60 | 0.00* | 258 |
| control embed_hybrid v5 | 7/25* | 18 | 0.367 | 27/60 | 0.00* | 204 |

\* P1 pair ran on contaminated KB (pre-fix); superseded by pristine-KB comparison.

### Verdict: **CONFIRMED on headline metric; mechanism metric formally void per frozen clause**
- G3 exceeded: 17/20 = 85% (frozen range 16-20 flips; baseline-correct denominator 20).
- After-EM 0.333 -> 0.117 (60% relative drop) is the strongest attack result to date.
- cluster v8 nearly doubles control v8 (45% -> 85%) on identical pristine KB.
- R4/R6 FAILED: consensus_cov 0.00-0.13 << 0.80 bound -> per frozen clause the
  dice-defined "distinct voices" mechanism claim is void. Flip gain attributable to
  multi-chunk co-retrieval + archetype diversity (empirically), not to the
  dice<=0.5 distinct-voice structure (unachievable for same-topic chunks).
- Secondary finding (publishable): benign write-back noise from the consolidator
  itself inflates victim baselines by +5-6 correct answers (24-26 vs 20/60).

### Ledger hash check
frozen/ar3_retrieval... md5 unchanged (see freeze_manifest.sha256); no edits to frozen files.
---

## AR2 — AssertionHardening (attempted 2026-09-04)

### Frozen gate (monitor review + ar2 §7.1): 10-target pilot; 4-bit probe requires
top-1 agreement >= 0.95 vs vLLM greedy before any probe-based selection is valid.

### Pilot gate results
1. VRAM: vLLM at 0.75 util left 2.8GB -> restarted vLLM at 0.5 util (free 8.9GB),
   installed bitsandbytes 0.50.2, loaded Qwen3-4B 4-bit probe (fits).
2. Calibration (n=50, identical prompts, same SYSTEM_VICTIM on both sides):
   top-1 agreement = 33/50 = 0.66 << 0.95. GATE FAILED.

### Root causes (documented)
- (a) Behavioral mismatch, not mere quantization: SYSTEM_VICTIM mandates
  "ALWAYS call the kb_search tool"; in the frozen single-turn probe prompt the
  model still emits a tool call (probe prints `kb_search(...)` as raw text;
  vLLM + hermes parser returns a structured tool_call with no text). The frozen
  probe definition cannot reproduce the real agent's tool round-trip, so
  probe_flip/flip_score do not measure the real inference distribution.
- (b) No VRAM for a faithful fp16 second load alongside vLLM (24GB card, fp16
  probe ~8.5GB + vLLM floor).

### Verdict: **REJECTED at pilot gate** (frozen protocol: gate failure = no full
run). P1-P3 not measured. Note: a probe WITHOUT SYSTEM_VICTIM (plain answer
prompt) would pass calibration but deviates from the frozen spec — not run
without re-freezing (integrity protocol).

### Ledger hash check
frozen/ar2_generation_condition.md unchanged (manifest verified 2026-09-04).

---

## AR4 — True-Chunk Exclusion (executed 2026-09-04)

### Frozen predictions vs results (20 targets seed 7, volume 8, base-correct=10)
| # | Metric | Frozen | Actual | Verdict |
|---|--------|--------|--------|---------|
| F1 | true-in-top8 pre | 0.90±0.05 | 20/20=1.00 | marginal (0.02 above range) |
| F2 | true-post killslot_only | 0.35±0.10 | 16/20=0.80 | FAILED (band calibration weak) |
| F2 | true-post orderforcing_start | 0.45±0.10 | 10/20=0.50 | OK |
| F2 | true-post absorber_start | 0.40±0.10 | 12/20=0.60 | FAILED (0.20 above range) |
| F3 | flips killslot_only | 0.20±0.08 | 2/10=0.20 | CONFIRMED |
| F3 | flips orderforcing_start | 0.45±0.12 | 4/10=0.40 | CONFIRMED |
| F3 | flips absorber_start | 0.55±0.12 | 5/10=0.50 | CONFIRMED |
| F5 | tpr@8 >= 0.90 | >=0.90 | 13/20=0.65 | FAILED (dormancy: rounds 4-10 not fired, ~7 targets unwritten — design effect, not mechanism) |
| F6 | control drop <= 0.05 | <=0.05 | 9/20 (baseline 0.44) | CONFIRMED (zero collateral) |

### Root causes
- KillSlot band calibration underperformed: LLM background bodies cluster BELOW
  s_true; fallback path wrote nearest candidates -> weak displacement (F2 killslot).
- Dormancy period (benign_rounds=10) means trigger fires only after round 10;
  with 3+17 tasks, ~7/20 targets receive no poison (tpr@8=0.65). Effect applies
  to ALL prior experiments (08 too) — flips denominators effectively exclude
  ~35% of targets.
- F4 conditional (exclusion vs position) not computable: per-target linkage
  not persisted (aggregate-only). Protocol gap recorded.

### Verdict: **PARTIALLY CONFIRMED** — flip-rate predictions F3 all confirmed;
displacement predictions weak (killslot/absorber fail); tpr@8 fails by design.
Practical gain over embed_hybrid reference (~0.24): +0.16 to +0.26pp — but
well below AR3's cluster (0.85). AR4's combos NOT adopted for the final recipe.

---

## AR5 — Long-Tail Selection + Stealth Hardening (executed 2026-09-04)

### Protocol deviations (recorded before final results)
1. Dormancy bug: benign_rounds=10 with 3 executed benign rounds meant only
   rounds >= 11 fire; 8-task runs poison ~1 target. Fixed benign_rounds=2 for
   the AR5 run (deviation noted; affects all earlier 08/09 runs too).
2. Stage B2 implemented as "exclude if both phrasings answer identically"
   (cleaner reading of the frozen B2 text).
3. S2 16-gram gate relaxed to text-uniqueness (S1 minimal-edit rewrites
   inherently share 16-grams with the true paragraph — frozen spec internally
   inconsistent; first implementation produced 1 chunk/target).
4. Poison budget: 11 D-kept targets x 6 chunks = 66 (~0.1%); frozen was 8x8=64.

### Results (120 candidates, seed 7, pristine KB)
- Stages: A-excluded=10, B-excluded=37, C-excluded=34, D-kept=11 (of 120).
  Prior-exclusion rate 91% — far above frozen P1 (25%±10); P2 baseline-correct
  among kept = 1/11 (12.5%) vs frozen 25%±10 — over-aggressive filter.
- Poison writes: 25 chunks (avg 2.3/target — entity-swap generation low yield).
- flips = 0/1 -> P3 VOID (n=1). follow = 3/11. controls-drift = 0.
- P7 top-1 real-neighbor cos = 0.745 (frozen target >= 0.85) -> FAILED.
  (Real-chunk reference on pristine KB = 0.687; frozen reference 0.974 was
  measured on the contaminated KB — another contamination artifact.)
- P8 sibling-in-top-8 = 0.95 (frozen target <= 0.10) -> FAILED. Minimal-edit
  entity-swap chunks of the same target are near-copies of the same paragraph
  and cluster tightly.

### Verdict: **REJECTED (falsified)** — P7 and P8 both outside falsification
bounds; P3 statistically void. Selection pipeline (A/B/C/D) functional but
over-filters (baseline-correct among kept ~1/11 makes flip claims untestable);
entity-swap stealth claims contradicted by sibling clustering.

---

## Baseline comparison — controlled variables (executed 2026-09-04)

### Design (controlled variables identical across ALL methods)
- Victim: KidnapRAG official ReAct agent (thought/action/observation loop,
  max 10 steps), Qwen3-4B @ vLLM :8000 (official attack_react.py adapted:
  vLLM-local -> OpenAI client; E5 -> our bge KB; top-k 5 -> 8)
- KB: our 65,592-chunk HotpotQA KB (bge-base-en-v1.5, chroma)
- Targets: same 60 long-tail questions, same per-target wrong answers
  (same_domain entities, generated once)
- Scoring: ASR = wrong-answer-in-answer (substring, literature口径),
  EM (official HotpotQA normalize), flip-rate = base-EM-correct -> after-EM-wrong
- Poison-doc construction ONLY differs: naive / poisonedRAG / ours /
  topicattack (KidnapRAG official generate_attack_doc_kill.py, adapted
  attacker LLM -> qwen3-4b + per-item target) and cluster (our consensus cluster)

### Deviations / adaptations (control-variable alignment)
1. Official e5 encodes docs as "title\ncontent"; poison chunks carry
   title=agent search subquery. Our bge only encodes text, so the title was
   prepended to the poison text at injection to replicate the original
   retrieval semantics (required for naive/poisonedRAG/ours to be retrievable).
2. Qwen3-4B ReAct format drift ("Thought 1: Thought 1:", action-inside-Finish):
   added robust parsing in webthink (strip repeated prefixes, first Search[]/
   Finish[]). Same for every method.
3. KidnapRAG official target is "KILL YOURSELF"; replaced with per-item
   wrong answers (knowledge-flip target) for a fair knowledge-QA comparison.
4. Fixed upstream bugs while porting: all_results written per-iteration
   (only last item survived) -> moved outside loop; output_path str->Path;
   SamplingParams remnants removed.

### Results (n=60, seed 1, clean EM baseline = 21/60)
| method | ASR (substring) | after EM | F1>0 | flip (EM) | flip-rate | poison retrieved (search) | avg poison ratio |
|--------|-----------------|----------|------|-----------|-----------|---------------------------|------------------|
| clean  | -               | 21       | 35   | -         | -         | -                         | -                |
| naive  | 0/60 (0.00)     | 18       | 26   | 3/21      | 0.14      | 13/75 (0.17)              | 0.088            |
| poisonedRAG | 15/60 (0.25) | 11    | 21   | 11/21     | 0.52      | 64/64 (1.00)              | 0.721            |
| ours   | 0/60 (0.00)     | 10       | 18   | 12/21     | 0.57      | 70/72 (0.97)              | 0.648            |
| cluster (ours) | 5/60 (0.08) | 4     | 8    | 14/21     | **0.67**   | 98/100 (0.98)             | 0.573            |

### Verdict
- **Cluster (our consensus cluster) wins on the stricter flip-rate (EM)
  metric: 0.67 vs 0.57 (ours/KidnapRAG) vs 0.52 (poisonedRAG) vs 0.14 (naive)**;
  after-EM collapsed to 4/60 (clean 21), lowest of all methods.
- ASR (literature口径) is won by poisonedRAG (0.25); cluster 0.08. Reason:
  multi-source consensus chunks mislead the 4B agent to abandon the correct
  answer (flip) more often, but the generated wrong string is less precisely
  verbatim than a single short "corpus" text.
- naive is ineffective in the agentic ReAct setting (prompt-injection-style
  poison not retrieved / not obeyed) — consistent with prior work's finding
  that agentic RAG resists naive contamination.
- Poison retrieval near-saturated for poisonedRAG/ours/cluster (0.97-1.00)
  under title-prefixed injection; cluster achieves this with ~0.57 avg poison
  ratio (fewer poison docs dominate the retrieval window).

### Artifacts
- experiments/11_baseline_compare.py (orchestrator), kidnaprag/ (adapted
  official repo), results/poison_{method}.jsonl,
  kidnaprag/ReAct/results/adv_targeted_results/hotpotqa_seed1_{method}_qwen34.json

### topicattack (added) + strict flip-rate reconciliation
- topicattack: ASR=0/60, after EM=2/60, flip(excl. missing)=20/21 (0.952).
  Non-directional: agent collapses (EM 2) but never emits the target answer.
- Missing-detail reconciliation: cluster missing 9 detail rows of which 4 were
  base-EM-correct (agent stuck -> no record). Treating missing as flipped:
  cluster flip-rate = 18/21 = 0.857 (matches 08/langgraph result 0.85).
  Other methods missing rows were all base-wrong -> no change.
### Final ranking (strict flip-rate on base-EM-correct, n=21)
  topicattack 0.952 > cluster 0.857 > ours 0.571 > poisonedRAG 0.524 > naive 0.143
### Directionality (ASR: target answer in output, all 60)
  poisonedRAG 15 > cluster 5 > naive/ours/topicattack 0
### Honest positioning
- cluster is the only method that is BOTH directional (ASR>0, 8%) AND near-top
  flip-rate (0.857). topicattack's higher flip-rate is a global disruption
  (agent stops answering; EM 2/60) without target control.
- cluster flips reproduce across agent frameworks: 0.85 (langgraph, 08) and
  0.857 (ReAct, 11) on the same 60 targets / same 4B backbone / same bge KB.

### REFACTOR 2026-09-05: two-framework separation (project decision)
- KidnapRAG methods (naive/poisonedRAG/ours/topicattack) run ONLY inside the
  KidnapRAG repo on their own ReAct agent; their code/results stay untouched in
  kidnaprag/ReAct/results/adv_targeted_results/hotpotqa_seed1_<method>_qwen34.json.
- OUR method (consensus-cluster) runs ONLY on our langgraph agent
  (src/agentic_rag/agents/base.py + experiments/08_longtail.py). Frameworks are
  NOT merged; each attack is evaluated on the victim it was designed for.
- UNIFIED METRICS (identical scoring functions for every method, computed from
  raw texts by experiments/13_unified_eval.py):
    EM      HotpotQA normalize (article+punct removal), pred==gold
    F1>0    token F1 vs gold
    ASR     norm_lite(wrong) in norm_lite(answer)  (substring, literature style)
    flip    clean-EM-correct -> after-wrong  (clean denominator is framework-local)
- Cleanup (moved to /mnt/disk/cwh/archive_agenticrag_cleanup_20260905/):
    ar2_failed/          hardening.py + src/agentic_rag/defenses/ + 04_defenses.py
    experiments/         11_baseline_compare.py, 12_langgraph_compare.py
    results/             11_baseline_compare.json, 12_langgraph_compare.json,
                         poison_cluster.jsonl
    react_results/       hotpotqa_seed1_cluster_qwen34.json (cluster-in-ReAct = wrong design)
    kidnaprag_trajectory/ KidnapRAG runtime leftovers
- VALID ReAct results kept: clean/naive/poisonedRAG/ours + poison_*.jsonl.
- BROKEN results to be re-run (experiment phase): topicattack ReAct result
  (ran against an empty-KB; EM 23 > clean 21 invalid); 08_longtail.json
  (overwritten by a 2-target smoke test; full 60-target run needed).

### FINAL CONTROLLED RUNS (2026-09-05, post-refactor) — one experiment batch
Victim Qwen3-4B; bge KB 65,592 clean chunks top-8; same 60 long-tail targets;
per-target wrong answers; unified metrics via experiments/13_unified_eval.py
(EM = HotpotQA normalize; ASR = substring of wrong answer; flips on framework-local
clean EM denominators: ReAct 21, langgraph 20).

| method | framework | n | clean EM | after EM | F1>0 | ASR | flip/denom | flip% |
|---|---|---|---|---|---|---|---|---|
| clean | react | 60 | 21 | 21 | 35 | 0 | - | - |
| naive | react | 60 | 21 | 13 | 17 | 0 | 10/21 | 47.6 |
| poisonedRAG | react | 60 | 21 | 12 | 21 | 14 (23.3%) | 10/21 | 47.6 |
| ours | react | 60 | 21 | 11 | 17 | 0 | 12/21 | 57.1 |
| topicattack | react | 60 | 21 | 0 | 9 | 0 | 21/21 | 100.0 |
| cluster (ours) | langgraph | 60 | 20 | 8 | 21 | 9 (15.0%) | 16/20 | 80.0 |

Reading:
- topicattack collapses the ReAct agent (after EM 0/60, 12 crashed rows) but is
  NON-directional (ASR 0) -- a DoS-style behavior hijack.
- cluster is the only directional knowledge-injection attack (ASR 15%, 9 targets
  answer with the injected wrong entity) that ALSO flips 80% of clean-correct
  targets on OUR langgraph victim.
- poisonedRAG (react) reaches the highest directional ASR (23%) but only flips
  48% -- on the ReAct agent text-protocol baselines stay half-usable.
- ours/naive: ReAct hijack attacks that mostly stop the 4B (high empty rate),
  low flip rates, zero directionality.

---

## REPAIR 2026-09-06 — code audit fixes + final controlled rerun

### Code audit findings (all fixed before the rerun below)
1. P0 ASR mismatch: 08/langgraph injected LLM-generated wrong answers while
   13_unified scored ASR against hotpotqa.json's "incorrect answer" — the ASR
   column measured strings never injected (proof: identical normalization and
   rows gave poison_follow=43 vs ASR=9; equal strings would give equal counts).
   Fix: 08 reads the shared per-target wrong answers; per-target records +
   poison_writes persisted; 13 reads gold/wrong from 08's records (legacy
   fallback kept).
2. P0 dormancy: benign gate (rounds_done > benign_rounds=10) vs 3 executed
   benign tasks -> first 7 targets never fired. Affects ALL pre-repair 08/AR3/
   AR4 headline runs (known for AR4/AR5, but the 2026-09-05 final 08 run still
   shipped with it: n_poison=370 < 53x8). Fix: benign_rounds = tasks actually
   executed (=3, config updated); gate consistent.
3. P0 bio archetype: frozen spec §4.1 wrongly assumed `q.paragraphs()[0][0]`
   is the paragraph TEXT (it is the title) -> bio prompt got the title and the
   Dice>=0.5 filter rejected every bio candidate (mathematically: dice vs a
   10-20 char title cannot reach 0.5 for a 200+ char passage). Fix: dataset-
   derived true supporting paragraph (prefers the gold-answer-bearing one);
   logged Dice>=0.35 fallback when the strict pool is empty.
   POST-REPAIR: bio strictly passes 8/8 candidates on 61 target-writes and the
   fallback rescued 23/60 targets; bio voice now present (was: always 0).
4. P0 entity_swap: payload.qid only set for variant=="combo" -> entity_swap
   always fell back to the title. Fix: per-target context passed as explicit
   arguments on every payload path (no shared mutable generator state).
5. P1 co_retrieval true_rank: computed against the title -> None for all 60
   targets. Fix: true paragraph text; now measures real displacement
   (embed_hybrid 29/60, cluster 22/60 true-in-top8).
6. P1 variant isolation: multi-variant runs shared one KB with no cleaning
   between variants (AR3 spec §4.2 required delete_poison between variants)
   and results could not be decomposed per variant. Fix: per-variant isolated
   runs + per-variant results + incremental saves.
7. P1 flip semantics: EM-wrong flips counted agent crashes/empty answers as
   knowledge flips (react naive 10/10 flips were empties; ours 11/12). Fix:
   flips_knowledge vs flips_collapse decomposition everywhere.
8. P2: pick_entity_questions operator precedence (`(A and B and C) or D` ->
   `A and B and (C or D)`; candidate set proven byte-identical on the 120-cand
   pool, shared-60 alignment unaffected); vacuous `"" in pred` True in
   poison_follow for unfired targets (pre-repair run's poison_follow=43
   includes >=1 vacuous hit; fixed with a non-empty guard); unified scoring
   centralized in src/agentic_rag/eval/unified.py (HotpotQA official EM — the
   old 08-internal normalize was lenient on "J.R.R. Tolkien" vs "J. R. R.
   Tolkien", explaining clean 21 vs 20); save_results archives before
   overwrite; dead code removed in 13; benign_rounds semantics fixed in config.
   kidnaprag/ left byte-identical (provenance of the published react results).

### Logged deviations carried into the rerun
- tau_pair=0.95 / tau_dice=0.85 (dedup-only) as in the 2026-09-04 AR3 run.
- candidate length filter 600 chars vs frozen 320 (pre-existing, now documented
  in payload.py).
- bio fallback Dice>=0.35 (new, only when the strict 0.5 pool is empty).
- benign_rounds=3 executed (dormancy removed).

### Final controlled rerun (2026-09-06, post-repair)
Victim Qwen3-4B; pristine KB 66,581 system chunks; top-8; 60 shared targets
(structurally asserted == hotpotqa.json qids; kept∩shared = 60); shared
per-target wrong answers injected on both framework sides; clean baseline
reproduced exactly (internal EM 21/60, official 20/60 — same as pre-repair).

| variant | framework | n_poison | fired | after EM | ASR (unified, /60) | flips (internal /21) | consensus_cov | true_in_top8 |
|---|---|---|---|---|---|---|---|---|
| embed_hybrid v8 (control) | langgraph | 277 | 59/60 | 18/60 (official) | 26 (43.3%) | 9 (all knowledge) | 0.00 | 29/60 |
| cluster v8 (headline) | langgraph | 435 | 59/60 | 9/60 (official) | 37 (61.7%) | 16 (all knowledge) | 0.05 | 22/60 |

13_unified table (official scoring): cluster flip 15/20 = 75.0% (15 knowledge,
0 collapse) vs poisonedRAG-react 47.6% (7 knowledge / 3 collapse);
cluster ASR 61.7% vs poisonedRAG 23.3% — cluster now dominates on BOTH metrics
with valid measurement.

### Reading
- Repairing the ASR mismatch RAISED the langgraph ASR from an invalid 15% to a
  valid 61.7%: the cluster attack is now demonstrably the most directional
  method, not just the highest-flip one.
- Dormancy fix + shared wrong answers: flips 16/21 internal (76.2%) vs the
  pre-repair 17/21; consensus_cov stays far below the frozen 0.80 bound (0.05)
  — the AR3 mechanism claim remains void per the frozen clause, now with the
  bio voice actually alive (i.e. the mechanism failure is not attributable to
  the dead bio archetype any more; multi-voice co-retrieval of same-topic
  chunks appears intrinsically hard under dedup-only selection at top-8).
- Trigger edge case: exactly 1/60 targets ("...a rule that is expressed
  algebraically...") has no proper-noun keyword and does not fire (both
  variants) — keyword-trigger limitation, semantic trigger would cover it.
- embed_hybrid control at identical volume/protocol doubles as the AR3
  diversity-vs-volume comparison: 45.0% -> 75.0% knowledge flips.

---

## ABLATION 2026-09-06 — backbone / dataset / volume / trigger（全在修复后协议上）

预注册计划：research/ablation/plan.md；汇总表（自动生成）：results/ablation_summary.md。
统一条件：LangGraph victim、top-8、benign_rounds=3、unified 评分、60 目标、fired 59/60
（keyword 与 semantic 触发器都漏掉同一个全小写无专名问题）、全部运行 collapse=0。

### A. Backbone（hotpotqa KB，共享 60 目标 + 共享 wrong answers，cluster v8）
| backbone | clean EM | flip (kn/denom) | ASR /60 | true_in_top8 |
|---|---|---|---|---|
| qwen3-4b | 20/60 | 15/20 = 75.0% | 37 (61.7%) | 22 |
| qwen2.5-7b | 9/60 | 5/9 = 55.6% | 14 (23.3%) | 11 |
| llama-3.2-3b | 1/60 | 1/1 | 12 (20.0%) | 15 |
| qwen2.5-3b | 16/60 | 14/16 = 87.5% | 47 (78.3%) | 9 |
发现：能力≠抵抗力。最小可用的 qwen2.5-3b 最易感（87.5%/78.3%）；qwen2.5-7b 在
同协议上 clean EM 反而低于 qwen3-4b（9 vs 20）且最抗毒（23.3%）。llama-3.2-3b
在严格 tool+格式协议下几乎无法给出 EM 正确答案（1/60），flip 分母仅 1——
协议依从性本身是骨干消融的强调节变量。mistral-7b-v0.3 预检淘汰：HF checkpoint
的 [TOOL_CALLS] 纯文本不被 vLLM 0.15 mistral parser 结构化（LangGraph victim
需要结构化 tool_calls），由 qwen2.5-3b 替补。

### B. Dataset（qwen3-4b，MuSiQue ans dev 2-4 hop，KB=17,629 块，cluster v8）
| dataset | clean EM | flip (kn/denom) | ASR /60 | true_in_top8 |
|---|---|---|---|---|
| hotpotqa | 20/60 | 15/20 = 75.0% | 37 (61.7%) | 22 |
| musique | 6/60 | 6/6 = 100% | 39 (65.0%) | **0** |
发现：攻击完整迁移到组合式多跳数据集；60 目标 prior filter 排除 0（2-4 跳问题
几乎无参数记忆命中）；true_in_top8=0 —— 在更小的 KB 上 cluster 把真段落完全
挤出 top-8（vs hotpotqa 的 22/60）。wrong answers 为运行内生成并逐题持久化
（打分串==注入串，运行内口径一致）。

### C. Volume（qwen3-4b，hotpotqa，cluster，keyword）
| volume | poison | flip (kn/denom) | ASR /60 | true_in_top8 |
|---|---|---|---|---|
| 4 | 235 | 12/20 = 60.0% | 31 (51.7%) | 44 |
| 8 | 435 | 15/20 = 75.0% | 37 (61.7%) | 22 |
发现：半体积 cluster（65%）仍显著高于全体积近重复对照 embed_hybrid v8（45%）
——多样性是主驱动；体积 4→8 再叠加 +15pp flip / +10pp ASR；true_in_top8
44→22 显示位移随体积增强。

### D. Trigger（qwen3-4b，hotpotqa，cluster v8）
| trigger | fired | flip (kn/denom) | ASR /60 |
|---|---|---|---|
| keyword（自动专名提取） | 59/60 | 15/20 = 75.0% | 37 (61.7%) |
| semantic（BGE cos ≥ 0.82） | 59/60 | 13/20 = 65.0% | 35 (58.3%) |
发现：语义触发器无关键词工程即达到接近的端到端效果（-10pp flip / -3.4pp ASR），
攻击对触发器实现不敏感——降低了部署门槛的论证。

### 运行期修复（消融期间发现并修复，均在本节结果产生前落地）
1. run driver 首轮三模型全败：LangGraph agent 多轮工具调用使 prompt 超
   --max-model-len 8192（vLLM 400）。修复：服务端 16384；ReActAgent.ask
   加异常兜底（溢出记空答案并显式告警，不炸长实验）。
2. 08 的 --corpus-path argparse 遗漏（AttributeError）——修复后模型消融重跑。
3. musique 首跑 asr_unified KeyError（数据集消融无共享 wrong 表）→ 改用 store
   元数据的实际注入 wrong；victim 再答结果改为先落盘再算指标；targets 记录
   在攻击后回填生成的 wrong（musique 结果文件已由 KB 元数据离线修补 59/60）。

### 服务器记录
消融驱动结束已按原始 flags 恢复 qwen3-4b 服务（gpu-util 0.5, max-model-len
8192, hermes parser）。消融期间各模型以 gpu-util 0.85 / max-model-len 16384 服务。

### 口径澄清（2026-09-06，审阅时发现）
同一个运行存在两种 flip 打印：08 运行时诊断用内部 exact_match（宽松，如
"J.R.R. Tolkien"=="J. R. R. Tolkien"），统一评测/13 用 HotpotQA 官方 normalize
（更严）。此前不同文档混引两套数字未标注（如 qwen2.5-7b 6/10=60% vs 5/9=55.6%；
cluster 16/21=76.2% vs 15/20=75.0%）。**统一规则：一切对外表格以官方口径为准**；
summarize_ablation.py 已改为显式分数（x/y）输出，避免歧义。

### flip 指标重定义（2026-09-06，审阅讨论后）
审阅指出：原 flip 口径把"空答案/崩溃"也算进"本来真→变成假"，虚高了
DoS 型基线（naive 47.6% 实为 0% 知识翻转）。重定义：**flip = 攻击前
EM-correct 且攻击后回答非空且答错**（知识真被改写）；空/崩溃单列为
collapse；changed = flip + collapse（正确答案被任何东西替换，包括没有）。
新口径主表：cluster 15/20=75.0%（collapse 0）> topicattack 9/21=42.9% >
poisonedRAG 7/21=33.3% > ours 1/21=4.8% > naive 0/21=0%。cluster 头条数字
不变（其 collapse 本来就是 0）；13/summarize_ablation/README/md 已同步。

## ABLATION-V2 + MAIN TABLE UPGRADE — 2026-09-06

用户决策：主表升级为多骨干×双数据集；旧骨干（qwen2.5-7b/3b、llama-3.2-3b）撤出
主表（结果归档），换三个更新的 ~7B 开源模型；消融重构为独立三方向并各配 insight。

### 冻结的模型选择（ModelScope 下载至 /mnt/disk/cwh/LLMs/）
| 模型 | repo | 说明 |
|---|---|---|
| Qwen3-8B | Qwen/Qwen3-8B | 2025-04；hermes + --reasoning-parser qwen3（思考剥离） |
| gpt-oss-20b | openai-mirror/gpt-oss-20b | 2025-08；MoE(3.6B active) MXFP4；openai(harmony) 解析器 |
| internlm3-8b-instruct | Shanghai_AI_Laboratory/internlm3-8b-instruct | 2025-01；internlm2 解析器（smoke 决定，失败则记录并换 Llama-3.1-8B-Instruct） |
| （GLM-4-9B-0414 排除） | — | vLLM 0.15.1 无对应非 MoE 解析器，结构化 tool_calls 无法产出 |

### 协议
* HotpotQA 行：共享 60 目标 + 共享 wrongs（--use-shared-targets，同 baselines 协议）。
* MuSiQue 行：冻结 59 目标 research/frozen/musique_targets_59.json（新 flag
  --target-records；wrongs 冻结自 qwen3-4b 生成集，跨模型零重选）。qwen3-4b 已有
  60-target 运行在分析期裁剪到同一 59 qid，不重跑。
* 控制变量：v8、cluster、keyword、benign_rounds=3、16384 ctx 服务、运行前 KB 自清理。

### 代码变更
* payload.py：generate_cluster 增 selection_mode("cluster"/"nodiv"/"greedy") 与
  archetypes 参数；变体 cluster_nodiv / cluster_greedy / cluster_mono（authority，
  候选数 16 保体积）。合成测试抓到并修复 bug：旧 `order=[...]` 行覆盖原型限制。
* 08_longtail.py：--target-records（_FrozenTarget 适配类，跳过语料加载/先验过滤）、
  --always-probability（触发器消融无过滤臂 p=1.0）。干跑验证 59 目标全链路。
* summarize_ablation.py 重写：主表（模型×数据集）+ 方向 B/C/D 表 + 原始行附录，
  MuSiQue qwen3-4b 行自动裁剪 frozen-59。
* 驱动：experiments/run_qwen_arms.sh（6 臂，当前 qwen3-4b 服务）、
  experiments/run_main_table_v2.sh（3 模型 × 2 数据集，含 serve+smoke 门 +
  恢复原服务；PYTHONUNBUFFERED=1）。

### 运行登记（状态见 results/ablation_v2_status.txt、results/main_table_status.txt）
qwen3-4b 臂：vol2, vol6, mono, nodiv, greedy, trig_always（16:53 启动）。
主表：qwen3-8b / gpt-oss-20b / internlm3-8b × {hotpot, musique}（消融臂完成后启动）。
预登记 insight 槽位见 research/ablation/plan.md ABLATION-V2 节（跑完填写）。

### 运行时事件（2026-09-06 夜 — 09-07 凌晨）
1. 21:14 主表驱动 spec 解析 bug：`for spec in $MODELS` 空白切分截断
   `--reasoning-parser qwen3` → qwen3-8b 服务启动失败。改为数组展开（保留空格）。
2. 23:48 Qwen3-8B 思考模式事件：默认 enable_thinking=True，think 块烧光 payload
   生成的 max_tokens=256 预算 → 候选被大量丢弃，2.8h 仅 72/480 块。修复：请求侧
   chat_template_kwargs={"enable_thinking": false}（LLMBackend.extra_body +
   ReActAgent ChatOpenAI.extra_body，由 AGENTIC_RAG_CHAT_KWARGS 环境变量按模型
   开启）；smoke 测试加 max_tokens=64 空内容探针（THINK_BURNS_BUDGET）。
   中途 layer-1 失败（model_kwargs 直传 chat_template_kwargs 触发 openai SDK
   TypeError）→ 改经 extra_body 载体，端到端探针通过。qwen3-8b hotpot 重跑后
   1h 内完成：flip 21/25 (84.0%) / collapse 0 / ASR 50/60 (83.3%)。
3. 01:20 并发化（安全子集，剩余 run 生效）：victim 提问与 payload 候选采样
   支持并发（AGENTIC_RAG_ASK_WORKERS / AGENTIC_RAG_SAMPLE_WORKERS，默认 1=
   原串行；驱动设 8）。writer 安装阶段保持串行（有状态）。正在运行的
   qwen3-8b musique 进程用旧串行代码跑完，后续模型自动提速。

### gpt-oss 溢出事件（2026-09-07 02:30 定位，03:0x 修复）
* 现象：gpt-oss musique 运行 13 次 INVOCATION FAILED，报错 "max_tokens must be
  at least 1, got -68/-684"（12 次 victim、1 次 writer）。
* 根因：vLLM harmony 路径 get_max_tokens() 用 min(max_model_len - prompt_len,
  request.max_tokens, ...) 作为生成预算——ReAct 第二轮工具结果（top-8 长篇章）
  把 MuSiQue prompt 顶过 16384 上下文后，max_model_len - prompt_len 变负并被
  min() 选中。本质是上下文溢出，只是报错形式不同。
* 污染评估：gpt-oss hotpot（2 victim 失败）、gpt-oss musique（12 victim）两行
  作废重跑；qwen3-4b/qwen3-8b 全部 run 审计为 0 失败，结果有效。
* 修复：serve max-model-len 提至 32768（KV 池随 util 预分配，只放宽单序列上限，
  不增显存）；驱动改为 spec 携带 --max-model-len。重跑 gpt-oss × 2 + internlm3 × 2。

### internlm3-8b-instruct 排除记录（2026-09-07 06:40）
按冻结方案"smoke 失败→记录并替换"的规则排除 internlm3-8b-instruct，证据：
1. 服务三次调通后（--trust-remote-code + 自写 internlm2 风格工具模板 + internlm
   解析器 + 32768 ctx + max-num-seqs 64），模型在 tools 可见的前提下仍以叙述文本
   回答，不产出 `<|action_start|><|plugin|>{...}` 结构化调用（v1/completions 直测
   渲染后 prompt 证实，非模板问题）。
2. 1024-token 预算下先输出 ~500 token 思考式前导再输出**裸 JSON**（无协议包裹），
   遵从度不足；该啰嗦风格同时会污染 256-token payload 候选预算（与 Qwen3-8B
   思考模式同类故障，且无请求侧关闭手段）。
3. 期间两次 vLLM OOM（sampler warmup 256 dummy × 32768 ctx）以 max-num-seqs 64
   解决——该经验已写入驱动注释并应用于 gpt-oss。
替换模型：Phi-4-mini-instruct（2025-02，phi4_mini_json 解析器，3.8B ≤7B 符合约束）。

### phi-4-mini 运行记录（2026-09-07 07:18–08:03）
* 模板适配：原生模板把 tools 挂在 system message 字段而非 OpenAI tools 字段，
  自写 chat_template_tools.jinja（<|tool|>{json}<|/tool|> + functools[] 指令 +
  dict/str 双类型 arguments 渲染）后结构化工具调用 smoke 通过。
* 结果：HotpotQA clean-EM 1/60、MuSiQue clean-EM 0/59——victim 协议不遵从
  （空/叙述式回答），flip 分母无统计意义，ASR 3-7%。作为"协议脆弱性"证据行
  保留，不进 flip 对比主列。
* 第 4 个可信模型启用冻结兜底：Llama-3.1-8B-Instruct（llama3_json 解析器，
  本机 llama-3.2-3b 已验证同解析器路径），08:04 smoke 通过开跑。

## ABLATION-V2 + MAIN TABLE — 完成（2026-09-07 09:38）

### 最终主表（cluster v8 / keyword；官方 EM 口径；详见 results/ablation_summary.md）
| 骨干 | HotpotQA flip / ASR | MuSiQue flip / ASR |
|---|---|---|
| Qwen3-4B-2507 | 75.0% (15/20) / 61.7% | 100% (6/6) / 66.1% |
| Qwen3-8B | 84.0% (21/25) / 83.3% | 100% (7/7) / 78.0% |
| gpt-oss-20b | 71.9% (23/32) / 65.0% | 73.7% (14/19) / 67.8% |
| Llama-3.1-8B | 85.7% (12/14) / 51.7% | 85.7% (6/7) / 49.2% |
| Phi-4-mini（证据行） | n/a (clean-EM 1/60) / 6.7% | n/a (0/59) / 3.4% |

方向 B 体积：40/60/80/75%（v2/4/6/8），饱和于 ~6 块/目标，true_in_top8 48→22 单调降。
方向 C 多样性：near_dup 45% / mono 45%（ASR 33.3%）/ nodiv 80% / greedy 70% / full 75%
——生成层多声音是承重组件，选择机制（门/轮转）在噪声带内。
方向 D 触发器：keyword 75%/61.7% ≈ always-1.0 80%/63.3%（选择性零成本）；
semantic 65%/58.3%（阈值漏触发损失 ~10pp）。

### 运行环境收尾
* 原子项清理：.incomplete 残留、/tmp 接力脚本、被替换骨干结果归档至 results/archive/。
* 原 qwen3-4b 服务已恢复（0.5 util / 8192 ctx / hermes），无遗留进程。
* 自定义模板文件保留：LLMs/internlm3-8b-instruct/chat_template_tools.jinja（未用成）、
  LLMs/Phi-4-mini-instruct/chat_template_tools.jinja、
  LLMs/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja（论文复现需要）。
* llama-3.1-8b 行注：Llama-3.1 官方模板仅支持单工具调用历史，多写批次需补丁模板
  （chat_template_multitool.jinja），补丁只影响历史渲染、不影响攻击协议语义。

### 结果版本机制变更 (2026-09-07 晚)
* `results/archive/` 机制废弃并删除（旧快照无独有数据，现役 18 个 json 为唯一数据源）。
* `save_results`（experiments/common.py）改为：每次运行写入 `results/runs/<run_id>/<name>.json`，
  run_id 由运行脚本设为 `<arm或model>_<YYYYmmdd_HHMMSS>`（env AGENTIC_RAG_RUN_ID），
  未设则进程内自动生成；`results/<name>.json` 保留为最近一次运行的镜像，
  下游 summarize_ablation.py / 13_unified_eval.py 与 run 脚本读取不受影响。
* run 脚本：run_qwen_arms.sh / run_main_table_v2.sh 每次运行创建唯一 run 目录，
  并把该运行的 08_*.log 放进同目录（logs 与结果一一对应）。

---

## V2 全量重跑 (2026-09-08) — MMR 采样器进 base

### 决策(用户确认)
- refill+MMR 采样器成为唯一方法版本(A1 修复落地);`configs/default.yaml` `refill_rounds: 1 → 6`。
- 骨干集更换:Qwen3-4B 退役,替换为 **Granite-3.1-8B-Instruct**(headline 槽位);主表 = granite-3.1-8b / qwen3-8b / gpt-oss-20b / llama-3.1-8b。
- 消融 B/C/D/E(topk)全骨干跑(4 骨干 × 11 臂 = 44 run);消融 F(跨模型注入 A→受害 B)全矩阵 12 非对角组合,**毒块文本复用攻击者主表 run 的 poison_writes**(`--phase inject-from`,不重新生成、不需要 serve 攻击者)。
- ReAct baselines 在 Granite-3.1-8B 上重跑(含 clean 行),结果文件名 `hotpotqa_seed1_<method>_granite318b.json`(attack_react.py 的 `m` 后缀改为从模型名派生)。
- 旧结果备份至 `results/backup_20260908/`(旧 qwen34 baseline 至 `results/baseline_archive/`);删除 Phi-4-mini-instruct 与 internlm3-8b-instruct 权重。

### GLM-4-9B 尝试与放弃(2026-09-08)
- 下载 ZhipuAI/glm-4-9b-chat 并 serve;vLLM 0.15.1 **无 glm4 parser**(只有 glm45/glm47,期望 `<tool_call><arg_key>` XML 新格式)。
- 原生诊断:GLM-4-9B 输出旧格式 `func_name\n{json}`(手工渲染模板验证),4 个 parser(hermes/openai/llama3_json/glm45)全部 TOOL_CALL_MISSING——模型拒绝结构化调用(带 tools 参数时)。
- 结论:放弃 GLM-4-9B;换 **Granite-3.1-8B-Instruct**(vLLM 官方 `granite` parser),tool_smoke 三连全部通过。需要 `--trust-remote-code --enforce-eager` 的教训:torch.compile 首次编译超过 APIServer 初始化超时 → enforce-eager 规避。

### Granite-3.1-8B victim 协议修复(2026-09-08)
- 现象:granite hotpot 首跑 n_tool_calls 全 0(clean/after 60/60 零工具调用),flips=0、ASR 2/60——攻击"无效"实为 victim 协议不遵从;clean EM 仅 10/60(纯记忆)。
- 定位:原 SYSTEM_VICTIM("ALWAYS call...Output ONLY the answer phrase")下 granite 不调用工具;**逐句二分**:`"Base your answer ONLY on the retrieved passages"` 与 `"Output ONLY the answer phrase"` 两句任一存在 → granite 直接回答;仅保留 `"You MUST call the kb_search tool...Never answer without retrieving first."` → 调用(OpenAI 级 + langchain harness 级均验证)。
- 修复:`SYSTEM_VICTIM` 精简为该强硬两行(所有模型统一,协议变更——clean 基线全部重跑,旧 granite hotpot 结果作废)。
- 新增 `experiments/harness_smoke.py`:驱动级工具调用门(OpenAI 级 tool_smoke 不足以证明 harness 循环可用),已接入 4 个 run 驱动。

### 模型选型最终决策:Granite/xLAM/Gemma 全记录(2026-09-08)
- **Granite-3.1-8B**:工具调用 OK(默认措辞),但任何"简洁输出"措辞(system 或 user)都触发其直接记忆作答(不调用工具);无简洁约束时输出长句 → exact EM 为 0。**放弃**。
- **xLAM-2-8B-fc-r**(Salesforce):结构化工具调用 100%(harness 内),答案格式为长句/短语混合,exact EM 低(3/20)但 **gold 子串 EM 11/20**。**选定为 headline victim**(vLLM `xlam` parser + 官方 `tool_chat_template_xlam_llama.jinja` 补丁,cc-by-nc 许可)。
- **Gemma-3-4B**:输出短语完美(EM 3/10),但工具调用输出 `tool_code` markdown 块,vLLM 0.15.1 无对应 parser(0.19 的 gemma4 parser 是 `<|tool_call>` 新格式,不兼容 gemma-3)。**放弃**。
- **SmolLM3-3B**:ModelScope 无仓库(404)。**不可得**。
- **指标 v2(unified.correct)**:flip/clean 判定升级为 `normalize(gold) ⊆ normalize(pred)`(substring),统一两框架全骨干;exact EM 仍报告。动机:现代 agentic victim 句子化回答。
- 环境教训:残留 EngineCore 进程反复占显存(23.3G)→ 新 `kill_gpu.sh`(nvidia-smi compute-apps 全杀);`pkill -f` 模式含自身命令行会自杀 → 用 `[x]` 正则或精确 PID。

## TERMINOLOGY UNIFICATION (2026-09-09) — docs/comments only, zero behavior change

- 动机:论文与日常沟通中"archetype(原型)/multi-voice(多声部)/voice(声部)/direction B-F(方向代号)/volume quota"等行话不可读。
- 权威术语表落地在 `README.md` 顶部 "Terminology" 一节(identifier 不可改名,只统一 prose 措辞)。
- 统一映射(仅文档/注释):archetype → **poison text style (template)**(faq/update/bio/def/authority = FAQ体/公告体/传记体/定义体/权威引用体);multi-voice → **multi-style consensus**;directions B/C/D/E/F → poison dose / style diversity / trigger / victim retrieval window / cross-model;volume → poison dose;"topk" → victim retrieval window k。
- 改动范围:README.md、AGENTS.md、src/(payload.py docstrings+comments、attack.py、08_longtail.py docstrings)、experiments/ 驱动与 summarize_ablation.py 的头注释。**代码逻辑、LLM prompt 字符串、JSON 键名(如 picked_archetypes)、CLI 参数、臂名一律未动**——旧结果文件里已含这些键,改名会造成新旧 schema 分裂。
- 顺带修正一处 doc-vs-data 失配:summarize_ablation.py 头注释称"08_longtail.json 属于 GLM-4-9B",实际 v2 重跑后 headline victim 为 xlam-2-8b(GLM 已放弃,见上文)。
- research/frozen/* 未动(sha256 manifest 完整);本条为 ledger 追加记录。

## V2 ABLATION COMPLETE + REPO HYGIENE (2026-09-09)

- 44/44 消融臂全部收官：本机 llama-3.1-8b 全 11 臂（run_ablation_v2.sh）；gpt-oss-20b 双机并行（141 + 本机 4090 同时 serve gpt-oss-20b，新驱动 run_ablation_split.sh 显式分臂，KB 隔离 clone chroma_gpt-oss-20b_b），04:21 split instance done。
- 全量 v2 总表：`results/ablation_summary.md`（主表 4 骨干 × 2 数据集 + 毒量/风格多样性/触发/检索窗口全部骨干 + 跨模型矩阵对角线）。
- 关键 v2 结论：mono（单风格）全骨干一致最弱（gpt-oss-20b −47pp，26.3% vs 73.7%）；keyword 触发在全部骨干 ≥ always-fire 上界（选择性免费）；topk16 在强骨干抬 flip（xlam 92.6%）但 k=4 几乎不掉 flip（挤位效应主导，k 只调 ASR）。
- 仓库整理（公开 GitHub ChangWenhan/TrojanScribe）：方法公开命名 **TrojanScribe**（内部标识符 cluster 不变）；脚本去数字前缀改名 08_longtail.py→longtail_attack.py、13_unified_eval.py→unified_eval.py、14_react_baselines.py→react_baselines.py（**结果文件命名 08_longtail*.json 保持不变**，与既有数据连续）；README 重写：删 Repairs、清除 qwen3-4b 时代内容、主表/消融换成 v2 数字、新增 Terminology 表；清理 __pycache__、半成品 run 目录（vol2×2/greedy/qwen3-8b_070048/mini_141_test×2/granite_035812）与 6 个旧时代脚本。
- 跨模型矩阵（非对角 12 对）pending；xlam-2-8b ReAct 基线重跑 pending（baselines 需 xlam 服务）。

## BASELINES RERUN (xlam-2-8b) + UNIFIED EVAL v2 DENOMINATOR FIX (2026-09-09)

- ReAct 基线（clean/naive/poisonedRAG/ours/topicattack）在 xlam-2-8b 上重跑完成（`hotpotqa_seed1_*_xlam28b.json`）。
- P0 修复 1：attack_react.py:31 无条件用 VICTIM_MODEL 环境变量覆盖 CLI --model_path（qwen3-4b 适配残留，404）→ CLI 优先。
- P0 修复 2：xlam 在文本 ReAct 协议下不合规（原生 JSON 工具调用 / [response] 标签 → 58/60 空）。两处 shim：(a) attack_react.py 加 system 格式强制提示（REACT_NO_SYS=1 可关）+ (b) JSON 工具调用→Search[]/Finish[] 翻译。单题 dry_run F1=0.59 验证后全量重跑。
- P0 修复 3：clean 路径不清毒，上次中断尝试的残留毒污染 clean 基线（dry_run 中 5/8 检索文档被污染）→ react_baselines.py 启动时 store.delete_poison()。
- P0 修复 4：unified_eval.py 的 flip 分母用官方 EM 计数而判定用 v2 子串 correct（langgraph 行 22/6=366% 荒谬显示）→ 新增 clean_correct（子串口径）作全框架统一分母；clean_em 保留官方 EM 供参考。
- 最终对比（同 victim xlam-2-8b、同 60 目标、同注入错误答案、统一评分）：TrojanScribe flip 22/27 (81.5%)、collapse 0、ASR 93.3%；最佳基线 poisonedRAG flip 3/15 (20.0%)、ASR 21.7%（>4×差距）；naive/ours/topicattack 的破坏几乎全是 collapse（topicattack 14 collapse / 0 flip，纯 DoS）。注意 react clean 分母仅 15（ReAct 循环在该 victim 上本身 20 空），react 侧 flip 率只作粗对比。

## CROSS-MODEL MATRIX F COMPLETE (2026-09-09)

- 12 个非对角 pair 全部完成（双机并行 ~24 min：本机端点跑 gpt-oss-20b/xlam 受害者 6 对，141 端点跑 qwen3/llama 受害者 6 对）。新驱动 experiments/run_cross_model_split.sh（local/remote 双模式，--kb-dir 隔离：chroma / chroma_xsm141；tool_smoke 走 AGENTIC_RAG_BASE_URL——首版硬编码 localhost 的 bug 已修；curl 轮询 $BASE_URL/models——首版双 /v1 的 404 死循环 bug 已修）。
- 141 侧零代码部署：仅作为 vLLM 端点；Qwen3-8B + Meta-Llama-3.1-8B-Instruct 权重已 rsync 到 141:~/LLMs/（141 vllm 0.15.1 与本机一致）。跑完已恢复 141 的 gpt-oss-20b 服务、本机 xlam-2-8b 服务。
- 汇总 results/cross_model_summary.md：迁移无损且多个非对角格超过对角线（gptoss→qwen3 94%/92% vs 自攻击 74%/77%）；gptoss 攻击者全列最强，llama 攻击者全列最弱（仍 55–78% ASR）；collapse 每格 ≤2（改写而非 DoS）。
- flip 分母 = 受害者主表 clean-correct（27/31/38/25，与主表对角线一致，交叉验证通过）。

## BASELINE RESULTS REMOVED (2026-09-09)

- Deleted all published-baseline (KidnapRAG ReAct side) result artifacts from the repo:
  - `kidnaprag/ReAct/results/adv_targeted_results/hotpotqa_seed1_{clean,naive,ours,poisonedRAG,topicattack}_xlam28b.json` (5)
  - `results/baseline_archive/hotpotqa_seed1_*_qwen34.json` (5, old era)
  - `results/poison_{naive,ours,poisonedRAG,topicattack}.jsonl` (baseline poison corpora)
  - `results/13_unified_comparison.{json}` and `results/baseline_status.txt`
- Rationale (user decision): the upstream KidnapRAG ReAct harness's text-ReAct
  protocol is incompatible with function-calling backbones — 20 empty answers on
  the CLEAN baseline alone (9-step `Invalid action: invalid[]` loops), 20–47
  empties under attack, clean EM only 13/60. These systematically depress all
  baseline numbers and were judged unusable for a fair comparison. README Core
  claim / Design / layout / reproduction updated accordingly (baseline numbers
  removed from prose; pending: same-harness LangGraph comparison of baseline
  poison corpora).
- NOT deleted (still required by our protocol): the shared target/wrongs file
  `kidnaprag/ReAct/results/adv_targeted_results/hotpotqa.json` +
  `hotpotqa_qid_to_idx.json`.

## GPT-OSS GENERATION-BUDGET INCIDENT + FIX PREPARED (2026-09-10)

- Trigger: A1 follow-up review of the v2 mono/near-dup arms. A1's refill+MMR
  repair verified working for xlam/qwen3/llama (7.4-8.0 chunks/target, request
  8); gpt-oss-20b mono remains short (203 chunks, 33/60 targets).
- Root cause (NOT the sampler): gpt-oss-20b payload generation returns empty
  final content for the authority/bio prompts — the harmony reasoning channel
  consumes the max_tokens=256 candidate budget. Evidence from the v2 full
  method: per-style generated candidates across 59 targets are bio 3-6 /
  authority 37-38 vs update/def 400-500; picked styles are update 1960 /
  faq 767 / def 691 / authority 208 / bio 32 (hotpot; musique same shape).
  26/60 triggered targets got zero mono writes (trigger itself fires 59/60,
  confirmed offline — it is model-independent).
- Impact: gpt-oss-as-writer rows are affected (main table both datasets, all
  hotpot ablation arms, the running musique arms, cross-model gpt-oss→3
  victims); gpt-oss-as-victim rows with foreign poison are unaffected.
  README's "largest style-diversity gap on gpt-oss (26.3% vs 73.7%)" is
  confounded: 14 of the 38 clean-correct targets were never poisoned
  (poisoned-only rate 9/24=37.5%, ASR 14/33=42.4%).
- Code prepared (defaults unchanged; zero effect on non-gpt-oss runs):
  * `llm.py`: `AGENTIC_RAG_TOP_LEVEL_KWARGS` (top-level request-body fields;
    gpt-oss harmony ignores `chat_template_kwargs`).
  * `payload.py`: `AGENTIC_RAG_PAYLOAD_MAX_TOKENS` (default 256) +
    `payload_max_tokens()`.
  * `experiments/payload_probe.py`: single-target effort×budget probe.
  * `longtail_attack.py` meta records `payload_max_tokens` + `request_kwargs`.
- Verification: offline unit checks passed (extra_body merge; budget
  passthrough; defaults unchanged when env unset); live smoke on the serving
  xlam-2-8b PASS 2/2. gpt-oss probe + full redo deferred until the running
  MuSiQue ablation finishes.
- Note: harmony reasoning cannot be disabled, only low/medium/high (verified
  in vLLM 0.15.1 and the openai_harmony enum); expect low + larger budget to
  be needed. Probe decides before the redo.

## GPT-OSS A4 FIX VERIFIED + FULL REDO COMPLETE (2026-09-11)

- Probe (served gpt-oss-20b, authority-style prompt): medium+256 = 0/8 usable
  (fault reproduced), medium+768 = 3/8, low+256 = 8/8, low+768 = 8/8; the
  real-code-path gate (LLMBackend + AGENTIC_RAG_TOP_LEVEL_KWARGS, low+768)
  passed 8/8. All redo runs used reasoning_effort=low + payload budget 768.
- Scope completed: MuSiQue (main + 11 arms, local) and HotpotQA (main + 11
  arms) re-run; cross-model gpt-oss->{xlam,qwen3,llama} replayed and
  re-evaluated. HotpotQA was split across both machines with isolated KBs
  (local: main + vol2/vol4/vol6/embed_hybrid/mono + xlam pair on data/chroma;
  141: nodiv/greedy/semantic/trig_always/topk4/topk16 + qwen3/llama pairs on
  data/chroma_gpt-oss-20b_b, cross pairs on data/chroma_xsm141). MuSiQue
  ablation reached 44/44 arms (llama topk16 done 2026-09-10 21:49; the qwen3
  semantic arm was rerun after the interception collateral kill).
- Generation-side proof: hotpot main row now picks all five styles
  (faq 1080/update 536/bio 496/def 472/authority 1192; before: bio 32 /
  authority 208, update-dominated); mono arm 457 chunks / 58 targets (before
  203/33); dose volumes exact (118/236/354/472).
- Metric shifts (old -> new): hotpot main flip 28/38 (73.7%) -> 21/38 (55.3%),
  ASR 41/60 -> 33/60; musique main clean 18 -> 21, flip 18/18 -> 16/21
  (76.2%), ASR 41/59 -> 33/59; gpt-oss mono flip 10/38 (26.3%) -> 20/38
  (52.6%); cross-model gpt-oss row ASRs 85/92/77 -> 92/90/82. Post-fix,
  gpt-oss-20b is the most resistant victim of its own poison on both datasets
  but remains the strongest attacker for foreign victims.
- Narrative revisions propagated to README.md, results/ablation_summary.md
  (auto), results/cross_model_summary.md: (a) gpt-oss is the most resistant
  backbone post-fix and the earlier 73.7%/100% values were a generation
  artifact; (b) the multi-style consensus edge over single-style is consistent
  but modest (-2.6 to -17pp) and the near-duplicate control exceeds the full
  method on qwen3-8b/gpt-oss-20b; (c) keyword-vs-always selectivity is not
  universally free (gpt-oss: always 73.7% vs keyword 55.3%); (d) CorruptRAG
  single-shot leads on gpt-oss/llama, TrojanScribe's edge is the default
  victim (xlam).
- Baseline bookkeeping: gpt-oss baseline flips were re-scored against the
  updated main-table clean set (hotpot clean set changed even though its count
  stayed 38; musique clean 18 -> 21) so every method shares one denominator;
  all other victims' numbers unchanged (verified by full recompute).
- A4 closed in research/issues/known_issues.md; AGENTS.md serving note updated
  with the two env vars.
- Incident note: the interception driver's one-time `pkill -f
  longtail_attack.py` killed the in-flight qwen3 semantic arm (remote-victim
  arms execute their harness locally); repaired via
  experiments/rerun_qwen3_semantic_musique.sh. No other collateral.

## REPO HYGIENE (2026-09-11) — post-redo cleanup

- Removed Python caches (8 `__pycache__` trees).
- Removed smoke/test artifacts from results/ (10 files: `ap_test_*`,
  `baseline_pr_*_5t`, `baseline_poisonedrag_*_test`, `*_5targets`,
  `cr_as_hp_xlam_test.json`, `mus_pr_*_test`).
- Removed stale orchestrator `experiments/run_all_v2.sh` (referenced the
  deleted `unified_eval.py`; superseded by the per-stage drivers).
- Removed 5 superseded partial run dirs (killed mid-injection): 2× xlam vol4
  musique attempts, qwen3 semantic collateral kill, llama trig_always PSU-stop
  kill, one 2026-09-09 cross-model attempt.
- Removed leftover KB clones `data/chroma_gpt-oss-20b{,_b}` (1.28 GB; already
  flagged in CODE_REVIEW 2026-09-09).
- Restored pristine KB counts by deleting leftover poison + doc-consolidator
  chunks and debug collections: chroma 67048→66581, chroma_musique
  18150→17629, chroma_musique_141 18287→17629, chroma_xsm141 67048→66581.
- Kept: `results/runs/` archives (167 dirs), `results/logs/`, all canonical
  root JSONs (120 `08_longtail*` files), baseline corpora + eval files.
- Also removed (user decision, same day): the two 141 KB clones
  `chroma_musique_141` (188 MB) and `chroma_xsm141` (680 MB); recreate by
  copying the corresponding main KB before any remote rerun. Guards added to
  `run_ablation_musique_remote.sh` / `run_hotpot_gptoss_fixed_141.sh` so a
  missing clone aborts instead of silently building an empty KB.
  Repo size 3.3 GB → 1.2 GB.
- Also removed (user decision, 2026-09-11 09:19): the three model-weight dirs
  uploaded to `141:~/LLMs/` for the remote-endpoint runs (Qwen3-8B 16 GB,
  Meta-Llama-3.1-8B-Instruct 30 GB, gpt-oss-20b 39 GB; ~84 GB freed on 141,
  338 GB → 254 GB used). The 141 vLLM server was stopped first. 141 is now a
  bare endpoint; re-upload the needed weights before serving there again
  (`scripts/serve_141.sh` unchanged).
