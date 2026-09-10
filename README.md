# TrojanScribe — AgenticRAG Write-Back Poisoning (Shared Knowledge Base)

**TrojanScribe** is the public name of our attack (method identifier `cluster` —
identifiers are stable, see Terminology below): a poisoned supply-chain subagent,
masquerading as the host's benign document-consolidation scribe, writes
consensus-style fabricated passages into a **shared knowledge base** through its
legitimate write-back channel.

Research codebase for **write-back poisoning of multi-user AgenticRAG** through a
supply-chain subagent, evaluated against published document-poisoning baselines
under controlled variables.

## Terminology (single source of truth)

The code and result files use a few internal names. This table maps every one of
them to its plain meaning. Arm names and JSON keys below are **identifiers — they
must not be renamed** (existing result files already use them); everything else
in this README and in code comments uses the plain term.

| identifier (code / results) | plain term | meaning |
|---|---|---|
| `cluster` (method name) | **consensus-style multi-template poisoning** — public name: **TrojanScribe** | for each target, fabricate ~8 short passages that all state the same injected wrong answer, each in a **different text style** so retrieved hits look like independent sources agreeing |
| `archetype` (code / JSON key `picked_archetypes`) | **poison text style (template)** | one of five fabricated styles: `faq` (FAQ entry), `update` (update notice), `bio` (biography), `def` (definition list), `authority` (authority quote) |
| "multi-voice" (old docs) | **multi-style consensus** | the property above: texts agree with each other but read as different source types |
| `embed_hybrid` (arm) | near-duplicate control | all poison chunks written in one near-identical style |
| `cluster_mono` (arm) | single-style arm | only the authority-quote template |
| `cluster_nodiv` (arm) | no-diversity-selection arm | sampler diversity term disabled (`lambda=1.0`) |
| `cluster_greedy` (arm) | no-template-anchor arm | no per-style anchor; plain relevance-greedy sampling |
| `vol2` / `vol4` / `vol6` (arms) | poison dose | 2 / 4 / 6 fabricated chunks per target (default = 8) |
| `topk4` / `topk16` (arms) | victim retrieval window | victim's `kb_search` top-k (default k=8) |
| `keyword` / `semantic` / `trig_always` (triggers) | keyword / semantic / always-fire trigger | fire when the task text contains an auto-extracted proper noun / embedding cosine to any target query ≥ 0.82 / always (p=1.0) |
| `flip` | knowledge flip | clean-correct target answered with a NON-EMPTY wrong answer after the attack |
| `collapse` | collapse (DoS) | clean-correct target answered with an empty answer or crash — never counted as a flip |
| `ASR` | attack success rate | answer contains the injected wrong-answer string |
| `true_in_top8` (JSON key) | true paragraph displaced? (inverted) | count of targets whose **true** supporting paragraph still ranks in the victim's top-8 |
| `08_longtail*.json` (result files) | — | historical data-file convention kept for stability with archived results; the experiment script itself is `longtail_attack.py` |
| AR1–AR5 | frozen hypotheses | pre-registered proposals in `research/frozen/` (read-only, sha256 manifest); e.g. AR3 = the consensus-style poisoning hypothesis implemented by `payload.generate_cluster` |

## Core claim

A poisoned subagent installed from an "open-source" platform writes consensus-style
fabricated passages into a **shared knowledge base** while behaving benignly in
every prior round. On a **LangGraph agent** victim it is both directional and
high-flip: on the default victim (xlam-2-8b, 60 long-tail HotpotQA targets) it
rewrites **81.5% of clean-correct targets** (22/27 — zero agent crashes) and
injects the shared target wrong answer into **93.3% of all answers
(ASR 56/60)**. Cross-model transfer (Section below) shows poison written by any
common open backbone transfers near-losslessly to every other victim.

## Baseline comparison — PoisonedRAG, AgentPoison & CorruptRAG

To compare against published document-poisoning attacks under **identical
conditions** (same 60 targets, same injected wrong answers, same bge KB,
same LangGraph victims, same unified metrics), we run each baseline's crafting
method on the same victim set. Poison texts are generated once (Qwen3-8B) and
evaluated against all four main-table victims.

- **PoisonedRAG** (USENIX Security 2025) black-box corpus crafting
  (`P = question + "." + II`, II = LLM-crafted 30-word corpus, verified so the
  wrong answer follows from the corpus alone). 5 chunks per target.
- **AgentPoison** (NeurIPS 2024, adapted) trigger + instruction: a trigger word
  is selected per target by Qwen3-8B sampling (the paper's GPT-3.5 target-asr
  role) and the poisoned doc embedding is pushed toward the question embedding
  under our bge retriever. Unlike the paper we do not append the trigger to the
  victim's question (write-back threat model), so the trigger only serves the
  generation side — the fair adaptation to our KB-only attacker.
- **CorruptRAG** (ACM SACMAT 2026) **single-shot** poisoning: ONE text per
  target. AS = fixed template (`question + "outdated corpus stating the
  incorrect answer [C]" + "latest data confirms the correct answer is [A]"`);
  AK = LLM-refined (Qwen3-8B, V=30 words, L=5 validation trials, fallback to AS).

| victim | clean | PR flip | AP flip | CR-AS flip | CR-AK flip | TrojanScribe flip | TS ASR |
|---|---|---|---|---|---|---|---|
| xlam-2-8b | 27/60 | 14/27 (51.9%) | 12/27 (44.4%) | 18/27 (66.7%) | 16/27 (59.3%) | **22/27 (81.5%)** | **56/60** |
| qwen3-8b | 31/60 | 21/31 (67.7%) | 18/31 (58.1%) | 22/31 (71.0%) | 24/31 (77.4%) | **23/31 (74.2%)** | **46/60** |
| gpt-oss-20b | 38/60 | 18/38 (47.4%) | 12/38 (31.6%) | 27/38 (71.1%) | 29/38 (76.3%) | **28/38 (73.7%)** | **41/60** |
| llama-3.1-8b | 25/60 | 19/25 (76.0%) | 11/25 (44.0%) | 21/25 (84.0%) | 21/25 (84.0%) | 19/25 (76.0%) | 32/60 |

TrojanScribe leads knowledge-flip on three of four victims (up to +30pp on
gpt-oss-20b). Among baselines, **CorruptRAG is the strongest**: with only ONE
poisoned text per target it matches or beats PoisonedRAG (5 chunks) on most
rows — its "outdated corpus / latest data" framing is a very effective single
shot, and it even edges out TrojanScribe on llama-3.1-8b flip (84% vs 76%,
though TrojanScribe's ASR there is higher on ASR it trails on llama). The
adapted AgentPoison is the weakest because its retrieval-side trigger mechanism
requires modifying the victim's query, which our KB-only attacker cannot do.
Same-harness, same-metric comparison (not the old ReAct-harness numbers,
removed 2026-09-09); code in `baseline/poisonedrag/`, `baseline/agentpoison/`
and `baseline/corruptrag/`.

### MuSiQue (59 frozen targets)

| victim | clean | PR flip | AP flip | CR-AS flip | CR-AK flip | TrojanScribe flip | TS ASR |
|---|---|---|---|---|---|---|---|
| xlam-2-8b | 7/59 | 7/7 (100%) | 6/7 (85.7%) | 5/7 (71.4%) | 6/7 (85.7%) | 7/7 (100%) | **51/59** |
| qwen3-8b | 9/59 | 9/9 (100%) | 7/9 (77.8%) | 8/9 (88.9%) | 7/9 (77.8%) | 9/9 (100%) | 47/59 |
| gpt-oss-20b | 18/59 | 16/18 (88.9%) | 9/18 (50.0%) | 15/18 (83.3%) | 17/18 (94.4%) | **18/18 (100%)** | **41/59** |
| llama-3.1-8b | 10/59 | 10/10 (100%) | 8/10 (80.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 8/10 (80.0%) | 26/59 |

On MuSiQue PoisonedRAG and TrojanScribe reach near-ceiling flip (small
clean-correct pools); CorruptRAG is close behind (a single shot flip 83-94% on
gpt-oss-20b/llama), AgentPoison consistently weakest. TrojanScribe's ASR leads
on 2/4 victims; CorruptRAG's ASR is highest on qwen3-8b.

Metric definitions: **flip** counts only clean-correct targets answered with a
NON-EMPTY wrong answer (true before the attack, false after); empty/crashed rows
are reported separately as `collapse`; `changed` = flip + collapse (the correct
answer was replaced by anything else, including nothing).

## Design

Our method runs on our LangGraph agent harness. Metric definitions below are
applied uniformly to every run of our method.

**Controlled variables across all runs:** same 60 long-tail HotpotQA targets
(rare, hard questions), same per-target wrong answers (the shared `hotpotqa.json`
"incorrect answer", injected verbatim by every method), same bge knowledge base
(66,581 clean chunks), top-8 retrieval, victim = the backbone under test
(vLLM, OpenAI-compatible).

**Unified metrics** (`src/agentic_rag/eval/unified.py`, applied to every
run's raw answers):

- `EM` — HotpotQA official normalize (article + punctuation removal)
- `F1>0` — token F1 vs gold
- `ASR` — substring of the **injected** wrong answer in the answer (literature style)
- `flip` — clean-correct target answered wrong after the attack, decomposed
  into `flips_knowledge` (non-empty wrong answer) and `flips_collapse` (empty
  answer / crashed row), so DoS breakage is never counted as a knowledge flip
  (clean denominators are per-victim clean baselines)

## Cross-model transfer — poison written by A, victimized by B (`results/cross_model_summary.md`)

Attacker A's persisted poison texts are **replayed verbatim** into the KB
(`--phase inject-from`, no regeneration — transferability of pre-written
poison), then victim B is evaluated; 12 off-diagonal pairs, 4×4 backbones
(diagonal = main table):

| attacker \ victim | xlam-2-8b | qwen3-8b | gpt-oss-20b | llama-3.1-8b |
|---|---|---|---|---|
| xlam-2-8b | — | 28/31 (90%), ASR 85% | 28/38 (74%), ASR 67% | 23/25 (92%), ASR 78% |
| qwen3-8b | 22/27 (81%), ASR 88% | — | 26/38 (68%), ASR 65% | 22/25 (88%), ASR 73% |
| gpt-oss-20b | 24/27 (89%), ASR 85% | 29/31 (94%), ASR 92% | — | 23/25 (92%), ASR 77% |
| llama-3.1-8b | 21/27 (78%), ASR 65% | 25/31 (81%), ASR 72% | 24/38 (63%), ASR 55% | — |

Key observations:

- **Transfer is essentially lossless and often exceeds the same-model
  diagonal**: every off-diagonal cell keeps ≥63% flip and ≥55% ASR, and
  several non-diagonal cells beat the victim's own diagonal attack (e.g.
  gpt-oss-20b poison on qwen3-8b: 94% flip / 92% ASR vs 74% / 77% on
  gpt-oss-20b itself). A poison corpus written once by any common open
  backbone poisons the whole fleet — there is no per-victim customization
  barrier to cross.
- Attacker quality ordering persists across victims: gpt-oss-20b poison is
  the strongest on every foreign victim; llama-3.1-8b poison the weakest
  (still 55–78% ASR).
- Collapse stays near zero (≤2 per cell): the damage is genuine knowledge
  rewriting, not agent breakage, regardless of which model wrote the poison.

## Main table — 4 backbones × 2 datasets (v2 rerun 2026-09-08, `results/ablation_summary.md`)

Same protocol per dataset: HotpotQA = shared 60 targets + shared wrongs; MuSiQue
= frozen 59-target set with frozen wrong answers, zero per-model re-selection.
Method = `cluster` at poison dose 8, keyword trigger, 3 benign rounds, flip
judged by the v2 substring-correct rule, victim = the served backbone (attacker
payload/writer use the same backbone).

### HotpotQA (60 shared targets)

| backbone | clean correct | flip (knowledge) | collapse | ASR | true paragraph still in top-8 |
|---|---|---|---|---|---|
| xlam-2-8b (default victim) | 27/60 | **22/27 (81.5%)** | 0 | **56/60 (93.3%)** | 13/60 |
| Qwen3-8B | 31/60 | 23/31 (74.2%) | 0 | 46/60 (76.7%) | 13/60 |
| gpt-oss-20b (MoE) | 38/60 | 28/38 (73.7%) | 0 | 41/60 (68.3%) | 16/60 |
| Llama-3.1-8B-Instruct | 25/60 | 19/25 (76.0%) | 1 | 32/60 (53.3%) | 8/60 |

### MuSiQue (59 frozen targets)

| backbone | clean correct | flip (knowledge) | collapse | ASR | true paragraph still in top-8 |
|---|---|---|---|---|---|
| xlam-2-8b (default victim) | 7/59 | **7/7 (100%)** | 0 | 51/59 (86.4%) | 2/59 |
| Qwen3-8B | 9/59 | 9/9 (100%) | 0 | 47/59 (79.7%) | 0/59 |
| gpt-oss-20b (MoE) | 18/59 | 18/18 (100%) | 0 | 41/59 (69.5%) | 3/59 |
| Llama-3.1-8B-Instruct | 10/59 | 8/10 (80.0%) | 0 | 26/59 (44.1%) | 0/59 |

**Insights.** (1) The attack fully generalizes across model families, scales
(dense 4B→8B, MoE 20B) and datasets — knowledge-flip stays at 74–100% everywhere,
and stronger backbones are NOT more resistant. (2) ASR is the backbone-sensitive
metric (44–93%), tracking each model's tendency to echo the injected string
verbatim. (3) On MuSiQue the true paragraph is displaced out of the top-8
retrieval window on essentially all targets — the attack captures retrieval
outright. (4) Two backbones were excluded for protocol reasons (weak tool-format
compliance); evidence in `research/monitor/experiment_ledger.md`.

## Ablations (v2, all 4 backbones, table in `results/ablation_summary.md`)

Every arm is run on EVERY backbone, HotpotQA, method = `cluster` unless stated.

- **Poison dose** (2/4/6/8 chunks per target): flip saturates around dose 4–8 on
  strong backbones (xlam 74→75→78→81.5%) while weaker-dose arms stay far below
  on gpt-oss-20b (53→55→66→74%); meanwhile the true paragraph vanishes from the
  retrieval window monotonically with dose (qwen3-8b true_in_top8 48→46→30→13) —
  displacement strength grows with dose even past the success saturation point.
- **Style diversity** (all dose 8): the load-bearing component is generating
  text in multiple distinct styles, not the selection machinery — the
  single-style arm (`cluster_mono`) is consistently the weakest (flip −6 to
  −47pp vs full; the largest gap on gpt-oss-20b, 26.3% vs 73.7%), while
  removing the selection machinery (`cluster_nodiv`, `cluster_greedy`) stays
  within noise of the full method. The near-duplicate control (`embed_hybrid`)
  matches on flip but collapses the true paragraph to ~1–6 in the top-8:
  near-duplicates flood the retrieval window instead of directional rewriting.
- **Trigger**: keyword (auto-extracted proper nouns) matches or beats the
  always-fire upper bound (p=1.0) on every backbone (e.g. llama-3.1-8b 76.0% vs
  60.0% flip) — **selectivity is effectively free**: the subagent can stay
  silent on all non-target traffic at no cost to the attack.
- **Victim retrieval window** (top-k 4/8/16): larger windows raise the flip
  rate on the stronger backbones (xlam-2-8b 81.5→92.6%, qwen3-8b 74.2→80.6% at
  k=16), while shrinking the window barely protects (flip at k=4 still
  68–82%): displacement of the true paragraph dominates, and window size mainly
  modulates how many poison chunks surface (ASR falls with k, e.g.
  llama-3.1-8b 53→32% at k=4).

## Repository layout

```
configs/            vLLM / knowledge-base / attack settings
src/agentic_rag/    our method: LangGraph victim agent + poisoned subagent chain
  eval/unified.py     single source of truth for EM/F1/ASR/flip scoring
experiments/
  longtail_attack.py    our method: shared targets -> clean baseline -> isolated
                        per-variant attack (persists per-target records + poison writes)
  summarize_ablation.py ablation + main-table summary -> results/ablation_summary.md
  run_*.sh              batch drivers (main table / ablation / cross-model)
baseline/           published attack baselines (same-harness, same-metric)
  poisonedrag/          PoisonedRAG (USENIX Security 2025): corpus crafting +
                        verification + per-victim evaluation scripts
results/            result JSONs (latest-run mirror) + runs/<run_id>/ timestamped
                    archives (kept out of this repo; naming note in Terminology)
research/           frozen hypotheses, experiment ledger, literature review
```

## Reproduction

1. Serve the default victim (xLAM-2-8B-fc-r):
   `vllm serve /path/to/xlam-2-8b-fc-r --port 8000 --gpu-memory-utilization 0.85 --max-model-len 16384 --served-model-name xlam-2-8b --enable-auto-tool-choice --tool-call-parser xlam --chat-template /path/to/xlam_chat_template.jinja`
2. Build the knowledge base: `data/chroma` (bge-base-en-v1.5 embeddings of
   HotpotQA dev distractor paragraphs, 66,581 chunks) — see `configs/default.yaml`
3. Our method (LangGraph):
   `python experiments/longtail_attack.py --targets 60 --volume 8 --variants cluster --use-shared-targets`
   (targets are structurally aligned to the shared 60-qid protocol and the run
   asserts full coverage; wrong answers are read from the shared target file)
4. Ablation summary:
   `python experiments/summarize_ablation.py`

## Environment

- conda env `agents` (Python 3.11; langgraph, chromadb, vllm, openai)
- victim / attacker model per main-table row (vLLM, localhost:8000; models
  downloaded via ModelScope). Serving notes (see
  `experiments/run_main_table_v2.sh`): Qwen3-8B needs `--reasoning-parser qwen3`
  plus request-level `enable_thinking: false`; gpt-oss-20b needs the harmony
  parser with `--max-num-seqs 64` (sampler-warmup OOM on 24 GB) and 32k context
  (harmony computes the generation budget as `max_model_len - prompt_len`, which
  goes negative once a ReAct tool result overflows); Llama-3.1-8B needs a patched
  chat template for multi-tool-call history.
- embedding: bge-base-en-v1.5 (CPU)
- datasets: HotpotQA dev (data/hotpot_dev_distractor_v1.json); MuSiQue
  (data/musique_ans_v1.0_dev.jsonl, knowledge base 17,629 chunks). Targets: the
  60 long-tail HotpotQA qids in `data/targets/hotpotqa.json` and the frozen
  59-target MuSiQue set in `research/frozen/musique_targets_59.json`
- concurrency: `AGENTIC_RAG_ASK_WORKERS` / `AGENTIC_RAG_SAMPLE_WORKERS` (default
  1 = serial; the drivers set 8) parallelize independent victim asks and poison
  candidate sampling; the writer install phase stays serial by design.
  Concurrency is only a speed-up: candidates are independent draws at the same
  prompt/temperature, so the distribution matches the serial branch (boundary
  cases may issue a few fewer calls than the strict 4n cap — see the note at
  `payload.py` `_gen_i`).

Known limitation: 1/60 targets ("...a rule that is expressed algebraically...")
contains no proper-noun keyword and is missed by the keyword trigger (59/60 fire
rate); the semantic trigger covers such cases.
