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
| `cluster_greedy` (arm) | no-template-anchor arm | no per-style anchor; global MMR sampling (λ=0.5) without anchors |
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
| qwen3-8b | 31/60 | 21/31 (67.7%) | 18/31 (58.1%) | 22/31 (71.0%) | **24/31 (77.4%)** | 23/31 (74.2%) | **46/60** |
| gpt-oss-20b | 38/60 | 19/38 (50.0%) | 11/38 (28.9%) | 28/38 (73.7%) | **30/38 (78.9%)** | 21/38 (55.3%) | 33/60 |
| llama-3.1-8b | 25/60 | 19/25 (76.0%) | 11/25 (44.0%) | **21/25 (84.0%)** | **21/25 (84.0%)** | 19/25 (76.0%) | 32/60 |

TrojanScribe is the strongest method on the default victim (xlam-2-8b:
81.5% vs 66.7% for the best baseline) and stays close on qwen3-8b (74.2% vs
77.4%); on gpt-oss-20b and llama-3.1-8b the single-shot **CorruptRAG**
baselines lead (78.9% / 84.0%). Among baselines, **CorruptRAG is the
strongest**: with only ONE poisoned text per target it matches or beats
PoisonedRAG (5 chunks) on most rows — its "outdated corpus / latest data"
framing is a very effective single shot. The adapted AgentPoison is the
weakest because its retrieval-side trigger mechanism requires modifying the
victim's query, which our KB-only attacker cannot do. Same-harness,
same-metric comparison (not the old ReAct-harness numbers, removed
2026-09-09); code in `baseline/poisonedrag/`, `baseline/agentpoison/` and
`baseline/corruptrag/`.

Note (2026-09-11): the gpt-oss-20b row was re-run with the payload
generation-budget fix (issue A4); its baselines were re-scored against that
victim's updated clean set so every method shares one denominator. The default
victim's ranking is unchanged and TrojanScribe remains by far the most
directional there (ASR 93.3% vs 73.3% for the best baseline, CorruptRAG-AS
44/60).

### MuSiQue (59 frozen targets)

| victim | clean | PR flip | AP flip | CR-AS flip | CR-AK flip | TrojanScribe flip | TS ASR |
|---|---|---|---|---|---|---|---|
| xlam-2-8b | 7/59 | 7/7 (100%) | 6/7 (85.7%) | 5/7 (71.4%) | 6/7 (85.7%) | 7/7 (100%) | **51/59** |
| qwen3-8b | 9/59 | 9/9 (100%) | 7/9 (77.8%) | 8/9 (88.9%) | 7/9 (77.8%) | 9/9 (100%) | 47/59 |
| gpt-oss-20b | 21/59 | 19/21 (90.5%) | 11/21 (52.4%) | 17/21 (81.0%) | 19/21 (90.5%) | 16/21 (76.2%) | 33/59 |
| llama-3.1-8b | 10/59 | 10/10 (100%) | 8/10 (80.0%) | 9/10 (90.0%) | 9/10 (90.0%) | 8/10 (80.0%) | 26/59 |

On MuSiQue TrojanScribe reaches ceiling flip on xlam-2-8b and qwen3-8b
(7/7 and 9/9; tiny clean-correct pools), together with PoisonedRAG. On
gpt-oss-20b the A4 redo grew the clean pool from 18 to 21 targets, so the
baseline flips resolve much better: PoisonedRAG and CorruptRAG-AK lead
(19/21, 90.5%) while TrojanScribe sits at 16/21 (76.2%); PoisonedRAG leads on llama-3.1-8b (10/10, vs 9/10 for CorruptRAG). AgentPoison
is the weakest baseline on 7 of 8 rows — the exception is MuSiQue xlam-2-8b,
where CorruptRAG-AS is lowest (5/7). Across both datasets the pattern is
consistent: TrojanScribe's edge is at its largest on the default victim, and
CorruptRAG's single-shot "outdated corpus" framing is the strongest published
baseline elsewhere.

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
| xlam-2-8b | — | 28/31 (90%), ASR 85% | 30/38 (79%), ASR 67% | 23/25 (92%), ASR 78% |
| qwen3-8b | 22/27 (81%), ASR 88% | — | 27/38 (71%), ASR 65% | 22/25 (88%), ASR 73% |
| gpt-oss-20b | 24/27 (89%), ASR 92% | 29/31 (94%), ASR 90% | — | 23/25 (92%), ASR 82% |
| llama-3.1-8b | 21/27 (78%), ASR 65% | 25/31 (81%), ASR 72% | 25/38 (66%), ASR 55% | — |

Key observations:

- **Transfer is essentially lossless and often exceeds the same-model
  diagonal**: every off-diagonal cell keeps ≥65% flip and ≥55% ASR, and
  several non-diagonal cells beat the victim's own diagonal attack (e.g.
  gpt-oss-20b poison on qwen3-8b: 94% flip / 90% ASR, while the same-model
  diagonal gpt-oss-20b cell is 55% / 55%). A poison corpus written once by any
  common open backbone poisons the whole fleet — there is no per-victim
  customization barrier to cross.
- Attacker quality ordering persists across victims: gpt-oss-20b poison is the
  strongest (or tied-strongest) on every foreign victim; llama-3.1-8b poison
  the weakest (still 55–72% ASR on foreign victims; 53% is llama's own
  main-table ASR).
- Collapse stays near zero (≤2 per cell): the damage is genuine knowledge
  rewriting, not agent breakage, regardless of which model wrote the poison.

## Main table — 4 backbones × 2 datasets (v2 rerun 2026-09-08; gpt-oss-20b rows re-run 2026-09-11; `results/ablation_summary.md`)

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
| gpt-oss-20b (MoE) | 38/60 | 21/38 (55.3%) | 1 | 33/60 (55.0%) | 9/60 |
| Llama-3.1-8B-Instruct | 25/60 | 19/25 (76.0%) | 1 | 32/60 (53.3%) | 8/60 |

### MuSiQue (59 frozen targets)

| backbone | clean correct | flip (knowledge) | collapse | ASR | true paragraph still in top-8 |
|---|---|---|---|---|---|
| xlam-2-8b (default victim) | 7/59 | **7/7 (100%)** | 0 | 51/59 (86.4%) | 2/59 |
| Qwen3-8B | 9/59 | 9/9 (100%) | 0 | 47/59 (79.7%) | 0/59 |
| gpt-oss-20b (MoE) | 21/59 | 16/21 (76.2%) | 1 | 33/59 (55.9%) | 1/59 |
| Llama-3.1-8B-Instruct | 10/59 | 8/10 (80.0%) | 0 | 26/59 (44.1%) | 0/59 |

**Insights.** (1) The attack generalizes across model families, scales and
datasets — knowledge-flip stays at 55–100% and ASR at 44–93% everywhere. After
the gpt-oss-20b redo (2026-09-11, generation-budget fix A4) the 20B MoE
reasoning model is the most resistant victim on both datasets (55.3% HotpotQA /
76.2% MuSiQue flip); the dense 8B backbones flip 74–100%. (2) ASR is the
backbone-sensitive metric (44–93%), tracking each model's tendency to echo the
injected string verbatim. (3) On MuSiQue the true paragraph is displaced out of
the top-8 retrieval window on essentially all targets — the attack captures
retrieval outright. (4) Two backbones were excluded for protocol reasons (weak
tool-format compliance); evidence in `research/monitor/experiment_ledger.md`.
The gpt-oss-20b rows above are the post-fix measurement; the earlier higher
numbers were an artifact of empty authority/bio candidates (A4), not attack
strength.

## Ablations (v2, all 4 backbones; table in `results/ablation_summary.md`)

Every arm is run on EVERY backbone, HotpotQA (bullets below) and MuSiQue (same
11 arms; compact reading at the end). All flip rates use the victim's
main-table clean set as the denominator (arms reuse it via `--clean-from`;
early xlam-2-8b arms measured their own clean and are re-scored against the
main-table clean at analysis time). All gpt-oss-20b rows were re-run
2026-09-11 with the payload generation-budget fix (A4), so its single-style and
near-duplicate arms are now volume-matched with the rest.

- **Poison dose** (2/4/6/8 chunks per target): flip rises with dose to a
  plateau around 4–8 on xlam-2-8b (74→74→78→81.5%), qwen3-8b (58→77→74→74%)
  and llama-3.1-8b (68→68→72→76%); gpt-oss-20b is non-monotonic
  (47→66→74→55%) within its smallest clean-correct pool. The true paragraph's
  presence in the victim's top-8 falls monotonically with dose on every
  backbone (xlam 48→44→29→13, qwen3 48→46→30→13, gpt-oss 46→42→26→9) —
  displacement keeps growing past the flip saturation point.
- **Style diversity** (all dose 8): the single-style arm (`cluster_mono`) is
  the weakest on every backbone, but the gap is modest and backbone-dependent:
  −2.6pp on gpt-oss-20b (52.6% vs 55.3%), −6.5pp on qwen3-8b, −16.0pp on
  llama-3.1-8b and −14.8pp on xlam-2-8b. Removing the selection machinery has
  backbone-dependent effects rather than a uniform one: `cluster_nodiv` /
  `cluster_greedy` stay within ±6pp of the full method on xlam-2-8b and
  qwen3-8b, but on gpt-oss-20b both score clearly higher (71.1% / 63.2% vs
  55.3%) and on llama-3.1-8b greedy is clearly lower (64.0% vs 76.0%). The
  near-duplicate control (`embed_hybrid`) is within 4pp of the full method on
  xlam-2-8b and llama-3.1-8b and exceeds it on qwen3-8b (80.6% vs 74.2%) and
  gpt-oss-20b (81.6% vs 55.3%), where it also floods the retrieval window
  (true_in_top8 ≈ 0). Consensus-style generation therefore gives a consistent
  but not universal edge over a single style, and at equal volume near-
  duplicates can be as strong or stronger.
- **Trigger**: keyword (auto-extracted proper nouns) is within noise of the
  always-fire upper bound (p=1.0) on three backbones (xlam-2-8b equal at 81.5%,
  qwen3-8b equal at 74.2%, llama-3.1-8b 76.0% vs 60.0% in keyword's favour),
  but on gpt-oss-20b always-fire scores 73.7% vs keyword 55.3%. Trigger
  implementation is therefore not decisive, but **selectivity is not
  universally free**.
- **Victim retrieval window** (top-k 4/8/16): flip is robust to the window on
  xlam-2-8b (81.5/81.5/92.6%) and qwen3-8b (77.4/74.2/80.6%); shrinking to k=4
  does not protect on any backbone (gpt-oss-20b 78.9% at k=4 vs 55.3% at k=8),
  and only llama-3.1-8b clearly loses at k=16 (68.0%). Window size mainly
  modulates ASR (xlam 85→93→92%, llama 32→53→62%): a larger window surfaces
  more poison chunks and more verbatim echo.
- **MuSiQue counterpart** (same 11 arms; clean pools 7/9/21/10): dose saturates
  by 4–6 (xlam 57→86→86→100%, qwen3 89/89/89/100%, gpt-oss 52→57→81→76%);
  the single-style arm is again at the bottom (clear-lowest 6/10 on
  llama-3.1-8b; tied-lowest 6/7 with the near-duplicate control on xlam-2-8b),
  and the near-duplicate control again exceeds the full method on gpt-oss-20b
  (19/21 vs 16/21); gpt-oss-20b's k=16 arm drops to 12/21 (57.1%) while k=4
  stays 19/21 (90.5%).

## Repository layout

```
configs/            vLLM / knowledge-base / attack settings
src/agentic_rag/    our method: LangGraph victim agent + poisoned subagent chain
  eval/unified.py     single source of truth for EM/F1/ASR/flip scoring
experiments/
  longtail_attack.py    our method: shared targets -> clean baseline -> isolated
                        per-variant attack (persists per-target records + poison writes)
  summarize_ablation.py ablation + main-table summary -> results/ablation_summary.md
  payload_probe.py      single-target candidate-generation probe (effort × budget grid)
  run_*.sh              batch drivers (main table / ablation / cross-model / baselines)
baseline/           published attack baselines (same-harness, same-metric)
  poisonedrag/          PoisonedRAG (USENIX Security 2025): corpus crafting +
                        verification + per-victim evaluation scripts
  agentpoison/          AgentPoison (NeurIPS 2024), adapted to the KB-only
                        write-back threat model: trigger selection + eval scripts
  corruptrag/           CorruptRAG (ACM SACMAT 2026) single-shot AS/AK templates
                        + dual-GPU evaluation driver
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
rate). The semantic trigger also fires 59/60 but misses a different target (its
embedding cosine 0.78 falls below the 0.82 threshold), so each trigger has its
own miss — the 59/60 rate is not a property of the target set.
