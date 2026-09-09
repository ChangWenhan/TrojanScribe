# AgentPoison (adapted) baseline

[AgentPoison: Red-teaming LLM Agents via Memory or Knowledge Base Backdoor
Poisoning](https://arxiv.org/abs/2407.12784) (NeurIPS 2024, official code:
`AI-secure/AgentPoison`). We adapt its *trigger + poisoned instruction* idea to
our write-back threat model (attacker can only write documents into the KB,
never modify the victim's question).

## What we keep from the paper

- **Trigger token sequence** per target: in the paper a small trigger is
  optimized so the poisoned document steers the victim LLM; we select a trigger
  word per target by **Qwen3-8B sampling** — the same role the paper's
  `target_asr` gives to GPT-3.5 (`algo/utils.py`).
- **Retriever-side steering**: the poisoned document's embedding is pushed
  toward the query embedding under the retriever (the paper's
  `algo/trigger_optimization.py` hotflip); we pick the trigger whose doc
  embedding is closest to `encode(question)` under **our** bge-base-en-v1.5.

## What we adapt (write-back threat model)

The paper appends the trigger to the victim's query at attack time
(`question += trigger_sequence`). Our attacker cannot do that, so:

- the trigger lives only inside the poisoned document,
- retrieval is by the plain question (semantic similarity), not query+trigger,
- therefore only the generation-side (instruction-following) value of the
  trigger survives — this is a faithful *lower bound* for AgentPoison under a
  KB-only attacker.

## Pipeline

```
baseline/agentpoison/gen_agentpoison_writes.py
  --phase gen        Qwen3-8B sampling: per target, try trigger-word candidates
                     (fragments of the wrong answer + fixed imperative phrases),
                     keep those that make Qwen3-8B answer with the wrong answer.
  --phase retriever  bge white-box: among the firing triggers, pick the one whose
                     poisoned-doc embedding is closest to encode(question).
```

Output uses the same `variants.cluster.poison_writes` layout, so
`longtail_attack.py --phase inject-from` / `--phase eval-after` work unchanged.
`run_agentpoison_victims.sh` evaluates the fixed poison against all four
main-table victims.

## Results

### HotpotQA (60 targets)

| victim | clean | AgentPoison flip | PoisonedRAG flip | TrojanScribe flip | AP ASR | PR ASR | TS ASR |
|---|---|---|---|---|---|---|---|
| xlam-2-8b | 27/60 | 12/27 (44.4%) | 14/27 (51.9%) | **22/27 (81.5%)** | 30/60 | 41/60 | **56/60** |
| qwen3-8b | 31/60 | 18/31 (58.1%) | 21/31 (67.7%) | **23/31 (74.2%)** | 36/60 | 40/60 | **46/60** |
| gpt-oss-20b | 38/60 | 12/38 (31.6%) | 18/38 (47.4%) | **28/38 (73.7%)** | 18/60 | 24/60 | **41/60** |
| llama-3.1-8b | 25/60 | 11/25 (44.0%) | 19/25 (76.0%) | 19/25 (76.0%) | 33/60 | **42/60** | 32/60 |

The adapted AgentPoison is the weakest of the three because its retrieval-side
trigger mechanism assumes the attacker can append the trigger to the victim's
query; under a KB-only attacker only the instruction-following side remains.
Qwen3-8B is the trigger-optimization model, so its own row (18/31) is not a
cross-model transfer; the other three rows are transfers from Qwen3-8B.

### MuSiQue (59 frozen targets)

| victim | clean | AgentPoison flip | PoisonedRAG flip | TrojanScribe flip | AP ASR | PR ASR | TS ASR |
|---|---|---|---|---|---|---|---|
| xlam-2-8b | 7/59 | 6/7 (85.7%) | 7/7 (100%) | 7/7 (100%) | 38/59 | 42/59 | **51/59** |
| qwen3-8b | 9/59 | 7/9 (77.8%) | 9/9 (100%) | 9/9 (100%) | **48/59** | 40/59 | 47/59 |
| gpt-oss-20b | 18/59 | 9/18 (50.0%) | 16/18 (88.9%) | **18/18 (100%)** | 29/59 | 26/59 | **41/59** |
| llama-3.1-8b | 10/59 | 8/10 (80.0%) | 10/10 (100%) | 8/10 (80.0%) | **40/59** | 34/59 | 26/59 |