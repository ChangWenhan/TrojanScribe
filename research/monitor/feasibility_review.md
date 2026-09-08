# Monitor Feasibility Review — 5 Frozen Proposals

**Date:** 2026-09-04 · **Reviewer:** Monitor
**Scope:** feasibility in OUR concrete environment (Qwen3-4B @ localhost:8000, chat/completions only; bge-base-en-v1.5 on CPU, white-box; 24GB VRAM shared with vLLM at ~75%; temp-0 experiments; `delete_poison` cleanup available).

---

## 1. Environment facts that bind every proposal

| Fact | Consequence |
|---|---|
| vLLM API: `/v1/chat/completions` only; **no forced decoding, no logprobs via API** | AR2's probe **must** load HF weights in a 2nd process (~3.5–4GB at 4-bit into ~6GB free VRAM — real OOM risk). AR5 Stage B1 (logprob-confidence filter) is **unavailable** → B2 (consistency) fallback only. |
| bge-base-en-v1.5 on CPU, white-box (`store._ef._model`, SentenceTransformer) | Embedding-gradient HotFlip (AR1 EmbGrad) **is** implementable on CPU via torch autograd (no GPU contention), but slow (~2–6 CPU-h per 20-target run). Black-box/MirrorS-only simplification captures most of the value (see AR1). |
| GPU-hours are the scarce resource (victim eval 20–60 targets ≈ 5–15 min; install ≈ 15–30 min) | Rank = (predicted flip gain) / (implementation cost + GPU runtime). AR2's second-model-load windows must be scheduled when vLLM is idle. |
| `08_longtail.py:68` hardcodes `["embed_hybrid"]`; `Target` dataclass already carries `gold_answer`; store metadata carries `source=hotpotqa:{qid}:{title}` | AR1/AR3/AR5 need a `--variants` flag added to 08; AR3 needs `gold_answer`/`true_paragraph` plumbed through `PoisonedAgent.run_task` (2-line change); AR4's true-chunk identification is exact via `source` metadata — no text matching needed. |
| Attacker-side generation uses temp 0.9 (candidate diversity); victim/selection are deterministic | All frozen protocols remain reproducible; only candidate *sampling* is stochastic (existing convention). |

---

## 2. Scoring table

| Proposal | (a) Implementation cost | (b) GPU/runtime | (c) Risk of failure | (d) Expected gain (frozen) | (e) Stealth risk | (f) Combinability | **Rank** |
|---|---|---|---|---|---|---|---|
| **AR1 MirrorGrad** | **M** — payload.py: `_QUERY_PARAPHRASE_PROMPT`, `_build_query_surface`, `_hotflip_prefix` (CPU autograd), `_margin_select`, `mirrorgrad` branch; 07: `outrank_gold` metric + variants arg; new GPU-free `09_retrieval_ablation.py`; 08: `--variants` flag. EmbGrad is the only hard part. | **Low** — EmbGrad 2–6 CPU-h (off-GPU); 1× 07 victim run (20 tgt) + 1× 08 (60 tgt) + installs ≈ **1.5–2 GPU-h**. | **M** — CPU-autograd fiddliness; main scientific risk: query-rewrite drift + generation ceiling (their threat #1/#2: flips may stay flat even at perfect retrieval — proposal is falsifiable in the "AR2's problem" direction). | **Medium flip gain; high retrieval gain** — TPR@1 0.85→0.95, but flips only 6→8/10 (07) and 10→15/24 (08). Ablation R8: MirrorS-pure already hits TPR@1 0.90 → EmbGrad adds only +0.05. | **M–H** — 6 verbatim-question prefixes/target (same fingerprint as today), pairwise cos ≤ 0.93 = still a tight cluster; HotFlip swaps add PPL anomaly. No defense exists in-pipeline, claims untested. | **Excellent** — retrieval-side only; chunks keep embed_hybrid I-body byte-identical → causal combination with AR2 (probe), AR3 (margin select cluster members), AR4 (margin objective already targets gold). | **4** |
| **AR2 AssertionHardening** | **L** — new `hardening.py` (transformers session, forced-decoding probe, GCG, PPL guard, 4-bit drift calibration), payload.py branch + 4 templates, 07 metrics + `--gcg-pilot`. Largest new component; must manage concurrent vLLM memory. | **Medium–High** — probe 1,800 fwd ≈ 20–40 GPU-min + GCG pilot 2,000–4,000 fwd ≈ 30–70 min + 07 run (30 tgt × 2 variants) ≈ 1 h ⇒ **≈ 2.5–3.5 GPU-h**, plus 4-bit calibration. | **H** — second 4B load into ~6GB free VRAM (4-bit ≈ 3.5–4GB + activations: tight, OOM-capable; fp16 8GB does not fit); 4-bit logit drift needs top-1 agreement ≥ 0.95 gate; probe–end-to-end gap (sole-context vs 8-chunk context) acknowledged (P2 capped below P1). | **HIGHEST flip gain** — poison-follow 0.55→0.80, flips on baseline-correct 0.24→0.60 (+0.36). Directly attacks the diagnosed generation-condition bottleneck. | **M** — T2 quotes gold explicitly (conflict-check exposure), GCG swaps PPL-guarded; probe ranks templates so risk is self-limiting. | **Best integrator** — probe machinery reusable: select AR3 cluster members, AR4 answer chunks, and replace AR5 Stage C; works on top of any retrieval variant. | **2** |
| **AR3 ConsensusCluster** | **M** — payload.py: 5 archetype prompts + `_topic_entity` + filters (cos ≤ 0.65, bigram-Dice ≤ 0.5, ≥6-gram ban) + `generate_cluster`; agent.py: pass gold_answer/true_paragraph (2 lines); 08: `--variants` + `co_retrieval` + stays tags; 07: variant + metrics. Pure prompt+selection work; no new infra. | **Medium** — 5 archetypes × 8 samples × 60 tgt ≈ 2,400 gen calls (≤256 tok each) ≈ 1–2 GPU-h; victim evals P1+P2 (60 tgt × 2) + P3 (20) + installs ⇒ **≈ 3–4 GPU-h**. | **M** — diversity constraints can reject candidates (fallback < 6 chunks); famous-entity ceiling acknowledged (S1: 3–4 FAMOUS stays); hard falsification bound built in (consensus_coverage < 0.80 ⇒ claim void). | **HIGH** — flips 0.42→0.70 (v5) / 0.75 (v8 headline) on the 08 protocol; poison-follow 0.60→0.80. Second only to AR2. | **Best structural stealth of the flip-seeking proposals** — no duplicate blocks, no byte-identical asserts, dispersed cluster defeats TrustRAG K-means/GRADA; BUT volume 300–480 chunks (0.46–0.74%) exceeds AgentPoison budget and `faq` keeps a verbatim-question chunk (A2 ablation quantifies). | **Good** — explicitly orthogonal to AR1/AR4 (must NOT ship in the same run — attribution); pairs well with AR2 probe (probe each archetype, keep flippers) and AR5 gates (S4/S5 applied to cluster members). | **1** |
| **AR4 True-Chunk Exclusion** | **M** — payload.py: `_KILLSLOT_PROMPT`, `_ABSORBER_PROMPT`, `_PREFIX(α)` helper, band calibration (pre-write `search(k=20)` + `_true_chunk_score` via `source` metadata), 5 combo branches; new `08_slot_warfare.py` (clone of 07) + true-in-top8 metric. No store changes. Band calibration against a 65k HNSW index is the fiddly part. | **Medium–High** — 5 combos × 20 tgt × v8 = 800 writes; 5× victim eval (5–15 min ea) + 5× install (15–30 min ea) ⇒ **≈ 2.5–4 GPU-h** (trim to 3 combos ⇒ ≈ 2–2.5 GPU-h). | **M–H** — HNSW approximate-search noise on band calibration (their F5 rank-7/8 forcing is the designated casualty); slot-structure prior (F1) is a guess; Absorber only excludes when pre-rank ∈ {7,8} (honest but narrow mechanism). | **High** — absorber_start 0.24→0.55 (+31pp); F4 conditional flip \| true-excluded = 0.65. Comparable to AR2's claim, wider intervals (±0.12). | **M** — KillSlot chunks contain only TRUE facts (content-review safe — the stealth core); but α=1.0 verbatim prefixes fingerprint 2 chunks/target and total volume 160–800 writes traces to one author. | **Good** — 1–2 answer chunks can be AR2-probed; AR1's margin objective is a complementary (not competing) displacement lever; must NOT run with AR3 (mechanism attribution conflict). | **3** |
| **AR5 Long-Tail + Stealth** | **M–L** — 08 selection pipeline rewritten (stages A–D; **B1 dropped → B2 only**, no llm.py change needed), payload.py `entity_swap` variant (needs gold-containing-chunk fetch machinery), domain-constrained `wrong_answer`, S2 variation loop, S4/S5 gates (k=12 neighbor search + 16-gram pool over 20k corpus chunks), S6 canary; 52-control group protocol. Largest experiment-script surgery; several new gates. | **Medium** — probes 120 cands × 3–4 calls ≈ 360–480 ≈ 30–60 min; 8 poisoned targets + 52 control victim asks + install ⇒ **≈ 1.5–2.5 GPU-h**. | **M–H** — headline is statistically fragile (8 poisoned targets ⇒ ~5–7 baseline-correct ⇒ binomial CI ±37pp at 95%); Stage D thresholds fit in-sample on n=8 (P13 out-of-sample check is the honest test); entity-swap can leak gold strings (risk 4); over-engineering rejection risk (risk 7). | **Medium in absolute terms** — flips 0.42→0.65 but on a *shrunk* poisoned set (8 targets ⇒ ~5 flips vs AR3's ~17); the real value is selection precision + the first measured stealth baseline (P7–P12, e.g., current 97% sibling-in-top-8 is a damning number for the whole line). | **Best-in-class (by design)** — 0.095% budget, real-neighbor band, sibling dispersion, no question prefix, no fixed assert, canary framing. The only proposal that *measures* detectability. | **Orthogonal wrapper** — selection + stealth gates apply to ANY payload (entity_swap, cluster members, probed assertions); natural final hardening pass over the winning variant. | **5** |

---

## 3. Ranked implementation order

**Rank 1 — AR3 (ConsensusCluster).** Highest flip gain per unit of implementation risk: pure prompt+selection work through the existing LLM+CPU-embedding path, zero new infrastructure, no GPU-memory hazard, and it attacks the diagnosed generation-condition bottleneck (0.42→0.75 headline on the 08 set). Its built-in falsification bound (consensus_coverage < 0.80) makes it the most honestly testable claim in the batch.

**Rank 2 — AR2 (AssertionHardening).** Biggest raw gain (+0.36 flips, poison-follow →0.80) and the only proposal that probes the actual victim — but the second 4B load into ~6GB free VRAM is a genuine OOM-grade technical risk, so it must run as a gated pilot (10 targets, 4-bit, top-1 agreement ≥ 0.95 calibration) scheduled in vLLM-idle windows before any full run. Its probe is also the future integrator for AR3/AR4/AR5.

**Rank 3 — AR4 (True-Chunk Exclusion).** High conditional gain (+31pp absorber_start; flip|excluded = 0.65) with moderate implementation, but the 5-combo protocol is GPU-hungry and HNSW band-calibration noise makes its rank-level predictions shaky. Trim to 3 combos (`killslot_only`, `orderforcing_start`, `absorber_start`) to stay inside the GPU budget.

**Rank 4 — AR1 (MirrorGrad).** Cheapest on GPU and the only proposal whose ablation run (exp 09) is entirely GPU-free — but its predicted *flip* gains are the smallest (+2/10, +5/24) and it is explicitly vulnerable to the generation-condition ceiling (i.e., its success partially depends on AR2/AR3 landing). Run the MirrorS+RegionCluster simplification by default (0.90 vs 0.95 TPR@1, saves the whole CPU-autograd stack); keep EmbGrad behind a flag.

**Rank 5 — AR5 (Long-Tail + Stealth).** Its flip claim is the least statistically stable (n≈6 binomial), and it needs the largest script surgery. Its unique contribution is the *first measured stealth baseline* (P7–P12) and a reusable selection filter — so it belongs at the end as a hardening/selection pass over whichever payload variant wins, not as a standalone headline run.

---

## 4. Target-set and script plan (execution plan)

**Same 60-target long-tail set (08 protocol: 120 candidates, seed 7, prior-filter; ~24 baseline-correct denominator) — run these three on the SAME set for direct flip-rate comparability:**
- **AR3 P1/P2:** `experiments/08_longtail.py --candidates 120 --targets 60 --volume 5,8 --variants embed_hybrid,cluster` (+ matched `embed_hybrid` volume-5 control — required, no historical value; co_retrieval + consensus_coverage + stays tags; delete_poison between variants).
- **AR1 (08 arm):** same script, `--variants embed_hybrid,mirrorgrad@v6` (use MirrorS-only unless the 09 ablation shows EmbGrad matters).
- **AR5:** same 120-candidate pool/seed; stages A–D (B2 fallback), 8 poisoned targets (`entity_swap` + S-gates) + 52 unpoisoned controls.

**Same 20-target mixed set (07 protocol: `split_sets` seed 7; 20 targets + 20 controls) — run these on the SAME set:**
- **AR4:** `experiments/08_slot_warfare.py --targets 20 --volume 8 --variants killslot_only,orderforcing_start,absorber_start` (trimmed combos; true-in-top8 pre/post; control-set collateral check).
- **AR1 (07 arm):** `experiments/07_scale_test.py --targets 20 --volume 6 --variants embed_hybrid,mirrorgrad --n-candidates 12` + `experiments/09_retrieval_ablation.py` (GPU-free, run first, any time).
- **AR3 P3:** `07_scale_test.py --targets 20 --volume 5 --variants embed_hybrid,cluster`.
- **AR2:** `07_scale_test.py --targets 30 --volume 8 --variants embed_hybrid,assertion_hardened --gcg-pilot` (30 targets is a superset of the 20 — reuse for both AR2 predictions and cross-checks).

**Ordering within the campaign (GPU-aware):**
1. `09_retrieval_ablation.py` (GPU-free) — validates AR1 pieces before any GPU spend.
2. AR3 P1/P2/P3 (dominant GPU consumer) — headline generation-condition test.
3. AR2 pilot (vLLM-idle window; 4-bit calibration gate; abort → B2-style API-only probe downgrade if top-1 agreement < 0.95) then full 30-target run.
4. AR4 slot warfare (3 combos).
5. AR1 07 + 08 arms (cheap; can fold into AR3's already-probed targets if scheduling is tight).
6. AR5 selection+stealth pass, applied to the winning payload.

**Total estimated GPU: ≈ 12–14 hours** (AR3 ~3.5–4h, AR2 ~2.5–3.5h incl. pilot, AR4 ~2–2.5h, AR1 ~1.5–2h, AR5 ~1.5–2.5h) + ~4–8 CPU-h for AR1 EmbGrad (optional) and AR5 gates.

**Comparability rules enforced by the monitor:**
- Every flip comparison uses a **matched-volume control rerun on the same targets** (AR1 §8, AR3 §8, AR2 §5.2) — never the historical CSV alone.
- AR3 and AR4 never share a run (attribution); AR2's probe + AR5's gates are the sanctioned cross-proposal integrations.
- Own-target poison ids only for all retrieval metrics (cross-target contamination is real at 120–480 chunks).
- AR2's second-model-load is the only step that can disturb the shared vLLM — schedule and gate it as above; never run it concurrently with victim evaluations.