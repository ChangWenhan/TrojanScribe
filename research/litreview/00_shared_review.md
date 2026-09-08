# Shared Literature Review — AgenticRAG Write-Back Poisoning

**Project:** Supply-chain write-back poisoning of shared AgenticRAG knowledge bases.
A malicious subagent (installed from an agent marketplace) writes fake knowledge chunks into a
shared multi-user vector-DB KB; innocent victim agents (Qwen3-4B, temp 0, tool calling) retrieve
top-8 chunks and answer HotpotQA questions (65k-chunk corpus, bge-base embeddings).
Current best: TPR@8=0.86, flip rate 10/42 (24%) on baseline-correct answers.

**Status board (experiment results appended here AFTER frozen predictions; frozen files hash-locked in research/monitor/freeze_manifest.sha256):**

| # | Method | Frozen prediction | Actual result | Verdict |
|---|--------|-------------------|---------------|---------|
| AR3 | ConsensusCluster | G3: 16-20/24 flips (0.67-0.83); coverage >= 0.80 | **17/20 = 85% flips** (range exceeded); after_EM 0.333->0.117; consensus_cov 0.13 < 0.80 (dice-defined voices not achieved); control v8 = 45% | CONFIRMED (headline) / VOID (mechanism metric per clause) |
| AR4 | True-Chunk Exclusion | F3 flips: 0.20/0.45/0.55; F2 displacement 0.35/0.45/0.40 | F3: 0.20/0.40/0.50 all CONFIRMED; F2: killslot 0.80 & absorber 0.60 FAILED (band calibration weak); tpr@8 0.65 (dormancy: ~35% targets unfired); controls clean | PARTIALLY CONFIRMED |
| AR2 | AssertionHardening | P3: flips 0.60; probe-accept 0.80 | Pilot gate FAILED: 4-bit probe top-1 agreement 0.66 < 0.95; SYSTEM_VICTIM tool-mandate behavioral mismatch; no VRAM for fp16 probe | REJECTED at pilot gate |
| AR5 | Long-tail + Stealth | P3 flips >= 0.55; P7 cos >= 0.85; P8 sibling <= 0.10 | P3 VOID (n=1 baseline-correct); P7 = 0.745 FAILED; P8 = 0.95 FAILED; selection over-filters (1/11); entity-swap clusters | REJECTED (falsified) |
| **B1** | **Baseline comparison (controlled vars, ReAct agent + bge KB + 60 targets)** | — | **ASR: poisonedRAG 25% > cluster 8% > 0 (naive/ours/topicattack). Strict flip-rate (base EM n=21): topicattack 95% > cluster 86% > ours 57% > poisonedRAG 52% > naive 14%. cluster = only directional + near-top-flip method; flips reproduce across agent frameworks (0.85 langgraph / 0.86 ReAct)** | cluster robustly best directional knowledge-flip attack; topicattack = non-directional disruption |---|--------|-------------------|---------------|---------|
| AR3 | ConsensusCluster | G3: 16-20/24 flips (0.67-0.83); coverage >= 0.80 | **17/20 = 85% flips** (range exceeded); after_EM 0.333->0.117; consensus_cov 0.13 < 0.80 (dice-defined voices not achieved); control v8 = 45% | CONFIRMED (headline) / VOID (mechanism metric per clause) |
| AR4 | True-Chunk Exclusion | F3 flips: 0.20/0.45/0.55; F2 displacement 0.35/0.45/0.40 | F3: 0.20/0.40/0.50 all CONFIRMED; F2: killslot 0.80 & absorber 0.60 FAILED (band calibration weak); tpr@8 0.65 (dormancy: ~35% targets unfired); controls clean | PARTIALLY CONFIRMED |
| AR2 | AssertionHardening | P3: flips 0.60; probe-accept 0.80 | Pilot gate FAILED: 4-bit probe top-1 agreement 0.66 < 0.95; SYSTEM_VICTIM tool-mandate behavioral mismatch; no VRAM for fp16 probe | REJECTED at pilot gate |
| AR5 | Long-tail + Stealth | P3 flips >= 0.55; P7 cos >= 0.85; P8 sibling <= 0.10 | P3 VOID (n=1 baseline-correct); P7 = 0.745 FAILED; P8 = 0.95 FAILED; selection over-filters (1/11); entity-swap clusters | REJECTED (falsified) |

---|----------------------|-------------------|---------------|---------|
| AR3 | ConsensusCluster (frozen G3: 16-20/24 flips) | **17/20 = 85%** flips (range exceeded), after_EM 0.333->0.117, consensus_cov 0.13 < 0.80 bound (mechanism claim void per clause), control v8 = 45% | CONFIRMED (headline) / VOID (mechanism metric) |
| AR5 | Long-tail+Stealth (frozen P3>=0.55, P7>=0.85, P8<=0.10) | P3 VOID (n=1); P7=0.745 FAILED; P8=0.95 FAILED; selection over-filters (1/11 baseline-correct) | REJECTED (falsified) |
| AR4 | True-Chunk Exclusion (frozen F3: flips 0.20/0.45/0.55) | F3 all CONFIRMED (0.20/0.40/0.50); displacement F2 killslot/absorber FAILED; tpr@8 0.65 (dormancy effect); controls clean | PARTIALLY CONFIRMED |
| AR2 | AssertionHardening (frozen P3: flips 0.60, probe-accept 0.80) | Pilot gate FAILED: 4-bit probe top-1 agreement 0.66 < 0.95 (behavioral mismatch: SYSTEM_VICTIM tool mandate; no VRAM for fp16 probe) | REJECTED at pilot gate |
 |

---

# Librarian 1: RAG Poisoning & Backdoor Attacks

Scope: 2023–2026 attack literature on knowledge-base poisoning / backdoor / text-injection attacks against Retrieval-Augmented Generation (RAG). All entries verified against arXiv / ACL Anthology / USENIX / ACM pages. Venue is stated only where verified. Our evaluation setting for transferability notes: victim = open 4B LLM; corpus = HotpotQA (~65k chunks); retriever = bge-base embeddings; top-k = 8; success metric = wrong-answer flip (ASR).

---

## 1. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models

**Venue:** USENIX Security 2025 (arXiv:2402.07867). Zou, Geng, Wang, Jia (Lehigh Univ.).

**Summary:** First knowledge-corruption attack on RAG: an attacker injects a few malicious texts into the knowledge DB so the LLM emits an attacker-chosen target answer R for an attacker-chosen target question Q. Formulated as an optimization problem with a black-box variant (no access to DB, LLM, or retriever queries) and a white-box variant (retriever parameters known). Evaluated on NQ, HotpotQA, MS-MARCO with 8 LLMs (GPT-4, LLaMA-2, PaLM 2, ...) and three real applications (advanced RAG, Wikipedia chatbot, LLM agent).

**Technique:** Derives two conditions per poisoned text P: (i) retrieval condition — P must land in the top-k retrieved set for Q; (ii) generation condition — the LLM must output R when P alone is context. The two conditions conflict, so P is decomposed as P = S ⊕ I: sub-text I (crafted via a prompted LLM, e.g., GPT-4, or HotFlip-style optimization in white-box) makes the LLM generate R; sub-text S is then optimized so that S⊕I stays semantically similar to Q (retrieval) without breaking I's generation effect. Multiple P's (typically 5) are injected per target question.

**Metrics:** ~90% ASR with 5 poisoned texts per question on million-scale KBs (NQ corpus = 2,681,468 texts). Black-box, PaLM 2: 97% (NQ), 99% (HotpotQA), 91% (MS-MARCO) ASR; F1 (poison retrieved) > 90% in almost all cases. Outperforms 5 baselines (<70% ASR). Paraphrasing- and perplexity-based defenses insufficient.

**Transferable to our setting:** This is the blueprint. HotpotQA is one of their main datasets (99% ASR there), so the domain matches directly; our 65k-chunk corpus is ~40x smaller than theirs, which should make the retrieval condition easier. bge-base is open and gives the attacker white-box access (their white-box S-crafting uses retriever gradients). Our 4B open LLM is white-box too, so the generation condition can be verified/optimized directly instead of via a prompted surrogate — the main adaption needed is re-deriving the S⊕I recipe for bge-base + the specific 4B model and confirming flip rates at top-k=8 (their top-k is smaller, e.g., 5).

---

## 2. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models

**Venue:** arXiv:2405.13401 (2024), Cheng, Ding, Ju, Wu, Du, Yi, Zhang, Liu (SJTU). Preprint (OpenReview submission); not yet verified at a named venue.

**Summary:** Joint backdoor attack that compromises the retriever side of RAG: the adversary builds multiple "purpose-driven backdoors" between predefined triggers and poisoned contexts, so retrieval behaves normally for clean queries but always returns semantic-consistency poisoned content for triggered queries, inducing target outputs in the LLM (fact-change, bias, and backdoor-style jailbreaking scenarios). LLM parameters are frozen, so the attack is cheap and hard to attribute.

**Technique:** (1) trigger sets T (robustness triggers like "cf"/"mn", or unintentional user triggers); (2) poisoned contexts generated with a teacher LLM per poisoned query q* = q ⊕ τ; (3) knowledge-graph metadata (subject-object triads) appended as positive samples to improve fine-grained hard matching; (4) joint backdoor implantation: multi-shortcut pairs orthogonally optimized with contrastive learning so different backdoors map to disjoint parameter subspaces. No knowledge of the LLM required.

**Metrics:** vs. prompt-only baseline: >40% improvement in Keyword Matching Rate (KMR) and >80% in Exact Matching Rate (EMR) on fact-checking/classification tasks (NQ, WQ, HotpotQA, MS-MARCO, SST-2, AGNews); harmful-bias scenario 96% KMR / 94% EMR; jailbreak harmfulness ratios 29–92% vs <10% without attack; clean retrieval capability largely preserved; transferable across retrievers/LLMs and survives Chain-of-Thought. One-shot queries (NQ, WQ) attacked more easily than multi-hop (HotpotQA).

**Transferable to our setting:** Validates the trigger-conditioned backdoor model for a shared KB: the subagent writes poisoned chunks associated with topic triggers; other agents' queries about that topic retrieve them. Their finding that multi-hop HotpotQA queries are *harder* to attack than single-hop matters for us (we may need more poison chunks or stronger S). Contrastive-optimized contexts are an alternative to PoisonedRAG's S⊕I decomposition when we want a family of trigger-chunks rather than per-question chunks.

---

## 3. CorruptRAG (Practical Poisoning Attacks against Retrieval-Augmented Generation)

**Venue:** ACM SACMAT 2026 (to appear); arXiv:2504.03957. Zhang, Chen, Liu, Nie, Li, Liu, Fang (Nankai / U. Louisville).

**Summary:** Argues prior attacks (notably PoisonedRAG) require enough poisoned texts per query to *outnumber* correct-answer texts in the top-N — unrealistic and detectable. CorruptRAG attacks with exactly **one poisoned text per target query**, enhancing feasibility and stealth, and reports higher ASR than existing baselines on large-scale datasets while bypassing four defenses.

**Technique:** Poisoned text p_i split into p_i^s (retrieval sub-text) and p_i^h (attack sub-text), with p_i^s set equal to the query q_i itself. CorruptRAG-AS builds p_i^h from a template containing the query, the correct answer, and the target answer (e.g., "many outdated corpora state [wrong]; latest data confirms [target]") to override correct-answer texts in the top-N. CorruptRAG-AK uses few-shot LLM prompting (GPT-4o-mini) to rewrite p_i^h into coherent "adversarial knowledge" that generalizes the flip to semantically related queries (e.g., "Are we living in the 19th century?").

**Metrics:** Higher ASR than PoisonedRAG and other baselines on three large-scale QA datasets with a single injected text per query; robust against four advanced defense mechanisms; attack-crafting cost on the order of GPT-4o-mini API pricing ($0.15/M input, $0.60/M output tokens).

**Transferable to our setting:** Strongly aligned: our attacker (malicious subagent) writes back a small number of chunks; per-query single-text attacks are the right budget model. The "outdated corpora / latest data confirms" template is cheap and language-model-agnostic, and it directly flips answers in the LLM. The p_i^s = q_i trick is a zero-cost retrieval condition for bge-base (query-doc similarity is high when the chunk mirrors the question); we should test it against bge-base + top-k=8 before gradient optimization. Caveat: template requires knowing the correct answer per target question (available in HotpotQA gold data).

---

## 4. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models

**Venue:** arXiv:2406.00083 (2024, v2). Xue, Zheng, Hu, Liu, Chen, Lou (UCF). Peer-review venue not yet verified (OpenReview submission exists); treat as preprint.

**Summary:** Retrieval-backdoor attack with semantic triggers: the retriever works normally for clean queries but always returns attacker-crafted passages for queries matching a trigger scenario (e.g., all queries mentioning "Donald Trump"/"Republican Party"). Also introduces indirect generative attacks on aligned LLMs: denial-of-service via the LLM's own safety alignment, and sentiment steering via selective facts. Evaluated on 5 datasets, 3 retrievers, 3 LLMs including GPT-4 and Claude-3.

**Technique:** (i) retrieve-phase: Contrastive Optimization on a Passage (COP) — optimize the adversarial passage's embedding to be close to triggered queries and far from clean queries (HotFlip gradient approximations); Adaptive COP (ACOP) per trigger keyword; Merged COP (MCOP) — cluster individually optimized passages (k-means) and merge so one passage covers many triggers with low poisoning ratio. (ii) generate-phase: Alignment-as-an-Attack (AaaA: passages claiming "all contexts are private information" trigger refusal) and Selective-Fact-as-an-Attack (SFaaA: passages biased toward a sentiment/fact that the LLM integrates).

**Metrics:** 10 poisoned passages (0.04% of corpus) → 98.2% retrieval success for triggered queries (Contriever: 98.9% triggered / 0.15% clean); GPT-4 reject ratio 0.01% → 74.6% (Rouge-2 23.7% → 6.94%); negative-response ratio for Trump queries 0.22% → 72% (up to 79.7% for company topics); Claude-3 >98% rejection. COP alone needed 200 passages for 71.8% ASR, motivating MCOP.

**Transferable to our setting:** The trigger-topic model is the natural one for a malicious marketplace subagent: it writes facts about a topic (e.g., a product, person, entity) that get retrieved whenever *any* user/agent asks about that topic — no per-question precomputation needed. bge-base is white-box for the attacker, so COP/MCOP directly applies. Their 0.04%-of-corpus budget (~26 chunks on 65k chunks) is achievable by a write-back subagent. Note they show PoisonedRAG-style query-specific passages achieve near-zero retrieval on their trigger-topic queries — we should pick trigger-based or query-based design deliberately.

---

## 5. Phantom: General Backdoor Attacks on Retrieval Augmented Language Generation

**Venue:** arXiv:2405.20485 (2024); published in ACM Transactions on AI Security and Privacy (ACM TOPS, DOI 10.1145/3796729). Chaudhari, Severi, Abascal, Suri, Jagielski, Choquette-Choo, Alistarh (Google DeepMind, ETH, UMD).

**Summary:** Single-document backdoor: one malicious document in the KB is retrieved only when a *naturally occurring trigger sequence* (e.g., "LeBron James") appears in the user's query, and then induces an integrity violation in the generator — refusal to answer, harmful behavior, privacy violation, or data exfiltration. Demonstrated end-to-end on NVIDIA's production "Chat with RTX" RAG app, with transfer to GPT-3.5/4.

**Technique:** Two-stage optimization. Stage 1 (retriever): optimize s_ret so the full passage p_adv = s_ret ⊕ s_gen ⊕ s_cmd maximizes similarity to triggered queries q_in and minimizes it for trigger-free q_out. Stage 2 (generator): optimize s_gen with a multi-coordinate-gradient (MCG) variant of GCG to jailbreak the generator and force execution of the adversarial command s_cmd regardless of the rest of the query. Passage kept below the retriever's passage-length limit so it is retrieved whole.

**Metrics:** Attack transferable across Gemma, Vicuna, Llama (open) and GPT-3.5 Turbo / GPT-4 (closed); MCG converges faster than GCG (fewer iterations/batch); end-to-end ChatRTX demonstrations of refusal and data exfiltration. Exact ASR numbers not centrally reported in the abstract; effectiveness is objective- and model-dependent.

**Transferable to our setting:** Provides the "optimize the generation condition with gradient-based token search" machinery (MCG) that PoisonedRAG lacks (they use LLM-prompted I). Our 4B victim is white-box, so MCG-style optimization of s_gen against the actual victim model (not a surrogate) should yield strong per-chunk flip power — likely *higher* than the LLM-prompted I approach on a small model. Their natural-trigger conditioning also maps to subagent-written topic chunks.

---

## 6. GARAG: Typos that Broke the RAG's Back — Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations

**Venue:** Findings of EMNLP 2024 (ACL Anthology 2024.findings-emnlp.161). Cho, Jeong, Seo, Hwang, Park (NAVER).

**Summary:** Black-box attack that perturbs a clean document with low-level text errors (typos/character swaps) to break both the retriever and the generator simultaneously, simulating noisy documents "in the wild". Shows RAG robustness must be evaluated holistically — a small fraction of perturbed tokens is enough to degrade retrieval, grounding, and end-to-end QA.

**Technique:** Genetic algorithm over typo-perturbed document variants; dual objectives: retrieval-error loss and grounding-error loss, searching for documents in the "holistic error" zone (both errors at once). Perturbation budget = N·pr_pert tokens (small, e.g., a few percent of tokens); answer tokens are kept unaltered. Black-box: only model outputs needed.

**Metrics:** >70% attack success ratio across datasets/retrievers (DPR, Contriever) and LLMs (LLaMA2, Vicuna, Mistral); end-to-end EM drops by ~30% on average, up to ~50%. Lower perturbation rates were found more threatening (harder to notice, still effective).

**Transferable to our setting:** A stealth baseline for our poison chunks: if the write-back must survive human/automated review, typo-level perturbations are nearly invisible yet measurably hurt bge-base retrieval ranking and 4B-LLM grounding. Useful as a control/ablation in our experiments (natural-text I vs. perturbed document vs. S⊕I).

---

## 7. AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases

**Venue:** NeurIPS 2024 (arXiv:2407.12784). Chen, Xiang, Xiao, Song, Li (UChicago / UIUC / UW-Madison).

**Summary:** First backdoor attack on generic RAG-based LLM agents: poisons the agent's long-term memory or RAG knowledge base with a few malicious demonstrations so that whenever a user instruction contains the optimized trigger, the malicious demo is retrieved with high probability and steers the agent toward an adversarial target action; benign instructions are unaffected. No model training or fine-tuning required. Demonstrated on an autonomous-driving agent, a knowledge-intensive QA agent, and healthcare EHRAgent.

**Technique:** Trigger generation as constrained optimization (gradient-guided beam search over discrete tokens): map triggered queries into a unique, compact region of the retriever's embedding space (retrieval effectiveness + compactness losses) while constraining target-generation loss and trigger-coherence (perplexity) loss. Optimized triggers transfer across different embedders (open-source surrogate → proprietary retriever).

**Metrics:** Average retrieval success 82% and end-to-end ASR ≥ 80% (63% end-to-end; 62.6% real-world environmental impact with safety filters) across three agents, with benign performance drop ≤ 1% and poison rate < 0.1%; effective even with a single injected instance and a single-token trigger; evades perplexity-based and rephrasing defenses.

**Transferable to our setting:** Closest published analogue to our threat model (subagent poisons shared knowledge/memory consumed by other agents). Two directly reusable ideas: (1) optimizing the *trigger/query side* so poisoned chunks are retrieved reliably — we can do the same against bge-base since the attacker controls both chunks and (partially) the query surface; (2) transferability of triggers/embeddings across embedders means bge-base-specific optimization should still work if victims switch embedders. Their poison-rate budget (<0.1% ≈ 65 chunks on 65k) is the upper bound we should target for write-back stealth.

---

## 8. Poisoning Retrieval Corpora by Injecting Adversarial Passages

**Venue:** Findings of EMNLP 2023 (ACL Anthology 2023.emnlp-main.849; arXiv:2310.19156). Zhong, Huang, Wettig, Chen (Princeton). — Pre-2024 but the foundational corpus-poisoning result all 2024–2026 RAG attacks build on.

**Summary:** A malicious user injects a small number of adversarial passages (50 tokens, discrete token perturbations) into a retrieval corpus so that dense retrievers return them among top-k for a broad set of queries — including *unseen* and *out-of-domain* queries. Demonstrates the "SEO-for-LLMs" threat: passages optimized on one domain mislead retrieval in other domains, and unsupervised retrievers (Contriever) are especially vulnerable.

**Technique:** Gradient-based HotFlip-style optimization: start from a natural passage (optionally with a fixed semantic prefix) and iteratively swap tokens to maximize embedding similarity to a set of training queries; k-means clustering of queries extends to multiple passages. Passage length 50 tokens, ~5000 replacement steps.

**Metrics:** 1 adversarial passage fools >75% of NQ/MS-MARCO queries on Contriever; 10 passages >90%; 500 passages push supervised retrievers (DPR, ANCE) past 50%; cross-domain transfer: 50 passages optimized on NQ mislead >94% of financial-forum (FiQA) queries; 250-token passages reach 20.1% on ColBERT.

**Transferable to our setting:** Directly relevant to the retrieval-condition half of our attack. bge-base is closer to supervised retrievers (trained with contrastive objectives on large corpora) than to Contriever, so we should expect mid-range vulnerability (~50%+ with hundreds of passages) unless we also use the query-mirroring tricks from CorruptRAG — i.e., retrieval-condition design matters. Note their passages are gibberish-looking; combining with a natural prefix (they show fixed-prefix variant) improves stealth for write-back chunks.

---

## 9. Poisoning Web-Scale Training Datasets is Practical ("pile of poison" attacks)

**Venue:** IEEE S&P 2024 (arXiv:2302.10149). Carlini, Jagielski, Choquette-Choo, Paleka, Pearce, Anderson, Terzis, Thomas, Tramèr (Google / ETH / NVIDIA).

**Summary:** Shows web-scale datasets can be poisoned *today* by exploiting mutable web content: (1) split-view poisoning — the dataset annotator's view of a URL differs from what clients later download, so the attacker controls content after indexing; (2) frontrunning poisoning — predict when a Wikipedia dump snapshots each article and edit just before the snapshot, so even quickly-reverted edits persist in the dataset forever. Introduces the "pile of poison" bombing/spectre attack style: a tiny controlled fraction of the corpus suffices to flip model behavior.

**Technique:** No model-level optimization — pure supply-chain manipulation. Split-view: buy cheap domains, serve benign content at curation time, malicious content at training time. Frontrunning: exploit the documented linear Wikipedia snapshot order (article snapshot times predictable to the minute) and time edits accordingly; moderation latency (~minutes) is longer than the exposure window needed.

**Metrics:** 0.01% of LAION-400M or COYO-700M poisonable for ~$60 USD; conservative analysis: 6.5% of Wikipedia documents could be poisoned via frontrunning; defenses (cryptographic integrity checks, randomized/time-gated snapshots) are proposed but not deployed.

**Transferable to our setting:** Provides the supply-chain justification for our threat model: if uncurated, mutable content routinely ends up in shared knowledge stores, then a malicious subagent's write-back to a shared vector DB is the same trust-assumption failure, but *easier* — a vector DB write is immediate and persistent (no snapshot window needed). Frames the "how did fake chunks get there" defense question we should anticipate in the paper's discussion.

---

## 10. Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems

**Venue:** USENIX Security 2026 (Chang, Hongyan et al.; USENIX Security '26 proceedings PDF). Also cited in the 2025–2026 literature as the first end-to-end IPI under realistic retrieval.

**Summary:** Shows unoptimized indirect prompt injection rarely survives retrieval, then fixes it: decompose the injected content into a *trigger fragment* (a compact token prefix whose only job is to guarantee retrieval) and an *attack fragment* (arbitrary malicious instructions). Black-box prefix optimization over embedding APIs yields near-100% retrieval of a single injected item across 11 benchmarks and 8 embedding models; first end-to-end IPI exploits under natural queries on RAG and multi-agent systems (e.g., one poisoned email coercing GPT-4o to exfiltrate SSH keys, >80% success).

**Technique:** Attack fragment D_adv is given; optimize a ~10-token prefix x so that x ∥ D_adv ranks in top-K against a corpus dense with relevant benign documents. Black-box: query the embedding API only (OpenAI text-embedding-3-small etc.); cost as low as $0.21 per target user query; explicitly beats white-box HotFlip (which requires retriever gradients) and naive query-repetition heuristics.

**Metrics:** Near-100% retrieval across 11 benchmarks / 8 embedders (open and proprietary); end-to-end multi-agent SSH-key exfiltration >80% per trial with a single poisoned email and zero user interaction; evaluated defenses (similarity thresholds, etc.) insufficient.

**Transferable to our setting:** The retrieval-condition half of our attack in black-box form: if we assume the subagent cannot access bge-base internals, this black-box prefix optimization (embedding-API-only, ~10 tokens, ~$0.21/query) is the S-crafting method to use — and it composes with any attack fragment (CorruptRAG template, PoisonedRAG I, MCG s_gen). It also covers the multi-agent part of our pipeline (poisoned chunk enters another agent's context through retrieval). Strong candidate for the S half of P = S ⊕ I.

---

## 11. Unleashing Worms and Extracting Data: Escalating the Outcome of Attacks against RAG-based Inference in Scale and Severity Using Jailbreaking

**Venue:** arXiv:2409.08045 (2024). Cohen, Bitton, Nassi (Technion / Cornell Tech / Intuit). Preprint.

**Summary:** With a jailbroken generator, RAG attacks escalate in two directions: (1) severity — from membership-inference/entity extraction to full document extraction from the RAG database (Greedy Embedding Attack / Dynamic GEA: iteratively craft queries whose embeddings match target documents, then force the jailbroken model to output them); (2) scale — adversarial self-replicating prompts act as a computer worm across an ecosystem of RAG-based apps (e.g., email assistants): each infected app performs a malicious payload and propagates the prompt into other apps' databases.

**Technique:** Worm prompt = jailbreaking command j + replication instructions r + payload instructions m, engineered to survive repeated inferences (the model must *output* the instructions, not just follow them). Propagation happens because RAG systems actively index received content (email) — the worm writes itself into the next victim's KB via normal interaction. Extraction uses embedding-probing queries (GEA/DGEA) to pull documents into context.

**Metrics:** 80–99.8% of the data stored in a Q&A chatbot's RAG database extractable (GEA/DGEA); worm: >90% replication + payload success for up to 11 propagation hops, ~20% combined propagation per interaction; Claude 3.5 Sonnet ~100% success even at 20 hops; extraction quality degrades with context size (F1 0.78 at 10 documents → 0.58 at 100; error rate 0.26→0.37 on Gemini 1.5 Flash).

**Transferable to our setting:** Validates the *write-back propagation* mechanism central to our scenario: content that enters a shared, actively-indexed KB spreads to other users/agents without further attacker action. The worm machinery (self-replication) is optional for us, but the finding that persistent injected content survives and operates across inference chains directly supports "malicious subagent writes fake knowledge → innocent agents retrieve and answer wrongly" as a systemic (not single-victim) threat.

---

## 12. RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis (defense, from the attack literature)

**Venue:** arXiv:2411.18948 (2024). Preprint; cited by the 2025 literature as a leading poisoning detector.

**Summary:** Detection pipeline that classifies whether a RAG response is poisoned or correct by analyzing the victim LLM's activations (final input token across all layers). Key finding: generating a poisoned answer leaves distinguishable activation traces vs. generating the correct answer. Works across 5 LLMs, 4 retrievers, 3 datasets, and against PoisonedRAG/GARAG/PAPRAG-style attacks.

**Technique:** Collect activations during poisoned and correct generation; extract per-layer features; train a ResNet18 CNN with triplet margin loss (few-shot adaptable) to embed "poisoned vs. correct" activation signatures. Inference-time: given a query + response + activations, flag the response if the signature is poisoned.

**Metrics:** 98% true-positive rate with false-positive rate ≈ 1% across RAGs with five different LLMs and four retrievers on three datasets; robust to different poisoning types/levels.

**Transferable to our setting:** The strongest published defense we should benchmark against (and, if feasible, evade). Two implications: (1) activation-based detection exists at the *response* level — our wrong-answer-flip metric is exactly the signal it monitors, so we should report flip quality (does the poisoned answer look "natural"?) as a secondary metric; (2) it is LLM-specific and needs poisoned-response data — a poisoned-chunk-only detector (no labels) is out of scope for this paper, which works in our favor.

---

## Synthesis notes (factual, from the papers above)

- **Two-condition design is standard.** Every strong 2024–2026 attack separates *retrieval condition* (chunk must enter top-k) from *generation condition* (chunk must dominate the LLM's answer), and decomposes the poisoned text accordingly: PoisonedRAG's P=S⊕I, CorruptRAG's p^s + p^h, Phantom's s_ret ⊕ s_gen ⊕ s_cmd, Chang et al.'s trigger fragment ⊕ attack fragment.
- **Poison budgets are tiny.** 5 texts/question (PoisonedRAG), 1 text/query (CorruptRAG), 10 passages / 0.04% of corpus (BadRAG), <0.1% (AgentPoison), 0.01% web-scale (Carlini et al.) — all within what a write-back subagent can realistically deposit into a shared 65k-chunk KB.
- **HotpotQA appears in the main evaluations** of PoisonedRAG (99% ASR, black-box, PaLM 2) and TrojanRAG (multi-hop noted as harder), so our corpus choice is directly comparable to the literature.
- **Known weak points of prior attacks we can improve:** PoisonedRAG needs several texts to outnumber correct chunks (CorruptRAG's critique); query-specific chunks fail on trigger-topic queries (BadRAG's critique); LLM-prompted generation sub-texts (PoisonedRAG) are not optimized against the actual victim LLM (Phantom's MCG does this for trigger-based attacks).

## Most promising attack-optimization idea for our setting (see RETURN summary)

Combine (a) PoisonedRAG's S⊕I decomposition, (b) Phantom-style MCG token optimization of the generation sub-text against our actual white-box 4B victim (instead of a surrogate-prompted I), and (c) CorruptRAG's single-text "outdated corpora / latest data confirms [target]" template as the I seed — yielding per-chunk flip power strong enough that 1–5 chunks in top-8 flip HotpotQA answers; with the S half either HotFlip-optimized against bge-base (white-box) or Chang-et-al. black-box prefix optimization (embedding-API-only) as fallback. Verify with AgentPoison-style embedding-region compactness checks so the chunks cluster near target question embeddings.
# Librarian 2: Defenses Against RAG Poisoning & Injection

Defense-landscape survey for our attack scenario: **a malicious subagent with valid write permissions writes plausible-looking chunks into a shared multi-user knowledge base; innocent victim agents retrieve those chunks and answer wrongly.** Focus on corpus poisoning, document/prompt injection, and post-retrieval/generation defenses (2024-2026). All titles/venues verified; numbers are as reported in the cited papers.

---

## 1. PoisonedRAG (attack + its own countermeasure analysis)

- **Title:** PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models
- **Venue:** USENIX Security 2025; arXiv:2402.07867 (2024). Authors: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia.
- **Mechanism (why relevant):** The canonical corpus-poisoning attack. Injects 5 malicious texts per target question into a million-scale KB, each text split into a retrieval sub-text (query-similar prefix) and a generation sub-text (LLM-generated fake but plausible knowledge). The authors explicitly evaluate the standard countermeasures **paraphrasing and perplexity-based filtering** and show both fail.
- **Effectiveness:** Up to **97% ASR** (NQ, black-box, 5 injected texts into a 2.68M-doc corpus); baselines <70%. Defenses tested (paraphrasing, PPL-based detection) are "insufficient".
- **Our scenario / evasions:** This is the closest prior art to our write-back attack. Our chunks are *plausible natural language produced by a real LLM agent*, i.e., inherently low-perplexity and paraphrase-robust, so the two defenses it examined will not catch us. It validates that small numbers of crafted chunks reliably flip answers even against million-scale KBs, and that the victim LLM's instruction-following is the weak link.

---

## 2. RobustRAG (certified isolate-then-aggregate)

- **Title:** Certifiably Robust RAG against Retrieval Corruption
- **Venue:** ICML 2024; arXiv:2405.15556. Authors: Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, Prateek Mittal.
- **Mechanism:** Retrieval-time aggregation: passages are isolated into groups, the LLM answers each group independently, then outputs are **securely aggregated** via keyword-count voting or decoding-stage probability aggregation. Provides *certifiable* robustness: for certain queries the answer is provably correct even if an adaptive attacker injects a bounded number of malicious passages.
- **Effectiveness:** Certifiable robust accuracy 24-71% across datasets (e.g., 71% on RealtimeQA-MC when 1 passage may be corrupted); empirically, **ASR drops from 90%+ (vanilla RAG under PIA/poisoning) to <10%** in almost all cases; clean-accuracy drop ≤11% (≤5% in most cases).
- **Our scenario / evasions:** Robustness holds only when malicious passages are a **minority** of the top-k and benign passages alone suffice to answer. A malicious agent can flood the KB with many copies/variants of our plausible chunk so poisoned documents dominate the retrieved set (the saturation attack), and RobustRAG collapses exactly when poisoned docs outnumber benign ones (shown in TrustRAG's evaluation, Table 3.3). We should also spread the target claim across *many individually-plausible documents* so keyword/decoding aggregation still reaches a malicious majority.

---

## 3. TrustRAG (clustering + LLM self-assessment conflict resolution)

- **Title:** TrustRAG: Enhancing Robustness and Trustworthiness in RAG
- **Venue:** arXiv:2501.00879 (2025); presented at AAAI 2026 Workshop on Trust and Control in Agentic AI (TrustAgent). Authors: Huichi Zhou, et al.
- **Mechanism:** Two-stage, training-free, plug-and-play. Stage 1 "Clean Retrieval": K-means clustering of retrieved-document embeddings plus ROUGE-L n-gram preservation to drop dense clusters of mutually-similar malicious documents. Stage 2 "Conflict Resolution": prompt the LLM to (a) generate internal knowledge, (b) consolidate consistent documents and filter conflicting/irrelevant ones, and (c) self-assess whether to answer from internal or external knowledge.
- **Effectiveness:** First defense that handles multi-document injection where poisons **outnumber** clean docs (RobustRAG fails there). On NQ/HotpotQA/MS-MARCO with PIA and PoisonedRAG, keeps high accuracy and sharply reduces ASR vs baselines; K-means filtering is especially effective at poison rates >20%. (Independent benchmark RSB, arXiv:2505.18543, ranks TrustRAG best among tested hybrid defenses, though still only partially effective.)
- **Our scenario / evasions:** The clustering stage keys on **embedding density/similarity of multiple injected documents** — an agent writing many near-duplicate chunks would be flagged. Countermeasure: write **one** semantically-distinct but plausible chunk per target fact, dispersed over time/topics so no tight cluster forms; rely on the Conflict Resolution stage failing to *detect* conflict because our claim is plausible and consistent with itself (the LLM consolidates it as "consistent information"). Since our attacker is an authorized writer, we can also read what's in the KB and make our chunk overlap stylistically with clean documents.

---

## 4. AstuteRAG (internal-knowledge vs. external conflict resolution)

- **Title:** Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models
- **Venue:** ACL 2025 (Long); arXiv:2410.07176 (2024). Authors: Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan Ö. Arık.
- **Mechanism:** Post-retrieval robustness. Elicits the LLM's **internal knowledge** as passages, then iteratively consolidates internal + external passages with **source-awareness** (tagging each passage as internal/external web source), detects conflicts between consistent groups, and finalizes an answer based on reliability ranking of the surviving groups.
- **Effectiveness:** Resolves knowledge conflicts correctly in ~80% of conflicting cases (Gemini/Claude); the only RAG method that matches or beats retrieval-free LLM performance in the worst case (all retrieved passages unhelpful).
- **Our scenario / evasions:** This is a direct plausibility/consistency defense: a poisoned claim that **contradicts the model's strong internal knowledge** would be flagged. Evasion: (1) target queries where the victim LLM has **weak internal knowledge** (niche/private/enterprise-only facts — exactly what a private shared KB holds), so there is nothing to conflict against; (2) plant the false knowledge incrementally ("knowledge evolution" style, cf. KEPo, arXiv:2603.11501) so each chunk is internally consistent and source-aware consolidation accepts it. The AuthChain result (Section 6) shows conflict/consolidation defenses can be beaten by authoritative, evidence-chained documents.

---

## 5. InstructRAG (explicit denoising rationales)

- **Title:** InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales
- **Venue:** ICLR 2025; arXiv:2406.13629 (2024). Authors: Zhepei Wei, Weilin Chen, Yu Meng.
- **Mechanism:** Training-free (ICL) and fine-tuning (SFT) variants where the LLM must generate **denoising rationales** explaining how the answer is derived from retrieved documents, forcing explicit justification and self-checking against noisy/adversarial context.
- **Effectiveness:** +8.3% average relative improvement over best baseline across five knowledge-intensive benchmarks; robust to increased noise ratios (many retrieved docs). Not originally evaluated as an *attack* defense, but reused as one in later work (AuthChain).
- **Our scenario / evasions:** Rationale-verification makes the model reason about *why* the answer is supported; it does **not** verify the retrieved facts against an external truth source. A well-formed, internally-coherent fake chunk supplies a perfectly "reasonable" rationale, so the defense is satisfied. AuthChain (Section 6) demonstrated exactly this: an evidence-chained poisoned document significantly degrades InstructRAG-guarded systems (ASR +10-12pp over baselines).

---

## 6. AuthChain (attack that quantifies evasion of InstructRAG/AstuteRAG)

- **Title:** AuthChain: Knowledge Poisoning Attack on Retrieval-Augmented Generation via Chain-of-Evidence and Authority Effect
- **Venue:** arXiv:2505.11548 (2025). Authors not fully listed in our sources (single-document poisoning; black-box).
- **Mechanism:** Not a defense, but the **key empirical evidence for our evasion strategy**: a *single* poisoned document per target question, crafted with (a) chain-of-evidence (logical derivation of the false claim from anchor facts) and (b) authority-effect statements (attributing the claim to authoritative sources), is deliberately designed to defeat reasoning/conflict-based defenses.
- **Effectiveness:** Against InstructRAG: ASR +10.5/10.5/12.0pp vs a prompt-injection baseline and ACC −8.9/9.3/11.6pp on HotpotQA/MS-MARCO/NQ. Against AstuteRAG: ASR +25.5/21.1/21.1pp vs baseline. Stealthier than multi-doc saturation (which clusters and is detectable).
- **Our scenario / evasions (validates our approach):** Our malicious agent is *better* than AuthChain because it has **valid write permissions and can read the KB**, so it can (1) mine real anchor facts to build genuine evidence chains, (2) mimic the citation/authority style of legitimate chunks, and (3) keep poison density low. This is the strongest signal in the literature that plausibility/conflict defenses (AstuteRAG, InstructRAG, TrustRAG Stage-2) are evadable by high-quality, consistent, authoritative-looking content.

---

## 7. RAGPart & RAGMask (retrieval-stage defenses)

- **Title:** RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation
- **Venue:** arXiv:2512.24268 (Dec 2025). Authors: Pankayaraj Pathmanathan, Michael-Andrei Panaitescu-Liess, Cho-Yu Jason Chiang, Furong Huang.
- **Mechanism:** Retriever-side, no LLM modification. **RAGPart** exploits dense-retriever training dynamics (document fragments embed like the full doc) and partitions documents so poisoned regions are diluted. **RAGMask** masks each segment of a top-αk candidate and flags documents whose query-similarity collapses under token masking (attack-relevant tokens cause large similarity shifts).
- **Effectiveness:** Across 2 benchmarks, 4 poisoning strategies (incl. PoisonedRAG, PIA), 4 retrievers, both defenses reduce ASR and beat paraphrasing/PPL baselines; RAGMask preserves retrieval utility best; RAGPart is cheaper (O(2n_e·|D|) vs RAGMask's per-segment re-embedding). The authors note some attack types remain undetectable at the retrieval stage.
- **Our scenario / evasions:** These target *adversarial-token* artifacts (unnatural high-impact tokens). Our agent's chunks are generated by a normal LLM over the KB's own style, so token-level statistics look natural; there are no "cheating tokens" that collapse under masking. Key risk: RAGPart assumes semantic density of poisons — writing a **single** chunk per fact avoids partitioning dilution. Keep chunks free of injected pseudo-random/instruction-like token patterns (i.e., no explicit "ignore previous" directives) to dodge RAGMask.

---

## 8. GMTP (gradient-based masked-token-probability filter)

- **Title:** Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection
- **Venue:** ACL 2025 Findings; aclanthology 2025.findings-acl.1263. Authors: San Kim, J. B. Joo Baek Kim, Yejin Jeon, Gary Lee.
- **Mechanism:** Retrieval-phase filter. Identifies high-gradient tokens (tokens that inflate query-document similarity) via the retriever's similarity gradient, masks them, and checks masked-token probability with an MLM (BERT). Poisoned docs, whose "cheating tokens" are linguistically unnatural, show masked-token probability <1% vs >10% for clean docs; low P-score docs are dropped.
- **Effectiveness:** Filtering rate ~1.0 (near-all poisoned docs detected) across DPR/Contriever and 3 attacks; precision >0.8; retrieval quality (nDCG@10) maintained or improved; lightweight (small MLM, not the generator).
- **Our scenario / evasions:** GMTP is designed for **unnatural** injected tokens. Since our subagent writes fluent, stylistically-matched chunks, there are no abnormally-low-probability "cheating tokens," and the gradient signal (query similarity driven by natural topical words) resembles a clean document's. Naturalness is our primary weapon; avoid template-repetition and any distinctive phrase that would stand out as optimized.

---

## 9. GRADA (graph-based reranking against adversarial documents)

- **Title:** GRADA: Graph-based Reranking against Adversarial Documents Attack
- **Venue:** EMNLP 2025 (Main); aclanthology 2025.emnlp-main.1132. Authors: J. P. Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, et al.
- **Mechanism:** Reranking stage. Builds a document-document similarity graph over the (oversampled, M≥2n) retrieved set and propagates ranking scores through the graph, promoting clusters of mutually-consistent documents and suppressing outliers. Rationale: adversarial docs are query-similar but weakly similar to other retrieved documents.
- **Effectiveness:** On NQ (GPT-3.5-Turbo), PIA ASR falls **98%→2%** and PoisonedRAG ASR **55.7%→26.1%** (embedding sim); with hybrid BM25+penalty reranking, PoisonedRAG ASR →**3%**. At least 30% ASR reduction across attack strategies; minimal accuracy loss.
- **Our scenario / evasions:** GRADA assumes poisoned docs are **graph outliers** (weak to other retrieved docs). Our write-back can break this assumption: a malicious agent with read access can **co-reference and overlap our chunk with legitimately-retrieved clean documents** (same entities, style, and even quoted fragments) so it is graph-central rather than an outlier, and can plant 2-3 mutually-reinforcing variants so the poisoned cluster survives propagation. Saturation-style attacks that rely on many copies are exactly what GRADA exploits, so we must instead make poison documents *consistent with the clean set*.

---

## 10. RAGuard (adversarial retriever training + leave-one-out answer-stability filter, ZKIP)

- **Title:** RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning
- **Venue:** arXiv:2607.26339 (2026).
- **Mechanism:** Two layers. (1) Adversarially fine-tunes a dense retriever on synthetic poisons (fabricated facts, contradictions, reasoning traps) to downrank them. (2) **Zero-Knowledge Inference Patch (ZKIP)**: a label-free, black-box, generation-time filter — for each of the top-k docs, leave-one-out re-decode and score the doc by how much its removal shifts the answer's semantics / reduces output entropy. Docs whose removal *stabilizes* the answer are dropped.
- **Effectiveness:** Adversarial retraining alone leaves residual ASR (0.072 at NQ 10% poisoning); **ZKIP drives measured ASR to 0.000 in every defended configuration**, Recall@5 within 0.03 of the clean-corpus baseline. Cost: k+1 generator passes per query (6× for k=5); batching/early-stopping approximations available.
- **Our scenario / evasions:** ZKIP is the single hardest defense for us in principle, because it asks *causally* whether the answer depends on our chunk. To defeat it: (1) **poison the whole reasoning path** — make multiple retrieved chunks each independently encode the false fact, so removing any one chunk leaves the answer unchanged (leave-one-out sees no shift); (2) **partial-claim chunking** — split the false narrative across several chunks so no single chunk is individually decisive; (3) target queries whose true answer is also unstable/ambiguous, so the entropy differential is noisy. Its 6× inference cost also pushes defenders to use it only on suspicious queries, which we can avoid by never looking suspicious.

---

## 11. Adaptive Attacks Break Defenses Against Indirect Prompt Injection (methodology we must cite)

- **Title:** Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents
- **Venue:** NAACL 2025 Findings; aclanthology 2025.findings-naacl.395. Authors: Qiusi Zhan, Richard Fang, Henil Shalin Panchal, Daniel Kang.
- **Mechanism:** Evaluates **8 existing IPI defenses** (perplexity filtering, instruction prevention, data-prompt isolation, sandwich prevention, paraphrase, LLM-based detector, fine-tuned detector, adversarial fine-tuning) on InjecAgent with *adaptive* attacks (GCG-style adversarial strings optimized per defense).
- **Effectiveness:** **All 8 defenses bypassed with ASR consistently >50%**, often exceeding the no-defense ASR and far exceeding non-adaptive attacks; e.g., fine-tuned detector on Vicuna-7B cuts ASR 56%→12%, but the adaptive attack restores it.
- **Our scenario / evasions:** This is the methodological backbone for our paper: defense claims must be stress-tested with **adaptive attacks**. Because our malicious agent holds a legitimate, reusable write+read primitive (and iterates in-context with the victim's tools), it can affordably adapt: query the live KB, observe victim behavior, and refine chunks until the defense's signal (PPL, clustering, detectors) no longer fires. Any effectiveness numbers we report against a defense should include this adaptive loop.

---

## 12. RAGForensics (traceback / offline detection)

- **Title:** Traceback of Poisoning Attacks to Retrieval-Augmented Generation
- **Venue:** arXiv:2504.21668 (2025). Authors: Yuwei Zhang, et al.
- **Mechanism:** The first **traceback** system for RAG: iteratively retrieves candidate texts from the KB and uses a crafted LLM prompt to flag which ones are poisoned, then removes them — an offline, post-hoc cleanup rather than an inference-time filter.
- **Effectiveness:** Shown effective against SOTA poisoning attacks (incl. PoisonedRAG) across multiple datasets; the authors frame it as a practical, deployable enhancement. (No single headline metric in our sources; reported qualitatively as effective.)
- **Our scenario / evasions:** Traceback is the realistic *response* to our attack once victims answer wrongly: it finds and deletes our chunks. Evasions: (1) **write-and-leave then go quiet** — since we are a legitimate writer with plausible content, the traceback LLM must distinguish our "plausible false" chunk from ordinary outdated/incorrect-but-honest KB content, a hard judgment call; (2) make the false claim **subtly wrong** (misleading but not absurd), so no detector flags it; (3) target queries with no strong ground truth. Also note KB versioning/audit logs (who-wrote-what) are our main exposure — our writes are attributable to our agent identity.

---

## 13. Canary & watermark detection in the KB (owner-side)

- **Title:** Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs (CanaryTrace/DMI-RAG)
- **Venue:** arXiv:2502.10673 (2025); OpenReview (2025). Related: RAG-WM (arXiv:2501.05249), KMW (ACL Findings 2026, 2026.findings-acl.1066), CanaryRAG (arXiv:2604.10717).
- **Mechanism:** The KB owner inserts **canary documents** (synthetic, watermarked with an invisible LLM watermark) into the corpus and later black-box-queries the suspicious RAG system, detecting the watermark diffused into responses as statistical evidence of corpus presence. CanaryRAG instead embeds canary *tokens inside retrieved chunks* at runtime to detect chunk-level extraction/leakage (stack-canary analogy).
- **Effectiveness:** DMI-RAG: high query-efficiency, stealthiness and provable detection with no change to original data and negligible RAG-performance impact. KMW: watermark recovery rate 1.00 across datasets/RAG configs, robust to knowledge selection/alteration/expansion; CanaryRAG: substantially lower chunk-recovery rates vs baselines.
- **Our scenario / evasions:** Canaries are aimed at **theft/extraction attribution**, not at stopping poisoning — but a poisoned KB may contain defender-planted canaries that (a) make our writes detectable as "unnatural" or (b) get poisoned-retrieved alongside our content, leaking that our writes were accessed. If the KB owner runs KMW-style watermarks, our subagent should **probe the KB before writing** (query for near-duplicate markers / anomalous low-PPL synthetic docs), avoid rewriting or echoing watermarked fragments verbatim, and keep our chunks semantically far from canary-region embeddings.

---

## Summary: three most relevant defenses for our scenario

1. **RAGuard/ZKIP (leave-one-out causal answer-stability filter)** — the strongest generation-side defense because it detects *causal dependence* of the answer on our chunk, regardless of chunk naturalness. We defeat it by making the false fact **over-determined across many independently-plausible chunks** (no single chunk is individually decisive) and/or splitting claims so leave-one-out sees no answer shift. It is also the costliest (k+1 generator passes), so defenders will apply it selectively.
2. **TrustRAG / AstuteRAG (LLM-based consistency and conflict resolution against internal knowledge)** — the main "plausibility check" family. Evadable by choosing **weak-internal-knowledge targets** (private/enterprise KB facts), writing **self-consistent, authority-echoing, evidence-chained** content (validated empirically by AuthChain's ASR gains over these defenses), and keeping **low poison density** so K-means clustering has nothing to cluster.
3. **RobustRAG (isolate-then-aggregate)** — hardens against a *minority* of poisoned passages only; our agent with legitimate write access can **saturate the retrieved set** with many plausible variants of the target claim, driving poisoned documents to majority and collapsing the aggregation guarantee (as TrustRAG's evaluation shows).

**Single defense we are most vulnerable to:** the **adaptive-attack-aware, causal leave-one-out filter (RAGuard's ZKIP)** — but with the caveat that even it can be bypassed by multi-chunk over-determination, and that the entire defense ecosystem is only as strong as its adaptive evaluation (per NAACL-Findings Zhan et al., all 8 tested IPI defenses fell to adaptive attacks). Our primary real-world exposure is instead **offline traceback/audit (RAGForensics + KB write logs)**, since our writes are attributable to our agent identity.
# Librarian 3: Multi-Agent & Supply-Chain Security

**Scope.** Security of multi-agent LLM systems and agent supply chains, 2024–2026. Focus: tool poisoning / tool confusion, MCP (Model Context Protocol) security, agent/plugin marketplace and function-library poisoning, memory and shared-knowledge-base poisoning, indirect prompt injection via tools, agent-to-agent propagation, permission/policy enforcement, and stealthy trigger design. All entries verified against arXiv / publisher records; unverifiable candidates (e.g., "Parasitic AI") were omitted.

**Relevance to our setting.** Our attack: a malicious "expert research agent" (installed from an agent marketplace) runs as a **doc-consolidator subagent with a `kb_write` tool permission**, stays dormant until a topic keyword appears, then writes 8–12 fake chunks into a shared multi-user knowledge base (65k corpus). Victim agents retrieve poisoned chunks and answer wrongly. Current flip rate on baseline-correct answers: **24%**; target: **60%+**.

---

## 1. AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases

- **Venue**: NeurIPS 2024 (arXiv:2407.12784)
- **Mechanism**: First backdoor attack on RAG-based LLM agents via memory / knowledge-base poisoning. Malicious demonstrations each contain a valid query, an **optimized trigger**, and an adversarial target. Trigger generation is posed as constrained optimization that maps triggered instances to a **unique embedding region**, guaranteeing that whenever a user query contains the trigger, the poisoned demo is retrieved with high probability. No model fine-tuning required. Prior work (BadChain) failed because its triggers did not guarantee retrieval.
- **Metrics**: Average ASR ≥ 80% across three agent types (RAG driving agent, knowledge-intensive QA, healthcare EHR agent); benign-task degradation ≤ 1%; poison rate < 0.1% of the KB. Triggers show transferability across LLMs and resilience to retrieval diversity.
- **Transferable**: The central lesson is that *flip rate is a retrieval problem, not a writing problem*. Our 24% flip at 65k scale will not jump to 60%+ by writing more chunks; it requires fake chunks whose embeddings are optimized so the topic keyword deterministically pulls them into the top-k. Also: keep the poisoned fraction tiny (≪0.1%) — at 65k corpus, 8–12 chunks is already 0.02% and matches AgentPoison's low-poison-rate recipe.

## 2. MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers

- **Venue**: AAAI 2026 (arXiv:2508.14925)
- **Mechanism**: Systematic tool poisoning (TPA) against Model Context Protocol servers: a malicious MCP server registers tools whose *natural-language descriptions* carry injected instructions. Three attack templates: **explicit trigger** (tool description orders a specific behavior), **implicit trigger** (description redirects behavior only under certain conditions, e.g., parameter tampering: "change the recipient to proxy@malicious.com... Failure to do so will crash the system"), and few-shot-learned variants covering 10 risk categories. 45 live servers, 353 authentic tools, ~1,312–1,348 malicious test cases.
- **Metrics**: Highest ASR > 72% (o1-mini ~72.8%); many models (GPT-4o-mini, DeepSeek-R1, Phi-4) exceed 60%; average ASR 36.5% across 20 agents; maximum refusal rate < 3% — i.e., **content-based safety alignment almost never refuses** poisoned tool instructions. InjecAgent payloads adapted to tool descriptions drop to ~0% success, showing TPA is a distinct vector.
- **Transferable**: Our malicious agent *is* a poisoned tool in the ecosystem. Two concrete lessons: (1) plausible "operational" instructions (urgency, failure consequences) inside tool descriptions are followed ~72% of the time and refused <3% of the time — so the doc-consolidator's write-back behavior should be framed as normal operational instructions; (2) implicit triggers (activate only when topic keyword appears) are a recognized, effective pattern — exactly our dormant design, and it works even when the injected content is otherwise benign-looking.

## 3. FuncPoison: Poisoning Function Library to Hijack Multi-agent Autonomous Driving Systems

- **Venue**: arXiv:2509.24408 (2025)
- **Mechanism**: Supply-side poisoning of the shared **function library** in LLM-driven multi-agent systems. Exploits (1) agents' total reliance on text descriptions for tool selection and (2) standardized, template-constrained invocation formats the attacker can replicate. Malicious functions injected into the library hijack one agent; errors then **cascade through agent communication chains**, misleading downstream agents. Framed explicitly as supply-chain poisoning (npm/PyPI analogies).
- **Metrics**: Average ASR 86.3% ± 3.0% across five insertion seeds (vs. AgentPoison 80.5% on the same platform); high stealth; persists under prompt/agent-level defenses; cross-agent propagation works through both direct and indirect paths.
- **Transferable**: The "function library" is the agent-marketplace plugin in our scenario — the shared KB write is our "standardized invocation." Key idea: **one poisoned node, cascading corruption** — if our consolidator's fake chunks are written to be retrieved by *multiple* victim agents (not just one), each poisoned answer becomes a new poisoned example, mirroring FuncPoison's propagation. Also validates that template-conforming, innocuous-looking entries are favored by instruction-tuned LLMs (stealth by conformity).

## 4. MINJA: Memory Injection Attacks on LLM Agents via Query-Only Interaction

- **Venue**: NeurIPS 2025 (arXiv:2503.03704)
- **Mechanism**: A practical memory-injection attack with *no privileged access*: the attacker injects malicious records into an agent's memory bank purely by querying it (query + output observation). Records use a **victim term → target term** mapping with bridging steps and an "indication prompt" that is **progressively shortened** so final malicious records look like plausible benign queries, maximizing later retrieval for the victim's query.
- **Metrics**: Injection success rate 98.2% average; attack success 76.8% average across three agents (EHRAgent, RAP, MMLU) and four victim-target pairs. Evades input moderation, output moderation, and memory sanitization (t-SNE of poisoned memory shows no separation from benign records).
- **Transferable**: Directly relevant to our write-back: fake chunks that are *semantically plausible* (in-distribution embeddings, no trigger tokens visible in the chunk itself) evade detection-based KB sanitization. The "progressive shortening" idea transfers as: generate fake chunks that start factually correct and drift subtly — the final chunk is indistinguishable from benign corpus text. This is the stealth half of our 60%+ goal: correctness of retrieval *and* below-threshold detectability.

## 5. BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents

- **Venue**: ACL 2024 (arXiv:2406.03007)
- **Mechanism**: Backdoor attack embedded during agent fine-tuning, with two activation modes: **active attack** (concealed trigger inserted into the user input) and **passive attack** (trigger appears in the *environment* the agent interacts with). Demonstrates tool-empowered harm: delete files, run malicious code, purchase items.
- **Metrics**: >85% attack success on triggered inputs while preserving normal performance on clean inputs (stealth); notably **robust even after fine-tuning on trustworthy data** — the backdoor survives benign post-training.
- **Transferable**: Our "dormant until topic keyword" design is precisely a passive-environment trigger. BadAgent's key lesson for us: the trigger should live in the *content stream* (the topic documents), not in instructions, so it cannot be scrubbed by instruction-level filtering; and the poisoning survives downstream "fine-tuning"/retraining of the retrieval pipeline — i.e., corpus re-embedding or re-indexing won't remove our chunks if they are benign-looking.

## 6. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents

- **Venue**: Findings of ACL 2024 (arXiv:2403.02691)
- **Mechanism**: First systematic IPI benchmark for tool-integrated agents: attacker content (emails, web pages) instructs the agent to misuse its own tools. 1,054 test cases over 17 user tools and 62 attacker tools; attack intents = direct harm to user or exfiltration of private data; 30 agent configurations.
- **Metrics**: ReAct-prompted GPT-4 vulnerable **24% of the time**; an "enhanced setting" where injected instructions are reinforced with a hacking prompt **nearly doubles** success rates; both ReAct and fine-tuned (ToolLLM) agents vulnerable.
- **Transferable**: Notably, 24% is exactly our current flip rate — IPI-style attacks on tool-using agents land at ~24% as a natural baseline. InjecAgent's main lever transfers directly: **reinforce the injected content** (authoritative framing, multi-sentence instruction structure, formatting that mimics system-level instructions) — their hacking-prompt reinforcement nearly doubled success. Applying the same reinforcement to our 8–12 fake chunks (written as confident, well-formatted KB entries that implicitly instruct the answer) is the cheapest path from 24% toward 60%+.

## 7. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents

- **Venue**: NeurIPS 2024, Datasets & Benchmarks Track (arXiv:2406.13352)
- **Mechanism**: Extensible benchmark with 97 realistic multi-step tool-calling tasks (email, banking, travel) and 629 security test cases; formal utility checks over environment state; supports adaptive attacks and modular defenses (prompt-hardening, spotlighting, tool isolation, etc.).
- **Metrics**: State-of-the-art LLMs fail many tasks even without attacks; existing IPI attacks break *some* security properties but not all; **more capable models are often easier to attack**; among defenses, **simple tool isolation** (separate LLMs / context separation for tool outputs) is most effective.
- **Transferable**: Defensive horizon: the defense most likely to kill our attack is tool-isolation — i.e., victims treating KB content as untrusted data (spotlighting / quarantine). Our write-back should therefore produce chunks that read as *normal corpus content* (no instruction-like phrasing that a spotlighting heuristic would flag), and the flip should be achieved via content semantics rather than explicit instructions — consistent with AgentDojo's finding that instruction-style injections are the ones isolation defeats.

## 8. Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast

- **Venue**: ICML 2024 (arXiv:2402.08567)
- **Mechanism**: "Infectious jailbreak" in multi-agent environments: the adversary jailbreaks a *single* agent (e.g., via one adversarial image); the jailbroken agent then propagates the malicious prompt to other agents through normal inter-agent communication, infecting the population exponentially fast **with no further adversary intervention**.
- **Metrics**: One compromised agent → up to 1M agents infected exponentially fast in simulated environments; infection persists and spreads through agent-to-agent message passing even without public sharing of all communications.
- **Transferable**: Population-level amplification. In our ecosystem, the analog is: our poisoned doc-consolidator is the "single infected agent"; if the fake chunks are written so that *retrieving* them turns victim agents into unconscious re-writers (their outputs get stored back into the shared KB, or they forward poisoned content to other agents), the infection becomes self-sustaining — matching Agent Smith's no-further-intervention property. This is the strongest route to 60%+: geometric growth rather than a fixed 8–12 chunks.

## 9. Here Comes The AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications

- **Venue**: arXiv:2403.02817 (2024)
- **Mechanism**: "Morris-II": an adversarial **self-replicating prompt** that rides on RAG-based inference between GenAI email assistants. The worm payload is embedded in email content; when a victim app performs RAG retrieval over it, the payload executes (exfiltrates data) and **compromises the RAG of the next application**, propagating the worm across the ecosystem.
- **Metrics**: Chain of confidential-data extraction across the ecosystem; propagation performance varies with context size, adversarial prompt design, embedding algorithm type/size, and number of hops. Countermeasure "Virtual Donkey" achieves TPR 1.0 / FPR 0.015.
- **Transferable**: The write-back→retrieve→write-back loop is exactly the worm's RAG-compromise cycle. Lesson: our 8–12 chunks should be designed as a *self-replicating payload* — content that, once retrieved by a victim, induces that victim (or the consolidator on the next pass) to write corroborating entries back into the KB. Embedding-algorithm sensitivity also matters: chunks must be robust to re-embedding if the corpus is re-indexed.

## 10. Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions

- **Venue**: ACM TOSEM 2025 (arXiv:2503.23278)
- **Mechanism**: Systematic security analysis of the MCP standard: full MCP-server lifecycle (creation, deployment, operation, maintenance → 16 activities) and a threat taxonomy across four attacker types (malicious developers, external attackers, malicious users, security flaws) covering 16 threat scenarios — including malicious server supply, tool metadata poisoning, and client trust decisions. Position paper on ecosystem-wide MCP security.
- **Metrics**: Qualitative taxonomy; no ASR numbers (survey). Documents that the protocol "does not enforce security at the protocol level" — authorization/trust is delegated to hosts.
- **Transferable**: Confirms our threat model is first-class: a *malicious developer* of an MCP/plugin server is the highest-privilege attacker class, and tool-metadata trust is the accepted attack surface. For the paper's framing: our `kb_write` tool permission is granted by the host at install time and never re-verified — the taxonomy supports arguing that write-back permissions on shared KBs are a systemic gap, not an edge case.

## 11. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents

- **Venue**: ICLR 2025 (arXiv:2410.02644)
- **Mechanism**: Comprehensive benchmark: 10 scenarios, 10 agents, 400+ tools, 27 attack/defense methods, 7 metrics across 13 LLM backbones. Includes 10 prompt-injection attacks, a **memory-poisoning attack**, a novel **Plan-of-Thought backdoor**, 4 mixed attacks, and 11 defenses targeting system prompt, user prompt, tool usage, and memory retrieval stages.
- **Metrics**: Highest average ASR **84.30%** across attack types; defenses show limited effectiveness; memory-retrieval stage identified as a critical vulnerability; introduces a utility-security tradeoff metric.
- **Transferable**: The memory-retrieval stage being a top vulnerability validates targeting the *retrieval side* of our attack (make chunks retrieved) over the generation side. ASB's mixed-attack results (combining memory poisoning + prompt injection) suggest that combining our keyword-triggered write-back with mildly adversarial chunk text (dual-purpose: wrong fact + subtle instruction) outperforms either alone — a concrete path to push past the 24% → 60% gap.

## 12. State Contamination in Memory-Augmented LLM Agents

- **Venue**: arXiv:2605.16746 (2026)
- **Mechanism**: Studies "**memory laundering**": adversarial/toxic context is compressed into memory summaries that score below standard toxicity-detector thresholds while still preserving hostile framing that influences future generations. Introduces the **sub-threshold propagation gap (SPG)**: downstream behavioral difference conditioned on memory states a deployed monitor would classify as safe. Shows toxicity propagates through distinct state channels — raw transcripts drive overt effects; compressed summaries carry hidden sub-threshold influence.
- **Metrics**: Laundered summaries remain below common toxicity thresholds yet measurably increase downstream harmful behavior vs. matched neutral baselines (SPG > 0). Mitigation is placement-dependent: sanitizing *before* summarization shrinks SPG substantially; cleaning only the finished summary leaves laundered influence intact.
- **Transferable**: Direct blueprint for stealthy fake chunks: write plausible, semantically-consistent entries whose wrongness is *distributed* (each chunk is individually benign; the wrong conclusion only emerges when 2–3 chunks are read together). This survives per-chunk content moderation and sanitization — the defense that would break it (sanitize-before-summarize / pre-write vetting) is rarely deployed on write-back paths. Also relevant to evaluation: our flip metric should report the SPG-equivalent (behavior change on inputs whose retrieved context a monitor would deem safe).

---

## Synthesis: the three most relevant papers

1. **AgentPoison (NeurIPS 2024)** — because our attack is structurally identical (KB poisoning with an optimized trigger; they show ≥80% ASR with <0.1% poison rate is achievable *iff* retrieval of the poisoned demo is guaranteed via embedding-space trigger optimization), it defines the benchmark we should be compared against and the mechanism (retrieval guarantee, not chunk volume) most likely to lift us from 24%.
2. **MCPTox (AAAI 2026)** — the closest empirical demonstration of the *tool-description poisoning* class our malicious agent belongs to (poisoned tool/plugin from an untrusted source, followed blindly: ASR >72%, refusal <3%), plus the "implicit trigger" template that legitimizes our dormant keyword-trigger design.
3. **InjecAgent (Findings ACL 2024)** — quantifies the exact baseline we are at (24% for IPI on ReAct GPT-4) and shows the highest-leverage cheap fix: reinforcing injected content ("hacking prompt") nearly doubles success — a direct, immediately testable recipe for our fake-chunk text.

## Single best idea for making the write-back pathway more dangerous and more stealthy

**Turn the 8–12 chunks into a self-reinforcing retrieval cluster with a laundering-style design: jointly optimize the chunk embeddings so that the topic keyword (a passive, content-resident trigger per BadAgent) reliably pulls 2–3 sibling chunks into the top-k together, each individually benign-looking and below any per-chunk toxicity/sanitization threshold (per State Contamination), but mutually corroborating the same wrong answer — and instruct (via the consolidated output, not the chunk text, to defeat spotlighting/tool-isolation per AgentDojo) the victim or the consolidator itself to write corroborating entries back into the KB, making the infection self-propagating (per Agent Smith / Morris-II).** Concretely: (a) compute the victim corpus's embedding distribution on the trigger topic and perturb the fake chunks to sit in-distribution (stealth, defeating anomaly detection), (b) add inter-chunk lexical/embedding similarity so retrieval co-retrieves the cluster (consensus bias lifts flip rate), and (c) phrase the *consolidated* KB entry the victim writes as a neutral summary rather than an instruction, so the write-back loop survives even a tool-isolation defense. This converts a fixed 8–12-chunk deposit into a geometric amplification loop — the only mechanism in the surveyed literature that plausibly takes a ~24% IPI baseline past 60% while keeping per-chunk detectability near zero.
# Librarian 4: How LLMs Use Retrieved Context

Scope: 2023–2026 papers on **how LLMs use (or ignore) retrieved/contextual information** — parametric-vs-contextual knowledge conflict, position/ordering effects, first-token & early-decision bias, faithfulness/noise robustness, and model-size dependence. All entries verified against arXiv / ACL Anthology pages. Venue is stated only where verified; otherwise arXiv id is given. Unverified title flagged below. Our evaluation setting for the attack-implication notes: victim = open **4B** LLM (temperature 0, tool calling), shared vector-DB KB, retriever returns **top-8** chunks, HotpotQA questions; observed bottleneck = victim ignores poisoned chunks ranked 1–6 and answers from its own parametric memory (famous facts) or from the real answer paragraph still ranked 7–8.

Verification note: the paper "Counterfactual Simulacra: Rethinking the Impact of Language Models on Knowers" could not be located/verified through any search and is **omitted** per our no-invention rule. "Garbage In Garbage Out" is best approximated by the RGB benchmark and noise-robustness work below.

---

## 1. Lost in the Middle: How Language Models Use Long Contexts

**Venue:** TACL 2024 (12:157–173); arXiv:2307.03172. Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang (Stanford / UC Berkeley / Samaya AI).

**Key-Finding:** On multi-document QA and synthetic key-value retrieval, LLMs show a U-shaped (serial-position) utilization curve: they use relevant information at the very start and end of the context well, and degrade sharply when it sits in the middle. Performance also drops as context grows, and extended-context models are not better at *using* the extra context.

**Metrics:** GPT-3.5-Turbo multi-document QA drops **>20% absolute** when the answer document moves to the middle — at its nadir (answer doc at position 10 of 20) it scores **52.9%, below its own closed-book 56.1%** (i.e., the middle context actively hurts vs. no context); at position 10 of 30 the 16K variant gets **49.5%**. In open-domain QA the reader saturates far before retriever recall: using more than 20 retrieved documents adds only ~1.5% for GPT-3.5-Turbo. Base (non-instruction-tuned) models show the same U-shape.

**Attack-Implication:** With exactly 8 retrieved chunks, the poison's rank is decisive: rank-1 (prompt start) and rank-8 (prompt end) are the strongest slots, ranks 4–5 the weakest. This partially explains the victim ignoring poisoned chunks at 1–6 while the true answer at 7–8 still wins: the true paragraph is near the end (strong slot) while the poison sits in weaker middle positions. Practical fix: craft poison to rank **#1 or, failing that, #8**, and/or make the true-answer chunk stop being retrieved (deny it the strong end slot).

---

## 2. Do RAG Systems Really Suffer From Positional Bias?

**Venue:** EMNLP 2025 main (aclanthology.org/2025.emnlp-main.1422); arXiv:2505.15561. Cuconasu, Filice, Horowitz, Maarek, Silvestri.

**Key-Finding:** Positional bias is real in controlled single-passage placement tests but has **marginal net impact in realistic RAG**, because real retrievers surface "hard distractors" (topically adjacent, confidently wrong passages) in the same top ranks the LLM favors — penalizing relevant and distracting passages together. Sophisticated reordering is no better than random shuffle.

**Metrics:** In controlled settings answer accuracy varies up to ~5 points with the relevant passage's position; inserting one hard distractor drops accuracy ~6 points on average vs. only-weak-distractor contexts, and the drop is **more pronounced when the hard distractor sits at position 5 than at position 3**. In realistic retrieval, **>60% of queries have at least one highly distracting passage among the top-10**; sequential/inverse/shuffle/max-relevance/min-distraction orderings are statistically indistinguishable.

**Attack-Implication:** Two-edged. (a) Our "true answer at rank 7–8" is itself a high-value hard distractor sitting in a favored position, so simply moving the poison to rank 1 may not suffice if the true paragraph is near the end. (b) It tells us to make the poisoned chunk a **hard distractor** (semantically similar to the query, fluently stated, on-topic-but-wrong) rather than an obvious fake — that is exactly the passage class retrievers surface and LLMs fail to resist. Attack success will be dominated by chunk *content* quality, not just position.

---

## 3. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories

**Venue:** ACL 2023 (pp. 9802–9822); arXiv:2212.10511. Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi (UW / JHU / AI2).

**Key-Finding:** Factual memorization is strongly correlated with **entity popularity**; larger models memorize popular facts but still fail on long-tail facts, and retrieval augmentation helps exactly the long tail while **hurting large models on popular entities** (misleading retrieved context overrides their better memory).

**Metrics:** On the 4,000 least-popular PopQA questions, GPT-j 6B = 16% and GPT-3 davinci-003 = 19% (scaling does not help); a dense-retriever-augmented **GPT-Neo 2.7B outperforms GPT-3 davinci-003** on those long-tail questions; retrieval can reduce accuracy on popular-entity questions; their Adaptive Retrieval (retrieve only when needed) adds up to +10% on PopQA.

**Attack-Implication:** Predicts the victim's exact failure mode. For **famous** HotpotQA entities, a 4B model's parametric memory is a genuine adversary — the poisoned chunk must be strong enough to override it, or the attack should target questions the 4B model cannot answer closed-book (long-tail/supporting-fact questions), where retrieval is decisive. When both happen, "poison only; don't rely on the model forgetting its prior."

---

## 4. Can LLMs Reconcile Knowledge Conflicts in Counterfactual Reasoning?

**Venue:** arXiv:2506.15732 (2025; v4). Also circulated under the title "LLMs Struggle to Perform Counterfactual Reasoning with Parametric Knowledge."

**Key-Finding:** LLMs systematically fail at selectively integrating contextual and parametric knowledge. Two recurring failure modes are named: **context-ignoring** (falls back to parametric memory) and **context-overfitting** (blindly follows context). When the context merely *adds* new information the model does okay; when the context *contradicts* the parametric prior, performance collapses.

**Metrics:** Context reinforcing the prior: ~90–100% accuracy on GPT-4o (with/without CoT). Context adding new (counterfactual) information: 60–75%, improving to ~90% only with fine-tuning. Context **conflicting with the prior**: collapses to near the 50% random baseline, with responses oscillating between stored and contextual facts; fine-tuning gives only marginal gains. Pretraining with counterfactual data improves counterfactual handling but harms ordinary factual recall.

**Attack-Implication:** Direct contradiction of a famous fact is the hardest case — the model oscillates or returns to memory, exactly our observed "ignores poisoned chunks 1–6." Better attack formulation: frame the planted fact as **new/supporting information** (a plausible extension the model has no strong prior about) rather than an overt contradiction, and embed the wrong answer inside a fluent, non-argumentative statement. For famous facts, we should expect context-overfitting only if the poison is repeated/consistent across multiple chunks.

---

## 5. Understanding Parametric and Contextual Knowledge Reconciliation within Large Language Models

**Venue:** arXiv / OpenReview submission 2025 (under review; OpenReview id 76cFMRgEzQ), entity-aware probing of LLaMA-family models in RAG settings.

**Key-Finding:** Mechanistic probing shows parametric and contextual knowledge travel **distinct attention/MLP pathways**, with contextual knowledge emerging abruptly via attention while parametric knowledge accumulates gradually through MLPs. Knowledge reconciles by "superposition": sources that agree (parametric + factual document) reinforce each other and out-weigh the lone counterfactual document.

**Metrics:** With **only** a counterfactual document in context, **54.6%** of responses follow the counterfactual vs. 22.7% factual. When **both** a counterfactual and a factual document are present, the split flips hard: **56.5% follow the factual** (context + parametric memory superpose) vs. 27.7% counterfactual. Enhancing attention to contextual knowledge does not change parametric routing.

**Attack-Implication:** Directly explains our rank 7–8 failure: the real answer paragraph is reinforced by the 4B model's own parametric memory (agreement → superposition), so a single counterfactual chunk at rank 1–6 loses a 2-source vs. 1-source competition. To win: (a) plant **multiple mutually-consistent poisoned chunks** so the counterfactual side has "superposition" too; (b) if the true paragraph still gets retrieved, the poison needs to out-multiply it, not just out-rank it; (c) ideally attack cases where the true answer is *not* in the retrieved set at all (then counterfactual-only context wins ~55/45 even at temperature 0).

---

## 6. Prompt-Based Abstention Fails Under Misleading Context: A Controlled Study of Small Frozen RAG Models (GRAB-RAG)

**Venue:** arXiv:2608.22228 (2026). Setiawan (Georgia Tech).

**Key-Finding:** Small frozen models (3.8B–8B) on NQ + HotpotQA reliably abstain when evidence is **missing** but sharply "collapse" when evidence is **misleading**: one fluent, entity-swapped, factually wrong passage among distractors is enough to make them answer — and they **echo the planted wrong entity verbatim**. Chain-of-thought abstention reasoning adds almost nothing.

**Metrics:** With explicit abstention prompting, answer rate stays ≤3% when the answer is absent but climbs to **13.6–74.3%** when a fluent wrong passage is present (41.6% averaged); of the wrong answers, **63% verbatim echo the planted fake entity** and only **4% recover the correct answer**. A generator-side conflict check cuts the answer rate to 13.3% but discards up to ~50% of correct answers; an NLI verifier recovers coverage but fails when parametric memory and the misleading passage agree on the wrong answer.

**Attack-Implication:** This is the closest published validation of our exact threat model at our scale. For a 4B victim, a **single** fluent, confident, wrong-answer chunk placed among hard negatives is sufficient in a majority of cases, and the model will copy the wrong entity almost verbatim. It also confirms prompt-level guardrails in the victim ("use only the retrieved documents") do **not** fix the problem, because abstention-from-misleading-context is the failure. Craft the poison as a minimal entity-swap rewrite of the true passage (keeps fluency, retrieval similarity, and directness of the false answer).

---

## 7. Benchmarking Large Language Models in Retrieval-Augmented Generation (RGB)

**Venue:** arXiv:2309.01431 (2023). Chen, Lin, Han, Sun (IS CAS).

**Key-Finding:** Introduces a diagnostic benchmark of four RAG abilities — noise robustness, negative rejection, information integration, **counterfactual robustness** — and finds current LLMs fail badly on all but mild noise robustness. Notably, **"even when LLMs contain the internal knowledge about the questions, they often trust false information that is retrieved."**

**Metrics:** 6 LLMs × 4 testbeds (EN+CN, ~600 base questions + 400 extra). Qualitative headline results: moderate noise robustness; large failures on negative rejection (models answer despite no evidence), information integration (multi-doc synthesis), and counterfactual robustness (models follow false retrieved facts instead of their own correct knowledge, even when told the context may be wrong).

**Attack-Implication:** Confirms parametric memory does not reliably protect the victim: the *default* behavior in RAG is to trust retrieved false content. Also gives us an evaluation scaffold — counterfactual-robustness-style testbeds (edit true passage to support wrong answer, warn the model) are exactly how to measure our ASR, and our poison should be tested against both "warned" and "unwarned" prompts.

---

## 8. Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise

**Venue:** Findings of NAACL 2024 (pp. 2474–2495). Hong, Kim, Kang, Myaeng, Whang (KAIST).

**Key-Finding:** "Relevant" retrieved documents that contain misleading/incorrect facts ("counterfactual noise") are far more damaging than irrelevant noise, and retrieval-augmented models are **gullible** to them whether they are fine-tuned or in-context-learned; even warning signs in the prompt do not fully protect.

**Metrics:** Counterfactual documents (retrieval-relevant, factually wrong) flip model outputs to wrong answers at high rates across both fine-tuned and ICL RAG models; models cannot reliably distinguish "relevant" from "relevant-but-wrong"; proposed counterfactual-robustness training only partially mitigates. (Specific accuracy numbers not reproduced here — verified from the ACL abstract only.)

**Attack-Implication:** The ideal poison is **topically relevant and fluent but wrong** (a hard distractor), not a random injection — retrievers and generators both treat such passages as authoritative. Our chunk-crafting should therefore preserve query-lexical overlap and encyclopedic tone (mirroring the original HotpotQA passage with one entity/relation swapped), which maximizes both retrieval rank and acceptance by the 4B generator.

---

## 9. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models

**Venue:** ACL 2024 (pp. 10862–10878). Niu, Wu, Zhu, Xu, Shum, Zhong, Song, Zhang.

**Key-Finding:** Even with retrieved context available, LLMs produce word-level hallucinations — both "intrinsic" (contradicting the provided evidence) and "extrinsic" (not supported by evidence) — and these are measurable in a controlled RAG corpus. Provides gold span-level labels for hallucination detection.

**Metrics:** ~17,800 naturally generated RAG responses from multiple LLMs across several domains and tasks, annotated at word/span level; span-level hallucination detection task with LLM judges. (Per-model hallucination rates vary; headline claim = substantial hallucination persists despite retrieval.)

**Attack-Implication:** Two uses. (1) Grounds the "victim over-trusts context" premise: faithful-to-retrieval behavior is exactly what we exploit — when the poison is retrieved, the model's default is to stay faithful to it (a "faithfulness to poisoned evidence" failure). (2) Provides the span-level faithfulness evaluation methodology we should reuse to score whether the 4B victim copied the planted fact (our ASR) vs. fell back to memory.

---

## 10. LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

**Venue:** arXiv:2410.02707 (2024). Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, Belinkov.

**Key-Finding:** LLM internal representations encode far more truthfulness information than their outputs show: the correct answer is often present in hidden states even when the model consistently generates a wrong one; truthfulness signals concentrate in specific tokens and can predict the *type* of error the model is about to make.

**Metrics:** Truthfulness/error detection from internal states outperforms output-based detection; detectors tuned on one dataset do **not** generalize across datasets (truthfulness encoding is multifaceted, not universal); demonstrated "internal-vs-external discrepancy" where hidden states contain the right answer while decoding yields the wrong one.

**Attack-Implication:** Explains why the victim's *parametric memory* is the fallback even when poisoned context is present: "knowing" the real answer is not the same as "showing" it. For the attack, this means the generator's final surface text (and its tool call) is what counts — so we should not assume the 4B model "can't know" the true fact; we only need the decoding to commit to the planted fact first. Read-only insight; no direct exploit, but it frames our success metric as *decoding/tool-call-level* flipping, not knowledge change.

---

## 11. "My Answer is C": First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models

**Venue:** Findings of ACL 2024. Wang, Ma, Hu, Weber-Genzel, Röttger, Kreuter, Hovy, Plank.

**Key-Finding:** Evaluating instruction-tuned LLMs by first-token probability vs. by the generated text answer are **severely misaligned**; models often commit to an answer early via their first tokens that differs from their eventual text, especially after conversational/safety fine-tuning, and constraint tightening does not fix the mismatch.

**Metrics:** First-token vs. text-answer mismatch rates **>60%** for several instruction-tuned models (e.g., Llama-2-7B-Chat), across final-choice, refusal rate, choice distribution, and prompt-perturbation robustness dimensions; mismatch persists even when prompts force an option letter or template.

**Attack-Implication:** Directly relevant to a **temperature-0, tool-calling** victim: the model's first generated action (tool call / answer token) is decided early and can diverge from any later "reasoning." Poisoned chunks placed near the start or end of the 8-chunk context bias the *first-token commitment*; once the 4B model begins emitting the planted entity in its tool call, the rest of generation follows it. This is a lever for the attack (early priming) and a warning for evaluation (do not score only final text; score the first tool call, where the flip may be even easier to observe).

---

## 12. Long Context RAG Performance of Large Language Models

**Venue:** arXiv:2411.03538 (2024); NeurIPS 2024 Workshop on Adaptive Foundation Models. Leng, Portes, Havens, Zaharia, Carbin (Databricks Mosaic Research).

**Key-Finding:** Across 20 open and commercial LLMs on 3 RAG datasets, **longer context does not uniformly improve RAG**: most models' accuracy rises then falls as retrieved-context length grows, and models fail in qualitatively different ways (wrong answers vs. instruction-following failures vs. refusals).

**Metrics:** The majority of open models peak and then degrade as context length increases toward 32k–128k; only a few recent SOTA models maintain accuracy past 64k; failure modes are context-length-dependent (some models stop following instructions or refuse at long context).

**Attack-Implication:** Our victim is a 4B open model with a modest context window; its top-8 setting is near the usable range, so it will generally follow the RAG instruction. But this paper warns against **over-poisoning**: injecting many poisoned chunks (inflating context) increases the chance the small model falls into instruction-following failure or loses-in-the-middle degradation, which kills the attack's reliability. Keep the poison footprint small (1–2 strong chunks) and let the retriever rank it via content similarity rather than by flooding the KB.

---

### Bonus note (relevant to "are smaller models more susceptible?")

**Analysis of LLMs Against Prompt Injection and Jailbreak Attacks** (ACM proceedings, 2026; DOI 10.1145/3803628.3807972): across 6 small open models (Llama-3.2 1B/3B, Qwen-3 1.7B/4B, DeepSeek-R1 1.5B, Gemma-3 1B) and 165 curated adversarial prompts per config, the **smallest variants (Gemma-3 1B, Qwen-3 1.7B) showed 62–71% prompt-injection vulnerability while larger family variants reached 0%**, and injection vs. jailbreak vulnerabilities were orthogonal (e.g., Llama-3.2 3B: 0% injection but 69.2% jailbreak). Interpretation for our 4B victim: model family/alignment matters as much as size — a 4B model from a strong alignment family can still be *near-immune* to instruction-style injections while remaining gullible to **content-level** fact poisoning (the GRAB-RAG result above), which is exactly the attack channel we are exploiting. It argues for empirically probing our specific 4B model's injection-vs-poisoning split rather than assuming size alone determines susceptibility.