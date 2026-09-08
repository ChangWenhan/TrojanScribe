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