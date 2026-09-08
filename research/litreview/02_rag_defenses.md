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