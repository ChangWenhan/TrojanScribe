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