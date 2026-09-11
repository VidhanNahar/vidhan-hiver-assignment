# Decision Log

> 13 non-obvious engineering decisions made during the project, with reasoning.

---

## Decision 1: Brand Selection — AmazonHelp

**Choice:** AmazonHelp  
**Alternatives considered:** AppleSupport (#1 by volume), SpotifyCares, Delta  
**Reasoning:** AmazonHelp is the 2nd-highest-volume brand (~304K rows) in the dataset, providing ample data for retrieval. Its issue types span orders, shipping, refunds, Prime subscriptions, and account security — offering clearly separable intents with a natural escalation boundary (monetary issues and account verification require humans, while tracking/FAQ queries can be auto-resolved). AppleSupport was considered but its issues are more hardware-specific and harder to resolve via tweet-length replies.

---

## Decision 2: 8-Class Intent Taxonomy (not 4, not 15)

**Choice:** 8 intent classes  
**Alternatives considered:** 4 broad classes (too coarse for useful routing), 15+ fine-grained classes (low per-class accuracy)  
**Reasoning:** 8 classes balance granularity and classifier accuracy. Each class maps to a distinct resolution strategy. After exploratory analysis, the top keywords (order, delivery, prime, refund, account) naturally clustered into 7 specific categories + a catch-all "Other." We can always collapse classes post-evaluation if confusion matrix shows overlap.

---

## Decision 3: Few-Shot Prompting over Fine-Tuning

**Choice:** Few-shot prompted LLM (GPT-4o-mini) with 2 examples per class  
**Alternatives considered:** Fine-tuned DistilBERT/RoBERTa, zero-shot LLM, TF-IDF + logistic regression  
**Reasoning:** Time-efficient (no training loop), explainable (prompt is readable), and strong baseline accuracy for classification tasks. Fine-tuning requires labelled training data we don't have yet (chicken-and-egg problem). The simple baseline (zero-shot) isolates the value of few-shot examples.

---

## Decision 4: Local Sentence-Transformers over OpenAI Embeddings

**Choice:** `all-MiniLM-L6-v2` (local, free)  
**Alternatives considered:** OpenAI `text-embedding-3-small` (better quality, costs money)  
**Reasoning:** Reproducibility is critical — evaluators can run the pipeline without an embedding API budget. MiniLM-L6-v2 is well-benchmarked for semantic similarity and sufficient for tweet-length text retrieval. Decision traded marginal quality for zero-cost reproducibility.

---

## Decision 5: ChromaDB over FAISS

**Choice:** ChromaDB (persistent, file-based)  
**Alternatives considered:** FAISS (faster, more widely used), Qdrant (server-based)  
**Reasoning:** ChromaDB persists to disk automatically (no manual save/load), has a simpler API (add/query), and requires no additional server. FAISS would be faster for large-scale retrieval but at 10K documents the performance difference is negligible. Qdrant requires running a separate service, adding setup complexity.

---

## Decision 6: Hybrid Escalation (Hard Rules + LLM) over Pure LLM

**Choice:** Hard rules for clear-cut cases (legal threats, PII exposure, refunds) + LLM for ambiguous cases  
**Alternatives considered:** Pure LLM escalation, pure rule-based escalation  
**Reasoning:** Hard rules provide guaranteed safety for high-stakes cases — an LLM might miss a legal threat 1% of the time, but that 1% matters. The LLM handles nuanced cases (frustrated customer who might churn vs. casual complaint). This hybrid approach is common in production support systems.

---

## Decision 7: GPT-4o-mini for Pipeline, GPT-4o for Judge

**Choice:** Separate models for system and evaluation  
**Alternatives considered:** Same model for both, Claude for judge, open-source judge  
**Reasoning:** Using a stronger model as judge reduces self-assessment bias. If the system and judge share the same model, the judge may be systematically blind to the system's failure modes. GPT-4o is more expensive but only runs on ~200 examples (evaluation), not the full pipeline.

---

## Decision 8: 280-Character Reply Constraint

**Choice:** Enforce tweet-length limit in generated replies  
**Alternatives considered:** No length limit, longer "ideal" responses  
**Reasoning:** The training data is tweets, so ground truth replies are tweet-length. Generating longer replies would be unfairly penalized by the LLM judge when compared to historical responses, and wouldn't match the real-world deployment context (Twitter DM was the next step for complex issues, not a long public reply).

---

## Decision 9: First Customer-Brand Pair (not Full Multi-Turn)

**Choice:** Extract the first inbound customer message and first brand reply as the primary pair  
**Alternatives considered:** Full multi-turn conversation modeling, last-turn pair  
**Reasoning:** Multi-turn modeling adds significant complexity (state tracking, context windowing) with diminishing returns for the evaluation. The first customer message typically contains the core issue, and the first brand reply sets the tone and resolution direction. Full-thread context is stored for reference but not used in generation. This is explicitly noted as a "What I'd do with one more week" improvement.

---

## Decision 10: 200 Golden Examples (not 150, not 250)

**Choice:** 200 hand-labelled examples  
**Alternatives considered:** 150 (minimum per assignment), 250 (maximum per assignment)  
**Reasoning:** 200 is the sweet spot — enough for reliable per-class metrics (≥5 examples per intent even for rare classes like "Account / Login Issue") while remaining feasible to hand-label in a single session. 150 would leave some classes under-represented; 250 would take proportionally more time without significantly improving statistical power at this scale.

---

## Decision 11: Stratified + Adversarial Sampling over Pure Random

**Choice:** 70% stratified random + 20% hard/ambiguous cases + 10% edge cases  
**Alternatives considered:** Pure random sampling, pure stratified, cluster-based  
**Reasoning:** Pure random sampling would over-represent the dominant "Other" class (75% of data) and under-test edge cases. Our adversarial component deliberately includes multi-intent messages, very short texts, non-English messages, and PII-containing tweets — exactly the cases where the system is most likely to fail. This gives us a more honest evaluation.

---

## Decision 12: 10K Subsample for Vector Index (not full 151K)

**Choice:** Index 10,000 randomly sampled threads in ChromaDB  
**Alternatives considered:** Full 151K index, 1K index, 50K index  
**Reasoning:** Embedding and indexing 151K documents with sentence-transformers takes significant time and memory. At 10K, retrieval quality is already strong (diverse coverage of issue types) and setup takes <2 minutes. This keeps the "reproduce in under 15 minutes" requirement feasible. The assignment explicitly says "a subsample is expected and encouraged."

---

## Decision 13: Self-Reported LLM Confidence (not Calibrated Logprobs)

**Choice:** Ask the LLM to self-report confidence as a float in its JSON output  
**Alternatives considered:** Token logprobs (not available via all APIs), MC dropout, ensemble disagreement  
**Reasoning:** Self-reported confidence is simple and interpretable. It's not well-calibrated out-of-the-box, but we explicitly note this in the "What is misleading" section. Logprobs would be more principled but add API complexity and aren't available for all models. This is an honest trade-off we document rather than hide.
