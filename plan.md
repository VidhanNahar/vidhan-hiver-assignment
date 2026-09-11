# Implementation Plan — AmazonHelp AI Support Agent

> **Brand:** AmazonHelp  
> **Dataset:** Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)  
> **Guiding Principle:** *"The proof is worth more than the system."*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Phase 1 — Data Acquisition & Exploration](#3-phase-1--data-acquisition--exploration)
4. [Phase 2 — Intent Taxonomy Design](#4-phase-2--intent-taxonomy-design)
5. [Phase 3 — Golden Evaluation Set](#5-phase-3--golden-evaluation-set)
6. [Phase 4 — Core Pipeline Implementation](#6-phase-4--core-pipeline-implementation)
7. [Phase 5 — Baselines](#7-phase-5--baselines)
8. [Phase 6 — Evaluation Harness](#8-phase-6--evaluation-harness)
9. [Phase 7 — Report & Decision Log](#9-phase-7--report--decision-log)
10. [Tech Stack](#10-tech-stack)
11. [Timeline](#11-timeline)
12. [Risk Register](#12-risk-register)

---

## 1. Project Overview

### What We're Building

An AI support agent for **AmazonHelp** that takes an incoming customer tweet and:

1. **Classifies** it into one of ~8 intent categories we define from the data.
2. **Drafts a reply** grounded in how AmazonHelp has historically responded to similar issues.
3. **Decides** whether to auto-handle or escalate to a human, with a stated reason.

### Why AmazonHelp?

- **High volume** in the dataset → more data for retrieval and pattern discovery.
- **Diverse issue types** — orders, shipping, refunds, Prime, Kindle, account security — gives us clearly separable intents.
- **Clear escalation boundary** — some issues (refunds, account takeover) obviously need humans; others (tracking links, how-to questions) can be auto-resolved. This makes the escalation task non-trivial but learnable.

### What We're NOT Building

- A production-grade chatbot with multi-turn state management.
- Fine-tuned models — we use prompting and RAG (time-efficient, explainable).
- Full-dataset processing — we work on a curated subsample (~2,000–5,000 AmazonHelp threads).

---

## 2. Repository Structure

```
vidhan-hiver-assignment/
├── README.md                        # Main report (≤6 pages) + reproduction instructions
├── plan.md                          # This file
├── Hiver SDE Intern Assignment.md   # Original assignment
│
├── data/
│   ├── raw/                         # Raw CSV from Kaggle (gitignored)
│   ├── processed/                   # Cleaned, filtered AmazonHelp threads
│   └── golden/                      # Hand-labelled golden eval set (150–250 examples)
│       ├── golden_set.json
│       └── labelling_notes.md       # Sampling & labelling methodology
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py              # Kaggle API download script
│   │   ├── preprocess.py            # Tweet cleaning, thread reconstruction
│   │   └── sample.py               # Stratified sampling for golden set
│   ├── pipeline/
│   │   ├── classifier.py            # Intent classification (LLM-based)
│   │   ├── retriever.py             # RAG: embed & retrieve historical responses
│   │   ├── responder.py             # Reply generation (grounded in retrieved context)
│   │   ├── escalation.py            # Escalation decision engine
│   │   └── agent.py                 # Orchestrator — ties all components together
│   ├── baselines/
│   │   ├── trivial.py               # Baseline 1: majority class + canned reply
│   │   └── simple.py                # Baseline 2: zero-shot LLM (no RAG, no few-shot)
│   └── eval/
│       ├── metrics.py               # Automated metrics (F1, precision, recall, etc.)
│       ├── llm_judge.py             # LLM-as-judge scoring for reply quality
│       ├── judge_calibration.py     # Human vs. LLM judge agreement analysis
│       └── run_eval.py              # Entry point: run full evaluation harness
│
├── notebooks/                       # Exploratory analysis (optional, not graded)
│   └── 01_data_exploration.ipynb
│
├── configs/
│   └── config.yaml                  # API keys placeholder, model selection, hyperparams
│
├── decision_log.md                  # 10–15 non-obvious decisions
├── requirements.txt
├── pyproject.toml
├── Makefile                         # make setup, make eval, make demo
└── .gitignore
```

---

## 3. Phase 1 — Data Acquisition & Exploration

### 3.1 Download

- Use the Kaggle API (`kaggle datasets download -d thoughtvector/customer-support-on-twitter`).
- Provide a fallback `download.py` script that also works without the Kaggle CLI.
- Raw data goes into `data/raw/` (gitignored; README tells users how to get it).

### 3.2 Preprocessing

**Input:** Raw CSV with columns `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`.

**Steps:**
1. **Filter** to only `AmazonHelp` threads (brand author_id).
2. **Thread reconstruction:** Link `in_response_to_tweet_id` chains to build multi-turn conversations. Extract the first customer inbound message as the "query" and the first AmazonHelp reply as the "historical response."
3. **Clean text:**
   - Remove `@mentions` from the start of tweets (but preserve mid-text mentions).
   - Decode HTML entities (`&amp;` → `&`).
   - Normalize URLs → `[URL]` placeholder.
   - Strip trailing whitespace, normalize unicode.
4. **Deduplicate** near-identical tweets (exact match + fuzzy threshold).
5. **Output:** `data/processed/amazon_threads.json` — each record:
   ```json
   {
     "thread_id": "...",
     "customer_text": "...",
     "brand_reply": "...",
     "full_thread": ["msg1", "msg2", ...],
     "created_at": "..."
   }
   ```

### 3.3 Exploratory Analysis

- Distribution of thread lengths.
- Most common keywords/phrases (word cloud or frequency table).
- Volume over time (to check for temporal patterns / events).
- Sample ~50 random threads manually to develop intuition for intent categories.

**Output of Phase 1:** Clean dataset of ~2,000–5,000 AmazonHelp customer→brand pairs.

---

## 4. Phase 2 — Intent Taxonomy Design

### Approach: Bottom-Up from Data

1. **Manual scan:** Read 100–150 random customer messages. Cluster them by what the customer wants.
2. **LLM-assisted clustering:** Send batches of ~50 messages to an LLM with the prompt: *"Group these customer messages into 5–10 categories based on the customer's primary intent."* Merge across batches.
3. **Refine:** Collapse overlapping categories, ensure each is actionable (i.e., leads to a different kind of response).

### Proposed Intent Taxonomy (Draft — to be refined after data exploration)

| # | Intent | Description | Example |
|---|--------|-------------|---------|
| 1 | **Order Status / Tracking** | Where is my order? When will it arrive? | "I ordered 3 days ago and it still says processing" |
| 2 | **Refund / Return** | I want my money back, item was wrong/damaged | "I received a broken item, need a refund" |
| 3 | **Delivery Problem** | Package not delivered, wrong address, stolen | "Says delivered but I never got it" |
| 4 | **Account / Login Issue** | Can't log in, locked out, suspicious activity | "My account got hacked and I can't reset" |
| 5 | **Product / Service Question** | How does X work? Is Y compatible? | "Does Prime include free shipping on this?" |
| 6 | **Prime / Subscription** | Prime billing, cancellation, benefits | "I was charged for Prime but I cancelled" |
| 7 | **Complaint / Frustration** | General dissatisfaction, poor experience, rant | "Worst customer service ever, 3 hours on hold" |
| 8 | **Other / Miscellaneous** | Doesn't fit above; praise, unrelated | "Thanks for the quick help!" |

> **Decision to Log:** Why 8 classes and not 15 or 4. Trade-off between granularity (useful for routing) and classifier accuracy (fewer classes → higher accuracy). 8 is our starting point; may collapse after seeing confusion matrix.

---

## 5. Phase 3 — Golden Evaluation Set

### 5.1 Sampling Strategy

We need 150–250 examples that are **representative but also stress-test edge cases**.

**Sampling method: Stratified + Adversarial**
- **70% Stratified random:** Sample proportionally from each intent bucket (after a first-pass LLM classification of the full processed set). Ensures every intent has ≥10 examples.
- **20% Hard / ambiguous cases:** Messages that sit near intent boundaries or contain multiple intents. Identified by low LLM classifier confidence or disagreement between two LLM runs.
- **10% Edge cases:** Very short messages (<5 words), messages with heavy slang/emoji, messages with PII, angry rants with no clear ask.

**Target: 200 examples.**

### 5.2 Labelling Schema

Each example gets:

```json
{
  "id": "golden_001",
  "customer_text": "...",
  "historical_brand_reply": "...",
  "labels": {
    "intent": "Refund / Return",
    "escalate": true,
    "escalation_reason": "Customer requests monetary refund — requires account verification and order lookup that cannot be done via public tweet.",
    "notes": "Multi-intent: also mentions delivery issue. Primary intent is refund."
  }
}
```

### 5.3 Labelling Process

1. **Solo pass:** Label all 200 examples myself.
2. **Consistency check:** Re-label 30 random examples 48 hours later (intra-annotator agreement). Report Cohen's κ.
3. **Document edge cases** in `labelling_notes.md`: ambiguous examples and the reasoning behind each judgement call.

**Output of Phase 3:** `data/golden/golden_set.json` + `data/golden/labelling_notes.md`

---

## 6. Phase 4 — Core Pipeline Implementation

### 6.1 Architecture

```
Customer Tweet
      │
      ▼
┌─────────────────────────┐
│  Intent Classifier      │  → intent label + confidence
│  (Few-shot LLM prompt)  │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      ▼              ▼
┌────────────┐  ┌─────────────────────┐
│ Escalation │  │ RAG Retriever       │
│ Decision   │  │ (Embed + Retrieve   │
│ Engine     │  │  top-k similar      │
│            │  │  historical replies) │
└─────┬──────┘  └─────────┬───────────┘
      │                   │
      │         ┌─────────▼───────────┐
      │         │ Reply Generator     │
      │         │ (Grounded in        │
      │         │  retrieved context) │
      │         └─────────┬───────────┘
      │                   │
      ▼                   ▼
┌─────────────────────────────────┐
│  Final Output:                  │
│  - intent                       │
│  - draft_reply                  │
│  - escalate (bool)              │
│  - escalation_reason            │
│  - confidence_score             │
└─────────────────────────────────┘
```

### 6.2 Intent Classifier (`src/pipeline/classifier.py`)

**Method:** Few-shot prompted LLM classification.

- **Prompt structure:**
  1. System: "You are an intent classifier for AmazonHelp customer support tweets."
  2. Taxonomy definition with descriptions + 2 examples per class.
  3. User message → classify.
  4. Output: JSON `{"intent": "...", "confidence": 0.0-1.0, "reasoning": "..."}`.
- **Model:** `gpt-4o-mini` (cost-effective, fast, good at classification).
- **Confidence:** Parsed from the LLM's self-reported confidence. We'll calibrate this against actual accuracy.

### 6.3 RAG Retriever (`src/pipeline/retriever.py`)

**Method:** Semantic search over historical AmazonHelp responses.

1. **Embedding:** Embed all historical `customer_text` using `all-MiniLM-L6-v2` (sentence-transformers, local, free).
2. **Vector store:** ChromaDB (local, file-based, simple).
3. **Retrieval:** Given an incoming message, retrieve top-5 most similar historical customer queries. Return their brand replies as context.
4. **Filtering:** Optionally filter by predicted intent to narrow retrieval scope.

**Decision to Log:** Embedding model choice — local sentence-transformers for reproducibility and zero cost.

### 6.4 Reply Generator (`src/pipeline/responder.py`)

**Method:** Grounded generation with retrieved context.

- **Prompt structure:**
  1. System: "You are AmazonHelp. Draft a helpful reply to this customer tweet. Match the brand's tone and style. Use the following real AmazonHelp responses as references for tone and approach."
  2. Retrieved historical examples (customer query + brand reply pairs).
  3. Current customer message.
  4. Output: Draft reply in AmazonHelp's voice.
- **Grounding constraints in prompt:**
  - "Do not make up order numbers, tracking links, or specific account details."
  - "If the issue requires account access, direct the customer to DM."
  - "Keep the reply under 280 characters (tweet length)."

### 6.5 Escalation Engine (`src/pipeline/escalation.py`)

**Method:** Rule-augmented LLM decision.

**Hard rules (always escalate):**
- Customer mentions legal action, lawyer, lawsuit.
- Customer mentions self-harm or safety.
- Customer requests refund/money back (requires account verification).
- Message contains PII (phone number, email, order number shared publicly).

**LLM-based soft decision:**
- For all other cases, ask the LLM: given the intent, customer sentiment, and draft reply, should this be auto-handled or escalated?
- Output: `{"escalate": bool, "reason": "...", "confidence": 0.0-1.0}`.

**Escalation reason taxonomy:**
- `requires_account_verification`
- `monetary_request`
- `angry_or_abusive`
- `complex_multi_issue`
- `pii_exposure`
- `safety_concern`
- `low_confidence_classification`
- `auto_handleable`

### 6.6 Orchestrator (`src/pipeline/agent.py`)

Ties it all together:

```python
def process_message(customer_text: str) -> AgentResponse:
    # Step 1: Classify intent
    classification = classifier.classify(customer_text)
    
    # Step 2: Retrieve similar historical interactions
    similar_threads = retriever.retrieve(customer_text, intent=classification.intent, top_k=5)
    
    # Step 3: Generate grounded reply
    draft_reply = responder.generate(customer_text, similar_threads, classification.intent)
    
    # Step 4: Escalation decision
    escalation = escalation_engine.decide(
        customer_text, classification, draft_reply
    )
    
    return AgentResponse(
        intent=classification.intent,
        confidence=classification.confidence,
        draft_reply=draft_reply,
        escalate=escalation.escalate,
        escalation_reason=escalation.reason,
    )
```

---

## 7. Phase 5 — Baselines

Two baselines are required. They must be run on the **same golden set**.

### 7.1 Baseline 1 — Trivial (Keyword + Canned Reply)

**Intent:** Keyword matching with a priority list.
- "refund" / "money back" → `Refund / Return`
- "tracking" / "where is" / "when will" → `Order Status / Tracking`
- "can't log in" / "password" / "hacked" → `Account / Login Issue`
- Default → `Other`

**Reply:** One canned reply per intent (e.g., "We're sorry to hear about this. Please DM us your order details so we can look into it.").

**Escalation:** Always escalate (trivially safe but useless as an auto-handler).

### 7.2 Baseline 2 — Simple (Zero-Shot LLM, No RAG)

**Intent:** Same LLM as main system, but zero-shot (no examples in prompt, just taxonomy).

**Reply:** LLM generates a reply with no retrieved context — just the customer message and a generic "respond as AmazonHelp" instruction.

**Escalation:** LLM decides with no rules, no context.

**Purpose:** Isolates the value added by few-shot examples, RAG retrieval, and hard escalation rules.

---

## 8. Phase 6 — Evaluation Harness

### 8.1 Automated Metrics (`src/eval/metrics.py`)

**Intent Classification:**
- Per-class Precision, Recall, F1.
- Macro-F1 (headline metric).
- Confusion matrix (to identify commonly confused intent pairs).

**Escalation Decision:**
- Precision, Recall, F1 for the `escalate=True` class.
- **False Negative Rate** is the critical metric — a missed escalation is worse than a false alarm.

### 8.2 LLM-as-Judge for Reply Quality (`src/eval/llm_judge.py`)

**Rubric (1–5 scale on each dimension):**

| Dimension | 1 (Poor) | 3 (Acceptable) | 5 (Excellent) |
|-----------|----------|-----------------|----------------|
| **Relevance** | Reply ignores the customer's issue | Addresses the issue but vaguely | Directly addresses the specific problem |
| **Groundedness** | Hallucinates details (fake order #, policies) | Mostly grounded, minor overreach | Only states verifiable info or directs to DM |
| **Tone** | Robotic, rude, or off-brand | Polite but generic | Empathetic, matches AmazonHelp's voice |
| **Actionability** | No next step for the customer | Vague next step | Clear, specific next step |
| **Safety** | Shares PII, makes promises it shouldn't | No harm, but doesn't proactively protect | Proactively directs PII to DM, sets correct expectations |

**Overall score:** Average of 5 dimensions.

**Judge model:** `gpt-4o` (or `gpt-4.1`) — stronger model than the system itself to reduce self-bias.

### 8.3 Judge Calibration (`src/eval/judge_calibration.py`)

This is critical for credibility.

1. **Manually score 40–50 examples** on the same rubric (5 dimensions × 1–5).
2. **Run the LLM judge** on the same 40–50 examples.
3. **Compute agreement:**
   - Per-dimension Pearson correlation (continuous scores).
   - Per-dimension Cohen's κ (after binning into Low/Medium/High).
   - Overall MAE (mean absolute error) between human and judge scores.
4. **Report:** "The LLM judge agrees with human scoring at r=X on relevance, r=Y on tone, ..." with a discussion of where it disagrees and why.
5. **If agreement is low** on a dimension, either recalibrate the rubric, add examples to the judge prompt, or flag that dimension as unreliable.

### 8.4 Evaluation Runner (`src/eval/run_eval.py`)

Single entry point:
```bash
python -m src.eval.run_eval --golden data/golden/golden_set.json --system main
python -m src.eval.run_eval --golden data/golden/golden_set.json --system baseline_trivial
python -m src.eval.run_eval --golden data/golden/golden_set.json --system baseline_simple
```

Outputs a results JSON + human-readable comparison table.

---

## 9. Phase 7 — Report & Decision Log

### 9.1 Report Structure (README.md, ≤6 pages equivalent)

1. **Problem Framing**
   - Why AmazonHelp, what "good" means (safe > helpful > fast), what we chose not to build.
2. **System Description**
   - Architecture diagram, component descriptions, prompt designs.
3. **Results**
   - Table: Main System vs. Baseline 1 vs. Baseline 2 on all metrics.
   - Intent confusion matrix.
   - Reply quality score distributions.
4. **Failure Analysis — Top 5 Failure Modes**
   - Each with: real example, what went wrong, hypothesis for why, potential fix.
   - e.g., "Multi-intent messages get classified as the less important intent."
   - e.g., "Sarcastic complaints get classified as praise."
5. **"What is misleading about my headline number?"**
   - Golden set bias (I labelled it, so I may have unconsciously designed for my system).
   - LLM judge inflation (LLM judges tend to be lenient).
   - Subsample ≠ production distribution.
   - Tweet-level eval ignores multi-turn conversation quality.
6. **What I'd do with one more week**
   - Multi-turn conversation handling.
   - Fine-tune a small classifier (distilbert) on the LLM-labelled data.
   - A/B test different retrieval strategies.
   - Add a feedback loop / active learning.

### 9.2 Decision Log (`decision_log.md`)

Target: 10–15 entries. Format:

```
## Decision: [Title]
**Choice:** [What I did]
**Alternatives considered:** [What I didn't do]
**Reasoning:** [Why]
```

Draft decisions to log:
1. Brand selection — AmazonHelp over SpotifyCares/AppleSupport.
2. 8-class intent taxonomy — not 4, not 15.
3. Few-shot over fine-tuned classifier.
4. Embedding model choice (local vs API).
5. ChromaDB over FAISS.
6. Hybrid escalation (hard rules + LLM) over pure LLM.
7. GPT-4o-mini for pipeline vs GPT-4o for judge (cost vs quality).
8. 280-char reply constraint (tweet realism).
9. Thread reconstruction strategy (first-pair only vs full thread).
10. Golden set size — 200 (not 150, not 250).
11. Stratified + adversarial sampling over pure random.
12. Confidence calibration approach.
13. Evaluation dimensions and rubric design.

---

## 10. Tech Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| **Language** | Python 3.11+ | Standard for ML/NLP |
| **LLM (Pipeline)** | `gpt-4o-mini` via OpenAI API | Cost-effective, fast, strong at classification |
| **LLM (Judge)** | `gpt-4o` / `gpt-4.1` | Stronger model for unbiased evaluation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, reproducible, good quality |
| **Vector Store** | ChromaDB | Simple, file-based, no server needed |
| **Data Processing** | pandas, json | Standard |
| **Evaluation** | scikit-learn (metrics), matplotlib/seaborn (plots) | Well-tested, familiar |
| **CLI / Demo** | Streamlit (optional) or plain CLI | Fast to build, impressive for demo |
| **Config** | YAML + python-dotenv | Simple, readable |
| **Build** | Makefile | `make setup`, `make eval`, `make demo` — reproducible |

---

## 11. Timeline

Estimated total: **5–7 days of focused work.**

| Day | Phase | Key Output |
|-----|-------|------------|
| **1** | Phase 1: Data download, preprocessing, exploration | `data/processed/amazon_threads.json`, EDA notes |
| **2** | Phase 2: Intent taxonomy + Phase 3 start: Begin golden set labelling | Finalized taxonomy, ~100 labels done |
| **3** | Phase 3 complete + Phase 4 start: Finish labelling, build classifier & retriever | `golden_set.json`, working classifier |
| **4** | Phase 4: Build responder, escalation engine, orchestrator | Full pipeline working end-to-end |
| **5** | Phase 5 + Phase 6: Baselines + evaluation harness + judge calibration | All baselines, eval results, judge agreement report |
| **6** | Phase 7: Report writing, failure analysis, decision log | README, `decision_log.md`, polished repo |
| **7** | Buffer: Polish, edge case fixes, final review | Submission-ready repo |

---

## 12. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| AmazonHelp data is too noisy / threads malformed | Pipeline produces garbage | Extensive preprocessing + manual spot-checks at Phase 1 |
| Intent categories are too ambiguous | Low classifier F1 | Iterate on taxonomy in Phase 2; collapse classes if needed |
| LLM API rate limits / costs | Slow eval, budget overrun | Use `gpt-4o-mini` for pipeline, batch requests, cache responses |
| LLM judge disagrees with human scores | Evaluation loses credibility | Calibrate thoroughly in Phase 6; adjust rubric or report disagreement honestly |
| Golden set too easy / not representative | Inflated metrics | Adversarial sampling strategy (20% hard cases, 10% edge cases) |
| Reproducibility issues | Evaluators can't run it | Pin all deps, provide sample data, test from clean env before submission |
| Reply generation hallucinates | Safety failure | Strong grounding constraints in prompt + escalation rules catch dangerous outputs |

---

## Summary of Success Criteria

The final system will be judged on:

1. ✅ **Runnable in <15 minutes** from a clean clone.
2. ✅ **Golden set exists** with documented methodology.
3. ✅ **Evaluation is rigorous** — automated metrics + LLM judge + human-judge agreement analysis.
4. ✅ **Two baselines** clearly show the value of the main system.
5. ✅ **Failure analysis is honest** — top 5 modes with real examples.
6. ✅ **"What is misleading"** section demonstrates self-awareness.
7. ✅ **Decision log** shows engineering maturity.

> The system doesn't need to be perfect. It needs to be *honestly evaluated*.
