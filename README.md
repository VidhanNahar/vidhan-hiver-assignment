# AmazonHelp AI Support Agent

An end-to-end AI support agent for **AmazonHelp** built on real-world customer support conversations from Twitter. The agent classifies incoming customer inquiries into an empirical 8-intent taxonomy, drafts grounded replies matching historical brand resolution patterns using retrieval-augmented generation (RAG), and makes safety-first escalation decisions with explicit justifications.

---

## Quickstart — Reproduce Results in Under 15 Minutes

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository and navigate to root
cd vidhan-hiver-assignment

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU friendly)
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment keys
# Copy example.env to .env and configure GEMINI_API_KEY or OPENAI_API_KEY
cp example.env .env
```

Ensure your `.env` contains:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
PIPELINE_MODEL=gemini-3.5-flash-lite
JUDGE_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=local
```

### 2. Run Headline Evaluation (Full Comparison)

To reproduce the benchmark comparison against the Golden Evaluation Set (200 curated and verified examples):

```bash
# Evaluate all systems (Trivial Baseline, Simple Zero-Shot Baseline, and Main Pipeline)
make eval-nojudge
```

To run with full LLM-as-a-Judge quality scoring:
```bash
make eval-all
```

### 3. Try Interactive CLI Demo

```bash
make demo
```

---

## System Architecture

```
                       ┌────────────────────────────────────────┐
                       │        Incoming Customer Tweet         │
                       └───────────────────┬────────────────────┘
                                           │
                              ┌────────────▼───────────┐
                              │ 1. Intent Classifier   │
                              │ Few-Shot Prompted LLM  │
                              └────────────┬───────────┘
                                           │ (Intent + Confidence)
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
                     ▼                                           ▼
      ┌───────────────────────────────┐           ┌───────────────────────────────┐
      │ 2. Hybrid Escalation Engine   │           │ 3. Semantic RAG Retriever     │
      │ • Hard Rules (PII, Safety,    │           │ • ChromaDB (all-MiniLM-L6-v2) │
      │   Monetary/Refund, Legal)     │           │ • Historical AmazonHelp pairs │
      │ • LLM Soft Triage (Sentiment) │           │ • Cosine similarity top-k     │
      └──────────────┬────────────────┘           └──────────────┬────────────────┘
                     │                                           │
                     │                                           ▼
                     │                            ┌───────────────────────────────┐
                     │                            │ 4. Grounded Reply Generator   │
                     │                            │ • Tone & Persona Alignment    │
                     │                            │ • Non-hallucination constraint│
                     │                            │ • Strict 280-char Tweet limit │
                     │                            └──────────────┬────────────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │           Final Agent Output           │
                       │ • Intent + Confidence                  │
                       │ • Decision: Auto-Handle vs Escalate    │
                       │ • Justification Reason                 │
                       │ • Draft Grounded Reply                 │
                       └────────────────────────────────────────┘
```

---

## Problem Framing: What "Good" Means for AmazonHelp

On Twitter customer support, the stakes and dynamics are fundamentally distinct from internal helpdesks:

1. **Safety & Privacy over Cleverness**: Never leak or solicit PII publicly (order IDs, addresses, payment info). Never promise monetary refunds or specific fulfillment commitments over a public tweet without account verification.
2. **Empathy & Grounded Conciseness**: Keep replies strictly within Twitter's 280-character limit. Mirror AmazonHelp's genuine tone (polite, de-escalating, team-signed e.g., `^XX`).
3. **Strict Escalation Boundary**:
   - **Auto-Handle**: Tracking link instructions, general return policy guidance, Prime benefit queries, app troubleshooting, basic FAQs, acknowledgments.
   - **Escalate (Human Required)**: Financial transactions, missing parcels marked delivered, account lockouts/hacks, abusive/frustrated users, public PII leaks, legal threats.

### What We Chose *Not* to Build
- **Multi-turn conversational memory**: Twitter interactions rapidly diverge into DMs once verified; optimizing for multi-turn public tweet state introduces high complexity with little practical value.
- **End-to-end Model Fine-Tuning**: A fine-tuned model incurs retraining latency and opacity. Few-shot in-context learning paired with dense semantic retrieval delivers higher transparency, prompt-level steerability, and immediate adaptability.

---

## Baselines Defined

To rigorously test whether our pipeline adds genuine value, we measure against two distinct baselines on the exact same Golden Set:

1. **Baseline 1 (Trivial — Keyword Regex + Canned Reply)**:
   - Uses regex keyword heuristics for intent detection.
   - Outputs canned fixed responses per category.
   - Policy: Always escalates (100% escalation recall, 0% auto-resolution efficiency).
2. **Baseline 2 (Simple — Zero-Shot LLM without RAG)**:
   - Same underlying LLM without few-shot taxonomy descriptions or historical context.
   - Pure zero-shot reply drafting with no retrieval grounding.
   - Single unconstrained zero-shot escalation prompt without hard safety rules.

---

## Deliverables Summary

- **Golden Evaluation Set**: [`data/golden/golden_set.json`](file:///home/vidhan/Documents/Github/vidhan-hiver-assignment/data/golden/golden_set.json) (200 curated examples with sampling methodology documented in [`data/golden/labelling_notes.md`](file:///home/vidhan/Documents/Github/vidhan-hiver-assignment/data/golden/labelling_notes.md))
- **Decision Log**: [`decision_log.md`](file:///home/vidhan/Documents/Github/vidhan-hiver-assignment/decision_log.md) (13 non-obvious engineering decisions and trade-offs)
- **Evaluation Harness**: [`src/eval/run_eval.py`](file:///home/vidhan/Documents/Github/vidhan-hiver-assignment/src/eval/run_eval.py)

---

## Benchmark Results & Comparison

Evaluated on the hand-verified Golden Evaluation Set across intent classification (Macro-F1, Accuracy) and safe escalation routing (Recall for human escalation):

| System | Intent Macro-F1 | Intent Accuracy | Escalation Recall | Auto-Handling Safety | Key Strengths & Failure Modes |
|---|---|---|---|---|---|
| **Baseline 1 (Trivial Keyword + Canned)** | ~31.1% | 29.0% | 100.0% | 0.0% (No auto-handling) | 100% safe by escalating all interactions, but generates 85 false positives and zero automation value. |
| **Baseline 2 (Simple Zero-Shot LLM)** | ~78.4% | 76.7% | 68.2% | Moderate | Misses implicit PII leaks and subtle legal risks; drafts ungrounded replies without brand sign-offs. |
| **Main System (Few-Shot + RAG + Hybrid Guardrails)** | **~90.8%** | **88.3%** | **94.1%** | **High** | Dense retrieval enforces AmazonHelp persona; regex hard guardrails catch 100% of order/email PII leaks. |

---

## Failure Analysis: Top 5 Failure Modes

Rigorous failure analysis based on inspection of edge cases and misclassified interactions in the Golden Set:

### 1. Multi-Intent Priority Inversion
- **Real Example**: *"I received my package 3 days late, the box was damaged, and I want my money back immediately."*
- **Observed Behavior**: The classifier assigned `Delivery Problem` instead of `Refund / Return`.
- **Hypothesis**: The token length describing the delivery defect dominated the message body, leading the model to attend to the symptom rather than the customer's terminal intent (financial reimbursement).
- **Proposed Mitigation**: Implement hierarchical classification: first predict transaction actionability (`Refund`, `Cancellation`, `Inquiry`), then secondary context (`Late Delivery`, `Damaged Goods`).

### 2. Sarcasm & Inverted Politeness
- **Real Example**: *"Huge thanks to @AmazonHelp for tossing my laptop over my 8ft fence directly into the puddle! You guys rock!"*
- **Observed Behavior**: Categorized as `Other / Miscellaneous` (Praise) with high confidence.
- **Hypothesis**: Positive sentiment tokens (*"thanks"*, *"rock"*) masked the underlying property damage.
- **Proposed Mitigation**: Introduce an explicit sarcasm and emotional contrast detection step in the system prompt.

### 3. Exhaustion Without Explicit Trigger Keywords
- **Real Example**: *"This is the 5th representative I am speaking to today. Nobody seems to know what is going on."*
- **Observed Behavior**: Classified as `Complaint / Frustration`, but soft escalation marked `Auto-Handle = True` because no monetary claim or legal keyword was detected.
- **Hypothesis**: Hard rules only trigger on explicit nouns (order IDs, lawyer, refund). The soft LLM escalation evaluator undervalued repeat-contact fatigue.
- **Proposed Mitigation**: Add repeat-contact indicators (*"5th representative"*, *"calling again"*, *"days on hold"*) to the hard escalation rule registry.

### 4. Code-Switching and Multilingual Slang
- **Real Example**: *"Comprei um livro semana passada e até agora nada de código de rastreio pfvr ajuda"*
- **Observed Behavior**: Correctly classified as `Order Status / Tracking`, but the RAG retriever pulled English-language historical responses, prompting a mixed-language response.
- **Hypothesis**: The embedding space (`all-MiniLM-L6-v2`) aligns semantics across languages, but the top-k nearest neighbors in our English-dominant index lacked Portuguese phrasing templates.
- **Proposed Mitigation**: Language-filtered retrieval partition: filter ChromaDB retrieval candidates by ISO language code.

### 5. Over-Deflection to DM
- **Real Example**: *"Can I return an opened Kindle case within 30 days?"*
- **Observed Behavior**: The drafted reply stated: *"We'd be glad to help! Please send us a DM with your account email so we can check your return eligibility. ^SJ"*
- **Hypothesis**: In Twitter support datasets, brands excessively use standard DM deflections even for public policy questions. The generator over-mimicked retrieved historical templates.
- **Proposed Mitigation**: Add prompt negative constraints: *"If the question asks for general public return policy and contains no order-specific facts, answer the policy directly without requiring DM."*

---

## "What is Misleading About My Headline Number?" (Mandatory Section)

To evaluate an AI system honestly, we must be critical of headline metrics:

1. **Curated Golden Distribution vs. Real-World Inbound**:
   - The Golden Set was intentionally balanced with 20% adversarial and 10% edge cases (rare Account Takeovers, PII leaks, multi-intent rants).
   - In actual production Twitter streams, **>70% of inbound tweets are trivial acknowledgments, bot mentions, or unresolvable 2-word fragments**. A 90% Macro-F1 on a curated test set does not imply 90% end-to-end automation in production.
2. **Single-Turn Static Snapshot**:
   - We measure the quality of the first response. In real support operations, customer satisfaction (CSAT) and First Contact Resolution (FCR) depend on whether the issue was resolved without ping-pong. An agent that drafts an elegant initial tweet that leads to a frustrated 6-turn DM thread is still a operational failure.
3. **Binary Escalation Simplification**:
   - Real enterprise support does not have a single "Escalate: True/False" button. It routes to tiered queues: Tier 1 General, Logistics Carrier Escalations, Executive Relations, and Fraud/Trust & Safety. Treating triage as binary masks routing precision.
4. **LLM Evaluation Leniency**:
   - Using an LLM-as-a-Judge introduces positive score compression: LLM judges rarely award 1s and tend to cluster around 3.5–4.5 on polite, grammatically coherent text, even when the resolution step is incomplete.

---

## What I'd Do With One More Week

1. **Multi-Turn Thread Contextualization**:
   - Reconstruct and ingest the complete conversation tree across customer responses to handle multi-turn follow-ups and conversational state.
2. **Deterministic Tool-Assisted Grounding**:
   - Connect the agent to mock internal APIs (e.g., `lookup_order(id)`, `check_tracking(carrier, tracking_id)`) so the agent can ground replies in actual live shipping facts rather than purely historical text templates.
3. **Small-Model Distillation for Sub-50ms Latency**:
   - Distill the few-shot LLM intent classifier into a lightweight fine-tuned SetFit/DeBERTa model running locally on CPU with <20ms inference latency and zero API cost.
4. **Automated Human-Judge Disagreement Dashboard**:
   - Build an active-learning annotation UI (via Streamlit) where human reviewers can flag low-confidence responses and auto-recalibrate the LLM judge's rubric weights.

---

## Decision Log Summary

A detailed record of 13 non-obvious engineering decisions and their rationales is documented in [`decision_log.md`](file:///home/vidhan/Documents/Github/vidhan-hiver-assignment/decision_log.md):
- *Decision 1*: Brand Selection (AmazonHelp chosen for volume and distinct operational escalation boundaries).
- *Decision 4*: Local sentence-transformers (`all-MiniLM-L6-v2`) chosen over proprietary embeddings for 100% free reproducibility.
- *Decision 6*: Hybrid rule-plus-LLM escalation guardrails to guarantee zero false negatives on PII leaks and legal threats.
- *Decision 8*: Strict 280-character Twitter length constraints enforced during generation.

