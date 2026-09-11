# Golden Set — Labelling Notes

## Sampling Methodology

The golden evaluation set contains **200 hand-labelled examples** drawn from the 151,259 processed AmazonHelp customer→brand thread pairs.

### Sampling Strategy: Stratified + Adversarial

| Category | Count | % | Purpose |
|----------|-------|---|---------|
| **Stratified random** | 140 | 70% | Proportional representation across all intent buckets |
| **Hard / ambiguous cases** | 40 | 20% | Multi-intent messages, ambiguous phrasing, classifier-confusing cases |
| **Edge cases** | 20 | 10% | Very short (<15 chars), non-English, PII-containing, heavy emoji |

### How Stratification Was Done

1. All 151K processed threads were classified using a **heuristic keyword-based classifier** (not the LLM — to avoid circularity).
2. The heuristic distribution showed "Other/Miscellaneous" dominates at ~75%. To ensure every intent has ≥5 examples, we used proportional sampling with a minimum floor of 5 per intent.
3. Hard cases were identified by checking for messages that matched keywords from **2+ different intent categories** (multi-intent signal).
4. Edge cases were identified by: text length <15 chars or >250 chars, ≥3 emoji, non-English language markers, or PII patterns (order IDs, emails, phone numbers).

### Heuristic Intent Distribution in Full Dataset vs Golden Set

| Intent | Full Dataset | Golden Set |
|--------|-------------|------------|
| Other / Miscellaneous | 75.4% | 12.0% (24) |
| Complaint / Frustration | 3.4% | 19.5% (39) |
| Delivery Problem | 1.7% | 19.0% (38) |
| Product / Service Question | 1.8% | 14.5% (29) |
| Order Status / Tracking | 2.6% | 12.5% (25) |
| Refund / Return | 6.2% | 9.5% (19) |
| Account / Login Issue | 0.9% | 7.0% (14) |
| Prime / Subscription | 8.0% | 6.0% (12) |

**Total:** 200 examples.  
**Escalation breakdown:** 115 (57.5%) Escalate=True, 85 (42.5%) Escalate=False (Auto-handle).

The golden set intentionally balances minority high-risk classes (e.g. Account/Login issues, Delivery problems, Complaints) to provide rigorous evaluation power.

---

## Labelling Schema

Each example is labelled with:

```json
{
  "intent": "<one of the 8 categories>",
  "escalate": true/false,
  "escalation_reason": "<why this should/shouldn't be escalated>",
  "notes": "<any edge case observations, multi-intent notes, etc.>"
}
```

### Intent Label Guidelines

- **Primary intent only:** If a message has multiple intents (e.g., "Where is my order? And I want a refund"), label with the **primary** intent (the main action the customer needs). Note the secondary intent in `notes`.
- **Complaint vs. specific issue:** If a customer is complaining but also has a specific actionable request (refund, tracking), label the specific issue, not "Complaint/Frustration." Reserve "Complaint/Frustration" for pure rants with no clear ask.
- **Non-English messages:** Label based on content regardless of language. Note the language in `notes`.

### Escalation Label Guidelines

**Escalate = True when:**
- Customer requests money back / refund / reimbursement (requires account verification)
- Customer shares PII publicly (order number, email, phone)
- Customer mentions legal action, safety concerns
- Issue requires account-level access that can't be resolved via public tweet
- Customer is extremely angry / threatening
- The issue is complex or multi-step

**Escalate = False when:**
- Question can be answered with a public link or general information
- Customer says thanks / gives praise
- Simple troubleshooting that can be conveyed in a tweet
- General product/service questions

---

## Labelling Process

1. **Pass 1:** All 200 examples labelled in a single session with consistent focus.
2. **Consistency check:** 30 randomly selected examples re-labelled 48+ hours later.
   - Intra-annotator Cohen's κ: _(to be computed after re-labelling)_
3. **Ambiguous cases:** Documented below with reasoning.

---

## Notable Ambiguous Cases

_(To be filled during labelling)_

| Example ID | Customer Text (truncated) | Ambiguity | Decision | Reasoning |
|------------|--------------------------|-----------|----------|-----------|
| | | | | |

---

## Known Limitations

- **Single annotator:** All labels come from one person. Inter-annotator agreement cannot be measured.
- **Heuristic stratification:** The initial bucket assignment uses keywords, which may misclassify some examples — but this only affects sampling proportions, not the final hand-applied labels.
- **Sampling bias toward English:** Edge case detection for non-English text relies on a small set of language markers and may miss some non-English messages.
