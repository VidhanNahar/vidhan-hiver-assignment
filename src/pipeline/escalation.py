"""
Escalation Engine — Hybrid rule + LLM-based triage.

Decides whether a customer message should be auto-handled or
escalated to a human agent, with a stated reason.

Uses hard rules for clear-cut cases (safety, PII, refunds)
and LLM-based soft decision for ambiguous cases.
"""

import json
import re
from dataclasses import dataclass
from openai import OpenAI

from src.config import OPENAI_API_KEY, PIPELINE_MODEL, get_openai_client
from src.pipeline.classifier import ClassificationResult


@dataclass
class EscalationResult:
    escalate: bool
    reason: str
    confidence: float
    triggered_by: str  # "hard_rule" or "llm_decision"


# ---------------------------------------------------------------------------
# Hard Rules (always escalate — no LLM needed)
# ---------------------------------------------------------------------------

# Legal threats
RE_LEGAL = re.compile(
    r"\b(lawyer|lawsuit|sue|suing|legal action|attorney|court|litigation|class action)\b",
    re.IGNORECASE,
)

# Safety / self-harm
RE_SAFETY = re.compile(
    r"\b(kill myself|suicide|self.harm|end my life|hurt myself|die)\b",
    re.IGNORECASE,
)

# PII in public tweet (order IDs, emails, phone numbers)
RE_ORDER_ID = re.compile(r"\d{3}-\d{7}-\d{7}")
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
RE_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

# Monetary request keywords
RE_MONETARY = re.compile(
    r"\b(refund|money back|reimburse|compensation|charged twice|overcharged|credit back|want my money)\b",
    re.IGNORECASE,
)


def _check_hard_rules(customer_text: str, classification: ClassificationResult) -> EscalationResult | None:
    """Check hard rules that always trigger escalation.

    Returns EscalationResult if a rule matches, None otherwise.
    """
    text = customer_text

    # Rule 1: Legal threats
    if RE_LEGAL.search(text):
        return EscalationResult(
            escalate=True,
            reason="Customer mentions legal action — requires human review.",
            confidence=1.0,
            triggered_by="hard_rule",
        )

    # Rule 2: Safety concerns
    if RE_SAFETY.search(text):
        return EscalationResult(
            escalate=True,
            reason="Potential safety concern detected — immediate human attention required.",
            confidence=1.0,
            triggered_by="hard_rule",
        )

    # Rule 3: PII exposure in public tweet
    if RE_ORDER_ID.search(text) or RE_EMAIL.search(text) or RE_PHONE.search(text):
        return EscalationResult(
            escalate=True,
            reason="Customer shared personal information (order ID, email, or phone) publicly — needs private follow-up to protect PII.",
            confidence=1.0,
            triggered_by="hard_rule",
        )

    # Rule 4: Monetary requests (refunds require account verification)
    if RE_MONETARY.search(text):
        return EscalationResult(
            escalate=True,
            reason="Customer requests monetary action (refund/reimbursement) — requires account verification by a human agent.",
            confidence=0.95,
            triggered_by="hard_rule",
        )

    # Rule 5: Low classifier confidence
    if classification.confidence < 0.4:
        return EscalationResult(
            escalate=True,
            reason=f"Intent classification confidence is low ({classification.confidence:.2f}) — human should review to ensure correct handling.",
            confidence=0.85,
            triggered_by="hard_rule",
        )

    return None


# ---------------------------------------------------------------------------
# LLM Soft Decision
# ---------------------------------------------------------------------------

ESCALATION_SYSTEM_PROMPT = """You are a customer support triage engine for AmazonHelp.

Given a customer message, its classified intent, and a draft reply, decide whether this interaction should be:
- **AUTO-HANDLED**: The draft reply is sufficient, safe, and addresses the customer's need.
- **ESCALATED**: A human agent should handle this because the issue is complex, sensitive, or the draft reply is inadequate.

Consider:
1. Is the customer angry, frustrated, or using threatening language?
2. Does the issue require access to the customer's account or private information?
3. Is the draft reply generic or does it actually help?
4. Could a wrong auto-response make things worse?
5. Is the customer a repeat complainer or first-time requester? (from context)

OUTPUT FORMAT (JSON only):
{"escalate": true/false, "reason": "<clear explanation>", "confidence": 0.0-1.0}"""


class EscalationEngine:
    """Hybrid escalation decision engine."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or PIPELINE_MODEL
        self.client = get_openai_client()

    def decide(
        self,
        customer_text: str,
        classification: ClassificationResult,
        draft_reply: str = "",
    ) -> EscalationResult:
        """Decide whether to escalate or auto-handle.

        Args:
            customer_text: The incoming customer message.
            classification: The intent classification result.
            draft_reply: The generated draft reply (if available).

        Returns:
            EscalationResult with the decision and reasoning.
        """
        # Check hard rules first
        hard_result = _check_hard_rules(customer_text, classification)
        if hard_result is not None:
            return hard_result

        # LLM soft decision for remaining cases
        return self._llm_decide(customer_text, classification, draft_reply)

    def _llm_decide(
        self,
        customer_text: str,
        classification: ClassificationResult,
        draft_reply: str,
    ) -> EscalationResult:
        """Use LLM for soft escalation decision."""
        user_prompt = (
            f"CUSTOMER MESSAGE:\n{customer_text}\n\n"
            f"CLASSIFIED INTENT: {classification.intent} (confidence: {classification.confidence:.2f})\n\n"
            f"DRAFT REPLY:\n{draft_reply}\n\n"
            f"Should this be AUTO-HANDLED or ESCALATED?"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ESCALATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            return EscalationResult(
                escalate=bool(parsed.get("escalate", True)),
                reason=parsed.get("reason", "No reason provided."),
                confidence=float(parsed.get("confidence", 0.5)),
                triggered_by="llm_decision",
            )

        except Exception as e:
            # On failure, escalate to be safe
            return EscalationResult(
                escalate=True,
                reason=f"Escalation decision failed ({str(e)}) — defaulting to escalate for safety.",
                confidence=0.0,
                triggered_by="error_fallback",
            )
