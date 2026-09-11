"""
Baseline 2 — Simple: Zero-shot LLM (no RAG, no few-shot examples).

Uses the same LLM as the main system but without retrieved context,
few-shot examples, or hard escalation rules.
Isolates the value added by RAG and prompt engineering.
"""

import json
from dataclasses import dataclass
from openai import OpenAI

from src.config import OPENAI_API_KEY, PIPELINE_MODEL, get_openai_client
from src.pipeline.classifier import VALID_INTENTS


@dataclass
class SimpleResponse:
    customer_text: str
    intent: str
    intent_confidence: float
    draft_reply: str
    escalate: bool
    escalation_reason: str


ZERO_SHOT_SYSTEM_PROMPT = """You are an AI assistant evaluating an incoming Twitter customer support message for AmazonHelp.
Categories:
{}

Tasks:
1. Classify the customer's primary intent into one of the categories above.
2. Draft a concise reply under 280 characters in AmazonHelp's voice ending with ^CS (zero-shot, no retrieved references).
3. Decide whether to auto-handle (false) or escalate to a human (true), with a short reason.

Respond with JSON only:
{{"intent": "<category>", "confidence": 0.0-1.0, "reply": "<draft reply>", "escalate": true/false, "reason": "<brief reason>"}}""".format(
    "\n".join(f"- {intent}" for intent in VALID_INTENTS)
)


class SimpleBaseline:
    """Zero-shot LLM baseline — no RAG, no few-shot, no hard rules."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or PIPELINE_MODEL
        self.client = get_openai_client()

    def process(self, customer_text: str) -> SimpleResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
                    {"role": "user", "content": customer_text},
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content.strip())
            intent = parsed.get("intent", "Other / Miscellaneous")
            if intent not in VALID_INTENTS:
                intent = "Other / Miscellaneous"
            confidence = float(parsed.get("confidence", 0.5))
            reply = parsed.get("reply", "Please DM us your details so we can help. ^CS")
            if len(reply) > 280:
                reply = reply[:277] + "..."
            escalate = bool(parsed.get("escalate", True))
            reason = parsed.get("reason", "No reason provided.")
        except Exception as e:
            intent = "Other / Miscellaneous"
            confidence = 0.0
            reply = "Please DM us your details so we can help. ^CS"
            escalate = True
            reason = f"Error fallback: {str(e)}"

        return SimpleResponse(
            customer_text=customer_text,
            intent=intent,
            intent_confidence=confidence,
            draft_reply=reply,
            escalate=escalate,
            escalation_reason=reason,
        )
