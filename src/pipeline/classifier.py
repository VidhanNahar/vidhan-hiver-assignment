"""
Intent Classifier — Few-shot LLM-based classification.

Classifies customer messages into one of 8 intent categories
using a few-shot prompted LLM.
"""

import json
from dataclasses import dataclass
from openai import OpenAI

from src.config import OPENAI_API_KEY, PIPELINE_MODEL, API_BASE_URL, get_openai_client


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    reasoning: str


# --- Intent taxonomy with descriptions and examples ---
INTENT_TAXONOMY = {
    "Order Status / Tracking": {
        "description": "Customer asks about order status, shipping updates, estimated delivery, or tracking information.",
        "examples": [
            "I ordered 3 days ago and it still says processing, when will it ship?",
            "Can you tell me where my package is? The tracking hasn't updated in 2 days.",
        ],
    },
    "Refund / Return": {
        "description": "Customer wants a refund, return, exchange, or reports being wrongly charged.",
        "examples": [
            "I received a broken item, I need a full refund please.",
            "I was charged twice for the same order, can I get my money back?",
        ],
    },
    "Delivery Problem": {
        "description": "Package not delivered, marked as delivered but not received, wrong item delivered, damaged in transit, stolen, or wrong address.",
        "examples": [
            "It says delivered but I never got the package, someone must have stolen it.",
            "I received completely the wrong item, this isn't what I ordered at all.",
        ],
    },
    "Account / Login Issue": {
        "description": "Customer can't log in, account locked/suspended/hacked, password reset issues, verification problems.",
        "examples": [
            "My account got hacked and now I'm locked out, I can't reset my password.",
            "You suspended my account for no reason, I need it back immediately.",
        ],
    },
    "Product / Service Question": {
        "description": "Customer asks how something works, product compatibility, feature questions, or general inquiries about Amazon services.",
        "examples": [
            "Does Prime include free shipping on all items or just some?",
            "How do I set up my new Echo Dot with my existing Alexa account?",
        ],
    },
    "Prime / Subscription": {
        "description": "Issues related to Prime membership, billing, cancellation, benefits, Prime Video, Prime Music, or free trial.",
        "examples": [
            "I was charged for Prime but I cancelled it last month, why?",
            "I signed up for a free trial and forgot to cancel, can I get a refund?",
        ],
    },
    "Complaint / Frustration": {
        "description": "General dissatisfaction, venting frustration, poor experience rant, threats to leave, or expressions of anger without a specific actionable request.",
        "examples": [
            "Worst customer service ever! I've been on hold for 3 hours and nobody helps.",
            "This is the third time you've messed up my order. I'm done with Amazon.",
        ],
    },
    "Other / Miscellaneous": {
        "description": "Praise, thanks, unrelated messages, messages in other languages that don't clearly fit above categories, or general conversation.",
        "examples": [
            "Thanks for the quick help, you guys are great!",
            "Hey @AmazonHelp do you have any job openings?",
        ],
    },
}

VALID_INTENTS = list(INTENT_TAXONOMY.keys())


def _build_system_prompt() -> str:
    """Build the system prompt with taxonomy and examples."""
    lines = [
        "You are an intent classifier for AmazonHelp customer support tweets.",
        "Classify each customer message into EXACTLY ONE of the following intent categories.",
        "If a message has multiple intents, choose the PRIMARY one (the main thing the customer needs help with).",
        "",
        "=== INTENT CATEGORIES ===",
        "",
    ]

    for intent, info in INTENT_TAXONOMY.items():
        lines.append(f"**{intent}**")
        lines.append(f"Description: {info['description']}")
        for i, ex in enumerate(info["examples"], 1):
            lines.append(f"  Example {i}: \"{ex}\"")
        lines.append("")

    lines.extend([
        "=== OUTPUT FORMAT ===",
        "Respond with a JSON object ONLY (no markdown, no extra text):",
        '{"intent": "<one of the categories above>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}',
        "",
        "Rules:",
        "- confidence should reflect how certain you are (0.5 = uncertain, 0.9+ = very sure)",
        "- If the message is in a non-English language, still classify it based on content",
        "- Always pick the most specific applicable category over 'Other / Miscellaneous'",
    ])

    return "\n".join(lines)


SYSTEM_PROMPT = _build_system_prompt()


class IntentClassifier:
    """Few-shot LLM intent classifier."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or PIPELINE_MODEL
        self.client = get_openai_client()

    def classify(self, customer_text: str) -> ClassificationResult:
        """Classify a customer message into an intent category."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": customer_text},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            intent = parsed.get("intent", "Other / Miscellaneous")
            # Validate intent is in our taxonomy
            if intent not in VALID_INTENTS:
                intent = "Other / Miscellaneous"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = parsed.get("reasoning", "")

            return ClassificationResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            # Fallback on any error
            return ClassificationResult(
                intent="Other / Miscellaneous",
                confidence=0.0,
                reasoning=f"Classification failed: {str(e)}",
            )
