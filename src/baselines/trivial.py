"""
Baseline 1 — Trivial: Keyword matching + canned replies.

The simplest possible system. Classifies intent via keyword lookup
and returns a fixed canned reply per intent. Always escalates.
"""

import re
from dataclasses import dataclass


@dataclass
class TrivialResponse:
    customer_text: str
    intent: str
    draft_reply: str
    escalate: bool
    escalation_reason: str


# Keyword patterns (checked in priority order)
KEYWORD_RULES = [
    (
        "Refund / Return",
        re.compile(r"\b(refund|money back|return|reimburse|charged twice|overcharged|credit back)\b", re.I),
    ),
    (
        "Account / Login Issue",
        re.compile(r"\b(can'?t log.?in|password|locked out|hacked|account blocked|suspended|can'?t access|verification)\b", re.I),
    ),
    (
        "Order Status / Tracking",
        re.compile(r"\b(where is my order|tracking|when will|order status|hasn'?t shipped|hasn'?t arrived|still processing|not shipped|has not shipped|still says processing|where'?s my|order.{0,5}ship|order.{0,5}deliver)\b", re.I),
    ),
    (
        "Delivery Problem",
        re.compile(r"\b(not delivered|lost package|wrong address|stolen|damaged|broken|says delivered|never received|wrong item)\b", re.I),
    ),
    (
        "Prime / Subscription",
        re.compile(r"\b(prime|subscription|membership|prime video|prime music|free trial|cancel prime)\b", re.I),
    ),
    (
        "Complaint / Frustration",
        re.compile(r"\b(worst|terrible|horrible|disgusting|unacceptable|ridiculous|pathetic|furious|waste of time|awful)\b", re.I),
    ),
    (
        "Product / Service Question",
        re.compile(r"\b(how do i|how to|does it|compatible|difference between|can i use)\b", re.I),
    ),
]

# One canned reply per intent
CANNED_REPLIES = {
    "Order Status / Tracking": "We're sorry for the concern. Please check your order status here: https://www.amazon.com/gp/your-account/order-history. If you need further help, DM us your details. ^CS",
    "Refund / Return": "We're sorry to hear about this. Please DM us your order details so we can look into a refund or return for you. ^CS",
    "Delivery Problem": "We're sorry for the trouble with your delivery. Please DM us your order number so we can investigate. ^CS",
    "Account / Login Issue": "We understand how frustrating that must be. Please try resetting your password here, or DM us for further assistance. ^CS",
    "Product / Service Question": "Great question! Please DM us or visit our Help page at https://www.amazon.com/help for more details. ^CS",
    "Prime / Subscription": "We'd be happy to help with your Prime membership. Please DM us your account details so we can look into this. ^CS",
    "Complaint / Frustration": "We're truly sorry for your experience. Your feedback matters to us. Please DM us so we can make this right. ^CS",
    "Other / Miscellaneous": "Thank you for reaching out! Please DM us your details and we'll be happy to help. ^CS",
}


class TrivialBaseline:
    """Keyword-based intent classification + canned replies."""

    def process(self, customer_text: str) -> TrivialResponse:
        # Classify by keyword matching
        intent = "Other / Miscellaneous"
        for intent_name, pattern in KEYWORD_RULES:
            if pattern.search(customer_text):
                intent = intent_name
                break

        # Canned reply
        reply = CANNED_REPLIES.get(intent, CANNED_REPLIES["Other / Miscellaneous"])

        return TrivialResponse(
            customer_text=customer_text,
            intent=intent,
            draft_reply=reply,
            escalate=True,  # Always escalate — trivially safe
            escalation_reason="Trivial baseline always escalates to human agent.",
        )
