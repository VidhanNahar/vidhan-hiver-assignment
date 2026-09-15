"""
Reply Generator — Grounded response drafting using RAG context.

Generates replies in AmazonHelp's voice, grounded in historically
similar brand responses. Enforces safety constraints.
"""

import json
import re
from dataclasses import dataclass
from openai import OpenAI

from src.config import OPENAI_API_KEY, PIPELINE_MODEL, get_openai_client
from src.pipeline.retriever import RetrievedThread


@dataclass
class GeneratedReply:
    reply: str
    grounding_note: str  # which retrieved examples influenced the reply


RE_SIG = re.compile(r"(\s*\^[A-Za-z]{2,3})$")
RE_ORDER_ID = re.compile(r"\d{3}-\d{7}-\d{7}")
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
RE_PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")


def truncate_tweet(text: str, max_len: int = 280) -> str:
    """Truncate tweet cleanly without slicing signatures or words arbitrarily."""
    if len(text) <= max_len:
        return text

    # Check for trailing agent signature like ^JM or ^CS
    sig_match = RE_SIG.search(text)
    suffix = ""
    if sig_match:
        suffix = sig_match.group(1)
        text = text[:sig_match.start()].strip()

    available = max_len - len(suffix) - 3  # room for '...'
    if available <= 0:
        return text[:max_len]

    truncated = text[:available]
    # Break on last whitespace to avoid chopping words or links mid-character
    last_space = truncated.rfind(" ")
    if last_space > available // 2:
        truncated = truncated[:last_space]

    return f"{truncated.rstrip()}...{suffix}"


def sanitize_reply(reply: str) -> str:
    """Validate and sanitize generated reply against PII leaks and character bounds."""
    # Redact any accidental order ID / PII echoes
    reply = RE_ORDER_ID.sub("[ORDER_ID]", reply)
    reply = RE_EMAIL.sub("[EMAIL]", reply)
    reply = RE_PHONE.sub("[PHONE]", reply)

    # Ensure signature exists
    if not RE_SIG.search(reply):
        reply = f"{reply.rstrip()} ^CS"

    # Enforce tweet length with signature preservation
    if len(reply) > 280:
        reply = truncate_tweet(reply, max_len=280)

    return reply


SYSTEM_PROMPT = """You are AmazonHelp, the official Amazon customer support account on Twitter.
Draft a helpful, empathetic reply to the customer tweet below.

RULES:
1. Match AmazonHelp's real tone: friendly, professional, empathetic, uses the customer's name when available.
2. Use the HISTORICAL EXAMPLES below as references for tone, structure, and typical resolution approaches. Do NOT copy them verbatim.
3. DO NOT make up order numbers, tracking links, specific account details, or policies you're unsure about.
4. If the issue requires account access or private info, direct the customer to DM.
5. Keep the reply under 280 characters (Twitter limit).
6. End with a team member initial like ^XX (e.g., ^JM) for authenticity.
7. If you genuinely cannot help with just the tweet text, ask a clarifying question.

OUTPUT FORMAT:
Respond with a JSON object ONLY:
{"reply": "<your drafted reply>", "grounding_note": "<brief note on which historical examples influenced your response style>"}"""


def _build_user_prompt(
    customer_text: str,
    retrieved: list[RetrievedThread],
    intent: str,
) -> str:
    """Build the user message with retrieved context."""
    lines = [
        f"DETECTED INTENT: {intent}",
        "",
        "=== HISTORICAL EXAMPLES (similar AmazonHelp interactions) ===",
        "",
    ]

    for i, r in enumerate(retrieved, 1):
        lines.append(f"Example {i} (similarity: {r.similarity_score:.2f}):")
        lines.append(f"  Customer: {r.customer_text[:200]}")
        lines.append(f"  AmazonHelp: {r.brand_reply[:200]}")
        lines.append("")

    lines.extend([
        "=== CURRENT CUSTOMER MESSAGE ===",
        "",
        customer_text,
    ])

    return "\n".join(lines)


class ReplyGenerator:
    """Generates grounded replies using RAG context."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or PIPELINE_MODEL
        self.client = get_openai_client(api_key=api_key)

    def generate(
        self,
        customer_text: str,
        retrieved: list[RetrievedThread],
        intent: str,
    ) -> GeneratedReply:
        """Generate a reply grounded in retrieved historical responses.

        Args:
            customer_text: The incoming customer message.
            retrieved: List of similar historical threads from the retriever.
            intent: The classified intent category.

        Returns:
            GeneratedReply with the drafted reply and grounding note.
        """
        user_prompt = _build_user_prompt(customer_text, retrieved, intent)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Slight creativity but mostly grounded
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            reply = parsed.get("reply", "").strip()
            grounding_note = parsed.get("grounding_note", "").strip()

            reply = sanitize_reply(reply)

            return GeneratedReply(
                reply=reply,
                grounding_note=grounding_note,
            )

        except Exception as e:
            return GeneratedReply(
                reply="We're sorry to hear about this. Please DM us your details so we can help! ^CS",
                grounding_note=f"Fallback reply due to error: {str(e)}",
            )
