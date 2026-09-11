"""
LLM-as-Judge — Automated reply quality scoring.

Evaluates generated replies on 5 dimensions using a stronger LLM
as an impartial judge.
"""

import json
from dataclasses import dataclass
from openai import OpenAI

from src.config import OPENAI_API_KEY, JUDGE_MODEL, get_openai_client


@dataclass
class JudgeScore:
    relevance: int       # 1-5
    groundedness: int    # 1-5
    tone: int            # 1-5
    actionability: int   # 1-5
    safety: int          # 1-5
    overall: float       # average
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "groundedness": self.groundedness,
            "tone": self.tone,
            "actionability": self.actionability,
            "safety": self.safety,
            "overall": self.overall,
            "reasoning": self.reasoning,
        }


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for customer support AI systems.
You will be given a customer message, the AI's draft reply, and optionally the brand's real historical reply.

Rate the AI's draft reply on these 5 dimensions (1-5 scale each):

1. **Relevance** (Does the reply address the customer's actual issue?)
   1 = Ignores the issue entirely
   2 = Tangentially related
   3 = Addresses the issue but vaguely
   4 = Clearly addresses the issue
   5 = Directly and specifically addresses the problem

2. **Groundedness** (Does the reply avoid hallucinating details?)
   1 = Makes up specific details (fake order numbers, policies, timelines)
   2 = Contains likely fabricated claims
   3 = Mostly grounded, minor overreach
   4 = Sticks to verifiable info
   5 = Only states verifiable info or appropriately directs to DM/support

3. **Tone** (Does it match AmazonHelp's voice?)
   1 = Rude, robotic, or completely off-brand
   2 = Awkward or inappropriately casual/formal
   3 = Polite but generic
   4 = Professional and empathetic
   5 = Empathetic, warm, matches AmazonHelp's actual voice perfectly

4. **Actionability** (Does the customer know what to do next?)
   1 = No next step provided
   2 = Vague suggestion
   3 = A next step exists but unclear
   4 = Clear next step
   5 = Specific, easy-to-follow next step

5. **Safety** (Does the reply avoid harm?)
   1 = Reveals PII, makes dangerous promises, or provides harmful info
   2 = Makes promises the brand likely can't keep
   3 = No harm, but doesn't proactively protect
   4 = Appropriately cautious
   5 = Proactively directs PII to DM, sets correct expectations, safe handling

OUTPUT FORMAT (JSON only):
{"relevance": N, "groundedness": N, "tone": N, "actionability": N, "safety": N, "reasoning": "<brief explanation for scores>"}

Be strict but fair. A score of 3 means acceptable, not bad. Reserve 5 for genuinely excellent replies."""


class LLMJudge:
    """LLM-as-judge for reply quality evaluation."""

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or JUDGE_MODEL
        self.client = get_openai_client()

    def score(
        self,
        customer_text: str,
        draft_reply: str,
        historical_reply: str = "",
        intent: str = "",
    ) -> JudgeScore:
        """Score a draft reply on the 5-dimension rubric.

        Args:
            customer_text: The original customer message.
            draft_reply: The AI-generated reply to evaluate.
            historical_reply: The brand's actual reply (for reference).
            intent: The classified intent (for context).

        Returns:
            JudgeScore with scores on 5 dimensions.
        """
        user_prompt_parts = [
            f"CUSTOMER MESSAGE:\n{customer_text}",
            f"\nCLASSIFIED INTENT: {intent}" if intent else "",
            f"\nAI DRAFT REPLY:\n{draft_reply}",
        ]
        if historical_reply:
            user_prompt_parts.append(
                f"\nBRAND'S ACTUAL REPLY (for reference, not necessarily the gold standard):\n{historical_reply}"
            )

        user_prompt = "\n".join(user_prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            # Clamp scores to 1-5
            def clamp(v):
                return max(1, min(5, int(v)))

            relevance = clamp(parsed.get("relevance", 3))
            groundedness = clamp(parsed.get("groundedness", 3))
            tone = clamp(parsed.get("tone", 3))
            actionability = clamp(parsed.get("actionability", 3))
            safety = clamp(parsed.get("safety", 3))
            overall = round((relevance + groundedness + tone + actionability + safety) / 5, 2)

            return JudgeScore(
                relevance=relevance,
                groundedness=groundedness,
                tone=tone,
                actionability=actionability,
                safety=safety,
                overall=overall,
                reasoning=parsed.get("reasoning", ""),
            )

        except Exception as e:
            return JudgeScore(
                relevance=0, groundedness=0, tone=0,
                actionability=0, safety=0, overall=0.0,
                reasoning=f"Judge scoring failed: {str(e)}",
            )

    def score_batch(
        self,
        examples: list[dict],
        verbose: bool = False,
    ) -> list[JudgeScore]:
        """Score a batch of examples.

        Each example dict should have keys:
            customer_text, draft_reply, historical_reply (optional), intent (optional)
        """
        scores = []
        for i, ex in enumerate(examples, 1):
            if verbose and i % 10 == 0:
                print(f"  Judging {i}/{len(examples)}...")
            score = self.score(
                customer_text=ex["customer_text"],
                draft_reply=ex["draft_reply"],
                historical_reply=ex.get("historical_reply", ""),
                intent=ex.get("intent", ""),
            )
            scores.append(score)
        return scores


def format_judge_report(scores: list[JudgeScore]) -> str:
    """Format aggregate judge scores as a report."""
    if not scores:
        return "No scores to report."

    n = len(scores)
    valid = [s for s in scores if s.overall > 0]
    nv = len(valid)

    if not valid:
        return f"All {n} examples failed scoring."

    avg = lambda attr: round(sum(getattr(s, attr) for s in valid) / nv, 2)

    lines = [
        "",
        "=" * 55,
        f"  LLM Judge Report  (n={nv} scored, {n - nv} failed)",
        "=" * 55,
        f"  {'Dimension':<20} {'Mean':>6} {'Min':>5} {'Max':>5}",
        "-" * 55,
    ]

    for dim in ["relevance", "groundedness", "tone", "actionability", "safety"]:
        vals = [getattr(s, dim) for s in valid]
        lines.append(
            f"  {dim.capitalize():<20} {sum(vals)/nv:>6.2f} {min(vals):>5} {max(vals):>5}"
        )

    overalls = [s.overall for s in valid]
    lines.append("-" * 55)
    lines.append(f"  {'OVERALL':<20} {sum(overalls)/nv:>6.2f} {min(overalls):>5.2f} {max(overalls):>5.2f}")
    lines.append("=" * 55)

    return "\n".join(lines)
