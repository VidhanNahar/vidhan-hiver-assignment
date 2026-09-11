"""
Agent Orchestrator — Ties all pipeline components together.

This is the main entry point for processing a customer message
through the full pipeline: classify → retrieve → respond → escalate.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

from src.pipeline.classifier import IntentClassifier, ClassificationResult
from src.pipeline.retriever import Retriever, RetrievedThread
from src.pipeline.responder import ReplyGenerator, GeneratedReply
from src.pipeline.escalation import EscalationEngine, EscalationResult


@dataclass
class AgentResponse:
    """Complete response from the AI support agent."""
    customer_text: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    draft_reply: str
    grounding_note: str
    escalate: bool
    escalation_reason: str
    escalation_confidence: float
    escalation_triggered_by: str
    retrieved_examples: list[dict]
    processing_time_ms: float

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty_print(self) -> str:
        """Format the response for human readability."""
        esc_status = "🚨 ESCALATE" if self.escalate else "✅ AUTO-HANDLE"
        lines = [
            "=" * 60,
            f"  CUSTOMER: {self.customer_text[:100]}{'...' if len(self.customer_text) > 100 else ''}",
            "-" * 60,
            f"  Intent:     {self.intent} ({self.intent_confidence:.0%})",
            f"  Decision:   {esc_status}",
            f"  Reason:     {self.escalation_reason[:100]}",
            "-" * 60,
            f"  DRAFT REPLY: {self.draft_reply}",
            "-" * 60,
            f"  Processed in {self.processing_time_ms:.0f}ms",
            "=" * 60,
        ]
        return "\n".join(lines)


class SupportAgent:
    """Main AI support agent orchestrator."""

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[Retriever] = None,
        responder: Optional[ReplyGenerator] = None,
        escalation_engine: Optional[EscalationEngine] = None,
    ):
        print("[Agent] Initializing components...")
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or Retriever()
        self.responder = responder or ReplyGenerator()
        self.escalation_engine = escalation_engine or EscalationEngine()
        print("[Agent] Ready.")

    def process(self, customer_text: str) -> AgentResponse:
        """Process a customer message through the full pipeline.

        Steps:
            1. Classify intent
            2. Retrieve similar historical interactions
            3. Generate grounded reply
            4. Decide escalation

        Args:
            customer_text: The incoming customer message.

        Returns:
            AgentResponse with all outputs.
        """
        start = time.time()

        # Step 1: Classify
        classification = self.classifier.classify(customer_text)

        # Step 2: Retrieve
        retrieved = self.retriever.retrieve(
            customer_text,
            intent=classification.intent,
        )

        # Step 3: Generate reply
        reply = self.responder.generate(
            customer_text, retrieved, classification.intent
        )

        # Step 4: Escalation decision
        escalation = self.escalation_engine.decide(
            customer_text, classification, reply.reply
        )

        elapsed_ms = (time.time() - start) * 1000

        return AgentResponse(
            customer_text=customer_text,
            intent=classification.intent,
            intent_confidence=classification.confidence,
            intent_reasoning=classification.reasoning,
            draft_reply=reply.reply,
            grounding_note=reply.grounding_note,
            escalate=escalation.escalate,
            escalation_reason=escalation.reason,
            escalation_confidence=escalation.confidence,
            escalation_triggered_by=escalation.triggered_by,
            retrieved_examples=[
                {
                    "customer_text": r.customer_text[:200],
                    "brand_reply": r.brand_reply[:200],
                    "similarity": r.similarity_score,
                }
                for r in retrieved
            ],
            processing_time_ms=elapsed_ms,
        )

    def process_batch(self, messages: list[str], verbose: bool = False) -> list[AgentResponse]:
        """Process a batch of customer messages."""
        results = []
        for i, msg in enumerate(messages, 1):
            if verbose:
                print(f"  Processing {i}/{len(messages)}...")
            result = self.process(msg)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# CLI Demo
# ---------------------------------------------------------------------------

def main():
    """Interactive CLI demo."""
    import sys

    agent = SupportAgent()

    print("\n" + "=" * 60)
    print("  AmazonHelp AI Support Agent — Interactive Demo")
    print("  Type a customer message, or 'quit' to exit.")
    print("=" * 60 + "\n")

    # Run some demo messages first
    demo_messages = [
        "I ordered a laptop 5 days ago and it still hasn't shipped. What's going on?",
        "My account was hacked and I can't reset my password. Please help!!",
        "I want a refund for order 123-4567890-1234567, the item arrived broken.",
        "Thanks for the quick help last time, you guys are great!",
    ]

    print("--- Demo Messages ---\n")
    for msg in demo_messages:
        response = agent.process(msg)
        print(response.pretty_print())
        print()

    # Interactive mode
    print("\n--- Interactive Mode ---\n")
    while True:
        try:
            user_input = input("Customer message> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = agent.process(user_input)
        print(response.pretty_print())
        print()


if __name__ == "__main__":
    main()
