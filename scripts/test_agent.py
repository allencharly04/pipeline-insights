"""Standalone CLI test for the Groq agent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import run_agent


QUESTIONS = [
    "What's our overall win rate?",
    "Which industry has the lowest win rate?",
    "Show me the top 5 sales reps by won revenue.",
    "How many deals are stuck in negotiation right now?",
    "Why are we losing deals?",
]


for q in QUESTIONS:
    print(f"\n{'=' * 70}")
    print(f"Q: {q}")
    print("=" * 70)
    result = run_agent(q)
    if result["tool_used"]:
        print(f"[Tool used: {result['tool_used']}({result['tool_args']})]")
    print(f"\nA: {result['answer']}")