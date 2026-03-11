import json
import os
import sys
import time
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.primitives import PythonInterpreter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"


def load_transcripts() -> str:
    """Load all JSON transcript files from the data directory and format as a single text block for RLM."""
    json_files = sorted(DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {DATA_DIR}")
        sys.exit(1)

    calls = []
    for json_file in json_files:
        with open(json_file) as f:
            calls.extend(json.load(f))

    lines = []
    for call in calls:
        call_id = call["id"]
        attrs = {a["label"]: a["value"] for a in call.get("attributes", [])}
        attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(f"\n=== Call {call_id} | {attrs_str} ===")
        for msg in call["messages"]:
            role = msg["role"].upper()
            lines.append(f"  [{role}] {msg['message']}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run main.py \"<your question about the transcripts>\"")
        sys.exit(1)

    question = sys.argv[1]

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set in .env file")
        sys.exit(1)

    lm = dspy.LM("gemini/gemini-3-flash-preview", api_key=api_key)
    dspy.configure(lm=lm)

    interpreter = PythonInterpreter()
    interpreter.start()

    transcripts = load_transcripts()

    rlm = dspy.RLM(
        "transcripts, question -> answer",
        max_iterations=100,
        max_llm_calls=200,
        verbose=True,
        interpreter=interpreter,
    )

    start = time.time()
    result = rlm(transcripts=transcripts, question=question)
    duration = time.time() - start
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result.answer)

    total_cost = sum(entry.get("cost") or 0 for entry in lm.history)
    total_tokens = sum(entry.get("usage", {}).get("total_tokens", 0) for entry in lm.history)
    print(f"\n--- Cost: ${total_cost:.6f} | Tokens: {total_tokens:,} | LLM calls: {len(lm.history)} | Duration: {duration:.1f}s ---")


if __name__ == "__main__":
    main()
