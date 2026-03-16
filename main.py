import json
import os
import sys
import time
from pathlib import Path

import dspy
from dotenv import load_dotenv
from litellm.exceptions import ContextWindowExceededError
from dspy.primitives import PythonInterpreter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"


class AnalyzeTranscripts(dspy.Signature):
    """You are an expert qualitative researcher analyzing interview transcripts.
    IMPORTANT: You must process ALL transcripts in the dataset — never sample or subset.
    Before calling SUBMIT, print the total number of transcripts processed and verify it matches the full dataset.
    When counting or classifying, report exact counts with supporting evidence."""

    transcripts: str = dspy.InputField(desc="Full text of all interview transcripts")
    question: str = dspy.InputField(desc="Research question to answer about the transcripts")
    answer: str = dspy.OutputField(desc="Detailed answer with exact counts, supporting evidence, and methodology notes")


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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    lm = dspy.LM("openai/gpt-5-mini", api_key=api_key,cache=False)
    dspy.configure(lm=lm)

    interpreter = PythonInterpreter()
    interpreter.start()

    transcripts = load_transcripts()

    rlm = dspy.RLM(
        AnalyzeTranscripts,
        max_iterations=15,
        max_llm_calls=200,
        verbose=True,
        interpreter=interpreter,
    )

    max_retries = 3
    start = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            result = rlm(transcripts=transcripts, question=question)
            break
        except ContextWindowExceededError:
            print(f"\n[Attempt {attempt}/{max_retries}] Context window exceeded — retrying with fresh RLM state...")
            if attempt == max_retries:
                duration = time.time() - start
                print("\n" + "=" * 60)
                print("ERROR: Context window exceeded on all retries.")
                print(f"\n--- Duration: {duration:.1f}s ---")
                sys.exit(1)
            interpreter = PythonInterpreter()
            interpreter.start()
            rlm = dspy.RLM(
                AnalyzeTranscripts,
                max_iterations=15,
                max_llm_calls=200,
                verbose=True,
                interpreter=interpreter,
            )

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
