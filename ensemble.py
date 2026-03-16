import asyncio
import os
import sys
import time

import dspy
from dotenv import load_dotenv
from dspy.primitives import PythonInterpreter

from main import AnalyzeTranscripts, load_transcripts

load_dotenv()

NUM_ENSEMBLE_RUNS = 3
MAX_ITERATIONS = 15
MAX_LLM_CALLS = 200


class AggregateAnswers(dspy.Signature):
    """You are synthesizing multiple independent research analyses of the same dataset.
    Each analysis answered the same question independently. Compare all answers,
    identify consensus themes, and produce a single unified answer.
    For counts that vary across answers, report the median and range."""

    question: str = dspy.InputField(desc="The research question that was analyzed")
    all_answers: list[str] = dspy.InputField(desc="All independent analysis answers, one per run")
    answer: str = dspy.OutputField(desc="Unified consensus answer with themes, counts, and confidence notes")


async def run_single(run_id: int, transcripts: str, question: str) -> dict | None:
    """Run a single RLM instance with its own pre-warmed interpreter."""
    print(f"[Run {run_id}] Starting...")
    start = time.time()
    interpreter = PythonInterpreter()
    interpreter.start()
    try:
        rlm = dspy.RLM(
            AnalyzeTranscripts,
            max_iterations=MAX_ITERATIONS,
            max_llm_calls=MAX_LLM_CALLS,
            verbose=True,
            interpreter=interpreter,
        )
        result = await rlm.acall(transcripts=transcripts, question=question)
        duration = time.time() - start
        print(f"[Run {run_id}] Completed in {duration:.1f}s")
        return {
            "run_id": run_id,
            "answer": result.answer,
            "duration": duration,
        }
    except Exception as e:
        duration = time.time() - start
        print(f"[Run {run_id}] Failed after {duration:.1f}s: {e}")
        return None
    finally:
        interpreter.shutdown()


async def run_ensemble(transcripts: str, question: str) -> None:
    """Run N parallel RLM instances and aggregate results via ChainOfThought."""
    print(f"Launching {NUM_ENSEMBLE_RUNS} parallel RLM runs...")
    print("=" * 60)

    start = time.time()

    tasks = [
        run_single(i + 1, transcripts, question)
        for i in range(NUM_ENSEMBLE_RUNS)
    ]
    raw_results = await asyncio.gather(*tasks)

    successful = [r for r in raw_results if r is not None]
    failed_count = NUM_ENSEMBLE_RUNS - len(successful)

    print("\n" + "=" * 60)
    print(f"Completed: {len(successful)}/{NUM_ENSEMBLE_RUNS} runs succeeded")
    if failed_count:
        print(f"Discarded: {failed_count} failed run(s)")

    if not successful:
        print("ERROR: All runs failed. Cannot produce an answer.")
        sys.exit(1)

    # Print individual run answers
    for r in successful:
        print(f"\n--- Run {r['run_id']} ({r['duration']:.1f}s) ---")
        print(r["answer"][:300] + ("..." if len(str(r["answer"])) > 300 else ""))

    # Aggregate via ChainOfThought
    print("\n" + "=" * 60)
    print("Aggregating results via ChainOfThought...")

    all_answers = [r["answer"] for r in successful]

    aggregator = dspy.ChainOfThought(AggregateAnswers, temperature=1.0)
    try:
        final = aggregator(question=question, all_answers=all_answers)
    except Exception as e:
        duration = time.time() - start
        print(f"\nAggregation failed: {e}")
        print("Falling back to first successful run's answer.")
        final_answer = successful[0]["answer"]
        print("\n" + "=" * 60)
        print("ENSEMBLE ANSWER (fallback — single run):")
        print("=" * 60)
        print(final_answer)
        _print_cost_summary(duration, len(successful))
        return

    duration = time.time() - start

    print("\n" + "=" * 60)
    print("ENSEMBLE ANSWER:")
    print("=" * 60)
    print(final.answer)

    _print_cost_summary(duration, len(successful))


def _print_cost_summary(duration: float, num_successful: int) -> None:
    lm = dspy.settings.lm
    total_cost = sum(entry.get("cost") or 0 for entry in lm.history)
    total_tokens = sum(
        entry.get("usage", {}).get("total_tokens", 0) for entry in lm.history
    )
    print(
        f"\n--- Cost: ${total_cost:.6f} | Tokens: {total_tokens:,} "
        f"| LLM calls: {len(lm.history)} | Runs: {num_successful}/{NUM_ENSEMBLE_RUNS} "
        f"| Duration: {duration:.1f}s ---"
    )


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run ensemble.py "<your question about the transcripts>"')
        sys.exit(1)

    question = sys.argv[1]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    lm = dspy.LM("openai/gpt-5-mini", api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    transcripts = load_transcripts()

    asyncio.run(run_ensemble(transcripts, question))


if __name__ == "__main__":
    main()
