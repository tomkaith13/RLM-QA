import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import dspy
from dotenv import load_dotenv
from dspy.primitives import PythonInterpreter

from build_index import search_transcripts
from main import load_transcripts

load_dotenv()

MAX_LENSES = 5
MAX_ITERATIONS = 25
MAX_LLM_CALLS = 200


# ── DSPy Signatures ──────────────────────────────────────────────────────────


class AnalyzeTranscriptsWithLens(dspy.Signature):
    """You are an expert qualitative researcher analyzing interview transcripts.
    IMPORTANT: You must process ALL transcripts in the dataset — never sample or subset.
    Before calling SUBMIT, print the total number of transcripts processed and verify it matches the full dataset.
    When counting or classifying, report exact counts with supporting evidence.
    You have been assigned a specific analytical lens. Use it to frame your
    analysis and prioritize which patterns to surface first, but do not let it
    blind you to other significant findings. If tools are available, 
    use them strategically in your analysis."""

    transcripts: str = dspy.InputField(desc="Full text of all interview transcripts")
    question: str = dspy.InputField(desc="Research question to answer about the transcripts")
    context: str = dspy.InputField(desc="Dataset summary and your assigned analytical lens for this run")
    answer: str = dspy.OutputField(desc="Detailed answer with exact counts, supporting evidence, and methodology notes")


class GenerateLenses(dspy.Signature):
    """You are a qualitative research methodologist designing analytical lenses
    for parallel independent analyses of interview transcript data.
    Each lens must be meaningfully distinct: vary the focal construct, the
    unit of analysis, or the interpretive frame. Lenses must be specific
    enough to steer an analyst's attention, not generic advice.
    Choose the right NUMBER of lenses for the question's complexity:
    - Simple factual/counting questions need only 1-2 lenses.
    - Moderately complex questions (filtering, cross-tabulation) need 2-3 lenses.
    - Open-ended interpretive/qualitative questions benefit from 3-5 lenses.
    Do not generate more lenses than the question warrants."""

    question: str = dspy.InputField(desc="The research question all runs will answer")
    data_summary: str = dspy.InputField(desc="Structural metadata about the transcript dataset")
    max_lenses: int = dspy.InputField(desc="Maximum number of lenses allowed — generate fewer if the question is simple")
    lenses: list[str] = dspy.OutputField(desc="The appropriate number of analytical lenses (up to max_lenses), each 1-3 sentences")


class AggregateAnswers(dspy.Signature):
    """You are synthesizing multiple independent research analyses of the same dataset.
    Each analysis answered the same question independently. Compare all answers,
    identify consensus themes, and produce a single unified answer.
    For counts that vary across answers, report the median and range."""

    question: str = dspy.InputField(desc="The research question that was analyzed")
    all_answers: list[str] = dspy.InputField(desc="All independent analysis answers, one per run")
    answer: str = dspy.OutputField(desc="Unified consensus answer with themes, counts, and confidence notes")


# ── Planning: data summary ───────────────────────────────────────────────────


def compute_data_summary(calls: list[dict[str, Any]]) -> str:
    """Extract structural metadata from raw call dicts and return a compact summary string."""
    n = len(calls)

    # Collect per-attribute value distributions, and explode JSON array values
    # into individual items for fields like "Which stores..." and "Do you buy..."
    attr_counts: dict[str, Counter] = defaultdict(Counter)
    for call in calls:
        for attr in call.get("attributes", []):
            label = attr.get("label", "").strip()
            value = attr.get("value", "").strip()
            if not label or not value:
                continue
            # If the value looks like a JSON array, count individual items
            if value.startswith("["):
                try:
                    items = json.loads(value)
                    if isinstance(items, list):
                        for item in items:
                            attr_counts[label][str(item).strip()] += 1
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            attr_counts[label][value] += 1

    # Age statistics (numeric)
    ages = []
    for val in attr_counts.get("Age", {}):
        try:
            ages.append(int(val))
        except ValueError:
            pass

    lines = [f"Dataset: {n} transcripts"]

    if len(ages) >= 2:
        q = statistics.quantiles(ages, n=4)
        lines.append(
            f"Ages: min={min(ages)}, max={max(ages)}, "
            f"median={statistics.median(ages):.0f}, "
            f"Q1={q[0]:.0f}, Q3={q[2]:.0f}"
        )
    elif ages:
        lines.append(f"Ages: {ages[0]} (single value)")

    lines.append("Attribute fields and top values:")
    for label, counter in attr_counts.items():
        top = counter.most_common(10)
        top_str = ", ".join(f"{v} ({c})" for v, c in top)
        if len(counter) > 10:
            top_str += f", ... ({len(counter)} total unique)"
        lines.append(f"  {label}: {top_str}")

    # Build a transcript format example from the first call
    if calls:
        sample = calls[0]
        sample_id = sample["id"]
        sample_attrs = {a["label"]: a["value"] for a in sample.get("attributes", [])}
        sample_attrs_str = ", ".join(f"{k}: {v}" for k, v in list(sample_attrs.items())[:3])
        lines.append("")
        lines.append("Transcript format (each transcript follows this structure):")
        lines.append(f"  === Call <call_id> | <comma-separated attributes> ===")
        lines.append(f"    [BOT] <interviewer message>")
        lines.append(f"    [USER] <participant response>")
        lines.append(f"    ... (conversation continues)")
        lines.append(f"  Example header: === Call {sample_id} | {sample_attrs_str}, ... ===")
        lines.append(f"  Attribute fields appear in the header, not as separate lines.")
        lines.append(f"  To extract an attribute (e.g., Gender+), parse the header pipe-delimited section.")

    return "\n".join(lines)


# ── Planning: lens generation ────────────────────────────────────────────────


def generate_lenses(question: str, data_summary: str, max_lenses: int) -> list[str]:
    """Use a single LLM call to generate up to max_lenses tailored analytical lenses."""
    try:
        predictor = dspy.Predict(GenerateLenses)
        result = predictor(
            question=question,
            data_summary=data_summary,
            max_lenses=max_lenses,
        )
        lenses = result.lenses
        if not isinstance(lenses, list):
            lenses = [str(lenses)]
    except Exception as e:
        print(f"Warning: Lens generation failed ({e}), using generic lenses.")
        lenses = []

    # Ensure at least one lens
    if not lenses:
        lenses = ["Analyze all transcripts comprehensively with no specific focal constraint."]

    return lenses[:max_lenses]


def assemble_context(data_summary: str, lens: str) -> str:
    """Combine the data summary and a single lens into the context string for one run."""
    return (
        f"=== DATASET SUMMARY ===\n{data_summary}\n\n"
        f"=== YOUR ANALYTICAL LENS ===\n{lens}"
    )


# ── Ensemble execution ───────────────────────────────────────────────────────


async def run_single(run_id: int, transcripts: str, question: str, context: str, *, use_search: bool = False) -> dict | None:
    """Run a single RLM instance with its own pre-warmed interpreter."""
    print(f"[Run {run_id}] Starting...")
    start = time.time()
    interpreter = PythonInterpreter()
    interpreter.start()
    try:
        tools = [search_transcripts] if use_search else []
        rlm = dspy.RLM(
            AnalyzeTranscriptsWithLens,
            max_iterations=MAX_ITERATIONS,
            max_llm_calls=MAX_LLM_CALLS,
            verbose=True,
            interpreter=interpreter,
            tools=tools,
        )
        result = await rlm.acall(transcripts=transcripts, question=question, context=context)
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


async def run_ensemble(transcripts: str, question: str, contexts: list[str], num_runs: int, *, use_search: bool = False) -> None:
    """Run N parallel RLM instances with per-run contexts and aggregate results."""
    print(f"Launching {num_runs} parallel RLM runs (search={'on' if use_search else 'off'})...")
    print("=" * 60)

    start = time.time()

    tasks = [
        run_single(i + 1, transcripts, question, contexts[i], use_search=use_search)
        for i in range(num_runs)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in raw_results if isinstance(r, dict)]
    failed_count = num_runs - len(successful)

    print("\n" + "=" * 60)
    print(f"Completed: {len(successful)}/{num_runs} runs succeeded")
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
        _print_cost_summary(duration, len(successful), num_runs)
        return

    duration = time.time() - start

    print("\n" + "=" * 60)
    print("ENSEMBLE ANSWER:")
    print("=" * 60)
    print(final.answer)

    _print_cost_summary(duration, len(successful), num_runs)


def _print_cost_summary(duration: float, num_successful: int, num_runs: int) -> None:
    lm = dspy.settings.lm
    total_cost = sum(entry.get("cost") or 0 for entry in lm.history)
    total_tokens = sum(
        entry.get("usage", {}).get("total_tokens", 0) for entry in lm.history
    )
    print(
        f"\n--- Cost: ${total_cost:.6f} | Tokens: {total_tokens:,} "
        f"| LLM calls: {len(lm.history)} | Runs: {num_successful}/{num_runs} "
        f"| Duration: {duration:.1f}s ---"
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run ensemble.py [--use-search-tool] "<your question about the transcripts>"')
        sys.exit(1)

    use_search = "--use-search-tool" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--use-search-tool"]
    if not args:
        print('Usage: uv run ensemble.py [--use-search-tool] "<your question about the transcripts>"')
        sys.exit(1)
    question = args[0]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    lm = dspy.LM("openai/gpt-5-mini", api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    transcripts, calls = load_transcripts()

    # Planning step: compute data summary and generate tailored lenses
    print("=" * 60)
    print("Planning step: analyzing dataset structure...")
    data_summary = compute_data_summary(calls)
    print(data_summary)
    print()

    print(f"Generating analytical lenses (up to {MAX_LENSES})...")
    lenses = generate_lenses(question=question, data_summary=data_summary, max_lenses=MAX_LENSES)
    print(f"  Planner chose {len(lenses)} lenses:")
    for i, lens in enumerate(lenses, 1):
        print(f"  Lens {i}: {lens}")
    print("=" * 60)
    print()

    contexts = [assemble_context(data_summary, lens) for lens in lenses]

    asyncio.run(run_ensemble(transcripts, question, contexts, num_runs=len(lenses), use_search=use_search))


if __name__ == "__main__":
    main()
