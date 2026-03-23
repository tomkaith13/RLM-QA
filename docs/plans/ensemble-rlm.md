# Ensemble RLM Design

## Problem

A single RLM run is non-deterministic. The same question on the same data can produce different strategies (regex vs sub-LM batching), different counts, and occasionally crash due to context window overflow. However, the **thematic findings are consistent** across runs — rankings and directional insights are stable even when exact numbers vary.

## Solution

Run N parallel RLM instances on the same question, then aggregate the results into a single consensus answer using DSPy's `MultiChainComparison`.

## Architecture

```
                         ┌─────────┐
                         │ Question │
                         └────┬────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │        (N parallel coroutines)
                ┌───▼───┐ ┌──▼──┐ ┌───▼───┐
                │ RLM 1 │ │ ... │ │ RLM N │    Each has own PythonInterpreter
                └───┬───┘ └──┬──┘ └───┬───┘    Each runs full deep-research
                    │        │        │
                    ▼        ▼        ▼
              ┌─────────────────────────────┐
              │   Collect successful runs   │   Discard failures (min 1 needed)
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │   ChainOfThought(Aggregate) │   Single LLM call
              │   (synthesize consensus)    │   Full answers as input
              └──────────────┬──────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │ Final Answer │
                      └─────────────┘
```

## Components

### 1. Parallel RLM Runs (coroutines via asyncio)

- Each run creates its own RLM instance with **no pre-supplied `interpreter`** — DSPy auto-creates a fresh `PythonInterpreter` (Deno subprocess) per `aforward()` call
- Uses `asyncio.gather()` with `return_exceptions=True` to run all N concurrently
- Failed runs (ContextWindowExceededError, etc.) are discarded
- Minimum 1 successful run required to produce an answer
- All runs share the same `dspy.LM` configured once in the main thread (thread-safe for reads)
- The `transcripts` string is passed to all runs (Python strings are immutable, no copy cost)

### 2. Aggregation (ChainOfThought)

A single `dspy.ChainOfThought` call takes all successful run answers concatenated as a labeled input string and synthesizes a consensus answer.

```python
all_answers = "\n\n".join(
    f"=== Run {r['run_id']} ===\n{r['answer']}"
    for r in successful
)
aggregator = dspy.ChainOfThought(AggregateAnswers, temperature=1.0)
final = aggregator(question=question, all_answers=all_answers)
```

The `AggregateAnswers` signature has `question` + `all_answers` as inputs and produces a unified `answer`. ChainOfThought adds a reasoning step so the LM explains its synthesis before producing the final output.

### 3. Error Handling

- Individual RLM failures are caught and logged, not fatal
- If all N runs fail, exit with error
- No retry on individual runs — discard and move on

## File Structure

| File | Description |
|---|---|
| `ensemble.py` | New entry point for ensemble runs |
| `main.py` | Unchanged — single-run mode still works |

## Configuration

| Constant | Value | Location |
|---|---|---|
| `NUM_ENSEMBLE_RUNS` | 5 | `ensemble.py` |
| `MAX_ITERATIONS` | 15 | `ensemble.py` (per RLM) |
| `MAX_LLM_CALLS` | 200 | `ensemble.py` (per RLM) |
| Model | GPT-5 Mini | Same model for RLM and aggregation |

## CLI Usage

```bash
uv run ensemble.py "What is the first tool respondents reach for online?"
```

## Key Constraints

1. **No shared `PythonInterpreter`** — each RLM must create its own (enforced by DSPy's thread-ownership check)
2. **`dspy.configure()` called once** from main context before launching coroutines
3. **OpenAI rate limits** — 5 concurrent RLMs each making up to 200 calls; ensure API tier has sufficient TPM headroom

## Expected Cost

| Component | Estimated Cost |
|---|---|
| 5 RLM runs (parallel) | ~$0.15-0.25 |
| 1 aggregation call | ~$0.01 |
| **Total per question** | **~$0.16-0.26** |

Wall-clock time: same as a single run (~3-10 min) since all 5 run concurrently.

## Amendment: Replacing MultiChainComparison with ChainOfThought (2026-03-16)

### Problem with MultiChainComparison

`MultiChainComparison` was designed for short-answer tasks (math, factoid QA) and is a poor fit for our long-form analytical outputs:

1. **Truncates answers to first line** — `forward()` calls `.split("\n")[0]` on both reasoning and answer, discarding all the detailed counts, evidence, and theme breakdowns that make our analyses valuable.
2. **No reasoning available from RLM** — `MultiChainComparison` expects `rationale`/`reasoning` fields from `ChainOfThought`. RLM doesn't produce these, so the reasoning slots are empty. The comparison LM has nothing substantive to compare.
3. **Degraded prompt** — With empty reasoning and single-line answers, the formatted attempts become `<<I'm trying to  I'm not sure but my prediction is {first line}>>`, which is worse than just showing the full answers directly.

### New Approach

Replace `MultiChainComparison` with a single `dspy.ChainOfThought` call using the `AggregateAnswers` signature. All successful run answers are concatenated into a single formatted input string, and the LM synthesizes them with full visibility into every answer.

```
              ┌─────────────────────────────┐
              │   ChainOfThought(Aggregate) │   Single LLM call
              │   Full answers as input     │   Sees complete text of all N answers
              └──────────────┬──────────────┘
```

This preserves the complete content of every run's answer and lets the LM reason about them holistically before producing a consensus.

## Why This Works

From our experiments, across multiple runs of the same question:
- **Themes and rankings are stable** (Google #1, ChatGPT #2, etc.)
- **Exact counts vary** (ChatGPT: 85-185 across runs)
- **Sub-LM verified answers are more accurate** than regex-only

The ensemble approach gives us:
- **Confidence through consensus** — themes appearing in 5/5 runs are high-confidence
- **Smoothed counts** — the aggregator sees all 5 sets of numbers and synthesizes
- **Resilience** — one bad run (crash, regex-only, max iterations) doesn't tank the result
