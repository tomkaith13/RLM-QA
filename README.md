# RLM Transcript QA

Ask natural language questions over hundreds of interview transcripts using [DSPy's Recursive Language Model (RLM)](https://dspy.ai/api/modules/RLM/).

## The Problem

We have 600+ interview transcripts (~1M+ tokens) from AI-moderated research calls about search engine and AI tool usage. The data is too large to fit in any single LLM context window. We need a way to ask complex analytical questions across the entire dataset.

## How It Works

**RLM (Recursive Language Model)** solves this by combining an LLM with a code interpreter in an iterative loop:

1. The **LM** (GPT-5 Mini) reasons about the question and writes Python code to analyze the data
2. The code executes in a **sandboxed Deno/Pyodide interpreter** with access to the full transcript text
3. Within that code, `llm_query()` and `llm_query_batched()` are available for semantic analysis tasks like topic extraction or sentiment classification
4. The LM reviews the code output and decides whether to write more code or submit a final answer
5. This loop repeats (up to N iterations) until the answer is ready

This approach lets the system process arbitrarily large datasets — the LLM never needs to "read" all the data at once. Instead, it writes code to iterate over it programmatically.

## Transcript Format

Transcripts are stored as JSON arrays in `data/`. Each file contains an array of call objects:

```json
{
  "id": "cmfizraj9040ine215oacg3es",
  "responseId": "cmfizqoa70409ne21pge28qln",
  "messages": [
    {
      "callId": "cmfizraj9040ine215oacg3es",
      "index": 0,
      "role": "bot",
      "time": "2025-09-14T01:03:18.707000Z",
      "message": "Hi there. I'm Elliot, your AI moderator..."
    },
    {
      "role": "user",
      "message": "Usually, I go to Google."
    }
  ],
  "attributes": [
    { "label": "Gender+", "value": "Female" },
    { "label": "Age", "value": "35" },
    { "label": "Ethnicity", "value": "White" },
    { "label": "US Census Region", "value": "South" }
  ]
}
```

- **messages**: The conversation between the AI moderator (`bot`) and the participant (`user`), ordered by `index`
- **attributes**: Demographic metadata about the participant (age, gender, region, income, etc.)

At load time, all JSON files in `data/` are merged and reformatted into a flat text block for RLM to process via code.

## Models

| Role | Model | Purpose |
|------|-------|---------|
| LM | `openai/gpt-5-mini` | Reasoning, code generation, orchestration, and semantic analysis |

The LM handles everything — deciding what code to write, executing semantic queries via `llm_query()`, and determining when to submit a final answer.

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Deno](https://deno.com/) runtime (for the sandboxed code interpreter)

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
```

### Install & Run

```bash
uv sync
uv run main.py "What are the most common topics discussed in these calls?"
```

### Example Output

```
ANSWER:
1. Search engine habits and preferences (primarily Google)
2. Comparison of traditional search vs. AI tools (ChatGPT, Gemini, Perplexity)
3. Trust and information verification
4. Specific use cases for AI (brainstorming, writing, recipes, travel)
5. User experience and interface preferences

--- Cost: $0.013731 | Tokens: 11,363 | LLM calls: 2 | Duration: 103.8s ---
```

## Screenshots

### Analyzing transcripts with Claude Code

Claude Code running a demographic query across 760 calls — filtering Hispanic respondents over 50 and their primary search engine.

![Claude Code Example](screenshots/claude-code-example.png)

### Direct RLM run with GPT-5 Mini

Running `main.py` directly from the terminal. The RLM processes all transcripts autonomously and returns a structured summary.

![Local Run Using GPT-5 Mini](screenshots/local-run-using-gpt5-mini.png)
