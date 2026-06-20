# Claim Verification System

Multi-modal evidence review system for the HackerRank Orchestrate hackathon. Verifies damage claims across **cars**, **laptops**, and **packages** using images, claim conversations, user history, and minimum evidence requirements.

---

## Architecture

The system uses a **3-agent sequential pipeline**, each powered by `gemma3n:e4b` via local Ollama:

```
Claims CSV
    │
    ▼
┌────────────────────────┐
│  Agent 1 — Image       │  Multimodal (images + text)
│  Analyst               │  Thinking ON
│  Outputs: description, │
│  quality flags, part,  │
│  issue type, validity  │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  Agent 2 — Evidence    │  Text-only
│  Checker               │  Thinking ON
│  Outputs: standard     │
│  met, reason, risk     │
│  flags                 │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│  Agent 3 — Verdict     │  Text-only
│  Writer                │  Thinking OFF
│  Outputs: final row    │
│  (claim_status, sev,   │
│  justification, etc.)  │
└──────────┬─────────────┘
           │
           ▼
      output.csv
```

**Skills** (stateless utility functions in `code/skills/`):
- `image_encoder.py` — Base64 encodes images with Level 1 in-memory cache
- `evidence_loader.py` — Loads and looks up minimum evidence requirements
- `history_loader.py` — Loads and looks up user claim history
- `output_validator.py` — Pydantic model validation of all output enums and column ordering

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally ([install guide](https://ollama.com))
- **gemma3n:e4b** model pulled in Ollama

```bash
# Install Ollama (if not already installed), then pull the model
ollama pull gemma3n:e4b
```

Verify Ollama is running:

```bash
ollama list
# Should show gemma3n:e4b in the list
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd hackerrank-orchestrate-june26
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
| Package | Purpose |
|---|---|
| `pandas` | CSV reading, DataFrame operations, output writing |
| `pydantic` | Output schema validation and enum enforcement |
| `pillow` | Image loading for base64 encoding |
| `requests` | HTTP calls to local Ollama API |
| `tqdm` | Progress bar for row-by-row processing |

### 4. Verify Ollama is running

```bash
# Ollama should be serving at http://localhost:11434
curl http://localhost:11434/api/tags
```

---

## Running the Pipeline

### Process test claims (produce output.csv)

From the **repository root**:

```bash
python code/main.py
```

This will:
1. Load `dataset/evidence_requirements.csv` and `dataset/user_history.csv` once
2. Read `dataset/claims.csv` row by row
3. Run Agent 1 → Agent 2 → Agent 3 sequentially for each row
4. Write validated results to `output.csv`

### Environment variables (all optional)

| Variable | Default | Description |
|---|---|---|
| `CLAIMS_CSV` | `dataset/claims.csv` | Input claims file |
| `EVIDENCE_CSV` | `dataset/evidence_requirements.csv` | Evidence requirements lookup |
| `HISTORY_CSV` | `dataset/user_history.csv` | User history lookup |
| `IMAGES_DIR` | `dataset` | Base directory for resolving image paths |
| `OUTPUT_CSV` | `output.csv` | Output predictions file |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |

Example with custom paths:

```bash
# Windows
set CLAIMS_CSV=dataset/claims.csv
set OUTPUT_CSV=output.csv
python code/main.py

# macOS / Linux
CLAIMS_CSV=dataset/claims.csv OUTPUT_CSV=output.csv python code/main.py
```

---

## Running the Evaluation Pipeline

The evaluator runs the full pipeline on `dataset/sample_claims.csv` (which contains both inputs and expected outputs) and compares predictions against ground truth.

From the **repository root**:

```bash
python code/evaluation/main.py
```

This will:
1. Run the 3-agent pipeline on all 20 sample rows
2. Compare predicted vs expected values for 6 key fields
3. Print a per-field accuracy summary table to stdout
4. Write a detailed report to `code/evaluation/evaluation_report.md`

### What gets evaluated

| Field | What it measures |
|---|---|
| `claim_status` | Final verdict accuracy (supported / contradicted / not_enough_information) |
| `issue_type` | Correct damage type identification (dent, scratch, crack, etc.) |
| `object_part` | Correct part identification (rear_bumper, screen, seal, etc.) |
| `severity` | Correct severity estimation (none, low, medium, high, unknown) |
| `evidence_standard_met` | Whether the system correctly assesses image sufficiency |
| `valid_image` | Whether the system correctly identifies usable images |

### Evaluation report contents

The generated `code/evaluation/evaluation_report.md` includes:
- Per-field accuracy scores
- Model call counts (per agent and total)
- Approximate token usage estimates
- Number of images processed and unique images cached
- Cost analysis ($0.00 for local Ollama inference)
- Runtime statistics (total and per-row average)
- Strategy notes on caching, batching, retry, and rate limiting

### Evaluation environment variables

| Variable | Default | Description |
|---|---|---|
| `SAMPLE_CSV` | `dataset/sample_claims.csv` | Sample claims with expected outputs |
| `EVAL_REPORT` | `code/evaluation/evaluation_report.md` | Report output path |

---

## Project Structure

```
code/
├── main.py                          # Pipeline orchestrator entry point
├── README.md                        # This file
├── agents/
│   ├── agent1_image_analyst.py      # Multimodal image analysis agent
│   ├── agent2_evidence_checker.py   # Evidence standard verification agent
│   └── agent3_verdict_writer.py     # Final verdict generation agent
├── skills/
│   ├── image_encoder.py             # Image → base64 with L1 cache
│   ├── evidence_loader.py           # Evidence requirements loader/lookup
│   ├── history_loader.py            # User history loader/lookup
│   └── output_validator.py          # Pydantic enum validation + column ordering
└── evaluation/
    ├── main.py                      # Evaluation pipeline entry point
    └── evaluation_report.md         # Generated after running evaluation
```

---

## Design Decisions

- **Single model**: Only `gemma3n:e4b` is loaded, as required by the project rules. No model swapping between agents.
- **Sequential processing**: Each row is processed one at a time through all 3 agents. No parallelism.
- **Level 1 cache**: In-memory dict keyed by exact `image_path` string. Prevents re-encoding the same image across rows. Not persisted to disk.
- **No-throw guarantee**: Every agent and the output validator are wrapped in try-except blocks. Failures produce safe fallback rows with `claim_status=not_enough_information` and `severity=unknown`.
- **Enum enforcement**: The output validator uses fuzzy matching (`difflib.get_close_matches`) to correct near-miss enum values before writing to CSV.
- **Forward-only data flow**: Agent 1 output feeds Agent 2, Agent 2 output feeds Agent 3. No agent re-analyzes prior agent decisions.
- **History cannot override verdicts**: User history can only add risk flags. It cannot flip a supported/contradicted verdict.
