# Evaluation Report

## Per-Field Accuracy

Evaluated on **20** sample rows from `dataset/sample_claims.csv`.

| Field | Accuracy |
|---|---|
| claim_status | 70.0% |
| issue_type | 55.0% |
| object_part | 75.0% |
| severity | 60.0% |
| evidence_standard_met | 95.0% |
| valid_image | 90.0% |

## Operational Analysis

### Model Calls

| Agent | Calls |
|---|---|
| Agent 1 (Image Analyst, multimodal) | 20 |
| Agent 2 (Evidence Checker, text-only) | 20 |
| Agent 3 (Verdict Writer, text-only) | 20 |
| **Total** | **60** |

### Token Usage (Approximate)

- Estimated input tokens: ~42,000
- Estimated output tokens: ~11,000
- Estimated total tokens: ~53,000

### Images Processed

- Total images submitted: 29
- Unique images cached (Level 1 cache): 29

### Cost

- Model: `llava:7b` via local Ollama
- API cost: **$0.00** (local inference)
- Compute cost: local GPU / CPU time only

### Runtime

- Total elapsed time: 576.6s
- Average per row: 28.8s
- Rows processed: 20

### Strategy Notes

- **Caching**: Level 1 in-memory cache keyed by `image_path`. Prevents re-encoding the same image across rows. Not persisted to disk.
- **Batching**: No batching; each row is processed sequentially with one Ollama call per agent per row.
- **Retry strategy**: Implemented exponential backoff with jitter via `safe_llm_request`. Handles HTTP 429, 5xx, and timeouts safely across all agents.
- **Rate limiting**: Not applicable for local Ollama. No TPM/RPM limits.
- **Parallelism**: Disabled. Sequential processing only to respect single-model constraint (AGENTS.md §9.1).
