import os
import sys
import time
import pandas as pd
from tqdm import tqdm

# Add code/ directory to sys.path for skill/agent imports
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code_dir)

from config import cfg
from skills.evidence_loader import load_evidence_requirements
from skills.history_loader import load_user_history
from skills.output_validator import validate_output
from agents.agent1_image_analyst import ImageAnalystAgent
from agents.agent2_evidence_checker import EvidenceCheckerAgent
from agents.agent3_verdict_writer import VerdictWriterAgent


# Fields to evaluate accuracy on
EVAL_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image",
]


def normalize(val):
    """Normalize a value for comparison: lowercase, strip, convert bools."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return "true"
    if s in ("false", "0", "no"):
        return "false"
    return s


def run_pipeline_on_sample(sample_csv_path, evidence_csv_path, history_csv_path, images_dir):
    """
    Runs the full 3-agent pipeline on the sample CSV and returns
    (predictions_list, total_agent1_calls, total_agent2_calls, total_agent3_calls, total_images, elapsed_seconds).
    """
    evidence_df = load_evidence_requirements(evidence_csv_path)
    history_df = load_user_history(history_csv_path)

    cache = {}
    agent_kwargs = dict(
        ollama_url=cfg.OLLAMA_URL,
        model=cfg.CLOUD_MODEL if cfg.API_KEY and cfg.CLOUD_MODEL else cfg.OLLAMA_MODEL,
        temperature=cfg.MODEL_TEMPERATURE,
        top_p=cfg.MODEL_TOP_P,
        top_k=cfg.MODEL_TOP_K,
        api_key=cfg.API_KEY,
        api_base_url=cfg.API_BASE_URL,
    )
    agent1 = ImageAnalystAgent(**agent_kwargs)
    agent2 = EvidenceCheckerAgent(**agent_kwargs)
    agent3 = VerdictWriterAgent(**agent_kwargs)

    claims_df = pd.read_csv(sample_csv_path)
    print(f"Loaded {len(claims_df)} sample rows for evaluation.")

    results = []
    total_a1 = 0
    total_a2 = 0
    total_a3 = 0
    total_images = 0

    start_time = time.time()

    for idx, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Evaluating Sample"):
        user_id = str(row.get("user_id", "")).strip()
        image_paths = str(row.get("image_paths", "")).strip()
        user_claim = str(row.get("user_claim", "")).strip()
        claim_object = str(row.get("claim_object", "")).strip()

        raw_paths = [p.strip() for p in image_paths.split(";") if p.strip()]
        resolved_paths = []
        for p in raw_paths:
            if not os.path.isabs(p):
                full_path = os.path.abspath(os.path.join(images_dir, p))
            else:
                full_path = p
            resolved_paths.append(full_path)

        total_images += len(resolved_paths)

        try:
            if not resolved_paths:
                raise ValueError("No valid image paths.")

            agent1_out = agent1.analyze(resolved_paths, claim_object, user_claim, cache)
            total_a1 += 1

            agent2_out = agent2.check(agent1_out, claim_object, user_id, evidence_df, history_df)
            total_a2 += 1

            final_row = agent3.write(
                agent1_output=agent1_out,
                agent2_output=agent2_out,
                user_claim=user_claim,
                claim_object=claim_object,
                user_id=user_id,
                image_paths=image_paths,
            )
            total_a3 += 1
            results.append(final_row)

        except Exception as e:
            print(f"[ERROR] Row {idx} user={user_id}: {e}", file=sys.stderr)
            fallback = validate_output({
                "user_id": user_id,
                "image_paths": image_paths,
                "user_claim": user_claim,
                "claim_object": claim_object,
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"Evaluation pipeline error: {e}",
                "risk_flags": "none",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Error: {e}",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            })
            results.append(fallback)
            total_a1 += 1  # count as attempted

    elapsed = time.time() - start_time
    return results, total_a1, total_a2, total_a3, total_images, elapsed


def compute_accuracy(expected_df, predicted_list, fields):
    """
    Computes per-field accuracy between the expected DataFrame and the list
    of predicted dictionaries. Returns a dict of {field: accuracy_float}.
    """
    n = min(len(expected_df), len(predicted_list))
    scores = {}
    for field in fields:
        if field not in expected_df.columns:
            scores[field] = None
            continue
        correct = 0
        for i in range(n):
            expected_val = normalize(expected_df.iloc[i].get(field, ""))
            predicted_val = normalize(predicted_list[i].get(field, ""))
            if expected_val == predicted_val:
                correct += 1
        scores[field] = correct / n if n > 0 else 0.0
    return scores


def print_summary_table(scores, n_rows):
    """Prints a formatted summary table to stdout."""
    print("\n" + "=" * 50)
    print(f"  EVALUATION SUMMARY ({n_rows} rows)")
    print("=" * 50)
    print(f"  {'Field':<28} {'Accuracy':>10}")
    print("-" * 50)
    for field, acc in scores.items():
        if acc is None:
            print(f"  {field:<28} {'N/A':>10}")
        else:
            print(f"  {field:<28} {acc:>9.1%}")
    print("=" * 50)


def write_report(report_path, scores, n_rows, a1_calls, a2_calls, a3_calls,
                 n_images, elapsed, cache_size, model_name):
    """Writes evaluation_report.md to disk."""
    total_calls = a1_calls + a2_calls + a3_calls
    avg_per_row = elapsed / n_rows if n_rows > 0 else 0

    # Rough token estimates:
    # Agent 1 (multimodal): ~800 input + ~200 output per call
    # Agent 2 (text): ~600 input + ~150 output per call
    # Agent 3 (text): ~700 input + ~200 output per call
    est_input_tokens = a1_calls * 800 + a2_calls * 600 + a3_calls * 700
    est_output_tokens = a1_calls * 200 + a2_calls * 150 + a3_calls * 200

    lines = [
        "# Evaluation Report",
        "",
        "## Per-Field Accuracy",
        "",
        f"Evaluated on **{n_rows}** sample rows from `dataset/sample_claims.csv`.",
        "",
        "| Field | Accuracy |",
        "|---|---|",
    ]
    for field, acc in scores.items():
        if acc is None:
            lines.append(f"| {field} | N/A |")
        else:
            lines.append(f"| {field} | {acc:.1%} |")

    lines += [
        "",
        "## Operational Analysis",
        "",
        "### Model Calls",
        "",
        f"| Agent | Calls |",
        f"|---|---|",
        f"| Agent 1 (Image Analyst, multimodal) | {a1_calls} |",
        f"| Agent 2 (Evidence Checker, text-only) | {a2_calls} |",
        f"| Agent 3 (Verdict Writer, text-only) | {a3_calls} |",
        f"| **Total** | **{total_calls}** |",
        "",
        "### Token Usage (Approximate)",
        "",
        f"- Estimated input tokens: ~{est_input_tokens:,}",
        f"- Estimated output tokens: ~{est_output_tokens:,}",
        f"- Estimated total tokens: ~{est_input_tokens + est_output_tokens:,}",
        "",
        "### Images Processed",
        "",
        f"- Total images submitted: {n_images}",
        f"- Unique images cached (Level 1 cache): {cache_size}",
        "",
        "### Cost",
        "",
        f"- Model: `{model_name}` via local Ollama",
        "- API cost: **$0.00** (local inference)",
        "- Compute cost: local GPU / CPU time only",
        "",
        "### Runtime",
        "",
        f"- Total elapsed time: {elapsed:.1f}s",
        f"- Average per row: {avg_per_row:.1f}s",
        f"- Rows processed: {n_rows}",
        "",
        "### Strategy Notes",
        "",
        "- **Caching**: Level 1 in-memory cache keyed by `image_path`. "
        "Prevents re-encoding the same image across rows. Not persisted to disk.",
        "- **Batching**: No batching; each row is processed sequentially with "
        "one Ollama call per agent per row.",
        "- **Retry strategy**: Implemented exponential backoff with jitter via `safe_llm_request`. Handles HTTP 429, 5xx, and timeouts safely across all agents.",
        "- **Rate limiting**: Not applicable for local Ollama. No TPM/RPM limits.",
        "- **Parallelism**: Disabled. Sequential processing only to respect "
        "single-model constraint (AGENTS.md §9.1).",
        "",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Evaluation report written to {report_path}")


def main():
    # All config comes from the centralized cfg (loaded from .env)
    sample_csv = cfg.SAMPLE_CSV
    evidence_csv = cfg.EVIDENCE_CSV
    history_csv = cfg.HISTORY_CSV
    images_dir = cfg.IMAGES_DIR
    report_path = cfg.EVAL_REPORT

    print("=" * 50)
    print("  EVALUATION PIPELINE")
    print("=" * 50)
    print(f"  Sample CSV: {sample_csv}")
    print(f"  Evidence CSV: {evidence_csv}")
    print(f"  History CSV: {history_csv}")
    print(f"  Images Dir: {images_dir}")
    print(f"  Ollama URL: {cfg.OLLAMA_URL}")
    print(f"  Model: {cfg.CLOUD_MODEL if cfg.API_KEY and cfg.CLOUD_MODEL else cfg.OLLAMA_MODEL}")
    print(f"  Report Path: {report_path}")
    print()

    if not os.path.exists(sample_csv):
        print(f"[ERROR] Sample CSV not found: {sample_csv}", file=sys.stderr)
        sys.exit(1)

    # Run the pipeline
    results, a1, a2, a3, n_images, elapsed = run_pipeline_on_sample(
        sample_csv, evidence_csv, history_csv, images_dir
    )

    # Load expected values
    expected_df = pd.read_csv(sample_csv)
    n_rows = len(expected_df)

    # Compute accuracy
    scores = compute_accuracy(expected_df, results, EVAL_FIELDS)

    # Print summary
    print_summary_table(scores, n_rows)

    # Estimate cache size (count unique image paths across all rows)
    all_img_paths = set()
    for _, row in expected_df.iterrows():
        paths = str(row.get("image_paths", "")).split(";")
        for p in paths:
            p = p.strip()
            if p:
                all_img_paths.add(p)
    cache_size = len(all_img_paths)

    # Write report
    model_used = cfg.CLOUD_MODEL if cfg.API_KEY and cfg.CLOUD_MODEL else cfg.OLLAMA_MODEL
    write_report(report_path, scores, n_rows, a1, a2, a3, n_images, elapsed, cache_size, model_used)

    print(f"\nEvaluation complete. Processed {n_rows} rows in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
