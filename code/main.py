import os
import sys
import pandas as pd
from tqdm import tqdm

# Add parent directory (code/) to sys.path so we can import agents and skills
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from skills.evidence_loader import load_evidence_requirements
from skills.history_loader import load_user_history
from skills.output_validator import validate_output
from agents.agent1_image_analyst import ImageAnalystAgent
from agents.agent2_evidence_checker import EvidenceCheckerAgent
from agents.agent3_verdict_writer import VerdictWriterAgent


def main():
    # 1. All config comes from the centralized cfg (loaded from .env)
    claims_csv_path = cfg.CLAIMS_CSV
    evidence_csv_path = cfg.EVIDENCE_CSV
    history_csv_path = cfg.HISTORY_CSV
    images_dir = cfg.IMAGES_DIR
    output_csv_path = cfg.OUTPUT_CSV

    print("Starting Claim Verification Pipeline...")
    print(f"  Claims input: {claims_csv_path}")
    print(f"  Evidence rules: {evidence_csv_path}")
    print(f"  User history: {history_csv_path}")
    print(f"  Images base dir: {images_dir}")
    print(f"  Output file: {output_csv_path}")
    print(f"  Ollama URL: {cfg.OLLAMA_URL}")
    print(f"  Model: {cfg.CLOUD_MODEL if cfg.API_KEY and cfg.CLOUD_MODEL else cfg.OLLAMA_MODEL}")

    # 2. Check and validate paths
    if not os.path.exists(evidence_csv_path):
        print(f"[ERROR] Evidence requirements file not found at: {evidence_csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(history_csv_path):
        print(f"[ERROR] User history file not found at: {history_csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(claims_csv_path):
        print(f"[ERROR] Claims input file not found at: {claims_csv_path}", file=sys.stderr)
        sys.exit(1)

    # 3. Load reference datasets once at startup
    evidence_df = load_evidence_requirements(evidence_csv_path)
    history_df = load_user_history(history_csv_path)

    # 4. Initialize cache and agents (model + sampling from config)
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

    # 5. Read input claims
    try:
        claims_df = pd.read_csv(claims_csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read claims input file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(claims_df)} rows to process.")
    results = []

    # 6. Process row by row sequentially
    for idx, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing Claims"):
        user_id = str(row.get("user_id", "")).strip()
        image_paths = str(row.get("image_paths", "")).strip()
        user_claim = str(row.get("user_claim", "")).strip()
        claim_object = str(row.get("claim_object", "")).strip()

        # Handle split and resolved paths for Agent 1
        raw_paths = [p.strip() for p in image_paths.split(";") if p.strip()]
        resolved_paths = []
        for p in raw_paths:
            if not os.path.isabs(p):
                # Prepend images_dir to resolve local path relative to workspace
                full_path = os.path.abspath(os.path.join(images_dir, p))
            else:
                full_path = p
            resolved_paths.append(full_path)

        try:
            if not resolved_paths:
                raise ValueError("No valid image paths provided for row.")

            # Step A: Image Analyst Agent (Agent 1)
            agent1_out = agent1.analyze(resolved_paths, claim_object, user_claim, cache)
            
            # Step B: Evidence Checker Agent (Agent 2)
            agent2_out = agent2.check(agent1_out, claim_object, user_id, evidence_df, history_df)
            
            # Step C: Verdict Writer Agent (Agent 3)
            final_row = agent3.write(
                agent1_output=agent1_out,
                agent2_output=agent2_out,
                user_claim=user_claim,
                claim_object=claim_object,
                user_id=user_id,
                image_paths=image_paths
            )
            
            results.append(final_row)

        except Exception as e:
            print(f"[ERROR] Failed to process row {idx} for user {user_id}: {e}", file=sys.stderr)
            # Create a safe fallback row using output_validator
            fallback_raw = {
                "user_id": user_id,
                "image_paths": image_paths,
                "user_claim": user_claim,
                "claim_object": claim_object,
                "evidence_standard_met": False,
                "evidence_standard_met_reason": f"Pipeline execution error: {e}",
                "risk_flags": "none",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Pipeline execution encountered an error: {e}",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown"
            }
            fallback_row = validate_output(fallback_raw)
            results.append(fallback_row)

    # 7. Write results to output.csv in the correct schema order
    try:
        out_df = pd.DataFrame(results)
        
        required_cols = [
            "user_id", "image_paths", "user_claim", "claim_object",
            "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
            "issue_type", "object_part", "claim_status", "claim_status_justification",
            "supporting_image_ids", "valid_image", "severity"
        ]
        # Re-ensure column ordering
        out_df = out_df[required_cols]
        out_df.to_csv(output_csv_path, index=False)
        print(f"Pipeline completed successfully. Output written to {output_csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
