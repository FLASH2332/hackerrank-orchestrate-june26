import sys
import os
import re
import json
import requests
import pandas as pd

# Add the parent directory (code/) to sys.path so we can import skills
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skills.evidence_loader import lookup_requirements
from skills.history_loader import lookup_user

PROMPT_VERSION = "v1"

PROMPT_TEMPLATE = """<|think|>
You are an expert claims checker. Your job is to determine whether the visual evidence meets the required standards and identify any risk flags.

Claim Context:
- Claim Object: {claim_object}
- User ID: {user_id}
- Visible Object Part (Agent 1): {visible_part}
- Visible Issue Type (Agent 1): {visible_issue}
- Image Description (Agent 1): {visible_description}
- Image Quality Issues (Agent 1): {image_quality_flags}

Required Evidence Standard:
{evidence_standard}

User Claim History Context:
- Past claim count: {past_claim_count}
- Accepted claims: {accept_claim}
- Claims in manual review: {manual_review_claim}
- Rejected claims: {rejected_claim}
- Last 90 days claim count: {last_90_days_claim_count}
- History flags: {history_flags}
- History summary: {history_summary}

Based on these details, you must evaluate if the visual evidence meets the minimum standard required to assess this claim.
Determine:
1. evidence_standard_met: true if the visible details and images are sufficient to inspect and evaluate the claimed issue; otherwise false.
2. evidence_standard_met_reason: A brief explanation of why the standard is met or not.
3. risk_flags: A list of risk flags identified from the images or the user history. Allowed risk flags:
   none, blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, non_original_image, text_instruction_present, user_history_risk, manual_review_required

If the user history contains history flags like "user_history_risk" or "manual_review_required", you must include them in the risk_flags list.

Your output must be a single JSON object. Do NOT wrap the JSON in markdown code blocks or formatting (no ```json). Do NOT include any intro or outro text. Respond ONLY with the raw JSON.

Format:
{{
  "evidence_standard_met": true/false,
  "evidence_standard_met_reason": "...",
  "risk_flags": ["...", "..."]
}}
"""

SAFE_DEFAULT_RESPONSE = {
    "evidence_standard_met": False,
    "evidence_standard_met_reason": "Failed to parse model response.",
    "risk_flags": ["none"]
}


def strip_thinking(text: str) -> str:
    """
    Strips thinking blocks like <|channel>thought ... <channel|>
    or standard <think>...</think> blocks from response before parsing.
    """
    cleaned = re.sub(r'<\|channel>thought.*?(?:<channel\|>|<\|channel\|>|$)', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<\|think>.*?(?:</\|think>|$)', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_json(text: str) -> str:
    """
    Extracts the JSON substring from response, stripping markdown blocks if present.
    """
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


class EvidenceCheckerAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "gemma3n:e4b"

    def check(self, agent1_output: dict, claim_object: str, 
              user_id: str, evidence_df: pd.DataFrame, 
              history_df: pd.DataFrame) -> dict:
        """
        Runs the evidence check using Agent 1 output, claim object, and user history.
        Returns a dictionary with verification results.
        """
        # 1. Lookup minimum requirements via evidence_loader skill
        issue_type = agent1_output.get("issue_type", "unknown")
        issue_family = "general claim review" if not issue_type or issue_type == "unknown" else issue_type
        evidence_standard = lookup_requirements(evidence_df, claim_object, issue_family)
        if not evidence_standard:
            evidence_standard = "No specific requirement found. Perform general visual inspection."

        # 2. Lookup user history via history_loader skill
        user_history = lookup_user(history_df, user_id)

        # 3. Format the checker prompt
        sys_prompt = PROMPT_TEMPLATE.format(
            claim_object=claim_object,
            user_id=user_id,
            visible_part=agent1_output.get("object_part", "unknown"),
            visible_issue=agent1_output.get("issue_type", "unknown"),
            visible_description=agent1_output.get("visible_description", ""),
            image_quality_flags=agent1_output.get("image_quality_flags", "none"),
            evidence_standard=evidence_standard,
            past_claim_count=user_history.get("past_claim_count", 0),
            accept_claim=user_history.get("accept_claim", 0),
            manual_review_claim=user_history.get("manual_review_claim", 0),
            rejected_claim=user_history.get("rejected_claim", 0),
            last_90_days_claim_count=user_history.get("last_90_days_claim_count", 0),
            history_flags=user_history.get("history_flags", "none"),
            history_summary=user_history.get("history_summary", "")
        )

        # 4. Construct text-only Ollama call payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": "Verify evidence and return standard check results in JSON."
                }
            ],
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64
            },
            "stream": False
        }

        try:
            # 5. Call Ollama chat API
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            
            resp_data = response.json()
            raw_content = resp_data.get("message", {}).get("content", "")
            
            # 6. Parse and strip thinking trace
            cleaned_content = strip_thinking(raw_content)
            json_content = extract_json(cleaned_content)
            
            data = json.loads(json_content)
            
            # Validate essential fields
            if "evidence_standard_met" not in data or "evidence_standard_met_reason" not in data:
                raise KeyError("Missing essential keys from Checker model output")
                
            # 7. Merge history risk flags, Agent 1 flags, and LLM flags to guarantee compliance
            hist_flags_str = user_history.get("history_flags", "none").lower()
            hist_flags = [f.strip() for f in hist_flags_str.split(";") if f.strip() and f.strip() != "none"]
            
            a1_flags_str = agent1_output.get("image_quality_flags", "none").lower()
            a1_flags = [f.strip() for f in a1_flags_str.split(";") if f.strip() and f.strip() != "none"]
            
            llm_flags = data.get("risk_flags", [])
            if isinstance(llm_flags, str):
                llm_flags = [f.strip() for f in llm_flags.split(";") if f.strip()]
                
            all_flags = set(llm_flags + hist_flags + a1_flags)
            
            cleaned_flags = []
            for flag in all_flags:
                flag_clean = flag.strip().lower().replace(" ", "_").replace("-", "_")
                if flag_clean == "none":
                    continue
                cleaned_flags.append(flag_clean)
                
            if not cleaned_flags:
                data["risk_flags"] = ["none"]
            else:
                data["risk_flags"] = sorted(list(set(cleaned_flags)))
                
            # Ensure evidence_standard_met is boolean
            if not isinstance(data["evidence_standard_met"], bool):
                if str(data["evidence_standard_met"]).lower() in ("true", "1", "yes"):
                    data["evidence_standard_met"] = True
                else:
                    data["evidence_standard_met"] = False
                    
            return data
            
        except Exception as e:
            # Log failure loudly
            print(f"[ERROR] Agent 2 failed to verify evidence: {e}", file=sys.stderr)
            return SAFE_DEFAULT_RESPONSE.copy()


if __name__ == "__main__":
    from unittest.mock import patch, MagicMock
    
    print("Running Agent 2 tests...")
    
    # Mock DataFrames
    columns_ev = ["requirement_id", "claim_object", "applies_to", "minimum_image_evidence"]
    data_ev = [["REQ_CAR_PANEL", "car", "dent", "Must see deformation on panels"]]
    evidence_mock_df = pd.DataFrame(data_ev, columns=columns_ev)
    
    columns_hist = [
        "user_id", "past_claim_count", "accept_claim", 
        "manual_review_claim", "rejected_claim", 
        "last_90_days_claim_count", "history_flags", "history_summary"
    ]
    data_hist = [["user_001", 2, 2, 0, 0, 1, "none", "Low-risk user"]]
    history_mock_df = pd.DataFrame(data_hist, columns=columns_hist)
    
    agent1_out = {
        "visible_description": "Dent on car panel",
        "image_quality_flags": "none",
        "object_part": "door",
        "issue_type": "dent",
        "valid_image": True
    }
    
    agent = EvidenceCheckerAgent()
    
    # Test Case 1: Mocked verification call
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "<|channel>thought\nDouble checking standard requirements...\n<channel|>{\"evidence_standard_met\": true, \"evidence_standard_met_reason\": \"Standard met because panel is visible\", \"risk_flags\": [\"none\"]}"
        }
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        res = agent.check(agent1_out, "car", "user_001", evidence_mock_df, history_mock_df)
        mock_post.assert_called_once()
        print("Test 1 (Mocked Verification):", res)
        assert res["evidence_standard_met"] is True
        assert res["risk_flags"] == ["none"]
        
    # Test Case 2: Verification with user history risk and parsing fallback
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 200
    mock_response_fail.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Invalid response"
        }
    }
    with patch("requests.post", return_value=mock_response_fail):
        res_fail = agent.check(agent1_out, "car", "user_001", evidence_mock_df, history_mock_df)
        print("Test 2 (Fallback):", res_fail)
        assert res_fail["evidence_standard_met"] is False
        assert "none" in res_fail["risk_flags"]
        
    print("All Agent 2 tests completed successfully!")
