import sys
import os
import re
import json
import requests

# Add parent directory (code/) to sys.path so we can import skills
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skills.output_validator import validate_output

PROMPT_VERSION = "v1"

PROMPT_TEMPLATE = """You are an expert claims verdict writer. Your job is to produce the final claims verdict based on the outputs of the Image Analyst (Agent 1) and the Evidence Checker (Agent 2).

You must trust the prior agents' findings and not re-derive them:
- Visible Object Part (Agent 1): {visible_part}
- Visible Issue Type (Agent 1): {visible_issue}
- Image Description (Agent 1): {visible_description}
- Valid Image (Agent 1): {valid_image}
- Evidence Standard Met (Agent 2): {evidence_standard_met}
- Evidence Standard Met Reason (Agent 2): {evidence_standard_met_reason}
- Risk Flags (Agent 2): {risk_flags}

Claim Details:
- Claim Object: {claim_object}
- User Claim: {user_claim}
- User ID: {user_id}
- Image Paths: {image_paths}

Based on this context, you must decide:
1. claim_status: supported, contradicted, or not_enough_information.
   - Use "supported" if the image evidence clearly shows the damage described in the user claim.
   - Use "contradicted" if the image shows a different object, a different object part, different damage type, or no damage at all where damage was claimed.
   - Use "not_enough_information" if the image set is not usable, the wrong angle, or does not show the claimed part clearly.
2. claim_status_justification: A concise, image-grounded explanation for the decision. Reference image IDs (e.g. img_1) when relevant.
3. supporting_image_ids: Semicolon-separated image IDs (e.g. "img_1" or "img_1;img_2") that support this verdict. If no image supports the decision, use "none".
4. severity: none, low, medium, high, or unknown. Estimate this based on the visible damage.

Rules to enforce:
- If evidence_standard_met is false, or valid_image is false: claim_status must be "not_enough_information", severity must be "unknown", and supporting_image_ids must be "none".
- User history cannot flip a supported or contradicted verdict.

Your response must be a single JSON object. Do NOT wrap the JSON in markdown code blocks or formatting (no ```json). Do NOT include any intro or outro text. Respond ONLY with the raw JSON.

Format:
{{
  "claim_status": "...",
  "claim_status_justification": "...",
  "supporting_image_ids": "...",
  "severity": "..."
}}
"""

SAFE_DEFAULT_RESPONSE = {
    "claim_status": "not_enough_information",
    "claim_status_justification": "Failed to parse model response.",
    "supporting_image_ids": "none",
    "severity": "unknown"
}


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


class VerdictWriterAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "gemma3n:e4b"

    def write(self, agent1_output: dict, agent2_output: dict,
              user_claim: str, claim_object: str, 
              user_id: str, image_paths: str) -> dict:
        """
        Combines information from Agent 1 and Agent 2 to generate the final output row dictionary.
        Sends context to Ollama (thinking OFF) to decide claim status, justification, supporting IDs, and severity.
        Validates the combined result using the output_validator skill.
        """
        risk_flags_list = agent2_output.get("risk_flags", ["none"])
        risk_flags_str = ";".join(risk_flags_list) if isinstance(risk_flags_list, list) else str(risk_flags_list)

        # 1. Format prompt (without <|think|> tag, per instructions)
        sys_prompt = PROMPT_TEMPLATE.format(
            visible_part=agent1_output.get("object_part", "unknown"),
            visible_issue=agent1_output.get("issue_type", "unknown"),
            visible_description=agent1_output.get("visible_description", ""),
            valid_image=str(agent1_output.get("valid_image", False)),
            evidence_standard_met=str(agent2_output.get("evidence_standard_met", False)),
            evidence_standard_met_reason=agent2_output.get("evidence_standard_met_reason", ""),
            risk_flags=risk_flags_str,
            claim_object=claim_object,
            user_claim=user_claim,
            user_id=user_id,
            image_paths=image_paths
        )

        # 2. Construct text-only Ollama call payload (Thinking OFF)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": "Generate final claims verdict JSON."
                }
            ],
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64
            },
            "stream": False
        }

        verdict_data = SAFE_DEFAULT_RESPONSE.copy()

        try:
            # 3. Call Ollama chat API
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            
            resp_data = response.json()
            raw_content = resp_data.get("message", {}).get("content", "")
            
            # Since thinking is OFF, we don't expect a reasoning trace, but just in case, extract JSON
            json_content = extract_json(raw_content)
            data = json.loads(json_content)
            
            # Extract keys
            verdict_data["claim_status"] = data.get("claim_status", "not_enough_information")
            verdict_data["claim_status_justification"] = data.get("claim_status_justification", "")
            verdict_data["supporting_image_ids"] = data.get("supporting_image_ids", "none")
            verdict_data["severity"] = data.get("severity", "unknown")
            
        except Exception as e:
            print(f"[ERROR] Agent 3 failed to generate verdict via Ollama: {e}", file=sys.stderr)
            verdict_data["claim_status_justification"] = f"Failed to get verdict from model: {e}"

        # 4. Compile complete output row dictionary
        raw_output = {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": agent2_output.get("evidence_standard_met", False),
            "evidence_standard_met_reason": agent2_output.get("evidence_standard_met_reason", ""),
            "risk_flags": risk_flags_str,
            "issue_type": agent1_output.get("issue_type", "unknown"),
            "object_part": agent1_output.get("object_part", "unknown"),
            "claim_status": verdict_data["claim_status"],
            "claim_status_justification": verdict_data["claim_status_justification"],
            "supporting_image_ids": verdict_data["supporting_image_ids"],
            "valid_image": agent1_output.get("valid_image", False),
            "severity": verdict_data["severity"]
        }

        # 5. Enforce hard rules in python (safety net)
        # If evidence standard is not met or image is invalid, override status and severity
        evidence_met = raw_output["evidence_standard_met"]
        is_valid_img = raw_output["valid_image"]
        
        # Coerce values if string
        if isinstance(evidence_met, str):
            evidence_met = evidence_met.strip().lower() in ("true", "1", "yes")
        if isinstance(is_valid_img, str):
            is_valid_img = is_valid_img.strip().lower() in ("true", "1", "yes")
            
        if not evidence_met or not is_valid_img:
            raw_output["claim_status"] = "not_enough_information"
            raw_output["severity"] = "unknown"
            raw_output["supporting_image_ids"] = "none"

        # 6. Validate the complete dictionary using output_validator skill
        validated_dict = validate_output(raw_output)
        
        return validated_dict


if __name__ == "__main__":
    from unittest.mock import patch, MagicMock
    
    print("Running Agent 3 tests...")
    
    agent1_out = {
        "visible_description": "Dent on car panel",
        "image_quality_flags": "none",
        "object_part": "rear_bumper",
        "issue_type": "dent",
        "valid_image": True
    }
    
    agent2_out = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Bumper is visible",
        "risk_flags": ["none"]
    }
    
    agent = VerdictWriterAgent()
    
    # Test Case 1: Mocked Verdict
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "{\"claim_status\": \"supported\", \"claim_status_justification\": \"Bumper dent is visible\", \"supporting_image_ids\": \"img_1\", \"severity\": \"medium\"}"
        }
    }
    
    with patch("requests.post", return_value=mock_response) as mock_post:
        res = agent.write(
            agent1_out, agent2_out, 
            "The back of the car has a dent.", "car", 
            "user_001", "images/sample/case_001/img_1.jpg"
        )
        mock_post.assert_called_once()
        print("Test 1 (Mocked Verdict):", res)
        assert res["claim_status"] == "supported"
        assert res["severity"] == "medium"
        assert res["valid_image"] is True
        assert res["evidence_standard_met"] is True
        
    # Test Case 2: Enforce invalid evidence rule in python
    agent2_out_invalid = {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "Too blurry to assess",
        "risk_flags": ["blurry_image"]
    }
    
    with patch("requests.post", return_value=mock_response):
        res_invalid = agent.write(
            agent1_out, agent2_out_invalid, 
            "The back of the car has a dent.", "car", 
            "user_001", "images/sample/case_001/img_1.jpg"
        )
        print("Test 2 (Evidence Standard Not Met Rule):", res_invalid)
        assert res_invalid["claim_status"] == "not_enough_information"
        assert res_invalid["severity"] == "unknown"
        assert res_invalid["supporting_image_ids"] == "none"

    print("All Agent 3 tests completed successfully!")
