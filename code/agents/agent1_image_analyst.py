import sys
import os
import re
import json
import requests

# Add the parent directory (code/) to sys.path so we can import skills
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skills.image_encoder import encode_image

PROMPT_VERSION = "v1"

PROMPT_TEMPLATE = """<|think|>
You are an expert claims examiner. Your job is to analyze the provided image(s) and extract visual details regarding a claim.
The claim is about: {claim_object}
User describes the damage as: {user_claim}

Analyze the image(s) carefully. You must determine:
1. visible_description: A brief description of what is visible in the image(s).
2. image_quality_flags: Set to "none" if the images are clear. Otherwise, identify any quality issues using ONLY these allowed flags (semicolon-separated):
   none, blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, non_original_image, text_instruction_present, user_history_risk, manual_review_required
3. object_part: The specific part of the object shown in the image.
   For car, use one of: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown
   For laptop, use one of: screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown
   For package, use one of: box, package_corner, package_side, seal, label, contents, item, unknown
4. issue_type: The type of issue visible. Use one of:
   dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
5. valid_image: Whether the image is usable for review (true or false). Set to false if it's the wrong object completely, empty, corrupted, or not readable.

Your output must be a single JSON object. Do NOT wrap the JSON in markdown code blocks or formatting (no ```json). Do NOT include any intro or outro text. Respond ONLY with the raw JSON.

Format:
{{
  "visible_description": "...",
  "image_quality_flags": "...",
  "object_part": "...",
  "issue_type": "...",
  "valid_image": true/false
}}
"""

SAFE_DEFAULT_RESPONSE = {
    "visible_description": "Failed to parse model output.",
    "image_quality_flags": "none",
    "object_part": "unknown",
    "issue_type": "unknown",
    "valid_image": False
}


def strip_thinking(text: str) -> str:
    """
    Strips thinking blocks like <|channel>thought ... <channel|>
    or standard <think>...</think> blocks from response before parsing.
    """
    # Strip <|channel>thought ... <channel|> (including unclosed ones at the end)
    cleaned = re.sub(r'<\|channel>thought.*?(?:<channel\|>|<\|channel\|>|$)', '', text, flags=re.DOTALL)
    # Strip standard <think>...</think>
    cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', cleaned, flags=re.DOTALL)
    # Strip <|think>...</|think>
    cleaned = re.sub(r'<\|think>.*?(?:</\|think>|$)', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def extract_json(text: str) -> str:
    """
    Extracts the JSON substring from response, stripping markdown blocks if present.
    """
    text = text.strip()
    # Strip markdown code block wrappers
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


class ImageAnalystAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "gemma3n:e4b"

    def analyze(self, image_paths: list[str], claim_object: str, 
                user_claim: str, cache: dict) -> dict:
        """
        Analyzes the images under the context of the user claim.
        Returns a dictionary with the analysis results.
        """
        # 1. Encode all images using image_encoder skill
        base64_images = []
        for path in image_paths:
            try:
                b64 = encode_image(path, cache)
                base64_images.append(b64)
            except Exception as e:
                print(f"[ERROR] Failed to encode image {path}: {e}", file=sys.stderr)
                return SAFE_DEFAULT_RESPONSE.copy()

        if not base64_images:
            print("[ERROR] No valid images were encoded.", file=sys.stderr)
            return SAFE_DEFAULT_RESPONSE.copy()

        # 2. Prepare the system prompt
        sys_prompt = PROMPT_TEMPLATE.format(
            claim_object=claim_object,
            user_claim=user_claim
        )

        # 3. Construct Ollama call
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": "Analyze the attached image(s) for the claim. Return the analysis JSON.",
                    "images": base64_images
                }
            ],
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "max_soft_tokens": 560
            },
            "stream": False
        }

        try:
            # 4. Make Ollama multimodal call
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=120)
            response.raise_for_status()
            
            resp_data = response.json()
            raw_content = resp_data.get("message", {}).get("content", "")
            
            # 5. Strip thinking blocks
            cleaned_content = strip_thinking(raw_content)
            
            # 6. Extract JSON content
            json_content = extract_json(cleaned_content)
            
            # 7. Parse and validate JSON structure
            data = json.loads(json_content)
            
            required_keys = ["visible_description", "image_quality_flags", "object_part", "issue_type", "valid_image"]
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"Missing required key: {key}")
                    
            # Ensure valid_image is mapped to a proper boolean
            if not isinstance(data["valid_image"], bool):
                if str(data["valid_image"]).lower() in ("true", "1", "yes"):
                    data["valid_image"] = True
                else:
                    data["valid_image"] = False
                    
            return data
            
        except Exception as e:
            # Log failure loudly
            print(f"[ERROR] Agent 1 failed to analyze or parse response: {e}", file=sys.stderr)
            return SAFE_DEFAULT_RESPONSE.copy()


if __name__ == "__main__":
    from unittest.mock import patch, MagicMock
    import tempfile
    from PIL import Image

    print("Running Agent 1 tests...")

    # Test Case 1: Stripping thinking block
    test_text = "<|channel>thought\nAnalyzing the bumpers...\n<channel|>{\"visible_description\": \"test\", \"image_quality_flags\": \"none\", \"object_part\": \"rear_bumper\", \"issue_type\": \"dent\", \"valid_image\": true}"
    cleaned = strip_thinking(test_text)
    print("Test 1 (strip_thinking):", repr(cleaned))
    assert "<|channel>thought" not in cleaned
    assert "Analyzing" not in cleaned
    assert "visible_description" in cleaned

    # Test Case 2: Extracting JSON
    test_json_md = "```json\n{\"test\": 123}\n```"
    extracted = extract_json(test_json_md)
    print("Test 2 (extract_json):", repr(extracted))
    assert extracted == '{"test": 123}'

    # Test Case 3: Mocked image analysis run
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Create a tiny temp image
        img = Image.new("RGB", (10, 10), color="blue")
        img.save(tmp_path, format="PNG")
        
        agent = ImageAnalystAgent()
        cache = {}
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "<|channel>thought\nLooks like car door damage.\n<channel|>{\"visible_description\": \"scratched door\", \"image_quality_flags\": \"none\", \"object_part\": \"door\", \"issue_type\": \"scratch\", \"valid_image\": true}"
            }
        }
        
        with patch("requests.post", return_value=mock_response) as mock_post:
            res = agent.analyze([tmp_path], "car", "Door scratch", cache)
            mock_post.assert_called_once()
            print("Test 3 (Mocked Analyze):", res)
            assert res["visible_description"] == "scratched door"
            assert res["object_part"] == "door"
            assert res["issue_type"] == "scratch"
            assert res["valid_image"] is True
            
        # Test Case 4: Default Fallback on error
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 200
        mock_response_fail.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Server error or unparseable non-JSON text"
            }
        }
        with patch("requests.post", return_value=mock_response_fail):
            res_fail = agent.analyze([tmp_path], "car", "Door scratch", cache)
            print("Test 4 (Fallback):", res_fail)
            assert res_fail["valid_image"] is False
            assert res_fail["visible_description"] == "Failed to parse model output."
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("All Agent 1 tests completed successfully!")
