import os
import difflib
from typing import Any, Dict, List, Set, Union
from pydantic import BaseModel, Field, model_validator

# Allowed values definition
CLAIM_STATUS_ALLOWED = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE_ALLOWED = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
}

CAR_PARTS_ALLOWED = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown"
}

LAPTOP_PARTS_ALLOWED = {
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown"
}

PACKAGE_PARTS_ALLOWED = {
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown"
}

RISK_FLAGS_ALLOWED = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
}

SEVERITY_ALLOWED = {"none", "low", "medium", "high", "unknown"}

CLAIM_OBJECT_ALLOWED = {"car", "laptop", "package"}


def match_closest(value: Any, allowed_set: Set[str], fallback: str) -> str:
    if value is None:
        return fallback
    # Convert to string, lowercase, strip, and replace spaces/hyphens with underscores
    val_str = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if val_str in allowed_set:
        return val_str
    # Try fuzzy matching
    matches = difflib.get_close_matches(val_str, list(allowed_set), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return fallback


def validate_object_part(part: Any, claim_obj: str) -> str:
    if claim_obj == "car":
        allowed = CAR_PARTS_ALLOWED
    elif claim_obj == "laptop":
        allowed = LAPTOP_PARTS_ALLOWED
    elif claim_obj == "package":
        allowed = PACKAGE_PARTS_ALLOWED
    else:
        allowed = CAR_PARTS_ALLOWED.union(LAPTOP_PARTS_ALLOWED).union(PACKAGE_PARTS_ALLOWED)
    
    return match_closest(part, allowed, "unknown")


def to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in ("true", "1", "yes", "y", "t"):
            return True
        return False
    return False


def validate_supporting_image_ids(ids: Any) -> str:
    if ids is None:
        return "none"
    if isinstance(ids, list):
        items = ids
    elif isinstance(ids, str):
        items = ids.split(";")
    else:
        items = [str(ids)]
        
    cleaned_items = []
    for item in items:
        cleaned_item = str(item).strip()
        if not cleaned_item:
            continue
        # Extract filename without extension if it looks like a path
        base = os.path.basename(cleaned_item)
        name, _ = os.path.splitext(base)
        name_clean = name.strip()
        if name_clean and name_clean.lower() != "none":
            cleaned_items.append(name_clean)
            
    if not cleaned_items:
        return "none"
    return ";".join(cleaned_items)


def validate_risk_flags(flags: Any) -> str:
    if flags is None:
        return "none"
    if isinstance(flags, list):
        items = flags
    elif isinstance(flags, str):
        items = flags.split(";")
    else:
        items = [str(flags)]
        
    validated_flags = set()
    for item in items:
        cleaned_item = str(item).strip()
        if not cleaned_item:
            continue
        cleaned_item = cleaned_item.lower().replace(" ", "_").replace("-", "_")
        if cleaned_item == "none":
            continue
        if cleaned_item in RISK_FLAGS_ALLOWED:
            validated_flags.add(cleaned_item)
        else:
            match = difflib.get_close_matches(cleaned_item, list(RISK_FLAGS_ALLOWED), n=1, cutoff=0.6)
            if match:
                validated_flags.add(match[0])
                
    if not validated_flags:
        return "none"
    validated_flags.discard("none")
    if not validated_flags:
        return "none"
    return ";".join(sorted(list(validated_flags)))


class ClaimRow(BaseModel):
    user_id: str = ""
    image_paths: str = ""
    user_claim: str = ""
    claim_object: str = "package"
    evidence_standard_met: bool = False
    evidence_standard_met_reason: str = ""
    risk_flags: str = "none"
    issue_type: str = "unknown"
    object_part: str = "unknown"
    claim_status: str = "not_enough_information"
    claim_status_justification: str = ""
    supporting_image_ids: str = "none"
    valid_image: bool = False
    severity: str = "unknown"

    @model_validator(mode='before')
    @classmethod
    def clean_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            data = {}
        
        # Extract basic values and clean strings
        user_id = str(data.get("user_id") or "").strip()
        image_paths = str(data.get("image_paths") or "").strip()
        user_claim = str(data.get("user_claim") or "").strip()
        
        claim_object = match_closest(data.get("claim_object"), CLAIM_OBJECT_ALLOWED, "package")
        
        evidence_standard_met = to_bool(data.get("evidence_standard_met"))
        valid_image = to_bool(data.get("valid_image"))
        
        evidence_standard_met_reason = str(data.get("evidence_standard_met_reason") or "").strip()
        claim_status_justification = str(data.get("claim_status_justification") or "").strip()
        
        issue_type = match_closest(data.get("issue_type"), ISSUE_TYPE_ALLOWED, "unknown")
        object_part = validate_object_part(data.get("object_part"), claim_object)
        claim_status = match_closest(data.get("claim_status"), CLAIM_STATUS_ALLOWED, "not_enough_information")
        severity = match_closest(data.get("severity"), SEVERITY_ALLOWED, "unknown")
        
        risk_flags = validate_risk_flags(data.get("risk_flags"))
        supporting_image_ids = validate_supporting_image_ids(data.get("supporting_image_ids"))
        
        return {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": evidence_standard_met,
            "evidence_standard_met_reason": evidence_standard_met_reason,
            "risk_flags": risk_flags,
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": claim_status_justification,
            "supporting_image_ids": supporting_image_ids,
            "valid_image": valid_image,
            "severity": severity,
        }


def validate_output(raw: dict) -> dict:
    """
    Validates all fields and enums for a claim output row.
    Guarantees that it never raises an exception and maintains
    the exact required column order.
    """
    try:
        model = ClaimRow.model_validate(raw)
        model_dict = model.model_dump()
    except Exception:
        # Fallback to defaults if model validation fails entirely
        default_model = ClaimRow()
        model_dict = default_model.model_dump()
        
    order = [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity"
    ]
    return {k: model_dict.get(k) for k in order}


if __name__ == "__main__":
    print("Running output_validator tests...")

    # Test Case 1: Perfectly valid data
    test_1 = {
        "user_id": "user_001",
        "image_paths": "images/sample/case_001/img_1.jpg",
        "user_claim": "Dent on the bumper",
        "claim_object": "car",
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Bumper is visible",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "rear_bumper",
        "claim_status": "supported",
        "claim_status_justification": "Bumper dent visible",
        "supporting_image_ids": "img_1",
        "valid_image": True,
        "severity": "medium"
    }
    res_1 = validate_output(test_1)
    print("Test 1 Result:", res_1)
    assert res_1["claim_object"] == "car"
    assert res_1["object_part"] == "rear_bumper"
    assert res_1["evidence_standard_met"] is True
    assert res_1["severity"] == "medium"
    
    # Test Case 2: Data with typos and incorrect formats
    test_2 = {
        "user_id": "user_002",
        "image_paths": "images/sample/case_002/img_1.jpg",
        "user_claim": "Laptop screen crack",
        "claim_object": "Laptap",  # Typos
        "evidence_standard_met": "true",  # String instead of bool
        "evidence_standard_met_reason": "Screen visible",
        "risk_flags": "blurry-image; wrong-object",  # Separators and typos
        "issue_type": "cracked",  # Needs matching to 'crack'
        "object_part": "scren",  # Scren -> screen
        "claim_status": "supparted",  # Supparted -> supported
        "claim_status_justification": "Screen crack visible",
        "supporting_image_ids": "images/sample/case_002/img_1.jpg",  # Path instead of ID
        "valid_image": "Yes",  # String instead of bool
        "severity": "high"
    }
    res_2 = validate_output(test_2)
    print("\nTest 2 Result:", res_2)
    assert res_2["claim_object"] == "laptop"
    assert res_2["evidence_standard_met"] is True
    assert "blurry_image" in res_2["risk_flags"]
    assert "wrong_object" in res_2["risk_flags"]
    assert res_2["issue_type"] == "crack"
    assert res_2["object_part"] == "screen"
    assert res_2["claim_status"] == "supported"
    assert res_2["supporting_image_ids"] == "img_1"
    assert res_2["valid_image"] is True

    # Test Case 3: Empty / missing data / complete invalid types
    test_3 = {
        "user_id": 12345,
        "claim_object": "invalid_obj",
        "object_part": "engine",  # invalid object part
        "risk_flags": None,
        "evidence_standard_met": "no",
        "severity": "extreme_danger"
    }
    res_3 = validate_output(test_3)
    print("\nTest 3 Result:", res_3)
    assert res_3["user_id"] == "12345"
    assert res_3["claim_object"] == "package"  # fallback object
    assert res_3["object_part"] == "unknown"   # parts list for package doesn't have engine
    assert res_3["risk_flags"] == "none"
    assert res_3["evidence_standard_met"] is False
    assert res_3["severity"] == "unknown"

    print("\nAll tests passed successfully!")
