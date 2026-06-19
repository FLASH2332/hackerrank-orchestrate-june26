import re
import pandas as pd

def load_evidence_requirements(path: str) -> pd.DataFrame:
    """
    Loads evidence requirements from the CSV file.
    Returns a DataFrame with the requirements.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        # Return empty DataFrame with correct column names if loading fails
        return pd.DataFrame(columns=[
            "requirement_id", "claim_object", "applies_to", "minimum_image_evidence"
        ])


def lookup_requirements(df: pd.DataFrame, claim_object: str, issue_family: str) -> str:
    """
    Looks up the minimum image evidence requirement based on claim_object and issue_family.
    Returns the requirement string, or an empty string if not found.
    """
    if df is None or df.empty:
        return ""
        
    obj = str(claim_object).strip().lower()
    fam = str(issue_family).strip().lower()
    
    # 1. Filter candidates by object (exact match or 'all')
    candidates = df[df['claim_object'].str.strip().str.lower().isin([obj, 'all'])].copy()
    if candidates.empty:
        return ""
        
    # Sort candidates so that claim_object matching obj is evaluated before 'all'
    candidates['is_exact_obj'] = candidates['claim_object'].str.strip().str.lower() == obj
    candidates = candidates.sort_values(by='is_exact_obj', ascending=False)
    
    # 2. Check for exact match in applies_to
    for _, row in candidates.iterrows():
        applies_to = str(row['applies_to']).strip().lower()
        if applies_to == fam:
            return str(row['minimum_image_evidence'])
            
    # 3. Check for substring match: does applies_to contain fam, or does fam contain applies_to?
    for _, row in candidates.iterrows():
        applies_to = str(row['applies_to']).strip().lower()
        if fam in applies_to or applies_to in fam:
            return str(row['minimum_image_evidence'])
            
    # 4. Check if any word matches
    fam_words = set(re.findall(r'\w+', fam))
    for _, row in candidates.iterrows():
        applies_to = str(row['applies_to']).strip().lower()
        applies_words = set(re.findall(r'\w+', applies_to))
        
        # Exclude common noise words
        ignore_words = {'or', 'and', 'part', 'damage', 'rows', 'claim', 'review'}
        overlap = (fam_words & applies_words) - ignore_words
        if overlap:
            return str(row['minimum_image_evidence'])
            
    return ""


if __name__ == "__main__":
    print("Running evidence_loader tests...")

    # Mock Dataframe for testing
    columns = ["requirement_id", "claim_object", "applies_to", "minimum_image_evidence"]
    data = [
        ["REQ_GENERAL", "all", "general claim review", "Evidence of general review."],
        ["REQ_CAR_PANEL", "car", "dent or scratch", "Car panel dent/scratch check."],
        ["REQ_CAR_GLASS", "car", "crack, broken, or missing part", "Car glass/light check."],
        ["REQ_LAPTOP_SCREEN", "laptop", "screen, keyboard, or trackpad", "Laptop screen check."],
        ["REQ_LAPTOP_HINGE", "laptop", "hinge, lid, corner, body, or port", "Laptop hinge check."],
        ["REQ_PACKAGE_EXTERIOR", "package", "crushed, torn, or seal damage", "Package exterior check."]
    ]
    df_mock = pd.DataFrame(data, columns=columns)
    
    # Test Case 1: Exact Match
    res_1 = lookup_requirements(df_mock, "car", "dent or scratch")
    print("Test 1 (Exact Match):", res_1)
    assert res_1 == "Car panel dent/scratch check."
    
    # Test Case 2: Substring Match
    res_2 = lookup_requirements(df_mock, "car", "dent")
    print("Test 2 (Substring Match):", res_2)
    assert res_2 == "Car panel dent/scratch check."
    
    # Test Case 3: Word Overlap Match
    res_3 = lookup_requirements(df_mock, "laptop", "cracked screen")
    print("Test 3 (Overlap Match):", res_3)
    assert res_3 == "Laptop screen check."

    # Test Case 4: General review fallback (matches applies_to "general claim review" via substring or words)
    res_4 = lookup_requirements(df_mock, "car", "general claim review")
    print("Test 4 (General Review):", res_4)
    assert res_4 == "Evidence of general review."

    # Test Case 5: No Match
    res_5 = lookup_requirements(df_mock, "car", "completely unrelated issue")
    print("Test 5 (No Match):", repr(res_5))
    assert res_5 == ""

    print("All evidence_loader tests passed!")
