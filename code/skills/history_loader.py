import pandas as pd

DEFAULT_USER_HISTORY = {
    "user_id": "",
    "past_claim_count": 0,
    "accept_claim": 0,
    "manual_review_claim": 0,
    "rejected_claim": 0,
    "last_90_days_claim_count": 0,
    "history_flags": "none",
    "history_summary": "New user with no prior claim history"
}


def load_user_history(path: str) -> pd.DataFrame:
    """
    Loads user claim history from the CSV file.
    Returns a DataFrame.
    """
    try:
        return pd.read_csv(path)
    except Exception:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=[
            "user_id", "past_claim_count", "accept_claim", 
            "manual_review_claim", "rejected_claim", 
            "last_90_days_claim_count", "history_flags", "history_summary"
        ])


def lookup_user(df: pd.DataFrame, user_id: str) -> dict:
    """
    Looks up a user's claim history by user_id.
    Returns a dictionary of history metrics or a default dictionary if not found.
    """
    if df is None or df.empty or not user_id:
        return DEFAULT_USER_HISTORY.copy()
        
    uid = str(user_id).strip()
    
    # Filter by user_id
    match = df[df['user_id'].astype(str).str.strip() == uid]
    if match.empty:
        # Case-insensitive retry
        match = df[df['user_id'].astype(str).str.strip().str.lower() == uid.lower()]
        if match.empty:
            res = DEFAULT_USER_HISTORY.copy()
            res["user_id"] = uid
            return res
            
    # Extract the first matching row
    row = match.iloc[0]
    
    # Convert fields to correct python types
    res = {
        "user_id": str(row.get("user_id", uid)).strip(),
        "past_claim_count": int(row.get("past_claim_count", 0)),
        "accept_claim": int(row.get("accept_claim", 0)),
        "manual_review_claim": int(row.get("manual_review_claim", 0)),
        "rejected_claim": int(row.get("rejected_claim", 0)),
        "last_90_days_claim_count": int(row.get("last_90_days_claim_count", 0)),
        "history_flags": str(row.get("history_flags", "none")).strip(),
        "history_summary": str(row.get("history_summary", "")).strip()
    }
    return res


if __name__ == "__main__":
    print("Running history_loader tests...")

    # Mock Dataframe for testing
    columns = [
        "user_id", "past_claim_count", "accept_claim", 
        "manual_review_claim", "rejected_claim", 
        "last_90_days_claim_count", "history_flags", "history_summary"
    ]
    data = [
        ["user_001", 2, 2, 0, 0, 1, "none", "Low-risk user"],
        ["user_002", 5, 2, 2, 1, 3, "user_history_risk", "High-risk user"]
    ]
    df_mock = pd.DataFrame(data, columns=columns)
    
    # Test Case 1: Existing User Match
    res_1 = lookup_user(df_mock, "user_002")
    print("Test 1 (Existing User):", res_1)
    assert res_1["user_id"] == "user_002"
    assert res_1["rejected_claim"] == 1
    assert res_1["history_flags"] == "user_history_risk"
    assert res_1["history_summary"] == "High-risk user"
    
    # Test Case 2: Case-Insensitive Match
    res_2 = lookup_user(df_mock, "USER_001")
    print("Test 2 (Case Insensitivity):", res_2)
    assert res_2["user_id"] == "user_001"
    assert res_2["past_claim_count"] == 2
    
    # Test Case 3: Missing User Fallback
    res_3 = lookup_user(df_mock, "user_999")
    print("Test 3 (Fallback/Not Found):", res_3)
    assert res_3["user_id"] == "user_999"
    assert res_3["history_flags"] == "none"
    assert res_3["past_claim_count"] == 0
    assert "no prior claim history" in res_3["history_summary"]

    print("All history_loader tests passed!")
