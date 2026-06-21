"""
Centralized configuration loader.

Reads all configurable values from environment variables (populated by .env
via python-dotenv). Every module imports from here instead of calling
os.environ.get() directly.

Usage:
    from config import cfg
    print(cfg.OLLAMA_MODEL)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from code/config.py)
_repo_root = Path(__file__).resolve().parent.parent
_env_path = _repo_root / ".env"
load_dotenv(_env_path)


class _Config:
    """Read-once configuration backed by environment variables."""

    # ── Model ──
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llava:7b")
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # ── Cloud API ──
    API_KEY: str = os.environ.get("API_KEY", "")
    API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    CLOUD_MODEL: str = os.environ.get("CLOUD_MODEL", os.environ.get("MODEL", ""))

    # ── Sampling ──
    MODEL_TEMPERATURE: float = float(os.environ.get("MODEL_TEMPERATURE", "1.0"))
    MODEL_TOP_P: float = float(os.environ.get("MODEL_TOP_P", "0.95"))
    MODEL_TOP_K: int = int(os.environ.get("MODEL_TOP_K", "64"))

    # ── Dataset paths ──
    CLAIMS_CSV: str = os.environ.get("CLAIMS_CSV", "dataset/claims.csv")
    SAMPLE_CSV: str = os.environ.get("SAMPLE_CSV", "dataset/sample_claims.csv")
    EVIDENCE_CSV: str = os.environ.get("EVIDENCE_CSV", "dataset/evidence_requirements.csv")
    HISTORY_CSV: str = os.environ.get("HISTORY_CSV", "dataset/user_history.csv")
    IMAGES_DIR: str = os.environ.get("IMAGES_DIR", "dataset")

    # ── Output paths ──
    OUTPUT_CSV: str = os.environ.get("OUTPUT_CSV", "output.csv")
    EVAL_REPORT: str = os.environ.get("EVAL_REPORT", "code/evaluation/evaluation_report.md")

    def __repr__(self) -> str:
        fields = [f for f in dir(self) if f.isupper() and not f.startswith("_")]
        lines = [f"  {f}={getattr(self, f)!r}" for f in sorted(fields)]
        return "Config(\n" + "\n".join(lines) + "\n)"


# Singleton — import this everywhere
cfg = _Config()


if __name__ == "__main__":
    print(cfg)
