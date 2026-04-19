"""
Configuration settings for DataInsight AI
"""
import os
from pathlib import Path

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Default matches a common Ollama tag; override with OLLAMA_MODEL if yours differs (see `ollama list`).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Max seconds to wait for Ollama during upload-time insight generation (thread pool).
INSIGHT_LLM_TIMEOUT_SEC = float(os.getenv("INSIGHT_LLM_TIMEOUT_SEC", "90"))

# Database: filesystem path for SQLite (used by CacheStorage)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

if os.getenv("SQLITE_DB_PATH"):
    SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH")).expanduser().resolve()
else:
    _db_url = os.getenv("DATABASE_URL", "")
    if _db_url.startswith("sqlite:///"):
        rel = _db_url.replace("sqlite:///", "", 1)
        _p = Path(rel)
        SQLITE_DB_PATH = _p.resolve() if _p.is_absolute() else (_BACKEND_ROOT / rel.lstrip("./")).resolve()
    else:
        SQLITE_DB_PATH = (_BACKEND_ROOT / "analysis_cache.db").resolve()

# Full cleaned datasets (CSV on disk; keyed by file_id). Legacy cache only; prefer raw_uploads.
CLEANED_EXPORT_DIR = (
    Path(os.getenv("CLEANED_EXPORT_DIR", "")).expanduser().resolve()
    if os.getenv("CLEANED_EXPORT_DIR")
    else (_BACKEND_ROOT / "cleaned_exports").resolve()
)

# Original upload bytes (CSV as uploaded). Used to rebuild cleaned exports on demand.
RAW_UPLOAD_DIR = (
    Path(os.getenv("RAW_UPLOAD_DIR", "")).expanduser().resolve()
    if os.getenv("RAW_UPLOAD_DIR")
    else (_BACKEND_ROOT / "raw_uploads").resolve()
)

# Analysis Configuration
SAMPLE_ROWS = 10
NULL_THRESHOLD = 0.5  # Drop columns with >50% nulls
METADATA_COLUMN_LIMIT = 50  # Max columns for LLM context

# Drop entire rows if any numeric cell is outside Tukey fences (same IQR idea as outlier reporting).
REMOVE_OUTLIER_ROWS = os.getenv("REMOVE_OUTLIER_ROWS", "true").lower() in (
    "1",
    "true",
    "yes",
)
OUTLIER_IQR_MULTIPLIER = float(os.getenv("OUTLIER_IQR_MULTIPLIER", "1.5"))

# Sign checks: remove rows with wrong sign if most values share one sign (see cleaning service).
SIGN_ANOMALY_DOMINANCE_FRACTION = float(
    os.getenv("SIGN_ANOMALY_DOMINANCE_FRACTION", "0.85")
)

# After imputation, drop rows that still have any missing value (e.g. NaT in dates).
DROP_ROWS_WITH_REMAINING_NULLS = os.getenv(
    "DROP_ROWS_WITH_REMAINING_NULLS", "true"
).lower() in ("1", "true", "yes")

# Pairwise Pearson |r| above this is listed as "significant" in the UI (default 0.7).
CORRELATION_STRONG_THRESHOLD = float(os.getenv("CORRELATION_STRONG_THRESHOLD", "0.7"))
