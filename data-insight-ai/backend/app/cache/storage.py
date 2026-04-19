"""
SQLite-based cache for storing analysis results
"""
import sqlite3
import json
import math
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from app.utils.config import SQLITE_DB_PATH, CLEANED_EXPORT_DIR, RAW_UPLOAD_DIR

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert objects so json.dumps succeeds (incl. numpy dict keys / NaN)."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            x = float(obj)
            return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize_for_json(obj.tolist())
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def _json_dumps(data: Any) -> str:
    return json.dumps(_sanitize_for_json(data), allow_nan=False)


class CacheStorage:
    """SQLite cache for analysis results"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else SQLITE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                preview_json TEXT,
                columns_json TEXT,
                summary_stats_json TEXT,
                correlation_json TEXT,
                outliers_json TEXT,
                insights_json TEXT,
                charts_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES analyses (file_id)
            )
        """)

        conn.commit()
        conn.close()

    def store_analysis(self, file_id: str, data: Dict[str, Any]) -> bool:
        """Store complete analysis result"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            correlation_bundle = {
                "matrix": data.get("correlation_matrix", {}),
                "significant_correlations": data.get("significant_correlations", []),
                "correlation_summary": data.get("correlation_summary"),
                "correlation_insights": data.get("correlation_insights") or {},
            }

            # Insert or replace analysis
            cursor.execute("""
                INSERT OR REPLACE INTO analyses
                (file_id, filename, preview_json, columns_json, summary_stats_json,
                 correlation_json, outliers_json, insights_json, charts_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                data.get("filename", ""),
                _json_dumps(data.get("preview", {})),
                _json_dumps(data.get("columns", [])),
                _json_dumps(data.get("summary_statistics", {})),
                _json_dumps(correlation_bundle),
                _json_dumps(data.get("outliers", {})),
                _json_dumps(data.get("insights", {})),
                _json_dumps(data.get("charts", {})),
                datetime.now().isoformat()
            ))

            conn.commit()
            return True
        finally:
            conn.close()

    def get_analysis(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result by file_id"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT filename, preview_json, columns_json, summary_stats_json,
                       correlation_json, outliers_json, insights_json, charts_json, created_at
                FROM analyses
                WHERE file_id = ?
            """, (file_id,))

            row = cursor.fetchone()
            if row:
                preview = json.loads(row[1]) if row[1] else {}
                cleaned = preview.get("cleaned_data")
                if not isinstance(cleaned, list):
                    cleaned = []
                c_raw = json.loads(row[4]) if row[4] else {}
                if isinstance(c_raw, dict) and "matrix" in c_raw:
                    correlation_matrix = c_raw.get("matrix") or {}
                    significant_correlations = c_raw.get("significant_correlations") or []
                    correlation_summary = c_raw.get("correlation_summary")
                    correlation_insights = c_raw.get("correlation_insights") or {}
                else:
                    from app.services.correlation import CorrelationService

                    correlation_matrix = c_raw if isinstance(c_raw, dict) else {}
                    significant_correlations = (
                        CorrelationService.significant_pairs_from_matrix(
                            correlation_matrix
                        )
                    )
                    correlation_summary = None
                    correlation_insights = {}
                return {
                    "filename": row[0],
                    "preview": preview,
                    "cleaned_data": cleaned,
                    "cleaning_report": preview.get("cleaning_report"),
                    "columns": json.loads(row[2]) if row[2] else [],
                    "summary_statistics": json.loads(row[3]) if row[3] else {},
                    "correlation_matrix": correlation_matrix,
                    "significant_correlations": significant_correlations,
                    "correlation_summary": correlation_summary,
                    "correlation_insights": correlation_insights,
                    "outliers": json.loads(row[5]) if row[5] else {},
                    "insights": json.loads(row[6]) if row[6] else None,
                    "charts": json.loads(row[7]) if row[7] else {},
                    "created_at": row[8]
                }
            return None
        finally:
            conn.close()

    def delete_analysis(self, file_id: str) -> bool:
        """Delete analysis result"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analyses WHERE file_id = ?", (file_id,))
            deleted = cursor.rowcount > 0
            cursor.execute("DELETE FROM chat_messages WHERE file_id = ?", (file_id,))
            conn.commit()
            for base in (CLEANED_EXPORT_DIR, RAW_UPLOAD_DIR):
                export_path = base / f"{file_id}.csv"
                try:
                    if export_path.is_file():
                        export_path.unlink()
                except OSError:
                    pass
            return deleted
        finally:
            conn.close()

    def list_file_ids(self) -> List[str]:
        """Return all stored analysis file IDs."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT file_id FROM analyses ORDER BY created_at DESC")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def store_chat_message(self, file_id: str, question: str, answer: str) -> int:
        """Store chat message"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_messages (file_id, question, answer)
                VALUES (?, ?, ?)
            """, (file_id, question, answer))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_chat_history(self, file_id: str) -> list[Dict[str, Any]]:
        """Get chat history for a file"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT question, answer, created_at
                FROM chat_messages
                WHERE file_id = ?
                ORDER BY created_at DESC
            """, (file_id,))

            return [
                {
                    "question": row[0],
                    "answer": row[1],
                    "created_at": row[2]
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
