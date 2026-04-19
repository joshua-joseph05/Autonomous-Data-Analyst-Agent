"""
Main analysis service for DataInsight AI
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait,
)

from app.schemas.schemas import ColumnMetadata
from app.utils.config import INSIGHT_LLM_TIMEOUT_SEC, CLEANED_EXPORT_DIR, RAW_UPLOAD_DIR
from app.cache.storage import CacheStorage
from app.services.cleaning import DataCleaningService
from app.services.correlation import CorrelationService
from app.services.upload import UploadService
from app.services.llm import LLMInferenceService
from app.utils.csv_io import read_csv_bytes
from app.utils.correlation_insight_rules import (
    build_column_correlation_components,
    filter_correlation_insight_lines,
    largest_component_ids,
    numeric_column_names,
)


class AnalysisService:
    """Main service for performing comprehensive data analysis"""

    def __init__(self, cache: CacheStorage):
        self.cache = cache
        self.cleaning = DataCleaningService()
        self.correlation = CorrelationService()
        self.upload = UploadService(cache)

    def analyze_file(self, file_content: bytes, filename: str = "upload.csv") -> Dict[str, Any]:
        """
        Perform complete analysis on uploaded file

        Args:
            file_content: Raw file bytes
            filename: Original upload filename

        Returns:
            Complete analysis result
        """
        preview = self.upload.upload_file(file_content, filename)

        # Keep original bytes so exports still work after restarts / missing cache files
        RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_UPLOAD_DIR / f"{preview.file_id}.csv").write_bytes(file_content)

        df = read_csv_bytes(file_content)

        # Clean data
        df, cleaning_report = self.cleaning.clean_dataframe(df)

        # Optional on-disk cache (same as download output)
        CLEANED_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_EXPORT_DIR / f"{preview.file_id}.csv", index=False)

        # Compute statistics
        summary_stats = self._compute_summary_statistics(df)

        # Compute correlations
        correlation_results = self.correlation.compute_correlations(df)

        # Detect outliers
        outliers = self.correlation.detect_outliers(df)

        # Generate charts info
        charts = self._generate_charts_info(df)

        columns_models = self._convert_columns(df.columns, df)
        columns_payload = [c.model_dump() for c in columns_models]

        # Sample rows for UI (string cells, JSON-safe)
        sample_df = df.head(20).copy()
        cleaned_data = sample_df.astype(object).where(sample_df.notna(), None).values.tolist()
        cleaned_data = [[("" if v is None else str(v)) for v in row] for row in cleaned_data]

        sig_pairs = correlation_results.get("significant_correlations", []) or []

        insights: Dict[str, Any]
        correlation_insights: Dict[str, Any] = {
            "correlated_pairs": [],
            "prediction_ideas": [],
        }
        timeout = float(INSIGHT_LLM_TIMEOUT_SEC)

        def _run_main_insights() -> Dict[str, Any]:
            return self._generate_llm_insights(columns_payload, summary_stats)

        def _run_corr_insights() -> Dict[str, Any]:
            return self._generate_correlation_insights_llm(
                columns_payload, sig_pairs
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_ins = pool.submit(_run_main_insights)
            fut_corr = pool.submit(_run_corr_insights)
            done, pending = wait(
                {fut_ins, fut_corr},
                timeout=timeout,
            )
            for fut in pending:
                fut.cancel()
            if fut_ins in done and fut_ins.done():
                try:
                    insights = fut_ins.result()
                except Exception:
                    insights = LLMInferenceService()._insights_setup_help(
                        "Insight generation failed unexpectedly."
                    )
            else:
                insights = self._insights_timeout_message()
            if fut_corr in done and fut_corr.done():
                try:
                    correlation_insights = fut_corr.result()
                except Exception:
                    correlation_insights = self._fallback_correlation_insights(
                        sig_pairs, columns_payload
                    )
            else:
                correlation_insights = self._fallback_correlation_insights(
                    sig_pairs, columns_payload
                )

        if not correlation_insights.get("correlated_pairs") and sig_pairs:
            correlation_insights = self._fallback_correlation_insights(
                sig_pairs, columns_payload
            )

        # Prepare complete result (cleaned_data also lives under preview for SQLite persistence)
        result = {
            "file_id": preview.file_id,
            "filename": preview.filename,
            "columns": columns_payload,
            "summary_statistics": summary_stats,
            "correlation_matrix": correlation_results.get("correlation_matrix"),
            "significant_correlations": sig_pairs,
            "outliers": outliers,
            "charts": charts,
            "correlation_summary": correlation_results.get("summary"),
            "correlation_insights": correlation_insights,
            "cleaned_data": cleaned_data,
            "cleaning_report": cleaning_report,
            "preview": {"cleaned_data": cleaned_data, "cleaning_report": cleaning_report},
            "insights": insights,
            "created_at": datetime.now().isoformat()
        }

        # Store in cache
        self.cache.store_analysis(preview.file_id, result)

        return result

    def _insights_timeout_message(self) -> Dict[str, Any]:
        """When Ollama is slow or the model is loading; avoid blaming a missing install."""
        sec = int(INSIGHT_LLM_TIMEOUT_SEC)
        return {
            "summary": (
                f"Calling Ollama for insights took longer than {sec} seconds, so the request was stopped. "
                "If Ollama is already running, the model may still be loading into memory or your machine is busy—wait a moment and upload again, "
                "or use a smaller model (for example `ollama pull llama3.2:1b` and set `OLLAMA_MODEL=llama3.2:1b`). "
                f"You can raise the limit with `INSIGHT_LLM_TIMEOUT_SEC` (currently {sec})."
            ),
            "key_findings": [],
            "recommendations": [],
        }

    def _generate_llm_insights(self, columns: list, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured insights via Ollama (bounded time; never blocks upload indefinitely)."""
        def _call() -> Dict[str, Any]:
            llm = LLMInferenceService()
            return llm.generate_insights(columns, summary, samples=None)

        timeout = INSIGHT_LLM_TIMEOUT_SEC
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_call)
                return fut.result(timeout=timeout)
        except FuturesTimeoutError:
            return self._insights_timeout_message()
        except Exception:
            return LLMInferenceService()._insights_setup_help(
                "Insight generation failed unexpectedly."
            )

    @staticmethod
    def _fallback_correlation_insights(
        pairs: list, columns: Optional[list] = None
    ) -> Dict[str, Any]:
        """Short data-driven notes when the correlation LLM is slow or unavailable."""
        correlated: list = []
        predict: list = []
        col_list = columns if isinstance(columns, list) else []
        names = numeric_column_names(
            [c for c in col_list if isinstance(c, dict)]
        )
        col_to_comp, comp_sizes = build_column_correlation_components(names, pairs)
        main_ids = largest_component_ids(comp_sizes)
        largest_size = max(comp_sizes.values()) if comp_sizes else 0
        allow_predict = largest_size >= 3

        for p in pairs[:14]:
            if not isinstance(p, dict):
                continue
            cols = p.get("columns") or []
            if not isinstance(cols, (list, tuple)) or len(cols) < 2:
                continue
            a, b = str(cols[0]), str(cols[1])
            ca, cb = col_to_comp.get(a), col_to_comp.get(b)
            if ca is None or cb is None or ca != cb:
                continue
            r = p.get("correlation")
            try:
                rf = float(r)
            except (TypeError, ValueError):
                continue
            direction = "tend to rise together" if rf > 0 else "one tends to fall as the other rises"
            correlated.append(
                f"{a} and {b} have Pearson r={rf:+.2f} ({direction}); linear association only — not necessarily causal."
            )
            if allow_predict and ca in main_ids:
                predict.append(
                    f"Modeling idea: try predicting {b} from {a} (or the reverse) with regression or tree models; check residual plots and holdout performance."
                )
        if not correlated:
            return {"correlated_pairs": [], "prediction_ideas": []}
        c_f, p_f = filter_correlation_insight_lines(
            correlated, predict, names, pairs
        )
        return {"correlated_pairs": c_f, "prediction_ideas": p_f}

    def _generate_correlation_insights_llm(
        self, columns: list, pairs: list
    ) -> Dict[str, Any]:
        """Correlation-panel copy via Ollama (same host/model as main insights)."""
        if not pairs:
            return {"correlated_pairs": [], "prediction_ideas": []}

        def _call() -> Dict[str, Any]:
            return LLMInferenceService().generate_correlation_insights(columns, pairs)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_call)
                raw = fut.result(timeout=INSIGHT_LLM_TIMEOUT_SEC)
        except (FuturesTimeoutError, Exception):
            return self._fallback_correlation_insights(pairs, columns)
        if not isinstance(raw, dict):
            return self._fallback_correlation_insights(pairs, columns)
        normalized = LLMInferenceService()._normalize_correlation_insights_payload(raw)
        names = numeric_column_names(
            [c for c in columns if isinstance(c, dict)]
        )
        c_f, p_f = filter_correlation_insight_lines(
            normalized.get("correlated_pairs") or [],
            normalized.get("prediction_ideas") or [],
            names,
            pairs,
        )
        normalized["correlated_pairs"] = c_f
        normalized["prediction_ideas"] = p_f
        fb = self._fallback_correlation_insights(pairs, columns)
        if not normalized.get("correlated_pairs"):
            normalized["correlated_pairs"] = fb["correlated_pairs"]
        if not normalized.get("prediction_ideas"):
            normalized["prediction_ideas"] = fb["prediction_ideas"]
        return normalized

    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for all columns"""
        stats = {
            "rows_count": len(df),
            "columns_count": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_cells_percent": float(round(df.isna().mean().mean() * 100, 2))
        }

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        stats["numeric_columns"] = numeric_cols
        stats["categorical_columns"] = categorical_cols

        # Compute stats for each column
        for col in df.columns:
            if col in numeric_cols:
                stats[f"{col}_stats"] = self.cleaning.process_numeric_column(df, col)
            elif col in categorical_cols:
                stats[f"{col}_stats"] = self.cleaning.process_categorical_column(df, col)
            else:
                stats[f"{col}_stats"] = {
                    "name": col,
                    "type": "unknown",
                    "count": int(len(df))
                }

        return stats

    def _convert_columns(self, columns: list, df: pd.DataFrame) -> list[ColumnMetadata]:
        """Convert column info to metadata objects"""
        metadata = []

        for col in columns:
            series = df[col]
            null_count = series.isna().sum()
            unique_count = series.nunique()

            min_max = None
            if pd.api.types.is_numeric_dtype(series):
                min_max = {
                    "min": float(series.min()),
                    "max": float(series.max())
                }

            metadata.append(ColumnMetadata(
                name=str(col),
                inferred_type="numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
                null_count=int(null_count),
                unique_count=int(unique_count),
                min_max=min_max
            ))

        return metadata

    def _generate_charts_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate chart recommendations for all columns"""
        charts = {}

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                charts[col] = {
                    "chart_type": "histogram",
                    "reason": "Numeric data - histogram shows distribution"
                }
            else:
                unique_count = series.nunique()
                if unique_count <= 5:
                    charts[col] = {
                        "chart_type": "bar",
                        "reason": "Low cardinality categorical - bar chart"
                    }
                else:
                    charts[col] = {
                        "chart_type": "bar",
                        "reason": "Categorical data - bar chart shows distribution"
                    }

        return charts

    def get_insights(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get insights for a file (lazy loading from cache)"""
        analysis = self.cache.get_analysis(file_id)
        if analysis:
            analysis["insights"] = self._generate_llm_insights(
                analysis["columns"],
                analysis["summary_statistics"]
            )
        return analysis

    def generate_insights(self, columns: list, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Public wrapper for on-demand insight generation."""
        return self._generate_llm_insights(columns, summary)
