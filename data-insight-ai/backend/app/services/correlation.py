"""
Correlation analysis service for DataInsight AI
"""
import math
import pandas as pd
from typing import Dict, Any, List
from app.utils.config import CORRELATION_STRONG_THRESHOLD


class CorrelationService:
    """Service for computing correlation statistics"""

    def compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute pairwise correlation matrix for numeric columns

        Args:
            df: Cleaned DataFrame

        Returns:
            Correlation results including matrix and significant pairs
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=['number']).copy()

        if numeric_df.empty:
            return {
                "correlation_matrix": {},
                "significant_correlations": [],
                "summary": "No numeric columns found for correlation analysis."
            }

        # Handle columns with too many nulls
        col_null_counts = numeric_df.isna().sum()
        low_null_cols = numeric_df.columns[col_null_counts <= 0.3 * len(df)].tolist()

        if len(low_null_cols) < 2:
            return {
                "correlation_matrix": {},
                "significant_correlations": [],
                "summary": f"Not enough numeric columns with low nulls. Found {len(low_null_cols)} usable columns."
            }

        numeric_df = numeric_df[low_null_cols]

        # Compute correlation matrix
        corr_matrix = numeric_df.corr()

        # Pearson correlation matrix (numeric columns with <=30% nulls only)
        corr_dict = {
            col: {other_col: float(round(val, 3)) if not pd.isna(val) else None
                   for other_col, val in corr_matrix[col].items()}
            for col in corr_matrix.columns
        }

        significant = self._find_significant_correlations(corr_matrix)

        return {
            "correlation_matrix": corr_dict,
            "significant_correlations": significant,
            "summary": self._generate_correlation_summary(corr_matrix)
        }

    def _find_significant_correlations(self, corr_matrix: pd.DataFrame) -> list[Dict[str, Any]]:
        """Return all valid pairwise correlations between numeric columns."""
        significant = []
        n_cols = len(corr_matrix.columns)

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_matrix.loc[col1, col2]

                if corr is not None and not pd.isna(corr):
                    significant.append({
                        "columns": [col1, col2],
                        "correlation": float(round(corr, 3)),
                        "strength": (
                            "strong"
                            if abs(corr) >= CORRELATION_STRONG_THRESHOLD
                            else "weak"
                        )
                    })

        # Sort by absolute correlation value
        significant.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return significant

    @staticmethod
    def significant_pairs_from_matrix(corr_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rebuild sorted pairwise correlations from a stored matrix dict
        (legacy SQLite rows only persisted the matrix).
        """
        if not isinstance(corr_dict, dict) or not corr_dict:
            return []
        cols = [c for c in corr_dict.keys() if isinstance(corr_dict.get(c), dict)]
        significant: List[Dict[str, Any]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                row = corr_dict.get(col1) or {}
                if not isinstance(row, dict):
                    continue
                v = row.get(col2)
                if v is None:
                    continue
                try:
                    corr = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(corr) or math.isinf(corr):
                    continue
                significant.append(
                    {
                        "columns": [col1, col2],
                        "correlation": float(round(corr, 3)),
                        "strength": (
                            "strong"
                            if abs(corr) >= CORRELATION_STRONG_THRESHOLD
                            else "weak"
                        ),
                    }
                )
        significant.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return significant

    def _generate_correlation_summary(self, corr_matrix: pd.DataFrame) -> str:
        """Generate human-readable summary of correlations"""
        if not corr_matrix.empty:
            n_corr = len(self._find_significant_correlations(corr_matrix))
            return (
                f"Found {n_corr} pairwise correlations "
                f"among {len(corr_matrix.columns)} numeric columns."
            )

        return "Unable to compute correlation summary."

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns using IQR method

        Args:
            df: Cleaned DataFrame

        Returns:
            Outlier detection results
        """
        results = {}

        # Select numeric columns
        numeric_df = df.select_dtypes(include=['number'])

        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()

            if len(col_data) < 4:
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

            results[col] = {
                "is_numeric": True,
                "outlier_count": len(outliers),
                "lower_bound": float(round(lower_bound, 2)),
                "upper_bound": float(round(upper_bound, 2)),
                "outlier_percent": float(round(len(outliers) / len(col_data) * 100, 2))
            }

        # For categorical columns, detect rare categories as outliers
        categorical_df = df.select_dtypes(include=['object', 'category']).copy()

        for col in categorical_df.columns:
            value_counts = categorical_df[col].value_counts()
            threshold = max(len(categorical_df[col]) * 0.01, 5)  # 1% or min 5

            rare_values = value_counts[value_counts <= threshold].index.tolist()

            if rare_values:
                results[col] = {
                    "is_numeric": False,
                    "outlier_count": len(rare_values),
                    "outlier_values": rare_values[:10],  # Limit to first 10
                    "note": f"Rare categories (freq <= {threshold}) detected"
                }

        return results

    def find_relationships(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Find columns correlated with a target column

        Args:
            df: DataFrame
            target_column: Target column name

        Returns:
            Relationship analysis results
        """
        target_col = df[target_column]

        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(target_col):
            return {
                "target_column": target_column,
                "method": "categorical",
                "relationships": [],
                "summary": f"Target column '{target_column}' is categorical. Use frequency analysis instead."
            }

        results = []
        numeric_df = df.select_dtypes(include=['number'])

        for col in numeric_df.columns:
            if col == target_column:
                continue

            try:
                corr = df[col].corr(target_col)
                if not pd.isna(corr) and abs(corr) > 0.3:
                    results.append({
                        "column": col,
                        "correlation": float(round(corr, 3)),
                        "direction": "positive" if corr > 0 else "negative",
                        "strength": "strong" if abs(corr) >= 0.7 else "moderate"
                    })
            except Exception:
                continue

        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        summary = ""
        if results:
            top_corr = results[0]
            summary = f"Strongest correlation: {top_corr['column']} ({top_corr['correlation']}) {top_corr['direction']}, {top_corr['strength']}."

        return {
            "target_column": target_column,
            "method": "numeric",
            "relationships": results[:10],  # Top 10
            "summary": summary
        }
