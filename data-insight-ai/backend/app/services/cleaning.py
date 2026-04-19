"""
Data cleaning service for DataInsight AI
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from app.utils.config import (
    NULL_THRESHOLD,
    REMOVE_OUTLIER_ROWS,
    OUTLIER_IQR_MULTIPLIER,
    SIGN_ANOMALY_DOMINANCE_FRACTION,
    DROP_ROWS_WITH_REMAINING_NULLS,
)


class DataCleaningService:
    """Service for cleaning and preparing data for analysis"""

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply full cleaning pipeline to DataFrame.

        Returns:
            (cleaned_df, report) where report has JSON-safe stats for this run.
        """
        report: Dict[str, Any] = {
            "rows_start": int(len(df)),
            "columns_start": int(len(df.columns)),
            "null_threshold_percent": int(round(NULL_THRESHOLD * 100)),
            "outlier_row_removal_enabled": bool(REMOVE_OUTLIER_ROWS),
            "drop_remaining_null_rows_enabled": bool(DROP_ROWS_WITH_REMAINING_NULLS),
            "iqr_multiplier": float(OUTLIER_IQR_MULTIPLIER),
            "sign_anomaly_threshold": float(SIGN_ANOMALY_DOMINANCE_FRACTION),
            "columns_dropped_high_null": [],
            "iqr_outlier_rows_removed": 0,
            "sign_anomaly_rows_removed": 0,
            "duplicate_rows_removed": 0,
            "low_cardinality_columns_as_category": [],
            "numeric_columns_filled": [],
            "categorical_columns_filled": [],
            "binary_categorical_columns_encoded": [],
            "single_value_categorical_columns_dropped": [],
            "categorical_columns_dropped_low_association": [],
            "rows_removed_unresolved_nulls": 0,
        }
        df = self.drop_high_null_columns(df, threshold=NULL_THRESHOLD, report=report)
        df = self.fix_data_types(df, report=report)
        if REMOVE_OUTLIER_ROWS:
            df = self.remove_rows_with_numeric_outliers(
                df, OUTLIER_IQR_MULTIPLIER, report=report
            )
            df = self.remove_rows_with_sign_inconsistencies(df, report=report)
        df = self.normalize_data(df, report=report)
        df = self.impute_numeric(df, report=report)
        df = self.impute_categorical(df, report=report)
        df = self.optimize_categorical_features(df, report=report)
        df = self.drop_weakly_related_categorical_columns(df, report=report)
        if DROP_ROWS_WITH_REMAINING_NULLS:
            df = self.remove_rows_with_remaining_nulls(df, report=report)
        # Final categorical pass after all row-level filtering, since some
        # columns may collapse to one level after rows are dropped.
        df = self.optimize_categorical_features(df, report=report)
        report["rows_end"] = int(len(df))
        report["columns_end"] = int(len(df.columns))
        return df, report

    def drop_high_null_columns(
        self,
        df: pd.DataFrame,
        threshold: float = NULL_THRESHOLD,
        report: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Drop columns with percentage of null values above threshold

        Args:
            df: DataFrame to clean
            threshold: Maximum null percentage (default 0.5 = 50%)

        Returns:
            DataFrame with high-null columns dropped
        """
        null_percentages = df.isna().mean()
        high_null_cols = null_percentages[null_percentages > threshold].index.tolist()

        if high_null_cols:
            print(f"Dropping columns with >{threshold*100}% nulls: {high_null_cols}")
        if report is not None:
            report["columns_dropped_high_null"] = [str(c) for c in high_null_cols]

        return df.drop(columns=high_null_cols, errors='ignore')

    def fix_data_types(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fix and improve data type inference

        Args:
            df: DataFrame to fix

        Returns:
            DataFrame with improved data types
        """
        # Coerce object columns that are mostly numeric (common CSV quirk).
        for col in list(df.select_dtypes(include=["object"]).columns):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if float(coerced.notna().mean()) >= 0.85:
                df[col] = coerced

        # Convert categorical columns to proper categories
        for col in df.select_dtypes(include=['object']).columns:
            unique_values = df[col].nunique()
            n_unique = len(df[col].dropna().unique())

            # If categorical has few unique values, convert to category type
            if unique_values < 50:
                df[col] = df[col].astype('category')
                if report is not None:
                    report.setdefault("low_cardinality_columns_as_category", []).append(
                        str(col)
                    )

        # Ensure numeric columns are proper numeric types
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    @staticmethod
    def _association_cat_num(cat: pd.Series, num: pd.Series) -> float:
        """
        Correlation ratio (eta) for categorical -> numeric association in [0,1].
        """
        pair = pd.DataFrame({"c": cat, "n": pd.to_numeric(num, errors="coerce")}).dropna()
        if pair.empty:
            return 0.0
        groups = pair.groupby("c")["n"]
        if groups.ngroups <= 1:
            return 0.0
        overall_mean = pair["n"].mean()
        ss_between = float(
            sum(len(g) * ((g.mean() - overall_mean) ** 2) for _, g in groups)
        )
        ss_total = float(((pair["n"] - overall_mean) ** 2).sum())
        if ss_total <= 0:
            return 0.0
        return float(np.sqrt(max(ss_between / ss_total, 0.0)))

    @staticmethod
    def _association_cat_cat(a: pd.Series, b: pd.Series) -> float:
        """
        Cramer's V for categorical-categorical association in [0,1].
        """
        pair = pd.DataFrame({"a": a, "b": b}).dropna()
        if pair.empty:
            return 0.0
        table = pd.crosstab(pair["a"], pair["b"])
        if table.empty:
            return 0.0
        n = float(table.values.sum())
        if n <= 0:
            return 0.0
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float((((table.values - expected) ** 2) / expected).sum())
        r, k = table.shape
        denom = float(min(r - 1, k - 1))
        if denom <= 0:
            return 0.0
        return float(np.sqrt(max((chi2 / n) / denom, 0.0)))

    def optimize_categorical_features(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Post-imputation categorical optimization:
        - drop categorical columns with only one unique non-null value
        - binary-encode categorical columns with exactly two unique non-null values
        """
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        encoded: List[str] = []
        dropped: List[str] = []

        for col in categorical_cols:
            # Ensure stable string comparison for categories / mixed types.
            non_null = df[col].dropna().map(str)
            unique_vals = sorted(non_null.unique().tolist())
            n_unique = len(unique_vals)

            if n_unique <= 1:
                df = df.drop(columns=[col], errors="ignore")
                dropped.append(str(col))
                continue

            if n_unique == 2:
                # Encode only non-numeric categorical outputs.
                # If both labels are numeric-like already, skip binary encoding.
                numeric_like = pd.to_numeric(pd.Series(unique_vals), errors="coerce")
                both_numeric_like = bool(numeric_like.notna().all())
                if both_numeric_like:
                    continue

                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                # Imputation runs before this, but keep a nullable-safe cast.
                df[col] = df[col].map(lambda x: mapping.get(str(x), np.nan))
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                encoded.append(str(col))

        if report is not None:
            prev_encoded = report.get("binary_categorical_columns_encoded", [])
            prev_dropped = report.get("single_value_categorical_columns_dropped", [])
            report["binary_categorical_columns_encoded"] = sorted(
                {str(x) for x in [*prev_encoded, *encoded]}
            )
            report["single_value_categorical_columns_dropped"] = sorted(
                {str(x) for x in [*prev_dropped, *dropped]}
            )
        if encoded:
            print(f"Binary-encoded categorical columns: {encoded}")
        if dropped:
            print(f"Dropped single-value categorical columns: {dropped}")

        return df

    def drop_weakly_related_categorical_columns(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Drop categorical columns that are not associated with most other features.

        Rule (strict):
        - For each categorical column, compute association to every other feature:
          - categorical vs numeric: correlation ratio (eta)
          - categorical vs categorical: Cramer's V
        - A relation is "meaningful" if association >= 0.30.
        - If fewer than 65% of comparable features are meaningfully related
          (and there are at least 5 comparable features), drop the column.
        """
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        numeric_cols = set(df.select_dtypes(include=["number"]).columns)
        datetime_cols = set(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
        dropped: List[str] = []

        assoc_threshold = 0.30
        min_comparable = 5
        required_fraction = 0.65

        for col in categorical_cols:
            strong_links = 0
            comparable = 0
            c = df[col]

            for other in df.columns:
                if other == col:
                    continue
                if other in datetime_cols:
                    # Datetime columns are sparse/high-cardinality and can
                    # distort categorical relevance scoring.
                    continue
                s = df[other]
                if other in numeric_cols:
                    assoc = self._association_cat_num(c, s)
                else:
                    assoc = self._association_cat_cat(c, s)
                comparable += 1
                if assoc >= assoc_threshold:
                    strong_links += 1

            if comparable < min_comparable:
                continue

            if comparable > 0 and (strong_links / comparable) < required_fraction:
                dropped.append(str(col))

        if dropped:
            df = df.drop(columns=dropped, errors="ignore")
            print(
                "Dropped weakly-related categorical columns "
                f"(assoc<{assoc_threshold} with most features): {dropped}"
            )
        if report is not None:
            report["categorical_columns_dropped_low_association"] = dropped

        return df

    def remove_rows_with_numeric_outliers(
        self,
        df: pd.DataFrame,
        iqr_multiplier: float = 1.5,
        report: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Remove rows that have at least one numeric value outside Tukey fences
        (Q1 - k*IQR, Q3 + k*IQR), using the same k as typical IQR outlier rules.

        Skips columns with fewer than 4 non-null values or zero IQR.
        Does not treat NaNs as anomalies (they are handled later by imputation).
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return df

        outlier_row = pd.Series(False, index=df.index)
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 4:
                continue
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            s = df[col]
            col_out = s.notna() & ((s < lower) | (s > upper))
            outlier_row = outlier_row | col_out

        removed = int(outlier_row.sum())
        if report is not None:
            report["iqr_outlier_rows_removed"] = removed
        if removed > 0:
            print(
                f"Removed {removed} row(s) with numeric IQR outliers "
                f"(any column outside [{iqr_multiplier}*IQR] fences)"
            )
            df = df.loc[~outlier_row].copy()
        return df

    def remove_rows_with_sign_inconsistencies(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Remove rows whose numeric values contradict the dominant sign in a column.

        A column is treated as "should be positive" if either:
        - strictly positive values make up at least SIGN_ANOMALY_DOMINANCE_FRACTION
          of non-null cells, or
        - there is exactly one non-positive value (<= 0) and at least one strictly
          positive value (catches one stray negative among positives).

        Symmetric rules apply for "should be negative" columns. Then drop rows that
        violate that sign (negative in a positive-dominant column, etc.).

        NaNs do not count toward the pattern and are not removed by this rule.
        """
        thr = SIGN_ANOMALY_DOMINANCE_FRACTION
        numeric_cols = df.select_dtypes(include=["number"]).columns
        bad = pd.Series(False, index=df.index)
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 2:
                continue
            s = df[col]
            n = len(col_data)
            pos_n = int((col_data > 0).sum())
            neg_n = int((col_data < 0).sum())
            nonpos_n = int((col_data <= 0).sum())
            nonneg_n = int((col_data >= 0).sum())

            dominant_positive = pos_n / n >= thr
            single_non_positive = nonpos_n == 1 and pos_n >= 1
            if dominant_positive or single_non_positive:
                bad |= s.notna() & (s < 0)

            dominant_negative = neg_n / n >= thr
            single_non_negative = nonneg_n == 1 and neg_n >= 1
            if dominant_negative or single_non_negative:
                bad |= s.notna() & (s > 0)

        removed = int(bad.sum())
        if report is not None:
            report["sign_anomaly_rows_removed"] = removed
        if removed > 0:
            print(
                f"Removed {removed} row(s) with sign inconsistent with "
                f"dominant positive/negative numeric columns (threshold {thr})"
            )
            df = df.loc[~bad].copy()
        return df

    def normalize_data(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Normalize data (date parsing, deduplication, etc.)

        Args:
            df: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        # Try to parse date-like object columns only. Do not coerce clearly numeric
        # strings (common in medical/CSV exports); that creates NaT and later
        # drops every row when null rows are removed.
        for col in df.select_dtypes(include=["object"]).columns:
            s = df[col]
            num_ratio = float(pd.to_numeric(s, errors="coerce").notna().mean())
            if num_ratio >= 0.70:
                continue
            try:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            except Exception:
                continue
            if float(parsed.notna().mean()) < 0.35:
                continue
            df[col] = parsed

        # Remove exact duplicates
        original_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = original_rows - len(df)

        if report is not None:
            report["duplicate_rows_removed"] = int(duplicates_removed)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")

        return df

    def impute_numeric(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Impute missing values in numeric columns

        Args:
            df: DataFrame with numeric columns

        Returns:
            DataFrame with imputed numeric columns
        """
        numeric_cols = df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            if df[col].isna().any():
                if report is not None:
                    report.setdefault("numeric_columns_filled", []).append(str(col))
                # Try median first (more robust to outliers)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

                # If median is also NaN, fill with 0
                if pd.isna(median_val):
                    df[col] = df[col].fillna(0)

        return df

    def impute_categorical(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Impute missing values in categorical columns

        Args:
            df: DataFrame with categorical columns

        Returns:
            DataFrame with imputed categorical columns
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if df[col].isna().any():
                if report is not None:
                    report.setdefault("categorical_columns_filled", []).append(str(col))
                # Use mode (most frequent value) for imputation
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                else:
                    # If no mode, use "Unknown" as default
                    df[col] = df[col].fillna("Unknown")

        return df

    def remove_rows_with_remaining_nulls(
        self, df: pd.DataFrame, report: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Drop rows that still contain any missing value after imputation.

        Numeric and object/category columns are filled by impute_*; datetime columns
        are not imputed here, so NaT and any other remaining NA cause the row to be
        removed instead of leaving a hole in the dataset.
        """
        before = len(df)
        df = df.dropna(how="any")
        removed = before - len(df)
        if report is not None:
            report["rows_removed_unresolved_nulls"] = int(removed)
        if removed > 0:
            print(
                f"Removed {removed} row(s) still containing null/NaT after imputation "
                f"(columns without a filled value)"
            )
        return df

    def process_numeric_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Process a single numeric column for analysis

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Processed column data
        """
        col_data = df[column].dropna()

        return {
            "name": column,
            "type": str(df[column].dtype),
            "count": int(len(col_data)),
            "mean": float(col_data.mean()) if not col_data.empty else None,
            "median": float(col_data.median()) if not col_data.empty else None,
            "std": float(col_data.std()) if not col_data.empty else None,
            "min": float(col_data.min()) if not col_data.empty else None,
            "max": float(col_data.max()) if not col_data.empty else None,
            "quartiles": {
                "q1": float(col_data.quantile(0.25)) if not col_data.empty else None,
                "q2": float(col_data.quantile(0.50)) if not col_data.empty else None,
                "q3": float(col_data.quantile(0.75)) if not col_data.empty else None,
            },
            "unique_count": int(col_data.nunique()) if not col_data.empty else 0,
            "null_count": int(df[column].isna().sum()),
            "distribution": (
                {str(k): int(v) for k, v in col_data.value_counts().head(10).items()}
                if not col_data.empty
                else {}
            ),
        }

    def process_categorical_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Process a single categorical column for analysis

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Processed column data
        """
        col_data = df[column]
        null_count = col_data.isna().sum()
        non_null = col_data.dropna()

        value_counts = non_null.value_counts()
        distribution = {str(k): int(v) for k, v in value_counts.head(10).items()}
        top_value = value_counts.index[0] if not value_counts.empty else None

        return {
            "name": column,
            "type": "categorical",
            "count": int(len(col_data)),
            "unique_count": int(non_null.nunique()),
            "null_count": int(null_count),
            "distribution": distribution,
            "top_values": {str(k): int(v) for k, v in value_counts.head(5).items()},
            "top_value": str(top_value) if top_value else None
        }

    def process_datetime_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Process a datetime column for analysis

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Processed column data
        """
        col_data = df[column]

        try:
            dt_series = pd.to_datetime(col_data, errors='coerce')
            null_count = col_data.isna().sum()
            non_null = col_data.dropna()

            return {
                "name": column,
                "type": "datetime",
                "count": int(len(col_data)),
                "null_count": int(null_count),
                "min_date": str(non_null.min()) if not non_null.empty else None,
                "max_date": str(non_null.max()) if not non_null.empty else None,
                "date_range": str(non_null.max() - non_null.min()) if not non_null.empty else None
            }
        except Exception:
            return {
                "name": column,
                "type": "unknown",
                "count": int(len(col_data)),
                "null_count": int(col_data.isna().sum())
            }
