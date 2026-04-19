"""
File upload handling service
"""
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from app.utils.config import METADATA_COLUMN_LIMIT
from app.utils.csv_io import read_csv_bytes
from app.schemas.schemas import ColumnMetadata, FilePreview
from app.cache.storage import CacheStorage


class UploadService:
    """Handle file uploads and validation"""

    def __init__(self, cache: CacheStorage):
        self.cache = cache

    def upload_file(self, file_content: bytes, filename: str = "upload.csv") -> FilePreview:
        """
        Handle file upload and generate preview

        Args:
            file_content: Raw file bytes
            filename: Original client filename (used for display and storage)

        Returns:
            FilePreview with metadata and first 20 rows
        """
        # Generate file ID
        file_id = str(uuid.uuid4())[:8]

        # Try to detect file type
        file_type = self._detect_file_type(file_content)

        if file_type == 'unknown':
            raise ValueError("Unsupported file format. Please upload a CSV file.")

        # Parse CSV (delimiter sniffing for tab/semicolon files named .csv)
        try:
            df = read_csv_bytes(file_content)
            df = df.head(METADATA_COLUMN_LIMIT * 5 + 10)
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or invalid CSV format.")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

        # Validate row count
        if len(df) == 0:
            raise ValueError("CSV file contains no data rows.")

        # Generate column metadata
        columns = self._generate_column_metadata(df)

        # Prepare preview data (first 20 rows as list of lists)
        preview_data = df.head(20).values.tolist()

        safe_name = Path(filename).name or "upload.csv"

        # Store in cache
        preview = {
            "file_id": file_id,
            "filename": safe_name,
            "file_size": len(file_content),
            "rows_count": len(df),
            "columns": [c.model_dump() for c in columns],
            "preview_data": preview_data
        }

        self.cache.store_analysis(file_id, preview)

        return FilePreview(
            file_id=file_id,
            filename=safe_name,
            file_size=len(file_content),
            rows_count=len(df),
            columns=columns,
            preview_data=preview_data
        )

    def _detect_file_type(self, content: bytes) -> str:
        """Detect file type from content"""
        # Check magic bytes
        if content[:4] == b'PK\x03\x04':
            return 'zip'
        if content[:3] == b'\xef\xbb\xbf':
            return 'csv'  # UTF-8 BOM

        # Assume CSV for text content
        try:
            content.decode('utf-8')
            return 'csv'
        except:
            pass

        return 'unknown'

    def _generate_column_metadata(self, df: pd.DataFrame) -> List[ColumnMetadata]:
        """Generate metadata for each column"""
        columns = []

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)

            # Determine inferred type
            inferred_type = self._infer_type(series, dtype)

            # Count nulls and unique values
            null_count = series.isna().sum()
            unique_count = series.nunique()

            # Get sample values
            sample_values = series.dropna().head(5).tolist()

            # Get min/max for numeric columns
            min_max = None
            if pd.api.types.is_numeric_dtype(series):
                min_max = {
                    "min": float(series.min()),
                    "max": float(series.max())
                }

            columns.append(ColumnMetadata(
                name=str(col),
                inferred_type=inferred_type,
                null_count=int(null_count),
                unique_count=int(unique_count),
                sample_values=sample_values,
                min_max=min_max
            ))

        return columns

    def _infer_type(self, series: pd.Series, dtype: str) -> str:
        """Infer column type"""
        # Handle pandas nullable types
        dtype = dtype.replace('>', '')  # Remove signed prefixes

        if 'float' in dtype or 'int' in dtype:
            return 'numeric'
        elif 'datetime' in dtype or 'time' in dtype:
            return 'datetime'
        elif dtype == 'object':
            # Check if mostly strings
            if all(isinstance(v, (str, np.str_)) for v in series.dropna().head(3)):
                return 'categorical'
            return 'text'
        else:
            return 'text'
