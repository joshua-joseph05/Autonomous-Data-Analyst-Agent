"""
Request/Response schemas for DataInsight AI API
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    """Request schema for file upload"""
    file: Optional[str] = None  # File is passed as multipart/form-data


class ColumnInfo(BaseModel):
    """Information about a dataset column"""
    name: str
    dtype: str
    unique_count: int
    null_count: int
    sample_values: Optional[list] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ColumnMetadata(BaseModel):
    """Metadata for a single column"""
    name: str
    inferred_type: str
    null_count: int
    unique_count: int
    sample_values: Optional[list] = None
    min_max: Optional[dict] = None


class FilePreview(BaseModel):
    """File upload response - preview data"""
    file_id: str
    filename: str
    file_size: int
    rows_count: int
    columns: list[ColumnMetadata]
    preview_data: list[list]  # First 20 rows as list of lists


class AnalysisResponse(BaseModel):
    """Response schema for analysis endpoint"""
    file_id: str
    filename: str
    columns: list[ColumnMetadata]
    summary_statistics: dict
    correlation_matrix: Optional[dict] = None
    outliers: dict
    insights: Optional[dict] = None
    charts: dict
    created_at: datetime


class ChartRequest(BaseModel):
    """Request schema for chart generation"""
    column_name: str
    chart_type: str = Field(default="auto", description="auto, line, bar, histogram, scatter")


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    question: str


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    question: str
    answer: str
    follow_up_questions: List[str] = Field(default_factory=list)
