"""
Pydantic models for DataInsight AI
"""
from datetime import datetime
from typing import Optional


class FileMetadata(BaseModel):
    """File upload metadata"""
    file_id: str
    filename: str
    file_size: int
    upload_time: datetime
    columns: list[dict]


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    file_id: str
    filename: str
    summary_statistics: dict
    correlation_matrix: Optional[dict]
    outliers: dict
    insights: Optional[dict]
    charts: dict
    created_at: datetime


class ChatMessage(BaseModel):
    """Chat message"""
    file_id: str
    question: str
    answer: str
    created_at: datetime
