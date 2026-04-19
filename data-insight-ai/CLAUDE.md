# DataInsight AI

Flask-based backend for a web app that analyzes messy CSV data and generates insights using a LOCAL LLM (Ollama).

## Project Overview

DataInsight AI is a production-quality MVP for analyzing CSV datasets with:
- **Backend**: Flask API with modular services
- **Frontend**: Next.js React application
- **LLM**: Ollama (local, no paid APIs)
- **Visualizations**: Plotly charts

## Architecture

```
data-insight-ai/
├── app/
│   ├── main.py              # Flask app factory (blueprints-based)
│   ├── __init__.py
│   ├── services/
│   │   ├── analysis.py      # Main analysis orchestration
│   │   ├── cleaning.py      # Data cleaning & preprocessing
│   │   ├── correlation.py   # Correlation & outlier detection
│   │   ├── llm.py          # Ollama LLM inference
│   │   └── upload.py       # File upload handling
│   ├── utils/
│   │   ├── config.py       # Configuration constants
│   │   └── prompts.py      # LLM prompt templates
│   ├── schemas/
│   │   ├── schemas.py      # Pydantic schemas (ColumnMetadata, FilePreview)
│   │   └── __init__.py
│   └── cache/
│       └── storage.py      # Redis/Memory cache layer
├── uploads/                # Temporary file storage
├── static/                 # Frontend static files
└── templates/              # Jinja2 templates (if used)
```

## Core Features

### 1. File Upload
- Accept CSV uploads via drag-and-drop
- Show preview (first 20 rows)
- Display column names and inferred types
- Generate unique `file_id` for tracking

### 2. Data Cleaning (pandas)
- **Missing values**: Drop columns with >50% nulls
- **Imputation**: Numeric (mean/median), Categorical (mode)
- **Type fixing**: Convert strings to dates/numbers as appropriate
- **Normalization**: Remove duplicates, handle inconsistent casing

### 3. Statistical Analysis (Non-LLM)
- Summary stats (mean, median, std, min, max, count, nulls)
- Correlation matrix (Pearson)
- Distribution analysis (histograms, histograms)
- Outlier detection (IQR method)

### 4. LOCAL LLM Integration (Ollama)

**API Call**:
```python
POST http://localhost:11434/api/generate
{
    "model": "llama3",  # or mistral, etc.
    "prompt": "...",
    "stream": false
}
```

**Send only:**
- Column names
- Data types
- Summary statistics
- 5–10 sample rows

**DO NOT send full dataset** (context overflow).

### 5. LLM Prompt Design

```
You are a data analyst.

Dataset Info:
- Columns: {column names}
- Types: {data types}
- Summary Stats: {mean, median, null counts, etc.}
- Sample Data: {5-10 rows}

Tasks:
1. Explain what this dataset represents
2. Identify trends or correlations
3. Highlight anomalies or unusual values
4. Provide 3–5 actionable insights
5. Suggest potential improvements

Output structured JSON with keys:
- summary
- key_findings
- recommendations
```

### 6. Visualization (Plotly)
- Line charts (time series)
- Bar charts (categories)
- Histograms (distributions)
- Scatter plots (correlations)
- Auto-select chart type based on column types

### 7. Optional Chat Feature
- Ask questions about the uploaded dataset
- Reuse summarized dataset context
- Maintain conversation per file

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload CSV file, return `file_id` |
| GET | `/analysis/{file_id}` | Get analysis results (cleaned data, stats, insights) |
| POST | `/chat/{file_id}` | Chat with dataset about a specific topic |

## Backend Endpoints Details

### POST /upload
```json
// Request
{
    "file_content": "<binary CSV data>"
}

// Response
{
    "file_id": "abc123...",
    "filename": "example.csv",
    "rows_count": 1000,
    "columns": [
        {
            "name": "column_name",
            "inferred_type": "numeric|categorical|datetime|text",
            "null_count": 5,
            "unique_count": 10,
            "sample_values": ["a", "b", "c"],
            "min_max": {"min": 0, "max": 100}
        }
    ],
    "preview_data": [["row1"], ["row2"], ...]
}
```

### GET /analysis/{file_id}
```json
{
    "file_id": "abc123...",
    "filename": "example.csv",
    "rows_count": 1000,
    "cleaned_data": [...],
    "summary_stats": {
        "rows": 1000,
        "missing_cells": 10,
        "numeric_columns": 5,
        "categorical_columns": 3,
        "columns": [...]
    },
    "correlations": {
        "matrix": {...},
        "significant": [...],
        "outliers": {...}
    },
    "insights": {
        "summary": "...",
        "key_findings": [...],
        "recommendations": [...]
    }
}
```

### POST /chat/{file_id}
```json
// Request
{
    "question": "What is the average value?"
}

// Response
{
    "answer": "The average value is...",
    "follow_up_questions": ["..."]
}
```

## Key Patterns

- **Service Layer**: Each `app/services/*.py` follows clean architecture with dedicated methods.
- **Type Hints**: All functions use `typing` module for strict type annotations.
- **Pandas-centric**: Data manipulation uses pandas; validate DataFrame state after transformations.
- **LLM Integration**: Ollama-based inference with fallback responses on error.
- **Caching**: Store analysis results to avoid recomputation.

## Common Workflows

### Data Cleaning Pipeline
1. Drop columns with >50% nulls
2. Convert categorical columns to category type (if <50 unique values)
3. Parse dates with `pd.to_datetime(..., format='mixed')`
4. Remove exact duplicates
5. Impute numeric columns (median → 0)
6. Impute categorical columns (mode → "Unknown")

### Correlation Analysis
1. Filter columns to ≤30% nulls
2. Compute Pearson correlation matrix
3. Flag significant correlations (|r| > 0.7)
4. Use IQR method for outlier detection

### LLM Insights
1. Pass column metadata + summary stats to LLM
2. Parse JSON from markdown code block (````json`)
3. Fallback to text if JSON parsing fails

## Configuration (`app/utils/config.py`)

```python
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # Options: llama3, mistral, etc.
SAMPLE_ROWS = 10
NULL_THRESHOLD = 0.5  # Drop columns with >50% nulls
METADATA_COLUMN_LIMIT = 50
```

## Testing Guidelines

- Always mock external dependencies (LLM, database) in unit tests
- Test cleaning pipeline end-to-end with synthetic data
- Verify correlation results with known datasets
- Check LLM fallback behavior when service unavailable

## Frontend Requirements (Next.js)

### Pages
- `/upload` - CSV upload UI with drag-and-drop
- `/dashboard/{file_id}` - Analysis results dashboard
- `/chat/{file_id}` - Chat interface
- `/settings` - Model selection (llama3, mistral, etc.)

### Components
- `UploadDropzone` - Drag-and-drop file upload
- `DataPreview` - Table with first 20 rows
- `ColumnMetadata` - Display column info (name, type, null count)
- `ChartPanel` - Plotly chart rendering
- `InsightPanel` - LLM insights display
- `ChatInterface` - Chat message list and input

### Styling
- Use Tailwind CSS or CSS modules
- Responsive design for mobile/tablet

## Notes

- `METADATA_COLUMN_LIMIT` controls how many columns to process during upload (prevents LLM context overflow)
- Use median imputation before mean for robustness to outliers
- Categorical type conversion happens after null dropping to avoid category issues with missing values
- Never send full dataset to LLM - only metadata and samples
- Cache analysis results to avoid recomputing for the same `file_id`
