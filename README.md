# DataInsight AI

**Autonomous Data Analyst Agent** — upload a CSV, explore it with statistics and correlations, and train simple ML models (linear regression or logistic-style classification) with optional LLM-guided feature selection.

## Features

- **Upload & cleaning** — CSV ingestion, automated cleaning pipeline, preview, and cleaned CSV export.
- **Exploratory analysis** — summary stats, correlation insights, and charts.
- **ML training** — automatic target/feature selection, validation-based feature engineering, and subset search.
- **Iterative agent loop** (optional) — multi-round LLM feature proposals with repeated holdout validation, per-iteration logs, and post-loop refinement (pair champion + subset search).
- **Local LLM (optional)** — [Ollama](https://ollama.com/) for upload-time insights and agent steps when configured.

## Tech stack

| Layer    | Stack |
|----------|--------|
| API      | Python 3, FastAPI, Uvicorn, Pandas, NumPy |
| Web app  | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS |
| Charts   | Plotly.js |
| Storage  | SQLite cache + on-disk CSV exports (configurable paths) |

## Repository layout

```
DataAnalysis-Project/
├── data-insight-ai/
│   ├── backend/          # FastAPI app (`app/main.py`)
│   └── frontend/         # Next.js UI
└── README.md
```

## Prerequisites

- **Python** 3.10+ recommended  
- **Node.js** 18+ and npm  
- **Ollama** (optional) — for LLM-backed insights and the agent loop  

## Quick start

### 1. Backend API

```bash
cd data-insight-ai/backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API serves OpenAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Frontend

```bash
cd data-insight-ai/frontend
npm install
npm run dev
```

Open the app (typically [http://localhost:3000](http://localhost:3000)).

By default the browser talks to the API at `http://localhost:8000`. Override with `NEXT_PUBLIC_API_URL` if the API runs elsewhere.

### 3. Ollama (optional)

Install Ollama, pull a model, then start the daemon. Example:

```bash
ollama pull llama3.2
```

The backend defaults to `OLLAMA_HOST=http://localhost:11434` and `OLLAMA_MODEL=llama3.2`. Adjust if your setup differs.

## Environment variables

### Backend (`data-insight-ai/backend`)

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama server URL (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model tag (default `llama3.2`) |
| `DATABASE_URL` / `SQLITE_DB_PATH` | SQLite location for analysis cache |
| `CLEANED_EXPORT_DIR` | Where cleaned CSVs are written |
| `RAW_UPLOAD_DIR` | Stored original uploads |
| `CORS_ORIGINS` | Allowed browser origins (comma-separated) |

### Frontend (`data-insight-ai/frontend`)

| Variable | Purpose |
|----------|---------|
| `NEXT_PUBLIC_API_URL` | FastAPI root URL (no trailing path segment; default `http://localhost:8000`) |
| `DATAINSIGHT_API_URL` | Server-side proxy to FastAPI (e.g. CSV export route in production) |

## API highlights

- `POST /upload` — upload CSV, returns `file_id` and preview.
- `GET /analysis/{file_id}` — full analysis payload for the dashboard.
- `POST /ml/train/{file_id}` — train model; add query `agent_loop=true` (or `1`) for the iterative agent loop.
