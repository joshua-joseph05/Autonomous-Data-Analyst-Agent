"""
FastAPI application for DataInsight AI
"""
import logging
import os
import json
import itertools
from math import comb
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.schemas import (
    FilePreview,
    ColumnMetadata,
    ChatRequest,
    ChatResponse,
)
from app.cache.storage import CacheStorage
from app.utils.config import CLEANED_EXPORT_DIR, RAW_UPLOAD_DIR
from app.utils.csv_io import read_csv_bytes
from app.services.analysis import AnalysisService
from app.services.cleaning import DataCleaningService
from app.services.llm import LLMInferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DataInsight AI",
    description="AI-powered data analysis platform",
    version="1.0.0"
)

_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize cache storage
cache = CacheStorage()

# Initialize analysis service
analysis_service = AnalysisService(cache)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DataInsight AI API",
        "version": "1.0.0"
    }


@app.post("/upload", response_model=FilePreview)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file for analysis

    Args:
        file: The CSV file to upload

    Returns:
        FilePreview with metadata and preview data
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        file_content = await file.read()
        filename = file.filename or "upload.csv"

        result = analysis_service.analyze_file(file_content, filename)

        preview_rows = result.get("cleaned_data") or []
        preview_data = preview_rows[:20]

        cols = result.get("columns") or []
        return FilePreview(
            file_id=result["file_id"],
            filename=result.get("filename", filename),
            file_size=len(file_content),
            rows_count=result.get("summary_statistics", {}).get("rows_count", 0),
            columns=[
                ColumnMetadata(
                    name=c.get("name", ""),
                    inferred_type=c.get("inferred_type", ""),
                    null_count=c.get("null_count", 0),
                    unique_count=c.get("unique_count", 0),
                    sample_values=(c.get("sample_values") or [])[:3],
                    min_max=c.get("min_max"),
                )
                for c in cols
            ],
            preview_data=preview_data,
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{file_id}", response_model=dict)
async def analyze_file(file_id: str):
    """
    Get analysis results for a file

    Args:
        file_id: File ID returned from upload endpoint

    Returns:
        Complete analysis results
    """
    try:
        # Get analysis from cache
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        analysis["file_id"] = file_id
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{file_id}", response_model=dict)
async def get_analysis_alias(file_id: str):
    """Alias for GET /analyze/{file_id} (frontend-friendly path)."""
    return await analyze_file(file_id)


def _load_full_cleaned_dataframe(file_id: str) -> pd.DataFrame:
    """
    Build the full cleaned DataFrame for an uploaded file.

    Prefers rebuilding from raw upload bytes so output always matches
    current cleaning rules.
    """
    if not file_id or "/" in file_id or "\\" in file_id or ".." in file_id:
        raise HTTPException(status_code=400, detail="Invalid file id")

    analysis = cache.get_analysis(file_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Upload the file again.",
        )

    raw_path = RAW_UPLOAD_DIR / f"{file_id}.csv"
    cleaned_path = CLEANED_EXPORT_DIR / f"{file_id}.csv"

    if raw_path.is_file():
        try:
            raw_bytes = raw_path.read_bytes()
            df = read_csv_bytes(raw_bytes)
            df, _ = DataCleaningService().clean_dataframe(df)
            return df
        except Exception as e:
            logger.exception("Could not rebuild cleaned dataframe for %s", file_id)
            raise HTTPException(
                status_code=500,
                detail=f"Could not rebuild cleaned dataframe: {e}",
            ) from e

    if cleaned_path.is_file():
        try:
            return pd.read_csv(cleaned_path)
        except Exception as e:
            logger.exception("Could not load cached cleaned dataframe for %s", file_id)
            raise HTTPException(
                status_code=500,
                detail=f"Could not load cleaned dataframe: {e}",
            ) from e

    raise HTTPException(
        status_code=404,
        detail=(
            "No stored copy of this upload is on the server. "
            "Upload the CSV again."
        ),
    )


@app.get("/exploratory/{file_id}/series")
async def get_exploratory_series(
    file_id: str,
    columns: Optional[str] = Query(
        None,
        description="Comma-separated column names (defaults to all numeric columns).",
    ),
):
    """
    Return full cleaned column series for exploratory charts.

    Response shape:
      {
        file_id: str,
        row_count: int,
        series: { "<col>": [number|null, ...] }
      }
    """
    try:
        df = _load_full_cleaned_dataframe(file_id)
        if columns and columns.strip():
            requested = [c.strip() for c in columns.split(",") if c.strip()]
        else:
            requested = []

        if requested:
            selected: List[str] = [c for c in requested if c in df.columns]
        else:
            selected = list(df.select_dtypes(include=["number"]).columns)

        if not selected:
            return {"file_id": file_id, "row_count": int(len(df)), "series": {}}

        out = {}
        for col in selected:
            s = pd.to_numeric(df[col], errors="coerce")
            vals = [None if pd.isna(v) else float(v) for v in s.tolist()]
            out[str(col)] = vals

        return {
            "file_id": file_id,
            "row_count": int(len(df)),
            "series": out,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exploratory series error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _is_probable_id_column(series: pd.Series, name: str) -> bool:
    n = len(series)
    if n == 0:
        return False
    nunique = int(series.nunique(dropna=True))
    unique_ratio = nunique / max(1, n)
    lname = (name or "").lower()
    if "id" in lname or lname.endswith("_key") or lname.endswith("uuid"):
        return True
    # near-unique numeric columns usually don't generalize as predictors.
    return unique_ratio > 0.95


def _is_coordinate_name(name: str) -> bool:
    n = (name or "").lower()
    return (
        n in {"lat", "latitude", "lon", "lng", "longitude"}
        or "latitude" in n
        or "longitude" in n
    )


def _target_name_prior(name: str) -> float:
    """Heuristic preference for real-world outcome-like target names."""
    n = (name or "").lower()
    outcome_tokens = [
        "price",
        "value",
        "score",
        "count",
        "amount",
        "income",
        "cost",
        "sale",
        "rating",
        "risk",
        "prob",
        "churn",
        "demand",
    ]
    penalty_tokens = ["index", "rank", "code", "zip", "latitude", "longitude", "lat", "lon", "lng", "id"]
    s = 0.0
    for t in outcome_tokens:
        if t in n:
            s += 0.15
    if "price" in n or "value" in n:
        s += 0.25
    for t in penalty_tokens:
        if t in n:
            s -= 0.12
    if _is_coordinate_name(name):
        s -= 0.25
    return s


def _is_likely_leakage_feature(feature: str, target: str) -> bool:
    """Conservative name-based leakage check."""
    f = (feature or "").lower()
    t = (target or "").lower()
    if f == t:
        return True
    bad_tokens = ["target", "label", "prediction", "predicted", "outcome", "response"]
    if any(tok in f for tok in bad_tokens):
        return True
    # If feature name almost contains the target name, likely derived/leaky.
    t_parts = [p for p in t.replace("-", "_").split("_") if len(p) >= 4]
    overlap = sum(1 for p in t_parts if p in f)
    return overlap >= 2


def _ols_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    x_train_i = np.column_stack([np.ones(len(x_train)), x_train])
    x_eval_i = np.column_stack([np.ones(len(x_eval)), x_eval])
    beta, *_ = np.linalg.lstsq(x_train_i, y_train, rcond=None)
    return x_eval_i @ beta


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _is_classification_target(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) < 20:
        return False
    nunique = int(s.nunique(dropna=True))
    if nunique < 2:
        return False
    if nunique > min(30, max(4, int(len(s) * 0.2))):
        return False
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        if np.isnan(vals).all():
            return False
        return bool(np.all(np.isclose(vals, np.round(vals), atol=1e-8)))
    return True


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(labels) == 0:
        return 0.0
    f1s = []
    for lab in labels:
        tp = float(np.sum((y_true == lab) & (y_pred == lab)))
        fp = float(np.sum((y_true != lab) & (y_pred == lab)))
        fn = float(np.sum((y_true == lab) & (y_pred != lab)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _fit_binary_logistic_gd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int = 900,
    lr: float = 0.05,
    l2: float = 1e-4,
    class_weight_balanced: bool = True,
) -> np.ndarray:
    """Binary logistic regression via GD; optional class-balanced sample weights."""
    x = np.column_stack([np.ones(len(x_train)), x_train])
    y = y_train.astype(float)
    n = max(1, len(x))
    if class_weight_balanced and len(np.unique(y)) >= 2:
        n_pos = float(np.sum(y >= 0.5))
        n_neg = float(n - n_pos)
        if n_pos > 0 and n_neg > 0:
            w_pos = n / (2.0 * n_pos)
            w_neg = n / (2.0 * n_neg)
            sw = np.where(y >= 0.5, w_pos, w_neg)
        else:
            sw = np.ones(n, dtype=float)
    else:
        sw = np.ones(n, dtype=float)
    sw_sum = float(np.sum(sw))
    w = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        p = _sigmoid(x @ w)
        err = p - y
        grad = (x.T @ (sw * err)) / sw_sum
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def _predict_binary_proba(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xi = np.column_stack([np.ones(len(x)), x])
    return _sigmoid(xi @ w)


def _predict_multiclass_proba_ovr(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    probs = []
    for lab in labels:
        y_bin = (y_train == lab).astype(float)
        w = _fit_binary_logistic_gd(x_train, y_bin)
        probs.append(_predict_binary_proba(x_eval, w))
    out = np.column_stack(probs)
    denom = np.sum(out, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return out / denom


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    f1 = _f1_macro(y_true, y_pred)
    return {"accuracy": acc, "f1": f1}


def _is_better_classification_score(
    cand_f1: float, cand_acc: float, best_f1: float, best_acc: float, tol: float = 1e-6
) -> bool:
    """Prefer higher F1; break ties with higher accuracy."""
    if cand_f1 > best_f1 + tol:
        return True
    if abs(cand_f1 - best_f1) <= tol and cand_acc > best_acc + tol:
        return True
    return False


def _repeated_validation_out_is_better(
    cand: Tuple[float, dict],
    incumbent: Tuple[float, dict],
    problem_type: str,
    tol: float,
) -> bool:
    """Compare two `_repeated_heldout_score_from_df` results (same protocol)."""
    if problem_type == "classification":
        cf1 = float(cand[1].get("validation_f1", cand[0]))
        cacc = float(cand[1].get("validation_accuracy", 0.0))
        bf1 = float(incumbent[1].get("validation_f1", incumbent[0]))
        bacc = float(incumbent[1].get("validation_accuracy", 0.0))
        return _is_better_classification_score(cf1, cacc, bf1, bacc, tol=tol)
    return float(cand[1].get("validation_r2", cand[0])) > float(
        incumbent[1].get("validation_r2", incumbent[0])
    ) + tol


def _subset_validation_combo_count(pool_size: int, max_k: int) -> int:
    """Count exhaustive subset evaluations for k=2..min(max_k, pool_size)."""
    if pool_size < 2:
        return 0
    mk = min(max_k, pool_size)
    total = 0
    for k in range(2, mk + 1):
        total += comb(pool_size, k)
    return int(total)


def _ordered_subset_from_pool(pool: List[str], names: set) -> List[str]:
    return [c for c in pool if c in names]


def _greedy_validation_feature_subset_search(
    df: pd.DataFrame,
    target: str,
    pool: List[str],
    problem_type: str,
    max_k: int,
    improve_tol: float,
    rng: np.random.Generator,
    *,
    pair_prefix: int = 18,
) -> Tuple[List[str], float, List[Dict[str, Any]]]:
    """
    Strong subset search when exhaustive combination counts are too large.

    Seeds from the best-scoring pair among the first `pair_prefix` pool columns,
    greedily adds features while validation improves, then prunes with backward
    elimination. All scores use `_repeated_heldout_score_from_df`.
    """
    history: List[Dict[str, Any]] = []
    if len(pool) < 2:
        return pool[: max(1, len(pool))], -1e9, history

    prefix = max(2, min(int(pair_prefix), len(pool)))
    best_pair: Optional[List[str]] = None
    best_pair_out: Optional[Tuple[float, dict]] = None

    def _append_hist(feats: List[str], out: Tuple[float, dict], ktag: int) -> None:
        score, aux = out
        history.append(
            {
                "k": int(ktag),
                "features": list(feats),
                "validation_score": float(score),
                "validation_r2": float(aux.get("validation_r2", score)),
                "validation_accuracy": float(aux.get("validation_accuracy", 0.0)),
                "validation_f1": float(aux.get("validation_f1", 0.0)),
                "search": "greedy",
            }
        )

    for i in range(prefix):
        for j in range(i + 1, prefix):
            pair = [pool[i], pool[j]]
            out = _repeated_heldout_score_from_df(df, target, pair, problem_type, rng)
            if not out:
                continue
            _append_hist(pair, out, 2)
            if best_pair_out is None or _repeated_validation_out_is_better(
                out, best_pair_out, problem_type, improve_tol
            ):
                best_pair = pair
                best_pair_out = out

    if best_pair is None or best_pair_out is None:
        fb = pool[:2]
        out = _repeated_heldout_score_from_df(df, target, fb, problem_type, rng)
        if not out:
            return fb, -1e9, history
        best_pair = fb
        best_pair_out = out
        _append_hist(fb, out, 2)

    current = _ordered_subset_from_pool(pool, set(best_pair))
    cur_out = _repeated_heldout_score_from_df(df, target, current, problem_type, rng)
    if not cur_out:
        return current, -1e9, history
    best_subset = list(current)
    best_out = cur_out

    while len(current) < min(max_k, len(pool)):
        best_add: Optional[str] = None
        best_cand_out: Optional[Tuple[float, dict]] = None
        for f in pool:
            if f in current:
                continue
            cand = _ordered_subset_from_pool(pool, set(current) | {f})
            out = _repeated_heldout_score_from_df(df, target, cand, problem_type, rng)
            if not out:
                continue
            _append_hist(cand, out, len(cand))
            if best_cand_out is None or _repeated_validation_out_is_better(
                out, best_cand_out, problem_type, improve_tol
            ):
                best_add = f
                best_cand_out = out
        if best_add is None or best_cand_out is None:
            break
        if not _repeated_validation_out_is_better(
            best_cand_out, cur_out, problem_type, improve_tol
        ):
            break
        current = _ordered_subset_from_pool(pool, set(current) | {best_add})
        cur_out = best_cand_out
        if _repeated_validation_out_is_better(cur_out, best_out, problem_type, improve_tol):
            best_subset = list(current)
            best_out = cur_out

    # Backward pass: drop redundant features if it improves the objective.
    changed = True
    while changed and len(best_subset) > 2:
        changed = False
        base_out = _repeated_heldout_score_from_df(
            df, target, best_subset, problem_type, rng
        )
        if not base_out:
            break
        for f in list(best_subset):
            trial = [x for x in best_subset if x != f]
            trial = _ordered_subset_from_pool(pool, set(trial))
            out = _repeated_heldout_score_from_df(df, target, trial, problem_type, rng)
            if not out:
                continue
            _append_hist(trial, out, len(trial))
            if _repeated_validation_out_is_better(out, base_out, problem_type, improve_tol):
                best_subset = trial
                best_out = out
                changed = True
                break

    best_score = float(best_out[0])
    return best_subset, best_score, history


def _best_binary_prob_threshold(
    y_true: np.ndarray, prob_pos: np.ndarray, pos_code: int, neg_code: int
) -> float:
    """Threshold on P(positive) that maximizes macro F1, then accuracy."""
    best_t = 0.5
    best_f1, best_acc = -1.0, -1.0
    y_int = y_true.astype(int)
    for t in np.linspace(0.05, 0.95, 37):
        y_hat = np.where(prob_pos >= t, pos_code, neg_code).astype(int)
        mm = _classification_metrics(y_int, y_hat)
        if _is_better_classification_score(
            float(mm["f1"]), float(mm["accuracy"]), best_f1, best_acc
        ):
            best_f1 = float(mm["f1"])
            best_acc = float(mm["accuracy"])
            best_t = float(t)
    return best_t


def _rank_features_by_target_relevance(
    df: pd.DataFrame, target: str, features: List[str], problem_type: str = "regression"
) -> List[str]:
    """Order features by target relevance (task-aware)."""
    if not features:
        return []
    keep = [f for f in features if f in df.columns and f != target]
    if not keep:
        return []
    tmp = df[keep + [target]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 8:
        return keep
    if problem_type == "classification":
        y = pd.Categorical(tmp[target]).codes.astype(float)
    else:
        y = pd.to_numeric(tmp[target], errors="coerce").to_numpy(dtype=float)
    scores = []
    for f in keep:
        try:
            x = pd.to_numeric(tmp[f], errors="coerce").to_numpy(dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if np.sum(mask) < 5:
                c = 0.0
            else:
                c = float(abs(np.corrcoef(x[mask], y[mask])[0, 1]))
        except Exception:
            c = 0.0
        if np.isnan(c):
            c = 0.0
        scores.append((f, c))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scores]


def _repeated_heldout_score_on_work(
    work: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    problem_type: str,
    rng: np.random.Generator,
) -> Optional[Tuple[float, dict]]:
    """
    Mean validation score on a fixed row matrix `work` (same rows for all k).
    Regression: R². Classification: maximize F1 then accuracy.
    """
    cols = [c for c in feature_cols if c in work.columns]
    if not cols or target not in work.columns:
        return None
    x_all = work[cols].to_numpy(dtype=float)
    if problem_type == "classification":
        y_all = pd.Categorical(work[target]).codes.astype(int)
    else:
        y_all = work[target].to_numpy(dtype=float)
    n = len(work)
    if n < 40:
        return None
    n_repeats = 4
    train_size = max(25, int(n * 0.7))
    val_size = max(10, int(n * 0.15))
    if n - train_size - val_size <= 0:
        return None

    fold_scores: List[float] = []
    fold_acc: List[float] = []
    fold_f1: List[float] = []
    for _ in range(n_repeats):
        idx = rng.permutation(n)
        x_perm = x_all[idx, :]
        y_perm = y_all[idx]
        x_train = x_perm[:train_size]
        y_train = y_perm[:train_size]
        x_val = x_perm[train_size : train_size + val_size]
        y_val = y_perm[train_size : train_size + val_size]
        if len(x_val) < 5:
            continue
        if problem_type == "classification":
            labels = np.unique(y_train)
            if len(labels) < 2:
                continue
            if len(labels) == 2:
                pos = int(labels[-1])
                neg = int(labels[0])
                y_bin = (y_train == pos).astype(float)
                w = _fit_binary_logistic_gd(x_train, y_bin)
                p_tr = _predict_binary_proba(x_train, w)
                thr = _best_binary_prob_threshold(y_train, p_tr, pos, neg)
                p = _predict_binary_proba(x_val, w)
                y_hat = np.where(p >= thr, pos, neg)
            else:
                probs = _predict_multiclass_proba_ovr(x_train, y_train, x_val, labels)
                y_hat = labels[np.argmax(probs, axis=1)]
            mm = _classification_metrics(y_val, y_hat)
            fold_acc.append(float(mm["accuracy"]))
            fold_f1.append(float(mm["f1"]))
            # Primary objective is F1; accuracy is a tie-breaker.
            fold_scores.append(float(mm["f1"]))
        else:
            pred = _ols_predict(x_train, y_train, x_val)
            fold_scores.append(_r2(y_val, pred))

    if not fold_scores:
        return None
    score = float(np.mean(fold_scores))
    aux: dict = {"validation_score": score}
    if problem_type == "classification":
        aux["validation_accuracy"] = float(np.mean(fold_acc)) if fold_acc else 0.0
        aux["validation_f1"] = float(np.mean(fold_f1)) if fold_f1 else 0.0
    else:
        aux["validation_r2"] = score
    return score, aux


def _repeated_heldout_score_from_df(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    problem_type: str,
    rng: np.random.Generator,
) -> Optional[Tuple[float, dict]]:
    """Same scoring as above, but builds a clean subset from `df` for the given columns."""
    cols = [c for c in feature_cols if c in df.columns]
    if not cols or target not in df.columns:
        return None
    work = df[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    return _repeated_heldout_score_on_work(work, cols, target, problem_type, rng)


def _agent_per_feature_validation_ablation(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    problem_type: str,
    cached_base: Optional[Tuple[float, dict]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Leave-one-feature-out under the same repeated-holdout protocol as the agent loop.

    For each feature f in the current set, score the model with all columns except f.
    Marginal metrics measure how much the full-set validation score drops when f is
    omitted (higher => f mattered more in that multivariate context).

    If `cached_base` is the `(score, aux)` tuple already computed for the full column
    set, it is reused so the headline iteration score is not recomputed.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return None

    base = cached_base
    if base is None:
        base = _repeated_heldout_score_from_df(
            df, target, cols, problem_type, np.random.default_rng(42)
        )
    if not base:
        return None
    _, base_aux = base
    if problem_type == "regression":
        base_primary = float(base_aux.get("validation_r2", base[0]))
    else:
        base_f1 = float(base_aux.get("validation_f1", base[0]))
        base_acc = float(base_aux.get("validation_accuracy", 0.0))

    rows: List[Dict[str, Any]] = []
    for f in cols:
        rest = [c for c in cols if c != f]
        out = _repeated_heldout_score_from_df(
            df, target, rest, problem_type, np.random.default_rng(42)
        )
        if not out:
            continue
        _, wo_aux = out
        if problem_type == "regression":
            wo = float(wo_aux.get("validation_r2", out[0]))
            rows.append(
                {
                    "feature": f,
                    "mean_val_r2_if_removed": wo,
                    "marginal_r2": base_primary - wo,
                }
            )
        else:
            wo_f1 = float(wo_aux.get("validation_f1", out[0]))
            wo_acc = float(wo_aux.get("validation_accuracy", 0.0))
            rows.append(
                {
                    "feature": f,
                    "mean_val_f1_if_removed": wo_f1,
                    "marginal_f1": base_f1 - wo_f1,
                    "mean_val_acc_if_removed": wo_acc,
                    "marginal_acc": base_acc - wo_acc,
                }
            )

    if problem_type == "regression":
        rows.sort(key=lambda r: float(r["marginal_r2"]), reverse=True)
    else:
        rows.sort(
            key=lambda r: (float(r["marginal_f1"]), float(r["marginal_acc"])),
            reverse=True,
        )
    return rows


def _sanitize_col_fragment(name: str) -> str:
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))
    return s[:48] or "col"


def _build_ai_feature_engineering_prompt(
    df: pd.DataFrame, target: str, base_features: List[str], problem_type: str
) -> str:
    feat_lines = []
    for c in base_features[:12]:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        feat_lines.append(
            f"- {c}: missing={float(s.isna().mean()):.3f}, std={float(s.dropna().std() if len(s.dropna()) else 0.0):.6f}"
        )
    features_txt = "\n".join(feat_lines) or "- (none)"
    objective = "maximize validation R²" if problem_type == "regression" else "maximize validation F1 and accuracy"
    return (
        "You are selecting feature engineering candidates for a machine learning model.\n"
        "Propose only relevant derived features created from EXISTING numeric features.\n"
        "Do NOT use constants, external data, or target leakage.\n"
        "Only use binary operations between two existing features.\n"
        "Allowed operators: multiply, divide, add, subtract.\n"
        "Return strict JSON only in this shape:\n"
        "{\n"
        "  \"candidates\": [\n"
        "    {\n"
        "      \"name\": \"short_descriptive_name\",\n"
        "      \"operator\": \"multiply|divide|add|subtract\",\n"
        "      \"feature_a\": \"existing_feature\",\n"
        "      \"feature_b\": \"existing_feature\",\n"
        "      \"reason\": \"why this may improve predictive signal\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Problem type: {problem_type}\n"
        f"Objective: {objective}\n"
        f"Target: {target}\n"
        "Available base features:\n"
        f"{features_txt}\n"
        "Return up to 8 candidates, best first."
    )


def _engineered_feature_candidates(
    df: pd.DataFrame,
    target: str,
    base_features: List[str],
    existing: set,
    problem_type: str,
) -> List[Tuple[str, pd.Series]]:
    """AI-proposed feature engineering from existing feature pairs only."""
    out: List[Tuple[str, pd.Series]] = []
    seen = set(existing)
    keep = [f for f in base_features if f in df.columns and f != target][:12]
    if len(keep) < 2:
        return out

    try:
        llm = LLMInferenceService()
        prompt = _build_ai_feature_engineering_prompt(df, target, keep, problem_type)
        text = llm._ollama_generate_text(prompt)
        parsed = llm._extract_balanced_json(text)
    except Exception:
        return out

    cands = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(cands, list):
        return out

    allowed_ops = {"multiply", "divide", "add", "subtract"}
    for item in cands[:12]:
        if not isinstance(item, dict):
            continue
        op = str(item.get("operator", "")).strip().lower()
        a = str(item.get("feature_a", "")).strip()
        b = str(item.get("feature_b", "")).strip()
        nm = str(item.get("name", "")).strip()
        if op not in allowed_ops or a not in keep or b not in keep or a == b:
            continue

        xa = pd.to_numeric(df[a], errors="coerce")
        xb = pd.to_numeric(df[b], errors="coerce")
        if op == "multiply":
            series = (xa * xb).astype(float)
            tag = "mul"
        elif op == "add":
            series = (xa + xb).astype(float)
            tag = "add"
        elif op == "subtract":
            series = (xa - xb).astype(float)
            tag = "sub"
        else:
            if float(xb.abs().median(skipna=True)) < 1e-8:
                continue
            series = (xa / xb.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).astype(float)
            tag = "ratio"

        if float(series.dropna().std()) < 1e-12:
            continue

        prefix = _sanitize_col_fragment(nm) if nm else f"{_sanitize_col_fragment(a)}__{_sanitize_col_fragment(b)}"
        col_name = f"__eng_{tag}__{prefix}"
        if len(col_name) >= 180 or col_name in seen:
            col_name = f"__eng_{tag}__{_sanitize_col_fragment(a)}__{_sanitize_col_fragment(b)}"
        if len(col_name) >= 180 or col_name in seen:
            continue
        out.append((col_name, series))
        seen.add(col_name)

    return out[:28]


def _maybe_apply_feature_engineering(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    problem_type: str,
    rng: np.random.Generator,
    max_add: int = 5,
) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Greedily add engineered columns only if repeated holdout score improves.
    Uses the same objective as feature growth (R² vs F1/accuracy blend).
    """
    improve_tol = 0.005 if problem_type == "regression" else 0.002
    meta: dict = {
        "attempted": True,
        "source": "ai",
        "added_columns": [],
        "baseline_validation_score": None,
        "final_validation_score": None,
        "steps": [],
    }

    base = [f for f in features if f in df.columns and f != target]
    if len(base) < 2:
        meta["attempted"] = False
        meta["note"] = "Not enough base features for engineering."
        return df, features, meta

    out_df = df.copy()
    current_features = list(base)
    source_features = list(base)

    base_score_t = _repeated_heldout_score_from_df(
        out_df, target, current_features, problem_type, rng
    )
    if base_score_t is None:
        meta["note"] = "Could not compute baseline validation score (too few rows)."
        return df, features, meta

    baseline_score, _ = base_score_t
    meta["baseline_validation_score"] = float(baseline_score)
    best_score = baseline_score

    for _round in range(max_add):
        # Build engineered candidates only from the original selected features.
        # Do not use previously engineered columns to create new ones.
        candidates = _engineered_feature_candidates(
            out_df, target, source_features, set(out_df.columns), problem_type
        )
        best_name = None
        best_cand_score = best_score
        best_series: Optional[pd.Series] = None

        for name, series in candidates:
            if name in out_df.columns:
                continue
            trial = out_df.copy()
            trial[name] = series
            sc_t = _repeated_heldout_score_from_df(
                trial, target, current_features + [name], problem_type, rng
            )
            if sc_t is None:
                continue
            sc, _ = sc_t
            if sc > best_cand_score + improve_tol:
                best_cand_score = sc
                best_name = name
                best_series = series

        if best_name is None or best_series is None:
            break

        out_df[best_name] = best_series
        current_features.append(best_name)
        meta["added_columns"].append(best_name)
        meta["steps"].append(
            {
                "column": best_name,
                "validation_score_after": float(best_cand_score),
            }
        )
        best_score = best_cand_score

    meta["final_validation_score"] = float(best_score)
    if not meta["added_columns"]:
        meta["note"] = "AI feature engineering ran, but no proposed feature improved validation score."
    return out_df, current_features, meta


def _choose_feature_count_by_validation(
    df: pd.DataFrame, target: str, ordered_features: List[str], problem_type: str = "regression"
) -> Tuple[List[str], dict]:
    """
    Start from top 3 features; add one feature at a time while validation score
    improves. Regression uses R². Classification maximizes F1, then accuracy.
    """
    work = (
        df[ordered_features + [target]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(work) < 40 or len(ordered_features) <= 3:
        return ordered_features, {
            "feature_growth": {"start_k": min(3, len(ordered_features)), "stopped_at_k": len(ordered_features), "validation_r2_by_k": []}
        }

    n = len(work)
    start_k = min(3, len(ordered_features))
    best_k = start_k
    best_score = -1.0
    best_f1 = -1.0
    best_acc = -1.0
    history = []
    improve_tol = 0.005 if problem_type == "regression" else 1e-6

    # Repeated shuffled holdout gives a more stable R² estimate than one split.
    # This reduces the chance that weak features are kept due to random noise.
    rng = np.random.default_rng(42)
    train_size = max(25, int(n * 0.7))
    val_size = max(10, int(n * 0.15))
    if n - train_size - val_size <= 0:
        return ordered_features, {
            "feature_growth": {"start_k": min(3, len(ordered_features)), "stopped_at_k": len(ordered_features), "validation_r2_by_k": []}
        }

    for k in range(start_k, len(ordered_features) + 1):
        out = _repeated_heldout_score_on_work(
            work, ordered_features[:k], target, problem_type, rng
        )
        if out is None:
            continue
        r2v, aux = out
        row = {"k": int(k), "validation_score": float(r2v)}
        if problem_type == "classification":
            row["validation_accuracy"] = float(aux.get("validation_accuracy", 0.0))
            row["validation_f1"] = float(aux.get("validation_f1", 0.0))
        else:
            row["validation_r2"] = float(r2v)
        history.append(row)
        if problem_type == "classification":
            cand_f1 = float(aux.get("validation_f1", 0.0))
            cand_acc = float(aux.get("validation_accuracy", 0.0))
            if _is_better_classification_score(
                cand_f1, cand_acc, best_f1, best_acc, tol=improve_tol
            ):
                best_f1 = cand_f1
                best_acc = cand_acc
                best_score = r2v
                best_k = k
        elif r2v > best_score + improve_tol:
            best_score = r2v
            best_k = k
        else:
            # stop when adding the next feature no longer improves performance
            if k > best_k:
                break

    selected = ordered_features[:best_k]
    meta = {
        "feature_growth": {
            "start_k": int(start_k),
            "stopped_at_k": int(best_k),
            "objective": "r2" if problem_type == "regression" else "maximize_f1_then_accuracy",
            "validation_r2_by_k": history,
        }
    }
    return selected, meta


def _optimize_feature_subset_by_validation(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    problem_type: str = "regression",
    *,
    max_pool: int = 12,
    max_k: int = 8,
    max_exhaustive_subset_evals: int = 5500,
    regression_improve_tol: float = 0.001,
) -> Tuple[List[str], dict]:
    """
    Search over feature subsets (bounded) and keep the subset with best
    repeated-holdout validation objective.

    Uses exhaustive enumeration when the number of subset scores to evaluate stays
    below `max_exhaustive_subset_evals`; otherwise falls back to a greedy search
    (pair seed + forward selection + backward pruning) on the same pool.
    """
    keep = [f for f in features if f in df.columns and f != target]
    if len(keep) <= 2:
        return keep, {"subset_optimization": {"applied": False, "reason": "too_few_features"}}

    ranked = _rank_features_by_target_relevance(df, target, keep, problem_type=problem_type)
    pool = ranked[: max(2, min(int(max_pool), len(ranked)))]
    if len(pool) <= 2:
        return pool, {"subset_optimization": {"applied": False, "reason": "too_few_features_after_pool"}}

    improve_tol = 1e-6 if problem_type == "classification" else float(regression_improve_tol)
    rng = np.random.default_rng(42)
    max_k_eff = min(int(max_k), len(pool))
    combo_count = _subset_validation_combo_count(len(pool), max_k_eff)
    strategy = "exhaustive"
    history: List[Dict[str, Any]] = []
    best_subset: List[str] = pool[: min(3, len(pool))]
    best_score = -1e9
    best_f1 = -1.0
    best_acc = -1.0

    if combo_count <= int(max_exhaustive_subset_evals):
        for k in range(2, max_k_eff + 1):
            for combo in itertools.combinations(pool, k):
                subset = list(combo)
                out = _repeated_heldout_score_from_df(df, target, subset, problem_type, rng)
                if out is None:
                    continue
                score, aux = out
                history.append(
                    {
                        "k": int(k),
                        "features": subset,
                        "validation_score": float(score),
                        "validation_r2": float(aux.get("validation_r2", score)),
                        "validation_accuracy": float(aux.get("validation_accuracy", 0.0)),
                        "validation_f1": float(aux.get("validation_f1", 0.0)),
                        "search": "exhaustive",
                    }
                )
                if problem_type == "classification":
                    cand_f1 = float(aux.get("validation_f1", 0.0))
                    cand_acc = float(aux.get("validation_accuracy", 0.0))
                    if _is_better_classification_score(
                        cand_f1, cand_acc, best_f1, best_acc, tol=improve_tol
                    ):
                        best_f1 = cand_f1
                        best_acc = cand_acc
                        best_score = float(score)
                        best_subset = subset
                elif score > best_score + improve_tol:
                    best_score = float(score)
                    best_subset = subset
    else:
        strategy = "greedy"
        best_subset, best_score, history = _greedy_validation_feature_subset_search(
            df,
            target,
            pool,
            problem_type,
            max_k_eff,
            improve_tol,
            rng,
            pair_prefix=min(18, len(pool)),
        )

    if not history:
        return keep, {"subset_optimization": {"applied": False, "reason": "no_valid_scored_subsets"}}

    history.sort(key=lambda x: x.get("validation_score", -1e9), reverse=True)
    return best_subset, {
        "subset_optimization": {
            "applied": True,
            "objective": "r2" if problem_type == "regression" else "maximize_f1_then_accuracy",
            "search_strategy": strategy,
            "subset_combo_count": int(combo_count),
            "max_exhaustive_subset_evals": int(max_exhaustive_subset_evals),
            "searched_pool_size": int(len(pool)),
            "max_k": int(max_k_eff),
            "selected_feature_count": int(len(best_subset)),
            "best_validation_score": float(best_score),
            "top_candidates": history[:12],
        }
    }


def _build_ai_target_feature_prompt(df: pd.DataFrame) -> str:
    num_df = df.select_dtypes(include=["number"]).copy()
    corr = num_df.corr().fillna(0.0) if num_df.shape[1] > 1 else pd.DataFrame()
    top_pairs = []
    if not corr.empty:
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                v = float(corr.loc[a, b])
                top_pairs.append((abs(v), a, b, v))
        top_pairs.sort(reverse=True, key=lambda x: x[0])
    top_pairs_txt = "\n".join(
        [f"- {a} vs {b}: r={v:+.3f}" for _, a, b, v in top_pairs[:20]]
    ) or "- (not enough numeric columns)"

    col_lines = []
    for c in df.columns:
        s = df[c]
        inferred = "numeric" if pd.api.types.is_numeric_dtype(s) else (
            "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "categorical"
        )
        col_lines.append(
            f"- {c}: type={inferred}, unique={int(s.nunique(dropna=True))}, missing={float(s.isna().mean()):.3f}"
        )
    cols_txt = "\n".join(col_lines[:120])

    rubric = (
        "You are given a dataset. Your task is to select an appropriate target variable for a supervised machine learning model.\n\n"
        "Follow these steps:\n\n"
        "Identify all candidate variables that could serve as a target:\n"
        "Prefer variables that represent outcomes (e.g., price, score, count, probability)\n"
        "Avoid identifiers (IDs, coordinates unless meaningful)\n"
        "Avoid variables that are clearly inputs or features\n"
        "Apply these selection criteria:\n"
        "The target should be meaningful to predict in a real-world context\n"
        "It should plausibly depend on other variables (cause → effect direction)\n"
        "It should not be derived directly from other variables (avoid leakage)\n"
        "It should have sufficient variability (not constant or near-constant)\n"
        "Detect and avoid data leakage:\n"
        "Exclude variables that directly encode or are computed from the target\n"
        "Exclude post-outcome variables\n"
        "Rank candidate targets based on:\n"
        "Predictability (correlation or relationship with other features)\n"
        "Interpretability (is the prediction useful?)\n"
        "Data quality (missing values, noise)\n"
        "Select the best target variable and justify your choice.\n"
        "Identify input features:\n"
        "Include variables that are not the target and not leaking information\n"
        "Optionally remove highly redundant features\n"
    )

    return (
        f"{rubric}\n\n"
        "Dataset profile:\n"
        f"Rows: {len(df)}\nColumns: {len(df.columns)}\n\n"
        "Columns:\n"
        f"{cols_txt}\n\n"
        "Top numeric correlations:\n"
        f"{top_pairs_txt}\n\n"
        "Respond as strict JSON only with keys:\n"
        "{\n"
        "  \"target_column\": \"...\",\n"
        "  \"feature_columns\": [\"...\", \"...\"],\n"
        "  \"justification\": \"...\",\n"
        "  \"ranked_candidate_targets\": [{\"column\":\"...\",\"reason\":\"...\"}]\n"
        "}\n"
    )


def _ai_select_regression_target_and_features(
    df: pd.DataFrame,
) -> Optional[Tuple[str, List[str], dict]]:
    """Use LLM to select target/features; return None if invalid/unavailable."""
    try:
        llm = LLMInferenceService()
        prompt = _build_ai_target_feature_prompt(df)
        text = llm._ollama_generate_text(prompt)
        parsed = llm._extract_balanced_json(text)
        if not isinstance(parsed, dict):
            return None

        target = str(parsed.get("target_column") or "").strip()
        features_raw = parsed.get("feature_columns")
        if not target or not isinstance(features_raw, list):
            return None

        num_df = df.select_dtypes(include=["number"]).copy()
        if target not in num_df.columns:
            return None

        features = []
        for f in features_raw:
            ff = str(f).strip()
            if not ff or ff == target:
                continue
            if ff not in num_df.columns:
                continue
            if _is_probable_id_column(num_df[ff], ff):
                continue
            if _is_likely_leakage_feature(ff, target):
                continue
            features.append(ff)
        # de-dup preserve order
        seen = set()
        features = [x for x in features if not (x in seen or seen.add(x))]
        if len(features) < 2:
            return None

        # rank AI-selected features by empirical relevance to target
        features = _rank_features_by_target_relevance(num_df, target, features)

        meta = {
            "source": "ai",
            "justification": str(parsed.get("justification") or "").strip(),
            "ranked_candidate_targets": parsed.get("ranked_candidate_targets") or [],
        }
        return target, features[:12], meta
    except Exception:
        return None


def _select_regression_target_and_features(df: pd.DataFrame) -> Tuple[str, List[str], dict]:
    """
    Generic target/feature selection:
    - enumerate plausible numeric targets
    - run feature filtering + forward selection per target
    - rank targets by predictability + interpretability + data quality
    """
    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least two numeric columns to train a model.",
        )

    candidate_cols = [c for c in num_df.columns if not _is_probable_id_column(num_df[c], str(c))]
    if len(candidate_cols) < 2:
        candidate_cols = list(num_df.columns)
    cand_df = num_df[candidate_cols].copy()
    corr = cand_df.corr().fillna(0.0).abs()
    np.fill_diagonal(corr.values, 0.0)

    candidate_summaries = []
    best_bundle: Optional[Tuple[str, List[str], float]] = None

    for target in corr.columns:
        tname = str(target)
        tseries = cand_df[target]
        if float(tseries.std(ddof=0) or 0.0) <= 1e-12:
            continue
        if _is_coordinate_name(tname):
            continue

        rel = corr[target].sort_values(ascending=False)
        feature_pool = [
            str(c)
            for c in rel.index
            if c != target
            and rel[c] >= 0.08
            and not _is_probable_id_column(cand_df[c], str(c))
            and not _is_likely_leakage_feature(str(c), tname)
        ]
        if not feature_pool:
            feature_pool = [str(c) for c in rel.index if c != target][: min(10, len(rel) - 1)]
        if not feature_pool:
            continue

        # Remove highly redundant features.
        pruned: List[str] = []
        for f in feature_pool:
            keep = True
            for g in pruned:
                pair_corr = float(abs(cand_df[[f, g]].corr().iloc[0, 1]))
                if pair_corr >= 0.95:
                    if rel[f] > rel[g]:
                        pruned.remove(g)
                        keep = True
                    else:
                        keep = False
                    break
            if keep:
                pruned.append(f)
        feature_pool = pruned[:12]
        if not feature_pool:
            continue

        work = cand_df[feature_pool + [target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(work) < 40:
            continue

        x_all = work[feature_pool].to_numpy(dtype=float)
        y_all = work[target].to_numpy(dtype=float)
        n = len(work)
        train_end = max(25, int(n * 0.7))
        val_end = max(train_end + 10, int(n * 0.85))
        if val_end >= n:
            val_end = n - 1
        x_train, y_train = x_all[:train_end], y_all[:train_end]
        x_val, y_val = x_all[train_end:val_end], y_all[train_end:val_end]
        if len(x_val) < 5:
            continue

        # Forward feature selection (validation R² gain).
        selected: List[str] = []
        selected_idx: List[int] = []
        best_val_r2 = -1.0
        remaining = list(range(len(feature_pool)))

        while remaining and len(selected) < 8:
            best_gain = 0.0
            best_j = None
            best_r2 = best_val_r2
            for j in remaining:
                trial_idx = selected_idx + [j]
                pred = _ols_predict(x_train[:, trial_idx], y_train, x_val[:, trial_idx])
                r2v = _r2(y_val, pred)
                gain = r2v - best_val_r2
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    best_r2 = r2v
            if best_j is None or best_gain < 0.003:
                break
            selected_idx.append(best_j)
            selected.append(feature_pool[best_j])
            remaining.remove(best_j)
            best_val_r2 = best_r2

        if not selected:
            selected = feature_pool[: min(4, len(feature_pool))]
            selected_idx = [feature_pool.index(f) for f in selected]
            pred = _ols_predict(x_train[:, selected_idx], y_train, x_val[:, selected_idx])
            best_val_r2 = _r2(y_val, pred)

        # Composite target score
        interpretability = _target_name_prior(tname)
        data_quality = 0.0 if float(tseries.isna().mean()) > 0.15 else 0.08
        predictability = max(best_val_r2, -0.2)
        composite = 0.75 * predictability + 0.15 * interpretability + 0.10 * data_quality

        candidate_summaries.append(
            {
                "column": tname,
                "score": float(composite),
                "validation_r2": float(best_val_r2),
                "selected_feature_count": int(len(selected)),
            }
        )

        if best_bundle is None or composite > best_bundle[2]:
            best_bundle = (tname, selected, float(composite))

    if best_bundle is None:
        raise HTTPException(
            status_code=400,
            detail="Could not find a reliable target/feature combination for this dataset.",
        )

    candidate_summaries.sort(key=lambda x: x["score"], reverse=True)
    target, selected_features, _ = best_bundle
    selected_features = _rank_features_by_target_relevance(cand_df, target, selected_features)
    return target, selected_features, {
        "candidate_targets_ranked": candidate_summaries[:8]
    }


def _train_linear_regression_closed_form(
    df: pd.DataFrame, target: str, features: List[str]
) -> dict:
    """Train/test split + OLS fit via normal equation (with intercept)."""
    work = df[features + [target]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 30:
        raise HTTPException(
            status_code=400,
            detail="Not enough clean rows to train a model after filtering missing values.",
        )

    X = work[features].to_numpy(dtype=float)
    y = work[target].to_numpy(dtype=float)

    n = len(work)
    idx = np.random.default_rng(42).permutation(n)
    X = X[idx]
    y = y[idx]
    split = max(20, int(n * 0.8))
    if n - split < 10:
        split = n - 10
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Add intercept column
    X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
    X_test_i = np.column_stack([np.ones(len(X_test)), X_test])
    X_all_i = np.column_stack([np.ones(len(X)), X])

    # Stable least squares solve
    beta, *_ = np.linalg.lstsq(X_train_i, y_train, rcond=None)
    y_pred = X_test_i @ beta
    y_pred_all = X_all_i @ beta

    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    coeffs = [
        {"feature": features[i], "coefficient": float(beta[i + 1])}
        for i in range(len(features))
    ]
    coeffs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return {
        "model_type": "linear_regression",
        "problem_type": "regression",
        "target_column": target,
        "feature_columns": features,
        "rows_used": int(len(work)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "evaluation": {
            # Full dataset used by the fitted model (after NA/inf filtering).
            "y_true": [float(v) for v in y.tolist()],
            "y_pred": [float(v) for v in y_pred_all.tolist()],
        },
        "coefficients": coeffs,
        "sample_predictions": [
            {"actual": float(a), "predicted": float(p)}
            for a, p in list(zip(y_test[:20], y_pred[:20]))
        ],
    }


def _train_logistic_classification(
    df: pd.DataFrame, target: str, features: List[str]
) -> dict:
    """Train/test split + logistic classification (binary or one-vs-rest multiclass)."""
    work = df[features + [target]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 30:
        raise HTTPException(
            status_code=400,
            detail="Not enough clean rows to train a classification model.",
        )

    X = work[features].to_numpy(dtype=float)
    y_labels = pd.Categorical(work[target])
    classes = np.array([str(c) for c in y_labels.categories])
    y = y_labels.codes.astype(int)
    if len(classes) < 2:
        raise HTTPException(
            status_code=400,
            detail="Classification target must contain at least 2 classes.",
        )

    n = len(work)
    idx = np.random.default_rng(42).permutation(n)
    X = X[idx]
    y = y[idx]
    split = max(20, int(n * 0.8))
    if n - split < 10:
        split = n - 10
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Training split has only one class; cannot train classifier.",
        )

    if len(classes) == 2:
        un = np.unique(y_train)
        pos, neg = int(un[-1]), int(un[0])
        w = _fit_binary_logistic_gd(X_train, (y_train == pos).astype(float))
        p_train = _predict_binary_proba(X_train, w)
        thr = _best_binary_prob_threshold(y_train, p_train, pos, neg)
        p_test = _predict_binary_proba(X_test, w)
        p_all = _predict_binary_proba(X, w)
        y_pred_test = np.where(p_test >= thr, pos, neg).astype(int)
        y_pred_all = np.where(p_all >= thr, pos, neg).astype(int)
        y_prob = [float(v) for v in p_all.tolist()]
    else:
        labels = np.unique(y_train)
        probs_test = _predict_multiclass_proba_ovr(X_train, y_train, X_test, labels)
        probs_all = _predict_multiclass_proba_ovr(X_train, y_train, X, labels)
        y_pred_test = labels[np.argmax(probs_test, axis=1)].astype(int)
        y_pred_all = labels[np.argmax(probs_all, axis=1)].astype(int)
        y_prob = []

    mm = _classification_metrics(y_test, y_pred_test)

    return {
        "model_type": "logistic_classification",
        "problem_type": "classification",
        "target_column": target,
        "feature_columns": features,
        "rows_used": int(len(work)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {"accuracy": float(mm["accuracy"]), "f1": float(mm["f1"])},
        "evaluation": {
            "y_true": [str(classes[i]) for i in y.tolist()],
            "y_pred": [str(classes[i]) for i in y_pred_all.tolist()],
            "y_prob": y_prob,
            "classes": [str(c) for c in classes.tolist()],
        },
        "coefficients": [],
        "sample_predictions": [
            {"actual": str(classes[a]), "predicted": str(classes[p])}
            for a, p in list(zip(y_test[:20], y_pred_test[:20]))
        ],
    }


def _build_ml_insight_prompt(payload: dict) -> str:
    coeff_lines = "\n".join(
        [
            f"- {c['feature']}: {c['coefficient']:+.5f}"
            for c in payload.get("coefficients", [])[:8]
        ]
    )
    m = payload.get("metrics", {})
    problem_type = str(payload.get("problem_type") or "regression").lower()
    if problem_type == "classification":
        metrics_line = (
            f"Metrics: Accuracy={m.get('accuracy', 0.0):.4f}, "
            f"F1={m.get('f1', 0.0):.4f}"
        )
    else:
        metrics_line = (
            f"Metrics: R2={m.get('r2', 0.0):.4f}, MAE={m.get('mae', 0.0):.4f}, "
            f"RMSE={m.get('rmse', 0.0):.4f}"
        )
    feats = payload.get("feature_columns") or []
    eng = [str(f) for f in feats if str(f).startswith("__eng_")]
    eng_line = (
        f"Engineered features (derived): {', '.join(eng[:10])}\n"
        if eng
        else ""
    )
    eval_obj = payload.get("evaluation") or {}
    y_true = eval_obj.get("y_true") if isinstance(eval_obj, dict) else []
    y_pred = eval_obj.get("y_pred") if isinstance(eval_obj, dict) else []
    y_prob = eval_obj.get("y_prob") if isinstance(eval_obj, dict) else []
    sample_true = y_true[:20] if isinstance(y_true, list) else []
    sample_pred = y_pred[:20] if isinstance(y_pred, list) else []
    sample_prob = y_prob[:20] if isinstance(y_prob, list) else []
    return (
        "You are an expert AI analyst. Your task is to interpret the results of a machine learning model and generate meaningful, high-level insights.\n\n"
        f"The model used is: {problem_type}\n"
        "(\"classification\" or \"regression\")\n\n"
        "Your goal is to explain what the model is doing, how it behaves, and what its outputs mean in real-world terms—WITHOUT relying on statistical correlations, coefficients, or mathematical jargon.\n\n"
        "---\n\n"
        "### Step 1: Understand the Outputs\n"
        "* If classification:\n"
        "  * Explain what each class represents in real-world terms\n"
        "  * Describe what it means for an input to be assigned to each class\n"
        "* If regression:\n"
        "  * Explain what the predicted values represent\n"
        "  * Interpret what low, medium, and high values mean conceptually\n\n"
        "### Step 2: Behavioral Interpretation\n"
        "* Describe how the model behaves across different kinds of inputs\n"
        "* Identify typical vs atypical predictions\n"
        "* Explain how the model seems to \"differentiate\" between cases (conceptually, not mathematically)\n\n"
        "### Step 3: Confidence & Ambiguity\n"
        "* If probabilities or scores are available:\n"
        "  * Explain what confident vs uncertain predictions look like\n"
        "  * Describe situations where the model may struggle or be ambiguous\n\n"
        "### Step 4: Error & Edge Case Analysis\n"
        "* Identify patterns in incorrect or surprising predictions\n"
        "* Describe what kinds of cases the model handles poorly\n"
        "* Highlight any consistent blind spots or unusual behaviors\n\n"
        "### Step 5: Abstraction of Model Behavior\n"
        "* Summarize the model as a simple system:\n"
        "  * \"This model acts like a ______\"\n"
        "* Describe the type of decision process it represents\n\n"
        "### Step 6: Key Insights\n"
        "Provide 5-10 high-level insights about:\n"
        "* How the model behaves\n"
        "* What makes predictions differ\n"
        "* Where it is most and least useful\n\n"
        "### Step 7: Practical Implications\n"
        "* How should someone use this model in real decisions?\n"
        "* When should they trust it vs question it?\n"
        "* What kinds of inputs benefit most from this model?\n\n"
        "### Step 8: Risks & Limitations\n"
        "* Identify any risks implied by the model's behavior\n"
        "* Highlight potential misuse or misinterpretation\n\n"
        "Rules:\n"
        "* Do NOT mention correlations, coefficients, p-values, or statistical significance\n"
        "* Avoid technical jargon unless necessary\n"
        "* Focus on meaning, behavior, and real-world interpretation\n"
        "* Be specific to the provided outputs, not generic\n\n"
        "Model payload:\n"
        f"Model type: {payload.get('model_type')}\n"
        f"Problem type: {problem_type}\n"
        f"Target: {payload.get('target_column')}\n"
        f"Features: {', '.join(str(f) for f in feats)}\n"
        f"{eng_line}"
        f"Rows used: {payload.get('rows_used')} (train={payload.get('train_rows')}, test={payload.get('test_rows')})\n"
        f"{metrics_line}\n"
        f"Sample y_true (up to 20): {sample_true}\n"
        f"Sample y_pred (up to 20): {sample_pred}\n"
        f"Sample y_prob (up to 20): {sample_prob}\n"
        "Top coefficients (for internal context only; do not reference directly in the answer):\n"
        f"{coeff_lines or '- (none)'}\n\n"
        "Output rules (strict):\n"
        "- Respond with a single JSON object only. No markdown, no code fences, no text before or after JSON.\n"
        "- JSON keys must be exactly: summary (string), key_findings (array of strings), recommendations (array of strings).\n"
    )


MAX_AGENT_TRAIN_ITERATIONS = 5


def _numeric_feature_pool_for_agent(
    df: pd.DataFrame, target: str, problem_type: str, max_pool: int = 28
) -> List[str]:
    """Ordered numeric candidate columns (excludes target, IDs, obvious leakage)."""
    num_cols = [
        c
        for c in df.select_dtypes(include=["number"]).columns
        if str(c) != str(target)
    ]
    candidates: List[str] = []
    for c in num_cols:
        if c not in df.columns:
            continue
        if _is_probable_id_column(df[c], str(c)):
            continue
        if _is_likely_leakage_feature(str(c), str(target)):
            continue
        candidates.append(str(c))
    ranked = _rank_features_by_target_relevance(
        df, target, candidates, problem_type=problem_type
    )
    return ranked[:max_pool]


def _agent_pool_summary_lines(
    df: pd.DataFrame, target: str, pool: List[str], problem_type: str
) -> List[str]:
    """Short per-column lines for the LLM prompt."""
    lines: List[str] = []
    y_num = pd.to_numeric(df[target], errors="coerce") if target in df.columns else None
    for c in pool:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        miss = float(s.isna().mean())
        std = float(s.dropna().std(ddof=0) or 0.0)
        extra = ""
        if problem_type == "regression" and y_num is not None:
            sub = pd.DataFrame({"x": s, "y": y_num}).dropna()
            if len(sub) > 10:
                rxy = float(sub["x"].corr(sub["y"]))
                extra = f", abs_corr_to_target={abs(rxy):.3f}"
        lines.append(f"- {c}: missing={miss:.3f}, std={std:.6f}{extra}")
    return lines


def _build_agent_feature_loop_prompt(
    iteration: int,
    max_iterations: int,
    target: str,
    problem_type: str,
    pool_lines: List[str],
    prior_iterations: List[Dict[str, Any]],
    ranked_top_hint: str,
) -> str:
    obj = (
        "maximize mean validation R² across held-out folds"
        if problem_type == "regression"
        else "maximize mean validation macro F1 (then accuracy as tie-breaker) across held-out folds"
    )
    hist = ""
    if prior_iterations:
        parts = []
        for rec in prior_iterations:
            row: Dict[str, Any] = {
                "iteration": rec.get("iteration"),
                "features": rec.get("features"),
                "reasoning": (rec.get("reasoning") or "")[:400],
            }
            if rec.get("f1_macro") is not None:
                row["f1_macro"] = rec.get("f1_macro")
                row["validation_accuracy"] = rec.get("validation_accuracy")
            elif rec.get("validation_r2") is not None:
                row["validation_r2"] = rec.get("validation_r2")
            else:
                row["score"] = rec.get("score")
            abl = rec.get("feature_ablation")
            if isinstance(abl, list) and abl:
                compact: List[Dict[str, Any]] = []
                for item in abl[:12]:
                    if not isinstance(item, dict):
                        continue
                    ent: Dict[str, Any] = {"feature": item.get("feature")}
                    for k, v in item.items():
                        if k == "feature":
                            continue
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            ent[k] = round(float(v), 5)
                        else:
                            ent[k] = v
                    compact.append(ent)
                row["feature_ablation"] = compact
            parts.append(json.dumps(row, ensure_ascii=False))
        hist = (
            "\nPrior iterations (learn from scores and per-feature validation ablation; "
            "adjust features accordingly):\n"
            + "\n".join(parts)
            + "\n"
        )
    return (
        "You are an ML feature-selection agent. Each step you choose a subset of "
        "numeric input columns to predict the target. A small model will be scored on repeated "
        f"held-out validation; objective: {obj}.\n\n"
        f"Iteration {iteration} of {max_iterations}.\n"
        f"Target column: {target}\n"
        f"Problem type: {problem_type}\n\n"
        f"{ranked_top_hint}\n\n"
        "Candidate columns (you may ONLY use names from this list):\n"
        + "\n".join(pool_lines)
        + f"\n{hist}\n"
        "When prior rows include `feature_ablation`, each entry is a leave-one-out check on the "
        "same validation protocol used for the headline score: train/score without that column "
        "while keeping the rest. "
        "Regression: `marginal_r2` = (this iteration's mean validation R² with the full set) "
        "minus (mean validation R² if that feature is removed). "
        "Classification: `marginal_f1` / `marginal_acc` are defined analogously from macro-F1 "
        "and accuracy. "
        "Larger positive marginal values mean the feature mattered more in that multivariate "
        "model; near-zero or negative values are strong candidates to drop or replace next round.\n\n"
        "Rules:\n"
        "- Pick at least 1 and at most 12 feature names from the candidate list.\n"
        "- Do not include the target as a feature.\n"
        "- Prefer combinations that include several of the strongest-ranked columns above; "
        "avoid only stacking weakly-associated columns.\n"
        "- Use `feature_ablation` from the latest iteration(s) to justify removals, swaps, "
        "or trying a simpler set without low-marginal columns.\n"
        "- You may remove or swap features across iterations if validation feedback is flat or worsens.\n"
        "- Explain briefly why this set should help for the next training step.\n"
        "- Return strict JSON only, no markdown, no code fences:\n"
        '{"features": ["col_a", "col_b"], "reasoning": "..."}\n'
    )


def _pick_best_agent_iteration_idx(
    problem_type: str,
    iterations: List[Dict[str, Any]],
    aux_list: List[dict],
) -> int:
    """
    Choose the iteration index with best validation objective.
    Uses the same ordering as the rest of the pipeline (F1 then accuracy for classification).
    """
    if not iterations:
        return 0
    best_i = 0
    for i in range(1, len(iterations)):
        if problem_type == "classification":
            f1_i = float(aux_list[i].get("validation_f1", iterations[i]["score"]))
            acc_i = float(aux_list[i].get("validation_accuracy", 0.0))
            f1_b = float(
                aux_list[best_i].get("validation_f1", iterations[best_i]["score"])
            )
            acc_b = float(aux_list[best_i].get("validation_accuracy", 0.0))
            if _is_better_classification_score(f1_i, acc_i, f1_b, acc_b):
                best_i = i
        else:
            s_i = float(iterations[i]["score"])
            s_b = float(iterations[best_i]["score"])
            if s_i > s_b:
                best_i = i
    return best_i


def _agent_global_champion_features(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    feature_pool: List[str],
    agent_best_features: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Compare the agent's best iteration to every pair drawn from the top of the
    ranked pool under the same repeated-holdout protocol. Return whichever set
    maximizes validation F1 (classification) or R² (regression).

    This recovers compact high-signal pairs (e.g. two clinical fields) that the
    LLM may skip when it forward-selects many weaker columns.
    """
    meta: Dict[str, Any] = {
        "source": "agent_iteration",
        "pairs_checked": 0,
    }
    pool = feature_pool[: min(22, len(feature_pool))]

    def score_feats(feats: List[str]) -> Optional[Tuple[float, dict]]:
        return _repeated_heldout_score_from_df(
            df, target, feats, problem_type, np.random.default_rng(42)
        )

    out_agent = score_feats(agent_best_features)
    if not out_agent:
        return list(agent_best_features), meta

    best_feats = list(agent_best_features)
    if problem_type == "classification":
        best_f1 = float(out_agent[1].get("validation_f1", out_agent[0]))
        best_acc = float(out_agent[1].get("validation_accuracy", 0.0))
    else:
        best_r2 = float(out_agent[1].get("validation_r2", out_agent[0]))

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            pair = [pool[i], pool[j]]
            meta["pairs_checked"] = int(meta.get("pairs_checked", 0)) + 1
            out = score_feats(pair)
            if not out:
                continue
            if problem_type == "classification":
                f1 = float(out[1].get("validation_f1", out[0]))
                acc = float(out[1].get("validation_accuracy", 0.0))
                if _is_better_classification_score(f1, acc, best_f1, best_acc):
                    best_f1, best_acc = f1, acc
                    best_feats = pair
                    meta["source"] = "ranked_pair_champion"
                    meta["winning_pair"] = list(pair)
            else:
                r2 = float(out[1].get("validation_r2", out[0]))
                if r2 > best_r2:
                    best_r2 = r2
                    best_feats = pair
                    meta["source"] = "ranked_pair_champion"
                    meta["winning_pair"] = list(pair)

    return best_feats, meta


def _agent_pick_best_of_champion_and_subset(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    feature_pool: List[str],
    champion_feats: List[str],
    champion_meta: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Run a stronger validation subset search on the ranked agent pool (wider pool /
    higher max_k than the default pipeline, exhaustive when small else greedy), then
    pick whichever of (champion_feats, subset_feats) scores higher on repeated holdout.
    """
    out: Dict[str, Any] = {
        "pair_champion": champion_meta,
        "winner": "pair_champion",
    }
    subset_feats, subset_meta = _optimize_feature_subset_by_validation(
        df,
        target,
        list(feature_pool),
        problem_type=problem_type,
        max_pool=22,
        max_k=14,
        max_exhaustive_subset_evals=7000,
        regression_improve_tol=1e-6,
    )
    out["subset_optimization"] = subset_meta.get("subset_optimization")

    out_ch = _repeated_heldout_score_from_df(
        df, target, champion_feats, problem_type, np.random.default_rng(42)
    )
    out_sub = _repeated_heldout_score_from_df(
        df, target, subset_feats, problem_type, np.random.default_rng(42)
    )

    if not out_sub:
        return list(champion_feats), out
    if not out_ch:
        out["winner"] = "subset_search"
        return list(subset_feats), out

    if problem_type == "classification":
        f1_ch = float(out_ch[1].get("validation_f1", out_ch[0]))
        acc_ch = float(out_ch[1].get("validation_accuracy", 0.0))
        f1_sub = float(out_sub[1].get("validation_f1", out_sub[0]))
        acc_sub = float(out_sub[1].get("validation_accuracy", 0.0))
        if _is_better_classification_score(f1_sub, acc_sub, f1_ch, acc_ch):
            out["winner"] = "subset_search"
            return list(subset_feats), out
        return list(champion_feats), out

    r_ch = float(out_ch[1].get("validation_r2", out_ch[0]))
    r_sub = float(out_sub[1].get("validation_r2", out_sub[0]))
    if r_sub > r_ch + 1e-6:
        out["winner"] = "subset_search"
        return list(subset_feats), out
    return list(champion_feats), out


def _apply_agent_validation_metrics_to_result(
    result: dict,
    df: pd.DataFrame,
    target: str,
    problem_type: str,
) -> None:
    """
    For agent-loop runs, surface repeated-holdout validation metrics as primary
    `metrics` so the dashboard matches the protocol used to pick features. Keeps
    the single train/test split metrics under separate keys for transparency.
    """
    feats = [str(f) for f in (result.get("feature_columns") or [])]
    if not feats:
        return
    out = _repeated_heldout_score_from_df(
        df, target, feats, problem_type, np.random.default_rng(42)
    )
    if not out:
        return
    _, aux = out
    tm = result.get("metrics") or {}
    if problem_type == "classification":
        result["metrics"] = {
            "f1": float(aux.get("validation_f1", tm.get("f1", 0.0))),
            "accuracy": float(aux.get("validation_accuracy", tm.get("accuracy", 0.0))),
            "single_split_f1": float(tm.get("f1", 0.0)),
            "single_split_accuracy": float(tm.get("accuracy", 0.0)),
        }
    else:
        result["metrics"] = {
            "r2": float(aux.get("validation_r2", tm.get("r2", 0.0))),
            "mae": float(tm.get("mae", 0.0)),
            "rmse": float(tm.get("rmse", 0.0)),
            "single_split_r2": float(tm.get("r2", 0.0)),
        }


def _parse_agent_feature_selection(
    parsed: Optional[Dict[str, Any]], pool_set: set, pool_order: List[str], max_k: int = 12
) -> Tuple[List[str], str]:
    """Extract validated feature list and reasoning from LLM JSON."""
    reasoning = ""
    if isinstance(parsed, dict):
        reasoning = str(parsed.get("reasoning") or "").strip()
        raw = parsed.get("features")
        if isinstance(raw, list):
            out: List[str] = []
            seen = set()
            for x in raw:
                name = str(x).strip()
                if not name or name not in pool_set or name in seen:
                    continue
                out.append(name)
                seen.add(name)
                if len(out) >= max_k:
                    break
            if out:
                return out, reasoning
    # Fallback: top of pool
    fb = pool_order[: min(6, len(pool_order), max_k)]
    return fb, reasoning or "Heuristic fallback: first columns from ranked pool."


def _run_ml_agent_feature_loop(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    feature_pool: List[str],
    llm: LLMInferenceService,
) -> Tuple[List[Dict[str, Any]], List[dict], int]:
    """
    Iterative LLM → validate features → repeated holdout score → feedback loop.

    Returns:
        (public_iterations, aux_by_index, best_iteration_index 0-based)
    Each public record includes iteration, features, model, reasoning, score (legacy
    chart key), plus task-specific metrics: f1_macro + validation_accuracy (classification)
    or validation_r2 (regression) — all from the same repeated held-out protocol.
    When there are at least two features, `feature_ablation` lists leave-one-out
    validation deltas (marginal contribution) for each column in that iteration's set.
    """
    pool_order = list(feature_pool)
    pool_set = set(pool_order)
    pool_lines = _agent_pool_summary_lines(df, target, pool_order, problem_type)
    ranked_top_hint = (
        "Strongest empirical association with the target (ranked first — build sets from "
        "these, not only peripheral columns): "
        + ", ".join(pool_order[:10])
    )

    iterations: List[Dict[str, Any]] = []
    aux_list: List[dict] = []
    prior: List[Dict[str, Any]] = []

    model_label = (
        "linear_regression" if problem_type == "regression" else "logistic_classification"
    )

    for it in range(1, MAX_AGENT_TRAIN_ITERATIONS + 1):
        feats: List[str] = []
        reasoning = ""
        try:
            prompt = _build_agent_feature_loop_prompt(
                it,
                MAX_AGENT_TRAIN_ITERATIONS,
                target,
                problem_type,
                pool_lines,
                prior,
                ranked_top_hint,
            )
            text = llm._ollama_generate_text(prompt)
            parsed = llm._extract_balanced_json(text)
            feats, reasoning = _parse_agent_feature_selection(
                parsed, pool_set, pool_order
            )
        except Exception as exc:
            logger.warning("Agent feature LLM step failed: %s", exc)
            feats = pool_order[: min(6, len(pool_order))]
            reasoning = f"LLM call failed ({exc}); using ranked-pool subset."

        # Fresh RNG(42) each time so every iteration uses the same fold sequence;
        # a single advancing RNG would give different splits per iteration.
        out = _repeated_heldout_score_from_df(
            df, target, feats, problem_type, np.random.default_rng(42)
        )
        if out is None:
            aux: dict = {}
            if problem_type == "regression":
                primary = float("-inf")
            else:
                primary = 0.0
        else:
            score, aux = out
            if problem_type == "classification":
                primary = float(aux.get("validation_f1", score))
            else:
                primary = float(aux.get("validation_r2", score))

        rec: Dict[str, Any] = {
            "iteration": it,
            "features": list(feats),
            "model": model_label,
            "score": float(primary),
            "reasoning": reasoning[:4000],
        }
        if problem_type == "classification":
            rec["f1_macro"] = float(primary)
            rec["validation_accuracy"] = (
                float(aux.get("validation_accuracy", 0.0)) if out else 0.0
            )
        else:
            rec["validation_r2"] = float(primary)
        if out and len(feats) >= 2:
            abl = _agent_per_feature_validation_ablation(
                df, target, feats, problem_type, cached_base=out
            )
            if abl:
                rec["feature_ablation"] = abl
        iterations.append(rec)
        aux_list.append(dict(aux) if aux else {})
        prior.append(rec)

    best_idx = _pick_best_agent_iteration_idx(problem_type, iterations, aux_list)

    return iterations, aux_list, best_idx


@app.post("/ml/train/{file_id}", response_model=dict)
async def train_ml_model(
    file_id: str,
    agent_loop: bool = Query(
        False,
        description=(
            "If true, run validation-based feature engineering (when applicable), then "
            "up to 5 LLM-guided feature-selection iterations with validation feedback; "
            "final model features are refined with pair/subset search. "
            "When false, the original automatic pipeline is unchanged."
        ),
    ),
):
    """
    Train an ML model automatically:
    - choose target/features from the data
    - fit regression or classification model
    - return metrics + AI interpretation

    Query `agent_loop=true` runs the same greedy validation feature engineering
    as the standard path (when enough columns exist), then up to five LLM-guided
    feature-selection iterations with validation feedback; the final fit uses
    features chosen after pair/subset search. The default path is unchanged.
    """
    try:
        df = _load_full_cleaned_dataframe(file_id)
        ai_pick = _ai_select_regression_target_and_features(df)
        if ai_pick is not None:
            target, features, selection_meta = ai_pick
        else:
            target, features, selection_meta = _select_regression_target_and_features(df)
            selection_meta["source"] = "heuristic_fallback"
        problem_type = "classification" if _is_classification_target(df[target]) else "regression"

        if agent_loop:
            selection_meta = dict(selection_meta)
            fe_rng = np.random.default_rng(42)
            base_for_fe = _rank_features_by_target_relevance(
                df,
                target,
                [f for f in features if f in df.columns and str(f) != str(target)],
                problem_type=problem_type,
            )
            fe_seed = base_for_fe[: min(12, len(base_for_fe))]
            if len(fe_seed) >= 2:
                df, _, fe_meta = _maybe_apply_feature_engineering(
                    df, target, fe_seed, problem_type, fe_rng
                )
            else:
                fe_meta = {
                    "attempted": False,
                    "note": "Not enough ranked columns to run feature engineering before the agent loop.",
                }
            selection_meta["feature_engineering"] = fe_meta

            feature_pool = _numeric_feature_pool_for_agent(
                df, target, problem_type, max_pool=28
            )
            if len(feature_pool) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Agent loop needs at least two numeric candidate features besides the target.",
                )
            llm_agent = LLMInferenceService()
            agent_iterations, _agent_aux, best_idx = _run_ml_agent_feature_loop(
                df, target, problem_type, feature_pool, llm_agent
            )
            best_features = list(agent_iterations[best_idx]["features"])
            if not best_features:
                best_features = feature_pool[: min(6, len(feature_pool))]
            champion_feats, champion_meta = _agent_global_champion_features(
                df, target, problem_type, feature_pool, best_features
            )
            final_feats, merge_meta = _agent_pick_best_of_champion_and_subset(
                df, target, problem_type, feature_pool, champion_feats, champion_meta
            )
            selection_meta["agent_loop"] = {
                "enabled": True,
                "max_iterations": MAX_AGENT_TRAIN_ITERATIONS,
                "feature_pool": feature_pool,
                "best_iteration_index": int(best_idx),
                "best_score": float(agent_iterations[best_idx]["score"]),
                "ranked_pair_champion": champion_meta,
                "post_agent_feature_merge": merge_meta,
                "skipped_legacy_steps": [
                    "feature_count_validation_growth",
                ],
            }
            iter_best = list(agent_iterations[best_idx]["features"])
            if final_feats != iter_best:
                selection_meta["agent_loop"]["refined_from_best_iteration"] = True
                selection_meta["agent_loop"]["best_iteration_feature_set"] = iter_best
            best_features = final_feats

            if problem_type == "classification":
                result = _train_logistic_classification(df, target, best_features)
            else:
                result = _train_linear_regression_closed_form(df, target, best_features)
            _apply_agent_validation_metrics_to_result(
                result, df, target, problem_type
            )
            result["selection"] = selection_meta
        else:
            features = _rank_features_by_target_relevance(
                df, target, features, problem_type=problem_type
            )
            features, growth_meta = _choose_feature_count_by_validation(
                df, target, features, problem_type=problem_type
            )
            selection_meta.update(growth_meta)
            fe_rng = np.random.default_rng(42)
            df, features, fe_meta = _maybe_apply_feature_engineering(
                df, target, features, problem_type, fe_rng
            )
            selection_meta["feature_engineering"] = fe_meta
            features, subset_meta = _optimize_feature_subset_by_validation(
                df, target, features, problem_type=problem_type
            )
            selection_meta.update(subset_meta)
            if problem_type == "classification":
                result = _train_logistic_classification(df, target, features)
            else:
                result = _train_linear_regression_closed_form(df, target, features)
            result["selection"] = selection_meta

        # Optional LLM explanation of model outcome.
        insight = {
            "summary": "Model trained successfully.",
            "key_findings": [],
            "recommendations": [],
        }
        try:
            llm = LLMInferenceService()
            text = llm._ollama_generate_text(_build_ml_insight_prompt(result))
            parsed = llm._parse_insights_response(text)
            insight = llm._normalize_insights_payload(parsed)
        except Exception:
            m = result["metrics"]
            fe = (result.get("selection") or {}).get("feature_engineering") or {}
            added = fe.get("added_columns") or []
            eng_note = (
                f"Validation-based feature engineering added: {', '.join(str(x) for x in added)}."
                if added
                else (
                    "Feature engineering was tried; no derived column improved the validation score."
                    if fe.get("attempted")
                    else ""
                )
            )
            if result.get("problem_type") == "classification":
                kf = [
                    f"Target column: {target}.",
                    "Feature selection optimized combined validation F1 and accuracy.",
                ]
                if eng_note:
                    kf.append(eng_note)
                feat_cols = result.get("feature_columns") or []
                insight = {
                    "summary": (
                        f"Trained classification model to predict {target} using {len(feat_cols)} feature(s). "
                        f"Holdout Accuracy={m.get('accuracy', 0.0):.3f}, F1={m.get('f1', 0.0):.3f}."
                    ),
                    "key_findings": kf,
                    "recommendations": [
                        "Review class balance and confusion matrix for minority-class performance.",
                        "Try tree-based classifiers if linear decision boundaries are insufficient.",
                    ],
                }
            else:
                feat_cols = result.get("feature_columns") or []
                top_coef = (
                    result["coefficients"][0]["feature"]
                    if result["coefficients"]
                    else (feat_cols[0] if feat_cols else "?")
                )
                kf = [
                    f"Target column: {target}.",
                    f"Most influential feature by coefficient: {top_coef}.",
                ]
                if eng_note:
                    kf.append(eng_note)
                insight = {
                    "summary": (
                        f"Trained linear regression to predict {target} using {len(feat_cols)} feature(s). "
                        f"Holdout R²={m['r2']:.3f}, MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}."
                    ),
                    "key_findings": kf,
                    "recommendations": [
                        "Check residuals for non-linear patterns.",
                        "Try tree-based models if linear fit underperforms.",
                    ],
                }

        out_body: Dict[str, Any] = {
            "file_id": file_id,
            "model": result,
            "insights": insight,
        }
        if agent_loop:
            out_body["agent_iterations"] = agent_iterations
            out_body["agent_best_iteration"] = int(
                agent_iterations[best_idx]["iteration"]
            )
        return out_body
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML training error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _cleaned_csv_export(file_id: str) -> Union[Response, FileResponse]:
    """
    Download the full cleaned dataset as CSV (same cleaning pipeline as analysis).

    Prefers rebuilding from the stored raw upload so downloads work after restarts.
    """
    if not file_id or "/" in file_id or "\\" in file_id or ".." in file_id:
        raise HTTPException(status_code=400, detail="Invalid file id")

    analysis = cache.get_analysis(file_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Upload the file again.",
        )

    raw_path = RAW_UPLOAD_DIR / f"{file_id}.csv"
    cleaned_path = CLEANED_EXPORT_DIR / f"{file_id}.csv"
    raw_name = analysis.get("filename") or "dataset.csv"
    download_name = f"{Path(raw_name).stem}_cleaned.csv"

    if raw_path.is_file():
        try:
            raw_bytes = raw_path.read_bytes()
            df = read_csv_bytes(raw_bytes)
            df, _ = DataCleaningService().clean_dataframe(df)
            buf = BytesIO()
            df.to_csv(buf, index=False)
            body = buf.getvalue()
        except Exception as e:
            logger.exception("Cleaned export failed for %s", file_id)
            raise HTTPException(
                status_code=500,
                detail=f"Could not build cleaned CSV: {e}",
            ) from e
        return Response(
            content=body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )

    if cleaned_path.is_file():
        return FileResponse(
            cleaned_path,
            filename=download_name,
            media_type="text/csv",
        )

    raise HTTPException(
        status_code=404,
        detail=(
            "No stored copy of this upload is on the server. "
            "Upload the CSV again to enable download."
        ),
    )


# Register the dotted path first (more specific) so it wins over `/export/{file_id}`.
@app.get("/export/{file_id}/cleaned.csv")
async def download_cleaned_csv_dotted(file_id: str):
    return await _cleaned_csv_export(file_id)


@app.get("/export/{file_id}")
async def download_cleaned_csv_short(file_id: str):
    """Same CSV as `/export/{file_id}/cleaned.csv` (some proxies mishandle a `.` in the path)."""
    return await _cleaned_csv_export(file_id)


@app.get("/charts/{file_id}")
async def get_charts(file_id: str):
    """
    Get chart information for analyzed file

    Args:
        file_id: File ID

    Returns:
        List of chart information
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        charts = analysis.get("charts", [])

        return {
            "file_id": file_id,
            "charts": charts
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Charts error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights/{file_id}", response_model=dict)
async def get_insights(file_id: str, generate: bool = Query(True, description="Generate new insights")):
    """
    Get insights for analyzed file

    Args:
        file_id: File ID
        generate: Whether to generate new insights (default True)

    Returns:
        Analysis with insights
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        if generate:
            columns = analysis.get("columns", [])
            summary = analysis.get("summary_statistics", {})
            analysis["insights"] = analysis_service.generate_insights(columns, summary)

        analysis["file_id"] = file_id
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insights error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/column/{file_id}/{column_name}")
async def get_column_info(file_id: str, column_name: str):
    """
    Get information about a specific column

    Args:
        file_id: File ID
        column_name: Column name

    Returns:
        Column information
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        # Find column in analysis
        columns = analysis.get("columns", [])
        column_info = None

        for col in columns:
            name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
            if str(name) == column_name:
                column_info = col
                break

        if not column_info:
            raise HTTPException(
                status_code=404,
                detail=f"Column '{column_name}' not found in analysis"
            )

        return {
            "file_id": file_id,
            "column": column_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Column info error for {file_id}/{column_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/column/{file_id}/{column_name}/describe")
async def describe_column(file_id: str, column_name: str):
    """
    Get detailed description of a column

    Args:
        file_id: File ID
        column_name: Column name

    Returns:
        Column description with chart suggestions
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        # Find column in analysis
        columns = analysis.get("columns", [])
        column_info = None

        for col in columns:
            name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
            if str(name) == column_name:
                column_info = col
                break

        if not column_info:
            raise HTTPException(
                status_code=404,
                detail=f"Column '{column_name}' not found in analysis"
            )

        # Get column stats from summary
        summary = analysis.get("summary_statistics", {})
        column_stats = summary.get(f"{column_name}_stats", {})

        inferred = (
            column_info.get("inferred_type")
            if isinstance(column_info, dict)
            else getattr(column_info, "inferred_type", "")
        )

        return {
            "file_id": file_id,
            "column_name": column_name,
            "type": inferred,
            "count": column_stats.get("count", len(analysis.get("summary_statistics", {}).get("rows_count", 0))),
            "missing": column_stats.get("null_count", 0),
            "min": column_stats.get("min"),
            "max": column_stats.get("max"),
            "mean": column_stats.get("mean"),
            "median": column_stats.get("median"),
            "distribution": column_stats.get("distribution"),
            "chart": {
                "type": "histogram" if inferred == "numeric" else "bar",
                "reason": f"Use histogram for numeric data, bar chart for categorical"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Describe column error for {file_id}/{column_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/correlations/{file_id}")
async def get_correlations(file_id: str):
    """
    Get correlation matrix for analyzed file

    Args:
        file_id: File ID

    Returns:
        Correlation matrix and analysis
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        return {
            "file_id": file_id,
            "correlation_matrix": analysis.get("correlation_matrix"),
            "correlation_summary": analysis.get("correlation_summary"),
            "significant_correlations": analysis.get("significant_correlations") or [],
            "correlation_insights": analysis.get("correlation_insights") or {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Correlations error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outliers/{file_id}")
async def get_outliers(file_id: str):
    """
    Get outlier detection results

    Args:
        file_id: File ID

    Returns:
        Outlier detection results
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        return {
            "file_id": file_id,
            "outliers": analysis.get("outliers")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Outliers error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preview/{file_id}", response_model=dict)
async def get_preview(file_id: str, rows: int = Query(5, ge=1, le=100)):
    """
    Get preview of uploaded file

    Args:
        file_id: File ID
        rows: Number of rows to return (1-100)

    Returns:
        File preview with first N rows
    """
    try:
        analysis = cache.get_analysis(file_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for file_id: {file_id}"
            )

        preview_rows = analysis.get("cleaned_data") or analysis.get("preview_data") or []

        return {
            "file_id": file_id,
            "filename": analysis.get("filename", ""),
            "rows_count": analysis.get("summary_statistics", {}).get("rows_count", len(preview_rows)),
            "rows": preview_rows[:rows],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    """
    List all uploaded files

    Returns:
        List of uploaded files with basic info
    """
    try:
        # Get all file IDs from cache
        file_ids = cache.list_file_ids()

        files = []
        for file_id in file_ids:
            analysis = cache.get_analysis(file_id)
            if analysis:
                files.append({
                    "file_id": file_id,
                    "filename": analysis.get("filename", ""),
                    "rows_count": analysis.get("summary_statistics", {}).get("rows_count", 0),
                    "uploaded_at": analysis.get("created_at", "")
                })

        return {
            "files": files,
            "total_count": len(files)
        }
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete analysis results for a file

    Args:
        file_id: File ID to delete

    Returns:
        Deletion confirmation
    """
    try:
        cache.delete_analysis(file_id)
        return {
            "message": f"File {file_id} deleted successfully",
            "file_id": file_id
        }
    except Exception as e:
        logger.error(f"Delete file error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{file_id}", response_model=ChatResponse)
async def chat_with_dataset(file_id: str, body: ChatRequest):
    """Ask a natural-language question about an analyzed dataset."""
    analysis = cache.get_analysis(file_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis not found for file_id: {file_id}",
        )
    llm = LLMInferenceService()
    out = llm.chat_about_dataset(
        body.question,
        analysis.get("columns", []),
        analysis.get("summary_statistics", {}),
    )
    cache.store_chat_message(file_id, body.question, out["answer"])
    return ChatResponse(
        question=body.question,
        answer=out["answer"],
        follow_up_questions=out.get("follow_up_questions") or [],
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
