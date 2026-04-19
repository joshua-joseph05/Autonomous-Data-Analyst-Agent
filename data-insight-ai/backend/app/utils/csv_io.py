"""
Robust CSV loading: delimiter sniffing and UTF-8 BOM handling.

Many user files use tabs, semicolons, or pipes while keeping a .csv extension.
Default comma parsing then mis-reads the table (wrong headers, empty rows after cleaning).

Excel / some ML exports prepend a dimension row (e.g. 768, 8) before real column names;
we detect and skip that so Pregnancies, Glucose, ... become headers.
"""
from __future__ import annotations

import csv
from io import StringIO
from typing import List, Tuple

import numpy as np
import pandas as pd


def _decode_text(content: bytes) -> str:
    return content.decode("utf-8-sig", errors="replace")


def _sniff_delimiter(sample: str) -> str | None:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        d = dialect.delimiter
        if isinstance(d, str) and len(d) == 1:
            return d
    except Exception:
        pass
    return None


def _split_line(line: str, sep: str) -> List[str]:
    if sep == r"\s+":
        return line.split()
    return [c.strip().strip('"') for c in line.split(sep)]


def _score_parse(df: pd.DataFrame) -> Tuple[float, int]:
    """Higher is better: many columns + high numeric-like content."""
    if df is None or df.empty:
        return -1.0, 0
    ncols = len(df.columns)
    if ncols < 2:
        return float(ncols), ncols
    unnamed = sum(1 for c in df.columns if str(c).startswith("Unnamed"))
    unnamed_pen = 0.25 * (unnamed / max(ncols, 1))
    fracs: List[float] = []
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        fracs.append(float(num.notna().mean()))
    avg_num = float(np.mean(fracs)) if fracs else 0.0
    score = float(ncols) * (1.0 - unnamed_pen) + 2.5 * avg_num
    return score, ncols


def _maybe_skip_excel_dimension_row(text: str, sep: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Skip a leading row like Excel exports: first row is `768,8,,,,` (rows, cols),
    second row is real headers (Pregnancies, Glucose, ...).
    """
    if sep == r"\s+":
        return df
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return df

    first_cells = _split_line(lines[0], sep)
    non_empty = [c for c in first_cells if c != ""]
    # Real data rows usually fill many columns; dimension rows are 2–4 numbers then blanks.
    if len(non_empty) > 6:
        return df
    if len(non_empty) < 2:
        return df
    if not (non_empty[0].isdigit() and non_empty[1].isdigit()):
        return df
    try:
        dim_a = int(non_empty[0])
        dim_b = int(non_empty[1])
    except ValueError:
        return df
    # Typical "N rows, M columns" metadata (not a normal patient row like 6,148,...).
    if dim_a < 10 or dim_b < 2:
        return df
    if dim_a > 100_000_000 or dim_b > 1_000_000:
        return df

    second_cells = _split_line(lines[1], sep)
    non_empty_second = [c for c in second_cells if c != ""]
    if len(non_empty_second) < 4:
        return df
    header_text = " ".join(non_empty_second)
    letter_count = sum(1 for c in header_text if c.isalpha())
    if letter_count < 12:
        return df

    c0, c1 = str(df.columns[0]), str(df.columns[1])
    looks_like_wrong_header = (c0.isdigit() and c1.isdigit()) or (
        str(c0).replace(".", "").isdigit() and str(c1).replace(".", "").isdigit()
    )
    if not looks_like_wrong_header:
        return df

    try:
        fixed = pd.read_csv(StringIO(text), sep=sep, skiprows=1, header=0, engine="python")
    except Exception:
        return df
    if fixed.empty or len(fixed.columns) < 2:
        return df
    # Prefer the fixed parse if it has fewer "Unnamed" junk headers.
    unnamed_before = sum(1 for c in df.columns if str(c).startswith("Unnamed"))
    unnamed_after = sum(1 for c in fixed.columns if str(c).startswith("Unnamed"))
    if unnamed_after > unnamed_before:
        return df
    score_before, _ = _score_parse(df)
    score_after, _ = _score_parse(fixed)
    if score_after + 0.01 < score_before:
        return df
    return fixed


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """
    Load CSV/TSV-like bytes into a DataFrame.

    Tries UTF-8 with BOM strip, sniffs delimiter when ambiguous, and scores parses
    so tab/semicolon files still named .csv are read correctly.
    """
    if not content or not content.strip():
        raise ValueError("File is empty.")

    text = _decode_text(content).strip()
    if not text:
        raise ValueError("File is empty.")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    sample = "\n".join(lines[:25]) if lines else text[:2000]
    sniffed = _sniff_delimiter(sample)

    candidates: List[str] = []
    if sniffed:
        candidates.append(sniffed)
    for sep in [",", "\t", ";", "|"]:
        if sep not in candidates:
            candidates.append(sep)

    best_df: pd.DataFrame | None = None
    best_sep: str = ","
    best_score = -1e18

    for sep in candidates:
        try:
            df = pd.read_csv(StringIO(text), sep=sep, engine="python")
        except Exception:
            continue
        score, ncols = _score_parse(df)
        if ncols >= 2 and score > best_score:
            best_score = score
            best_df = df
            best_sep = sep

    if best_df is None or len(best_df.columns) < 2:
        try:
            df_ws = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")
            score, ncols = _score_parse(df_ws)
            if ncols >= 2 and score > best_score:
                best_score = score
                best_df = df_ws
                best_sep = r"\s+"
        except Exception:
            pass

    if best_df is None:
        try:
            best_df = pd.read_csv(StringIO(text))
            best_sep = ","
        except Exception as e:
            raise ValueError(f"Could not parse CSV: {e}") from e

    if best_df is None or len(best_df.columns) < 1:
        raise ValueError("Could not parse CSV: no columns detected.")

    if best_sep != r"\s+":
        best_df = _maybe_skip_excel_dimension_row(text, best_sep, best_df)

    return best_df
