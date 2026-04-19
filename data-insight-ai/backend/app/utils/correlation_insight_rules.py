"""
Correlation-panel text rules: avoid mixing columns that live in different
correlation "worlds" — e.g. a pair that only strongly ties to each other vs
columns that tie into the broader numeric network.

We approximate that by building an undirected graph on numeric columns:
edges for |Pearson r| >= CORRELATION_CLUSTER_EDGE_THRESHOLD. Connected
components partition columns; the largest component(s) are the "main"
network. Bullets that bridge distinct components, or modeling lines that pull
in off-network columns, are dropped.
"""
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

# Moderate |r|; only used to decide who clusters with whom, not significance UI.
CORRELATION_CLUSTER_EDGE_THRESHOLD = 0.35


def numeric_column_names(columns: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for c in columns:
        if not isinstance(c, dict):
            continue
        if str(c.get("inferred_type", "")).lower() != "numeric":
            continue
        name = str(c.get("name", "")).strip()
        if name:
            out.append(name)
    return out


def mentioned_column_names(line: str, names: Sequence[str]) -> List[str]:
    """Which dataset column names appear in this line (underscore-safe boundaries)."""
    if not line or not names:
        return []
    low = line.lower()
    found: List[str] = []
    for name in sorted(names, key=len, reverse=True):
        if len(name) < 2:
            continue
        pat = r"(?<![a-z0-9_])" + re.escape(name.lower()) + r"(?![a-z0-9_])"
        if re.search(pat, low):
            found.append(name)
    return found


def build_column_correlation_components(
    column_names: Sequence[str],
    pairs: Sequence[Dict[str, Any]],
    threshold: float = CORRELATION_CLUSTER_EDGE_THRESHOLD,
) -> Tuple[Dict[str, int], Dict[int, int]]:
    """
    Map each column name -> component id, and component id -> size.
    Isolated columns (no edge at threshold) are singleton components.
    """
    nodes = sorted({n for n in column_names if n})
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for p in pairs:
        if not isinstance(p, dict):
            continue
        cols = p.get("columns") or []
        if not isinstance(cols, (list, tuple)) or len(cols) < 2:
            continue
        a, b = str(cols[0]), str(cols[1])
        if a not in adj or b not in adj:
            continue
        try:
            r = abs(float(p.get("correlation", 0.0)))
        except (TypeError, ValueError):
            continue
        if r >= threshold:
            adj[a].add(b)
            adj[b].add(a)

    col_to_comp: Dict[str, int] = {}
    comp_sizes: Dict[int, int] = {}
    next_id = 0
    visited: Set[str] = set()
    for start in nodes:
        if start in visited:
            continue
        stack = [start]
        comp_members: List[str] = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_members.append(u)
            for v in adj[u]:
                if v not in visited:
                    stack.append(v)
        for u in comp_members:
            col_to_comp[u] = next_id
        comp_sizes[next_id] = len(comp_members)
        next_id += 1
    return col_to_comp, comp_sizes


def largest_component_ids(comp_sizes: Dict[int, int]) -> Set[int]:
    if not comp_sizes:
        return set()
    mx = max(comp_sizes.values())
    return {cid for cid, sz in comp_sizes.items() if sz == mx}


def line_spans_multiple_correlation_components(
    line: str,
    column_names: Sequence[str],
    col_to_comp: Dict[str, int],
) -> bool:
    """True if the line references columns from two+ different correlation components."""
    names = [n for n in column_names if n]
    mentioned = mentioned_column_names(line, names)
    if len(mentioned) < 2:
        return False
    comps: Set[int] = set()
    for m in mentioned:
        cid = col_to_comp.get(m)
        if cid is not None:
            comps.add(cid)
    return len(comps) > 1


def line_mentions_outside_main_correlation_network(
    line: str,
    column_names: Sequence[str],
    col_to_comp: Dict[str, int],
    main_component_ids: Set[int],
) -> bool:
    """True if any mentioned column is not in one of the largest component(s)."""
    names = [n for n in column_names if n]
    mentioned = mentioned_column_names(line, names)
    for m in mentioned:
        cid = col_to_comp.get(m)
        if cid is None:
            continue
        if cid not in main_component_ids:
            return True
    return False


def filter_correlation_insight_lines(
    correlated_lines: Sequence[str],
    prediction_lines: Sequence[str],
    column_names: Sequence[str],
    pairs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Apply component-based filters to both bullet lists."""
    names = [n for n in column_names if n]
    col_to_comp, comp_sizes = build_column_correlation_components(names, pairs)
    main_ids = largest_component_ids(comp_sizes)
    if len(comp_sizes) <= 1:
        return list(correlated_lines), list(prediction_lines)

    c_out = [
        ln
        for ln in correlated_lines
        if ln and not line_spans_multiple_correlation_components(ln, names, col_to_comp)
    ]
    p_out = [
        ln
        for ln in prediction_lines
        if ln
        and not line_mentions_outside_main_correlation_network(
            ln, names, col_to_comp, main_ids
        )
        and not line_spans_multiple_correlation_components(ln, names, col_to_comp)
    ]
    return c_out, p_out


def format_column_group(names: Sequence[str]) -> str:
    return ", ".join(names) if names else "(none)"


def summarize_correlation_network_for_prompt(
    column_names: Sequence[str],
    pairs: Sequence[Dict[str, Any]],
) -> Tuple[str, str, str]:
    """
    Human-readable lists for the LLM: main bulk vs off-network columns,
    plus a one-line note about the graph rule.
    """
    names = [n for n in column_names if n]
    col_to_comp, comp_sizes = build_column_correlation_components(names, pairs)
    main_ids = largest_component_ids(comp_sizes)
    mx = max(comp_sizes.values()) if comp_sizes else 0
    main_cols = sorted(n for n, cid in col_to_comp.items() if cid in main_ids)
    narrow_cols = sorted(n for n, cid in col_to_comp.items() if cid not in main_ids)
    note = (
        f"Link threshold |r|≥{CORRELATION_CLUSTER_EDGE_THRESHOLD} "
        f"(Pearson). Largest mutually linked group has {mx} column(s)."
    )
    return format_column_group(main_cols), format_column_group(narrow_cols), note
