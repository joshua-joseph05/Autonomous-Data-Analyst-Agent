"""
LLM inference service for DataInsight AI
"""
import ast
import json
import os
import requests
from typing import Dict, Any, Optional, List
from app.utils.config import OLLAMA_HOST, OLLAMA_MODEL
from app.utils.prompts import (
    DATASET_INSIGHT_PROMPT,
    CHART_DESCRIPTION_PROMPT,
    CORRELATION_INSIGHT_PROMPT,
)
from app.utils.correlation_insight_rules import (
    filter_correlation_insight_lines,
    numeric_column_names,
    summarize_correlation_network_for_prompt,
)


class LLMInferenceService:
    """Service for calling LLM via Ollama"""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self._requested_model = model
        self.model = model  # may be rewritten to a concrete tag from /api/tags

    @staticmethod
    def _resolve_model_tag(requested: str, installed: List[str]) -> Optional[str]:
        """
        Map a short name (e.g. llama3) to an installed Ollama tag (e.g. llama3:latest).
        """
        if not requested or not installed:
            return None
        if requested in installed:
            return requested
        prefix = requested + ":"
        for name in installed:
            if name.startswith(prefix):
                return name
        req_base = requested.split(":")[0]
        for name in installed:
            base = name.split(":")[0]
            if base == req_base:
                return name
        for name in installed:
            base = name.split(":")[0]
            if req_base in base or base in req_base:
                return name
        return None

    def _insights_setup_help(self, detail: str = "") -> Dict[str, Any]:
        msg = (
            "AI insights use Ollama on this machine. "
            f"1) Start Ollama so {self.host} responds. "
            f"2) Install a model, e.g. `ollama pull {self._requested_model}` (or run `ollama list` and set OLLAMA_MODEL to an exact name). "
            "3) Restart the API after changing env vars. "
            "Optional: OLLAMA_AUTO_PULL=1 lets the server trigger `ollama pull` when the model is missing."
        )
        if detail:
            msg = f"{msg} Details: {detail}"
        return {
            "summary": msg,
            "key_findings": [],
            "recommendations": [],
        }

    def _ensure_model_loaded(self) -> bool:
        """Resolve OLLAMA_MODEL to an installed tag; optional pull when OLLAMA_AUTO_PULL=1."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models if m.get("name")]

            resolved = self._resolve_model_tag(self._requested_model, model_names)
            if resolved:
                self.model = resolved
                return True

            if os.getenv("OLLAMA_AUTO_PULL", "").lower() in ("1", "true", "yes"):
                load_resp = requests.post(
                    f"{self.host}/api/pull",
                    json={"name": self._requested_model},
                    timeout=120,
                )
                if load_resp.status_code != 200:
                    return False
                # Re-fetch tags after pull
                response = requests.get(f"{self.host}/api/tags", timeout=10)
                if response.status_code != 200:
                    return False
                model_names = [
                    m.get("name", "")
                    for m in response.json().get("models", [])
                    if m.get("name")
                ]
                resolved = self._resolve_model_tag(self._requested_model, model_names)
                if resolved:
                    self.model = resolved
                    return True
                return False

            print(
                f"Model '{self._requested_model}' not found in Ollama. "
                f"Installed: {model_names[:5]}{'...' if len(model_names) > 5 else ''}. "
                "Set OLLAMA_MODEL to one of these names, or OLLAMA_AUTO_PULL=1 to pull."
            )
            return False
        except Exception as e:
            print(f"Error checking model: {e}")
            return False

    def generate_insights(self, columns: list, summary: dict, samples: list = None) -> Dict[str, Any]:
        """
        Generate insights from dataset using LLM

        Args:
            columns: List of column metadata
            summary: Summary statistics
            samples: Optional sample data

        Returns:
            Insights dictionary
        """
        # Prepare prompt
        prompt = self._build_insight_prompt(columns, summary, samples)

        try:
            text = self._ollama_generate_text(prompt)
            parsed = self._parse_insights_response(text)
            return self._normalize_insights_payload(parsed)
        except Exception as e:
            print(f"LLM error: {e}")
            return self._insights_setup_help(str(e))

    def _build_correlation_insight_prompt(self, columns: list, pairs: list) -> str:
        lines: List[str] = []
        for c in columns:
            if isinstance(c, dict):
                name = str(c.get("name", "")).strip()
                if not name:
                    continue
                lines.append(f"- {name}: {c.get('inferred_type', '')}")
        pair_lines: List[str] = []
        for p in pairs[:18]:
            if not isinstance(p, dict):
                continue
            cols = p.get("columns") or []
            if not isinstance(cols, (list, tuple)) or len(cols) < 2:
                continue
            pair_lines.append(
                f"- {cols[0]} vs {cols[1]}: r={p.get('correlation')} ({p.get('strength', '')})"
            )
        num_names = numeric_column_names(
            [c for c in columns if isinstance(c, dict)]
        )
        main_net, narrow_net, net_note = summarize_correlation_network_for_prompt(
            num_names, pairs
        )
        return CORRELATION_INSIGHT_PROMPT.format(
            columns="\n".join(lines[:40]) or "(none)",
            main_network_columns=main_net,
            narrow_network_columns=narrow_net,
            network_note=net_note,
            pairs="\n".join(pair_lines) or "(no pairs)",
        )

    def _normalize_correlation_insights_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce LLM JSON into two string lists for the correlation panel."""

        def collect(*keys: str) -> List[str]:
            for key in keys:
                val = self._maybe_parse_json_list(raw.get(key))
                if not isinstance(val, list):
                    continue
                out: List[str] = []
                for item in val:
                    s = self._stringify_insight_item(item)
                    if s:
                        out.append(s)
                if out:
                    return out[:25]
            return []

        return {
            "correlated_pairs": collect(
                "correlated_pairs",
                "pair_insights",
                "correlated_columns",
                "key_findings",
            ),
            "prediction_ideas": collect(
                "prediction_ideas",
                "prediction_suggestions",
                "modeling_ideas",
                "predictors",
            ),
        }

    def generate_correlation_insights(self, columns: list, pairs: list) -> Dict[str, Any]:
        """LLM commentary on correlation pairs and modeling directions."""
        if not pairs:
            return {"correlated_pairs": [], "prediction_ideas": []}
        prompt = self._build_correlation_insight_prompt(columns, pairs)
        try:
            text = self._ollama_generate_text(prompt)
            parsed: Dict[str, Any] = {}
            bal = self._extract_balanced_json(text)
            if isinstance(bal, dict):
                parsed = bal
            elif text.strip().startswith("{"):
                try:
                    obj = json.loads(text.strip())
                    if isinstance(obj, dict):
                        parsed = obj
                except json.JSONDecodeError:
                    pass
            normalized = self._normalize_correlation_insights_payload(parsed)
            names = numeric_column_names(
                [c for c in columns if isinstance(c, dict)]
            )
            c_lines, p_lines = filter_correlation_insight_lines(
                normalized.get("correlated_pairs") or [],
                normalized.get("prediction_ideas") or [],
                names,
                pairs,
            )
            return {"correlated_pairs": c_lines, "prediction_ideas": p_lines}
        except Exception as e:
            print(f"LLM correlation insight error: {e}")
            return {"correlated_pairs": [], "prediction_ideas": []}

    def generate_insight(self, column_name: str, column_type: str, column_range: tuple) -> Dict[str, Any]:
        """
        Generate description for a single column

        Args:
            column_name: Column name
            column_type: Column type
            column_range: (min, max) for numeric columns

        Returns:
            Column description
        """
        prompt = CHART_DESCRIPTION_PROMPT.format(
            column_name=column_name,
            column_type=column_type,
            column_range=column_range,
            categories=""  # Simplified for now
        )

        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            print(f"LLM error for {column_name}: {e}")
            return {
                "column_description": f"Column '{column_name}' is a {column_type} column.",
                "suggested_chart": "bar" if column_type == "categorical" else "line",
                "reason": "Standard chart for this column type."
            }

    def _build_insight_prompt(self, columns: list, summary: dict, samples: list = None) -> str:
        """Build prompt for insight generation"""
        # Format columns
        col_strs = []
        for col in columns:
            col_strs.append(
                f"{col['name']}: {col['inferred_type']}"
            )
        columns_str = ", ".join(col_strs)

        # Format summary stats (truncated)
        summary_str = f"\nSummary:\n- Rows: {summary.get('rows_count', 0)}\n- Missing cells: {summary.get('missing_cells', 0)}"
        if 'numeric_columns' in summary:
            summary_str += f"\n- Numeric columns: {len(summary['numeric_columns'])}"
        if 'categorical_columns' in summary:
            summary_str += f"\n- Categorical columns: {len(summary['categorical_columns'])}"

        # Format sample data
        samples_str = f"\nSample:\n{samples[:3] if samples else 'No sample data provided'}" if samples else ""

        return DATASET_INSIGHT_PROMPT.format(
            columns=columns_str,
            types=", ".join(set(c['inferred_type'] for c in columns)),
            summary=summary_str,
            samples=samples_str
        )

    def _ollama_generate_text(self, prompt: str) -> str:
        """Call Ollama /api/generate and return raw model text."""
        if not self._ensure_model_loaded():
            raise RuntimeError(
                f"No usable model for OLLAMA_MODEL={self._requested_model!r} at {self.host}"
            )

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.text}")

        result = response.json()
        return (result.get("response") or "").strip()

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API and parse JSON (chart / generic prompts)."""
        text = self._ollama_generate_text(prompt)
        try:
            return self._parse_response(text)
        except Exception as e:
            print(f"Could not parse LLM response as JSON: {e}")
            return {
                "summary": text[:500] if text else "No response received.",
                "key_findings": [],
                "recommendations": [],
            }

    @staticmethod
    def _extract_balanced_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse first balanced {...} JSON object from text."""
        start = text.find("{")
        if start < 0:
            return None
        snippet = text[start:]
        depth = 0
        in_str = False
        esc = False
        quote: Optional[str] = None
        for j, ch in enumerate(snippet):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
                    quote = None
            else:
                if ch in ('"', "'"):
                    in_str = True
                    quote = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = snippet[: j + 1]
                        try:
                            obj = json.loads(chunk)
                            return obj if isinstance(obj, dict) else None
                        except json.JSONDecodeError:
                            # Some models emit Python-style dicts with single quotes.
                            # Accept them safely via literal_eval as a fallback.
                            try:
                                obj2 = ast.literal_eval(chunk)
                                return obj2 if isinstance(obj2, dict) else None
                            except Exception:
                                return None
        return None

    def _parse_insights_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON insight object from model output (markdown fences, preamble, etc.)."""
        import re

        if not text:
            return {}

        # ```json ... ```
        m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                bal = self._extract_balanced_json(inner)
                if bal:
                    return bal

        # ``` ... ``` (no language tag)
        m = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if m:
            inner = m.group(1).strip()
            if inner.startswith("{") or "{" in inner:
                try:
                    obj = json.loads(inner)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    bal = self._extract_balanced_json(inner)
                    if bal:
                        return bal

        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        bal = self._extract_balanced_json(text)
        if bal:
            return bal

        return {"summary": text.strip(), "key_findings": [], "recommendations": []}

    @staticmethod
    def _stringify_insight_item(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            title = item.get("title") or item.get("heading")
            body = item.get("text") or item.get("description") or item.get("detail")
            if title and body:
                return f"{title}: {body}"
            if title:
                return str(title)
            parts = [f"{k}: {v}" for k, v in item.items() if v is not None and k not in ("title", "text")]
            return "; ".join(parts) if parts else json.dumps(item, ensure_ascii=False)
        return str(item).strip()

    def _unwrap_embedded_insights_json(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Some models return the full insight object as a JSON string in `summary`
        while leaving top-level lists empty. Merge parsed inner fields so the UI
        receives a proper shape.
        """
        out: Dict[str, Any] = dict(raw)
        for _ in range(3):
            s = out.get("summary")
            if not isinstance(s, str):
                break
            t = s.strip()
            if not t.startswith("{"):
                break
            if "key_findings" not in t and "recommendations" not in t:
                break
            inner: Optional[Dict[str, Any]] = None
            try:
                cand = json.loads(t)
                if isinstance(cand, dict):
                    inner = cand
            except json.JSONDecodeError:
                inner = self._extract_balanced_json(t)
            if not isinstance(inner, dict):
                break
            # Merge even when lists are stringified; downstream normalization
            # will parse JSON-array strings where possible.
            out = {**out, **inner}
        return out

    @staticmethod
    def _maybe_parse_json_list(val: Any) -> Any:
        """If the model stringified an array, parse it back to a list."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            t = val.strip()
            if t.startswith("["):
                try:
                    parsed = json.loads(t)
                    return parsed if isinstance(parsed, list) else val
                except json.JSONDecodeError:
                    pass
        return val

    def _normalize_insights_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce LLM JSON into summary (str) + string lists for the UI."""
        if not isinstance(raw, dict):
            return {
                "summary": str(raw).strip()[:8000],
                "key_findings": [],
                "recommendations": [],
            }

        raw = self._unwrap_embedded_insights_json(raw)

        def flatten_summary(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val.strip()
            if isinstance(val, dict):
                lines = []
                for k, v in val.items():
                    if isinstance(v, (dict, list)):
                        try:
                            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:500]}")
                        except TypeError:
                            lines.append(f"{k}: {str(v)[:500]}")
                    else:
                        lines.append(f"{k}: {v}")
                return " ".join(lines)[:8000]
            return str(val).strip()[:8000]

        def collect_strings(*keys: str) -> List[str]:
            out: List[str] = []
            for key in keys:
                val = self._maybe_parse_json_list(raw.get(key))
                if not isinstance(val, list):
                    continue
                for item in val:
                    s = self._stringify_insight_item(item)
                    if s:
                        out.append(s)
            return out[:40]

        summary = flatten_summary(raw.get("summary"))
        findings = collect_strings(
            "key_findings",
            "findings",
            "key_findings_list",
            "notable_relationships",
            "notable_relationship",
            "anomalies",
            "anomaly",
            "patterns",
        )
        recs = collect_strings(
            "recommendations",
            "recommendation",
            "next_steps",
            "actions",
        )

        if not findings:
            for key in ("notable_relationships", "anomalies", "patterns"):
                block = raw.get(key)
                if isinstance(block, dict):
                    findings.append(self._stringify_insight_item(block))
                elif isinstance(block, str) and block.strip():
                    findings.append(block.strip())

        if not summary and findings:
            summary = findings[0][:2000]
            findings = findings[1:]

        if not summary:
            summary = "No summary returned; see key findings below."

        return {
            "summary": summary[:12000],
            "key_findings": findings[:30],
            "recommendations": recs[:30],
        }

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON"""
        import re

        # Try to extract JSON from response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        bal = LLMInferenceService._extract_balanced_json(text)
        if bal:
            return bal

        # If all else fails, return text as summary
        return {
            "summary": text.strip()[:1000] if text else "No response received.",
            "key_findings": [],
            "recommendations": []
        }

    def chat_about_dataset(self, question: str, columns: list, summary: dict) -> Dict[str, Any]:
        """Answer a natural-language question about an analyzed dataset (plain text via Ollama)."""
        if not self._ensure_model_loaded():
            help_text = self._insights_setup_help()["summary"]
            return {"answer": help_text, "follow_up_questions": []}

        col_lines = []
        for c in columns[:50]:
            if isinstance(c, dict):
                col_lines.append(f"- {c.get('name')}: {c.get('inferred_type')}")
            else:
                col_lines.append(f"- {getattr(c, 'name', '?')}: {getattr(c, 'inferred_type', '?')}")
        prompt = (
            "You are a data analyst. Answer the user's question using the dataset summary below.\n\n"
            f"Rows: {summary.get('rows_count', 'unknown')}\n"
            f"Missing cells: {summary.get('missing_cells', 'unknown')}\n"
            f"Columns:\n{chr(10).join(col_lines)}\n\n"
            f"Question: {question}\n\n"
            "Respond in clear prose. If the summary is insufficient, say what is missing."
        )
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=90,
            )
            if response.status_code != 200:
                return {
                    "answer": f"LLM error ({response.status_code}): {response.text[:300]}",
                    "follow_up_questions": [],
                }
            text = (response.json().get("response") or "").strip()
            return {"answer": text or "No response.", "follow_up_questions": []}
        except Exception as e:
            return {"answer": f"Could not reach LLM: {e}", "follow_up_questions": []}
