import os, json, math, time, re
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq

ChartSuggestion = Dict[str, Any]

ALLOWED_TYPES = {"BarChart", "Bar", "XAxis", "YAxis", "Tooltip", "ResponsiveContainer", "PieChart", "Pie", "Cell", "LineChart", "Line", "Legend", "CartesianGrid", "ScatterChart", "Scatter",}

def _is_number(x: Any) -> bool:
    if x is None or x == "":
        return False
    try:
        return math.isfinite(float(str(x).replace(",", "")))
    except Exception:
        return False

def _profile_columns(columns: List[str], rows: List[List[Any]], max_profile_rows: int = 500) -> Dict[str, Any]:
    sample = rows[:max_profile_rows] if rows else []
    stats: Dict[str, Any] = {}
    for i, col in enumerate(columns):
        values = [r[i] if i < len(r) else None for r in sample]
        non_null = [v for v in values if v not in (None, "")]
        unique_count = len(set(non_null))
        numeric_ratio = (sum(1 for v in non_null if _is_number(v)) / len(non_null)) if non_null else 0.0
        # keep only lightweight profile info; the LLM decides visualization
        stats[col] = {
            "nonNullCount": len(non_null),
            "uniqueCount": unique_count,
            "numericRatio": round(numeric_ratio, 3),
            "sampleValues": [str(v) if v is not None else None for v in values[:10]],
        }
    return {
        "rowCount": len(rows),
        "columnCount": len(columns),
        "columns": columns,
        "stats": stats,
    }

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # ```json\n{...}\n``` or ```\n{...}\n```
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()

def _validate_suggestion(obj: Dict[str, Any]) -> Optional[ChartSuggestion]:
    if not isinstance(obj, dict):
        return None
    t = str(obj.get("type", "")).lower()
    if t not in ALLOWED_TYPES:
        return None
    # Normalize fields
    out: ChartSuggestion = {
        "type": t,
        "x": obj.get("x"),
        "y": obj.get("y"),
        "series": obj.get("series"),
        "reason": obj.get("reason"),
    }
    # Optional fields sanity
    if "y" in out and isinstance(out["y"], list):
        out["y"] = [str(v) for v in out["y"]]
    if "x" in out and out["x"] is not None:
        out["x"] = str(out["x"])
    if "series" in out and out["series"] is not None:
        out["series"] = str(out["series"])
    if "reason" in out and out["reason"] is not None:
        out["reason"] = str(out["reason"])
    return out

class ChartSuggester:
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        timeout_s: int = 12,
        max_rows_for_llm: int = 200,
        retries: int = 2,
        retry_backoff_s: float = 0.8,
    ):
        self.model_name = model_name or os.getenv("CHART_MODEL_NAME", os.getenv("MODEL_NAME", "llama-3.1-8b-instant"))
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_rows_for_llm = max_rows_for_llm
        self.retries = max(0, retries)
        self.retry_backoff_s = retry_backoff_s

    def _build_prompt(self, question: Optional[str], columns: List[str], rows: List[List[Any]]) -> str:
        sample_rows = rows[: self.max_rows_for_llm]
        profile = _profile_columns(columns, rows)

        instructions = (
            "You are a senior data visualization expert. Decide the single best chart to visualize the provided tabular data.\n"
            "Return EXACTLY ONE compact JSON object with keys: type, x, y, series, reason. No markdown, no extra text.\n"
            "Constraints:\n"
            f"- Allowed type values: {sorted(ALLOWED_TYPES)}\n"
            "- x is the name of the x-axis column (or null if not applicable)\n"
            "- y is the name of the y-axis column OR an array of metric columns (or null if not applicable)\n"
            "- series is an optional grouping/series column name (or null)\n"
            "- reason is a short one-line rationale for the choice\n"
            "Do not include any commentary outside JSON. Do not output code fences.\n"
        )

        context = {
            "question": question or "",
            "columns": columns,
            "sampleRows": sample_rows,
            "profile": profile,
        }

        return (
            instructions
            + "\nContext:\n"
            + json.dumps(context, ensure_ascii=False)
            + "\nJSON:"
        )

    def _invoke_llm_once(self, prompt: str) -> Optional[ChartSuggestion]:
        llm = ChatGroq(model=self.model_name, temperature=self.temperature, timeout=self.timeout_s)
        out = llm.invoke(prompt)
        if not out:
            return None
        text = out if isinstance(out, str) else str(out)
        text = _strip_code_fences(text)
        try:
            data = json.loads(text)
        except Exception:
            # Attempt to extract trailing JSON object
            m = re.search(r"\{[\s\S]*\}\s*$", text)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except Exception:
                return None
        return _validate_suggestion(data)

    def suggest_chart(self, question: Optional[str], columns: List[str], rows: List[List[Any]]) -> ChartSuggestion:
        prompt = self._build_prompt(question, columns, rows)

        # Retry with simple backoff to handle transient failures or minor formatting issues
        attempt = 0
        last_err: Optional[str] = None
        while attempt <= self.retries:
            try:
                res = self._invoke_llm_once(prompt)
                if res:
                    return res
                last_err = "invalid_or_empty_llm_response"
            except Exception as e:
                last_err = str(e) or "unknown_error"
            if attempt < self.retries:
                time.sleep(self.retry_backoff_s * (attempt + 1))
            attempt += 1

        # As a last resort in production, default to table to avoid breaking UX
        return {"type": "table", "x": None, "y": None, "series": None, "reason": f"fallback_due_to_{last_err or 'llm_failure'}"}