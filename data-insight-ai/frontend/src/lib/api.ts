import axios from 'axios';

/** Trailing slashes removed; used by axios and CSV export fetch. */
export function getApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (raw) return raw.replace(/\/+$/, '');
  return 'http://localhost:8000';
}

const API_BASE_URL = getApiBaseUrl();

const api = axios.create({
  baseURL: API_BASE_URL,
});

export interface ColumnMetadata {
  name: string;
  inferred_type: string;
  null_count: number;
  unique_count: number;
  sample_values: string[];
  min_max?: { min: number; max: number };
}

/** Response from POST /upload */
export interface FilePreview {
  file_id: string;
  filename: string;
  file_size: number;
  rows_count: number;
  columns: ColumnMetadata[];
  preview_data: string[][];
}

/** Per-upload stats from the backend cleaning pipeline (JSON-safe). */
export interface CleaningReport {
  rows_start: number;
  rows_end: number;
  columns_start: number;
  columns_end: number;
  null_threshold_percent: number;
  columns_dropped_high_null: string[];
  iqr_outlier_rows_removed: number;
  sign_anomaly_rows_removed: number;
  duplicate_rows_removed: number;
  low_cardinality_columns_as_category: string[];
  numeric_columns_filled: string[];
  categorical_columns_filled: string[];
  binary_categorical_columns_encoded: string[];
  single_value_categorical_columns_dropped: string[];
  categorical_columns_dropped_low_association: string[];
  rows_removed_unresolved_nulls: number;
  outlier_row_removal_enabled: boolean;
  drop_remaining_null_rows_enabled: boolean;
  iqr_multiplier: number;
  sign_anomaly_threshold: number;
}

function asNumber(v: unknown, fallback = 0): number {
  return typeof v === 'number' && !Number.isNaN(v) ? v : fallback;
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.map((x) => String(x)) : [];
}

function asBool(v: unknown, fallback = false): boolean {
  return typeof v === 'boolean' ? v : fallback;
}

export function parseCleaningReport(raw: unknown): CleaningReport | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.rows_start !== 'number') return null;
  return {
    rows_start: asNumber(o.rows_start),
    rows_end: asNumber(o.rows_end),
    columns_start: asNumber(o.columns_start),
    columns_end: asNumber(o.columns_end),
    null_threshold_percent: asNumber(o.null_threshold_percent, 50),
    columns_dropped_high_null: asStringArray(o.columns_dropped_high_null),
    iqr_outlier_rows_removed: asNumber(o.iqr_outlier_rows_removed),
    sign_anomaly_rows_removed: asNumber(o.sign_anomaly_rows_removed),
    duplicate_rows_removed: asNumber(o.duplicate_rows_removed),
    low_cardinality_columns_as_category: asStringArray(
      o.low_cardinality_columns_as_category
    ),
    numeric_columns_filled: asStringArray(o.numeric_columns_filled),
    categorical_columns_filled: asStringArray(o.categorical_columns_filled),
    binary_categorical_columns_encoded: asStringArray(
      o.binary_categorical_columns_encoded
    ),
    single_value_categorical_columns_dropped: asStringArray(
      o.single_value_categorical_columns_dropped
    ),
    categorical_columns_dropped_low_association: asStringArray(
      o.categorical_columns_dropped_low_association
    ),
    rows_removed_unresolved_nulls: asNumber(o.rows_removed_unresolved_nulls),
    outlier_row_removal_enabled: asBool(o.outlier_row_removal_enabled, true),
    drop_remaining_null_rows_enabled: asBool(
      o.drop_remaining_null_rows_enabled,
      true
    ),
    iqr_multiplier: asNumber(o.iqr_multiplier, 1.5),
    sign_anomaly_threshold: asNumber(o.sign_anomaly_threshold, 0.85),
  };
}

export interface AnalysisResult {
  file_id: string;
  filename: string;
  rows_count: number;
  cleaned_data: string[][];
  cleaning_report: CleaningReport | null;
  summary_stats: {
    rows: number;
    missing_cells: number;
    numeric_columns: number;
    categorical_columns: number;
    columns: ColumnMetadata[];
  };
  correlations: {
    matrix: Record<string, unknown>;
    significant: Array<{
      column1: string;
      column2: string;
      correlation: number;
    }>;
    outliers: Record<string, unknown>;
    /** LLM (or heuristic fallback) commentary on pairs and modeling ideas */
    insights: {
      correlated_pairs: string[];
      prediction_ideas: string[];
    };
  };
  insights: {
    summary: string;
    key_findings: string[];
    recommendations: string[];
  };
  charts: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  answer: string;
  follow_up_questions: string[];
}

export interface ExploratorySeriesResponse {
  file_id: string;
  row_count: number;
  series: Record<string, Array<number | null>>;
}

/** Leave-one-feature-out validation deltas for one agent iteration (same protocol as headline score). */
export interface MLAgentFeatureAblation {
  feature: string;
  /** Regression: mean validation R² when this feature is removed from the set. */
  mean_val_r2_if_removed?: number;
  /** Regression: full-set mean val R² minus mean val R² without this feature. */
  marginal_r2?: number;
  mean_val_f1_if_removed?: number;
  marginal_f1?: number;
  mean_val_acc_if_removed?: number;
  marginal_acc?: number;
}

/** One step of the optional LLM agent loop for feature selection (backend). */
export interface MLAgentIteration {
  iteration: number;
  features: string[];
  model: string;
  /** Legacy chart series; equals f1_macro (classification) or validation_r2 (regression). */
  score: number;
  reasoning: string;
  /** Mean macro F1 across repeated validation folds (classification). */
  f1_macro?: number;
  /** Mean accuracy across those same folds (classification). */
  validation_accuracy?: number;
  /** Mean R² across repeated validation folds (regression). */
  validation_r2?: number;
  /** Present when |features| >= 2: marginal validation impact per feature. */
  feature_ablation?: MLAgentFeatureAblation[];
}

export interface MLTrainResponse {
  file_id: string;
  agent_iterations?: MLAgentIteration[];
  /** 1-based iteration number that produced the best validation score */
  agent_best_iteration?: number;
  model: {
    model_type: string;
    problem_type?: 'regression' | 'classification';
    target_column: string;
    feature_columns: string[];
    rows_used: number;
    train_rows: number;
    test_rows: number;
    metrics:
      | { r2: number; mae: number; rmse: number }
      | {
          accuracy: number;
          f1: number;
          /** Present for agent-loop: one random train/test split */
          single_split_f1?: number;
          single_split_accuracy?: number;
        }
      | {
          r2: number;
          mae: number;
          rmse: number;
          single_split_r2?: number;
        };
    evaluation?: {
      y_true: Array<number | string>;
      y_pred: Array<number | string>;
      y_prob?: number[];
    };
    coefficients: Array<{ feature: string; coefficient: number }>;
    sample_predictions: Array<{ actual: number | string; predicted: number | string }>;
    /** Training metadata (e.g. agent_loop, feature_engineering) from the API */
    selection?: Record<string, unknown>;
  };
  insights: {
    summary: string;
    key_findings: string[];
    recommendations: string[];
  };
}

function asInsightSummary(v: unknown): string {
  const raw =
    typeof v === 'string'
      ? v
      : v != null && typeof v === 'object'
      ? JSON.stringify(v)
      : '';
  return cleanInsightText(raw);
}

function cleanInsightText(text: string): string {
  return text
    .replace(/\*\*/g, '')
    .replace(/^\s*[\*\-•]+\s*/, '')
    .replace(/^\s*\d+[\.\)]\s*/, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function isUsefulInsightLine(text: string): boolean {
  if (!text) return false;
  if (!/[a-z0-9]/i.test(text)) return false;
  if (/^[,.;:!?'"`~\-\s]+$/.test(text)) return false;
  const tokens = text.split(/\s+/).filter(Boolean);
  if (tokens.length === 1) {
    const t = tokens[0].toLowerCase();
    if (t.length < 4) return false;
    if (['there', 'here', 'also', 'thus', 'therefore'].includes(t)) return false;
  }
  return true;
}

function asInsightStringList(v: unknown): string[] {
  const normalizeList = (items: unknown[]): string[] =>
    items
      .map((item) =>
        cleanInsightText(typeof item === 'string' ? item : JSON.stringify(item))
      )
      .filter(isUsefulInsightLine);

  if (typeof v === 'string') {
    const t = v.trim();
    if (t.startsWith('[')) {
      try {
        const parsed = JSON.parse(t) as unknown;
        if (Array.isArray(parsed)) {
          return normalizeList(parsed);
        }
      } catch {
        return [];
      }
    }
    return [];
  }
  if (!Array.isArray(v)) return [];
  return normalizeList(v);
}

function extractFirstJsonObject(text: string): Record<string, unknown> | null {
  const start = text.indexOf('{');
  if (start < 0) return null;
  const s = text.slice(start);
  let depth = 0;
  let inStr = false;
  let esc = false;
  let quote = '';
  for (let i = 0; i < s.length; i += 1) {
    const ch = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === quote) {
        inStr = false;
        quote = '';
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      inStr = true;
      quote = ch;
      continue;
    }
    if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        const chunk = s.slice(0, i + 1);
        try {
          const parsed = JSON.parse(chunk) as unknown;
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
          }
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function extractPseudoInsightsObject(text: string): Record<string, unknown> | null {
  const t = text.trim();
  if (!t.includes('summary') || !t.includes('key_findings')) return null;

  const summaryMatch = t.match(
    /['"]summary['"]\s*:\s*(['"])([\s\S]*?)\1\s*,\s*['"](key_findings|findings)['"]/i
  );
  const summary = summaryMatch?.[2]?.trim() ?? '';

  const findingsBlock = t.match(/['"]key_findings['"]\s*:\s*\[([\s\S]*?)\]/i)?.[1] ?? '';
  const recsBlock = t.match(/['"]recommendations['"]\s*:\s*\[([\s\S]*?)\]/i)?.[1] ?? '';

  const extractList = (block: string): string[] => {
    const out: string[] = [];
    const re = /['"]([^'"]+?)['"]/g;
    let m: RegExpExecArray | null = re.exec(block);
    while (m) {
      const v = m[1].trim();
      if (v) out.push(v);
      m = re.exec(block);
    }
    return out;
  };

  const key_findings = extractList(findingsBlock);
  const recommendations = extractList(recsBlock);
  if (!summary && key_findings.length === 0 && recommendations.length === 0) {
    return null;
  }
  return { summary, key_findings, recommendations };
}

/** Accept insights as object or a single JSON string (bad cache / older API). */
function coerceInsightsRecord(raw: unknown): Record<string, unknown> {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    return raw as Record<string, unknown>;
  }
  if (typeof raw === 'string') {
    const t = raw.trim();
    const parsed = t.startsWith('{')
      ? (() => {
          try {
            return JSON.parse(t) as unknown;
          } catch {
            return null;
          }
        })()
      : extractFirstJsonObject(t);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    const pseudo = extractPseudoInsightsObject(t);
    if (pseudo) return pseudo;
  }
  return {};
}

/**
 * Models sometimes put the full `{ summary, key_findings, recommendations }`
 * JSON as a string in `summary`. Merge inner fields so the dashboard layout
 * stays correct (also fixes already-stored bad payloads).
 */
function unwrapEmbeddedInsightsJson(
  raw: Record<string, unknown>
): Record<string, unknown> {
  let out: Record<string, unknown> = { ...raw };
  for (let i = 0; i < 3; i += 1) {
    const s = out.summary;
    if (typeof s !== 'string') break;
    const t = s.trim();
    if (!t.includes('{')) break;
    if (!t.includes('key_findings') && !t.includes('recommendations')) break;
    let inner: Record<string, unknown> | null = null;
    if (t.startsWith('{')) {
      try {
        const parsed = JSON.parse(t) as unknown;
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          inner = parsed as Record<string, unknown>;
        }
      } catch {
        inner = extractFirstJsonObject(t) ?? extractPseudoInsightsObject(t);
      }
    } else {
      inner = extractFirstJsonObject(t) ?? extractPseudoInsightsObject(t);
    }
    if (!inner) break;
    // Merge even when arrays were stringified; `asInsightStringList` handles
    // JSON-array strings later.
    out = { ...out, ...inner };
  }
  return out;
}

function normalizeAnalysis(raw: Record<string, unknown>, fileId: string): AnalysisResult {
  const summary = (raw.summary_statistics as Record<string, unknown>) || {};
  const numericList = Array.isArray(summary.numeric_columns)
    ? (summary.numeric_columns as unknown[])
    : [];
  const categoricalList = Array.isArray(summary.categorical_columns)
    ? (summary.categorical_columns as unknown[])
    : [];

  const significantRaw = Array.isArray(raw.significant_correlations)
    ? (raw.significant_correlations as Record<string, unknown>[])
    : [];
  const significant = significantRaw.map((s) => {
    const cols = Array.isArray(s.columns) ? (s.columns as string[]) : [];
    return {
      column1: cols[0] ?? '',
      column2: cols[1] ?? '',
      correlation: typeof s.correlation === 'number' ? s.correlation : 0,
    };
  });

  const corrInsightsObj =
    raw.correlation_insights &&
    typeof raw.correlation_insights === 'object' &&
    !Array.isArray(raw.correlation_insights)
      ? (raw.correlation_insights as Record<string, unknown>)
      : {};
  const correlationAi = {
    correlated_pairs: asInsightStringList(corrInsightsObj.correlated_pairs),
    prediction_ideas: asInsightStringList(corrInsightsObj.prediction_ideas),
  };

  const insightsRaw = unwrapEmbeddedInsightsJson(
    coerceInsightsRecord(raw.insights)
  );
  const columnsRaw = Array.isArray(raw.columns) ? raw.columns : [];

  const previewObj = (raw.preview as Record<string, unknown>) || {};
  const cleaningReport = parseCleaningReport(
    raw.cleaning_report ?? previewObj.cleaning_report
  );
  const fromPreview = Array.isArray(previewObj.cleaned_data)
    ? (previewObj.cleaned_data as unknown[][])
    : [];
  const rawCleaned = Array.isArray(raw.cleaned_data)
    ? (raw.cleaned_data as unknown[][])
    : [];
  const sourceRows = rawCleaned.length ? rawCleaned : fromPreview;
  const cleaned = sourceRows.map((row) =>
    (Array.isArray(row) ? row : []).map((c) => (c == null ? '' : String(c)))
  );

  return {
    file_id: (raw.file_id as string) || fileId,
    filename: (raw.filename as string) || 'dataset',
    rows_count: (summary.rows_count as number) ?? 0,
    cleaned_data: cleaned,
    cleaning_report: cleaningReport,
    summary_stats: {
      rows: (summary.rows_count as number) ?? 0,
      missing_cells: (summary.missing_cells as number) ?? 0,
      numeric_columns: numericList.length,
      categorical_columns: categoricalList.length,
      columns: columnsRaw.map((c: Record<string, unknown>) => ({
        name: String(c.name ?? ''),
        inferred_type: String(c.inferred_type ?? ''),
        null_count: Number(c.null_count ?? 0),
        unique_count: Number(c.unique_count ?? 0),
        sample_values: Array.isArray(c.sample_values)
          ? (c.sample_values as string[]).map(String)
          : [],
        min_max: c.min_max as ColumnMetadata['min_max'],
      })),
    },
    correlations: {
      matrix: (raw.correlation_matrix as Record<string, unknown>) || {},
      significant,
      outliers: (raw.outliers as Record<string, unknown>) || {},
      insights: correlationAi,
    },
    insights: {
      summary: asInsightSummary(insightsRaw.summary),
      key_findings: asInsightStringList(insightsRaw.key_findings),
      recommendations: asInsightStringList(insightsRaw.recommendations),
    },
    charts: (raw.charts as Record<string, unknown>) || {},
  };
}

export const uploadFile = async (file: File): Promise<FilePreview> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<FilePreview>('/upload', formData);

  return response.data;
};

export const getAnalysis = async (fileId: string): Promise<AnalysisResult> => {
  const response = await api.get<Record<string, unknown>>(
    `/analysis/${encodeURIComponent(fileId)}`
  );
  return normalizeAnalysis(response.data, fileId);
};

export const getExploratorySeries = async (
  fileId: string,
  columns: string[]
): Promise<ExploratorySeriesResponse> => {
  const cleanedCols = columns.map((c) => c.trim()).filter(Boolean);
  const response = await api.get<ExploratorySeriesResponse>(
    `/exploratory/${encodeURIComponent(fileId)}/series`,
    {
      params: cleanedCols.length
        ? { columns: cleanedCols.join(',') }
        : undefined,
    }
  );
  return response.data;
};

export const trainMlModel = async (
  fileId: string,
  options?: { agentLoop?: boolean }
): Promise<MLTrainResponse> => {
  const params = options?.agentLoop === true ? { agent_loop: 'true' } : undefined;
  const response = await api.post<MLTrainResponse>(
    `/ml/train/${encodeURIComponent(fileId)}`,
    undefined,
    { params }
  );
  return response.data;
};

/** Full cleaned table (all rows) as produced by the backend cleaning pipeline. */
export const downloadCleanedCsv = async (
  fileId: string,
  originalFilename: string
): Promise<void> => {
  // Same-origin Next route (must match `src/app/api/export/[fileId]/route.ts` — no extra path segments).
  const url = `/api/export/${encodeURIComponent(fileId)}`;
  const res = await fetch(url, { method: 'GET', credentials: 'same-origin' });
  const bodyText = await res.text();

  if (!res.ok) {
    const trimmed = bodyText.trim();
    let msg = bodyText.slice(0, 400) || `Download failed (HTTP ${res.status})`;
    if (trimmed.startsWith('<!DOCTYPE') || trimmed.startsWith('<html')) {
      msg = `Download route returned a web page (HTTP ${res.status}), not the CSV API. Restart Next after adding /api/export, or check the dev server log.`;
    } else if (trimmed.startsWith('{')) {
      try {
        const j = JSON.parse(trimmed) as { detail?: unknown };
        if (typeof j.detail === 'string') msg = j.detail;
        else if (Array.isArray(j.detail)) {
          msg = j.detail
            .map((x: { msg?: string }) => x?.msg || JSON.stringify(x))
            .join('; ');
        }
      } catch {
        /* keep msg */
      }
    }
    throw new Error(msg);
  }

  const ct = (res.headers.get('content-type') || '').toLowerCase();
  if (ct.includes('application/json') || ct.includes('text/html')) {
    throw new Error(
      'Unexpected response (not CSV). Check NEXT_PUBLIC_API_URL points at the FastAPI server.'
    );
  }

  const blob = new Blob([bodyText], { type: 'text/csv;charset=utf-8' });
  const stem = originalFilename.replace(/\.[^/.]+$/, '') || 'dataset';
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objectUrl;
  a.download = `${stem}_cleaned.csv`;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  window.setTimeout(() => {
    a.remove();
    URL.revokeObjectURL(objectUrl);
  }, 2500);
};

export const chatWithDataset = async (
  fileId: string,
  question: string
): Promise<ChatResponse> => {
  const response = await api.post<{
    answer: string;
    follow_up_questions?: string[];
  }>(`/chat/${encodeURIComponent(fileId)}`, { question });
  return {
    answer: response.data.answer,
    follow_up_questions: response.data.follow_up_questions ?? [],
  };
};
