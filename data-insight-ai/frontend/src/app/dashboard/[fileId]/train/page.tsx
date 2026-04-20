'use client';

import Link from 'next/link';
import { Suspense, useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import {
  trainMlModel,
  type MLAgentIteration,
  type MLTrainResponse,
} from '@/lib/api';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

/** Primary metric for charts/cards (explicit fields preferred, else legacy `score`). */
function agentIterationPrimaryMetric(
  step: MLAgentIteration,
  kind: 'classification' | 'regression'
): number {
  if (kind === 'classification') {
    return typeof step.f1_macro === 'number' ? step.f1_macro : step.score;
  }
  return typeof step.validation_r2 === 'number' ? step.validation_r2 : step.score;
}

type InsightsView = {
  summary: string;
  keyFindings: string[];
  recommendations: string[];
};

function cleanInsightLine(text: string): string {
  return text
    .replace(/^\s*[\*\-•]+\s*/, '')
    .replace(/^\s*\d+[\.\)]\s*/, '')
    .replace(/^\s*:+\s*/, '')
    .replace(/\*\*/g, '')
    .trim();
}

function isUsefulInsightLine(text: string): boolean {
  if (!text) return false;
  if (!/[a-z0-9]/i.test(text)) return false;
  if (/^[,.;:!?'"`~\-\s]+$/.test(text)) return false;
  return true;
}

function splitInsightBullets(text: string): string[] {
  return text
    .split(/\n+|\r+|\t+|(?:^|\s)[*•-]\s+/)
    .map((x) => cleanInsightLine(x))
    .filter(isUsefulInsightLine)
    .map((x) => cleanInsightLine(x));
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
    /['"]?summary['"]?\s*:\s*(['"])([\s\S]*?)\1\s*,[\s\S]*?(key_findings|findings)\s*:/i
  );
  const summary = summaryMatch?.[2]?.trim() ?? '';

  const findingsBlock =
    t.match(/['"]?(key_findings|findings)['"]?\s*:\s*\[([\s\S]*?)\]/i)?.[2] ?? '';
  const recsBlock =
    t.match(/['"]?recommendations['"]?\s*:\s*\[([\s\S]*?)\]/i)?.[1] ?? '';

  const extractList = (block: string): string[] => {
    const out: string[] = [];
    const re = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'/g;
    let m: RegExpExecArray | null = re.exec(block);
    while (m) {
      const raw = (m[1] ?? m[2] ?? '').replace(/\\(["'])/g, '$1');
      const v = cleanInsightLine(raw);
      if (isUsefulInsightLine(v)) out.push(v);
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

function normalizeInsights(raw: MLTrainResponse['insights']): InsightsView {
  const summaryText = (raw.summary || '').trim();
  const out: InsightsView = {
    summary: cleanInsightLine(summaryText),
    keyFindings: [...(raw.key_findings || [])]
      .map((x) => cleanInsightLine(String(x)))
      .filter(isUsefulInsightLine),
    recommendations: [...(raw.recommendations || [])]
      .map((x) => cleanInsightLine(String(x)))
      .filter(isUsefulInsightLine),
  };

  const hasEmbeddedObject =
    /key_findings\s*:|recommendations\s*:|\{\s*["']?summary["']?\s*:/i.test(
      summaryText
    );

  // Always try to unwrap embedded object-style payloads when detected.
  if (hasEmbeddedObject) {
    const compact = summaryText.replace(/\n/g, ' ').trim();
    const embedded =
      extractFirstJsonObject(compact) ?? extractPseudoInsightsObject(compact);
    if (embedded) {
      const s = typeof embedded.summary === 'string' ? embedded.summary.trim() : '';
      if (s) out.summary = cleanInsightLine(s);
      const k = embedded.key_findings;
      const r = embedded.recommendations;
      if (Array.isArray(k)) {
        out.keyFindings = k
          .map((x) => cleanInsightLine(String(x)))
          .filter(isUsefulInsightLine);
      }
      if (Array.isArray(r)) {
        out.recommendations = r
          .map((x) => cleanInsightLine(String(x)))
          .filter(isUsefulInsightLine);
      }
    }
  }

  // Fallback: sometimes the model dumps all sections into `summary` with
  // markdown labels (e.g. **Summary:**, **Key Findings:**, **Recommendations:**).
  if (out.keyFindings.length === 0 || out.recommendations.length === 0) {
    const compact = summaryText.replace(/\n/g, ' ').trim();
    const embedded =
      extractFirstJsonObject(compact) ?? extractPseudoInsightsObject(compact);
    if (embedded) {
      const s = typeof embedded.summary === 'string' ? embedded.summary.trim() : '';
      if (s) out.summary = s;
      const k = embedded.key_findings;
      const r = embedded.recommendations;
      if (Array.isArray(k) && out.keyFindings.length === 0) {
        out.keyFindings = k
          .map((x) => cleanInsightLine(String(x)))
          .filter(isUsefulInsightLine);
      }
      if (Array.isArray(r) && out.recommendations.length === 0) {
        out.recommendations = r
          .map((x) => cleanInsightLine(String(x)))
          .filter(isUsefulInsightLine);
      }
    }
    const summaryMatch = compact.match(/\*\*Summary:?\*\*\s*(.*?)(?=\*\*Key Findings:?\*\*|\*\*Recommendations:?\*\*|$)/i);
    const findingsMatch = compact.match(/\*\*Key Findings:?\*\*\s*(.*?)(?=\*\*Recommendations:?\*\*|$)/i);
    const recsMatch = compact.match(/\*\*Recommendations:?\*\*\s*(.*)$/i);

    if (summaryMatch?.[1]) out.summary = summaryMatch[1].trim();
    if (findingsMatch?.[1] && out.keyFindings.length === 0) {
      out.keyFindings = splitInsightBullets(findingsMatch[1]);
    }
    if (recsMatch?.[1] && out.recommendations.length === 0) {
      out.recommendations = splitInsightBullets(recsMatch[1]);
    }

    // Additional fallback when labels are present but regex parsing is imperfect.
    if (
      (out.keyFindings.length === 0 || out.recommendations.length === 0) &&
      /\*\*Key Findings/i.test(compact)
    ) {
      const keyIx = compact.search(/\*\*Key Findings:?\*\*/i);
      const recIx = compact.search(/\*\*Recommendations:?\*\*/i);
      if (keyIx >= 0) {
        const keyPart =
          recIx > keyIx
            ? compact.slice(keyIx, recIx).replace(/\*\*Key Findings:?\*\*/i, '')
            : compact.slice(keyIx).replace(/\*\*Key Findings:?\*\*/i, '');
        if (out.keyFindings.length === 0) out.keyFindings = splitInsightBullets(keyPart);
      }
      if (recIx >= 0) {
        const recPart = compact
          .slice(recIx)
          .replace(/\*\*Recommendations:?\*\*/i, '');
        if (out.recommendations.length === 0) {
          out.recommendations = splitInsightBullets(recPart);
        }
      }
    }

    // Last-resort tolerant extraction for malformed objects where keys are
    // unquoted or punctuation is injected (e.g. ", ? key_findings: [...]").
    if (
      (out.keyFindings.length === 0 || out.recommendations.length === 0) &&
      /key_findings\s*:/i.test(compact)
    ) {
      const arrFromKey = (key: string): string[] => {
        const m = compact.match(new RegExp(`${key}\\s*:\\s*\\[([\\s\\S]*?)\\]`, 'i'));
        if (!m?.[1]) return [];
        const block = m[1];
        const items: string[] = [];
        const re = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'/g;
        let k: RegExpExecArray | null = re.exec(block);
        while (k) {
          const rawItem = (k[1] ?? k[2] ?? '').replace(/\\(["'])/g, '$1');
          const line = cleanInsightLine(rawItem);
          if (isUsefulInsightLine(line)) items.push(line);
          k = re.exec(block);
        }
        return items;
      };
      if (out.keyFindings.length === 0) out.keyFindings = arrFromKey('key_findings');
      if (out.recommendations.length === 0) out.recommendations = arrFromKey('recommendations');
    }
  }

  // Strip common wrapper prefixes if they survive parsing.
  out.summary = out.summary
    .replace(/^here is the analysis\s*:\s*/i, '')
    .replace(/^analysis\s*:\s*/i, '')
    .trim();
  out.summary = cleanInsightLine(out.summary);
  return out;
}

function detectProblemType(
  yTrue: Array<number | string>,
  yPred: Array<number | string>
): 'regression' | 'classification' {
  if (!yTrue.length || !yPred.length) return 'regression';
  const numeric = yTrue.every((x) => typeof x === 'number' && Number.isFinite(x));
  if (!numeric) return 'classification';
  const vals = yTrue as number[];
  const unique = new Set(vals.map((v) => Math.round(v * 1e6) / 1e6)).size;
  const isIntegerLike = vals.every((v) => Math.abs(v - Math.round(v)) < 1e-9);
  // Heuristic fallback only: very low-cardinality integer target is likely classification.
  // Keep this strict to avoid misclassifying regression targets that are integer-valued.
  if (isIntegerLike && unique <= 10 && unique <= Math.max(2, Math.floor(vals.length * 0.02))) {
    return 'classification';
  }
  return 'regression';
}

function confusionMatrix(
  yTrue: Array<number | string>,
  yPred: Array<number | string>
): { labels: string[]; matrix: number[][] } {
  const labels = Array.from(
    new Set([...yTrue.map(String), ...yPred.map(String)])
  ).sort();
  const idx = new Map(labels.map((l, i) => [l, i]));
  const mat = labels.map(() => labels.map(() => 0));
  for (let i = 0; i < Math.min(yTrue.length, yPred.length); i += 1) {
    const a = idx.get(String(yTrue[i]));
    const p = idx.get(String(yPred[i]));
    if (a == null || p == null) continue;
    mat[a][p] += 1;
  }
  return { labels, matrix: mat };
}

function TrainModelPageContent() {
  const params = useParams<{ fileId: string }>();
  const searchParams = useSearchParams();
  const fileId = decodeURIComponent(params.fileId || '');
  const loopParam =
    searchParams.get('agent_loop') ?? searchParams.get('agentLoop') ?? '';
  const agentLoop = /^(1|true|yes|on)$/i.test(loopParam);
  const [result, setResult] = useState<MLTrainResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await trainMlModel(fileId, { agentLoop });
        setResult(data);
      } catch (err: any) {
        const detail = err?.response?.data?.detail;
        setError(
          typeof detail === 'string' ? detail : err?.message || 'Training failed'
        );
      } finally {
        setLoading(false);
      }
    };
    if (fileId) run();
  }, [fileId, agentLoop]);

  if (loading) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-10">
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading ML model...</p>
          </div>
        </div>
      </main>
    );
  }

  if (error || !result) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-10 space-y-4">
        <Link
          href={`/dashboard/${encodeURIComponent(fileId)}`}
          className="text-sm text-indigo-700 hover:text-indigo-800 hover:underline"
        >
          ← Back to dashboard
        </Link>
        <div className="rounded-lg border border-red-200 bg-red-50 p-6">
          <h1 className="text-lg font-semibold text-red-800 mb-1">
            Could not train model
          </h1>
          <p className="text-sm text-red-700">{error || 'Unknown error'}</p>
        </div>
      </main>
    );
  }

  const m = result.model;
  const insights = normalizeInsights(result.insights);
  const evalYTrue = m.evaluation?.y_true ?? m.sample_predictions.map((p) => p.actual);
  const evalYPred = m.evaluation?.y_pred ?? m.sample_predictions.map((p) => p.predicted);
  const metrics = m.metrics as Record<string, unknown>;
  const hasClsMetrics =
    typeof metrics.accuracy === 'number' || typeof metrics.f1 === 'number';
  const hasRegMetrics =
    typeof metrics.r2 === 'number' ||
    typeof metrics.mae === 'number' ||
    typeof metrics.rmse === 'number';

  let inferredType: 'regression' | 'classification';
  if (m.problem_type === 'classification' || hasClsMetrics) {
    inferredType = 'classification';
  } else if (m.problem_type === 'regression' || hasRegMetrics) {
    inferredType = 'regression';
  } else if (/class/i.test(m.model_type)) {
    inferredType = 'classification';
  } else if (/regression/i.test(m.model_type)) {
    inferredType = 'regression';
  } else {
    inferredType = detectProblemType(
      evalYTrue as number[],
      evalYPred as number[]
    );
  }

  const actuals = evalYTrue.map((v) => Number(v));
  const preds = evalYPred.map((v) => Number(v));
  const residuals = actuals.map((v, i) => v - (preds[i] ?? 0));
  const minV =
    actuals.length > 0
      ? Math.min(...actuals, ...preds)
      : 0;
  const maxV =
    actuals.length > 0
      ? Math.max(...actuals, ...preds)
      : 1;

  return (
    <main className="mx-auto max-w-5xl px-4 py-10 space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Link
          href={`/dashboard/${encodeURIComponent(fileId)}`}
          className="text-sm text-indigo-700 hover:text-indigo-800 hover:underline"
        >
          ← Back to dashboard
        </Link>
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-600">
          <span className="text-gray-500 uppercase tracking-wide">
            Auto-selected model
          </span>
          {agentLoop ? (
            <Link
              href={`/dashboard/${encodeURIComponent(fileId)}/train`}
              className="text-blue-700 hover:underline"
            >
              Switch to standard training
            </Link>
          ) : (
            <Link
              href={`/dashboard/${encodeURIComponent(fileId)}/train?agent_loop=1`}
              className="text-blue-700 hover:underline"
            >
              Train with LLM agent loop (5 iterations)
            </Link>
          )}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-6">
        <h1 className="text-2xl font-bold text-gray-900">ML Model Results</h1>
        <p className="mt-2 text-sm text-gray-600">
          Model: <span className="font-medium">{m.model_type}</span> | Target:{' '}
          <span className="font-medium">{m.target_column}</span>
        </p>
        <p className="mt-1 text-sm text-gray-600">
          Features: <span className="font-medium">{m.feature_columns.join(', ')}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {inferredType === 'classification' ? (
          <>
            <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="text-sm text-gray-500">Accuracy</div>
              <div className="text-2xl font-semibold text-gray-900">
                {typeof metrics.accuracy === 'number'
                  ? metrics.accuracy.toFixed(3)
                  : '—'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="text-sm text-gray-500">F1 (macro)</div>
              <div className="text-2xl font-semibold text-gray-900">
                {typeof metrics.f1 === 'number' ? metrics.f1.toFixed(3) : '—'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="text-sm text-gray-500">Model</div>
              <div className="text-sm font-semibold text-gray-900 leading-snug">
                {m.model_type.replace(/_/g, ' ')}
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="text-sm text-gray-500">R²</div>
              <div className="text-2xl font-semibold text-gray-900">
                {typeof metrics.r2 === 'number' ? metrics.r2.toFixed(3) : '—'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="text-sm text-gray-500">MAE</div>
              <div className="text-2xl font-semibold text-gray-900">
                {typeof metrics.mae === 'number' ? metrics.mae.toFixed(3) : '—'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="text-sm text-gray-500">RMSE</div>
              <div className="text-2xl font-semibold text-gray-900">
                {typeof metrics.rmse === 'number' ? metrics.rmse.toFixed(3) : '—'}
              </div>
            </div>
          </>
        )}
      </div>

      {typeof (metrics as Record<string, unknown>).single_split_f1 === 'number' ? (
        <p className="mb-4 text-xs text-gray-600">
          Top metrics use the same repeated holdout estimate as the agent loop (aligned with the
          iteration chart). Single random 80/20 split on the trained model: F1{' '}
          {Number((metrics as Record<string, unknown>).single_split_f1).toFixed(3)}, accuracy{' '}
          {Number((metrics as Record<string, unknown>).single_split_accuracy).toFixed(3)}.
        </p>
      ) : null}
      {typeof (metrics as Record<string, unknown>).single_split_r2 === 'number' ? (
        <p className="mb-4 text-xs text-gray-600">
          R² above uses repeated holdout (agent protocol). Single split R²:{' '}
          {Number((metrics as Record<string, unknown>).single_split_r2).toFixed(3)}.
        </p>
      ) : null}

      <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">
          Iterative agent training (validation over time)
        </h2>
        {result.agent_iterations && result.agent_iterations.length > 0 ? (
          <>
            <p className="text-sm text-gray-700 mb-4">
              Best LLM iteration (validation):{' '}
              <span className="font-semibold">
                {result.agent_best_iteration ?? '—'}
              </span>
              . Training first applied the same validation-based feature engineering as the standard
              path (when applicable), then the 5-round agent run below. Final features may still be
              adjusted by pair champion / subset search when those beat the best LLM iteration.
            </p>

            <div className="mb-8 rounded-xl border border-indigo-100 bg-white shadow-sm p-4">
              <h3 className="text-base font-semibold text-gray-900 mb-1">
                {inferredType === 'classification'
                  ? 'Macro F1 over iterations'
                  : 'Validation R² over iterations'}
              </h3>
              <p className="text-xs text-gray-600 mb-3">
                Each point is the mean across repeated held-out validation folds for that
                iteration&apos;s features (
                {inferredType === 'classification'
                  ? 'macro F1; list below also shows mean validation accuracy per iteration'
                  : 'R²; higher is better'}
                ).
              </p>
              <div className="w-full min-h-[300px]">
                <Plot
                  data={[
                    {
                      type: 'scatter',
                      mode: 'lines+markers',
                      x: result.agent_iterations.map((s) => s.iteration),
                      y: result.agent_iterations.map((s) =>
                        agentIterationPrimaryMetric(s, inferredType)
                      ),
                      name:
                        inferredType === 'classification'
                          ? 'Macro F1 (mean val.)'
                          : 'R² (mean val.)',
                      line: { color: '#4f46e5', width: 2, shape: 'spline' },
                      marker: { color: '#6366f1', size: 11, line: { color: '#fff', width: 1 } },
                    },
                    ...(typeof result.agent_best_iteration === 'number'
                      ? [
                          {
                            type: 'scatter' as const,
                            mode: 'markers' as const,
                            x: result.agent_iterations
                              .filter(
                                (s) => s.iteration === result.agent_best_iteration
                              )
                              .map((s) => s.iteration),
                            y: result.agent_iterations
                              .filter(
                                (s) => s.iteration === result.agent_best_iteration
                              )
                              .map((s) =>
                                agentIterationPrimaryMetric(s, inferredType)
                              ),
                            name: 'Best LLM iteration',
                            marker: {
                              color: '#059669',
                              size: 16,
                              symbol: 'diamond',
                              line: { color: '#fff', width: 1 },
                            },
                          },
                        ]
                      : []),
                  ]}
                  layout={{
                    title: {
                      text:
                        inferredType === 'classification'
                          ? 'Macro F1 vs iteration'
                          : 'R² vs iteration',
                      font: { size: 14 },
                    },
                    autosize: true,
                    height: 320,
                    margin: { l: 56, r: 24, t: 48, b: 48 },
                    xaxis: {
                      title: 'Iteration',
                      dtick: 1,
                      tick0: 1,
                      range: [
                        0.5,
                        Math.max(
                          5.5,
                          ...result.agent_iterations.map((s) => s.iteration)
                        ),
                      ],
                    },
                    yaxis: {
                      title:
                        inferredType === 'classification'
                          ? 'Mean validation F1'
                          : 'Mean validation R²',
                    },
                    legend: {
                      orientation: 'h',
                      yanchor: 'bottom',
                      y: 1.02,
                      xanchor: 'right',
                      x: 1,
                    },
                    plot_bgcolor: '#f8fafc',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: { family: 'system-ui, sans-serif', size: 12, color: '#334155' },
                  }}
                  config={{ responsive: true, displaylogo: false }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler
                />
              </div>
              {Boolean(
                (m.selection as { agent_loop?: { refined_from_best_iteration?: boolean } } | undefined)
                  ?.agent_loop?.refined_from_best_iteration
              ) ? (
                <p className="mt-3 text-xs text-gray-600">
                  The green marker is the best <strong>LLM</strong> iteration. Final features may
                  differ after a <strong>pair champion</strong> step and/or the same{' '}
                  <strong>subset search</strong> used in standard training (whichever scores higher
                  on repeated holdout — see features under ML Model Results).
                </p>
              ) : null}
            </div>

            <ol className="space-y-4">
              {result.agent_iterations.map((step) => (
                <li
                  key={step.iteration}
                  className="rounded-xl border border-slate-200 bg-white shadow-sm p-4 text-sm"
                >
                  <div className="font-semibold text-gray-900">
                    Iteration {step.iteration}{' '}
                    <span className="font-normal text-gray-600">
                      · {step.model.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div className="mt-1 text-sm text-gray-800">
                    {inferredType === 'classification' ? (
                      <>
                        <span className="text-gray-500">Macro F1 (mean on val folds):</span>{' '}
                        {(step.f1_macro ?? step.score).toFixed(4)}
                        {' · '}
                        <span className="text-gray-500">Accuracy (mean on val folds):</span>{' '}
                        {typeof step.validation_accuracy === 'number'
                          ? step.validation_accuracy.toFixed(4)
                          : '—'}
                      </>
                    ) : (
                      <>
                        <span className="text-gray-500">R² (mean on val folds):</span>{' '}
                        {(step.validation_r2 ?? step.score).toFixed(4)}
                      </>
                    )}
                  </div>
                  <p className="mt-1 text-gray-700">
                    <span className="text-gray-500">Features: </span>
                    {step.features.join(', ')}
                  </p>
                  <p className="mt-2 text-gray-600 leading-relaxed">{step.reasoning}</p>
                </li>
              ))}
            </ol>
          </>
        ) : (
          <div className="rounded-xl border border-dashed border-indigo-300 bg-white/95 p-5 text-sm text-slate-700">
            <p className="mb-3">
              This result used the <strong>standard</strong> training path (one automatic pipeline).
              To see a <strong>chart and per-iteration log</strong> (5 LLM rounds with validation
              feedback), run training with the agent loop enabled.
            </p>
            {agentLoop ? (
              <p className="text-amber-800 text-xs">
                Agent mode was requested but no iteration data was returned. Try refreshing or
                check the API response.
              </p>
            ) : (
              <Link
                href={`/dashboard/${encodeURIComponent(fileId)}/train?agent_loop=1`}
                className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-indigo-700"
              >
                Run agent-loop training (5 iterations + chart)
              </Link>
            )}
          </div>
        )}
      </div>

      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Model Graph</h2>
        {inferredType === 'regression' ? (
          <div className="space-y-8">
            <p className="text-sm text-gray-600">
              Regression diagnostics using holdout predictions.
            </p>
            <div className="w-full min-h-[320px]">
              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'markers',
                    x: actuals,
                    y: preds,
                    marker: { color: '#2563eb', size: 8, opacity: 0.7 },
                    name: 'Predictions',
                  },
                  {
                    type: 'scatter',
                    mode: 'lines',
                    x: [minV, maxV],
                    y: [minV, maxV],
                    line: { color: '#ef4444', dash: 'dash', width: 2 },
                    name: 'Perfect fit',
                  },
                ]}
                layout={{
                  title: { text: 'Predicted vs Actual' },
                  autosize: true,
                  height: 340,
                  margin: { l: 70, r: 20, t: 50, b: 60 },
                  xaxis: { title: 'Actual values' },
                  yaxis: { title: 'Predicted values' },
                  plot_bgcolor: '#f8fafc',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  font: { family: 'system-ui, sans-serif', size: 12, color: '#334155' },
                  legend: { orientation: 'h' as const, y: 1.12 },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
            <div className="w-full min-h-[320px]">
              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'markers',
                    x: preds,
                    y: residuals,
                    marker: { color: '#0ea5e9', size: 7, opacity: 0.7 },
                    name: 'Residuals',
                  },
                  {
                    type: 'scatter',
                    mode: 'lines',
                    x: [Math.min(...preds), Math.max(...preds)],
                    y: [0, 0],
                    line: { color: '#ef4444', dash: 'dash', width: 2 },
                    name: 'Zero error',
                  },
                ]}
                layout={{
                  title: { text: 'Residuals vs Predicted' },
                  autosize: true,
                  height: 340,
                  margin: { l: 70, r: 20, t: 50, b: 60 },
                  xaxis: { title: 'Predicted values' },
                  yaxis: { title: 'Residuals (actual - predicted)' },
                  plot_bgcolor: '#f8fafc',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  font: { family: 'system-ui, sans-serif', size: 12, color: '#334155' },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
            <div className="w-full min-h-[320px]">
              <Plot
                data={[
                  {
                    type: 'histogram',
                    x: residuals,
                    marker: { color: '#8b5cf6' },
                    nbinsx: Math.min(40, Math.max(10, Math.round(Math.sqrt(residuals.length || 1)))),
                    name: 'Residual distribution',
                  },
                ]}
                layout={{
                  title: { text: 'Histogram of Residuals' },
                  autosize: true,
                  height: 340,
                  margin: { l: 70, r: 20, t: 50, b: 60 },
                  xaxis: { title: 'Residual' },
                  yaxis: { title: 'Frequency' },
                  plot_bgcolor: '#f8fafc',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  font: { family: 'system-ui, sans-serif', size: 12, color: '#334155' },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            <p className="text-sm text-gray-600">
              Classification diagnostics using holdout predictions.
            </p>
            {(() => {
              const cm = confusionMatrix(
                evalYTrue as Array<number | string>,
                evalYPred as Array<number | string>
              );
              return (
                <div className="w-full min-h-[320px]">
                  <Plot
                    data={[
                      {
                        type: 'heatmap',
                        x: cm.labels,
                        y: cm.labels,
                        z: cm.matrix,
                        colorscale: 'Blues',
                        colorbar: { title: 'Count' },
                      },
                    ]}
                    layout={{
                      title: { text: 'Confusion Matrix' },
                      autosize: true,
                      height: 360,
                      margin: { l: 80, r: 20, t: 50, b: 60 },
                      xaxis: { title: 'Predicted label' },
                      yaxis: { title: 'True label' },
                      plot_bgcolor: '#f8fafc',
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      font: { family: 'system-ui, sans-serif', size: 12, color: '#334155' },
                    }}
                    config={{ responsive: true, displaylogo: false }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler
                  />
                </div>
              );
            })()}
          </div>
        )}
        <div className="mt-4 text-xs text-gray-500">
          Problem type auto-detected as <span className="font-medium">{inferredType}</span>.
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">AI Insights</h2>
        <h3 className="text-sm font-semibold text-gray-800 mb-2">Key Findings</h3>
        <ul className="space-y-2">
          {insights.keyFindings.length > 0 ? insights.keyFindings.map((x, i) => (
            <li key={i} className="flex items-start text-sm text-gray-700">
              <span className="mr-2 text-green-600">•</span>
              <span>{x}</span>
            </li>
          )) : <li className="text-sm text-gray-500">No key findings returned.</li>}
        </ul>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Feature Contribution (Coefficients)
        </h2>
        {inferredType === 'classification' && (!m.coefficients || m.coefficients.length === 0) ? (
          <p className="text-sm text-gray-600">
            This classifier does not expose a simple coefficient table in the UI.
            Use the confusion matrix and class metrics above to judge feature-level
            behavior, or train a linear model for an explicit weight list.
          </p>
        ) : (
          <>
            <p className="text-sm text-gray-600 mb-4">
              Coefficients come from the trained linear regression equation. A positive
              coefficient means that, holding other features constant, increasing that
              feature tends to increase the predicted target. A negative coefficient
              means the opposite. Larger absolute values indicate stronger influence
              in this linear model, but coefficient size is also affected by each
              feature&apos;s scale.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-gray-700">Feature</th>
                    <th className="px-3 py-2 text-left text-gray-700">Coefficient</th>
                  </tr>
                </thead>
                <tbody>
                  {(m.coefficients || []).map((c, i) => (
                    <tr key={i} className="border-t border-gray-100">
                      <td className="px-3 py-2 text-gray-800">{c.feature}</td>
                      <td className="px-3 py-2 text-gray-700">
                        {c.coefficient > 0 ? '+' : ''}
                        {c.coefficient.toFixed(5)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </main>
  );
}

export default function TrainModelPage() {
  return (
    <Suspense
      fallback={
        <main className="mx-auto max-w-5xl px-4 py-10">
          <div className="flex min-h-[40vh] items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-600">
            Loading…
          </div>
        </main>
      }
    >
      <TrainModelPageContent />
    </Suspense>
  );
}
