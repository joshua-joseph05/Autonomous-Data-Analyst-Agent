'use client';

import dynamic from 'next/dynamic';
import type { ComponentType, CSSProperties } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { getExploratorySeries } from '@/lib/api';
import type { AnalysisResult } from '@/lib/api';

const Plot = dynamic(
  () => import('react-plotly.js'),
  {
    ssr: false,
    loading: () => (
      <div className="h-72 flex items-center justify-center rounded-lg bg-gray-50 text-sm text-gray-500">
        Loading chart…
      </div>
    ),
  }
) as ComponentType<{
  data: unknown[];
  layout: Record<string, unknown>;
  config?: Record<string, unknown>;
  style?: CSSProperties;
  useResizeHandler?: boolean;
}>;

function colIndex(columns: AnalysisResult['summary_stats']['columns'], name: string) {
  return columns.findIndex((c) => c.name === name);
}

function parseNumericColumn(
  rows: string[][],
  colIdx: number
): number[] {
  const out: number[] = [];
  for (const row of rows) {
    if (colIdx < 0 || colIdx >= row.length) continue;
    const v = parseFloat(String(row[colIdx]).replace(/,/g, ''));
    if (!Number.isNaN(v)) out.push(v);
  }
  return out;
}

function parseFullSeriesColumn(series?: Array<number | null>): number[] {
  if (!Array.isArray(series)) return [];
  const out: number[] = [];
  for (let i = 0; i < series.length; i += 1) {
    const v = series[i];
    if (typeof v === 'number' && !Number.isNaN(v)) out.push(v);
  }
  return out;
}

function buildCorrelationHeatmap(
  matrix: Record<string, unknown>
): { data: object[]; labels: string[] } | null {
  const keys = Object.keys(matrix).filter(
    (k) => matrix[k] != null && typeof matrix[k] === 'object'
  );
  if (keys.length < 2) return null;
  const labels = [...keys].sort();
  const z = labels.map((row) =>
    labels.map((col) => {
      const rowObj = matrix[row] as Record<string, unknown> | undefined;
      const raw = rowObj?.[col];
      if (typeof raw === 'number' && !Number.isNaN(raw)) return raw;
      return null;
    })
  );
  return {
    labels,
    data: [
      {
        type: 'heatmap' as const,
        x: labels,
        y: labels,
        z,
        colorscale: 'RdBu',
        zmid: 0,
        reversescale: true,
        colorbar: { title: 'Pearson r' },
      },
    ],
  };
}

function computeMainCorrelationNetwork(
  matrix: Record<string, unknown>,
  allowed: string[],
  threshold = 0.35
): Set<string> {
  const keys = Object.keys(matrix).filter(
    (k) =>
      allowed.includes(k) && matrix[k] != null && typeof matrix[k] === 'object'
  );
  if (keys.length < 2) return new Set(keys);

  const adj = new Map<string, Set<string>>();
  for (const k of keys) adj.set(k, new Set());

  for (const a of keys) {
    const row = matrix[a] as Record<string, unknown>;
    for (const b of keys) {
      if (a === b) continue;
      const raw = row?.[b];
      if (typeof raw !== 'number' || Number.isNaN(raw)) continue;
      if (Math.abs(raw) >= threshold) {
        adj.get(a)?.add(b);
        adj.get(b)?.add(a);
      }
    }
  }

  const visited = new Set<string>();
  let best = new Set<string>();

  for (const start of keys) {
    if (visited.has(start)) continue;
    const stack = [start];
    const comp = new Set<string>();
    while (stack.length) {
      const u = stack.pop() as string;
      if (visited.has(u)) continue;
      visited.add(u);
      comp.add(u);
      const neigh = Array.from(adj.get(u) ?? []);
      for (let i = 0; i < neigh.length; i += 1) {
        const v = neigh[i];
        if (!visited.has(v)) stack.push(v);
      }
    }
    if (comp.size > best.size) best = comp;
  }

  return best;
}

function correlationCentralityScore(
  matrix: Record<string, unknown>,
  name: string,
  within: Set<string>
): number {
  const row = matrix[name] as Record<string, unknown> | undefined;
  if (!row) return 0;
  let sum = 0;
  const others = Array.from(within);
  for (let i = 0; i < others.length; i += 1) {
    const other = others[i];
    if (other === name) continue;
    const raw = row[other];
    if (typeof raw === 'number' && !Number.isNaN(raw)) sum += Math.abs(raw);
  }
  return sum;
}

function pickMostImportantNumericColumn(
  matrix: Record<string, unknown>,
  numericColumnNames: string[]
): string | null {
  const main = computeMainCorrelationNetwork(matrix, numericColumnNames);
  const candidates = Array.from(main);
  if (candidates.length === 0) return numericColumnNames[0] ?? null;

  let best: { name: string; score: number } | null = null;
  for (const name of candidates) {
    const score = correlationCentralityScore(matrix, name, main);
    if (!best || score > best.score) best = { name, score };
  }
  return best?.name ?? (numericColumnNames[0] ?? null);
}

const baseLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: '#fafafa',
  font: { family: 'system-ui, sans-serif', size: 12, color: '#374151' },
  margin: { l: 48, r: 24, t: 16, b: 48 },
  autosize: true,
};

const plotConfig = {
  displayModeBar: true,
  displaylogo: false,
  responsive: true,
};

interface ExploratoryAnalysisProps {
  analysis: AnalysisResult;
}

export default function ExploratoryAnalysis({ analysis }: ExploratoryAnalysisProps) {
  const { columns } = analysis.summary_stats;
  const rows = analysis.cleaned_data;
  const numericCols = columns.filter((c) => c.inferred_type.includes('numeric'));
  const numericNames = numericCols.map((c) => c.name);
  const [fullSeries, setFullSeries] = useState<Record<string, Array<number | null>>>(
    {}
  );

  const importantColName = useMemo(() => {
    const matrix = analysis.correlations.matrix as Record<string, unknown>;
    const picked = pickMostImportantNumericColumn(matrix, numericNames);
    return picked ?? numericNames[0] ?? '';
  }, [analysis, numericNames]);

  const selectedScatterPair = useMemo(() => {
    const matrix = analysis.correlations.matrix as Record<string, unknown>;
    const main = computeMainCorrelationNetwork(matrix, numericNames);
    const sigs = analysis.correlations.significant.filter(
      (s) => main.has(s.column1) && main.has(s.column2)
    );
    return (
      sigs.find(
        (s) => s.column1 === importantColName || s.column2 === importantColName
      ) ?? sigs[0] ?? null
    );
  }, [analysis, numericNames, importantColName]);

  const requestedFullSeriesColumns = useMemo(() => {
    const cols = new Set<string>();
    if (importantColName) cols.add(importantColName);
    if (selectedScatterPair?.column1) cols.add(selectedScatterPair.column1);
    if (selectedScatterPair?.column2) cols.add(selectedScatterPair.column2);
    return Array.from(cols);
  }, [importantColName, selectedScatterPair]);

  useEffect(() => {
    let cancelled = false;
    if (!analysis.file_id || requestedFullSeriesColumns.length === 0) {
      setFullSeries({});
      return;
    }
    getExploratorySeries(analysis.file_id, requestedFullSeriesColumns)
      .then((res) => {
        if (!cancelled) setFullSeries(res.series || {});
      })
      .catch(() => {
        if (!cancelled) setFullSeries({});
      });
    return () => {
      cancelled = true;
    };
  }, [analysis.file_id, requestedFullSeriesColumns]);

  const importantHistSpec = useMemo(() => {
    if (!importantColName) return null;
    let vals = parseFullSeriesColumn(fullSeries[importantColName]);
    if (vals.length < 2 && rows.length > 0) {
      const idx = colIndex(columns, importantColName);
      vals = parseNumericColumn(rows, idx);
    }
    if (vals.length < 2) return null;
    return {
      title: `Histogram: ${importantColName}`,
      description:
        'This feature is selected because it connects strongly to many other numeric columns (based on the correlation network). The histogram shows its distribution: skew, clusters, and typical ranges.',
      data: [
        {
          type: 'histogram' as const,
          x: vals,
          nbinsx: Math.min(30, Math.max(10, Math.round(Math.sqrt(vals.length)))),
          marker: { color: '#3b82f6', line: { color: '#fff', width: 0.5 } },
        },
      ],
      layout: {
        ...baseLayout,
        margin: { l: 92, r: 42, t: 24, b: 82 },
        xaxis: {
          title: { text: importantColName, font: { size: 15 } },
          tickfont: { size: 13 },
        },
        yaxis: {
          title: { text: 'Count (full cleaned rows)', font: { size: 15 } },
          tickfont: { size: 13 },
        },
        height: 300,
      },
    };
  }, [importantColName, columns, rows, fullSeries]);

  const boxSpec = useMemo(() => {
    if (!importantColName) return null;
    let vals = parseFullSeriesColumn(fullSeries[importantColName]);
    if (vals.length < 2 && rows.length > 0) {
      const idx = colIndex(columns, importantColName);
      vals = parseNumericColumn(rows, idx);
    }
    if (vals.length < 2) return null;
    return {
      title: `Box plot: ${importantColName}`,
      description:
        'A box plot summarizes the median, quartiles (middle 50%), and potential outliers. It is a fast way to see spread and extreme values without focusing on bins.',
      data: [
        {
          type: 'box' as const,
          y: vals,
          name: importantColName,
          boxpoints: 'outliers' as const,
          jitter: 0.35,
          pointpos: -1.6,
          marker: { color: '#10b981', opacity: 0.6 },
          line: { color: '#059669' },
        },
      ],
      layout: {
        ...baseLayout,
        margin: { l: 92, r: 42, t: 24, b: 82 },
        xaxis: {
          title: { text: 'Distribution', font: { size: 15 } },
          tickfont: { size: 13 },
        },
        yaxis: {
          title: { text: importantColName, font: { size: 15 } },
          tickfont: { size: 13 },
        },
        height: 320,
      },
    };
  }, [importantColName, columns, rows, fullSeries]);

  const scatterSpec = useMemo(() => {
    const sig = selectedScatterPair;
    if (!sig) return null;
    const ix = colIndex(columns, sig.column1);
    const iy = colIndex(columns, sig.column2);
    if (ix < 0 || iy < 0) return null;
    const t1 = columns[ix]?.inferred_type ?? '';
    const t2 = columns[iy]?.inferred_type ?? '';
    if (!t1.includes('numeric') || !t2.includes('numeric')) return null;
    const sx = fullSeries[sig.column1];
    const sy = fullSeries[sig.column2];
    const xs: number[] = [];
    const ys: number[] = [];
    if (Array.isArray(sx) && Array.isArray(sy) && sx.length === sy.length) {
      for (let i = 0; i < sx.length; i += 1) {
        const x = sx[i];
        const y = sy[i];
        if (typeof x === 'number' && typeof y === 'number') {
          xs.push(x);
          ys.push(y);
        }
      }
    }
    if (xs.length < 2) {
      for (const row of rows) {
        const x = parseFloat(String(row[ix]).replace(/,/g, ''));
        const y = parseFloat(String(row[iy]).replace(/,/g, ''));
        if (!Number.isNaN(x) && !Number.isNaN(y)) {
          xs.push(x);
          ys.push(y);
        }
      }
    }
    if (xs.length < 2) return null;
    return {
      title: `Scatter: ${sig.column1} vs ${sig.column2}`,
      description: `Each point is one row in the full cleaned dataset when available. Pearson r≈${sig.correlation.toFixed(2)} in the full numeric table (not recomputed here). Scatter plots reveal curvature, clusters, and outliers that correlation alone can hide.`,
      data: [
        {
          type: 'scatter' as const,
          mode: 'markers',
          x: xs,
          y: ys,
          marker: { size: 8, opacity: 0.65, color: '#6366f1' },
        },
      ],
      layout: {
        ...baseLayout,
        xaxis: { title: sig.column1 },
        yaxis: { title: sig.column2 },
        height: 320,
      },
    };
  }, [selectedScatterPair, columns, rows, fullSeries]);

  const heatmapSpec = useMemo(() => {
    const built = buildCorrelationHeatmap(
      analysis.correlations.matrix as Record<string, unknown>
    );
    if (!built) return null;
    return {
      title: 'Correlation heatmap',
      description:
        'Red-blue cells are pairwise Pearson correlations among numeric columns used in the matrix. Darker red is stronger positive linear association; darker blue is stronger negative. Diagonal is always 1. Use this to spot redundant features or candidate relationships.',
      data: built.data,
      layout: {
        ...baseLayout,
        height: Math.min(520, 120 + built.labels.length * 28),
        xaxis: { side: 'bottom' as const, tickangle: -40 },
        yaxis: { side: 'left' as const },
      },
    };
  }, [analysis]);

  const cards = [
    importantHistSpec,
    boxSpec,
    scatterSpec,
    heatmapSpec,
  ].filter(Boolean) as Array<{
    title: string;
    description: string;
    data: unknown[];
    layout: Record<string, unknown>;
  }>;

  if (cards.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-2">
          Exploratory analysis
        </h2>
        <p className="text-sm text-gray-600">
          Not enough structured preview data to build charts. Try re-uploading after
          analysis completes.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-1">
        Exploratory analysis
      </h2>
      <p className="text-sm text-gray-600 mb-6">
        Quick views from full cleaned column data (with preview fallback if the
        full-series fetch is unavailable). Charts are labeled; read each caption
        for what the view is meant to show.
      </p>
      <div className="space-y-10">
        {cards.map((spec, i) => (
          <div
            key={i}
            className="border border-gray-100 rounded-xl p-4 md:p-5 bg-gray-50/50"
          >
            <h3 className="text-base font-semibold text-gray-900 mb-2">
              {spec.title}
            </h3>
            <p className="text-sm text-gray-600 leading-relaxed mb-4 max-w-3xl">
              {spec.description}
            </p>
            <div className="w-full min-h-[280px]">
              <Plot
                data={spec.data}
                layout={spec.layout}
                config={plotConfig}
                style={{ width: '100%', height: '100%', minHeight: 280 }}
                useResizeHandler
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
