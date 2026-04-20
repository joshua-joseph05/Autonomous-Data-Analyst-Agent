'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getAnalysis, downloadCleanedCsv, AnalysisResult } from '@/lib/api';
import ExploratoryAnalysis from '@/components/ExploratoryAnalysis';

interface DashboardProps {
  fileId: string;
}

export default function Dashboard({ fileId }: DashboardProps) {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exportingCleaned, setExportingCleaned] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    const loadAnalysis = async () => {
      try {
        const data = await getAnalysis(fileId);
        setAnalysis(data);
        setError(null);
      } catch (err: any) {
        const detail = err?.response?.data?.detail;
        const detailStr = Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d?.msg || JSON.stringify(d)).join('; ')
          : typeof detail === 'string'
            ? detail
            : err?.response?.data?.message;
        setError(detailStr || err?.message || 'Failed to load analysis');
      } finally {
        setLoading(false);
      }
    };

    loadAnalysis();
  }, [fileId]);

  const handleDownloadCleaned = async () => {
    if (!analysis) return;
    setExportError(null);
    setExportingCleaned(true);
    try {
      await downloadCleanedCsv(analysis.file_id, analysis.filename);
    } catch (err: unknown) {
      const ax = err as {
        response?: { data?: unknown; status?: number };
        message?: string;
      };
      const status = ax?.response?.status;
      let msg = ax?.message || 'Download failed';
      const data = ax?.response?.data;
      if (data instanceof Blob) {
        const text = await data.text();
        try {
          const parsed = JSON.parse(text) as {
            detail?: string | Array<{ msg?: string }>;
          };
          if (typeof parsed?.detail === 'string') {
            msg = parsed.detail;
          } else if (Array.isArray(parsed?.detail)) {
            msg = parsed.detail
              .map((d) => d?.msg || JSON.stringify(d))
              .join('; ');
          }
        } catch {
          if (text.includes('<!DOCTYPE') || text.includes('<html')) {
            msg =
              `Got an HTML page instead of a CSV (HTTP ${status ?? '?'}). ` +
              'Confirm the API is running and NEXT_PUBLIC_API_URL is the FastAPI base URL (e.g. http://localhost:8000).';
          } else if (text.trim()) {
            msg = text.trim().slice(0, 300);
          }
        }
      }
      setExportError(msg);
    } finally {
      setExportingCleaned(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <svg className="w-5 h-5 text-red-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div>
            <h3 className="text-red-800 font-medium">Error loading analysis</h3>
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 relative">
        <div className="absolute top-6 right-6 flex flex-col items-end gap-2">
          <Link
            href={`/dashboard/${encodeURIComponent(fileId)}/train`}
            className="inline-flex items-center justify-center rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-emerald-700"
          >
            Train ML model
          </Link>
          <Link
            href={`/dashboard/${encodeURIComponent(fileId)}/train?agent_loop=1`}
            className="inline-flex items-center justify-center rounded-lg border border-indigo-300 bg-indigo-50/40 px-4 py-2 text-sm font-medium text-indigo-800 hover:bg-indigo-100/50"
          >
            Agent loop (5 rounds + chart)
          </Link>
        </div>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 flex-1">
            <h1 className="text-2xl font-bold text-gray-900">
              Analysis Dashboard
            </h1>
            <p className="text-slate-600 mt-1 truncate" title={analysis.filename}>
              {analysis.filename}
            </p>
            <p className="text-sm text-slate-500 mt-2">
              Download the full cleaned CSV (all rows, same cleaning as analysis).
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handleDownloadCleaned}
                disabled={exportingCleaned}
                className="inline-flex items-center justify-center rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:opacity-50 disabled:pointer-events-none"
              >
                {exportingCleaned ? 'Preparing download…' : 'Download cleaned CSV'}
              </button>
            </div>
            {exportError ? (
              <p className="text-sm text-red-600 mt-2">{exportError}</p>
            ) : null}
          </div>
        </div>
      </div>

      {/* How this file was cleaned (server-reported stats) */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-2">
          How this dataset was cleaned
        </h2>
        {analysis.cleaning_report ? (
          <div className="space-y-3 text-sm text-slate-700">
            <p>
              For <span className="font-medium text-gray-900">{analysis.filename}</span>
              , the uploaded table started with{' '}
              <span className="font-medium text-gray-900">
                {analysis.cleaning_report.rows_start.toLocaleString()} rows
              </span>{' '}
              and{' '}
              <span className="font-medium text-gray-900">
                {analysis.cleaning_report.columns_start} columns
              </span>
              . After cleaning it has{' '}
              <span className="font-medium text-gray-900">
                {analysis.cleaning_report.rows_end.toLocaleString()} rows
              </span>{' '}
              and{' '}
              <span className="font-medium text-gray-900">
                {analysis.cleaning_report.columns_end} columns
              </span>
              . Charts, statistics, correlations, and the CSV download all use this
              cleaned version.
            </p>
            <p>
              <span className="font-medium text-slate-800">High-missing columns.</span>{' '}
              Using a{' '}
              {analysis.cleaning_report.null_threshold_percent}% missing-value cutoff,{' '}
              {analysis.cleaning_report.columns_dropped_high_null.length === 0
                ? 'no columns were dropped for being too sparse.'
                : `${analysis.cleaning_report.columns_dropped_high_null.length} column(s) were removed: ${analysis.cleaning_report.columns_dropped_high_null.join(', ')}.`}
            </p>
            <p>
              <span className="font-medium text-slate-800">Column types.</span>{' '}
              {analysis.cleaning_report.low_cardinality_columns_as_category.length ===
              0
                ? 'No text columns were promoted to category type.'
                : `${analysis.cleaning_report.low_cardinality_columns_as_category.length} low-cardinality text column(s) were stored as categories: ${analysis.cleaning_report.low_cardinality_columns_as_category.join(', ')}. Numeric columns were coerced to numbers (unparseable values became empty cells).`}
            </p>
            <p>
              <span className="font-medium text-slate-800">Row checks.</span>{' '}
              {analysis.cleaning_report.outlier_row_removal_enabled ? (
                <>
                  {analysis.cleaning_report.iqr_outlier_rows_removed === 0
                    ? 'No rows were removed as numeric IQR outliers.'
                    : `${analysis.cleaning_report.iqr_outlier_rows_removed} row(s) were removed because at least one numeric value sat outside the Tukey fences (IQR multiplier ${analysis.cleaning_report.iqr_multiplier}).`}{' '}
                  {analysis.cleaning_report.sign_anomaly_rows_removed === 0
                    ? 'No rows were removed for sign mismatches (e.g. a lone negative in an otherwise positive-dominant column).'
                    : `${analysis.cleaning_report.sign_anomaly_rows_removed} row(s) were removed for sign mismatches (dominant-positive/negative rule at ${(analysis.cleaning_report.sign_anomaly_threshold * 100).toFixed(0)}% or a single inconsistent value).`}
                </>
              ) : (
                'IQR and sign-based row removal were disabled on the server for this run.'
              )}
            </p>
            <p>
              <span className="font-medium text-slate-800">Duplicates and dates.</span>{' '}
              {analysis.cleaning_report.duplicate_rows_removed === 0
                ? 'No exact duplicate rows were found.'
                : `${analysis.cleaning_report.duplicate_rows_removed} duplicate row(s) were removed.`}{' '}
              Date-like columns were parsed where possible; any cells that stayed
              empty after parsing are counted in the final gap step below.
            </p>
            <p>
              <span className="font-medium text-slate-800">Imputation.</span>{' '}
              {analysis.cleaning_report.numeric_columns_filled.length === 0
                ? 'No numeric columns needed missing-value imputation.'
                : `Numeric gaps were filled (median, or 0 if needed) in ${analysis.cleaning_report.numeric_columns_filled.length} column(s): ${analysis.cleaning_report.numeric_columns_filled.join(', ')}.`}{' '}
              {analysis.cleaning_report.categorical_columns_filled.length === 0
                ? 'No category-style columns needed imputation.'
                : `Category-style gaps were filled (mode, or "Unknown") in ${analysis.cleaning_report.categorical_columns_filled.length} column(s): ${analysis.cleaning_report.categorical_columns_filled.join(', ')}.`}
            </p>
            <p>
              <span className="font-medium text-slate-800">Categorical encoding.</span>{' '}
              {analysis.cleaning_report.binary_categorical_columns_encoded.length === 0
                ? 'No categorical columns had exactly two distinct values, so no binary encoding was applied.'
                : `${analysis.cleaning_report.binary_categorical_columns_encoded.length} binary categorical column(s) were encoded to 0/1: ${analysis.cleaning_report.binary_categorical_columns_encoded.join(', ')}.`}{' '}
              {analysis.cleaning_report.single_value_categorical_columns_dropped
                .length === 0
                ? 'No single-value categorical columns were dropped.'
                : `${analysis.cleaning_report.single_value_categorical_columns_dropped.length} categorical column(s) were dropped because they had only one distinct value: ${analysis.cleaning_report.single_value_categorical_columns_dropped.join(', ')}.`}
            </p>
            <p>
              <span className="font-medium text-slate-800">Feature relevance pruning.</span>{' '}
              {analysis.cleaning_report.categorical_columns_dropped_low_association
                .length === 0
                ? 'No categorical columns were dropped for low association with other features.'
                : `${analysis.cleaning_report.categorical_columns_dropped_low_association.length} categorical column(s) were dropped because they were not meaningfully related to most other features: ${analysis.cleaning_report.categorical_columns_dropped_low_association.join(', ')}.`}
            </p>
            <p>
              <span className="font-medium text-slate-800">Remaining gaps.</span>{' '}
              {analysis.cleaning_report.drop_remaining_null_rows_enabled ? (
                analysis.cleaning_report.rows_removed_unresolved_nulls === 0 ? (
                  <>After imputation, every cell had a value; no rows were dropped for leftover nulls.</>
                ) : (
                  <>
                    {analysis.cleaning_report.rows_removed_unresolved_nulls} row(s) were
                    dropped because at least one cell was still null (for example an
                    unparsed date).
                  </>
                )
              ) : (
                'Dropping rows for leftover nulls was disabled on the server for this run.'
              )}
            </p>
          </div>
        ) : (
          <div className="space-y-2 text-sm text-gray-600">
            <p>
              This analysis was stored before per-dataset cleaning statistics were
              added. Upload the file again to see a step-by-step report for this
              dataset.
            </p>
            <p className="text-xs text-gray-500">
              The pipeline still applies: drop sparse columns, fix types, optional IQR
              and sign row removal, parse dates and drop duplicates, impute, then drop
              rows with any remaining nulls (when enabled on the server).
            </p>
          </div>
        )}
        <p className="text-xs text-gray-500 mt-4">
          Summary row counts above match the cleaned table. Re-upload after changing
          the file to refresh this report.
        </p>
      </div>

      {/* Summary Stats */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Summary Statistics</h2>
        <div className="grid grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-3xl font-bold text-blue-600">
              {analysis.summary_stats.missing_cells}
            </div>
            <div className="text-sm text-blue-700 mt-1">Missing cells</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-3xl font-bold text-green-600">
              {analysis.summary_stats.numeric_columns}
            </div>
            <div className="text-sm text-green-700 mt-1">Numeric columns</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-3xl font-bold text-purple-600">
              {analysis.summary_stats.categorical_columns}
            </div>
            <div className="text-sm text-purple-700 mt-1">Categorical columns</div>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-3xl font-bold text-orange-600">
              {analysis.summary_stats.columns.length}
            </div>
            <div className="text-sm text-orange-700 mt-1">Total columns</div>
          </div>
        </div>
      </div>

      {/* Column Metadata */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Column Details</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-gray-700">
                  Column
                </th>
                <th className="px-4 py-3 text-left font-semibold text-gray-700">
                  Type
                </th>
                <th className="px-4 py-3 text-left font-semibold text-gray-700">
                  Null Count
                </th>
                <th className="px-4 py-3 text-left font-semibold text-gray-700">
                  Unique
                </th>
                <th className="px-4 py-3 text-left font-semibold text-gray-700">
                  Range
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {analysis.summary_stats.columns.map((col, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-gray-900">
                    {col.name}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {col.inferred_type}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {col.null_count}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {col.unique_count}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {col.min_max
                      ? `${col.min_max.min.toLocaleString()} - ${col.min_max.max.toLocaleString()}`
                      : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <ExploratoryAnalysis analysis={analysis} />

      {/* Correlations */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">
          Correlation Analysis
        </h2>
        <div className="space-y-4">
          {analysis.correlations.significant.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {analysis.correlations.significant.slice(0, 6).map(
                (corr, idx) => (
                  <div
                    key={idx}
                    className="p-4 border border-gray-200 rounded-lg bg-gray-50"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 truncate flex-1 mr-2">
                        {corr.column1}
                      </span>
                      <span
                        className={`
                          px-3 py-1 rounded-full text-xs font-medium
                          ${Math.abs(corr.correlation) > 0.8
                            ? 'bg-red-100 text-red-700'
                            : Math.abs(corr.correlation) > 0.6
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-blue-100 text-blue-700'
                          }
                        `}
                      >
                        {corr.correlation > 0 ? '+' : ''}{corr.correlation.toFixed(2)}
                      </span>
                      <span className="text-sm text-gray-500 ml-2 truncate">
                        {corr.column2}
                      </span>
                    </div>
                  </div>
                )
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500 text-center py-8">
              No significant correlations found
            </p>
          )}

          {analysis.correlations.insights.correlated_pairs.length > 0 && (
            <div className="border-t border-gray-100 pt-5 mt-2 space-y-5">
              <h3 className="text-sm font-semibold text-gray-800">
                Correlation Insights
              </h3>
              <p className="text-xs text-gray-500">
                Based on Pearson linear correlations. Suggestions favor columns that
                link into the same correlation network and avoid mixing columns that
                sit in separate weakly linked groups. Correlation is not causation—
                validate with domain knowledge and proper modeling.
              </p>
              {analysis.correlations.insights.correlated_pairs.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">
                    Related columns
                  </h4>
                  <ul className="space-y-2">
                    {analysis.correlations.insights.correlated_pairs.map(
                      (line, idx) => (
                        <li
                          key={idx}
                          className="flex items-start text-sm text-gray-600"
                        >
                          <svg
                            className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                          {line}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Insights */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">AI Insights</h2>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Summary
            </h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              {analysis.insights.summary}
            </p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Key Findings
            </h3>
            <ul className="space-y-2">
              {analysis.insights.key_findings.map((finding, idx) => (
                <li
                  key={idx}
                  className="flex items-start text-sm text-gray-600"
                >
                  <svg
                    className="w-5 h-5 text-green-500 mr-2 flex-shrink-0 mt-0.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  {finding}
                </li>
              ))}
            </ul>
          </div>

        </div>
      </div>

      {/* Data Table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-gray-800">
            Cleaned dataset preview
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Sample of cleaned rows. Use{' '}
            <span className="font-medium text-gray-700">Download cleaned CSV</span>{' '}
            at the top for the full file.
          </p>
        </div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">Preview</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                {analysis.summary_stats.columns.map((col, idx) => (
                  <th
                    key={idx}
                    className="px-4 py-3 text-left font-semibold text-gray-700 border-b border-gray-200"
                  >
                    {col.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {analysis.cleaned_data.slice(0, 10).map((row, rowIndex) => (
                <tr
                  key={rowIndex}
                  className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                >
                  {row.map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="px-4 py-3 text-sm text-gray-600 border-b border-gray-50"
                    >
                      {cell || <span className="text-gray-300 italic">NULL</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Showing first 10 rows
        </p>
      </div>
    </div>
  );
}
