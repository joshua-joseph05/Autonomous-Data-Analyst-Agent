'use client';

import { useState } from 'react';
import { uploadFile, FilePreview, ColumnMetadata } from '@/lib/api';

interface FileUploaderProps {
  onFileUploaded: (data: FilePreview) => void;
  onFileError: (error: string) => void;
}

export default function FileUploader({
  onFileUploaded,
  onFileError,
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [previewData, setPreviewData] = useState<FilePreview | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    setIsUploading(true);

    try {
      const data = await uploadFile(file);
      setPreviewData(data);
      onFileUploaded(data);
    } catch (error: any) {
      const detail = error?.response?.data?.detail;
      const detailStr = Array.isArray(detail)
        ? detail.map((d: { msg?: string }) => d?.msg || JSON.stringify(d)).join('; ')
        : typeof detail === 'string'
          ? detail
          : error?.response?.data?.message;
      onFileError(detailStr || error?.message || 'Failed to upload file');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center
          transition-all duration-200 cursor-pointer
          ${isDragging
            ? 'border-blue-500 bg-blue-50'
            : previewData
            ? 'border-green-500 bg-green-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
        `}
      >
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="cursor-pointer"
        >
          <div className="flex flex-col items-center">
            <svg
              className="w-12 h-12 text-gray-400 mb-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9m0 0h1m-3 2h4a4 4 0 010 8h-1"
              />
            </svg>

            <div className="text-gray-600 font-medium mb-2">
              {isUploading
                ? 'Uploading...'
                : previewData
                ? 'File uploaded successfully!'
                : 'Drag and drop your CSV file here'}
            </div>

            <p className="text-sm text-gray-500 mb-4">
              or
            </p>

            <div
              className={`
                px-6 py-2 rounded-full font-medium text-sm transition-colors
                ${previewData
                  ? 'bg-green-100 text-green-700'
                  : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                }
              `}
            >
              Browse files
            </div>

            <p className="text-xs text-gray-400 mt-4">
              Supports CSV files up to 10MB
            </p>
          </div>
        </label>
      </div>

      {previewData && (
        <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-800">
              Preview: {previewData.filename}
            </h3>
            <span className="text-sm text-gray-500">
              {previewData.rows_count.toLocaleString()} rows
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr>
                  {previewData.columns.map((col, idx) => (
                    <th
                      key={idx}
                      className="px-4 py-2 text-left font-medium text-gray-700 border-b border-gray-200"
                      style={{ minWidth: '120px' }}
                    >
                      <div className="flex flex-col">
                        <span>{col.name}</span>
                        <span className="text-xs text-gray-400 uppercase">
                          {col.inferred_type}
                        </span>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(previewData.preview_data || []).slice(0, 5).map((row, rowIndex) => (
                  <tr
                    key={rowIndex}
                    className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                  >
                    {row.map((cell, cellIndex) => (
                      <td
                        key={cellIndex}
                        className="px-4 py-2 text-sm text-gray-600 border-b border-gray-100 truncate max-w-[150px]"
                        title={cell}
                      >
                        {cell || <span className="text-gray-300 italic">NULL</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 text-xs text-gray-500">
            Showing first{' '}
            {Math.min(5, (previewData.preview_data || []).length)} of{' '}
            {previewData.rows_count.toLocaleString()} rows
          </div>
        </div>
      )}
    </div>
  );
}
