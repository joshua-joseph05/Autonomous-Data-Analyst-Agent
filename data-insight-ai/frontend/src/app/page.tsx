'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import FileUploader from '@/components/FileUploader';
import type { FilePreview } from '@/lib/api';

export default function HomePage() {
  const router = useRouter();
  const [uploadError, setUploadError] = useState<string | null>(null);

  const onUploaded = (data: FilePreview) => {
    setUploadError(null);
    router.push(`/dashboard/${encodeURIComponent(data.file_id)}`);
  };

  return (
    <main className="mx-auto max-w-4xl px-4 py-12">
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900">
          Autonomous Data Analyst Agent
        </h1>
        <p className="mt-2 text-gray-600">
          Upload a CSV dataset and get AI insights, correlations, and more
        </p>
      </div>
      {uploadError && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {uploadError}
        </div>
      )}
      <FileUploader
        onFileUploaded={onUploaded}
        onFileError={(msg) => setUploadError(msg)}
      />
    </main>
  );
}
