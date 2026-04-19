'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import Dashboard from '@/components/Dashboard';

export default function DashboardPage() {
  const params = useParams();
  const fileId = typeof params.fileId === 'string' ? params.fileId : '';

  if (!fileId) {
    return (
      <div className="p-8 text-center text-gray-600">
        Missing file id.{' '}
        <Link href="/" className="text-blue-600 underline">
          Upload a file
        </Link>
      </div>
    );
  }

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <div className="mb-6">
        <Link
          href="/"
          className="text-sm font-medium text-blue-600 hover:text-blue-800"
        >
          ← Upload another file
        </Link>
      </div>
      <Dashboard fileId={fileId} />
    </main>
  );
}
