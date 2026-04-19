import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const DEFAULT_PYTHON_API = 'http://127.0.0.1:8000';

/**
 * URL the Next.js server uses to reach FastAPI (must be the uvicorn root, no /api prefix).
 * - Prefer DATAINSIGHT_API_URL in Docker / split deployments.
 * - If NEXT_PUBLIC_API_URL points at this Next app (same origin), ignore it — that would 404.
 * - Otherwise use NEXT_PUBLIC_API_URL's origin (drops paths like /api on a wrong public URL).
 */
function backendBaseUrl(request: Request): string {
  const dedicated = process.env.DATAINSIGHT_API_URL?.trim();
  if (dedicated && /^https?:\/\//i.test(dedicated)) {
    return dedicated.replace(/\/+$/, '');
  }

  const self = new URL(request.url);
  const selfOrigin = `${self.protocol}//${self.host}`;

  const pub = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (pub && /^https?:\/\//i.test(pub)) {
    try {
      const u = new URL(pub);
      const pubOrigin = `${u.protocol}//${u.host}`;
      if (pubOrigin === selfOrigin) {
        return DEFAULT_PYTHON_API;
      }
      return u.origin.replace(/\/+$/, '');
    } catch {
      /* fall through */
    }
  }

  return DEFAULT_PYTHON_API;
}

export async function GET(
  request: Request,
  context: { params: { fileId: string } }
) {
  const fileId = context.params.fileId;
  if (
    !fileId ||
    fileId.includes('..') ||
    fileId.includes('/') ||
    fileId.includes('\\')
  ) {
    return NextResponse.json({ detail: 'Invalid file id' }, { status: 400 });
  }

  const base = backendBaseUrl(request);
  // Prefer short path — same handler as `/export/{id}/cleaned.csv` on FastAPI.
  const upstreamUrl = `${base}/export/${encodeURIComponent(fileId)}`;

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, { cache: 'no-store' });
  } catch {
    return NextResponse.json(
      {
        detail:
          'Could not reach the Python API. Start the backend (uvicorn on port 8000) or set DATAINSIGHT_API_URL / NEXT_PUBLIC_API_URL.',
      },
      { status: 502 }
    );
  }

  const body = await upstream.arrayBuffer();

  if (upstream.status === 404) {
    const snippet = new TextDecoder().decode(body.slice(0, 300)).trim();
    const isGenericFastApi =
      snippet.includes('Not Found') &&
      !snippet.includes('No stored copy') &&
      !snippet.includes('Analysis not found');
    if (isGenericFastApi) {
      return NextResponse.json(
        {
          detail:
            `The Python API at ${upstreamUrl} returned 404 (route not found). ` +
            `Start uvicorn from the backend folder, or set DATAINSIGHT_API_URL to the FastAPI root (example: http://localhost:8000).`,
        },
        { status: 502 }
      );
    }
  }

  const headers = new Headers();
  if (process.env.NODE_ENV === 'development') {
    headers.set('X-Debug-Upstream-Url', upstreamUrl);
  }
  const ct = upstream.headers.get('content-type');
  if (ct) headers.set('Content-Type', ct);
  const cd = upstream.headers.get('content-disposition');
  if (cd) headers.set('Content-Disposition', cd);

  return new NextResponse(body, { status: upstream.status, headers });
}
