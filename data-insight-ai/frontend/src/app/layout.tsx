import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DataInsight AI',
  description: 'AI-assisted CSV exploration and analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
