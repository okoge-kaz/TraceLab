import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

// Build-time loader for the card headline numbers. build-payload.mjs aggregates each table
// experiment's headline.json into public/data/headlines.json, keyed by experiment slug. The
// Overview gallery's stat cards render these few numbers; clicking through shows the full table.
// Resolved from process.cwd() (web/app during astro dev/build), mirroring lib/summary.ts.

export interface HeadlineStat {
  label: string;
  value: string;
}

export type Headlines = Record<string, HeadlineStat[]>;

function resolveHeadlinesPath(): string | null {
  const candidates = [
    resolve(process.cwd(), 'public/data/headlines.json'), // build-payload output (cwd = web/app)
    resolve(process.cwd(), 'app/public/data/headlines.json'), // cwd = web
  ];
  return candidates.find((c) => existsSync(c)) ?? null;
}

let cached: Headlines | null = null;

export function loadHeadlines(): Headlines {
  if (cached) return cached;
  const path = resolveHeadlinesPath();
  cached = path ? (JSON.parse(readFileSync(path, 'utf-8')) as Headlines) : {};
  return cached;
}

/** Headline stat lines for an experiment slug (empty when none have been rendered yet). */
export function headlinesFor(slug: string | undefined): HeadlineStat[] {
  if (!slug) return [];
  return loadHeadlines()[slug] ?? [];
}
