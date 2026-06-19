import type { CompareData, CompareEntry, CompareGroup } from '../lib/compare';
import { copyFor, normalizeLocale, type Locale } from './index';

function tr(locale: Locale, text: string | undefined): string | undefined {
  if (!text || locale === 'en') return text;
  return copyFor(locale).compare.translations[text] ?? text;
}

function localizeEntry(locale: Locale, entry: CompareEntry): CompareEntry {
  if ('kind' in entry && entry.kind === 'subsection') {
    return { ...entry, title: tr(locale, entry.title) ?? entry.title, note: tr(locale, entry.note) };
  }
  return { ...entry, metric: tr(locale, entry.metric) ?? entry.metric };
}

function localizeGroup(locale: Locale, group: CompareGroup): CompareGroup {
  return {
    ...group,
    name: tr(locale, group.name) ?? group.name,
    note: tr(locale, group.note) ?? group.note,
    rows: group.rows.map((row) => localizeEntry(locale, row)),
  };
}

export function localizeCompareData(data: CompareData, rawLocale: unknown): CompareData {
  const locale = normalizeLocale(rawLocale);
  if (locale === 'en') return data;
  return {
    ...data,
    groups: data.groups.map((group) => localizeGroup(locale, group)),
  };
}
