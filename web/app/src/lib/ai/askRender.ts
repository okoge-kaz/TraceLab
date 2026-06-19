// Pure DOM builders for the Ask-the-trace assistant. No global state — every function takes its
// inputs and returns (or fills) an element. SVG icon markup is lifted verbatim from the mockup so
// the visuals match exactly. All user/model text is HTML-escaped before insertion.

import { marked } from 'marked';
import DOMPurify from 'dompurify';

import type { AskStoredImage, AskStoredMessage } from './askHistory';
import { browserCopy } from '../../i18n/browser';

/** Create an element with an optional class and innerHTML. */
function el(tag: string, cls?: string, html?: string): HTMLElement {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html != null) node.innerHTML = html;
  return node;
}

export function escapeHtml(s: unknown): string {
  return String(s ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string,
  );
}

// ---------------------------------------------------------------------------
// Markdown answer renderer
// The model returns CommonMark/GFM (headings, ordered + bullet lists, links, tables, fenced code).
// We render it with `marked`, then run the result through DOMPurify before it ever touches the DOM.
// The answer is untrusted model output, so no model-supplied HTML, scripts, or javascript: URLs are
// trusted — only the sanitized element subset survives. (The gallery README path renders markdown at
// build time over trusted repo files; this runs in the browser over model output, hence the sanitize.)
// ---------------------------------------------------------------------------

marked.setOptions({
  gfm: true, // GitHub-flavored: tables, strikethrough, autolinks
  breaks: true, // a single newline becomes <br>, matching the old formatter's feel
});

// Model-supplied links open in a new tab without leaking the opener. DOMPurify already strips
// dangerous URL schemes (javascript:, data:) and inline event handlers; this only hardens the
// rel/target on the anchors that survive sanitization.
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.nodeName === 'A' && node.getAttribute('href')) {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});

/** Render a model answer (CommonMark/GFM) to sanitized HTML for the chat bubble. */
export function formatAnswer(text: string): string {
  const raw = String(text ?? '').trim();
  if (!raw) return '<p></p>';
  const html = marked.parse(raw, { async: false }) as string;
  // Drop inline images: generated plots are delivered separately as `display_images` artifacts and
  // rendered as `.plot` cards (buildPlot). Any `![…](path)` the model writes points at a sandbox/out
  // path the browser can't load, so it would only ever render as a broken-image icon.
  const clean = DOMPurify.sanitize(html, { USE_PROFILES: { html: true }, FORBID_TAGS: ['img'] });
  // A lone image becomes an empty <p> once its <img> is stripped — drop those so there's no dead gap.
  return clean.replace(/<p>\s*<\/p>/g, '');
}

// ---------------------------------------------------------------------------
// Message builders
// ---------------------------------------------------------------------------

export function buildUserMessage(text: string): HTMLElement {
  const t = browserCopy().assistantRuntime;
  return el('div', 'msg user', `<div class="who">${escapeHtml(t.userLabel)}</div><div class="bubble">${escapeHtml(text)}</div>`);
}

const DL_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M7 11l5 5 5-5M5 21h14"/></svg>';
const CHEV_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>';

/** Trigger a browser download for a data URL (used by the plot caption's "Download PNG"). */
function downloadDataUrl(dataUrl: string, name: string): void {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** A `.plot` card wrapping an <img> from a generated artifact, with a downloadable caption. */
export function buildPlot(image: AskStoredImage, index: number): HTMLElement {
  const t = browserCopy().assistantRuntime;
  const caption = image.caption || t.generatedPlot;
  const plot = el('div', 'plot');
  plot.innerHTML =
    `<img src="${escapeHtml(image.dataUrl)}" alt="${escapeHtml(caption)}">` +
    `<div class="cap"><span>${escapeHtml(caption)}</span>` +
    `<span class="dl" role="button" tabindex="0">${DL_ICON}${escapeHtml(t.downloadPng)}</span></div>`;
  const dl = plot.querySelector<HTMLElement>('.dl');
  const fileName = caption.split('/').pop() || `plot-${index + 1}.png`;
  dl?.addEventListener('click', () => downloadDataUrl(image.dataUrl, fileName));
  return plot;
}

/** The folded "Show the code & output" block. `code` is escaped and shown verbatim. */
export function buildCodeFold(code: string): HTMLElement {
  const t = browserCopy().assistantRuntime;
  const fold = el('details', 'code-fold');
  fold.innerHTML =
    `<summary><span class="chev">${CHEV_ICON}</span>${escapeHtml(t.showCodeOutput)}</summary>` +
    `<pre>${escapeHtml(code)}</pre>`;
  return fold;
}

/** A full assistant turn: answer bubble, then any plots, then the code fold. */
export function buildBotMessage(message: AskStoredMessage): HTMLElement {
  const t = browserCopy().assistantRuntime;
  const node = el('div', 'msg bot');
  node.appendChild(el('div', 'who', escapeHtml(t.assistantLabel)));
  node.appendChild(el('div', 'bubble', formatAnswer(message.text)));
  (message.images ?? []).forEach((img, i) => {
    const spacer = el('div');
    spacer.style.height = '12px';
    node.appendChild(spacer);
    node.appendChild(buildPlot(img, i));
  });
  if (message.code) {
    const spacer = el('div');
    spacer.style.height = '12px';
    node.appendChild(spacer);
    node.appendChild(buildCodeFold(message.code));
  }
  return node;
}

// ---------------------------------------------------------------------------
// Live progress — an append-only checklist driven by the real streamed events.
// Each line is keyed so an event can either create a step or update one already shown. Only the
// latest line spins; starting a new step settles the previous active one to a ✓. `note()` drops a
// standalone ⚠ line (e.g. a provider retry) above the active step without disturbing it. The caller
// (askTrace) owns the event→step mapping; this class only knows how to draw and settle lines.
// ---------------------------------------------------------------------------

const TICK_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';
const WARN_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v5M12 16.5h.01"/></svg>';

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0.0s';
  if (ms < 1000) return `${Math.max(0.1, ms / 1000).toFixed(1)}s`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

export class LiveProgress {
  readonly element: HTMLElement;
  private readonly lines = new Map<string, HTMLElement>();
  private readonly labels = new Map<string, string>();
  private readonly starts = new Map<string, number>();
  private readonly finished = new Set<string>();
  private current: HTMLElement | null = null;
  private currentKey: string | null = null;

  constructor() {
    this.element = el('div', 'working');
  }

  /** Start (or update) a keyed step as the active, spinning line. Settles the previous active step. */
  begin(key: string, text: string): void {
    const existing = this.lines.get(key);
    if (existing) {
      this.labels.set(key, text);
      this.setText(existing, text);
      return;
    }
    if (this.current && this.currentKey) {
      const label = this.labels.get(this.currentKey) ?? '';
      this.setText(this.current, this.withDuration(this.currentKey, label));
      this.finished.add(this.currentKey);
      this.mark(this.current, 'done');
    }
    const line = this.lineEl('active', TICK_ICON, text);
    this.lines.set(key, line);
    this.labels.set(key, text);
    this.starts.set(key, performance.now());
    this.finished.delete(key);
    this.element.appendChild(line);
    this.current = line;
    this.currentKey = key;
  }

  /** Mark a keyed step done (✓), creating it if it never had an active phase. */
  done(key: string, text?: string): void {
    const existing = this.lines.get(key);
    const label = text ?? this.labels.get(key) ?? '';
    if (existing) {
      this.labels.set(key, label);
      this.setText(existing, this.withDuration(key, label));
      this.mark(existing, 'done');
      this.finished.add(key);
      if (this.current === existing) {
        this.current = null;
        this.currentKey = null;
      }
      return;
    }
    if (this.current) {
      if (this.currentKey) {
        const currentLabel = this.labels.get(this.currentKey) ?? '';
        this.setText(this.current, this.withDuration(this.currentKey, currentLabel));
        this.finished.add(this.currentKey);
      }
      this.mark(this.current, 'done');
      this.current = null;
      this.currentKey = null;
    }
    this.labels.set(key, label);
    this.starts.set(key, performance.now());
    this.finished.add(key);
    const line = this.lineEl('done', TICK_ICON, label);
    this.lines.set(key, line);
    this.element.appendChild(line);
  }

  /** Drop a standalone ⚠ note above the active step. Does not disturb the running step. */
  note(text: string): void {
    const line = this.lineEl('warn', WARN_ICON, text);
    if (this.current) this.element.insertBefore(line, this.current);
    else this.element.appendChild(line);
  }

  /** Mark every step done — used when the turn finishes successfully. */
  complete(): void {
    this.lines.forEach((line, key) => {
      const label = this.labels.get(key) ?? '';
      this.setText(line, this.withDuration(key, label));
      this.finished.add(key);
      this.mark(line, 'done');
    });
    this.current = null;
    this.currentKey = null;
  }

  remove(): void {
    this.element.remove();
  }

  private lineEl(state: 'active' | 'done' | 'warn', icon: string, text: string): HTMLElement {
    const line = el('div', `work-step ${state}`, `<span class="tick">${icon}</span><span class="ws-label"></span>`);
    this.setText(line, text);
    return line;
  }

  private setText(line: HTMLElement, text: string): void {
    const label = line.querySelector<HTMLElement>('.ws-label');
    if (label) label.textContent = text; // event-derived text → textContent, never innerHTML
  }

  private withDuration(key: string, text: string): string {
    if (this.finished.has(key)) return text;
    const started = this.starts.get(key);
    if (started == null) return text;
    return `${text} · ${formatDuration(performance.now() - started)}`;
  }

  private mark(line: HTMLElement, state: 'done' | 'warn'): void {
    line.classList.remove('active');
    line.classList.add(state);
  }
}

// ---------------------------------------------------------------------------
// Contribute nudge — one-time, dismissible prompt after the first answer
// ---------------------------------------------------------------------------

export interface ContributeNudgeHandlers {
  /** User accepted. `hasTrace` distinguishes "contribute now" vs "go analyze a trace first". */
  onAccept: () => void;
}

const HEART_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6.5l-1.1-1.1a4 4 0 0 0-5.7 5.6L12 18l6.8-7a4 4 0 0 0-5.7-5.6L12 6.5z"/></svg>';
const CHECK_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';
const X_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>';

/** Build the contribute nudge card. `hasTrace` tunes the copy + the accept behavior. */
export function buildContributeNudge(hasTrace: boolean, handlers: ContributeNudgeHandlers): HTMLElement {
  const t = browserCopy().assistantRuntime.nudge;
  const card = el('div', 'contrib-nudge');
  card.innerHTML =
    `<span class="cn-ico">${HEART_ICON}</span>` +
    '<div class="cn-body">' +
    `<div class="cn-title">${escapeHtml(hasTrace ? t.traceTitle : t.publicTitle)}</div>` +
    `<div class="cn-sub">${escapeHtml(hasTrace ? t.traceBody : t.publicBody)}</div>` +
    '<div class="cn-actions">' +
    `<button class="btn btn-primary cn-yes">${escapeHtml(hasTrace ? t.traceCta : t.publicCta)}</button>` +
    `<button class="btn btn-ghost cn-no">${escapeHtml(t.later)}</button>` +
    '</div></div>' +
    `<button class="cn-x" aria-label="${escapeHtml(t.dismiss)}">${X_ICON}</button>`;

  const dismiss = (): void => {
    card.classList.add('out');
    window.setTimeout(() => card.remove(), 280);
  };
  card.querySelector('.cn-no')?.addEventListener('click', dismiss);
  card.querySelector('.cn-x')?.addEventListener('click', dismiss);
  card.querySelector('.cn-yes')?.addEventListener('click', () => {
    if (hasTrace) {
      card.innerHTML =
        `<span class="cn-ico">${CHECK_ICON}</span>` +
        `<div class="cn-body"><div class="cn-title">${escapeHtml(t.thanksTitle)}</div>` +
        `<div class="cn-sub">${escapeHtml(t.thanksBody)}</div></div>`;
      window.setTimeout(dismiss, 3200);
    }
    handlers.onAccept();
  });
  return card;
}

// ---------------------------------------------------------------------------
// Empty state — greeting + example chips, and the two "Your trace" gates
// ---------------------------------------------------------------------------

const CHIP_ICONS: Record<string, string> = {
  tool: '<path d="M14.7 6.3a4 4 0 0 0-5 5l-6 6 2 2 6-6a4 4 0 0 0 5-5l-2.5 2.5-2-2 2.5-2.5z"/>',
  cache:
    '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  time: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  token: '<path d="M12 2l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/><path d="M3 17l9 5 9-5"/>',
};

/** The greeting + clickable example chips. `onPick` receives the chip's question text. */
export function buildEmptyState(source: 'public' | 'user', onPick: (question: string) => void): HTMLElement {
  const t = browserCopy().assistantRuntime;
  const empty = t.empty;
  const hello =
    source === 'public'
      ? `<div class="hello">${empty.publicHello}</div><p class="blurb">${escapeHtml(empty.publicBlurb)}</p>`
      : `<div class="hello">${empty.userHello}</div><p class="blurb">${escapeHtml(empty.userBlurb)}</p>`;

  const chips = empty.examples[source]
    .map(
      (c) =>
        `<button class="chip" data-q="${escapeHtml(c.text)}"><span class="ci">` +
        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${
          CHIP_ICONS[(c.icon as keyof typeof CHIP_ICONS)] ?? CHIP_ICONS.tool
        }</svg>` +
        `</span>${escapeHtml(c.text)}</button>`,
    )
    .join('');

  const wrap = el('div', 'empty', `${hello}<div class="chips-label">${escapeHtml(t.tryOne)}</div><div class="chips">${chips}</div>`);
  wrap.querySelectorAll<HTMLButtonElement>('.chip').forEach((chip) => {
    chip.addEventListener('click', () => onPick(chip.dataset.q || ''));
  });
  return wrap;
}

export interface NoTraceGateHandlers {
  onGoAnalyze: () => void;
  onUsePublic: () => void;
}

/** Gate shown when "Your trace" is selected but no trace has been analyzed yet. */
export function buildNoTraceGate(handlers: NoTraceGateHandlers): HTMLElement {
  const t = browserCopy().assistantRuntime.noTrace;
  const gate = el('div', 'gate no-trace');
  gate.innerHTML =
    '<div class="g-ico"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg></div>' +
    `<h3>${t.title}</h3>` +
    `<p>${t.body}</p>` +
    '<div class="gate-actions">' +
    `<button class="btn btn-primary" data-act="analyze">${escapeHtml(t.analyze)}</button>` +
    `<button class="btn btn-ghost" data-act="public">${escapeHtml(t.public)}</button>` +
    '</div>';
  gate.querySelector('[data-act="analyze"]')?.addEventListener('click', handlers.onGoAnalyze);
  gate.querySelector('[data-act="public"]')?.addEventListener('click', handlers.onUsePublic);
  return gate;
}

export interface ConsentGateHandlers {
  onConsent: () => void;
  onUsePublic: () => void;
}

/**
 * Consent gate before any cloud LLM call over the user's trace. The copy reflects the real flow:
 * the model's generated code runs locally in the browser; only small aggregated results (never raw
 * rows) are sent to the model.
 */
export function buildConsentGate(handlers: ConsentGateHandlers): HTMLElement {
  const t = browserCopy().assistantRuntime.consent;
  const li = (text: string): string =>
    '<li><span class="li-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg></span>' +
    `<span>${text}</span></li>`;

  const gate = el('div', 'gate');
  gate.innerHTML =
    '<div class="g-ico"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 4v5c0 4.4-3.1 7.9-8 9-4.9-1.1-8-4.6-8-9V7l8-4z"/><path d="M9 12l2 2 4-4"/></svg></div>' +
    `<h3>${t.title}</h3>` +
    `<p>${t.body}</p>` +
    '<ul>' +
    t.bullets.map(li).join('') +
    '</ul>' +
    `<label class="consent"><input type="checkbox" data-role="consent"><span>${escapeHtml(t.checkbox)}</span></label>` +
    '<div class="gate-actions">' +
    `<button class="btn btn-primary" data-act="enable" disabled>${escapeHtml(t.enable)}</button>` +
    `<button class="btn btn-ghost" data-act="public">${escapeHtml(t.public)}</button>` +
    '</div>' +
    `<p class="no-trace-hint">${t.hint}</p>`;

  const box = gate.querySelector<HTMLInputElement>('[data-role="consent"]');
  const enable = gate.querySelector<HTMLButtonElement>('[data-act="enable"]');
  box?.addEventListener('change', () => {
    if (enable) enable.disabled = !box.checked;
  });
  enable?.addEventListener('click', handlers.onConsent);
  gate.querySelector('[data-act="public"]')?.addEventListener('click', handlers.onUsePublic);
  return gate;
}
