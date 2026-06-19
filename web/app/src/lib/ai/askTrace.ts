// Controller for the Ask-the-trace assistant. Wires the markup from AskTrace.astro to the real
// WebSocket client (askClient.askTurn), drives the stepped "working" strip from streamed events,
// renders answers/plots/code via askRender, and persists conversations via askHistory.
//
// This replaces the mockup's monolithic IIFE: behavior is the same (open/close/expand dock, source
// switch with "Your trace" lock + consent gate, history drawer, contribute nudge) but the canned
// response engine is gone — scope and answers are the model's job now.

import { askTurn, newSessionId, type AskEvent, type AskMessage, type AskResult, type AskSource } from './askClient';
import {
  AskHistory,
  titleFromMessage,
  type AskHistorySource,
  type AskStoredImage,
  type AskStoredMessage,
  type Conversation,
} from './askHistory';
import {
  LiveProgress,
  buildBotMessage,
  buildConsentGate,
  buildContributeNudge,
  buildEmptyState,
  buildNoTraceGate,
  buildUserMessage,
  escapeHtml,
} from './askRender';
import { PyodideToolExecutorClient, type TraceReady } from './pyodideToolExecutor';
import {
  ASSISTANT_PREWARM_EVENT,
  TRACE_CLEARED_EVENT,
  TRACE_CONTRIBUTED_EVENT,
  TRACE_READY_EVENT,
  getAnalyzedTrace,
  traceFileId,
  type TraceContributedDetail,
  type TraceReadyDetail,
} from './traceStore';
import type { PyodideArtifact, PyodideToolResult } from '../worker/protocol';
import { browserCopy, template } from '../../i18n/browser';

/** The shape shared by a server tool-result summary and a browser PyodideToolResult. */
interface RunResultLike {
  stdout?: string[];
  error?: unknown;
  display_images?: unknown[];
  artifacts?: Array<{ is_image?: boolean }>;
}

/** If the last stdout line is a JSON array (the prompt tells the model to print one), its length. */
function rowsFromStdout(stdout?: string[]): number | null {
  if (!Array.isArray(stdout)) return null;
  for (let i = stdout.length - 1; i >= 0; i--) {
    const line = (stdout[i] || '').trim();
    if (line[0] !== '[') continue;
    try {
      const parsed = JSON.parse(line);
      if (Array.isArray(parsed)) return parsed.length;
    } catch {
      /* not JSON — keep scanning earlier lines */
    }
  }
  return null;
}

/** A real, honest one-liner for a finished query, derived from the actual tool output. */
function assistantCopy() {
  return browserCopy().assistantRuntime;
}

function summarizeRun(result: RunResultLike | null | undefined, scope: string): string {
  const t = assistantCopy().progress;
  if (!result) return template(t.ranQuery, { scope });
  if (result.error) return t.queryError;
  const rows = rowsFromStdout(result.stdout);
  if (rows != null) {
    return template(t.ranRows, {
      scope,
      rows: rows.toLocaleString(),
      rowWord: rows === 1 ? t.row : t.rows,
    });
  }
  const images = result.display_images?.length || (result.artifacts || []).filter((a) => a?.is_image).length;
  if (images) return template(t.chart, { scope });
  return template(t.ranQuery, { scope });
}

/**
 * Translates the real streamed events (and browser tool callbacks) into LiveProgress steps, one
 * instance per turn. Every line reflects something that actually happened — sandbox warm/cold, each
 * model↔tool loop turn, the query result, provider retries — instead of a fixed four-step script.
 */
class TurnProgress {
  private readonly t = assistantCopy().progress;
  private turn = 0;
  // Key of the current "soft" spinner (the model-is-thinking line: 'ask' or `compose-N`). A hard
  // spinner ('sandbox'/'prep'/`run-N`) isn't tracked here — begin()/done() of the next line settles
  // it. Tracking the soft one lets us relabel it correctly when the wait resolves into a query vs.
  // the final answer, and keep *something* spinning during every model call (no dead gaps).
  private wait = '';

  constructor(
    private readonly prog: LiveProgress,
    private readonly source: AskSource,
  ) {}

  /** First line, shown as soon as the turn starts (before any event arrives). */
  start(): void {
    if (this.source === 'user') this.prog.begin('prep', this.t.preparingTrace);
    else this.prog.begin('sandbox', this.t.preparingSandbox);
  }

  /** User source: the in-browser executor finished loading the trace; the model call begins now. */
  traceReady(): void {
    this.prog.done('prep', this.t.traceReady);
    this.wait = 'ask';
    this.prog.begin('ask', this.t.readingQuestion);
  }

  /** A server `event` frame. */
  handle(event: AskEvent): void {
    const v = (event.value ?? {}) as Record<string, unknown>;
    switch (event.label) {
      case 'e2b':
        this.sandbox(String(v.status ?? ''));
        break;
      case 'openrouter_retry':
        if (v.status === 'retrying') {
          const attempt = Number(v.attempt) || 1;
          const tries = (Number(v.max_retries) || 0) + 1;
          this.prog.note(template(this.t.providerRetry, { attempt, tries }));
        }
        break;
      case 'generation_retry':
        if (v.status === 'retrying') this.prog.note(this.t.generationRetry);
        break;
      case 'model_turn':
        this.modelTurn(v);
        break;
      case 'tool_result':
        // The browser source drives its run line from toolStart/toolEnd; ignore the server echo.
        if (this.source !== 'user') {
          this.prog.done(`run-${this.turn || 1}`, summarizeRun(v.summary as RunResultLike, this.t.poolScope));
          this.openCompose(); // keep a spinner up while the model reads results / composes
        }
        break;
      case 'final':
        this.openCompose();
        break;
    }
  }

  /** Browser tool execution started (user source). */
  toolStart(): void {
    this.prog.begin(`run-${this.turn || 1}`, this.t.runningTrace);
  }

  /** Browser tool execution finished (user source). */
  toolEnd(result: PyodideToolResult): void {
    this.prog.done(`run-${this.turn || 1}`, summarizeRun(result as RunResultLike, this.t.traceScope));
    this.openCompose();
  }

  private sandbox(status: string): void {
    if (!status) return;
    if (status === 'sandbox_prepared') return;
    if (status === 'sandbox_expired') {
      this.prog.note(this.t.sandboxExpired);
      return;
    }
    if (status.startsWith('stale_sandbox_') || status.startsWith('sandbox_timeout_refresh_')) return;
    const warm = status.startsWith('reuse') || status === 'using_sandbox' || status === 'sandbox_ready';
    if (status === 'reuse_sandbox') {
      this.prog.done('sandbox', this.t.reusedSandbox);
      this.openQuestionWait();
      return;
    }
    if (status === 'sandbox_ready') {
      this.prog.done('sandbox', warm ? this.t.reusedSandbox : this.t.sandboxReady);
      this.openQuestionWait();
    } else {
      this.prog.begin('sandbox', warm ? this.t.reusingSandbox : this.t.startingSandbox);
    }
  }

  private openQuestionWait(): void {
    if (this.turn !== 0 || this.wait) return;
    this.wait = 'ask';
    this.prog.begin('ask', this.t.readingQuestion);
  }

  private modelTurn(v: Record<string, unknown>): void {
    const toolCalls = Array.isArray(v.tool_calls) ? v.tool_calls : [];
    if (!toolCalls.length) {
      // No query this turn → the model is composing the answer.
      this.openCompose();
      return;
    }
    // A query: settle whatever we were waiting on (the model just produced it), then record it.
    this.settleWait();
    this.turn = Number(v.turn) || this.turn + 1;
    this.prog.done(
      `query-${this.turn}`,
      template(this.turn === 1 ? this.t.wroteQuery : this.t.refinedQuery, { turn: this.turn }),
    );
    // Public path: the query now runs server-side. (Browser runs open their own line via toolStart.)
    if (this.source !== 'user') this.prog.begin(`run-${this.turn}`, this.t.runningPool);
  }

  /** Open the "Composing the answer…" spinner once per wait; it persists until the strip is removed. */
  private openCompose(): void {
    const key = `compose-${this.turn}`;
    if (this.wait === key) return;
    this.settleWait();
    this.wait = key;
    this.prog.begin(key, this.t.composing);
  }

  /** Settle the current soft spinner: the model finished thinking and either queried or answered. */
  private settleWait(): void {
    if (!this.wait) return;
    this.prog.done(this.wait, this.turn === 0 ? this.t.readQuestion : this.t.reviewedResults);
    this.wait = '';
  }
}

interface ControllerState {
  source: AskHistorySource;
  traceAvailable: boolean;
  userConsented: boolean;
  busy: boolean;
  publicContributeNudgeShown: boolean;
  contributeNudgedTraceId: string | null;
  activeId: string | null;
}

/** Map the UI source ('public'|'user') to the client's source value ('syfi'|'user'). */
function clientSource(source: AskHistorySource): AskSource {
  return source === 'public' ? 'syfi' : 'user';
}

class AskTraceController {
  private readonly assistant: HTMLElement;
  private readonly body: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly send: HTMLButtonElement;
  private readonly seg: HTMLElement;
  private readonly composer: HTMLElement;
  private readonly headSub: HTMLElement;
  private readonly srcCapText: HTMLElement;
  private readonly footNote: HTMLElement;
  private readonly histList: HTMLElement;
  private readonly btnHistory: HTMLButtonElement;

  private readonly history = new AskHistory();
  private readonly state: ControllerState = {
    source: 'public',
    traceAvailable: false,
    userConsented: false,
    busy: false,
    publicContributeNudgeShown: false,
    contributeNudgedTraceId: null,
    activeId: null,
  };

  private readonly sessionId = newSessionId();
  private abort: AbortController | null = null;

  // Lazily-created Pyodide executor for the user source, plus its loaded-trace handle.
  private traceFile: File | null = null;
  private traceId: string | null = null;
  private traceContributed = false;
  private executor: PyodideToolExecutorClient | null = null;
  private traceReady: TraceReady | null = null;
  private executorLoad: Promise<TraceReady> | null = null;

  constructor(root: { [k: string]: HTMLElement }) {
    this.assistant = root.assistant;
    this.body = root.body;
    this.input = root.input as HTMLTextAreaElement;
    this.send = root.send as HTMLButtonElement;
    this.seg = root.seg;
    this.composer = root.composer;
    this.headSub = root.headSub;
    this.srcCapText = root.srcCapText;
    this.footNote = root.footNote;
    this.histList = root.histList;
    this.btnHistory = root.btnHistory as HTMLButtonElement;
  }

  // ---- lifecycle ----------------------------------------------------------

  mount(): void {
    this.bindControls();
    this.bindTraceEvents();

    // A trace may already be ready if the user analyzed before opening the assistant.
    const existing = getAnalyzedTrace();
    if (existing) {
      this.traceFile = existing.file;
      this.traceId = existing.traceId;
      this.traceContributed = existing.contributed;
      this.state.traceAvailable = true;
    }

    this.history.create('public');
    this.state.activeId = this.history.list()[0].id;
    this.updateSourceUI();
    this.renderHistory();
    this.renderThread();
  }

  private bindControls(): void {
    document.getElementById('launcher')?.addEventListener('click', () => this.open());
    // The in-page teaser lives in a surface, not the dock; open if something clicks it.
    document.getElementById('askTeaser')?.addEventListener('click', () => this.open('public'));

    document.getElementById('btnClose')?.addEventListener('click', () => this.close());
    document.getElementById('btnExpand')?.addEventListener('click', () => {
      document.body.classList.add('assistant-expanded');
      this.setHistory(true);
    });
    document.getElementById('btnCollapse')?.addEventListener('click', () => {
      document.body.classList.remove('assistant-expanded');
      this.setHistory(false);
    });
    document.getElementById('scrim')?.addEventListener('click', () => {
      document.body.classList.remove('assistant-expanded');
      this.setHistory(false);
    });
    this.btnHistory.addEventListener('click', () => this.toggleHistory());
    document.getElementById('histScrim')?.addEventListener('click', () => this.setHistory(false));
    document.getElementById('btnNew')?.addEventListener('click', () => this.startNewConversation());
    document.getElementById('histNew')?.addEventListener('click', () => this.startNewConversation());

    this.seg.querySelectorAll<HTMLButtonElement>('button').forEach((b) =>
      b.addEventListener('click', () => this.selectSource((b.dataset.src as AskHistorySource) || 'public')),
    );

    // While a turn is in flight the send button is a Stop button (interrupt); otherwise it sends.
    this.send.addEventListener('click', () => {
      if (this.state.busy) this.interrupt();
      else void this.ask();
    });
    this.input.addEventListener('input', () => {
      this.autoGrow();
      this.syncSend();
    });
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        void this.ask(); // ask() no-ops while busy; Stop is explicit (button / Esc)
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      // A turn is streaming → Esc interrupts it first (a second Esc then closes the panel).
      if (this.state.busy && document.body.classList.contains('assistant-open')) {
        this.interrupt();
        return;
      }
      if (this.assistant.classList.contains('hist-open') && !this.isExpanded()) this.setHistory(false);
      else if (this.isExpanded()) {
        document.body.classList.remove('assistant-expanded');
        this.setHistory(false);
      } else if (document.body.classList.contains('assistant-open')) this.close();
    });
  }

  private bindTraceEvents(): void {
    // Local-executor analysis just started — boot the executor (incl. matplotlib) in the background so
    // it overlaps the server-side sanitize+compute and is warm before the user ever opens "Your trace".
    window.addEventListener(ASSISTANT_PREWARM_EVENT, () => this.prewarmBoot());
    window.addEventListener(TRACE_READY_EVENT, (e: Event) => {
      const detail = (e as CustomEvent<TraceReadyDetail>).detail;
      if (!detail?.file) return;
      const newTrace = detail.traceId !== this.traceId;
      this.traceFile = detail.file;
      this.traceId = detail.traceId;
      this.traceContributed = detail.contributed;
      if (newTrace) {
        // A new trace replaced a previously-loaded one — drop the stale executor so the next
        // user-source question reloads against the new file.
        if (this.traceReady || this.executorLoad) this.resetExecutor();
      }
      this.markTraceReady();
    });
    window.addEventListener(TRACE_CONTRIBUTED_EVENT, (e: Event) => {
      const detail = (e as CustomEvent<TraceContributedDetail>).detail;
      if (!detail?.traceId || detail.traceId !== this.traceId) return;
      this.traceContributed = true;
      this.state.contributeNudgedTraceId = detail.traceId;
    });
    window.addEventListener('contrib:success', (e: Event) => {
      const file = (e as CustomEvent<{ file?: File }>).detail?.file;
      if (!file) return;
      const traceId = traceFileId(file);
      if (traceId !== this.traceId) return;
      this.traceContributed = true;
      this.state.contributeNudgedTraceId = traceId;
    });
    window.addEventListener(TRACE_CLEARED_EVENT, () => {
      this.state.traceAvailable = false;
      this.traceFile = null;
      this.traceId = null;
      this.traceContributed = false;
      this.resetExecutor();
      this.updateSourceUI();
      if (this.state.source === 'user') this.renderThread();
      this.syncSend();
    });
  }

  // ---- open / close / expand ---------------------------------------------

  private isExpanded(): boolean {
    return document.body.classList.contains('assistant-expanded');
  }

  private defaultSource(): AskHistorySource {
    const live = document.querySelector('.surface.live');
    return live && live.id === 'analyze' ? 'user' : 'public';
  }

  private open(forceSrc?: AskHistorySource): void {
    this.selectSource(forceSrc ?? this.defaultSource());
    document.body.classList.add('assistant-open');
    this.renderHistory();
    this.renderThread();
    window.setTimeout(() => this.input.focus(), 80);
  }

  private close(): void {
    document.body.classList.remove('assistant-open', 'assistant-expanded');
    this.setHistory(false);
    this.btnHistory.classList.remove('on');
  }

  private setHistory(open: boolean): void {
    this.assistant.classList.toggle('hist-open', open);
    this.btnHistory.classList.toggle('on', this.assistant.classList.contains('hist-open'));
  }

  private toggleHistory(): void {
    this.setHistory(!this.assistant.classList.contains('hist-open'));
  }

  private goToAnalyze(): void {
    this.close();
    const tab = document.querySelector<HTMLButtonElement>('nav.tabs button[data-surface="analyze"]');
    tab?.click();
  }

  // ---- source switching ---------------------------------------------------

  private activeConv(): Conversation | undefined {
    return this.history.byId(this.state.activeId);
  }

  /** Switch data source. A conversation with messages can't change source — start a new one. */
  private selectSource(src: AskHistorySource): void {
    if (this.state.busy) this.cancelActiveTurn(); // a turn is streaming — interrupt it, then switch
    const active = this.activeConv();
    if (this.state.source === src && active && active.source === src) {
      this.updateSourceUI();
      return;
    }
    this.state.source = src;
    if (active && active.messages.length === 0) {
      active.source = src;
      this.history.touch();
    } else {
      const conv = this.history.create(src);
      this.state.activeId = conv.id;
    }
    this.updateSourceUI();
    this.renderThread();
    this.renderHistory();
    this.syncSend(); // refresh the composer button (clears any Stop state from an interrupted turn)
    // Intent signal: the user moved to "Your trace" — warm the executor now so their first
    // question doesn't pay the cold Pyodide boot + DuckDB build (covers consent-reading + typing).
    if (src === 'user' && this.state.traceAvailable) this.prefetchExecutor();
  }

  private updateSourceUI(): void {
    const t = assistantCopy().source;
    this.seg.dataset.active = this.state.source;
    this.seg.dataset.trace = this.state.traceAvailable ? 'ready' : 'none';
    this.seg
      .querySelectorAll<HTMLButtonElement>('button')
      .forEach((b) => b.classList.toggle('on', b.dataset.src === this.state.source));

    if (this.state.source === 'public') {
      this.headSub.textContent = t.publicHead;
      this.srcCapText.textContent = t.publicCaption;
      this.footNote.textContent = t.publicFoot;
    } else if (!this.state.traceAvailable) {
      this.headSub.textContent = t.noTraceHead;
      this.srcCapText.textContent = t.noTraceCaption;
      this.footNote.textContent = t.privateFoot;
    } else {
      this.headSub.textContent = t.userHead;
      this.srcCapText.textContent = this.state.userConsented ? t.userConsentedCaption : t.userPrivateCaption;
      this.footNote.textContent = t.privateFoot;
    }
  }

  // ---- history ------------------------------------------------------------

  private renderHistory(): void {
    const t = assistantCopy().history;
    this.histList.innerHTML = '';
    const convs = this.history.list().filter((c) => c.messages.length || c.id === this.state.activeId);
    if (!convs.length) {
      this.histList.innerHTML = `<div class="hist-empty">${escapeHtml(t.empty)}</div>`;
      return;
    }
    convs.forEach((c) => {
      const item = document.createElement('button');
      item.className = 'hist-item' + (c.id === this.state.activeId ? ' active' : '');
      const title = c.title || t.newConversation;
      const label = c.source === 'public' ? t.public : t.user;
      const when = c.ts === 'Now' ? t.now : c.ts;
      item.innerHTML =
        `<span class="ti">${escapeHtml(title)}</span>` +
        `<span class="mt"><span class="hist-pill ${c.source}">${label}</span>${escapeHtml(when)}</span>`;
      item.addEventListener('click', () => this.loadConversation(c.id));
      this.histList.appendChild(item);
    });
  }

  private loadConversation(id: string): void {
    if (this.state.busy) this.cancelActiveTurn(); // interrupt the streaming turn, then switch threads
    const conv = this.history.byId(id);
    if (!conv) return;
    this.state.activeId = id;
    this.state.source = conv.source;
    this.updateSourceUI();
    this.renderThread();
    this.renderHistory();
    this.syncSend();
    if (!this.isExpanded()) this.setHistory(false);
    this.input.focus();
  }

  private startNewConversation(): void {
    if (this.state.busy) this.cancelActiveTurn(); // interrupt the streaming turn before opening a new thread
    const active = this.activeConv();
    if (!active || active.messages.length) {
      const conv = this.history.create(this.state.source);
      this.state.activeId = conv.id;
    }
    this.updateSourceUI();
    this.renderThread();
    this.renderHistory();
    this.syncSend();
    if (!this.isExpanded()) this.setHistory(false);
    this.input.focus();
  }

  // ---- thread rendering ---------------------------------------------------

  private renderThread(): void {
    const conv = this.activeConv();
    if (!conv || !conv.messages.length) {
      this.renderEmpty();
      return;
    }
    this.composer.classList.remove('locked');
    this.body.innerHTML = '';
    conv.messages.forEach((m) => this.body.appendChild(this.buildMessageNode(m)));
    this.scrollDown();
  }

  private buildMessageNode(m: AskStoredMessage): HTMLElement {
    return m.role === 'user' ? buildUserMessage(m.text) : buildBotMessage(m);
  }

  private renderEmpty(): void {
    this.body.innerHTML = '';

    if (this.state.source === 'user' && !this.state.traceAvailable) {
      this.composer.classList.add('locked');
      this.body.appendChild(
        buildNoTraceGate({
          onGoAnalyze: () => this.goToAnalyze(),
          onUsePublic: () => this.selectSource('public'),
        }),
      );
      return;
    }

    if (this.state.source === 'user' && !this.state.userConsented) {
      this.composer.classList.add('locked');
      this.body.appendChild(
        buildConsentGate({
          onConsent: () => {
            this.state.userConsented = true;
            this.prefetchExecutor(); // a question is imminent — make sure the executor is warming
            this.updateSourceUI();
            this.renderThread();
            this.composer.classList.remove('locked');
            this.syncSend();
            this.input.focus();
          },
          onUsePublic: () => this.selectSource('public'),
        }),
      );
      return;
    }

    this.composer.classList.remove('locked');
    this.body.appendChild(
      buildEmptyState(this.state.source, (question) => {
        this.input.value = question;
        this.autoGrow();
        this.syncSend();
        void this.ask();
      }),
    );
  }

  // ---- asking a turn ------------------------------------------------------

  private async ask(): Promise<void> {
    if (this.state.busy) return;
    const q = this.input.value.trim();
    if (!q) return;
    if (this.state.source === 'user' && (!this.state.traceAvailable || !this.state.userConsented)) return;

    const conv = this.activeConv();
    if (!conv) return;

    this.state.busy = true;
    this.syncSend(); // flips the send button into Stop mode for the duration of the turn

    const userMessage: AskStoredMessage = { role: 'user', text: q };
    conv.messages.push(userMessage);
    if (!conv.title) conv.title = titleFromMessage(q);
    conv.ts = assistantCopy().history.now;
    if (conv.messages.length === 1) this.body.innerHTML = '';
    this.body.appendChild(buildUserMessage(q));
    this.input.value = '';
    this.autoGrow();
    this.scrollDown();
    this.renderHistory();
    this.history.touch();

    const progress = new LiveProgress();
    this.body.appendChild(progress.element);
    this.scrollDown();

    // Own this turn's abort controller locally. If the user interrupts or switches mid-turn,
    // cancelActiveTurn() aborts + nulls this.abort, so the `this.abort !== abort` guards below let the
    // unwinding turn bail out without rendering a stale answer or clobbering newer state.
    const abort = new AbortController();
    this.abort = abort;
    try {
      const result = await this.runTurn(conv, progress, abort.signal);
      if (this.abort !== abort) {
        progress.remove(); // superseded by an interrupt / source-or-thread switch — drop the result
        return;
      }
      progress.remove();
      const assistantMessage = this.toStoredMessage(result);
      conv.messages.push(assistantMessage);
      this.history.touch();
      this.body.appendChild(buildBotMessage(assistantMessage));
      this.scrollDown();
      this.state.busy = false;
      this.syncSend();
      if (this.shouldShowContributeNudge()) {
        window.setTimeout(() => this.showContributeNudge(), 550);
      }
    } catch (err) {
      progress.remove();
      // Superseded (covers every user-initiated abort, since cancelActiveTurn nulls this.abort): the
      // canceller already reset busy + the button, so don't touch shared state or render anything.
      if (this.abort !== abort) return;
      if ((err as Error)?.name !== 'AbortError') {
        const errorMessage: AskStoredMessage = {
          role: 'assistant',
          text: template(assistantCopy().errors.failed, { message: String((err as Error)?.message ?? err) }),
        };
        conv.messages.push(errorMessage);
        this.history.touch();
        this.body.appendChild(buildBotMessage(errorMessage));
        this.scrollDown();
      }
      this.state.busy = false;
      this.syncSend();
    } finally {
      if (this.abort === abort) this.abort = null;
    }
  }

  /** Stop the in-flight turn (Stop button / Esc). The aborted ask() unwinds via its superseded guards. */
  private interrupt(): void {
    if (!this.state.busy) return;
    this.cancelActiveTurn();
    this.syncSend();
  }

  /**
   * Abort the active turn (if any) and clear busy synchronously, so an interrupt or a source/thread
   * switch leaves consistent state immediately. The in-flight ask() still unwinds asynchronously, but
   * its `this.abort !== abort` guards keep it from rendering or re-touching shared state.
   */
  private cancelActiveTurn(): void {
    if (this.abort) {
      this.abort.abort();
      this.abort = null;
    }
    this.state.busy = false;
  }

  /** Run one assistant turn, driving the live progress checklist from the real streamed events. */
  private async runTurn(conv: Conversation, progress: LiveProgress, signal: AbortSignal): Promise<AskResult> {
    const source = clientSource(conv.source);
    const messages: AskMessage[] = conv.messages.map((m) => ({
      role: m.role,
      content: m.text,
    }));

    const tp = new TurnProgress(progress, source);
    tp.start();

    const onEvent = (event: AskEvent): void => tp.handle(event);
    const onToolStart = (): void => tp.toolStart();
    const onToolEnd = (_id: string, result: PyodideToolResult): void => tp.toolEnd(result);

    if (source === 'syfi') {
      return askTurn({ source, sessionId: this.sessionId, messages, locale: browserCopy().locale, onEvent, signal });
    }

    // User source: ensure the trace is loaded into a Pyodide executor first.
    const ready = await this.ensureExecutor();
    tp.traceReady();
    return askTurn({
      source,
      sessionId: this.sessionId,
      messages,
      executor: this.executor!,
      dbPath: ready.dbPath,
      traceContext: ready.traceContext,
      locale: browserCopy().locale,
      onEvent,
      onToolStart,
      onToolEnd,
      signal,
    });
  }

  /** Map an AskResult onto a stored assistant message: answer text, plots, joined code. */
  private toStoredMessage(result: AskResult): AskStoredMessage {
    const images: AskStoredImage[] = (result.display_images || [])
      .filter((art: PyodideArtifact) => !!art.data_url)
      .map((art: PyodideArtifact) => ({ dataUrl: art.data_url as string, caption: art.path || assistantCopy().generatedPlot }));
    const code = (result.tool_events || [])
      .map((e) => e.code)
      .filter((c): c is string => !!c)
      .join('\n\n');
    return {
      role: 'assistant',
      text: result.content || '',
      images: images.length ? images : undefined,
      code: code || undefined,
    };
  }

  private showContributeNudge(): void {
    if (!this.shouldShowContributeNudge()) return;
    const hasTrace = this.state.traceAvailable;
    if (hasTrace) this.state.contributeNudgedTraceId = this.traceId;
    else this.state.publicContributeNudgeShown = true;
    const card = buildContributeNudge(hasTrace, {
      onAccept: () => {
        if (hasTrace) {
          // A trace is already analyzed: close the dock and let Analyze open the real
          // contribution dialog over the page (it owns the analyzed bytes + consent flow).
          this.close();
          window.dispatchEvent(new CustomEvent('ask:contribute'));
        } else {
          this.goToAnalyze();
        }
      },
    });
    this.body.appendChild(card);
    this.scrollDown();
  }

  private shouldShowContributeNudge(): boolean {
    if (!this.state.traceAvailable) return !this.state.publicContributeNudgeShown;
    if (!this.traceId || this.traceContributed) return false;
    return this.state.contributeNudgedTraceId !== this.traceId;
  }

  // ---- Pyodide executor (user source) ------------------------------------

  /** Lazily create the executor and load the trace; reused across turns in the same session. */
  private ensureExecutor(): Promise<TraceReady> {
    if (this.traceReady) return Promise.resolve(this.traceReady);
    if (this.executorLoad) return this.executorLoad;
    if (!this.traceFile) return Promise.reject(new Error(assistantCopy().errors.noTrace));

    if (!this.executor) this.executor = new PyodideToolExecutorClient();
    const file = this.traceFile;
    this.executorLoad = this.executor
      .loadTrace(file)
      .then((ready) => {
        this.traceReady = ready;
        return ready;
      })
      .catch((err) => {
        this.executorLoad = null;
        throw err;
      });
    return this.executorLoad;
  }

  /**
   * Warm the in-browser executor (boot Pyodide + materialize the trace's DuckDB) ahead of the first
   * question, so the user isn't waiting on a cold start. Fire-and-forget and idempotent — ensureExecutor
   * dedupes, and the real ask re-runs it and surfaces any error. This is purely local work (no cloud
   * call), so it's safe to start before consent. Triggered on intent: selecting "Your trace",
   * analyzing while on that source, or consenting.
   */
  private prefetchExecutor(): void {
    if (!this.traceFile || this.traceReady || this.executorLoad) return;
    void this.ensureExecutor().catch(() => {
      /* ignore — the first ask retries ensureExecutor and reports any failure */
    });
  }

  /**
   * Boot the executor's Pyodide + plotting stack (the ~28 s matplotlib/font-cache tax) WITHOUT a trace,
   * so it overlaps the local-executor's server-side compute instead of landing on the user's first
   * question. The trace loads later via ensureExecutor, reusing THIS same warm worker — and, because
   * the local path primes the DuckDB cache, that load is a cache HIT. Idempotent; no-op once loaded.
   */
  private prewarmBoot(): void {
    if (this.traceReady || this.executorLoad) return; // already loaded / loading the trace
    if (!this.executor) this.executor = new PyodideToolExecutorClient();
    this.executor.warm();
  }

  private resetExecutor(): void {
    this.executor?.terminate();
    this.executor = null;
    this.traceReady = null;
    this.executorLoad = null;
  }

  private markTraceReady(): void {
    this.state.traceAvailable = true;
    this.updateSourceUI();
    if (this.state.source === 'user') {
      this.renderThread();
      this.prefetchExecutor(); // already on "Your trace" when analysis finished — warm it now
    }
    this.syncSend();
  }

  // ---- small helpers ------------------------------------------------------

  private scrollDown(): void {
    this.body.scrollTop = this.body.scrollHeight;
  }

  private autoGrow(): void {
    this.input.style.height = 'auto';
    this.input.style.height = `${Math.min(this.input.scrollHeight, 140)}px`;
  }

  private syncSend(): void {
    // While a turn streams, the button is an always-enabled Stop control; otherwise it's Send, gated
    // on a non-empty question and (for "Your trace") an available + consented trace.
    if (this.state.busy) {
      this.send.classList.add('stop');
      this.send.disabled = false;
      this.send.setAttribute('aria-label', assistantCopy().controls.stop);
      this.send.title = assistantCopy().controls.stop;
      return;
    }
    this.send.classList.remove('stop');
    this.send.removeAttribute('title');
    this.send.setAttribute('aria-label', assistantCopy().controls.send);
    this.send.disabled =
      !this.input.value.trim() ||
      (this.state.source === 'user' && (!this.state.traceAvailable || !this.state.userConsented));
  }
}

/** Bootstrap the assistant. Safe to call once the AskTrace markup is in the DOM. */
export function mountAskTrace(): void {
  const ids = [
    'assistant',
    'asstBody',
    'asstInput',
    'asstSend',
    'seg',
    'composer',
    'headSub',
    'srcCapText',
    'footNote',
    'histList',
    'btnHistory',
  ] as const;

  const found: Record<string, HTMLElement> = {};
  for (const id of ids) {
    const node = document.getElementById(id);
    if (!node) return; // markup not present (e.g. detail pages) — nothing to mount
    found[id] = node;
  }

  const controller = new AskTraceController({
    assistant: found.assistant,
    body: found.asstBody,
    input: found.asstInput,
    send: found.asstSend,
    seg: found.seg,
    composer: found.composer,
    headSub: found.headSub,
    srcCapText: found.srcCapText,
    footNote: found.footNote,
    histList: found.histList,
    btnHistory: found.btnHistory,
  });
  controller.mount();
}
