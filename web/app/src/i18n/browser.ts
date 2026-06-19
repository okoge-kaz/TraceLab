import { uiCopy, template as renderTemplate } from './index';

type BrowserCopy = (typeof uiCopy)[keyof typeof uiCopy]['browser'];

declare global {
  interface Window {
    __TRACELAB_I18N__?: BrowserCopy;
  }
}

export function browserCopy(): BrowserCopy {
  return window.__TRACELAB_I18N__ ?? uiCopy.en.browser;
}

export function template(text: string, values: Record<string, string | number>): string {
  return renderTemplate(text, values);
}
