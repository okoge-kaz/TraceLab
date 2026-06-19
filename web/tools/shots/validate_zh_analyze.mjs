import { chromium } from 'playwright';
import { resolve, dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync, existsSync } from 'node:fs';
import { createServer } from 'node:http';

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, '../../app/dist');
const tar = process.env.TAR || resolve(here, '../../../test_workload_claude_sessions.tar.gz');

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.wasm': 'application/wasm',
  '.map': 'application/json',
};

const server = createServer((req, res) => {
  let p = decodeURIComponent((req.url || '/').split('?')[0].split('#')[0]);
  if (p.endsWith('/')) p += 'index.html';
  let f = join(dist, p);
  if (!existsSync(f) && existsSync(f + '.html')) f += '.html';
  if (!existsSync(f) || extname(f) === '') f = join(dist, p, 'index.html');
  try {
    const body = readFileSync(f);
    res.writeHead(200, { 'content-type': MIME[extname(f)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
});

await new Promise((resolveListen) => server.listen(0, '127.0.0.1', resolveListen));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1700 }, deviceScaleFactor: 1 });

const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`PAGEERROR: ${e.message}`));

console.log('goto', `${base}/zh/#analyze`);
await page.goto(`${base}/zh/#analyze`, { waitUntil: 'domcontentloaded' });
await page.click('nav.tabs button[data-surface="analyze"]').catch(() => {});
await page.waitForSelector('#file-input', { state: 'attached', timeout: 20000 });
console.log('uploading', tar);
await page.setInputFiles('#file-input', tar);

let done = false;
const statusTexts = [];
for (let i = 0; i < 180; i++) {
  const title = await page.$eval('#dz-title', (el) => el.textContent || '').catch(() => '');
  if (title) statusTexts.push(title);
  if (/分析完成|Analysis complete/.test(title)) {
    done = true;
    console.log('analysis done:', title);
    break;
  }
  if (/无法|Couldn.t read/.test(title)) {
    console.log('analysis error:', title);
    break;
  }
  if (i % 10 === 0) console.log('status:', title);
  await page.waitForTimeout(1000);
}

await page.waitForSelector('#analytics-dashboard canvas', { timeout: 20000 }).catch(() => {});
await page.click('#launcher').catch(() => {});
await page.waitForTimeout(800);

const visibleText = await page.evaluate(() => document.body.innerText);
const suspects = [
  'Now',
  'unrecognized file',
  'Most 步 land',
  'Most steps land',
  'active days',
  'day span',
  'served from cache',
  'avg / step',
  ' users',
  'No sessions match',
  'Click a round',
  'Loading timeline',
  'Could not load',
  'Round ',
  'user-initiated',
  'tool-step',
  'Cached input',
  'Fresh input',
  'Tool calls',
  'Tool latency',
  'Generation time',
  'Prefix cache hit ratio',
  'Human input wait',
  'human wait',
  'Download PNG',
  'Try one',
  'What would you like',
  'Use the public pool',
  'No conversations yet',
  'Reading the public',
  'Answers run',
  'Detecting format',
  'Building your database',
].filter((s) => visibleText.includes(s) || statusTexts.some((t) => t.includes(s)));

console.log('completed:', done);
console.log('suspects:', JSON.stringify(suspects, null, 2));
console.log('sample visible text:');
console.log(visibleText.slice(0, 2500));
console.log('logs:', logs.slice(-20).join('\n'));

await browser.close();
server.close();
process.exit(suspects.length || !done ? 1 : 0);
