// Optional real-browser end-to-end test (not run by pytest by default).
//   npm i puppeteer-core @sparticuz/chromium   (in a scratch dir), serve the repo root on :8000, then
//   node tests/browser/e2e_checker.mjs <dir-with-test-tifs>
// It drives docs/executive_summary.html in headless Chromium: uploads files through the real <input>, reads the rendered verdicts.
process.env.AWS_EXECUTION_ENV = process.env.AWS_EXECUTION_ENV || 'AWS_Lambda_nodejs20.x';
const chromium = (await import('@sparticuz/chromium')).default;
const puppeteer = (await import('puppeteer-core')).default;
const fs = (await import('fs')).default;
const dir = process.argv[2];
const base = process.env.SITE || 'http://127.0.0.1:8000/docs/';
const cases = JSON.parse(fs.readFileSync(`${dir}/cases.json`, 'utf8'));   // [{file, expect: 'PASS'|'FAIL', mustContain: [...]}]
const browser = await puppeteer.launch({ args: [...chromium.args, '--no-sandbox'], executablePath: await chromium.executablePath(), headless: 'shell' });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
await page.setViewport({ width: 1100, height: 1400 });
let failed = 0;
for (const c of cases) {
  await page.goto(base + 'executive_summary.html', { waitUntil: 'networkidle0' });
  const input = await page.$('#checker input[type=file]');
  if (!input) { console.log('NO INPUT FOUND'); failed++; continue; }
  await input.uploadFile(`${dir}/${c.file}`);
  await page.waitForFunction(() => /PASS|FAIL|Could not read|ZIP/.test(document.querySelector('#checker .out').textContent), { timeout: 60000 });
  const text = await page.$eval('#checker .out', e => e.textContent);
  const verdict = /PASS:/.test(text) ? 'PASS' : (/FAIL:|Could not read/.test(text) ? 'FAIL' : '?');
  const ok = verdict === c.expect && (c.mustContain || []).every(s => text.includes(s));
  console.log(ok ? 'ok  ' : 'BAD ', c.file, '->', verdict, ok ? '' : text.slice(0, 300));
  if (!ok) failed++;
  if (c.shot) { const el = await page.$('#checker'); await el.screenshot({ path: `${dir}/${c.shot}` }); }
}
// copy buttons
await page.goto(base + 'index.html', { waitUntil: 'networkidle0' });
const before = await page.$eval('[data-copy]', e => e.textContent);
await page.click('[data-copy]');
await new Promise(r => setTimeout(r, 300));
const after = await page.$eval('[data-copy]', e => e.textContent);
console.log(/Copied|Selected/.test(after) ? 'ok  ' : 'BAD ', 'copy button:', JSON.stringify(before), '->', JSON.stringify(after));
if (!/Copied|Selected/.test(after)) failed++;
console.log('page errors:', JSON.stringify(errors));
if (errors.length) failed++;
await browser.close();
process.exit(failed ? 1 : 0);
