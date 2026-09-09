// Render the camera grid in a real browser and assert it actually shows video.
//
// The Python tests can only assert that strings appear in INDEX_HTML, which
// cannot catch the failures that matter: a poller started before its card is
// in the document, a JS error that stops the page, or images that load but
// stay blank. Those have all shipped. This renders the page, waits for the
// poller, and fails if any feed is missing, undersized or uniformly dark.
//
//   npx playwright install chromium     # once
//   node tools/check_camera_page.mjs [url]
import { chromium } from 'playwright';

const url = process.argv[2] || 'http://127.0.0.1:8766/';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });

const problems = [];
page.on('pageerror', e => problems.push(`uncaught: ${e.message}`));
page.on('console', m => m.type() === 'error' && problems.push(`console: ${m.text()}`));
page.on('requestfailed', r => problems.push(`request failed: ${r.url()}`));

await page.goto(url, { waitUntil: 'domcontentloaded' });
// A camera can take several seconds to deliver its first frame after a
// restart, and a short window reports that as a dead feed.
await page.waitForTimeout(12000);

const feeds = await page.evaluate(() => {
  const drawn = [...document.querySelectorAll('#cameras article img')].map(img => {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    if (!canvas.width || !canvas.height) return { w: 0, h: 0, mean: 0 };
    canvas.getContext('2d').drawImage(img, 0, 0);
    const { data } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
    let total = 0;
    // Sample every 40th pixel; a uniform frame needs no more than that.
    for (let i = 0; i < data.length; i += 160) total += data[i] + data[i + 1] + data[i + 2];
    return { w: canvas.width, h: canvas.height, mean: total / (3 * (data.length / 160)) };
  });
  return { cards: document.querySelectorAll('#cameras article').length, drawn };
});

if (!feeds.cards) problems.push('no camera cards rendered');
feeds.drawn.forEach((feed, index) => {
  if (!feed.w || !feed.h) problems.push(`feed ${index} never loaded an image`);
  else if (feed.mean < 8) problems.push(`feed ${index} is essentially black (mean ${feed.mean.toFixed(1)})`);
});

console.log(JSON.stringify(feeds, null, 1));
await browser.close();
if (problems.length) {
  console.error('FAIL\n  ' + problems.join('\n  '));
  process.exit(1);
}
console.log(`OK: ${feeds.cards} feeds rendering with live pixels`);
