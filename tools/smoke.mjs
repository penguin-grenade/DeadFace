// Headless smoke test: builds nothing, starts the Vite dev server, opens the
// game in ?demo mode (auto look + auto fire), collects console errors and
// writes screenshots to ./screenshots.
//
//   npm run smoke            # uses $CHROMIUM or Playwright's bundled chromium
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { mkdirSync } from 'node:fs';

const outDir = new URL('../screenshots/', import.meta.url).pathname;
mkdirSync(outDir, { recursive: true });

const server = await createServer({ server: { port: 5199, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const url = 'http://127.0.0.1:5199/?demo';

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
// Keep the test offline: web fonts are optional (the CSS has fallback stacks).
await context.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.fulfill({ status: 200, contentType: 'text/css', body: '' }));
const page = await context.newPage();
const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text());
  if (process.env.VERBOSE) console.log(`[${m.type()}] ${m.text()}`);
});
page.on('requestfailed', (r) => errors.push('requestfailed ' + r.url()));
page.on('response', (r) => { if (r.status() >= 400) errors.push(r.status() + ' ' + r.url()); });
page.on('pageerror', (e) => errors.push(String(e)));

await page.goto(url);
await page.waitForFunction(() => window.__game, null, { timeout: 120_000 });
const shots = Number(process.env.SHOTS || 3);
for (let i = 0; i < shots; i++) {
  await page.waitForTimeout(Number(process.env.INTERVAL || 4000));
  await page.screenshot({ path: `${outDir}/demo-${i}.png` });
}
const stats = await page.evaluate(() => {
  const g = window.__game;
  const info = g.renderer.renderer.info;
  return { ammo: g.weapon.hudText, calls: info.render.calls, triangles: info.render.triangles, pos: g.player.position.toArray().map((v) => +v.toFixed(2)) };
});
console.log('stats', stats);
await page.close();

// Functional check: aim at the nearest mannequin and fire until it drops.
const page2 = await context.newPage();
page2.on('pageerror', (e) => errors.push(String(e)));
await page2.goto(url.replace('?demo', ''), { timeout: 120_000 });
await page2.waitForFunction(() => window.__game, null, { timeout: 120_000 });
const combat = await page2.evaluate(async () => {
  const g = window.__game;
  const m = g.targets.mannequins[0];
  const V = g.camera.position.constructor;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  // Software GL renders slowly, so re-aim before every shot (recoil climbs)
  // and wait for the smoothed camera to settle on the target.
  const aim = async () => {
    const cam = g.camera.getWorldPosition(new V());
    const t = m.body.translation();
    const dx = t.x - cam.x, dy = t.y + 1.25 - cam.y, dz = t.z - cam.z;
    g.player.yawTarget = Math.atan2(-dx, -dz);
    g.player.pitchTarget = Math.atan2(dy, Math.hypot(dx, dz));
    for (let i = 0; i < 100; i++) {
      if (Math.abs(g.player.yaw - g.player.yawTarget) < 0.002 && Math.abs(g.player.pitch - g.player.pitchTarget) < 0.002) break;
      await sleep(100);
    }
  };
  let shots = 0;
  while (!m.down && shots < 10) {
    await aim();
    g.weapon.cooldown = 0;
    g.weapon.aim = 1; // aimed fire: tight spread
    g.weapon.fire();
    shots++;
  }
  return { down: m.down, shots, ammo: g.weapon.ammo };
});
await page2.waitForTimeout(1500);
await page2.evaluate(() => (document.getElementById('start').hidden = true));
await page2.screenshot({ path: `${outDir}/combat.png` });
console.log('combat', combat);
if (!combat.down) errors.push('mannequin did not go down after 10 shots');


await browser.close();
await server.close();

if (errors.length) {
  console.error('Console errors:\n' + errors.join('\n'));
  process.exit(1);
}
console.log(`OK, screenshots in ${outDir}`);
