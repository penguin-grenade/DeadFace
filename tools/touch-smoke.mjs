// Touch controls check: a phone-sized landscape screen with multi-touch, driven through the
// DevTools protocol (real touch events, so the game sees touch pointers). Starts the game with a
// tap, then walks with the stick, sprints, looks, fires, aims, reloads, moves and looks with two
// thumbs at once, and pauses. Fails on a missed check or a console error.
//
//   node tools/touch-smoke.mjs        # writes screenshots/touch-*.png
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { mkdirSync } from 'node:fs';

const W = 844;
const H = 390;
const outDir = new URL('../screenshots/', import.meta.url).pathname;
mkdirSync(outDir, { recursive: true });
const server = await createServer({ server: { port: 5195, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const context = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: Number(process.env.DPR || 1), isMobile: true, hasTouch: true });
await context.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.fulfill({ status: 200, contentType: 'text/css', body: '' }));
const page = await context.newPage();
const errors = [];
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
page.on('pageerror', (e) => errors.push(String(e)));
const cdp = await context.newCDPSession(page);
const touch = (type, points) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: points.map(([x, y], id) => ({ x, y, id })) });
const wait = (ms) => page.waitForTimeout(ms);
// Software rendering is slow: give screenshots time.
const shot = (name) => page.screenshot({ path: `${outDir}/touch-${name}.png`, timeout: 180_000 });
const state = () =>
  page.evaluate(() => {
    const g = window.__game;
    const t = g.player.body.translation();
    const i = g.weapon.input;
    return { x: t.x, z: t.z, yaw: g.player.yaw, moveX: i.moveX, moveY: i.moveY, sprint: i.sprint, sprinting: g.player.sprinting, aim: i.aim, weaponAim: g.weapon.aim, ammo: g.weapon.ammo, reloading: g.weapon.reloading, touch: i.touch };
  });
const centre = (sel) =>
  page.evaluate((sel) => {
    const r = document.querySelector(sel).getBoundingClientRect();
    return [r.x + r.width / 2, r.y + r.height / 2];
  }, sel);
const tap = async (sel) => {
  const p = await centre(sel);
  await touch('touchStart', [p]);
  await wait(80);
  await touch('touchEnd', []);
  await wait(250);
};
let failed = 0;
const check = (name, ok, info) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${info === undefined ? '' : ` ${JSON.stringify(info)}`}`);
  if (!ok) failed++;
};

await page.goto('http://127.0.0.1:5195/?fixedres');
await page.waitForFunction(() => window.__game && !document.getElementById('cta').hidden, null, { timeout: 300_000 });
check('start screen says tap', (await page.textContent('#cta')) === 'Tap to start');
await shot('start');

await touch('touchStart', [[W / 2, H / 2]]);
await wait(60);
await touch('touchEnd', []);
await wait(1500);
let s = await state();
check('tap starts touch play', s.touch && (await page.isHidden('#start')) && (await page.isVisible('.touch')));
await wait(3000);
await shot('play');

// Walk: left thumb down, pushed up to the ring.
s = await state();
const start = s;
await touch('touchStart', [[110, 300]]);
await touch('touchMove', [[110, 280]]);
await touch('touchMove', [[110, 252]]);
await wait(2500);
s = await state();
// Game time runs slower than the clock under software rendering, so only the direction is checked.
const ahead = -Math.sin(start.yaw) * (s.x - start.x) - Math.cos(start.yaw) * (s.z - start.z);
check('stick walks forward', s.moveY > 0.9 && Math.abs(s.moveX) < 0.05 && ahead > 0.05, { moveY: +s.moveY.toFixed(2), metres: +ahead.toFixed(2) });
await touch('touchMove', [[112, 190]]);
await wait(600);
s = await state();
check('pushing past the ring sprints', s.sprint && s.sprinting);
await shot('sprint');
await touch('touchEnd', []);
await wait(300);
s = await state();
check('letting go stops', s.moveX === 0 && s.moveY === 0 && !s.sprint);

// Look: right thumb drags left to right.
const yaw0 = s.yaw;
await touch('touchStart', [[560, 150]]);
for (let k = 1; k <= 10; k++) await touch('touchMove', [[560 + k * 15, 150]]);
await touch('touchEnd', []);
await wait(500);
s = await state();
check('dragging right turns right', s.yaw < yaw0 - 0.2, { turned: +(yaw0 - s.yaw).toFixed(2) });

// Fire, aim, reload.
const ammo0 = s.ammo;
await tap('.tbtn.fire:not(.l)');
await wait(300);
s = await state();
check('fire button shoots', s.ammo === ammo0 - 1, { ammo: s.ammo });
await tap('.tbtn.fire.l');
await wait(300);
s = await state();
check('left fire button shoots', s.ammo === ammo0 - 2, { ammo: s.ammo });
await tap('.tbtn.aim');
const raised = await page.waitForFunction(() => window.__game.weapon.aim > 0.9, null, { timeout: 30_000 }).then(() => true, () => false);
s = await state();
check('aim button raises the gun', s.aim && raised, { aim: +s.weaponAim.toFixed(2) });
await shot('aim');
await tap('.tbtn.aim');
const lowered = await page.waitForFunction(() => window.__game.weapon.aim < 0.1, null, { timeout: 30_000 }).then(() => true, () => false);
s = await state();
check('aim button again lowers it', !s.aim && lowered);
await tap('.tbtn.reload');
s = await state();
check('reload button reloads', s.reloading);

// Two thumbs: walk and look at once.
await wait(2500);
s = await state();
const both0 = s;
await touch('touchStart', [[110, 300]]);
await touch('touchStart', [[110, 300], [560, 150]]);
for (let k = 1; k <= 8; k++) await touch('touchMove', [[110, 300 - k * 6], [560 - k * 12, 150]]);
await wait(800);
s = await state();
check('move and look together', s.moveY > 0.5 && s.yaw > both0.yaw + 0.1, { moveY: +s.moveY.toFixed(2), turned: +(s.yaw - both0.yaw).toFixed(2) });
await touch('touchEnd', [[560 - 96, 150]]);
await touch('touchEnd', []);

// Pause, then resume.
await wait(300);
await tap('.tbtn.pause');
await wait(300);
s = await state();
check('pause button shows the menu', !s.touch && (await page.isVisible('#start')) && (await page.isHidden('.touch')));
await shot('pause');
await touch('touchStart', [[W / 2, H - 30]]);
await wait(60);
await touch('touchEnd', []);
await wait(800);
s = await state();
check('tap resumes', s.touch && (await page.isHidden('#start')));

check('no console errors', errors.length === 0, errors.slice(0, 5));
await browser.close();
await server.close();
if (failed) {
  console.log(`${failed} check(s) failed`);
  process.exit(1);
}
console.log('OK');
