// Headless screenshots from fixed viewpoints, for judging the look while tuning.
//
//   node tools/shots.mjs [name ...]        # default: every viewpoint below
//   QUERY=oldlevel node tools/shots.mjs    # extra URL query
//   W=1600 H=900 node tools/shots.mjs
// Writes screenshots/<name>.png. Uses $CHROMIUM or Playwright's bundled chromium.
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { mkdirSync } from 'node:fs';

// Feet position (three.js world space), yaw (0 = looking down -Z), pitch (radians, + = up).
const VIEWS = {
  spawn: { pos: [0, 0, 6.5], yaw: 0, pitch: -0.05 },
  hall: { pos: [-1.2, 0, 1.5], yaw: 0.35, pitch: 0.05 },
  range: { pos: [0.5, 0, -6], yaw: 0.05, pitch: -0.04 },
  racks: { pos: [6.5, 0, 2.5], yaw: -0.9, pitch: 0.0 },
  office: { pos: [-6.1, 0, -9.2], yaw: 0.9, pitch: -0.05 },
  puddle: { pos: [2.0, 0, -1.0], yaw: 2.6, pitch: -0.45 },
  up: { pos: [0, 0, -2], yaw: 0.3, pitch: 0.9 },
  // Same cameras as blender/build_level.py PREVIEW_CAMS (use with QUERY=clean&tm=agx&exp=2).
  b_spawn: { pos: [0, 0, 6.5], yaw: 0, pitch: -0.035, eye: 1.55 },
  b_hall: { pos: [-9.5, 0, 6.8], yaw: -0.663, pitch: -0.105, eye: 2.2 },
  b_range: { pos: [0.5, 0, -5], yaw: -0.14, pitch: -0.035, eye: 1.6 },
  b_racks: { pos: [7, 0, 4.5], yaw: -0.314, pitch: 0, eye: 1.6 },
  b_office: { pos: [-4, 0, -6.5], yaw: 0.611, pitch: -0.07, eye: 1.6 },
  bench: { pos: [3.0, 0, 5.8], yaw: 0.2, pitch: -0.5 },
  // Weapon close-ups under the spawn work light (aim = hold right mouse).
  gun: { pos: [0.4, 0, 4.4], yaw: 0.25, pitch: -0.2 },
  gun_ads: { pos: [0.4, 0, 4.4], yaw: 0.25, pitch: -0.2, aim: true },
  // Range mannequins: up close (the one at x 4.3, z -7) and down the range.
  dummy: { pos: [3.63, 0, -4.9], yaw: -0.31, pitch: -0.12 },
  dummy_far: { pos: [1.2, 0, 0.5], yaw: -0.28, pitch: -0.08 },
};

const names = process.argv.slice(2).filter((a) => !a.startsWith('-'));
const list = names.length ? names : Object.keys(VIEWS);
const outDir = new URL('../screenshots/', import.meta.url).pathname;
mkdirSync(outDir, { recursive: true });

const server = await createServer({ server: { port: 5198, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const query = `?fixedres${process.env.QUERY ? `&${process.env.QUERY}` : ''}`;
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM || undefined,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const context = await browser.newContext({ viewport: { width: Number(process.env.W || 1280), height: Number(process.env.H || 720) } });
await context.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.fulfill({ status: 200, contentType: 'text/css', body: '' }));
const page = await context.newPage();
const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error' || m.type() === 'warning') errors.push(`[${m.type()}] ${m.text()}`);
  if (process.env.VERBOSE) console.log(`[${m.type()}] ${m.text()}`);
});
page.on('pageerror', (e) => errors.push(String(e)));
const t0 = Date.now();
await page.goto(`http://127.0.0.1:5198/${query}`);
await page.waitForFunction(() => window.__game, null, { timeout: 300_000 });
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
await page.evaluate(() => {
  document.getElementById('start').hidden = true;
});

for (const name of list) {
  const v = VIEWS[name];
  if (!v) {
    console.warn('unknown view', name);
    continue;
  }
  await page.evaluate((v) => {
    const g = window.__game;
    g.player.body.setTranslation({ x: v.pos[0], y: v.pos[1] + 0.9, z: v.pos[2] }, true);
    g.player.body.setNextKinematicTranslation({ x: v.pos[0], y: v.pos[1] + 0.9, z: v.pos[2] });
    g.player.yaw = g.player.yawTarget = g.player.prevYaw = v.yaw;
    g.player.pitch = g.player.pitchTarget = g.player.prevPitch = v.pitch;
    g.player.velocity.set(0, 0, 0);
    g.weapon.input.aim = !!v.aim;
  }, v);
  // Let springs, auto exposure and shadow caches settle.
  await page.waitForTimeout(Number(process.env.SETTLE || 5000));
  await page.screenshot({ path: `${outDir}/${name}.png` });
  const info = await page.evaluate(() => {
    const g = window.__game;
    const r = g.renderer.renderer.info;
    // Where the front sight post lands on screen (NDC), to check the ADS sight picture.
    const sight = g.weapon.view.position.clone().set(0, 0.0487, -0.153);
    g.weapon.model?.localToWorld(sight);
    sight.project(g.player.camera);
    return { calls: r.render.calls, tris: r.render.triangles, programs: r.programs?.length, aim: +g.weapon.aim.toFixed(3), sight: [+sight.x.toFixed(3), +sight.y.toFixed(3)] };
  });
  console.log(name, JSON.stringify(info));
}
if (errors.length) console.log('console:\n' + [...new Set(errors)].slice(0, 30).join('\n'));
await browser.close();
await server.close();
