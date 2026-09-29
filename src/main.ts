import * as THREE from 'three';
import { Renderer, type Quality } from './engine/renderer';
import { assetManifest, assetProblems, loadGLTF, policyBlocks, reportAssetProblem } from './engine/assets';
import { Physics } from './engine/physics';
import { Input } from './engine/input';
import { Audio } from './engine/audio';
import { buildLevel, makeMaterials, type Level } from './game/level';
import { loadBakedLevel } from './game/bakedLevel';
import { levelUniforms } from './game/levelShading';
import { Player } from './game/player';
import { Weapon } from './game/weapon';
import { Effects } from './game/effects';
import { Targets, type ModelLibrary } from './game/targets';

const params = new URLSearchParams(location.search);
/** ?demo runs without pointer lock: slow look-around + periodic fire, used by the smoke test. */
const DEMO = params.has('demo');

async function loadModels(): Promise<ModelLibrary> {
  const lib: ModelLibrary = {};
  const { models } = await assetManifest();
  await Promise.all(
    (['mannequin', 'steel_target', 'can', 'box_small'] as const).map(async (name) => {
      if (!models.includes(name)) return;
      try {
        const gltf = await loadGLTF(`models/${name}.glb`);
        gltf.scene.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) o.castShadow = o.receiveShadow = true;
        });
        lib[name] = gltf.scene;
      } catch (e) {
        reportAssetProblem(name, e);
      }
    }),
  );
  return lib;
}

async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const hud = document.getElementById('overlay')!;
  const ammoEl = document.getElementById('ammo')!;
  const stampEl = document.getElementById('stamp-time')!;
  const hintEl = document.getElementById('hint')!;
  const startEl = document.getElementById('start')!;
  const loadingEl = document.getElementById('loading')!;
  const ctaEl = document.getElementById('cta')!;
  const noteEl = document.getElementById('asset-note')!;

  const scene = new THREE.Scene();
  // Wide lens; the barrel distortion pass adds the fisheye look on top.
  const camera = new THREE.PerspectiveCamera(78, innerWidth / innerHeight, 0.02, 120);
  const renderer = new Renderer(canvas, scene, camera);

  const physics = new Physics();
  const [, mats, models] = await Promise.all([physics.init(), makeMaterials(), loadModels()]);

  const input = new Input(canvas);
  const audio = new Audio();
  // The Blender-baked warehouse when it's there (public/level/), else the procedural range.
  const baked = params.has('oldlevel')
    ? null
    : await loadBakedLevel(scene, physics, renderer.renderer).catch((e) => {
        reportAssetProblem('warehouse level', e);
        return null;
      });
  const level: Level = baked ?? buildLevel(scene, physics, renderer.renderer, mats);
  const effects = new Effects(scene);

  const player = new Player(physics, input, audio, camera);
  scene.add(player.root);
  player.spawn(level.spawn, level.spawnYaw);

  const weapon = new Weapon(scene, physics, audio, input, player, effects);
  await weapon.load();

  const targets = new Targets(scene, physics, mats, models);
  level.targetSpots.forEach((p, i) => targets.addMannequin(p, Math.atan2(level.spawn.x - p.x, level.spawn.z - p.z), i === 1 ? 1.5 : 0));
  level.plateSpots.forEach((p) => targets.addPlate(p));
  level.propSpots.forEach((p, i) => (i < 4 ? targets.addCan(p) : targets.addBox(p)));
  if (baked) {
    for (const { object, radius } of targets.movingObjects()) baked.track(object, radius);
    baked.track(weapon.view, 0.5, false);
    renderer.post.setVolumetricLights(
      [...baked.lamps.filter((l) => l.volumetric > 0).map((l) => ({ light: l.light, strength: l.volumetric })), { light: weapon.flashlight, strength: 1.4 }],
      baked.moon ? { ...baked.moon, color: baked.moon.color.clone().multiplyScalar(4) } : null,
    );
  }

  let score = 0;
  let hintTimer = 0;
  const hint = (text: string) => {
    hintEl.textContent = text;
    hintEl.classList.add('show');
    hintTimer = 1.6;
  };
  targets.onKill = (head) => {
    score++;
    hint(head ? `Headshot  ·  ${score} down` : `Target down  ·  ${score}`);
  };

  // Settle physics before first frame so props rest on surfaces.
  for (let i = 0; i < 30; i++) physics.world.step();

  loadingEl.hidden = true;
  ctaEl.hidden = false;
  if (assetProblems.length) {
    const blocked = policyBlocks.size ? ` The host blocked: ${[...policyBlocks].join(', ')}.` : '';
    noteEl.textContent = `Some files didn't load, so parts of the range will look simpler: ${assetProblems.join('; ')}.${blocked}`;
    noteEl.hidden = false;
  }
  if (DEMO) startEl.hidden = true;
  const start = () => {
    audio.init();
    input.requestLock();
    startEl.hidden = true;
  };
  startEl.addEventListener('click', start);
  startEl.addEventListener('keydown', (e) => {
    if (e.code === 'Enter' || e.code === 'Space') {
      e.preventDefault();
      start();
    }
  });
  document.addEventListener('pointerlockchange', () => {
    // Esc releases the lock: show the menu again.
    if (!input.locked && !input.freeLook) startEl.hidden = DEMO;
  });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape' && input.freeLook) {
      input.freeLook = false;
      startEl.hidden = false;
    }
  });

  const setQuality = (q: Quality) => {
    renderer.setQuality(q);
    hint(`Quality: ${q}`);
  };

  let last = performance.now();
  let time = 0;
  let demoFire = 0;

  const frame = () => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 1 / 20);
    last = now;
    time += dt;

    if (DEMO) {
      input.mouseDX = Math.sin(time * 0.35) * 3;
      input.mouseDY = Math.cos(time * 0.5) * 0.4;
      demoFire -= dt;
      if (demoFire <= 0) {
        input.firePressed = true;
        demoFire = 0.6;
      }
    }

    if (input.keyPressed('Digit1')) setQuality('low');
    if (input.keyPressed('Digit2')) setQuality('medium');
    if (input.keyPressed('Digit3')) setQuality('high');
    if (input.keyPressed('KeyH')) hud.classList.toggle('minimal');
    if (input.keyPressed('KeyT')) targets.resetProps();

    physics.update(dt, (step) => player.fixedUpdate(step, weapon.aiming));
    player.update(dt, time);
    weapon.update(dt, time);
    targets.update(dt, time);
    level.update(dt, time);
    effects.update(dt, canvas.clientHeight);

    // Post: motion blur follows camera angular velocity (screen-space approx).
    const u = renderer.bodycam.uniforms;
    const av = player.angularVel;
    const blurScale = 0.012;
    u.uBlur.value.set(
      THREE.MathUtils.clamp(av.x * blurScale, -0.04, 0.04),
      THREE.MathUtils.clamp(-av.y * blurScale, -0.04, 0.04),
    );
    u.uShutter.value.set(THREE.MathUtils.clamp(-av.x * 0.006, -0.03, 0.03), THREE.MathUtils.clamp(av.y * 0.004, -0.02, 0.02));
    u.uFlash.value = weapon.flashAmount;

    renderer.render(time, dt);
    input.endFrame();

    // HUD
    ammoEl.textContent = weapon.hudText;
    stampEl.textContent = new Date().toISOString().replace('T', ' T').slice(0, 20) + 'Z';
    if (hintTimer > 0) {
      hintTimer -= dt;
      if (hintTimer <= 0) hintEl.classList.remove('show');
    }

    requestAnimationFrame(frame);
  };
  hud.classList.add('minimal');

  // Debug views: ?clean drops the bodycam look, volumetrics and viewmodel and matches the Blender
  // preview camera (16 mm on 36 mm film, 16:9); ?exp= fixes the exposure (Blender's +1 EV is exp=2);
  // ?debug=1..5 shows single shading terms; ?fixedres turns off dynamic resolution.
  if (params.has('exp')) renderer.post.manualExposure = Number(params.get('exp'));
  if (params.has('debug')) levelUniforms.uDebug.value = Number(params.get('debug'));
  if (params.has('fixedres')) renderer.dynamicResolution = false;
  if (params.has('novol')) renderer.post.volumetrics = false;
  if (params.has('clean')) {
    renderer.post.contrast = 0;
    renderer.bodycamEnabled = false;
    renderer.post.bloomStrength = 0;
    renderer.post.volumetrics = false;
    weapon.view.visible = false;
    hud.hidden = true;
    camera.fov = 64.6;
    camera.updateProjectionMatrix();
  }
  requestAnimationFrame(frame);

  // Debug handle for the smoke test / console tinkering.
  (window as unknown as Record<string, unknown>).__game = { scene, camera, player, weapon, physics, renderer, targets, level };
}

main().catch((e) => {
  console.error(e);
  const el = document.getElementById('loading');
  if (!el) return;
  el.className = 'error';
  el.textContent =
    `The range could not start in this browser (${e instanceof Error ? e.message : e}). ` +
    'It needs WebGL 2 and WebAssembly: try a current desktop Chrome, Edge or Firefox, or run it locally with npm run dev.';
});
