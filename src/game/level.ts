import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { Physics, SurfaceKind } from '../engine/physics';
import * as tex from './textures';

/** Scale BoxGeometry UVs so textures keep a constant world-space density. */
function worldUVBox(w: number, h: number, d: number, tile: number) {
  const g = new THREE.BoxGeometry(w, h, d);
  const uv = g.attributes.uv as THREE.BufferAttribute;
  // Face order: +x, -x, +y, -y, +z, -z (4 verts each).
  const dims: [number, number][] = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (let f = 0; f < 6; f++) {
    for (let k = 0; k < 4; k++) {
      const i = f * 4 + k;
      uv.setXY(i, uv.getX(i) * dims[f][0] / tile, uv.getY(i) * dims[f][1] / tile);
    }
  }
  return g;
}

export interface Materials {
  concrete: THREE.MeshStandardMaterial;
  floor: THREE.MeshStandardMaterial;
  plaster: THREE.MeshStandardMaterial;
  wood: THREE.MeshStandardMaterial;
  steel: THREE.MeshStandardMaterial;
  rusty: THREE.MeshStandardMaterial;
  cardboard: THREE.MeshStandardMaterial;
}

export async function makeMaterials(): Promise<Materials> {
  const std = (t: tex.TextureSet, extra: THREE.MeshStandardMaterialParameters = {}) =>
    new THREE.MeshStandardMaterial({ ...t, normalScale: new THREE.Vector2(1, 1), ...extra });
  const [concreteT, floorT, plasterT, woodT, steelT, rustyT, boxT] = await Promise.all([
    tex.loadOrGenerate('concrete_wall', () => tex.concrete(11)),
    tex.loadOrGenerate('concrete_floor', () => tex.concrete(23)),
    tex.loadOrGenerate('plaster', () => tex.paintedPlaster(5)),
    tex.loadOrGenerate('wood', () => tex.woodPlanks(3)),
    tex.loadOrGenerate('steel', () => tex.paintedMetal(4, [0.05, 0.06, 0.07])),
    tex.loadOrGenerate('rusty', () => tex.paintedMetal(9, [0.3, 0.06, 0.04])),
    tex.loadOrGenerate('cardboard', () => tex.cardboard(5)),
  ]);
  return {
    concrete: std(concreteT),
    floor: std(floorT, { normalScale: new THREE.Vector2(0.6, 0.6) }),
    plaster: std(plasterT),
    wood: std(woodT),
    steel: std(steelT, { metalness: 0.75 }),
    rusty: std(rustyT, { metalness: 0.5 }),
    cardboard: std(boxT),
  };
}

export interface Level {
  spawn: THREE.Vector3;
  spawnYaw: number;
  targetSpots: THREE.Vector3[];
  plateSpots: THREE.Vector3[];
  propSpots: THREE.Vector3[];
  update(dt: number, time: number): void;
}

interface FlickerLamp {
  light: THREE.SpotLight;
  bulb: THREE.MeshStandardMaterial;
  base: number;
  phase: number;
  broken: boolean;
}

export function buildLevel(scene: THREE.Scene, physics: Physics, renderer: THREE.WebGLRenderer, m: Materials): Level {
  scene.background = new THREE.Color(0x020203);
  scene.fog = new THREE.FogExp2(0x0b0c0e, 0.028);

  // Very dim image-based lighting so metals and wet spots have something to reflect.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.06;
  scene.add(new THREE.HemisphereLight(0x8090a0, 0x201810, 0.04));

  const box = (
    x: number, y: number, z: number,
    w: number, h: number, d: number,
    mat: THREE.Material, surface: SurfaceKind,
    opts: { tile?: number; rotY?: number; collide?: boolean; parent?: THREE.Object3D } = {},
  ) => {
    const mesh = new THREE.Mesh(worldUVBox(w, h, d, opts.tile ?? 2), mat);
    mesh.position.set(x, y, z);
    if (opts.rotY) mesh.rotation.y = opts.rotY;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    (opts.parent ?? scene).add(mesh);
    if (opts.collide !== false) physics.addStaticBox(mesh, new THREE.Vector3(w, h, d), surface);
    return mesh;
  };

  // --- Shell: 24 x 26 m warehouse, 5 m ceiling --------------------------------
  const X0 = -12, X1 = 12, Z0 = -18, Z1 = 8, H = 5;
  const cx = (X0 + X1) / 2, cz = (Z0 + Z1) / 2, W = X1 - X0, D = Z1 - Z0;
  const floor = box(cx, -0.25, cz, W, 0.5, D, m.floor, 'concrete', { tile: 3 });
  floor.castShadow = false;
  const ceil = box(cx, H + 0.25, cz, W, 0.5, D, m.concrete, 'concrete', { tile: 4 });
  ceil.castShadow = false;
  box(cx, H / 2, Z0 - 0.25, W, H, 0.5, m.plaster, 'plaster', { tile: H });
  box(cx, H / 2, Z1 + 0.25, W, H, 0.5, m.plaster, 'plaster', { tile: H });
  box(X0 - 0.25, H / 2, cz, 0.5, H, D, m.plaster, 'plaster', { tile: H });
  box(X1 + 0.25, H / 2, cz, 0.5, H, D, m.plaster, 'plaster', { tile: H });

  // Roof beams (visual only).
  for (let z = Z0 + 3; z < Z1; z += 5) box(cx, H - 0.25, z, W, 0.4, 0.25, m.steel, 'metal', { collide: false });

  // Concrete columns.
  for (const x of [-6, 6]) for (const z of [-12, -5, 2]) box(x, H / 2, z, 0.6, H, 0.6, m.concrete, 'concrete', { tile: 2 });

  // Floor safety stripes.
  const stripeMat = new THREE.MeshStandardMaterial({ color: 0xb8932a, roughness: 0.85, polygonOffset: true, polygonOffsetFactor: -2 });
  for (const x of [-3.2, 3.2]) {
    const s = new THREE.Mesh(new THREE.PlaneGeometry(0.12, D - 2), stripeMat);
    s.rotation.x = -Math.PI / 2;
    s.position.set(x, 0.002, cz);
    s.receiveShadow = true;
    scene.add(s);
  }

  // --- Office in the back-left corner (drywall partition with a doorway) ------
  const oh = 3;
  box(-8.25, oh / 2, -8, 7.5, oh, 0.15, m.plaster, 'plaster', { tile: oh }); // front wall, left part
  box(-4.55, oh / 2, -10.6, 0.15, oh, 5.2, m.plaster, 'plaster', { tile: oh }); // side wall (to -13.2)
  box(-4.55, oh / 2, -16.3, 0.15, oh, 3.4, m.plaster, 'plaster', { tile: oh }); // side wall after door
  box(-4.55, oh - 0.25, -13.9, 0.15, 0.5, 1.4, m.plaster, 'plaster', { tile: oh }); // lintel
  box(-8.2, oh + 0.05, -13, 7.6, 0.1, 10, m.plaster, 'plaster', { tile: 3 }); // office ceiling
  // Office furniture
  box(-9, 0.75, -15.5, 2, 0.06, 0.9, m.wood, 'wood');
  for (const [lx, lz] of [[-9.9, -15.9], [-8.1, -15.9], [-9.9, -15.1], [-8.1, -15.1]]) box(lx, 0.36, lz, 0.06, 0.72, 0.06, m.steel, 'metal');
  box(-11.6, 0.9, -11.5, 0.6, 1.8, 2.4, m.steel, 'metal'); // filing cabinet bank

  // --- Crates, pallets, barrels, shelving --------------------------------------
  const crate = (x: number, z: number, s = 1.1, y = 0, rot = 0) =>
    box(x, y + s / 2, z, s, s, s, m.wood, 'wood', { tile: s, rotY: rot });
  crate(-9, 4);
  crate(-9, 5.2, 1.1, 0, 0.1);
  crate(-9.1, 4.5, 1.0, 1.1, 0.3);
  crate(8.5, -1.5, 1.2, 0, -0.2);
  crate(9.2, -0.2, 1.0);
  crate(2.2, -8.5, 1.1, 0, 0.5);
  crate(-1.5, -3, 0.9, 0, 0.8);

  // Pallets
  for (const [px, pz, r] of [[4.5, 4.5, 0.2], [-4.5, -1.5, -0.4], [9, -14, 0]]) box(px, 0.07, pz, 1.2, 0.14, 1.0, m.wood, 'wood', { rotY: r });

  // Barrels
  const barrelGeo = new THREE.CylinderGeometry(0.29, 0.29, 0.88, 24);
  const barrel = (x: number, z: number, mat: THREE.Material) => {
    const mesh = new THREE.Mesh(barrelGeo, mat);
    mesh.position.set(x, 0.44, z);
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
    const R = physics.R;
    const body = physics.world.createRigidBody(R.RigidBodyDesc.fixed().setTranslation(x, 0.44, z));
    const col = physics.world.createCollider(R.ColliderDesc.cylinder(0.44, 0.29), body);
    physics.tag(col, { surface: 'metal', mesh });
  };
  barrel(10.4, 6.8, m.rusty);
  barrel(9.8, 6.9, m.steel);
  barrel(10.5, 6.2, m.steel);
  barrel(-10.8, -3.5, m.rusty);
  barrel(5.5, -16.9, m.rusty);

  // Shelving rack along the right wall with room for boxes.
  const shelfX = 11;
  for (const z of [-11, -8.5, -6]) for (const dz of [-0.45, 0.45]) {
    box(shelfX - 0.45, 1.25, z + dz, 0.06, 2.5, 0.06, m.steel, 'metal');
    box(shelfX + 0.45, 1.25, z + dz, 0.06, 2.5, 0.06, m.steel, 'metal');
  }
  for (const y of [0.1, 1.0, 1.9]) box(shelfX, y, -8.5, 1.0, 0.04, 6, m.steel, 'metal', { tile: 1 });

  // Workbench near spawn for the plinking targets.
  box(3.2, 0.9, 0.2, 2.2, 0.08, 0.8, m.wood, 'wood');
  for (const [lx, lz] of [[2.2, -0.1], [4.2, -0.1], [2.2, 0.5], [4.2, 0.5]]) box(lx, 0.43, lz, 0.07, 0.86, 0.07, m.steel, 'metal');

  // Sandbag / berm behind the steel plates.
  box(0, 1.0, -17.2, 10, 2, 1.2, m.wood, 'wood', { tile: 1.2 });

  // --- High strip windows (emissive, moonlit) ----------------------------------
  const winMat = new THREE.MeshStandardMaterial({ color: 0x0, emissive: 0x6d85a8, emissiveIntensity: 0.9 });
  for (let z = Z0 + 2; z < Z1 - 1; z += 4) {
    const w = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 0.7), winMat);
    w.position.set(X1 - 0.001, 4.1, z);
    w.rotation.y = -Math.PI / 2;
    scene.add(w);
    // Faint cold spill from each window.
    const spill = new THREE.SpotLight(0x7890b8, 1.5, 14, 0.9, 0.9, 2);
    spill.position.set(X1 - 0.3, 4.1, z);
    spill.target.position.set(X1 - 5, 0, z + 1);
    scene.add(spill, spill.target);
  }

  // --- Hanging industrial lamps ------------------------------------------------
  const lamps: FlickerLamp[] = [];
  const shadeGeo = new THREE.ConeGeometry(0.42, 0.35, 24, 1, true);
  const bulbGeo = new THREE.SphereGeometry(0.09, 16, 8);
  const cordGeo = new THREE.CylinderGeometry(0.008, 0.008, 1, 6);
  const shadeMat = m.steel.clone();
  shadeMat.side = THREE.DoubleSide;
  const lamp = (x: number, z: number, intensity: number, broken = false, top = H, drop = 1.4) => {
    const y = top - drop;
    const shade = new THREE.Mesh(shadeGeo, shadeMat);
    shade.position.set(x, y + 0.12, z);
    shade.castShadow = true;
    const bulbMat = new THREE.MeshStandardMaterial({ color: 0x111111, emissive: 0xffd9a0, emissiveIntensity: 8 });
    const bulb = new THREE.Mesh(bulbGeo, bulbMat);
    bulb.position.set(x, y, z);
    const cord = new THREE.Mesh(cordGeo, m.steel);
    cord.scale.y = top - y;
    cord.position.set(x, (top + y) / 2, z);
    scene.add(shade, bulb, cord);

    const light = new THREE.SpotLight(0xffd6a0, intensity, 22, Math.PI / 2.6, 0.55, 2);
    light.position.set(x, y - 0.05, z);
    light.target.position.set(x, 0, z);
    light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    light.shadow.bias = -0.0004;
    light.shadow.normalBias = 0.02;
    light.shadow.camera.near = 0.2;
    light.shadow.camera.far = 12;
    scene.add(light, light.target);
    lamps.push({ light, bulb: bulbMat, base: intensity, phase: Math.random() * 100, broken });
  };
  lamp(0, 4, 28);
  lamp(0, -5, 22, true);
  lamp(-8, 1, 16);
  lamp(0, -13.5, 34);
  lamp(-8.2, -11.5, 9, false, oh, 0.5); // office

  // Warm light spill for the office through the doorway is handled by the office lamp.

  const update = (_dt: number, time: number) => {
    for (const l of lamps) {
      if (!l.broken) continue;
      // Fluorescent-style stutter: mostly on, occasional dropouts.
      const t = time + l.phase;
      const n = Math.sin(t * 13.1) * Math.sin(t * 7.7) * Math.sin(t * 2.3);
      const on = n > -0.35 ? 1 : 0.05 + Math.random() * 0.2;
      l.light.intensity = l.base * on;
      l.bulb.emissiveIntensity = 8 * on;
    }
  };

  return {
    spawn: new THREE.Vector3(0, 0.1, 6),
    spawnYaw: 0,
    targetSpots: [
      new THREE.Vector3(-2.5, 0, -11),
      new THREE.Vector3(3.5, 0, -13),
      new THREE.Vector3(8, 0, -3.5),
      new THREE.Vector3(-8.5, 0, -13), // inside the office
    ],
    plateSpots: [new THREE.Vector3(-3, 0, -16), new THREE.Vector3(0, 0, -16), new THREE.Vector3(3, 0, -16)],
    propSpots: [
      new THREE.Vector3(2.5, 0.95, 0.2),
      new THREE.Vector3(3.0, 0.95, 0.3),
      new THREE.Vector3(3.5, 0.95, 0.1),
      new THREE.Vector3(3.9, 0.95, 0.3),
      new THREE.Vector3(shelfX, 1.05, -9.5),
      new THREE.Vector3(shelfX, 1.05, -7.5),
      new THREE.Vector3(shelfX, 1.95, -8.8),
      new THREE.Vector3(shelfX, 0.15, -6.5),
    ],
    update,
  };
}
