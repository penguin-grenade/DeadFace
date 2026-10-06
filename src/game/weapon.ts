import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import { GROUP_DEBRIS, type HitInfo, type Physics } from '../engine/physics';
import { assetManifest, loadGLTF, reportAssetProblem } from '../engine/assets';
import type { Audio } from '../engine/audio';
import type { Input } from '../engine/input';
import type { Player } from './player';
import { ArmRig } from './arms';
import { WeaponSway } from './sway';
import type { Effects } from './effects';
import { flashSprite } from './textures';

const MAG_SIZE = 15;
const FIRE_INTERVAL = 0.11;
const RELOAD_TIME = 1.9;

interface Casing {
  body: RAPIER.RigidBody;
  mesh: THREE.Mesh;
  age: number;
  clinked: boolean;
}

/** Procedural polymer-frame pistol used when public/models/pistol.glb is absent. */
function buildProceduralPistol() {
  const g = new THREE.Group();
  const polymer = new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.7, metalness: 0.05 });
  const steel = new THREE.MeshStandardMaterial({ color: 0x1d1e20, roughness: 0.35, metalness: 0.9 });
  const add = (geo: THREE.BufferGeometry, mat: THREE.Material, x: number, y: number, z: number, parent: THREE.Object3D = g) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    m.receiveShadow = true;
    parent.add(m);
    return m;
  };
  // Frame + grip (grip raked back ~18 degrees)
  add(new THREE.BoxGeometry(0.03, 0.022, 0.16), polymer, 0, 0.0, -0.06);
  const grip = add(new THREE.BoxGeometry(0.029, 0.11, 0.048), polymer, 0, -0.055, 0.005);
  grip.rotation.x = -0.32;
  add(new THREE.BoxGeometry(0.028, 0.012, 0.05), polymer, 0, -0.018, -0.03); // trigger guard bottom
  const trig = add(new THREE.BoxGeometry(0.006, 0.02, 0.006), steel, 0, -0.01, -0.035);
  trig.rotation.x = 0.3;
  // Slide
  const slide = new THREE.Group();
  slide.name = 'Slide';
  g.add(slide);
  add(new THREE.BoxGeometry(0.029, 0.03, 0.19), steel, 0, 0.026, -0.07, slide);
  for (let i = 0; i < 6; i++) add(new THREE.BoxGeometry(0.03, 0.024, 0.003), polymer, 0, 0.027, 0.005 + i * -0.006, slide); // serrations
  add(new THREE.BoxGeometry(0.004, 0.006, 0.004), steel, 0, 0.044, -0.155, slide); // front sight
  add(new THREE.BoxGeometry(0.02, 0.006, 0.005), steel, 0, 0.044, 0.015, slide); // rear sight
  add(new THREE.CylinderGeometry(0.0055, 0.0055, 0.01, 12).rotateX(Math.PI / 2), steel, 0, 0.026, -0.166, slide); // barrel
  // Weapon-mounted light under the rail
  add(new THREE.BoxGeometry(0.03, 0.028, 0.06), polymer, 0, -0.022, -0.12);
  const lens = new THREE.Mesh(new THREE.CircleGeometry(0.012, 16), new THREE.MeshStandardMaterial({ color: 0x222222, emissive: 0xfff4e0, emissiveIntensity: 3 }));
  lens.name = 'LightLens';
  lens.position.set(0, -0.022, -0.1505);
  lens.rotation.y = Math.PI;
  g.add(lens);

  const muzzle = new THREE.Object3D();
  muzzle.name = 'Muzzle';
  muzzle.position.set(0, 0.026, -0.175);
  g.add(muzzle);
  const eject = new THREE.Object3D();
  eject.name = 'Ejection';
  eject.position.set(0.015, 0.035, -0.04);
  g.add(eject);
  const lightMount = new THREE.Object3D();
  lightMount.name = 'LightMount';
  lightMount.position.set(0, -0.022, -0.155);
  g.add(lightMount);
  return g;
}

/** Simple gloved hands + sleeves so the gun isn't floating. */
function buildArms() {
  const g = new THREE.Group();
  const glove = new THREE.MeshStandardMaterial({ color: 0x1b1a18, roughness: 0.85 });
  const sleeve = new THREE.MeshStandardMaterial({ color: 0x1f2630, roughness: 0.95 });
  const mk = (geo: THREE.BufferGeometry, mat: THREE.Material) => {
    const m = new THREE.Mesh(geo, mat);
    m.receiveShadow = true;
    g.add(m);
    return m;
  };
  // Firing hand wraps the grip.
  const rh = mk(new THREE.BoxGeometry(0.045, 0.075, 0.06), glove);
  rh.position.set(0.012, -0.058, 0.012);
  rh.rotation.x = -0.32;
  // Support hand cups from the left.
  const lh = mk(new THREE.BoxGeometry(0.04, 0.07, 0.07), glove);
  lh.position.set(-0.024, -0.06, 0.0);
  lh.rotation.set(-0.32, 0, 0.35);
  // Forearms running back and down out of frame.
  const armGeo = new THREE.CylinderGeometry(0.038, 0.048, 0.42, 12);
  const ra = mk(armGeo, sleeve);
  ra.position.set(0.09, -0.16, 0.2);
  ra.rotation.set(1.05, 0, -0.45);
  const la = mk(armGeo, sleeve);
  la.position.set(-0.11, -0.16, 0.2);
  la.rotation.set(1.05, 0, 0.5);
  return g;
}

export class Weapon {
  readonly view = new THREE.Group(); // camera-space pivot
  private model!: THREE.Object3D;
  private slide?: THREE.Object3D;
  private muzzle!: THREE.Object3D;
  private eject!: THREE.Object3D;
  private flash: THREE.Mesh;
  private flashLight: THREE.PointLight;
  readonly flashlight: THREE.SpotLight;
  private flashTime = 0;
  private ray = new THREE.Raycaster();
  private rig: ArmRig | null = null;

  ammo = MAG_SIZE;
  reserve = 45;
  private cooldown = 0;
  private reloadT = -1;
  /** 0 at the hip, 1 aimed down the sights. */
  aim = 0;
  private kickZ = 0;
  private kickVel = 0;
  private kickRot = 0;
  private kickRotVel = 0;
  private slideT = 1;
  private wallBlock = 0;
  private sway = new WeaponSway();
  private localVel = new THREE.Vector3();
  /** Smoothed pose position before sway is added on top. */
  private basePos = new THREE.Vector3(0.075, -0.15, -0.37);
  private casings: Casing[] = [];
  private casingGeo = new THREE.CylinderGeometry(0.0048, 0.0048, 0.019, 10);
  private brass = new THREE.MeshStandardMaterial({ color: 0xb48a3c, metalness: 1, roughness: 0.3 });
  private reloadStage = 0;

  onShot?: (hit: ReturnType<Physics['raycast']>, dir: THREE.Vector3) => void;

  constructor(
    private scene: THREE.Scene,
    private physics: Physics,
    private audio: Audio,
    private input: Input,
    private player: Player,
    private effects: Effects,
  ) {
    player.camera.add(this.view);

    const flashMat = new THREE.MeshBasicMaterial({ map: flashSprite(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, color: 0xffffff });
    this.flash = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.16), flashMat);
    this.flash.visible = false;
    this.flashLight = new THREE.PointLight(0xffb060, 0, 9, 2);

    this.flashlight = new THREE.SpotLight(0xfff2e0, 9, 30, 0.38, 0.5, 2);
    this.flashlight.castShadow = true;
    this.flashlight.shadow.mapSize.set(1024, 1024);
    this.flashlight.shadow.camera.near = 0.1;
    this.flashlight.shadow.bias = -0.0003;
    this.flashlight.shadow.normalBias = 0.02;
  }

  async load() {
    const { models } = await assetManifest();
    const load = async (name: string) => {
      if (!models.includes(name)) return null;
      try {
        const gltf = await loadGLTF(`models/${name}.glb`);
        gltf.scene.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) {
            // No full body to go with it, so the viewmodel casts no shadow (it also keeps the lamps' cached shadows static).
            o.castShadow = false;
            o.receiveShadow = true;
          }
        });
        return gltf.scene;
      } catch (e) {
        reportAssetProblem(name, e);
        return null;
      }
    };
    const [model, arms] = await Promise.all([load('pistol'), load('arms')]);
    this.model = model ?? buildProceduralPistol();
    this.slide = this.model.getObjectByName('Slide') ?? undefined;
    this.muzzle = this.model.getObjectByName('Muzzle') ?? this.model;
    this.eject = this.model.getObjectByName('Ejection') ?? this.model;
    const mount = this.model.getObjectByName('LightMount') ?? this.muzzle;

    this.view.add(this.model);
    this.view.add(arms ?? buildArms());
    this.rig = arms ? ArmRig.from(arms) : null;
    this.muzzle.add(this.flash);
    this.muzzle.add(this.flashLight);
    this.flashLight.position.set(0, 0, -0.05);
    mount.add(this.flashlight);
    const tgt = new THREE.Object3D();
    tgt.position.set(0, 0, -5);
    mount.add(tgt);
    this.flashlight.target = tgt;
  }

  get reloading() {
    return this.reloadT >= 0;
  }

  get aiming() {
    return this.aim > 0.5;
  }

  private startReload() {
    if (this.reloading || this.ammo === MAG_SIZE || this.reserve === 0) return;
    this.reloadT = 0;
    this.reloadStage = 0;
  }

  private fire() {
    if (this.reloading) return;
    if (this.ammo <= 0) {
      this.audio.dryFire();
      this.cooldown = 0.25;
      return;
    }
    this.ammo--;
    this.cooldown = FIRE_INTERVAL;

    const cam = this.player.camera;
    const origin = cam.getWorldPosition(new THREE.Vector3());
    const dir = this.player.forward;
    const spread = THREE.MathUtils.lerp(0.014, 0.0025, this.aim) + this.player.moveAmount * 0.02;
    dir.x += (Math.random() - 0.5) * spread;
    dir.y += (Math.random() - 0.5) * spread;
    dir.z += (Math.random() - 0.5) * spread;
    dir.normalize();

    const hit = this.trace(origin, dir);
    if (hit) {
      const surface = hit.tag?.surface ?? 'concrete';
      const body = hit.collider.parent();
      const movable = body && !body.isFixed();
      this.effects.impact(hit.point, hit.normal, surface, dir, movable ? hit.tag?.mesh : undefined);
      this.audio.impact(surface === 'metal' ? 'metal' : surface === 'wood' || surface === 'cardboard' || surface === 'rubber' || surface === 'fibreglass' ? 'wood' : surface === 'flesh' || surface === 'fabric' ? 'flesh' : 'concrete', hit.distance);
      hit.tag?.onHit?.(hit, dir);
      if (body && body.isDynamic()) {
        body.applyImpulseAtPoint(dir.clone().multiplyScalar(2.2), hit.point, true);
      }
    }
    this.onShot?.(hit, dir);

    // Feedback
    this.audio.gunshot();
    const kick = THREE.MathUtils.lerp(1, 0.6, this.aim);
    this.player.kick(0.028 * kick + Math.random() * 0.01, (Math.random() - 0.5) * 0.02 * kick);
    this.kickVel += 1.6 * kick;
    this.kickRotVel += 14 * kick;
    this.slideT = 0;
    this.flashTime = 0.05;
    this.flash.rotation.z = Math.random() * Math.PI;
    this.flash.scale.setScalar(0.7 + Math.random() * 0.6);
    this.effects.muzzleSmoke(this.muzzle.getWorldPosition(new THREE.Vector3()).addScaledVector(dir, 0.12), dir);
    this.ejectCasing();
  }

  /**
   * The bullet's path. Targets' colliders only approximate their shape, so a hit on one
   * ('precise' tag) is checked against the visible mesh: it moves onto the surface, or the
   * bullet carries on past (between the legs, beside an arm).
   */
  private trace(origin: THREE.Vector3, dir: THREE.Vector3): HitInfo | null {
    const passed = new Set<number>();
    const filter = (c: RAPIER.Collider) => !passed.has(c.parent()?.handle ?? -1);
    for (let i = 0; i < 4; i++) {
      const hit = this.physics.raycast(origin, dir, 120, this.player.collider, passed.size ? filter : undefined);
      if (!hit?.tag?.precise || !hit.tag.mesh) return hit;
      this.ray.set(origin, dir);
      this.ray.near = Math.max(0, hit.distance - 0.8);
      this.ray.far = hit.distance + 0.8;
      // Decals parented to the target are planes; only the target's own surface counts.
      const vis = this.ray.intersectObject(hit.tag.mesh, true).find((v) => (v.object as THREE.Mesh).geometry?.type !== 'PlaneGeometry');
      if (vis) {
        hit.point.copy(vis.point);
        hit.distance = vis.distance;
        if (vis.face) hit.normal.copy(vis.face.normal).transformDirection(vis.object.matrixWorld);
        return hit;
      }
      const body = hit.collider.parent();
      if (!body) return hit;
      passed.add(body.handle);
    }
    return null;
  }

  private ejectCasing() {
    const R = this.physics.R;
    const p = this.eject.getWorldPosition(new THREE.Vector3());
    const q = this.player.camera.getWorldQuaternion(new THREE.Quaternion());
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    const v = right.multiplyScalar(2.2 + Math.random()).add(up.multiplyScalar(1.8 + Math.random())).add(back.multiplyScalar(0.3));
    v.add(this.player.velocity);
    const body = this.physics.world.createRigidBody(
      R.RigidBodyDesc.dynamic().setTranslation(p.x, p.y, p.z).setLinvel(v.x, v.y, v.z).setAngvel({ x: Math.random() * 30, y: 20, z: Math.random() * 30 }).setCcdEnabled(true),
    );
    // Casings are tiny; collide with world but not with the player capsule.
    const col = this.physics.world.createCollider(R.ColliderDesc.cylinder(0.0095, 0.0048).setDensity(8000).setRestitution(0.4).setFriction(0.6), body);
    col.setCollisionGroups(GROUP_DEBRIS);
    const mesh = new THREE.Mesh(this.casingGeo, this.brass);
    this.scene.add(mesh);
    this.physics.sync(body, mesh);
    this.casings.push({ body, mesh, age: 0, clinked: false });
    if (this.casings.length > 30) this.removeCasing(this.casings.shift()!);
  }

  private removeCasing(c: Casing) {
    this.physics.unsync(c.body);
    this.physics.world.removeRigidBody(c.body);
    c.mesh.removeFromParent();
  }

  update(dt: number, time: number) {
    const i = this.input;
    this.cooldown -= dt;

    if (i.keyPressed('KeyR')) this.startReload();
    if (i.keyPressed('KeyF')) this.flashlight.visible = !this.flashlight.visible;
    if (i.firePressed && this.cooldown <= 0 && !this.player.sprinting) this.fire();
    else if (i.firePressed && this.ammo === 0 && !this.reloading) this.startReload();

    const wantAim = i.aim && !this.reloading && !this.player.sprinting;
    this.aim += ((wantAim ? 1 : 0) - this.aim) * Math.min(1, 12 * dt);

    // Reload timeline
    let reloadPose = 0;
    if (this.reloading) {
      this.reloadT += dt;
      const t = this.reloadT / RELOAD_TIME;
      reloadPose = Math.sin(Math.min(1, t) * Math.PI);
      if (this.reloadStage === 0 && t > 0.15) { this.audio.reload('out'); this.reloadStage++; }
      if (this.reloadStage === 1 && t > 0.55) { this.audio.reload('in'); this.reloadStage++; }
      if (this.reloadStage === 2 && t > 0.8) { this.audio.reload('slide'); this.reloadStage++; this.slideT = 0; }
      if (t >= 1) {
        const need = MAG_SIZE - this.ammo;
        const take = Math.min(need, this.reserve);
        this.ammo += take;
        this.reserve -= take;
        this.reloadT = -1;
      }
    }

    // Wall proximity: raise the muzzle instead of clipping into geometry.
    const origin = this.player.camera.getWorldPosition(new THREE.Vector3());
    const wallHit = this.physics.raycast(origin, this.player.forward, 0.85, this.player.collider);
    const block = wallHit ? 1 - Math.max(0, (wallHit.distance - 0.3) / 0.55) : 0;
    this.wallBlock += (block - this.wallBlock) * Math.min(1, 10 * dt);

    // Recoil springs
    this.kickVel += (-this.kickZ * 400 - this.kickVel * 28) * dt;
    this.kickZ += this.kickVel * dt;
    this.kickRotVel += (-this.kickRot * 380 - this.kickRotVel * 24) * dt;
    this.kickRot += this.kickRotVel * dt;

    // Inertia: the gun trails turns and walking starts, swings past and settles (sway.ts).
    const av = this.player.angularVel;
    const v = this.player.velocity;
    const yaw = this.player.root.rotation.y;
    this.localVel.set(
      v.x * Math.cos(yaw) - v.z * Math.sin(yaw),
      this.player.verticalSpeed,
      v.x * Math.sin(yaw) + v.z * Math.cos(yaw),
    );
    const sway = this.sway;
    sway.update(dt, time, av.x, av.y, this.localVel, this.aim, this.player.motionScale);

    // Hip: compressed low ready, right of centre. ADS: sight line on the camera axis (front post
    // top is 48.8 mm above the pistol origin), arms extended so both forearms rise from the frame edge.
    const hip = new THREE.Vector3(0.075, -0.15, -0.37);
    const ads = new THREE.Vector3(0, -0.0488, -0.5);
    const pos = hip.lerp(ads, this.aim);
    const bob = this.player.bob;
    const sprint = this.player.sprinting ? 1 : 0;
    pos.x += bob.x * 0.6 * (1 - this.aim * 0.8);
    pos.y += bob.y * 0.8 * (1 - this.aim * 0.8) - reloadPose * 0.06 - sprint * 0.04;
    pos.z += this.kickZ * 0.05 + this.wallBlock * 0.1;
    this.basePos.lerp(pos, Math.min(1, 25 * dt));
    this.view.position.copy(this.basePos);
    // The trail turns the gun about the arms rather than its own grip: it swings out to the side
    // it lags on (and drops when looking up), as hands held out from the chest do.
    const PIVOT = 0.28;
    this.view.position.x += -Math.sin(sway.trailYaw) * PIVOT + sway.shift.x;
    this.view.position.y += Math.sin(sway.trailPitch) * PIVOT + sway.shift.y;
    this.view.position.z += sway.shift.z;
    // Hip pose is slightly canted so the side of the slide and both hands read on camera.
    const hipCant = 1 - this.aim;
    this.view.rotation.set(
      this.kickRot * 0.04 + this.wallBlock * 0.9 + reloadPose * -0.5 - sprint * 0.35 + hipCant * 0.02 + sway.trailPitch,
      sway.trailYaw + sprint * 0.5 + reloadPose * 0.4 + hipCant * 0.14,
      reloadPose * 0.6 + sprint * 0.2 - hipCant * 0.1 + sway.cant,
      'YXZ',
    );
    this.rig?.update(this.player.camera);

    if (this.slide) {
      this.slideT = Math.min(1, this.slideT + dt / 0.07);
      this.slide.position.z = Math.sin(this.slideT * Math.PI) * 0.03;
    }

    this.flashTime -= dt;
    this.flash.visible = this.flashTime > 0;
    this.flashLight.intensity = this.flashTime > 0 ? 25 : 0;

    // Casings: tinkle on first bounce, despawn after a while.
    for (const c of this.casings) {
      c.age += dt;
      if (!c.clinked && c.age > 0.1 && c.body.linvel().y > -0.5 && c.body.translation().y < 0.4) {
        c.clinked = true;
        this.audio.shellCasing();
      }
    }
    while (this.casings.length && this.casings[0].age > 20) this.removeCasing(this.casings.shift()!);
  }

  get hudText() {
    return this.reloading ? 'RELOADING' : `${this.ammo} / ${this.reserve}`;
  }

  /** Muzzle flash intensity for the post shader (brief exposure bump). */
  get flashAmount() {
    return Math.max(0, this.flashTime / 0.05);
  }

  addReserve(n: number) {
    this.reserve += n;
  }
}
