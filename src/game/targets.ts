import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import { GROUP_PROP, GROUP_TARGET, type HitInfo, type Physics, type SurfaceKind } from '../engine/physics';
import type { Materials } from './level';

/** Optional authored assets from the Blender pipeline, keyed by file stem. */
export type ModelLibrary = Partial<Record<'mannequin' | 'steel_target' | 'can' | 'box_small', THREE.Object3D>>;

interface Mannequin {
  body: RAPIER.RigidBody;
  group: THREE.Group;
  home: THREE.Vector3;
  health: number;
  down: boolean;
  downTime: number;
  facing: number;
  track: number; // metres of side-to-side travel, 0 = static
  phase: number;
}

interface Plate {
  pivot: THREE.Object3D;
  body: RAPIER.RigidBody;
  angle: number;
  vel: number;
  hinge: THREE.Vector3;
}

interface Prop {
  body: RAPIER.RigidBody;
  mesh: THREE.Object3D;
  home: THREE.Vector3;
}

export class Targets {
  private mannequins: Mannequin[] = [];
  private plates: Plate[] = [];
  private props: Prop[] = [];
  onKill?: (headshot: boolean) => void;
  onHitMarker?: () => void;

  constructor(
    private scene: THREE.Scene,
    private physics: Physics,
    private mats: Materials,
    private models: ModelLibrary,
  ) {}

  // --- Mannequins -------------------------------------------------------------
  addMannequin(at: THREE.Vector3, facing: number, track = 0) {
    const R = this.physics.R;
    const w = this.physics.world;
    const body = w.createRigidBody(
      R.RigidBodyDesc.kinematicPositionBased()
        .setTranslation(at.x, at.y, at.z)
        .setRotation(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, facing, 0))),
    );
    const group = new THREE.Group();
    const cloth = new THREE.MeshStandardMaterial({ color: 0x3b3d33, roughness: 0.95 });
    const skin = new THREE.MeshStandardMaterial({ color: 0xb9a58c, roughness: 0.55 });
    const vest = new THREE.MeshStandardMaterial({ color: 0x23262a, roughness: 0.8 });

    if (this.models.mannequin) {
      group.add(this.models.mannequin.clone());
    } else {
      const part = (geo: THREE.BufferGeometry, mat: THREE.Material, x: number, y: number, z: number, rx = 0, rz = 0) => {
        const m = new THREE.Mesh(geo, mat);
        m.position.set(x, y, z);
        m.rotation.set(rx, 0, rz);
        m.castShadow = m.receiveShadow = true;
        group.add(m);
        return m;
      };
      const limb = new THREE.CapsuleGeometry(0.07, 0.36, 6, 12);
      part(limb, cloth, -0.11, 0.3, 0);
      part(limb, cloth, 0.11, 0.3, 0);
      part(new THREE.CapsuleGeometry(0.075, 0.34, 6, 12), cloth, -0.11, 0.72, 0);
      part(new THREE.CapsuleGeometry(0.075, 0.34, 6, 12), cloth, 0.11, 0.72, 0);
      part(new THREE.CapsuleGeometry(0.17, 0.12, 8, 16), cloth, 0, 1.0, 0); // hips
      part(new THREE.CapsuleGeometry(0.19, 0.32, 8, 16), vest, 0, 1.3, 0); // torso
      part(new THREE.CapsuleGeometry(0.055, 0.5, 6, 12), cloth, -0.26, 1.2, 0, 0, 0.18);
      part(new THREE.CapsuleGeometry(0.055, 0.5, 6, 12), cloth, 0.26, 1.2, 0, 0, -0.18);
      part(new THREE.CylinderGeometry(0.05, 0.06, 0.1, 12), skin, 0, 1.6, 0);
      const head = part(new THREE.SphereGeometry(0.11, 24, 16), skin, 0, 1.73, 0);
      head.scale.set(0.9, 1.1, 1);
    }
    this.scene.add(group);

    const m: Mannequin = { body, group, home: at.clone(), health: 3, down: false, downTime: 0, facing, track, phase: Math.random() * 6 };
    const onHit = (hit: HitInfo, dir: THREE.Vector3) => this.hitMannequin(m, hit, dir);
    // One collider per body part (the figure faces +z, its stand plate under the feet), so shots
    // between the legs or past an arm carry on; bullets then confirm hits against the mesh.
    const tilt = (a: number) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), a);
    const parts: [RAPIER.ColliderDesc, SurfaceKind, string | null][] = [
      [R.ColliderDesc.ball(0.108).setTranslation(0, 1.747, 0.005), 'fibreglass', 'head'],
      [R.ColliderDesc.capsule(0.02, 0.055).setTranslation(0, 1.587, -0.016), 'fibreglass', 'torso'],
      [R.ColliderDesc.cuboid(0.175, 0.225, 0.1825).setTranslation(0, 1.295, 0.0075), 'fabric', 'torso'],
      [R.ColliderDesc.cuboid(0.175, 0.1, 0.13).setTranslation(0, 0.98, -0.02), 'fabric', 'torso'],
      [R.ColliderDesc.cuboid(0.24, 0.006, 0.21).setTranslation(0, 0.006, 0.045), 'metal', null],
    ];
    for (const s of [-1, 1]) {
      parts.push(
        [R.ColliderDesc.capsule(0.2325, 0.05).setTranslation(s * 0.229, 1.183, 0.006).setRotation(tilt(s * 0.1456)), 'fibreglass', 'arm'],
        [R.ColliderDesc.capsule(0.045, 0.045).setTranslation(s * 0.278, 0.817, 0.042), 'fibreglass', 'arm'],
        [R.ColliderDesc.capsule(0.14, 0.085).setTranslation(s * 0.098, 0.727, 0.0), 'fabric', 'legs'],
        [R.ColliderDesc.capsule(0.145, 0.065).setTranslation(s * 0.107, 0.307, 0.006), 'fabric', 'legs'],
        [R.ColliderDesc.cuboid(0.05, 0.04, 0.135).setTranslation(s * 0.13, 0.052, 0.056), 'fibreglass', 'legs'],
      );
    }
    for (const [desc, surface, part] of parts) {
      const col = w.createCollider(desc.setDensity(300).setCollisionGroups(GROUP_TARGET), body);
      // The stand rings like steel but a shot to it doesn't count against the target.
      this.physics.tag(col, part ? { surface, mesh: group, onHit, part, precise: true } : { surface, mesh: group, precise: true });
    }
    this.physics.sync(body, group);
    this.mannequins.push(m);
  }

  private hitMannequin(m: Mannequin, hit: HitInfo, dir: THREE.Vector3) {
    this.onHitMarker?.();
    if (m.down) return;
    const headshot = hit.tag?.part === 'head';
    m.health -= headshot ? 3 : hit.tag?.part === 'torso' ? 1.5 : 1;
    if (m.health > 0) return;
    m.down = true;
    m.downTime = 0;
    m.body.setBodyType(this.physics.R.RigidBodyType.Dynamic, true);
    m.body.setLinearDamping(0.3);
    m.body.setAngularDamping(0.6);
    m.body.applyImpulseAtPoint(dir.clone().multiplyScalar(90), hit.point, true);
    this.onKill?.(headshot);
  }

  // --- Steel plates -------------------------------------------------------------
  addPlate(at: THREE.Vector3) {
    const R = this.physics.R;
    const w = this.physics.world;
    const standMat = this.mats.steel;
    // Stand: two uprights and a crossbar.
    const stand = new THREE.Group();
    const post = new THREE.BoxGeometry(0.05, 1.5, 0.05);
    for (const x of [-0.35, 0.35]) {
      const p = new THREE.Mesh(post, standMat);
      p.position.set(x, 0.75, 0);
      p.castShadow = true;
      stand.add(p);
    }
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.05, 0.05), standMat);
    bar.position.set(0, 1.5, 0);
    bar.castShadow = true;
    stand.add(bar);
    stand.position.copy(at);
    this.scene.add(stand);

    // Plate hangs from the bar on chains; pivot at the bar.
    const pivot = new THREE.Object3D();
    pivot.position.set(at.x, at.y + 1.5, at.z);
    this.scene.add(pivot);
    const plateMat = new THREE.MeshStandardMaterial({ color: 0xd8d6cc, roughness: 0.55, metalness: 0.4 });
    const plate = this.models.steel_target
      ? this.models.steel_target.clone()
      : new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.012, 32).rotateX(Math.PI / 2), plateMat);
    plate.position.set(0, -0.45, 0);
    plate.castShadow = true;
    pivot.add(plate);
    const chainGeo = new THREE.CylinderGeometry(0.004, 0.004, 0.24, 4);
    for (const x of [-0.12, 0.12]) {
      const c = new THREE.Mesh(chainGeo, standMat);
      c.position.set(x, -0.12, 0);
      pivot.add(c);
    }

    const body = w.createRigidBody(R.RigidBodyDesc.kinematicPositionBased().setTranslation(at.x, at.y + 1.05, at.z));
    const col = w.createCollider(R.ColliderDesc.cylinder(0.006, 0.22).setCollisionGroups(GROUP_TARGET), body);
    const p: Plate = { pivot, body, angle: 0, vel: 0, hinge: pivot.position.clone() };
    this.physics.tag(col, {
      surface: 'metal',
      mesh: plate,
      onHit: (_hit, dir) => {
        p.vel += Math.sign(-dir.z || 1) * -7;
        this.onHitMarker?.();
      },
    });
    this.plates.push(p);
  }

  // --- Loose props ------------------------------------------------------------
  addCan(at: THREE.Vector3) {
    const R = this.physics.R;
    const mat = new THREE.MeshStandardMaterial({ color: [0xa02020, 0x1f5fa0, 0xd0c040][Math.floor(Math.random() * 3)], metalness: 0.8, roughness: 0.35 });
    const mesh = this.models.can?.clone() ?? new THREE.Mesh(new THREE.CylinderGeometry(0.033, 0.033, 0.122, 16), mat);
    mesh.castShadow = true;
    this.scene.add(mesh);
    const body = this.physics.world.createRigidBody(R.RigidBodyDesc.dynamic().setTranslation(at.x, at.y + 0.061, at.z).setCcdEnabled(true));
    const col = this.physics.world.createCollider(R.ColliderDesc.cylinder(0.061, 0.033).setDensity(400).setRestitution(0.3).setCollisionGroups(GROUP_PROP), body);
    this.physics.tag(col, { surface: 'metal', mesh });
    this.physics.sync(body, mesh);
    this.props.push({ body, mesh, home: at.clone().setY(at.y + 0.061) });
  }

  addBox(at: THREE.Vector3, size = 0.35) {
    const R = this.physics.R;
    const mesh = this.models.box_small?.clone() ?? new THREE.Mesh(new THREE.BoxGeometry(size, size * 0.8, size), this.mats.cardboard);
    mesh.castShadow = mesh.receiveShadow = true;
    this.scene.add(mesh);
    const body = this.physics.world.createRigidBody(R.RigidBodyDesc.dynamic().setTranslation(at.x, at.y + size * 0.4, at.z).setRotation(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.random(), 0))));
    const col = this.physics.world.createCollider(R.ColliderDesc.cuboid(size / 2, size * 0.4, size / 2).setDensity(120).setCollisionGroups(GROUP_PROP), body);
    this.physics.tag(col, { surface: 'cardboard', mesh });
    this.physics.sync(body, mesh);
    this.props.push({ body, mesh, home: at.clone().setY(at.y + size * 0.4) });
  }

  /** Everything that moves, with a rough radius (for shadow/ambient tracking). */
  movingObjects(): { object: THREE.Object3D; radius: number }[] {
    return [
      ...this.mannequins.map((m) => ({ object: m.group as THREE.Object3D, radius: 1.2 })),
      ...this.plates.map((p) => ({ object: p.pivot, radius: 0.5 })),
      ...this.props.map((p) => ({ object: p.mesh, radius: 0.3 })),
    ];
  }

  resetProps() {
    for (const p of this.props) {
      p.body.setTranslation(p.home, true);
      p.body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      p.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      p.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    }
  }

  update(dt: number, time: number) {
    for (const m of this.mannequins) {
      if (m.down) {
        m.downTime += dt;
        if (m.downTime > 7) {
          // Stand back up.
          m.down = false;
          m.health = 3;
          m.body.setBodyType(this.physics.R.RigidBodyType.KinematicPositionBased, true);
          m.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
          m.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
          m.body.setTranslation(m.home, true);
          m.body.setRotation(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, m.facing, 0)), true);
          m.group.traverse((o) => {
            // Drop decals so the respawned target is clean.
            if ((o as THREE.Mesh).isMesh && (o as THREE.Mesh).geometry.type === 'PlaneGeometry') o.removeFromParent();
          });
        }
        continue;
      }
      if (m.track > 0) {
        const off = Math.sin(time * 0.6 + m.phase) * m.track;
        const q = m.body.rotation();
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(new THREE.Quaternion(q.x, q.y, q.z, q.w));
        const p = m.home.clone().addScaledVector(right, off);
        m.body.setNextKinematicTranslation(p);
      }
    }

    for (const p of this.plates) {
      // Pendulum
      p.vel += (-p.angle * 55 - p.vel * 1.2) * dt;
      p.angle += p.vel * dt;
      p.pivot.rotation.x = p.angle;
      const c = new THREE.Vector3(0, -0.45, 0).applyAxisAngle(new THREE.Vector3(1, 0, 0), p.angle).add(p.hinge);
      p.body.setNextKinematicTranslation(c);
      p.body.setNextKinematicRotation(new THREE.Quaternion().setFromEuler(new THREE.Euler(p.angle + Math.PI / 2, 0, 0)));
    }

    // Anything that fell out of the world goes home.
    for (const p of this.props) if (p.body.translation().y < -5) p.body.setTranslation(p.home, true);
  }
}
