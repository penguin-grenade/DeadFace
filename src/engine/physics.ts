import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';

export type SurfaceKind = 'metal' | 'concrete' | 'wood' | 'flesh' | 'plaster' | 'cardboard' | 'glass';

export interface HitInfo {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  distance: number;
  collider: RAPIER.Collider;
  tag?: ColliderTag;
}

/** Game-side data attached to a collider so a raycast hit knows what it struck. */
export interface ColliderTag {
  surface: SurfaceKind;
  /** Mesh bullet decals should parent to (so they move with dynamic props). */
  mesh?: THREE.Object3D;
  /** Optional hit handler, e.g. a target reacting to damage. */
  onHit?: (hit: HitInfo, dir: THREE.Vector3) => void;
  part?: string;
}

/**
 * Collision groups: (membership << 16) | filter.
 * Bit 0 = player, bit 1 = debris (shell casings). Debris ignores the player,
 * and bullets/character queries ignore debris.
 */
export const GROUP_PLAYER = (0x0001 << 16) | 0xfffd;
export const GROUP_DEBRIS = (0x0002 << 16) | 0xfffe;
export const QUERY_NO_DEBRIS = (0x0001 << 16) | 0xfffd;

interface Synced {
  body: RAPIER.RigidBody;
  object: THREE.Object3D;
}

export class Physics {
  world!: RAPIER.World;
  readonly tags = new Map<number, ColliderTag>();
  private synced: Synced[] = [];
  private acc = 0;
  readonly step = 1 / 60;

  async init() {
    await RAPIER.init();
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    this.world.timestep = this.step;
  }

  get R() {
    return RAPIER;
  }

  tag(collider: RAPIER.Collider, tag: ColliderTag) {
    this.tags.set(collider.handle, tag);
  }

  /** Static box collider matching an axis-aligned or rotated mesh. */
  addStaticBox(mesh: THREE.Mesh, size: THREE.Vector3, surface: SurfaceKind) {
    mesh.updateMatrixWorld(true);
    const p = new THREE.Vector3();
    const q = new THREE.Quaternion();
    mesh.matrixWorld.decompose(p, q, new THREE.Vector3());
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(p.x, p.y, p.z).setRotation(q),
    );
    const col = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(size.x / 2, size.y / 2, size.z / 2).setFriction(0.9),
      body,
    );
    this.tag(col, { surface, mesh });
    return col;
  }

  /** Keep a Three object glued to a rigid body each frame. */
  sync(body: RAPIER.RigidBody, object: THREE.Object3D) {
    this.synced.push({ body, object });
  }

  unsync(body: RAPIER.RigidBody) {
    this.synced = this.synced.filter((s) => s.body !== body);
  }

  update(dt: number, beforeStep?: (dt: number) => void) {
    this.acc += Math.min(dt, 0.1);
    let steps = 0;
    while (this.acc >= this.step && steps < 4) {
      beforeStep?.(this.step);
      this.world.step();
      this.acc -= this.step;
      steps++;
    }
    for (const s of this.synced) {
      const t = s.body.translation();
      const r = s.body.rotation();
      s.object.position.set(t.x, t.y, t.z);
      s.object.quaternion.set(r.x, r.y, r.z, r.w);
    }
  }

  raycast(origin: THREE.Vector3, dir: THREE.Vector3, maxDist: number, exclude?: RAPIER.Collider): HitInfo | null {
    const ray = new RAPIER.Ray(origin, dir);
    const hit = this.world.castRayAndGetNormal(ray, maxDist, true, undefined, QUERY_NO_DEBRIS, exclude);
    if (!hit) return null;
    const point = origin.clone().addScaledVector(dir, hit.timeOfImpact);
    return {
      point,
      normal: new THREE.Vector3(hit.normal.x, hit.normal.y, hit.normal.z),
      distance: hit.timeOfImpact,
      collider: hit.collider,
      tag: this.tags.get(hit.collider.handle),
    };
  }
}
