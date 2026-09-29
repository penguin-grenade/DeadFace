import RAPIER from '@dimforge/rapier3d-compat';
import * as THREE from 'three';

export type SurfaceKind = 'metal' | 'concrete' | 'wood' | 'flesh' | 'plaster' | 'cardboard' | 'glass' | 'fabric' | 'rubber';

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
 *
 * The level has two collision representations: simple boxes/cylinders the
 * player walks against (WORLD) and exact triangle meshes of the visible
 * geometry that bullets, shell casings and loose props hit (SHOT). Targets and
 * props are DYNAMIC. Bullet rays query as QUERY.
 */
export const G = {
  PLAYER: 0x0001,
  DEBRIS: 0x0002,
  WORLD: 0x0004,
  SHOT: 0x0008,
  DYNAMIC: 0x0010,
  QUERY: 0x0020,
} as const;
const groups = (member: number, filter: number) => ((member << 16) | filter) >>> 0;
export const GROUP_PLAYER = groups(G.PLAYER, G.WORLD | G.DYNAMIC);
export const GROUP_DEBRIS = groups(G.DEBRIS, G.SHOT | G.DYNAMIC);
export const GROUP_WORLD = groups(G.WORLD, G.PLAYER);
export const GROUP_SHOT = groups(G.SHOT, G.QUERY | G.DEBRIS | G.DYNAMIC);
/** Loose props: stand on the exact level geometry, get pushed by the player. */
export const GROUP_PROP = groups(G.DYNAMIC, G.PLAYER | G.DEBRIS | G.SHOT | G.DYNAMIC | G.QUERY);
/** Targets: block the player and take bullets; once knocked down they fall onto the exact level geometry. */
export const GROUP_TARGET = groups(G.DYNAMIC, G.PLAYER | G.DEBRIS | G.SHOT | G.DYNAMIC | G.QUERY);
/** Bullet rays: exact level geometry and anything dynamic. */
export const QUERY_SHOT = groups(G.QUERY, G.SHOT | G.DYNAMIC);

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
    // The procedural fallback level has no separate bullet mesh, so its boxes serve both.
    const col = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(size.x / 2, size.y / 2, size.z / 2).setFriction(0.9).setCollisionGroups(groups(G.WORLD | G.SHOT, 0xffff)),
      body,
    );
    this.tag(col, { surface, mesh });
    return col;
  }

  /** Fixed collider from a level description (walk-against blockers). */
  addFixed(desc: RAPIER.ColliderDesc, pos: THREE.Vector3Like, quat: THREE.QuaternionLike | null, surface: SurfaceKind, group = GROUP_WORLD) {
    const bd = RAPIER.RigidBodyDesc.fixed().setTranslation(pos.x, pos.y, pos.z);
    if (quat) bd.setRotation({ x: quat.x, y: quat.y, z: quat.z, w: quat.w });
    const col = this.world.createCollider(desc.setFriction(0.9).setCollisionGroups(group), this.world.createRigidBody(bd));
    this.tag(col, { surface });
    return col;
  }

  /** Exact triangle collider for bullets and small debris (world-space vertices). */
  addTrimesh(vertices: Float32Array, indices: Uint32Array, surface: SurfaceKind, mesh?: THREE.Object3D) {
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    const col = this.world.createCollider(
      RAPIER.ColliderDesc.trimesh(vertices, indices).setFriction(0.8).setRestitution(0.2).setCollisionGroups(GROUP_SHOT),
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
    const hit = this.world.castRayAndGetNormal(ray, maxDist, true, undefined, QUERY_SHOT, exclude);
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
