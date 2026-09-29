import * as THREE from 'three';
import type { SurfaceKind } from '../engine/physics';
import { bulletHole, softSprite } from './textures';

const MAX_PARTICLES = 600;
const MAX_DECALS = 160;

/** Billboard particles with per-particle size, alpha and colour. */
class ParticlePool {
  readonly points: THREE.Points;
  private pos = new Float32Array(MAX_PARTICLES * 3);
  private vel = new Float32Array(MAX_PARTICLES * 3);
  private col = new Float32Array(MAX_PARTICLES * 3);
  private alpha = new Float32Array(MAX_PARTICLES);
  private size = new Float32Array(MAX_PARTICLES);
  private life = new Float32Array(MAX_PARTICLES);
  private maxLife = new Float32Array(MAX_PARTICLES);
  private grow = new Float32Array(MAX_PARTICLES);
  private drag = new Float32Array(MAX_PARTICLES);
  private gravity = new Float32Array(MAX_PARTICLES);
  private next = 0;
  private geo = new THREE.BufferGeometry();

  constructor(additive: boolean, map: THREE.Texture) {
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.geo.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1));
    this.geo.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));
    const mat = new THREE.ShaderMaterial({
      uniforms: { map: { value: map }, scale: { value: 600 } },
      vertexShader: /* glsl */ `
        attribute float aAlpha;
        attribute float aSize;
        varying float vAlpha;
        varying vec3 vColor;
        uniform float scale;
        void main() {
          vAlpha = aAlpha;
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = min(aSize * scale / -mv.z, 96.0);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D map;
        varying float vAlpha;
        varying vec3 vColor;
        void main() {
          vec4 t = texture2D(map, gl_PointCoord);
          gl_FragColor = vec4(vColor * t.rgb, t.a * vAlpha * vAlpha);
          #include <colorspace_fragment>
        }`,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.points = new THREE.Points(this.geo, mat);
    this.points.frustumCulled = false;
  }

  emit(p: THREE.Vector3, v: THREE.Vector3, color: THREE.Color, size: number, life: number, opts: { grow?: number; drag?: number; gravity?: number } = {}) {
    const i = this.next;
    this.next = (this.next + 1) % MAX_PARTICLES;
    this.pos.set([p.x, p.y, p.z], i * 3);
    this.vel.set([v.x, v.y, v.z], i * 3);
    this.col.set([color.r, color.g, color.b], i * 3);
    this.size[i] = size;
    this.life[i] = life;
    this.maxLife[i] = life;
    this.alpha[i] = 1;
    this.grow[i] = opts.grow ?? 0;
    this.drag[i] = opts.drag ?? 0;
    this.gravity[i] = opts.gravity ?? 0;
  }

  update(dt: number) {
    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (this.life[i] <= 0) {
        this.alpha[i] = 0;
        continue;
      }
      this.life[i] -= dt;
      const k = Math.max(0, 1 - this.drag[i] * dt);
      this.vel[i * 3] *= k;
      this.vel[i * 3 + 1] = this.vel[i * 3 + 1] * k - this.gravity[i] * dt;
      this.vel[i * 3 + 2] *= k;
      this.pos[i * 3] += this.vel[i * 3] * dt;
      this.pos[i * 3 + 1] += this.vel[i * 3 + 1] * dt;
      this.pos[i * 3 + 2] += this.vel[i * 3 + 2] * dt;
      this.size[i] += this.grow[i] * dt;
      this.alpha[i] = Math.max(0, this.life[i] / this.maxLife[i]);
    }
    for (const a of ['position', 'color', 'aAlpha', 'aSize']) this.geo.attributes[a].needsUpdate = true;
  }
}

export class Effects {
  private sparks: ParticlePool;
  private dust: ParticlePool;
  private decals: THREE.Mesh[] = [];
  private decalGeo = new THREE.PlaneGeometry(1, 1);
  private hardMat: THREE.MeshStandardMaterial;
  private softMat: THREE.MeshStandardMaterial;
  private tmpV = new THREE.Vector3();
  private tmpQ = new THREE.Quaternion();

  constructor(private scene: THREE.Scene) {
    this.sparks = new ParticlePool(true, softSprite([255, 250, 220], [255, 150, 40]));
    this.dust = new ParticlePool(false, softSprite([200, 200, 200], [160, 160, 160]));
    scene.add(this.sparks.points, this.dust.points);
    const decal = (t: THREE.Texture) =>
      new THREE.MeshStandardMaterial({
        map: t,
        transparent: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        roughness: 0.9,
      });
    this.hardMat = decal(bulletHole('hard'));
    this.softMat = decal(bulletHole('soft'));
  }

  impact(point: THREE.Vector3, normal: THREE.Vector3, surface: SurfaceKind, dir: THREE.Vector3, parent?: THREE.Object3D) {
    this.decal(point, normal, surface, parent);
    const reflect = dir.clone().reflect(normal);
    const rand = () => (Math.random() - 0.5) * 2;

    if (surface === 'metal') {
      for (let i = 0; i < 18; i++) {
        this.tmpV.copy(reflect).multiplyScalar(3 + Math.random() * 5).add(new THREE.Vector3(rand(), rand(), rand()).multiplyScalar(3));
        this.sparks.emit(point, this.tmpV, new THREE.Color(1, 0.75, 0.35).multiplyScalar(3), 0.025, 0.15 + Math.random() * 0.25, { gravity: 9.8, drag: 1 });
      }
    }
    const dustColor: Record<SurfaceKind, [number, number, number]> = {
      concrete: [0.55, 0.53, 0.5],
      plaster: [0.8, 0.8, 0.75],
      wood: [0.45, 0.33, 0.2],
      cardboard: [0.5, 0.38, 0.25],
      metal: [0.3, 0.3, 0.3],
      flesh: [0.6, 0.55, 0.5],
      glass: [0.8, 0.85, 0.9],
      fabric: [0.5, 0.46, 0.38],
      fibreglass: [0.86, 0.85, 0.8],
      rubber: [0.12, 0.12, 0.12],
    };
    const c = dustColor[surface];
    const count = surface === 'metal' ? 3 : 10;
    for (let i = 0; i < count; i++) {
      this.tmpV.copy(normal).multiplyScalar(0.6 + Math.random() * 1.8).add(new THREE.Vector3(rand(), rand(), rand()).multiplyScalar(0.5));
      this.dust.emit(point.clone().addScaledVector(normal, 0.02), this.tmpV, new THREE.Color(...c), 0.05 + Math.random() * 0.05, 0.6 + Math.random() * 0.9, { grow: 0.35, drag: 3.5, gravity: 0.4 });
    }
    // Chips/debris
    if (surface !== 'metal' && surface !== 'flesh') {
      for (let i = 0; i < 6; i++) {
        this.tmpV.copy(reflect).multiplyScalar(1.5 + Math.random() * 2).add(new THREE.Vector3(rand(), rand() + 1, rand()));
        this.dust.emit(point, this.tmpV, new THREE.Color(...c).multiplyScalar(0.6), 0.012, 0.6, { gravity: 9.8, drag: 0.5 });
      }
    }
  }

  muzzleSmoke(p: THREE.Vector3, dir: THREE.Vector3) {
    for (let i = 0; i < 4; i++) {
      const v = dir.clone().multiplyScalar(0.4 + Math.random() * 0.8).add(new THREE.Vector3(0, 0.15, 0));
      this.dust.emit(p, v, new THREE.Color(0.35, 0.35, 0.35), 0.015, 0.6 + Math.random() * 0.5, { grow: 0.2, drag: 2.5, gravity: -0.15 });
    }
  }

  private decal(point: THREE.Vector3, normal: THREE.Vector3, surface: SurfaceKind, parent?: THREE.Object3D) {
    const hard = surface === 'concrete' || surface === 'plaster' || surface === 'metal';
    const mesh = new THREE.Mesh(this.decalGeo, hard ? this.hardMat : this.softMat);
    const s = hard ? 0.09 + Math.random() * 0.05 : surface === 'fibreglass' ? 0.04 : 0.05;
    mesh.scale.set(s, s, s);
    mesh.receiveShadow = true;
    this.tmpQ.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    mesh.quaternion.copy(this.tmpQ);
    mesh.rotateZ(Math.random() * Math.PI * 2);
    mesh.position.copy(point).addScaledVector(normal, 0.003);
    this.scene.add(mesh);
    if (parent) parent.attach(mesh);
    this.decals.push(mesh);
    if (this.decals.length > MAX_DECALS) this.decals.shift()!.removeFromParent();
  }

  update(dt: number, viewportHeight: number) {
    const scale = viewportHeight * 0.9;
    (this.sparks.points.material as THREE.ShaderMaterial).uniforms.scale.value = scale;
    (this.dust.points.material as THREE.ShaderMaterial).uniforms.scale.value = scale;
    this.sparks.update(dt);
    this.dust.update(dt);
  }
}
