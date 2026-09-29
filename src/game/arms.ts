import * as THREE from 'three';

/**
 * Viewmodel arms posed with two-bone IK every frame, on the rig blender/make_hands.py builds.
 * The shoulders stay put beside the chest-mounted camera and the hands stay on the pistol, so the
 * elbows follow the gun as it's raised, lowered, canted and kicked back. Each arm has an upper
 * arm, a forearm in two parts (the elbow end follows the bend, the wrist end keeps the hand's
 * twist, so the turn of the forearm spreads along the sleeve) and a hand that rides on the gun.
 */

// Camera space (three.js axes), right arm; the left mirrors x. Same as SHOULDER / POLE in make_hands.py.
const SHOULDER = new THREE.Vector3(0.185, -0.03, 0.1);
const POLE = new THREE.Vector3(1.0, -0.6, 0.2);

interface Arm {
  shoulder: THREE.Vector3;
  pole: THREE.Vector3;
  upper: THREE.Bone;
  fore: THREE.Bone;
  twist: THREE.Bone;
  a: number;
  b: number;
  wrist: THREE.Vector3;
  /** Rest bend normal: the twist reference the hand keeps. */
  n0: THREE.Vector3;
  /** Rest frames (x along the bone, z the bend normal) and rest rotations. */
  upper0: THREE.Matrix4;
  fore0: THREE.Matrix4;
  qUpper: THREE.Quaternion;
  qFore: THREE.Quaternion;
  qTwist: THREE.Quaternion;
}

const _m = new THREE.Matrix4();
const _r = new THREE.Matrix4();
const _t = new THREE.Matrix4();
const _S = new THREE.Vector3();
const _P = new THREE.Vector3();
const _E = new THREE.Vector3();
const _d = new THREE.Vector3();
const _u = new THREE.Vector3();
const _v = new THREE.Vector3();
const _n = new THREE.Vector3();
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();

/** Rotation whose columns are x (along the bone), z × x and the bend normal z. */
function frame(out: THREE.Matrix4, x: THREE.Vector3, z: THREE.Vector3) {
  _y.crossVectors(z, x);
  return out.makeBasis(x, _y, z);
}

export class ArmRig {
  private arms: Arm[];

  private constructor(
    private rig: THREE.Object3D,
    arms: Arm[],
  ) {
    this.arms = arms;
  }

  /** The rig in a loaded arms.glb, or null for a model without one. */
  static from(root: THREE.Object3D): ArmRig | null {
    const arms: Arm[] = [];
    let rig: THREE.Object3D | null = null;
    for (const [sfx, sign] of [['R', 1], ['L', -1]] as const) {
      const bone = (n: string) => root.getObjectByName(`${n}_${sfx}`) as THREE.Bone | undefined;
      const upper = bone('upper');
      const fore = bone('fore');
      const twist = bone('twist');
      const hand = bone('hand');
      if (!upper || !fore || !twist || !hand || upper.parent !== hand.parent) return null;
      rig = upper.parent;
      const S0 = upper.position.clone();
      const E0 = fore.position.clone();
      const W0 = hand.position.clone();
      const n0 = new THREE.Vector3().subVectors(E0, S0).cross(_d.subVectors(W0, E0)).normalize();
      arms.push({
        shoulder: SHOULDER.clone().setX(SHOULDER.x * sign),
        pole: POLE.clone().setX(POLE.x * sign).normalize(),
        upper,
        fore,
        twist,
        a: S0.distanceTo(E0),
        b: E0.distanceTo(W0),
        wrist: W0,
        n0,
        upper0: frame(new THREE.Matrix4(), _x.subVectors(E0, S0).normalize(), n0),
        fore0: frame(new THREE.Matrix4(), _x.subVectors(W0, E0).normalize(), n0),
        qUpper: upper.quaternion.clone(),
        qFore: fore.quaternion.clone(),
        qTwist: twist.quaternion.clone(),
      });
    }
    root.traverse((o) => {
      // Bones move far from the bind pose; the bind-pose bounds would cull the sleeves.
      if ((o as THREE.SkinnedMesh).isSkinnedMesh) o.frustumCulled = false;
    });
    return rig ? new ArmRig(rig, arms) : null;
  }

  /** Pose both arms for the current gun placement. Call after the viewmodel pivot has moved. */
  update(camera: THREE.Camera) {
    this.rig.updateWorldMatrix(true, false);
    // camera space -> rig space
    _m.copy(this.rig.matrixWorld).invert().multiply(camera.matrixWorld);
    for (const arm of this.arms) {
      const { a, b, wrist: W } = arm;
      const S = _S.copy(arm.shoulder).applyMatrix4(_m);
      const pole = _P.copy(arm.pole).transformDirection(_m);
      // Out of reach: the shoulder comes forward along the arm. Too close: it goes back.
      _d.subVectors(W, S);
      let L = _d.length();
      const lo = Math.abs(a - b) + 0.01;
      const hi = a + b - 0.002;
      if (L > hi || L < lo) {
        const Lc = THREE.MathUtils.clamp(L, lo, hi);
        S.copy(W).addScaledVector(_d, -Lc / L);
        L = Lc;
      }
      const u = _u.copy(_d.subVectors(W, S)).divideScalar(L);
      const ca = THREE.MathUtils.clamp((a * a + L * L - b * b) / (2 * a * L), -1, 1);
      const v = _v.copy(pole).addScaledVector(u, -pole.dot(u));
      if (v.lengthSq() < 1e-8) v.set(0, -1, 0).addScaledVector(u, u.y);
      v.normalize();
      const E = _E.copy(S).addScaledVector(u, a * ca).addScaledVector(v, a * Math.sqrt(1 - ca * ca));
      const n = _n.crossVectors(v, u).normalize();

      // Upper arm and the elbow end of the forearm turn with the bend plane.
      frame(_r, _x.subVectors(E, S).normalize(), n).multiply(_t.copy(arm.upper0).transpose());
      arm.upper.position.copy(S);
      arm.upper.quaternion.setFromRotationMatrix(_r).multiply(arm.qUpper);
      frame(_r, _x.subVectors(W, E).normalize(), n).multiply(_t.copy(arm.fore0).transpose());
      arm.fore.position.copy(E);
      arm.fore.quaternion.setFromRotationMatrix(_r).multiply(arm.qFore);
      // The wrist end keeps the hand's twist: the rest bend normal, squared up to the forearm.
      _z.copy(arm.n0).addScaledVector(_x, -arm.n0.dot(_x)).normalize();
      frame(_r, _x, _z).multiply(_t.copy(arm.fore0).transpose());
      arm.twist.position.copy(E);
      arm.twist.quaternion.setFromRotationMatrix(_r).multiply(arm.qTwist);
    }
  }
}
