import * as THREE from 'three';

/**
 * Weapon inertia for the viewmodel. The pistol is treated as a mass held out on springy arms:
 * when the body turns, the gun keeps going the way it was for a moment, trails the turn while it
 * lasts, swings a little past centre when the turn stops and settles. The same goes for starting
 * and stopping a walk or strafe. Turning also cants the gun into the turn, a beat behind it.
 *
 * Each axis is a damped oscillator in camera space, kicked by the change in the camera's own
 * velocity (what a free mass would see) and pulled along by drag while the camera moves. Aiming
 * locks the arms out: stiffer springs, smaller kicks, so the sights wander less and settle faster.
 */

interface Tune {
  /** Spring stiffness (1/s²) and damping ratio. */
  k: number;
  zeta: number;
  /** Fraction of a change in camera velocity the gun doesn't follow at once. */
  inertia: number;
  /** Steady trail per unit of camera velocity, as a spring force (offset = -drag·rate / k). */
  drag: number;
}

const HIP = {
  turn: { k: 120, zeta: 0.55, inertia: 0.4, drag: 7.0 },
  pitch: { k: 140, zeta: 0.6, inertia: 0.3, drag: 4.5 },
  move: { k: 100, zeta: 0.5, inertia: 0.14, drag: 0.5 },
  roll: { k: 80, zeta: 0.45 },
  /** Cant per rad/s of turn, and per m/s of strafe. */
  rollTurn: 0.05,
  rollStrafe: 0.035,
  /** Breathing / muscle drift amplitude (rad). */
  idle: 0.008,
};
const ADS = {
  turn: { k: 220, zeta: 0.65, inertia: 0.17, drag: 2.7 },
  pitch: { k: 240, zeta: 0.68, inertia: 0.14, drag: 2.2 },
  move: { k: 180, zeta: 0.6, inertia: 0.06, drag: 0.2 },
  roll: { k: 140, zeta: 0.6 },
  rollTurn: 0.016,
  rollStrafe: 0.01,
  idle: 0.003,
};

const MAX_TRAIL = 0.22; // rad, soft limit for a flick
const MAX_ROLL = 0.24;
const MAX_SHIFT = 0.05; // m
const STEP = 1 / 120;

class Osc {
  x = 0;
  v = 0;
  /** Integrate toward `target` with an extra constant `force`. */
  step(h: number, k: number, zeta: number, target: number, force: number) {
    const c = 2 * zeta * Math.sqrt(k);
    this.v += (k * (target - this.x) - c * this.v + force) * h;
    this.x += this.v * h;
  }
}

const mix = (a: Tune, b: Tune, t: number, out: Tune) => {
  out.k = THREE.MathUtils.lerp(a.k, b.k, t);
  out.zeta = THREE.MathUtils.lerp(a.zeta, b.zeta, t);
  out.inertia = THREE.MathUtils.lerp(a.inertia, b.inertia, t);
  out.drag = THREE.MathUtils.lerp(a.drag, b.drag, t);
  return out;
};
const soft = (x: number, max: number) => max * Math.tanh(x / max);

export class WeaponSway {
  private yaw = new Osc();
  private pitch = new Osc();
  private roll = new Osc();
  private move = [new Osc(), new Osc(), new Osc()];
  private prevTurn = 0;
  private prevPitch = 0;
  private prevVel = new THREE.Vector3();
  private turnT: Tune = { ...HIP.turn };
  private pitchT: Tune = { ...HIP.pitch };
  private moveT: Tune = { ...HIP.move };

  /** Gun yaw relative to the camera (rad, + = muzzle left), i.e. how far it trails a turn. */
  trailYaw = 0;
  /** Gun pitch relative to the camera (rad, + = muzzle up). */
  trailPitch = 0;
  /** Cant (rad, + = top of the slide leans left). */
  cant = 0;
  /** Translation in camera space (m), from walking and strafing starts and stops. */
  readonly shift = new THREE.Vector3();

  /**
   * @param turnRate camera yaw rate, rad/s, + = turning left
   * @param pitchRate camera pitch rate, rad/s, + = looking up
   * @param vel player velocity in camera-yaw space (x right, z back), m/s
   * @param aim 0 at the hip, 1 aimed
   * @param scale overall motion multiplier (reduced motion)
   */
  update(dt: number, time: number, turnRate: number, pitchRate: number, vel: THREE.Vector3, aim: number, scale: number) {
    if (dt <= 0) return;
    mix(HIP.turn, ADS.turn, aim, this.turnT);
    mix(HIP.pitch, ADS.pitch, aim, this.pitchT);
    mix(HIP.move, ADS.move, aim, this.moveT);
    const rollK = THREE.MathUtils.lerp(HIP.roll.k, ADS.roll.k, aim);
    const rollZ = THREE.MathUtils.lerp(HIP.roll.zeta, ADS.roll.zeta, aim);
    const rollTarget =
      THREE.MathUtils.lerp(HIP.rollTurn, ADS.rollTurn, aim) * turnRate - THREE.MathUtils.lerp(HIP.rollStrafe, ADS.rollStrafe, aim) * vel.x;

    // The camera just changed speed; the gun's mass hasn't yet.
    this.yaw.v -= this.turnT.inertia * (turnRate - this.prevTurn);
    this.pitch.v -= this.pitchT.inertia * (pitchRate - this.prevPitch);
    for (let i = 0; i < 3; i++) this.move[i].v -= this.moveT.inertia * (vel.getComponent(i) - this.prevVel.getComponent(i));
    this.prevTurn = turnRate;
    this.prevPitch = pitchRate;
    this.prevVel.copy(vel);

    const n = Math.min(12, Math.ceil(dt / STEP));
    const h = dt / n;
    for (let s = 0; s < n; s++) {
      this.yaw.step(h, this.turnT.k, this.turnT.zeta, 0, -this.turnT.drag * turnRate);
      this.pitch.step(h, this.pitchT.k, this.pitchT.zeta, 0, -this.pitchT.drag * pitchRate);
      this.roll.step(h, rollK, rollZ, soft(rollTarget, MAX_ROLL), 0);
      for (let i = 0; i < 3; i++) this.move[i].step(h, this.moveT.k, this.moveT.zeta, 0, -this.moveT.drag * vel.getComponent(i));
    }
    // A flick can't throw the gun out of the hands: keep the oscillators inside the soft limits.
    for (const o of [this.yaw, this.pitch]) if (Math.abs(o.x) > MAX_TRAIL * 1.5) { o.x = Math.sign(o.x) * MAX_TRAIL * 1.5; o.v *= 0.5; }

    // Breathing and small muscle corrections: a slow figure eight, tighter when aimed.
    const idle = THREE.MathUtils.lerp(HIP.idle, ADS.idle, aim);
    const driftYaw = idle * (Math.sin(time * 0.83) * 0.7 + Math.sin(time * 2.3 + 1.1) * 0.3);
    const driftPitch = idle * (Math.sin(time * 1.66 + 0.4) * 0.5 + Math.sin(time * 0.37) * 0.5);

    this.trailYaw = (soft(this.yaw.x, MAX_TRAIL) + driftYaw) * scale;
    this.trailPitch = (soft(this.pitch.x, MAX_TRAIL) + driftPitch) * scale;
    this.cant = soft(this.roll.x, MAX_ROLL) * scale;
    this.shift.set(soft(this.move[0].x, MAX_SHIFT), soft(this.move[1].x, MAX_SHIFT), soft(this.move[2].x, MAX_SHIFT)).multiplyScalar(scale);
  }
}
