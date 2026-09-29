import * as THREE from 'three';
import type RAPIER from '@dimforge/rapier3d-compat';
import { GROUP_PLAYER, type Physics } from '../engine/physics';
import type { Input } from '../engine/input';
import type { Audio } from '../engine/audio';

const RADIUS = 0.3;
const HALF = 0.6; // capsule half-height (cylinder part)
const CENTER = HALF + RADIUS; // body centre above feet
const EYE_STAND = 1.5; // chest/shoulder-mounted camera, a bit below the eyes
const EYE_CROUCH = 1.0;

/** Critically damped spring for camera offsets. */
class Spring {
  value = 0;
  vel = 0;
  constructor(public stiffness: number, public damping: number) {}
  update(dt: number, target = 0) {
    const a = (target - this.value) * this.stiffness - this.vel * this.damping;
    this.vel += a * dt;
    this.value += this.vel * dt;
    return this.value;
  }
}

export class Player {
  readonly root = new THREE.Object3D(); // yaw
  readonly pitchNode = new THREE.Object3D();
  readonly shakeNode = new THREE.Object3D();
  body!: RAPIER.RigidBody;
  collider!: RAPIER.Collider;
  private controller!: RAPIER.KinematicCharacterController;

  private yaw = 0;
  private pitch = 0;
  private yawTarget = 0;
  private pitchTarget = 0;
  private prevYaw = 0;
  private prevPitch = 0;
  readonly angularVel = new THREE.Vector2(); // rad/s, yaw & pitch

  readonly velocity = new THREE.Vector3();
  private vy = 0;
  grounded = false;
  private wasGrounded = true;
  private eye = EYE_STAND;
  private bobPhase = 0;
  private stepSign = 1;
  sensitivity = 0.0022;
  /** Head bob / sway multiplier; toned down for people who ask for reduced motion. */
  private motionScale = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0.3 : 1;

  // Camera feel springs
  private landDip = new Spring(120, 14);
  private recoilPitch = new Spring(180, 16);
  private recoilYaw = new Spring(160, 16);
  private recoilRoll = new Spring(140, 12);
  private roll = 0;

  /** 0..1, how fast we're moving relative to sprint speed (drives weapon bob). */
  moveAmount = 0;
  sprinting = false;
  bob = new THREE.Vector3();

  constructor(
    private physics: Physics,
    private input: Input,
    private audio: Audio,
    readonly camera: THREE.PerspectiveCamera,
  ) {
    this.root.add(this.pitchNode);
    this.pitchNode.add(this.shakeNode);
    this.shakeNode.add(camera);
  }

  spawn(at: THREE.Vector3, yaw: number) {
    const R = this.physics.R;
    const w = this.physics.world;
    this.body = w.createRigidBody(R.RigidBodyDesc.kinematicPositionBased().setTranslation(at.x, at.y + CENTER, at.z));
    this.collider = w.createCollider(R.ColliderDesc.capsule(HALF, RADIUS).setCollisionGroups(GROUP_PLAYER), this.body);
    this.controller = w.createCharacterController(0.02);
    this.controller.enableAutostep(0.35, 0.2, false);
    this.controller.enableSnapToGround(0.3);
    this.controller.setMaxSlopeClimbAngle((50 * Math.PI) / 180);
    this.controller.setApplyImpulsesToDynamicBodies(true);
    this.controller.setCharacterMass(85);
    this.yaw = this.yawTarget = this.prevYaw = yaw;
  }

  /** Called from weapon on fire. Values in radians. */
  kick(pitch: number, yaw: number) {
    this.pitchTarget += pitch * 0.55; // part of the recoil stays (muzzle climb)
    this.yawTarget += yaw * 0.5;
    this.recoilPitch.vel += pitch * 60;
    this.recoilYaw.vel += yaw * 50;
    this.recoilRoll.vel += (Math.random() - 0.5) * 0.9;
  }

  get position() {
    return this.root.position;
  }

  get forward() {
    return new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.getWorldQuaternion(new THREE.Quaternion()));
  }

  fixedUpdate(dt: number, aiming: boolean) {
    const i = this.input;
    const crouch = i.key('KeyC') || i.key('ControlLeft');
    this.sprinting = i.key('ShiftLeft') && !aiming && !crouch && i.key('KeyW');
    const speed = crouch ? 1.3 : this.sprinting ? 4.8 : aiming ? 1.8 : 2.6;

    const f = (i.key('KeyW') ? 1 : 0) - (i.key('KeyS') ? 1 : 0);
    const s = (i.key('KeyD') ? 1 : 0) - (i.key('KeyA') ? 1 : 0);
    const wish = new THREE.Vector3(s, 0, -f);
    if (wish.lengthSq() > 0) wish.normalize().multiplyScalar(speed);
    wish.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);

    // Heavy-ish acceleration: gear and body armour.
    const accel = this.grounded ? 9 : 1.5;
    this.velocity.x += (wish.x - this.velocity.x) * Math.min(1, accel * dt);
    this.velocity.z += (wish.z - this.velocity.z) * Math.min(1, accel * dt);

    if (this.grounded && i.keyPressed('Space')) this.vy = 3.6;
    this.vy -= 9.81 * dt;

    const desired = { x: this.velocity.x * dt, y: this.vy * dt, z: this.velocity.z * dt };
    this.controller.computeColliderMovement(this.collider, desired, undefined, GROUP_PLAYER);
    const mv = this.controller.computedMovement();
    this.grounded = this.controller.computedGrounded();
    if (this.grounded && this.vy < 0) {
      if (!this.wasGrounded && this.vy < -2) {
        this.landDip.vel -= Math.min(6, -this.vy) * 0.9;
        this.audio.footstep(1.4);
      }
      this.vy = 0;
    }
    this.wasGrounded = this.grounded;
    const t = this.body.translation();
    this.body.setNextKinematicTranslation({ x: t.x + mv.x, y: t.y + mv.y, z: t.z + mv.z });

    const eyeTarget = crouch ? EYE_CROUCH : EYE_STAND;
    this.eye += (eyeTarget - this.eye) * Math.min(1, 10 * dt);
  }

  update(dt: number, time: number) {
    const i = this.input;
    this.yawTarget -= i.mouseDX * this.sensitivity;
    this.pitchTarget -= i.mouseDY * this.sensitivity;
    this.pitchTarget = THREE.MathUtils.clamp(this.pitchTarget, -1.45, 1.45);

    // Slight rotational inertia: a camera strapped to a vest never snaps.
    const k = 1 - Math.exp(-28 * dt);
    this.yaw += (this.yawTarget - this.yaw) * k;
    this.pitch += (this.pitchTarget - this.pitch) * k;
    this.angularVel.set((this.yaw - this.prevYaw) / Math.max(dt, 1e-4), (this.pitch - this.prevPitch) / Math.max(dt, 1e-4));
    this.prevYaw = this.yaw;
    this.prevPitch = this.pitch;

    const t = this.body.translation();
    this.root.position.set(t.x, t.y - CENTER + this.eye, t.z);
    this.root.rotation.y = this.yaw;
    this.pitchNode.rotation.x = this.pitch;

    // --- Head bob (gait-driven) ------------------------------------------------
    const hSpeed = Math.hypot(this.velocity.x, this.velocity.z);
    this.moveAmount = this.grounded ? Math.min(1, hSpeed / 4.8) : 0;
    const freq = this.sprinting ? 2.35 : 1.85; // steps per second / 2
    const prevPhase = this.bobPhase;
    this.bobPhase += dt * freq * Math.PI * 2 * Math.min(1, hSpeed / 1.5);
    const amp = this.moveAmount * (this.sprinting ? 1.6 : 1.0) * this.motionScale;
    const bobY = -Math.abs(Math.sin(this.bobPhase)) * 0.045 * amp;
    const bobX = Math.cos(this.bobPhase) * 0.03 * amp;
    const bobRoll = Math.cos(this.bobPhase) * 0.012 * amp;
    this.bob.set(bobX, bobY, 0);
    // Footstep on each half cycle.
    if (this.grounded && hSpeed > 0.5 && Math.floor(prevPhase / Math.PI) !== Math.floor(this.bobPhase / Math.PI)) {
      this.stepSign *= -1;
      this.audio.footstep(this.sprinting ? 1.1 : 0.7);
      this.recoilRoll.vel += this.stepSign * 0.05 * amp;
    }

    // Breathing sway when idle.
    const breathe = Math.sin(time * 1.6) * 0.004 * this.motionScale;
    const swayX = Math.sin(time * 0.7) * 0.0025 + Math.sin(time * 1.9) * 0.001;

    // Strafe roll.
    const local = this.velocity.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), -this.yaw);
    const targetRoll = -local.x * 0.008;
    this.roll += (targetRoll - this.roll) * Math.min(1, 6 * dt);

    const dip = this.landDip.update(dt);
    const rp = this.recoilPitch.update(dt);
    const ry = this.recoilYaw.update(dt);
    const rr = this.recoilRoll.update(dt);

    this.shakeNode.position.set(bobX, bobY + dip * 0.1 + breathe, 0);
    this.shakeNode.rotation.set(rp + breathe * 0.5, ry + swayX, this.roll + bobRoll + rr * 0.08, 'YXZ');
  }
}
