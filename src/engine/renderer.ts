import * as THREE from 'three';
import { BodycamShader, CopyShader } from './BodycamShader';
import { PostFX } from './postfx';

export type Quality = 'low' | 'medium' | 'high';

/**
 * WebGL renderer + the HDR camera pipeline (see postfx.ts), with dynamic
 * resolution: the internal resolution drops when frames run long and climbs
 * back when there is headroom. The bodycam pass upscales to the canvas, and its
 * grain and sharpening hide most of the difference.
 */
export class Renderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly post: PostFX;
  readonly bodycam: THREE.ShaderMaterial;
  private copy: THREE.ShaderMaterial;
  /** false shows the straight tone-mapped image (debug). */
  bodycamEnabled = true;
  dynamicResolution = true;
  private quality: Quality = 'high';
  private scale = 1;
  private frameAvg = 1 / 60;
  private sinceResize = 0;

  constructor(
    canvas: HTMLCanvasElement,
    readonly scene: THREE.Scene,
    readonly camera: THREE.PerspectiveCamera,
  ) {
    const r = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false });
    r.outputColorSpace = THREE.SRGBColorSpace;
    // Tone mapping happens in the post pipeline; the scene renders to linear HDR.
    r.toneMapping = THREE.NoToneMapping;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    this.renderer = r;
    this.post = new PostFX(r, 4);
    this.bodycam = new THREE.ShaderMaterial({ ...BodycamShader, uniforms: THREE.UniformsUtils.clone(BodycamShader.uniforms), depthTest: false, depthWrite: false });
    this.copy = new THREE.ShaderMaterial({ ...CopyShader, uniforms: THREE.UniformsUtils.clone(CopyShader.uniforms), depthTest: false, depthWrite: false });
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  setQuality(q: Quality) {
    this.quality = q;
    const p = this.post;
    p.volumetrics = q !== 'low';
    p.volumeSteps = q === 'high' ? 24 : 14;
    p.invalidate();
    this.bodycam.uniforms.uBlurSamples.value = q === 'low' ? 1 : q === 'medium' ? 5 : 9;
    this.scale = 1;
    this.resize();
  }

  private get baseRatio() {
    return Math.min(window.devicePixelRatio, this.quality === 'high' ? 1.5 : 1) * (this.quality === 'low' ? 0.75 : 1);
  }

  resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const ratio = this.baseRatio;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(w, h, false);
    this.post.setSize(w * ratio * this.scale, h * ratio * this.scale);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.bodycam.uniforms.uResolution.value.set(w * ratio, h * ratio);
  }

  /** Internal resolution relative to the canvas (dynamic resolution). */
  get resolutionScale() {
    return this.scale;
  }

  private adaptResolution(dt: number) {
    if (!this.dynamicResolution) return;
    this.frameAvg += (dt - this.frameAvg) * 0.05;
    this.sinceResize += dt;
    if (this.sinceResize < 1.5) return;
    let next = this.scale;
    if (this.frameAvg > 1 / 45) next = Math.max(0.5, this.scale * 0.85);
    else if (this.frameAvg < 1 / 58 && this.scale < 1) next = Math.min(1, this.scale * 1.1);
    if (Math.abs(next - this.scale) > 0.01) {
      this.scale = next;
      this.sinceResize = 0;
      this.resize();
    }
  }

  render(time: number, dt = 1 / 60) {
    this.adaptResolution(dt);
    this.bodycam.uniforms.uTime.value = time;
    this.post.render(this.scene, this.camera, dt, time, this.bodycamEnabled ? this.bodycam : this.copy);
  }
}
