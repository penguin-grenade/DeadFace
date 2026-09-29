import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { BodycamShader } from './BodycamShader';

export type Quality = 'low' | 'medium' | 'high';

export class Renderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly composer: EffectComposer;
  readonly bodycam: ShaderPass;
  readonly bloom: UnrealBloomPass;
  private quality: Quality = 'high';

  constructor(
    canvas: HTMLCanvasElement,
    readonly scene: THREE.Scene,
    readonly camera: THREE.PerspectiveCamera,
  ) {
    const r = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    this.renderer = r;

    const size = new THREE.Vector2(window.innerWidth, window.innerHeight);
    const target = new THREE.WebGLRenderTarget(size.x, size.y, {
      type: THREE.HalfFloatType,
      samples: 4,
    });
    this.composer = new EffectComposer(r, target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new UnrealBloomPass(size, 0.35, 0.6, 0.92);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.bodycam = new ShaderPass(BodycamShader);
    this.composer.addPass(this.bodycam);

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  setQuality(q: Quality) {
    this.quality = q;
    this.bloom.enabled = q !== 'low';
    this.bodycam.uniforms.uBlurSamples.value = q === 'low' ? 1 : q === 'medium' ? 5 : 9;
    this.renderer.shadowMap.enabled = true;
    this.resize();
  }

  resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio, this.quality === 'high' ? 1.5 : 1);
    const scale = this.quality === 'low' ? 0.75 : 1;
    this.renderer.setPixelRatio(dpr * scale);
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(dpr * scale);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.bodycam.uniforms.uResolution.value.set(w * dpr * scale, h * dpr * scale);
  }

  render(time: number) {
    this.bodycam.uniforms.uTime.value = time;
    this.composer.render();
  }
}
