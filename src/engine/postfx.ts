import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';

/**
 * HDR camera pipeline, replacing EffectComposer:
 *
 *   scene (HDR, MSAA, depth)
 *     -> volumetric in-scattering (half res, raymarched through the lamps' shadow maps)
 *     -> bloom (energy-conserving mip chain, the lens veil around lamps)
 *     -> auto exposure (GPU luminance reduction, bodycam-style adaptation)
 *     -> AgX tone curve -> sRGB
 *     -> bodycam lens/sensor pass (see BodycamShader) -> screen
 */

const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;

const hdrTarget = (w: number, h: number, opts: THREE.RenderTargetOptions = {}) =>
  new THREE.WebGLRenderTarget(w, h, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: false,
    generateMipmaps: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    ...opts,
  });

export interface VolumetricSpot {
  light: THREE.SpotLight;
  /** Multiplier on the physical in-scattering (artistic). */
  strength: number;
}

export interface MoonShafts {
  direction: THREE.Vector3; // travel direction of the light
  color: THREE.Color; // irradiance, linear
  shadow: THREE.DepthTexture;
  matrix: THREE.Matrix4; // world -> shadow uv/depth
}

function volumetricShader(numSpots: number, moon: boolean, steps: number) {
  const spot = (i: number) => /* glsl */ `
    {
      vec3 d = p - uSpotPos[ ${i} ];
      float d2 = max( dot( d, d ), 1e-4 );
      vec3 l = d * inversesqrt( d2 );
      float att = smoothstep( uSpotCone[ ${i} ].x, uSpotCone[ ${i} ].y, dot( l, uSpotDir[ ${i} ] ) );
      if ( att > 0.0 ) {
        float sh = 1.0;
        vec4 sc = uSpotMatrix[ ${i} ] * vec4( p, 1.0 );
        sc.xyz /= sc.w;
        if ( sc.w > 0.0 && sc.x > 0.0 && sc.x < 1.0 && sc.y > 0.0 && sc.y < 1.0 && sc.z < 1.0 ) {
          sh = textureLod( uSpotShadow[ ${i} ], vec3( sc.xy, sc.z + uSpotBias[ ${i} ] ), 0.0 );
        }
        L += uSpotColor[ ${i} ] * ( att * sh * phase( dot( l, -rd ) ) / max( d2, 0.25 ) );
      }
    }`;
  const spots = Array.from({ length: numSpots }, (_, i) => spot(i)).join('\n');
  return /* glsl */ `
precision highp float;
precision highp sampler2DShadow;
#include <common>
uniform sampler2D tDepth;
uniform mat4 uProjInv;
uniform mat4 uViewInv;
uniform vec3 uCamPos;
uniform float uScatter;
uniform float uExtinction;
uniform float uG;
uniform float uFrame;
uniform float uMaxDist;
uniform float uTime;
#if ${numSpots} > 0
uniform vec3 uSpotPos[ ${Math.max(numSpots, 1)} ];
uniform vec3 uSpotDir[ ${Math.max(numSpots, 1)} ];
uniform vec3 uSpotColor[ ${Math.max(numSpots, 1)} ];
uniform vec2 uSpotCone[ ${Math.max(numSpots, 1)} ];
uniform mat4 uSpotMatrix[ ${Math.max(numSpots, 1)} ];
uniform float uSpotBias[ ${Math.max(numSpots, 1)} ];
uniform sampler2DShadow uSpotShadow[ ${Math.max(numSpots, 1)} ];
#endif
#if ${moon ? 1 : 0}
uniform vec3 uMoonDir;
uniform vec3 uMoonColor;
uniform mat4 uMoonMatrix;
uniform sampler2DShadow uMoonShadow;
#endif
varying vec2 vUv;

float phase( float c ) {
  float g2 = uG * uG;
  return ( 1.0 - g2 ) / ( 4.0 * PI * pow( max( 1.0 + g2 - 2.0 * uG * c, 1e-4 ), 1.5 ) );
}

// Slowly drifting dust density, so shafts are not perfectly uniform.
float dust( vec3 p ) {
  vec3 q = p * 0.35 + vec3( 0.0, uTime * 0.03, uTime * 0.02 );
  float n = sin( q.x * 1.7 + sin( q.z * 1.3 ) ) * sin( q.z * 1.9 + sin( q.y * 2.1 ) ) * sin( q.y * 1.3 + q.x * 0.7 );
  return 0.75 + 0.45 * n;
}

void main() {
  float depth = texture2D( tDepth, vUv ).x;
  vec4 vp = uProjInv * vec4( vUv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0 );
  vp /= vp.w;
  vec3 wp = ( uViewInv * vp ).xyz;
  vec3 ray = wp - uCamPos;
  float dist = length( ray );
  vec3 rd = ray / max( dist, 1e-4 );
  float tMax = min( dist, uMaxDist );
  float stepLen = tMax / float( ${steps} );
  float jitter = fract( 52.9829189 * fract( dot( gl_FragCoord.xy, vec2( 0.06711056, 0.00583715 ) ) ) + uFrame * 0.61803398875 );
  vec3 acc = vec3( 0.0 );
  float T = 1.0;
  for ( int s = 0; s < ${steps}; s ++ ) {
    vec3 p = uCamPos + rd * ( ( float( s ) + jitter ) * stepLen );
    vec3 L = vec3( 0.0 );
${spots}
#if ${moon ? 1 : 0}
    {
      vec4 mc = uMoonMatrix * vec4( p, 1.0 );
      float msh = 0.0;
      if ( mc.x > 0.0 && mc.x < 1.0 && mc.y > 0.0 && mc.y < 1.0 ) msh = textureLod( uMoonShadow, vec3( mc.xy, mc.z - 0.002 ), 0.0 );
      L += uMoonColor * ( msh * phase( dot( uMoonDir, -rd ) ) );
    }
#endif
    float sigma = uScatter * dust( p );
    acc += T * L * sigma * stepLen;
    T *= exp( -uExtinction * stepLen );
  }
  gl_FragColor = vec4( acc, T );
}`;
}

const DOWN_FRAG = /* glsl */ `
uniform sampler2D tSrc;
uniform vec2 uTexel;
uniform bool uKaris;
uniform sampler2D tVol;
uniform bool uUseVol;
varying vec2 vUv;
vec3 src( vec2 uv ) {
  vec3 c = texture2D( tSrc, uv ).rgb;
  if ( uUseVol ) {
    vec4 v = texture2D( tVol, uv );
    c = c * v.a + v.rgb;
  }
  return c;
}
float karis( vec3 c ) { return 1.0 / ( 1.0 + dot( c, vec3( 0.2126, 0.7152, 0.0722 ) ) ); }
void main() {
  vec2 d = uTexel;
  vec3 a = src( vUv + d * vec2( -2.0, 2.0 ) ), b = src( vUv + d * vec2( 0.0, 2.0 ) ), c = src( vUv + d * vec2( 2.0, 2.0 ) );
  vec3 e = src( vUv + d * vec2( -2.0, 0.0 ) ), f = src( vUv ), g = src( vUv + d * vec2( 2.0, 0.0 ) );
  vec3 h = src( vUv + d * vec2( -2.0, -2.0 ) ), i = src( vUv + d * vec2( 0.0, -2.0 ) ), j = src( vUv + d * vec2( 2.0, -2.0 ) );
  vec3 k = src( vUv + d * vec2( -1.0, 1.0 ) ), l = src( vUv + d * vec2( 1.0, 1.0 ) );
  vec3 m = src( vUv + d * vec2( -1.0, -1.0 ) ), n = src( vUv + d * vec2( 1.0, -1.0 ) );
  vec3 o;
  if ( uKaris ) {
    // Karis average per 2x2 block: one blazing texel (a bulb) cannot flicker the whole bloom.
    vec3 g0 = ( a + b + e + f ) * 0.25, g1 = ( b + c + f + g ) * 0.25, g2 = ( e + f + h + i ) * 0.25, g3 = ( f + g + i + j ) * 0.25, g4 = ( k + l + m + n ) * 0.25;
    float w0 = karis( g0 ) * 0.125, w1 = karis( g1 ) * 0.125, w2 = karis( g2 ) * 0.125, w3 = karis( g3 ) * 0.125, w4 = karis( g4 ) * 0.5;
    o = ( g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4 ) / ( w0 + w1 + w2 + w3 + w4 );
  } else {
    o = f * 0.125 + ( a + c + h + j ) * 0.03125 + ( b + e + g + i ) * 0.0625 + ( k + l + m + n ) * 0.125;
  }
  gl_FragColor = vec4( max( o, 0.0 ), 1.0 );
}`;

const UP_FRAG = /* glsl */ `
uniform sampler2D tSrc; // coarser level (upsampled)
uniform sampler2D tAdd; // same-size downsample level
uniform vec2 uTexel;
uniform float uRadius;
varying vec2 vUv;
void main() {
  vec2 d = uTexel * uRadius;
  vec3 s = texture2D( tSrc, vUv ).rgb * 4.0;
  s += ( texture2D( tSrc, vUv + vec2( d.x, 0.0 ) ).rgb + texture2D( tSrc, vUv - vec2( d.x, 0.0 ) ).rgb + texture2D( tSrc, vUv + vec2( 0.0, d.y ) ).rgb + texture2D( tSrc, vUv - vec2( 0.0, d.y ) ).rgb ) * 2.0;
  s += texture2D( tSrc, vUv + d ).rgb + texture2D( tSrc, vUv - d ).rgb + texture2D( tSrc, vUv + vec2( d.x, -d.y ) ).rgb + texture2D( tSrc, vUv + vec2( -d.x, d.y ) ).rgb;
  gl_FragColor = vec4( s / 16.0 + texture2D( tAdd, vUv ).rgb, 1.0 );
}`;

// Log-luminance with centre weighting, into a small target.
const LUM_FRAG = /* glsl */ `
uniform sampler2D tSrc;
varying vec2 vUv;
void main() {
  float l = dot( texture2D( tSrc, vUv ).rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
  vec2 c = vUv * 2.0 - 1.0;
  float w = exp( -dot( c, c ) * 1.6 ) + 0.15; // centre-weighted metering
  gl_FragColor = vec4( log2( l + 1e-4 ) * w, w, 0.0, 1.0 );
}`;

// 32x32 -> 1x1 average, then adapt towards the target exposure.
const ADAPT_FRAG = /* glsl */ `
uniform sampler2D tLum;
uniform sampler2D tPrev;
uniform float uDt;
uniform float uKey;
uniform float uMinExp;
uniform float uMaxExp;
uniform float uSpeedUp;
uniform float uSpeedDown;
uniform bool uReset;
varying vec2 vUv;
void main() {
  vec2 s = vec2( 0.0 );
  for ( int y = 0; y < 32; y ++ ) for ( int x = 0; x < 32; x ++ ) s += texelFetch( tLum, ivec2( x, y ), 0 ).rg;
  float avgLog = s.x / max( s.y, 1e-4 );
  float target = clamp( log2( uKey ) - avgLog, log2( uMinExp ), log2( uMaxExp ) );
  float prev = texelFetch( tPrev, ivec2( 0 ), 0 ).r;
  // Opening up (scene got darker) is slower than stopping down, like a real camera.
  float speed = target > prev ? uSpeedUp : uSpeedDown;
  float ev = uReset ? target : prev + ( target - prev ) * ( 1.0 - exp( -uDt * speed ) );
  gl_FragColor = vec4( ev, exp2( avgLog ), 0.0, 1.0 );
}`;

const TONEMAP_FRAG = /* glsl */ `
#include <common>
#include <tonemapping_pars_fragment>
uniform sampler2D tScene;
uniform sampler2D tVol;
uniform bool uUseVol;
uniform sampler2D tBloom;
uniform sampler2D tExposure;
uniform float uBloom;
uniform float uBloomScale;
uniform float uExposureBias;
uniform float uManualExposure;
uniform float uContrast;
varying vec2 vUv;
void main() {
  vec3 c = texture2D( tScene, vUv ).rgb;
  if ( uUseVol ) {
    vec4 v = texture2D( tVol, vUv );
    c = c * v.a + v.rgb;
  }
  c = c * ( 1.0 - uBloom ) + texture2D( tBloom, vUv ).rgb * ( uBloom * uBloomScale );
  float ev = uManualExposure > 0.0 ? log2( uManualExposure ) : texelFetch( tExposure, ivec2( 0 ), 0 ).r;
  c *= exp2( ev + uExposureBias );
  c = AgXToneMapping( c );
  vec4 o = sRGBTransferOETF( vec4( c, 1.0 ) );
  // A touch of S-curve on top of AgX's gentle base look: deeper blacks, firmer mids.
  o.rgb = mix( o.rgb, o.rgb * o.rgb * ( 3.0 - 2.0 * o.rgb ), uContrast );
  gl_FragColor = o;
}`;

export class PostFX {
  readonly sceneRT: THREE.WebGLRenderTarget;
  private volRT: THREE.WebGLRenderTarget;
  private ldrRT: THREE.WebGLRenderTarget;
  private down: THREE.WebGLRenderTarget[] = [];
  private up: THREE.WebGLRenderTarget[] = [];
  private lumRT: THREE.WebGLRenderTarget;
  private adapt: THREE.WebGLRenderTarget[];
  private adaptIndex = 0;
  private quad = new FullScreenQuad();
  private volMat: THREE.ShaderMaterial | null = null;
  private downMat: THREE.ShaderMaterial;
  private upMat: THREE.ShaderMaterial;
  private lumMat: THREE.ShaderMaterial;
  private adaptMat: THREE.ShaderMaterial;
  private toneMat: THREE.ShaderMaterial;
  private spots: VolumetricSpot[] = [];
  private moon: MoonShafts | null = null;
  private frame = 0;
  private resetExposure = true;
  private width = 1;
  private height = 1;

  volumetrics = true;
  /** Scattering coefficient per metre (dusty warehouse air). */
  scatter = 0.009;
  extinction = 0.005;
  anisotropy = 0.55;
  volumeSteps = 20;
  bloomStrength = 0.06;
  bloomLevels = 6;
  exposureKey = 0.06;
  exposureBias = 0;
  minExposure = 0.3;
  maxExposure = 3.5;
  contrast = 0.35;
  /** > 0 disables auto exposure. */
  manualExposure = 0;

  constructor(
    private renderer: THREE.WebGLRenderer,
    samples = 4,
  ) {
    this.sceneRT = hdrTarget(1, 1, { depthBuffer: true, samples, depthTexture: new THREE.DepthTexture(1, 1, THREE.UnsignedIntType) });
    this.volRT = hdrTarget(1, 1);
    this.ldrRT = new THREE.WebGLRenderTarget(1, 1, { type: THREE.UnsignedByteType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.lumRT = hdrTarget(32, 32, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.adapt = [0, 1].map(() => hdrTarget(1, 1, { type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }));
    const mat = (fragmentShader: string, uniforms: Record<string, THREE.IUniform>) =>
      new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader, uniforms, depthTest: false, depthWrite: false });
    this.downMat = mat(DOWN_FRAG, {
      tSrc: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uKaris: { value: false },
      tVol: { value: null },
      uUseVol: { value: false },
    });
    this.upMat = mat(UP_FRAG, { tSrc: { value: null }, tAdd: { value: null }, uTexel: { value: new THREE.Vector2() }, uRadius: { value: 1 } });
    this.lumMat = mat(LUM_FRAG, { tSrc: { value: null } });
    this.adaptMat = mat(ADAPT_FRAG, {
      tLum: { value: this.lumRT.texture },
      tPrev: { value: null },
      uDt: { value: 0 },
      uKey: { value: 0.13 },
      uMinExp: { value: 0.35 },
      uMaxExp: { value: 9 },
      uSpeedUp: { value: 1.1 },
      uSpeedDown: { value: 2.6 },
      uReset: { value: true },
    });
    this.toneMat = mat(TONEMAP_FRAG, {
      tScene: { value: this.sceneRT.texture },
      tVol: { value: this.volRT.texture },
      uUseVol: { value: false },
      tBloom: { value: null },
      tExposure: { value: null },
      uBloom: { value: 0.06 },
      uBloomScale: { value: 1 },
      uExposureBias: { value: 0 },
      uManualExposure: { value: 0 },
      uContrast: { value: 0.35 },
      toneMappingExposure: { value: 1 },
    });
  }

  /** Texture holding (log2 exposure, average luminance) in its single texel. */
  get exposureTexture() {
    return this.adapt[this.adaptIndex].texture;
  }

  get output() {
    return this.ldrRT.texture;
  }

  /** Rebuild the volumetric shader (after changing volumeSteps). */
  invalidate() {
    this.volMat?.dispose();
    this.volMat = null;
  }

  setVolumetricLights(spots: VolumetricSpot[], moon: MoonShafts | null) {
    this.spots = spots;
    this.moon = moon;
    this.volMat?.dispose();
    this.volMat = null;
  }

  setSize(width: number, height: number) {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    this.sceneRT.setSize(this.width, this.height);
    this.ldrRT.setSize(this.width, this.height);
    this.volRT.setSize(Math.max(1, this.width >> 1), Math.max(1, this.height >> 1));
    for (const t of [...this.down, ...this.up]) t.dispose();
    this.down = [];
    this.up = [];
    let w = this.width >> 1;
    let h = this.height >> 1;
    for (let i = 0; i < this.bloomLevels && w >= 2 && h >= 2; i++) {
      this.down.push(hdrTarget(w, h));
      this.up.push(hdrTarget(w, h));
      w >>= 1;
      h >>= 1;
    }
  }

  private pass(material: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget | null) {
    this.quad.material = material;
    this.renderer.setRenderTarget(target);
    this.quad.render(this.renderer);
  }

  private buildVolumetric() {
    const n = this.spots.length;
    const u: Record<string, THREE.IUniform> = {
      tDepth: { value: this.sceneRT.depthTexture },
      uProjInv: { value: new THREE.Matrix4() },
      uViewInv: { value: new THREE.Matrix4() },
      uCamPos: { value: new THREE.Vector3() },
      uScatter: { value: this.scatter },
      uExtinction: { value: this.extinction },
      uG: { value: this.anisotropy },
      uFrame: { value: 0 },
      uMaxDist: { value: 28 },
      uTime: { value: 0 },
      uSpotPos: { value: Array.from({ length: n }, () => new THREE.Vector3()) },
      uSpotDir: { value: Array.from({ length: n }, () => new THREE.Vector3()) },
      uSpotColor: { value: Array.from({ length: n }, () => new THREE.Color()) },
      uSpotCone: { value: Array.from({ length: n }, () => new THREE.Vector2()) },
      uSpotMatrix: { value: Array.from({ length: n }, () => new THREE.Matrix4()) },
      uSpotBias: { value: new Array<number>(n).fill(0) },
      uSpotShadow: { value: new Array<THREE.Texture | null>(n).fill(null) },
      uMoonDir: { value: new THREE.Vector3() },
      uMoonColor: { value: new THREE.Color() },
      uMoonMatrix: { value: new THREE.Matrix4() },
      uMoonShadow: { value: null },
    };
    this.volMat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: volumetricShader(n, !!this.moon, this.volumeSteps),
      uniforms: u,
      depthTest: false,
      depthWrite: false,
    });
  }

  private updateVolumetric(camera: THREE.PerspectiveCamera, time: number) {
    if (!this.volMat) this.buildVolumetric();
    const u = this.volMat!.uniforms;
    u.uProjInv.value.copy(camera.projectionMatrixInverse);
    u.uViewInv.value.copy(camera.matrixWorld);
    u.uCamPos.value.setFromMatrixPosition(camera.matrixWorld);
    u.uScatter.value = this.scatter;
    u.uExtinction.value = this.extinction;
    u.uG.value = this.anisotropy;
    u.uFrame.value = this.frame % 64;
    u.uTime.value = time;
    const dir = new THREE.Vector3();
    this.spots.forEach((s, i) => {
      const L = s.light;
      L.updateMatrixWorld();
      L.target.updateMatrixWorld();
      u.uSpotPos.value[i].setFromMatrixPosition(L.matrixWorld);
      dir.setFromMatrixPosition(L.target.matrixWorld).sub(u.uSpotPos.value[i]).normalize();
      u.uSpotDir.value[i].copy(dir);
      const on = L.visible && L.intensity > 0;
      u.uSpotColor.value[i].copy(L.color).multiplyScalar(on ? L.intensity * s.strength : 0);
      u.uSpotCone.value[i].set(Math.cos(L.angle), Math.cos(L.angle * (1 - L.penumbra)));
      u.uSpotMatrix.value[i].copy(L.shadow.matrix);
      u.uSpotBias.value[i] = L.shadow.bias;
      u.uSpotShadow.value[i] = L.castShadow ? (L.shadow.map?.depthTexture ?? null) : null;
      // A lamp without a shadow map yet: park its matrix so every lookup is "lit".
      if (!u.uSpotShadow.value[i]) u.uSpotMatrix.value[i].makeScale(0, 0, 0).setPosition(-1, -1, 2);
    });
    if (this.moon) {
      u.uMoonDir.value.copy(this.moon.direction);
      u.uMoonColor.value.copy(this.moon.color);
      u.uMoonMatrix.value.copy(this.moon.matrix);
      u.uMoonShadow.value = this.moon.shadow;
    }
  }

  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, dt: number, time: number, final: THREE.ShaderMaterial) {
    const r = this.renderer;
    this.frame++;
    r.setRenderTarget(this.sceneRT);
    r.render(scene, camera);

    const useVol = this.volumetrics && (this.spots.length > 0 || !!this.moon);
    if (useVol) {
      this.updateVolumetric(camera, time);
      this.pass(this.volMat!, this.volRT);
    }

    // Bloom chain (the first downsample also folds in the volumetrics).
    const dm = this.downMat.uniforms;
    for (let i = 0; i < this.down.length; i++) {
      const src = i === 0 ? this.sceneRT : this.down[i - 1];
      dm.tSrc.value = src.texture;
      dm.uTexel.value.set(1 / src.width, 1 / src.height);
      dm.uKaris.value = i === 0;
      dm.uUseVol.value = i === 0 && useVol;
      dm.tVol.value = this.volRT.texture;
      this.pass(this.downMat, this.down[i]);
    }
    const um = this.upMat.uniforms;
    const last = this.down.length - 1;
    for (let i = last - 1; i >= 0; i--) {
      const src = i === last - 1 ? this.down[last] : this.up[i + 1];
      um.tSrc.value = src.texture;
      um.tAdd.value = this.down[i].texture;
      um.uTexel.value.set(1 / src.width, 1 / src.height);
      this.pass(this.upMat, this.up[i]);
    }
    const bloomTex = this.down.length > 1 ? this.up[0].texture : this.down[0]?.texture;

    // Auto exposure from the 1/8-resolution level.
    const meterSrc = this.down[Math.min(2, this.down.length - 1)];
    this.lumMat.uniforms.tSrc.value = meterSrc.texture;
    this.pass(this.lumMat, this.lumRT);
    const prev = this.adapt[this.adaptIndex];
    this.adaptIndex ^= 1;
    const am = this.adaptMat.uniforms;
    am.tPrev.value = prev.texture;
    am.uDt.value = dt;
    am.uKey.value = this.exposureKey;
    am.uMinExp.value = this.minExposure;
    am.uMaxExp.value = this.maxExposure;
    am.uReset.value = this.resetExposure;
    this.resetExposure = false;
    this.pass(this.adaptMat, this.adapt[this.adaptIndex]);

    const tm = this.toneMat.uniforms;
    tm.uUseVol.value = useVol;
    tm.tBloom.value = bloomTex;
    tm.tExposure.value = this.adapt[this.adaptIndex].texture;
    tm.uBloom.value = this.bloomStrength;
    // The up chain sums every level: normalise so the bloom carries the scene's energy.
    tm.uBloomScale.value = 1 / Math.max(1, this.down.length);
    tm.uExposureBias.value = this.exposureBias;
    tm.uManualExposure.value = this.manualExposure;
    tm.uContrast.value = this.contrast;
    this.pass(this.toneMat, this.ldrRT);

    final.uniforms.tDiffuse.value = this.ldrRT.texture;
    if (final.uniforms.tExposure) final.uniforms.tExposure.value = this.adapt[this.adaptIndex].texture;
    this.pass(final, null);
  }

  resetAdaptation() {
    this.resetExposure = true;
  }
}
