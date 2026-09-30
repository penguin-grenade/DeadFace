import { Vector2 } from 'three';

/**
 * Final-stage bodycam lens + sensor emulation, applied in display (sRGB) space
 * after tone mapping:
 *  - barrel / fisheye distortion with per-channel chromatic aberration
 *  - rolling-shutter skew and directional motion blur from camera angular velocity
 *  - unsharp mask (bodycam firmware over-sharpens)
 *  - sensor noise that grows with the auto-exposure gain (luma + chroma), so
 *    dark rooms get grainy the way real body-worn cameras do
 *  - mild macroblocking in flat dark areas (stream compression)
 *  - vignette, desaturation, lifted blacks, soft highlight clip
 */
export const BodycamShader = {
  name: 'BodycamShader',
  uniforms: {
    tDiffuse: { value: null },
    tExposure: { value: null },
    uTime: { value: 0 },
    uResolution: { value: new Vector2(1, 1) },
    uDistortion: { value: 0.26 },
    uChromatic: { value: 0.012 },
    uBlur: { value: new Vector2(0, 0) },
    uBlurSamples: { value: 8 },
    uShutter: { value: new Vector2(0, 0) },
    uGrain: { value: 0.055 },
    uVignette: { value: 0.55 },
    uSharpen: { value: 0.35 },
    uSaturation: { value: 0.84 },
    uFlash: { value: 0 },
    uBlocks: { value: 1 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform sampler2D tExposure;
    uniform float uTime;
    uniform vec2 uResolution;
    uniform float uDistortion;
    uniform float uChromatic;
    uniform vec2 uBlur;
    uniform int uBlurSamples;
    uniform vec2 uShutter;
    uniform float uGrain;
    uniform float uVignette;
    uniform float uSharpen;
    uniform float uSaturation;
    uniform float uFlash;
    uniform float uBlocks;
    varying vec2 vUv;

    // Barrel distortion. Scaled so the corners of the output still land
    // inside the source image (no black borders).
    vec2 barrel(vec2 uv, float k) {
      vec2 p = uv * 2.0 - 1.0;
      float aspect = uResolution.x / uResolution.y;
      p.x *= aspect;
      float r2 = dot(p, p);
      float cornerR2 = aspect * aspect + 1.0;
      float fit = 1.0 / (1.0 + k * cornerR2 * 0.55);
      p *= (1.0 + k * r2 * 0.55) * fit;
      p.x /= aspect;
      return p * 0.5 + 0.5;
    }

    vec3 blurred(vec2 uv) {
      if (uBlurSamples <= 1 || dot(uBlur, uBlur) < 1e-8) {
        return texture2D(tDiffuse, uv).rgb;
      }
      vec3 acc = vec3(0.0);
      float n = float(uBlurSamples);
      for (int i = 0; i < 16; i++) {
        if (i >= uBlurSamples) break;
        float t = (float(i) / (n - 1.0)) - 0.5;
        acc += texture2D(tDiffuse, uv + uBlur * t).rgb;
      }
      return acc / n;
    }

    float hash(vec2 p) {
      p = fract(p * vec2(443.897, 441.423));
      p += dot(p, p.yx + 19.19);
      return fract((p.x + p.y) * p.x);
    }

    void main() {
      // Rolling shutter: rows are read top to bottom, so a fast pan leans verticals.
      vec2 uv = vUv;
      uv.x += uShutter.x * (0.5 - uv.y);
      uv.y += uShutter.y * (0.5 - uv.y) * 0.5;

      vec2 uvR = barrel(uv, uDistortion + uChromatic);
      vec2 uvG = barrel(uv, uDistortion);
      vec2 uvB = barrel(uv, uDistortion - uChromatic);

      vec3 col;
      col.r = blurred(uvR).r;
      col.g = blurred(uvG).g;
      col.b = blurred(uvB).b;

      // Unsharp mask on the green-channel sample position.
      vec2 px = 1.0 / uResolution;
      vec3 nb = texture2D(tDiffuse, uvG + vec2(px.x, 0.0)).rgb
              + texture2D(tDiffuse, uvG - vec2(px.x, 0.0)).rgb
              + texture2D(tDiffuse, uvG + vec2(0.0, px.y)).rgb
              + texture2D(tDiffuse, uvG - vec2(0.0, px.y)).rgb;
      col += (col - nb * 0.25) * uSharpen;

      // Stream compression: flat, dark regions collapse towards their 8x8 block average.
      if (uBlocks > 0.0) {
        vec2 cell = floor(vUv * uResolution / 8.0);
        vec2 cuv = (cell + 0.5) * 8.0 / uResolution;
        vec3 blk = texture2D(tDiffuse, barrel(cuv, uDistortion)).rgb;
        float flatness = 1.0 - smoothstep(0.015, 0.06, length(col - blk));
        float dark = 1.0 - smoothstep(0.08, 0.3, dot(blk, vec3(0.299, 0.587, 0.114)));
        col = mix(col, blk, flatness * dark * 0.35 * uBlocks);
      }

      // Sensor response: desaturate, lift blacks, soft clip highlights.
      float luma = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(vec3(luma), col, uSaturation);
      col = col * 0.96 + 0.016;
      col += uFlash * vec3(1.0, 0.85, 0.6) * 0.25;

      // Vignette on the distorted frame.
      vec2 v = vUv * 2.0 - 1.0;
      float vig = 1.0 - uVignette * pow(dot(v * vec2(0.85, 1.0), v * vec2(0.85, 1.0)) * 0.5, 1.4);
      col *= vig;

      // Noise scales with the auto-exposure gain: a dark room pushed bright is grainy.
      float ev = texelFetch(tExposure, ivec2(0), 0).r;
      float gain = clamp(ev / 3.0, 0.0, 1.2);
      float t = fract(uTime * 13.37) * 1000.0;
      float g = hash(vUv * uResolution + t) - 0.5;
      vec3 chroma = vec3(hash(vUv * uResolution + t + 17.0), hash(vUv * uResolution + t + 41.0), hash(vUv * uResolution + t + 73.0)) - 0.5;
      float amt = uGrain * (0.45 + 1.1 * gain) * (1.35 - luma);
      col += g * amt + chroma * amt * 0.45 * gain;

      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }
  `,
};

/** Straight copy to the screen (debug views without the bodycam look). */
export const CopyShader = {
  name: 'CopyShader',
  uniforms: { tDiffuse: { value: null } },
  vertexShader: BodycamShader.vertexShader,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    varying vec2 vUv;
    void main() { gl_FragColor = texture2D(tDiffuse, vUv); }
  `,
};
