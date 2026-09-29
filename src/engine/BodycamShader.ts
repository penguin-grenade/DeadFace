import { Vector2 } from 'three';

/**
 * Final-stage bodycam lens + sensor emulation, applied in display (sRGB) space
 * after tone mapping:
 *  - barrel / fisheye distortion with per-channel chromatic aberration
 *  - directional motion blur driven by camera angular velocity
 *  - cheap unsharp mask (bodycam firmware over-sharpens)
 *  - luma-weighted sensor grain, vignette, slight desaturation, highlight clip
 */
export const BodycamShader = {
  name: 'BodycamShader',
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uResolution: { value: new Vector2(1, 1) },
    uDistortion: { value: 0.26 },
    uChromatic: { value: 0.012 },
    uBlur: { value: new Vector2(0, 0) },
    uBlurSamples: { value: 8 },
    uGrain: { value: 0.07 },
    uVignette: { value: 0.55 },
    uSharpen: { value: 0.35 },
    uSaturation: { value: 0.82 },
    uFlash: { value: 0 },
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
    uniform float uTime;
    uniform vec2 uResolution;
    uniform float uDistortion;
    uniform float uChromatic;
    uniform vec2 uBlur;
    uniform int uBlurSamples;
    uniform float uGrain;
    uniform float uVignette;
    uniform float uSharpen;
    uniform float uSaturation;
    uniform float uFlash;
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
      vec2 uvR = barrel(vUv, uDistortion + uChromatic);
      vec2 uvG = barrel(vUv, uDistortion);
      vec2 uvB = barrel(vUv, uDistortion - uChromatic);

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

      // Sensor response: desaturate, lift blacks, soft clip highlights.
      float luma = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(vec3(luma), col, uSaturation);
      col = col * 0.96 + 0.018;
      col += uFlash * vec3(1.0, 0.85, 0.6) * 0.25;

      // Vignette on the distorted frame.
      vec2 v = vUv * 2.0 - 1.0;
      float vig = 1.0 - uVignette * pow(dot(v * vec2(0.85, 1.0), v * vec2(0.85, 1.0)) * 0.5, 1.4);
      col *= vig;

      // Grain: stronger in shadows (sensor gain), animated per frame.
      float g = hash(vUv * uResolution + fract(uTime * 13.37) * 1000.0) - 0.5;
      float gAmt = uGrain * (1.4 - luma);
      col += g * gAmt;

      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }
  `,
};
