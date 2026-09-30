import * as THREE from 'three';

/**
 * Shader patches for the baked level (see blender/build_level.py for the bake side).
 *
 * - Lightmapped and vertex-lit surfaces take their indirect light (and all light
 *   from the baked-only sources: moon, street light, office panel) from the bake,
 *   and never from the environment probe.
 * - Reflections come from one HDR probe, box-projected onto the hall so they sit
 *   in the right place, then "normalised": a reflection is dimmed where the
 *   surface itself is darker than what the probe saw (corners, under racks,
 *   inside the office), which is what keeps a single probe from glowing everywhere.
 * - The mask texture adds grime, specular occlusion and, on the floor, damp
 *   patches and standing puddles (darker, glossier, flat normal).
 * - Emissive bulbs dim together with the lamp they belong to when it flickers.
 */

export const MAX_FLICKER = 4;

/** Shared by every patched material, so updating a value here updates the whole level. */
export const levelUniforms = {
  uProbePos: { value: new THREE.Vector3() },
  uBoxMin: { value: new THREE.Vector3(-1e3, -1e3, -1e3) },
  uBoxMax: { value: new THREE.Vector3(1e3, 1e3, 1e3) },
  /** Local-to-probe irradiance ratio that counts as "as bright as the probe". */
  uNormScale: { value: 2.0 },
  uNormMin: { value: 0.04 },
  uWetness: { value: 1.0 },
  uGrime: { value: 1.0 },
  uLevelMask: { value: null as THREE.Texture | null },
  uVertexLight: { value: 1.0 },
  uFlickerPos: { value: Array.from({ length: MAX_FLICKER }, () => new THREE.Vector3(1e4, 1e4, 1e4)) },
  uFlicker: { value: new Array<number>(MAX_FLICKER).fill(1) },
  uTime: { value: 0 },
  /** 0 normal; 1 baked/ambient diffuse; 2 reflections; 3 real-time direct; 4 albedo; 5 roughness. */
  uDebug: { value: 0 },
};

export type LevelShadingKind =
  /** Static, lightmapped (uv1 + mask). */
  | 'lightmap'
  /** Static, lit by the baked per-vertex light in the `bakedLight` attribute. */
  | 'vertex'
  /** Anything that moves: real-time lights + box-projected probe. */
  | 'dynamic';

export interface LevelShadingOptions {
  /** Floor: damp patches and puddles from the mask's red channel. */
  wet?: boolean;
  /** Emissive surfaces that dim with a flickering lamp nearby. */
  flicker?: boolean;
}

const LUMA = 'vec3( 0.2126, 0.7152, 0.0722 )';

const vertexHead = /* glsl */ `
varying vec3 vLevelPos;
#ifdef LEVEL_VTX
attribute vec4 bakedLight;
varying vec3 vBakedLight;
#endif
`;

const vertexBody = /* glsl */ `
vLevelPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
#ifdef LEVEL_VTX
// Stored as sqrt() so 8-bit vertex colours keep the darks.
vBakedLight = bakedLight.rgb * bakedLight.rgb;
#endif
`;

const fragmentHead = /* glsl */ `
varying vec3 vLevelPos;
uniform vec3 uProbePos;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform float uNormScale;
uniform float uNormMin;
uniform float uWetness;
uniform float uGrime;
uniform float uTime;
uniform int uDebug;
#ifdef LEVEL_MASK
uniform sampler2D uLevelMask;
#endif
#ifdef LEVEL_VTX
varying vec3 vBakedLight;
uniform float uVertexLight;
#endif
#ifdef LEVEL_FLICKER
uniform vec3 uFlickerPos[ ${MAX_FLICKER} ];
uniform float uFlicker[ ${MAX_FLICKER} ];
#endif

// Irradiance from the real-time level lamps (not the flashlight), accumulated in the light loop.
vec3 gLevelDirect = vec3( 0.0 );

vec3 levelBoxProject( vec3 dir, float roughness ) {
	vec3 tMax = ( uBoxMax - vLevelPos ) / dir;
	vec3 tMin = ( uBoxMin - vLevelPos ) / dir;
	vec3 t = max( tMax, tMin );
	float d = max( min( min( t.x, t.y ), t.z ), 0.0 );
	vec3 projected = normalize( vLevelPos + dir * d - uProbePos );
	return normalize( mix( projected, dir, roughness * roughness ) );
}
`;

// Grime and wetness, right after the albedo map is applied.
const afterMap = /* glsl */ `
float levelWet = 0.0;
float levelGrime = 0.0;
float levelSpecOcc = 1.0;
#ifdef LEVEL_MASK
	vec3 levelMaskTexel = texture2D( uLevelMask, vLightMapUv ).rgb;
	levelGrime = levelMaskTexel.g * uGrime;
	levelSpecOcc = levelMaskTexel.b;
	#ifdef LEVEL_WET
		levelWet = levelMaskTexel.r * uWetness;
	#endif
#endif
float levelPuddle = smoothstep( 0.5, 0.78, levelWet );
float levelDamp = saturate( levelWet * 1.6 );
#if defined( LEVEL_STATIC )
	// Tints above 1 exist for saturated paints; keep albedo physical.
	diffuseColor.rgb = min( diffuseColor.rgb, vec3( 0.92 ) );
#endif
diffuseColor.rgb *= mix( vec3( 1.0 ), vec3( 0.6, 0.56, 0.5 ), levelGrime );
// Porous concrete darkens when damp; standing water darkens it a little more.
diffuseColor.rgb *= 1.0 - 0.42 * levelDamp - 0.18 * levelPuddle;
`;

const afterRoughness = /* glsl */ `
roughnessFactor = mix( roughnessFactor, min( roughnessFactor + 0.18, 1.0 ), levelGrime );
roughnessFactor = mix( roughnessFactor, roughnessFactor * 0.45, levelDamp );
roughnessFactor = mix( roughnessFactor, 0.03, levelPuddle );
`;

const afterNormal = /* glsl */ `
#ifdef LEVEL_WET
	normal = normalize( mix( normal, nonPerturbedNormal, max( levelPuddle, levelDamp * 0.35 ) ) );
#endif
`;

const lightsMaps = /* glsl */ `
vec3 levelBaked = vec3( 0.0 );
#if defined( RE_IndirectDiffuse )
	#if defined( USE_LIGHTMAP )
		levelBaked = texture2D( lightMap, vLightMapUv ).rgb * lightMapIntensity;
		irradiance += levelBaked;
	#elif defined( LEVEL_VTX )
		levelBaked = vBakedLight * uVertexLight;
		irradiance += levelBaked;
	#elif defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	vec3 iblRadiance = getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#if defined( LEVEL_STATIC ) && defined( ENVMAP_TYPE_CUBE_UV )
		vec3 levelWorldN = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		float levelProbe = dot( PI * textureCubeUV( envMap, envMapRotation * levelWorldN, 1.0 ).rgb, ${LUMA} );
		float levelLocal = dot( levelBaked + gLevelDirect, ${LUMA} );
		float levelNorm = clamp( uNormScale * levelLocal / max( levelProbe, 1e-5 ), uNormMin, 1.0 );
		// Mirror-like standing water shows the box-projected probe as it is: its directional content is right.
		iblRadiance *= mix( levelNorm, 1.0, levelPuddle * 0.75 );
	#endif
	iblRadiance *= computeSpecularOcclusion( saturate( dot( geometryNormal, geometryViewDir ) ), levelSpecOcc, material.roughness );
	radiance += iblRadiance;
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif
`;

const flickerEmissive = /* glsl */ `
#ifdef LEVEL_FLICKER
	for ( int i = 0; i < ${MAX_FLICKER}; i ++ ) {
		totalEmissiveRadiance *= mix( uFlicker[ i ], 1.0, smoothstep( 0.35, 0.6, distance( vLevelPos, uFlickerPos[ i ] ) ) );
	}
#endif
`;

const debugView = /* glsl */ `
if ( uDebug == 1 ) outgoingLight = reflectedLight.indirectDiffuse;
else if ( uDebug == 2 ) outgoingLight = reflectedLight.indirectSpecular;
else if ( uDebug == 3 ) outgoingLight = reflectedLight.directDiffuse + reflectedLight.directSpecular;
else if ( uDebug == 4 ) outgoingLight = diffuseColor.rgb;
else if ( uDebug == 5 ) outgoingLight = vec3( material.roughness );
`;

function patchedEnvmapChunk() {
  const src = THREE.ShaderChunk.envmap_physical_pars_fragment;
  const hook = 'reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );';
  if (!src.includes(hook)) throw new Error('three envmap chunk changed: box projection hook missing');
  return src.replace(hook, `${hook}\n\t\t\t#ifdef LEVEL_BOXPROJ\n\t\t\treflectVec = levelBoxProject( reflectVec, roughness );\n\t\t\t#endif`);
}

function patchedLightsBegin() {
  let src = THREE.ShaderChunk.lights_fragment_begin;
  // Only the level lamps (distance 0) count as "local light" for reflection normalisation:
  // the flashlight lighting a wall should not make the wall reflect more of the hall.
  const spotStart = src.indexOf('#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )');
  const spotEnd = src.indexOf('#pragma unroll_loop_end', spotStart);
  if (spotStart < 0 || spotEnd < 0) throw new Error('three light loop chunk changed');
  const call = 'RE_Direct( directLight, geometryPosition,';
  const spot = src.slice(spotStart, spotEnd).replace(
    call,
    `#ifdef LEVEL_STATIC\n\t\tif ( spotLight.distance == 0.0 ) gLevelDirect += saturate( dot( geometryNormal, directLight.direction ) ) * directLight.color;\n\t\t#endif\n\t\t${call}`,
  );
  src = src.slice(0, spotStart) + spot + src.slice(spotEnd);
  // Directional and rect-area lights exist only to light moving things; static surfaces have them baked.
  src = src
    .replace('#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )', '#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct ) && !defined( LEVEL_STATIC )')
    .replace(
      '#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )',
      '#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea ) && !defined( LEVEL_STATIC )',
    );
  return src;
}

let envChunk: string | null = null;
let lightsChunk: string | null = null;

function replaceOnce(src: string, hook: string, text: string) {
  if (!src.includes(hook)) throw new Error(`shader hook missing: ${hook}`);
  return src.replace(hook, text);
}

/** Patch a standard/physical material in place. Safe to call once per material. */
export function applyLevelShading(mat: THREE.MeshStandardMaterial, kind: LevelShadingKind, opts: LevelShadingOptions = {}) {
  envChunk ??= patchedEnvmapChunk();
  lightsChunk ??= patchedLightsBegin();
  const defines: Record<string, string> = { ...(mat.defines ?? {}), LEVEL_BOXPROJ: '' };
  if (kind !== 'dynamic') defines.LEVEL_STATIC = '';
  if (kind === 'lightmap') defines.LEVEL_MASK = '';
  if (kind === 'vertex') defines.LEVEL_VTX = '';
  if (opts.wet) defines.LEVEL_WET = '';
  if (opts.flicker) defines.LEVEL_FLICKER = '';
  mat.defines = defines;
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, levelUniforms);
    let vs = shader.vertexShader;
    vs = replaceOnce(vs, '#include <common>', `#include <common>\n${vertexHead}`);
    vs = replaceOnce(vs, '#include <worldpos_vertex>', `#include <worldpos_vertex>\n${vertexBody}`);
    shader.vertexShader = vs;
    let fs = shader.fragmentShader;
    fs = replaceOnce(fs, '#include <common>', `#include <common>\n${fragmentHead}`);
    fs = replaceOnce(fs, '#include <envmap_physical_pars_fragment>', envChunk!);
    fs = replaceOnce(fs, '#include <map_fragment>', `#include <map_fragment>\n${afterMap}`);
    fs = replaceOnce(fs, '#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>\n${afterRoughness}`);
    fs = replaceOnce(fs, '#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${afterNormal}`);
    fs = replaceOnce(fs, '#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${flickerEmissive}`);
    fs = replaceOnce(fs, '#include <lights_fragment_begin>', lightsChunk!);
    fs = replaceOnce(fs, '#include <lights_fragment_maps>', lightsMaps);
    fs = replaceOnce(fs, '#include <opaque_fragment>', `${debugView}\n#include <opaque_fragment>`);
    shader.fragmentShader = fs;
  };
  mat.customProgramCacheKey = () => `level:${Object.keys(defines).sort().join(',')}`;
  mat.needsUpdate = true;
  return mat;
}
