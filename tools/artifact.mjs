// Package dist/ for hosts that only serve common web types (e.g. a claude.ai
// Artifact): inlines the CSS, strips the document skeleton, converts .glb
// models to embedded glTF JSON (.gltf.json) and points assets.json at them.
//
//   npm run build && node tools/artifact.mjs [outDir]
// Prints the published-path -> source-path map for the upload.
import { readFileSync, writeFileSync, mkdirSync, readdirSync, cpSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';

const dist = resolve('dist');
const out = resolve(process.argv[2] || 'dist-artifact');
rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, 'models'), { recursive: true });

const html = readFileSync(join(dist, 'index.html'), 'utf8');
const cssHref = html.match(/<link rel="stylesheet" crossorigin href="\.\/(assets\/[^"]+\.css)">/)[1];
const jsSrc = html.match(/<script type="module" crossorigin src="\.\/(assets\/[^"]+\.js)"><\/script>/)[1];
const fonts = [...html.matchAll(/<link rel="(?:preconnect|stylesheet)" href="https:\/\/fonts[^>]*>/g)].map((m) => m[0]);
const body = html.match(/<body>([\s\S]*)<\/body>/)[1].trim();
const title = html.match(/<title>[^<]*<\/title>/)[0];
const css = readFileSync(join(dist, cssHref), 'utf8');
writeFileSync(
  join(out, 'index.html'),
  `${title}\n${fonts.join('\n')}\n<style>\n${css}\n</style>\n${body}\n<script type="module" crossorigin src="./${jsSrc}"></script>\n`,
);

const files = {};
cpSync(join(dist, 'assets'), join(out, 'assets'), { recursive: true });
files[jsSrc] = join(out, jsSrc);
cpSync(join(dist, 'textures'), join(out, 'textures'), { recursive: true });
for (const f of readdirSync(join(out, 'textures'))) files[`textures/${f}`] = join(out, 'textures', f);

// GLB -> glTF JSON with the binary chunk as a data: URI.
for (const f of readdirSync(join(dist, 'models')).filter((n) => n.endsWith('.glb'))) {
  const buf = readFileSync(join(dist, 'models', f));
  let off = 12;
  let json = null;
  let bin = null;
  while (off < buf.length) {
    const len = buf.readUInt32LE(off);
    const type = buf.readUInt32LE(off + 4);
    const chunk = buf.subarray(off + 8, off + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(chunk.toString('utf8'));
    if (type === 0x004e4942) bin = chunk;
    off += 8 + len;
  }
  if (bin) json.buffers[0].uri = `data:application/octet-stream;base64,${bin.toString('base64')}`;
  const name = f.replace(/\.glb$/, '.gltf.json');
  writeFileSync(join(out, 'models', name), JSON.stringify(json));
  files[`models/${name}`] = join(out, 'models', name);
}

const manifest = JSON.parse(readFileSync(join(dist, 'assets.json'), 'utf8'));
manifest.modelSuffix = '.gltf.json';
writeFileSync(join(out, 'assets.json'), JSON.stringify(manifest, null, 2) + '\n');
files['assets.json'] = join(out, 'assets.json');

writeFileSync(join(out, 'files.json'), JSON.stringify(files, null, 1));
console.log(`packaged ${Object.keys(files).length} files + index.html into ${out}`);
