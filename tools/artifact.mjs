// Package dist/ for locked-down static hosts (e.g. a claude.ai Artifact), which serve only common
// web types and let the page fetch() nothing but its own files: inlines the CSS, strips the
// document skeleton, wraps .glb and .hdr files as base64 JSON strings (the game decodes them
// itself rather than through data: or blob: requests), gives every asset a content-hashed name so
// a cached copy from an older publish can't be picked up, and embeds the asset manifest, with the
// map from each public/ path to its published name, in the page.
//
//   npm run build && node tools/artifact.mjs [outDir]
// Writes outDir/files.json, the published-path -> source-path map for the upload.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const dist = resolve('dist');
const out = resolve(process.argv[2] || 'dist-artifact');
rmSync(out, { recursive: true, force: true });

const html = readFileSync(join(dist, 'index.html'), 'utf8');
const cssHref = html.match(/<link rel="stylesheet" crossorigin href="\.\/(assets\/[^"]+\.css)">/)[1];
const jsSrc = html.match(/<script type="module" crossorigin src="\.\/(assets\/[^"]+\.js)"><\/script>/)[1];
const fonts = [...html.matchAll(/<link rel="(?:preconnect|stylesheet)" href="https:\/\/fonts[^>]*>/g)].map((m) => m[0]);
const body = html.match(/<body>([\s\S]*)<\/body>/)[1].trim();
const title = html.match(/<title>[^<]*<\/title>/)[0];
const css = readFileSync(join(dist, cssHref), 'utf8');

const files = {}; // published path -> source file
const renamed = {}; // public/ path -> published path
const put = (published, data) => {
  const file = join(out, published);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, data);
  files[published] = file;
};
// "models/pistol.glb" -> "models/pistol.<hash>.glb.json" holding the file as a base64 string.
const publish = (path) => {
  const data = readFileSync(join(dist, path));
  const hash = createHash('sha256').update(data).digest('hex').slice(0, 10);
  const wrap = /\.(glb|hdr)$/.test(path);
  const name = path.replace(/(\.[^./]+)$/, `.${hash}$1`) + (wrap ? '.json' : '');
  put(name, wrap ? JSON.stringify(data.toString('base64')) : data);
  renamed[path] = name;
};
const walk = (dir) =>
  readdirSync(join(dist, dir)).flatMap((f) => (statSync(join(dist, dir, f)).isDirectory() ? walk(`${dir}/${f}`) : [`${dir}/${f}`]));

for (const dir of ['models', 'textures', 'level']) {
  try {
    walk(dir).forEach(publish);
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
}
put(jsSrc, readFileSync(join(dist, jsSrc))); // Vite already hashed the bundle's name

const manifest = { ...JSON.parse(readFileSync(join(dist, 'assets.json'), 'utf8')), files: renamed };
const manifestTag = `<script type="application/json" id="asset-manifest">${JSON.stringify(manifest)}</script>`;
writeFileSync(
  join(out, 'index.html'),
  `${title}\n${fonts.join('\n')}\n<style>\n${css}\n</style>\n${body}\n${manifestTag}\n<script type="module" crossorigin src="./${jsSrc}"></script>\n`,
);

writeFileSync(join(out, 'files.json'), JSON.stringify(files, null, 1));
const mb = Object.values(files).reduce((s, f) => s + statSync(f).size, 0) / 2 ** 20;
console.log(`packaged ${Object.keys(files).length} files (${mb.toFixed(1)} MB) + index.html into ${out}`);
