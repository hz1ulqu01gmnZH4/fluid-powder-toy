// Extracts the composed WGSL strings from main.js without running the app.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'main.js'), 'utf8');
const marker = '// Tiny column-major mat4 helpers';
const cut = src.indexOf(marker);
if (cut === -1) throw new Error(`marker comment ${JSON.stringify(marker)} not found in main.js`);
const head = src.slice(0, cut);
const fn = new Function(head + '\nreturn { N, SUBSTEPS, DT, MAX_SOURCES, NK, WGSL_SIM, WGSL_RENDER, WGSL_MASS };');
const out = fn();
fs.writeFileSync(path.join(__dirname, 'sim.wgsl'), out.WGSL_SIM);
fs.writeFileSync(path.join(__dirname, 'render.wgsl'), out.WGSL_RENDER);
fs.writeFileSync(path.join(__dirname, 'mass.wgsl'), out.WGSL_MASS);
console.log(JSON.stringify({ N: out.N, SUBSTEPS: out.SUBSTEPS, DT: out.DT, MAX_SOURCES: out.MAX_SOURCES, NK: out.NK }));
