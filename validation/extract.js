// Extracts the composed WGSL strings from main.js without running the app.
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'main.js'), 'utf8');
const marker = '// Tiny column-major mat4 helpers';
const head = src.slice(0, src.indexOf(marker));
const fn = new Function(head + '\nreturn { N, SUBSTEPS, DT, WGSL_SIM, WGSL_RENDER };');
const out = fn();
fs.writeFileSync(path.join(__dirname, 'sim.wgsl'), out.WGSL_SIM);
fs.writeFileSync(path.join(__dirname, 'render.wgsl'), out.WGSL_RENDER);
console.log(JSON.stringify({ N: out.N, SUBSTEPS: out.SUBSTEPS, DT: out.DT }));
