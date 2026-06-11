'use strict';

// ============================================================================
// Hydra Terra — WebGPU fluid & erosion sandbox
//
// Simulation: heightfield shallow-water "virtual pipes" model (Mei et al. 2007)
// with hydraulic erosion (capacity / dissolve / deposit + semi-Lagrangian
// sediment advection) and thermal erosion (talus slippage). Everything lives
// in GPU storage buffers; the CPU never reads the field back. Mouse picking is
// done by a single-thread compute raymarch that writes the hit point into a
// small buffer which the pour pass and the cursor-ring shader read directly.
// ============================================================================

const N = 256;                 // sim + mesh grid resolution (cells)
const SUBSTEPS = 3;            // sim substeps per frame
const DT = 0.055;              // sim timestep per substep

// ---------------------------------------------------------------------------
// WGSL — shared helper snippets (templated into both modules)
// ---------------------------------------------------------------------------

const WGSL_COMMON = /* wgsl */`
const N : u32 = ${N}u;
const NI : i32 = ${N};
const NF : f32 = ${N}.0;
const PI : f32 = 3.14159265;

fn cIdx(x : i32, y : i32) -> u32 {
  return u32(clamp(y, 0, NI - 1)) * N + u32(clamp(x, 0, NI - 1));
}

// --- value noise / fbm -----------------------------------------------------
fn hash21(p : vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q = q + dot(q, q + vec2f(45.32));
  return fract(q.x * q.y);
}
fn vnoise(p : vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm(p0 : vec2f) -> f32 {
  var p = p0;
  var amp = 0.5;
  var s = 0.0;
  for (var i = 0; i < 5; i++) {
    s += amp * vnoise(p);
    p = p * 2.03 + vec2f(17.3, 9.1);
    amp *= 0.5;
  }
  return s;
}
fn ridgedFbm(p0 : vec2f) -> f32 {
  var p = p0;
  var amp = 0.55;
  var s = 0.0;
  var w = 1.0;
  for (var i = 0; i < 6; i++) {
    var n = 1.0 - abs(2.0 * vnoise(p) - 1.0);
    n = n * n * w;
    w = clamp(n * 1.6, 0.0, 1.0);
    s += amp * n;
    p = p * 2.11 + vec2f(31.7, 4.3);
    amp *= 0.5;
  }
  return s;
}

// Procedural island height — used by terrain generation.
fn genHeight(gp : vec2f, seed : f32) -> f32 {
  let uv = gp / NF;
  let c = uv - vec2f(0.5);
  let off = vec2f(seed * 37.71, seed * 91.27);
  let p = uv * 5.0 + off;
  let warp = vec2f(fbm(p + vec2f(0.0, 5.2)), fbm(p + vec2f(3.1, 1.7))) * 1.7;
  var m = ridgedFbm(p * 0.85 + warp * 0.7);
  m = pow(max(m, 0.0), 1.55);
  let rolling = fbm(p * 0.55 + warp) * 0.55;
  let fall = 1.0 - smoothstep(0.30, 0.92, length(c) * 2.0);
  var h = (m * 30.0 + rolling * 9.0) * fall + fall * 2.5;
  return h;
}
`;

// ---------------------------------------------------------------------------
// WGSL — simulation module (all compute passes share one bind group)
// ---------------------------------------------------------------------------

const WGSL_SIM = /* wgsl */`
${WGSL_COMMON}

struct SimU {
  rayOrigin : vec4f,           // xyz origin, w unused
  rayDir    : vec4f,           // xyz dir, w = hover flag (cast pick or not)
  dt        : f32,
  time      : f32,
  rate      : f32,             // pour rate (depth/sec at brush center)
  radius    : f32,             // brush radius in cells
  pourActive: f32,
  seed      : f32,
  pad0      : f32,
  pad1      : f32,
};

@group(0) @binding(0) var<uniform> U : SimU;
@group(0) @binding(1) var<storage, read_write> terrain : array<f32>;
@group(0) @binding(2) var<storage, read_write> water   : array<f32>;
@group(0) @binding(3) var<storage, read_write> flux    : array<vec4f>;  // x:-X y:+X z:-Y w:+Y
@group(0) @binding(4) var<storage, read_write> vel     : array<vec2f>;
@group(0) @binding(5) var<storage, read_write> sed     : array<f32>;
@group(0) @binding(6) var<storage, read_write> scratch : array<f32>;    // ping target for advect / thermal
@group(0) @binding(7) var<storage, read_write> pick    : array<vec4f>;  // [0] = (hitX, hitZ, hit?, t)
@group(0) @binding(8) var<storage, read_write> orig    : array<f32>;    // terrain as generated

const GRAV   : f32 = 9.81;
const VMAX   : f32 = 12.0;
const EVAP   : f32 = 0.015;
const KC     : f32 = 1.0;     // sediment capacity
const KS     : f32 = 0.30;    // dissolve rate
const KD     : f32 = 0.35;    // deposit rate
const TALUS  : f32 = 0.95;    // tan of talus angle (per unit cell)
const KT     : f32 = 0.50;    // thermal creep rate

fn totalH(x : i32, y : i32) -> f32 {
  if (x < 0 || y < 0 || x >= NI || y >= NI) {
    return -25.0;              // open boundary: water falls off the map edge
  }
  let i = cIdx(x, y);
  return terrain[i] + water[i];
}
fn fluxAt(x : i32, y : i32) -> vec4f {
  if (x < 0 || y < 0 || x >= NI || y >= NI) {
    return vec4f(0.0);
  }
  return flux[cIdx(x, y)];
}

fn bilTW(p : vec2f) -> f32 {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  let a = terrain[cIdx(x, y)]     + water[cIdx(x, y)];
  let b = terrain[cIdx(x + 1, y)] + water[cIdx(x + 1, y)];
  let c = terrain[cIdx(x, y + 1)] + water[cIdx(x, y + 1)];
  let d = terrain[cIdx(x + 1, y + 1)] + water[cIdx(x + 1, y + 1)];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn bilSed(p : vec2f) -> f32 {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  let a = sed[cIdx(x, y)];
  let b = sed[cIdx(x + 1, y)];
  let c = sed[cIdx(x, y + 1)];
  let d = sed[cIdx(x + 1, y + 1)];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// --- terrain generation ------------------------------------------------------
@compute @workgroup_size(8, 8)
fn genTerrain(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  let h = genHeight(vec2f(f32(gid.x), f32(gid.y)), U.seed);
  terrain[i] = h;
  orig[i] = h;
  water[i] = 0.0;
  sed[i] = 0.0;
  scratch[i] = 0.0;
  flux[i] = vec4f(0.0);
  vel[i] = vec2f(0.0);
}

@compute @workgroup_size(8, 8)
fn clearWater(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  water[i] = 0.0;
  sed[i] = 0.0;
  flux[i] = vec4f(0.0);
  vel[i] = vec2f(0.0);
}

// --- mouse pick: raymarch the surface heightfield ---------------------------
@compute @workgroup_size(1)
fn pickCast() {
  var hit = vec4f(0.0);
  if (U.rayDir.w > 0.5) {
    let ro = U.rayOrigin.xyz;
    let rd = normalize(U.rayDir.xyz);
    var t = 0.0;
    var prevT = 0.0;
    var prevD = ro.y - bilTW(ro.xz);
    for (var i = 0; i < 500; i++) {
      let p = ro + rd * t;
      let inMap = p.x >= 0.0 && p.x <= NF - 1.0 && p.z >= 0.0 && p.z <= NF - 1.0;
      var dh = p.y + 30.0;     // far below map: only hit while above it
      if (inMap) { dh = p.y - bilTW(p.xz); }
      if (dh < 0.0 && inMap) {
        let tt = prevT + (t - prevT) * prevD / max(prevD - dh, 1e-4);
        let hp = ro + rd * tt;
        hit = vec4f(hp.x, hp.z, 1.0, tt);
        break;
      }
      prevT = t;
      prevD = dh;
      t += clamp(dh * 0.45, 0.3, 8.0);
      if (t > 1500.0) { break; }
    }
  }
  pick[0] = hit;
}

// --- pour water at the picked point ------------------------------------------
@compute @workgroup_size(8, 8)
fn addWater(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let pk = pick[0];
  if (U.pourActive < 0.5 || pk.z < 0.5) { return; }
  let d = distance(vec2f(f32(gid.x), f32(gid.y)), pk.xy);
  if (d < U.radius) {
    let w = 0.5 + 0.5 * cos(d / U.radius * PI);
    water[gid.y * N + gid.x] += U.rate * U.dt * w;
  }
}

// --- pipe-model outflow flux --------------------------------------------------
@compute @workgroup_size(8, 8)
fn fluxPass(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  let h = terrain[i] + water[i];
  var f = flux[i];
  let dh = vec4f(
    h - totalH(x - 1, y),
    h - totalH(x + 1, y),
    h - totalH(x, y - 1),
    h - totalH(x, y + 1));
  f = max(vec4f(0.0), f * 0.995 + U.dt * GRAV * dh);
  let total = f.x + f.y + f.z + f.w;
  if (total > 1e-6) {
    let k = min(1.0, water[i] / (total * U.dt));
    f = f * k;
  }
  flux[i] = f;
}

// --- depth integration, velocity field, evaporation --------------------------
@compute @workgroup_size(8, 8)
fn depthVel(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  let fc = flux[i];
  let inL = fluxAt(x - 1, y).y;
  let inR = fluxAt(x + 1, y).x;
  let inB = fluxAt(x, y - 1).w;
  let inT = fluxAt(x, y + 1).z;
  let d0 = water[i];
  let dv = U.dt * ((inL + inR + inB + inT) - (fc.x + fc.y + fc.z + fc.w));
  var d1 = max(0.0, d0 + dv);

  let wx = 0.5 * (inL - fc.x + fc.y - inR);
  let wy = 0.5 * (inB - fc.z + fc.w - inT);
  let dAvg = max(0.05, 0.5 * (d0 + d1));
  var v = vec2f(wx, wy) / dAvg;
  let sp = length(v);
  if (sp > VMAX) { v = v * (VMAX / sp); }
  vel[i] = v;

  d1 = d1 * (1.0 - EVAP * U.dt);
  if (d1 < 1e-5) { d1 = 0.0; }
  water[i] = d1;
}

// --- hydraulic erosion / deposition -------------------------------------------
@compute @workgroup_size(8, 8)
fn erosion(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;

  let tC = terrain[i];
  let tL = terrain[cIdx(x - 1, y)];
  let tR = terrain[cIdx(x + 1, y)];
  let tB = terrain[cIdx(x, y - 1)];
  let tT = terrain[cIdx(x, y + 1)];
  let grad = 0.5 * vec2f(tR - tL, tT - tB);
  let sinA = length(grad) / sqrt(1.0 + dot(grad, grad));
  let slope = max(0.05, sinA);

  let sp = length(vel[i]);
  let depthFac = clamp(water[i] * 6.0, 0.0, 1.0);   // dry film doesn't carve
  let cap = KC * slope * sp * depthFac;
  let st = sed[i];

  // relief clamp vs the 4-neighbour mean keeps the erode/deposit feedback from
  // checkerboarding: a local pit stops digging, a local bump stops growing
  let avg4 = 0.25 * (tL + tR + tB + tT);
  if (cap > st) {
    var a = min(KS * (cap - st) * U.dt, 0.05);
    a = min(a, max(0.0, tC - (avg4 - 0.9)));
    terrain[i] = tC - a;
    sed[i] = st + a;
    water[i] += a;                                   // dissolved volume joins the flow
  } else {
    var a = min(KD * (st - cap) * U.dt, st);
    a = min(a, max(0.0, (avg4 + 0.3) - tC));   // deposits fill lows, never build bumps
    terrain[i] = tC + a;
    sed[i] = st - a;
    water[i] = max(0.0, water[i] - a);
  }
}

// --- semi-Lagrangian sediment advection (writes scratch) ----------------------
@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  let p = vec2f(f32(gid.x), f32(gid.y)) - vel[i] * U.dt;
  scratch[i] = bilSed(p);
}

// --- thermal erosion: symmetric gather over 8 neighbours (writes scratch) -----
@compute @workgroup_size(8, 8)
fn thermal(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  let h = terrain[i];
  var delta = 0.0;
  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      if (ox == 0 && oy == 0) { continue; }
      let nx = x + ox;
      let ny = y + oy;
      if (nx < 0 || ny < 0 || nx >= NI || ny >= NI) { continue; }
      let dist = length(vec2f(f32(ox), f32(oy)));
      let dh = h - terrain[cIdx(nx, ny)];
      let lim = TALUS * dist;
      if (dh > lim) {
        delta -= KT * (dh - lim) * U.dt / dist;      // give material away
      } else if (-dh > lim) {
        delta += KT * (-dh - lim) * U.dt / dist;     // receive material
      }
    }
  }
  scratch[i] = h + delta * 0.5;
}
`;

// ---------------------------------------------------------------------------
// WGSL — render module (sky + terrain + water share one bind group)
// ---------------------------------------------------------------------------

const WGSL_RENDER = /* wgsl */`
${WGSL_COMMON}

struct RU {
  viewProj : mat4x4f,
  camPos   : vec4f,
  camRight : vec4f,
  camUp    : vec4f,
  camFwd   : vec4f,            // w = tan(fov/2)
  sun      : vec4f,
  misc     : vec4f,            // x: time, y: pourActive, z: brushRadius, w: aspect
};

@group(0) @binding(0) var<uniform> R : RU;
@group(0) @binding(1) var<storage, read> terrain : array<f32>;
@group(0) @binding(2) var<storage, read> water   : array<f32>;
@group(0) @binding(3) var<storage, read> sed     : array<f32>;
@group(0) @binding(4) var<storage, read> vel     : array<vec2f>;
@group(0) @binding(5) var<storage, read> orig    : array<f32>;
@group(0) @binding(6) var<storage, read> pick    : array<vec4f>;

fn bilBuf(p : vec2f, which : i32) -> f32 {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  var a = 0.0; var b = 0.0; var c = 0.0; var d = 0.0;
  if (which == 0) {
    a = terrain[cIdx(x, y)];     b = terrain[cIdx(x + 1, y)];
    c = terrain[cIdx(x, y + 1)]; d = terrain[cIdx(x + 1, y + 1)];
  } else if (which == 1) {
    a = water[cIdx(x, y)];       b = water[cIdx(x + 1, y)];
    c = water[cIdx(x, y + 1)];   d = water[cIdx(x + 1, y + 1)];
  } else if (which == 2) {
    a = sed[cIdx(x, y)];         b = sed[cIdx(x + 1, y)];
    c = sed[cIdx(x, y + 1)];     d = sed[cIdx(x + 1, y + 1)];
  } else {
    a = orig[cIdx(x, y)];        b = orig[cIdx(x + 1, y)];
    c = orig[cIdx(x, y + 1)];    d = orig[cIdx(x + 1, y + 1)];
  }
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn bilVelMag(p : vec2f) -> f32 {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  let a = length(vel[cIdx(x, y)]);     let b = length(vel[cIdx(x + 1, y)]);
  let c = length(vel[cIdx(x, y + 1)]); let d = length(vel[cIdx(x + 1, y + 1)]);
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn bilT(p : vec2f) -> f32 { return bilBuf(p, 0); }
fn bilW(p : vec2f) -> f32 { return bilBuf(p, 1); }
fn bilS(p : vec2f) -> f32 { return bilBuf(p, 2); }
fn bilO(p : vec2f) -> f32 { return bilBuf(p, 3); }
fn bilTW(p : vec2f) -> f32 { return bilT(p) + bilW(p); }

fn terrNormal(p : vec2f) -> vec3f {
  let e = 1.0;
  let dx = bilT(p + vec2f(e, 0.0)) - bilT(p - vec2f(e, 0.0));
  let dy = bilT(p + vec2f(0.0, e)) - bilT(p - vec2f(0.0, e));
  return normalize(vec3f(-dx / (2.0 * e), 1.0, -dy / (2.0 * e)));
}
fn surfNormal(p : vec2f) -> vec3f {
  let e = 1.0;
  let dx = bilTW(p + vec2f(e, 0.0)) - bilTW(p - vec2f(e, 0.0));
  let dy = bilTW(p + vec2f(0.0, e)) - bilTW(p - vec2f(0.0, e));
  return normalize(vec3f(-dx / (2.0 * e), 1.0, -dy / (2.0 * e)));
}

// --- sky --------------------------------------------------------------------
fn skyColor(rd : vec3f) -> vec3f {
  let sd = R.sun.xyz;
  let up = clamp(rd.y, -1.0, 1.0);
  var col = mix(vec3f(0.66, 0.74, 0.85), vec3f(0.16, 0.34, 0.68), pow(clamp(up, 0.0, 1.0), 0.55));
  col = mix(col, vec3f(0.46, 0.52, 0.62), exp(-abs(up) * 9.0) * 0.35);
  let s = clamp(dot(rd, sd), 0.0, 1.0);
  col += vec3f(1.0, 0.88, 0.65) * pow(s, 800.0) * 24.0;   // disc
  col += vec3f(1.0, 0.75, 0.45) * pow(s, 7.0) * 0.15;     // glow
  if (up > 0.015) {
    let cp = rd.xz / (up + 0.12) * 38.0 + vec2f(R.misc.x * 0.9, R.misc.x * 0.22);
    let cl = fbm(cp * 0.035);
    let cov = smoothstep(0.52, 0.78, cl) * clamp(up * 4.0, 0.0, 1.0);
    col = mix(col, vec3f(0.95, 0.96, 1.0) * (0.75 + 0.35 * s), cov * 0.55);
  }
  if (up < 0.0) {
    col = mix(col, vec3f(0.25, 0.27, 0.3), clamp(-up * 4.0, 0.0, 0.8));
  }
  return col;
}

// --- sun visibility: soft shadow raymarch over the surface -------------------
fn shadowRay(p0 : vec3f) -> f32 {
  let sd = R.sun.xyz;
  var t = 1.0;
  var res = 1.0;
  for (var i = 0; i < 26; i++) {
    let p = p0 + sd * t;
    if (p.y > 46.0) { break; }
    if (p.x < 0.0 || p.z < 0.0 || p.x > NF - 1.0 || p.z > NF - 1.0) { break; }
    let d = p.y - bilTW(p.xz);
    res = min(res, 6.0 * d / t);
    if (res < 0.02) { break; }
    t += clamp(d, 0.45, 7.0);
  }
  return clamp(res, 0.0, 1.0);
}

fn aces(x : vec3f) -> vec3f {
  let v = x * (2.51 * x + vec3f(0.03)) / (x * (2.43 * x + vec3f(0.59)) + vec3f(0.14));
  return clamp(v, vec3f(0.0), vec3f(1.0));
}
fn finish(c0 : vec3f) -> vec3f {
  var c = aces(c0 * 0.95);
  let l = dot(c, vec3f(0.299, 0.587, 0.114));
  c = clamp(mix(vec3f(l), c, 1.3), vec3f(0.0), vec3f(1.0));
  return pow(c, vec3f(1.0 / 2.2));
}
fn applyFog(col : vec3f, wp : vec3f) -> vec3f {
  let toP = wp - R.camPos.xyz;
  let dist = length(toP);
  let f = 1.0 - exp(-dist * 0.00042);
  return mix(col, skyColor(normalize(toP)) * 0.95, f);
}

// --- terrain material ---------------------------------------------------------
fn terrainAlbedo(wp : vec3f, n : vec3f) -> vec3f {
  let h = wp.y;
  let slope = 1.0 - n.y;
  let rnd = fbm(wp.xz * 0.16);
  let detail = fbm(wp.xz * 0.55 + h * 0.18);

  let grass = mix(vec3f(0.10, 0.24, 0.05), vec3f(0.26, 0.37, 0.10), rnd);
  let rock  = mix(vec3f(0.27, 0.24, 0.21), vec3f(0.49, 0.46, 0.42), detail);
  let sand  = vec3f(0.66, 0.58, 0.42);
  let snow  = vec3f(0.92, 0.94, 0.98);

  var a = mix(grass, rock, smoothstep(0.12, 0.34, slope + (rnd - 0.5) * 0.1));
  a = mix(sand, a, smoothstep(1.0, 3.0, h + rnd * 1.2));
  a = mix(a, snow, smoothstep(21.0, 25.0, h + rnd * 5.0) * (1.0 - smoothstep(0.18, 0.5, slope)));

  // erosion scars expose rock; deposition leaves silt (wide tap — the raw
  // per-cell delta is noisy and would dapple the basin)
  let q = wp.xz;
  let delta = 0.25 * ((bilT(q) - bilO(q))
            + (bilT(q + vec2f(1.6, 0.0)) - bilO(q + vec2f(1.6, 0.0)))
            + (bilT(q + vec2f(-0.8, 1.4)) - bilO(q + vec2f(-0.8, 1.4)))
            + (bilT(q + vec2f(-0.8, -1.4)) - bilO(q + vec2f(-0.8, -1.4))));
  a = mix(a, rock * 0.85, clamp(-delta * 0.7, 0.0, 1.0) * 0.55);
  a = mix(a, vec3f(0.62, 0.52, 0.36), clamp(delta * 0.9, 0.0, 1.0) * 0.6);
  return a;
}

// ============================ SKY PASS =======================================
struct SkyOut {
  @builtin(position) pos : vec4f,
  @location(0) ndc : vec2f,
};
@vertex
fn vsSky(@builtin(vertex_index) vi : u32) -> SkyOut {
  var out : SkyOut;
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - vec2f(1.0);
  out.pos = vec4f(xy, 1.0, 1.0);
  out.ndc = xy;
  return out;
}
@fragment
fn fsSky(in : SkyOut) -> @location(0) vec4f {
  let tf = R.camFwd.w;
  let aspect = R.misc.w;
  let rd = normalize(
    R.camFwd.xyz +
    R.camRight.xyz * in.ndc.x * tf * aspect +
    R.camUp.xyz * in.ndc.y * tf);
  return vec4f(finish(skyColor(rd)), 1.0);
}

// ============================ TERRAIN PASS ===================================
struct TOut {
  @builtin(position) pos : vec4f,
  @location(0) wp : vec3f,
};
@vertex
fn vsTerrain(@builtin(vertex_index) vi : u32) -> TOut {
  var out : TOut;
  let x = f32(vi % N);
  let y = f32(vi / N);
  let wp = vec3f(x, terrain[vi], y);
  out.wp = wp;
  out.pos = R.viewProj * vec4f(wp, 1.0);
  return out;
}
@fragment
fn fsTerrain(in : TOut) -> @location(0) vec4f {
  let p = in.wp.xz;
  let n = terrNormal(p);
  let alb = terrainAlbedo(in.wp, n);
  let sd = R.sun.xyz;

  let sh = shadowRay(in.wp + n * 0.35);
  let ndl = clamp(dot(n, sd), 0.0, 1.0);

  // cheap cavity AO from the height laplacian
  let lap = bilT(p + vec2f(2.0, 0.0)) + bilT(p - vec2f(2.0, 0.0))
          + bilT(p + vec2f(0.0, 2.0)) + bilT(p - vec2f(0.0, 2.0)) - 4.0 * bilT(p);
  let ao = clamp(1.0 + lap * 0.06, 0.55, 1.0);

  let skyAmb = (0.34 + 0.28 * n.y) * vec3f(0.42, 0.52, 0.70);
  let bounce = clamp(dot(n, normalize(vec3f(-sd.x, 0.25, -sd.z))), 0.0, 1.0)
             * vec3f(0.30, 0.24, 0.18) * 0.35;

  var col = alb * (ndl * sh * vec3f(1.35, 1.22, 1.00) * 1.55 + skyAmb * ao + bounce);

  // darken just-submerged shoreline so beaches read as wet
  let wd = bilW(p);
  col *= mix(1.0, 0.62, smoothstep(0.0, 0.04, wd));

  col = applyFog(col, in.wp);

  // cursor brush ring
  let pk = pick[0];
  if (pk.z > 0.5) {
    let d = distance(p, pk.xy);
    let r = R.misc.z;
    let ring = (1.0 - smoothstep(0.25, 0.9, abs(d - r))) * 0.8
             + (1.0 - smoothstep(0.0, r, d)) * 0.10;
    let glow = mix(vec3f(0.35, 0.85, 1.0), vec3f(0.55, 1.0, 1.0), R.misc.y);
    col += glow * ring * (0.4 + 0.6 * R.misc.y);
  }

  return vec4f(finish(col), 1.0);
}

// ============================ WATER PASS =====================================
struct WOut {
  @builtin(position) pos : vec4f,
  @location(0) wp : vec3f,
  @location(1) depth : f32,
};
@vertex
fn vsWater(@builtin(vertex_index) vi : u32) -> WOut {
  var out : WOut;
  let x = f32(vi % N);
  let y = f32(vi / N);
  let d = water[vi];
  var h = terrain[vi] + d;
  if (d < 0.0012) {
    h = terrain[vi] - 0.06;        // sink dry verts just below ground
  }
  let wp = vec3f(x, h, y);
  out.wp = wp;
  out.depth = d;
  out.pos = R.viewProj * vec4f(wp, 1.0);
  return out;
}
@fragment
fn fsWater(in : WOut) -> @location(0) vec4f {
  let p = in.wp.xz;
  let d = bilW(p);
  if (d < 0.0012 || in.depth < 0.0008) { discard; }

  let t = R.misc.x;
  var n = surfNormal(p);

  // animated ripple detail, fading out on thin films
  let rs = clamp(d * 3.0, 0.0, 1.0) * 0.22;
  let e = 0.35;
  let rp = p * 1.35 + vec2f(t * 0.9, t * 0.55);
  let r0 = vnoise(rp) + 0.5 * vnoise(p * 2.9 - vec2f(t * 1.3, t * 0.7));
  let rx = vnoise(rp + vec2f(e, 0.0)) + 0.5 * vnoise(p * 2.9 - vec2f(t * 1.3, t * 0.7) + vec2f(e, 0.0)) - r0;
  let ry = vnoise(rp + vec2f(0.0, e)) + 0.5 * vnoise(p * 2.9 - vec2f(t * 1.3, t * 0.7) + vec2f(0.0, e)) - r0;
  n = normalize(n + vec3f(-rx, 0.0, -ry) * rs / e);

  let V = normalize(R.camPos.xyz - in.wp);
  let sd = R.sun.xyz;
  let fres = 0.025 + 0.975 * pow(1.0 - clamp(dot(n, V), 0.0, 1.0), 5.0);

  let sh = shadowRay(in.wp + vec3f(0.0, 0.25, 0.0));
  let refl = skyColor(reflect(-V, n));
  let spec = pow(clamp(dot(reflect(-V, n), sd), 0.0, 1.0), 320.0) * sh * 4.0;

  // depth absorption — what fraction of the bed below shows through
  let mud = clamp(bilS(p) * 0.6, 0.0, 0.8);
  let absorb = mix(vec3f(0.42, 0.16, 0.10), vec3f(0.55, 0.50, 0.40), mud);
  let transmit = exp(-d * absorb * 2.6);

  let deepTint = mix(vec3f(0.04, 0.18, 0.23), vec3f(0.20, 0.17, 0.10), mud);
  var body = deepTint * (1.0 - transmit) * (0.55 + 0.75 * sh);

  // foam where the flow is fast — wide blurred velocity tap so it bands, not speckles
  let sp = 0.25 * (bilVelMag(p) + bilVelMag(p + vec2f(1.5, 0.0))
                 + bilVelMag(p + vec2f(-0.75, 1.3)) + bilVelMag(p + vec2f(-0.75, -1.3)));
  let foamN = vnoise(p * 1.7 + vec2f(t * 1.8, -t * 1.2)) * 0.6 + vnoise(p * 4.1 - vec2f(t * 1.1)) * 0.4;
  let foam = smoothstep(2.6, 5.5, sp) * smoothstep(0.4, 0.8, foamN) * clamp(d * 8.0, 0.0, 1.0) * 0.55;

  var rgb = body + refl * fres + vec3f(spec);
  rgb += vec3f(0.9, 0.95, 1.0) * foam * (0.35 + 0.65 * sh);
  rgb = applyFog(rgb, in.wp);

  var alpha = clamp(1.0 - dot(transmit, vec3f(0.333)) * (1.0 - fres), 0.05, 1.0);
  alpha = max(alpha, foam * 0.85);
  let shore = smoothstep(0.002, 0.06, d);
  alpha *= shore;
  rgb *= shore;

  return vec4f(finish(rgb) * alpha, alpha);   // premultiplied
}
`;

// ---------------------------------------------------------------------------
// Tiny column-major mat4 helpers
// ---------------------------------------------------------------------------

function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}
function mat4Perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect;
  o[5] = f;
  o[10] = far / (near - far);
  o[11] = -1;
  o[14] = (near * far) / (near - far);
  return o;
}
function mat4LookAt(eye, target, up) {
  const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let zl = Math.hypot(zx, zy, zz);
  const z = [zx / zl, zy / zl, zz / zl];
  const x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  const xl = Math.hypot(x[0], x[1], x[2]);
  x[0] /= xl; x[1] /= xl; x[2] /= xl;
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1,
  ]);
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function fail(msg) {
  const el = document.getElementById('fail');
  el.style.display = 'flex';
  if (msg) document.getElementById('failMsg').textContent = msg;
}

async function main() {
  const canvas = document.getElementById('c');
  if (!navigator.gpu) { fail(); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { fail('No WebGPU adapter found. Try enabling WebGPU in your browser settings.'); return; }
  const device = await adapter.requestDevice();
  device.lost.then((info) => { if (info.reason !== 'destroyed') fail('GPU device lost: ' + info.message); });

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // --- buffers ---------------------------------------------------------------
  const cells = N * N;
  const mk = (bytes) => device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const bufTerrain = mk(cells * 4);
  const bufWater   = mk(cells * 4);
  const bufFlux    = mk(cells * 16);
  const bufVel     = mk(cells * 8);
  const bufSed     = mk(cells * 4);
  const bufScratch = mk(cells * 4);
  const bufPick    = mk(16);
  const bufOrig    = mk(cells * 4);

  const simUBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const renUBuf = device.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // --- grid index buffer (vertices are implicit from vertex_index) -----------
  const quads = (N - 1) * (N - 1);
  const indices = new Uint32Array(quads * 6);
  let k = 0;
  for (let y = 0; y < N - 1; y++) {
    for (let x = 0; x < N - 1; x++) {
      const i = y * N + x;
      indices[k++] = i;     indices[k++] = i + 1;     indices[k++] = i + N;
      indices[k++] = i + 1; indices[k++] = i + N + 1; indices[k++] = i + N;
    }
  }
  const indexBuf = device.createBuffer({ size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(indexBuf, 0, indices);

  // --- simulation pipelines ---------------------------------------------------
  const simModule = device.createShaderModule({ code: WGSL_SIM });
  const simBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4, 5, 6, 7, 8].map((b) => ({
        binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' },
      })),
    ],
  });
  const simLayout = device.createPipelineLayout({ bindGroupLayouts: [simBGL] });
  const simBG = device.createBindGroup({
    layout: simBGL,
    entries: [
      { binding: 0, resource: { buffer: simUBuf } },
      { binding: 1, resource: { buffer: bufTerrain } },
      { binding: 2, resource: { buffer: bufWater } },
      { binding: 3, resource: { buffer: bufFlux } },
      { binding: 4, resource: { buffer: bufVel } },
      { binding: 5, resource: { buffer: bufSed } },
      { binding: 6, resource: { buffer: bufScratch } },
      { binding: 7, resource: { buffer: bufPick } },
      { binding: 8, resource: { buffer: bufOrig } },
    ],
  });
  const simPipe = {};
  for (const ep of ['genTerrain', 'clearWater', 'pickCast', 'addWater', 'fluxPass', 'depthVel', 'erosion', 'advect', 'thermal']) {
    simPipe[ep] = device.createComputePipeline({
      layout: simLayout,
      compute: { module: simModule, entryPoint: ep },
    });
  }

  // --- render pipelines --------------------------------------------------------
  const SAMPLES = 4;
  const renModule = device.createShaderModule({ code: WGSL_RENDER });
  const renBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4, 5, 6].map((b) => ({
        binding: b,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' },
      })),
    ],
  });
  const renLayout = device.createPipelineLayout({ bindGroupLayouts: [renBGL] });
  const renBG = device.createBindGroup({
    layout: renBGL,
    entries: [
      { binding: 0, resource: { buffer: renUBuf } },
      { binding: 1, resource: { buffer: bufTerrain } },
      { binding: 2, resource: { buffer: bufWater } },
      { binding: 3, resource: { buffer: bufSed } },
      { binding: 4, resource: { buffer: bufVel } },
      { binding: 5, resource: { buffer: bufOrig } },
      { binding: 6, resource: { buffer: bufPick } },
    ],
  });

  const skyPipe = device.createRenderPipeline({
    layout: renLayout,
    vertex: { module: renModule, entryPoint: 'vsSky' },
    fragment: { module: renModule, entryPoint: 'fsSky', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
    multisample: { count: SAMPLES },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
  });
  const terrainPipe = device.createRenderPipeline({
    layout: renLayout,
    vertex: { module: renModule, entryPoint: 'vsTerrain' },
    fragment: { module: renModule, entryPoint: 'fsTerrain', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: SAMPLES },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  const waterPipe = device.createRenderPipeline({
    layout: renLayout,
    vertex: { module: renModule, entryPoint: 'vsWater' },
    fragment: {
      module: renModule, entryPoint: 'fsWater',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: SAMPLES },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  // --- render targets (recreated on resize) -----------------------------------
  let msaaTex = null, depthTex = null;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === w && canvas.height === h && msaaTex) return;
    canvas.width = w;
    canvas.height = h;
    if (msaaTex) msaaTex.destroy();
    if (depthTex) depthTex.destroy();
    msaaTex = device.createTexture({
      size: [w, h], sampleCount: SAMPLES, format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthTex = device.createTexture({
      size: [w, h], sampleCount: SAMPLES, format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  window.addEventListener('resize', resize);
  resize();

  // --- camera & input ----------------------------------------------------------
  const cam = {
    yaw: 0.9, pitch: 0.62, dist: 290,
    target: [N / 2, 6, N / 2],
    fov: (45 * Math.PI) / 180,
  };
  const mouse = { x: 0, y: 0, over: false, pouring: false, orbiting: false, lx: 0, ly: 0 };

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    if (e.button === 0) mouse.pouring = true;
    if (e.button === 2 || e.button === 1) {
      mouse.orbiting = true;
      mouse.lx = e.clientX;
      mouse.ly = e.clientY;
    }
  });
  canvas.addEventListener('pointerup', (e) => {
    if (e.button === 0) mouse.pouring = false;
    if (e.button === 2 || e.button === 1) mouse.orbiting = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.over = true;
    if (mouse.orbiting) {
      cam.yaw += (e.clientX - mouse.lx) * 0.005;
      cam.pitch = Math.min(1.5, Math.max(0.08, cam.pitch + (e.clientY - mouse.ly) * 0.004));
      mouse.lx = e.clientX;
      mouse.ly = e.clientY;
    }
  });
  canvas.addEventListener('pointerleave', () => { mouse.over = false; mouse.pouring = false; });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.dist = Math.min(700, Math.max(40, cam.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // --- UI ------------------------------------------------------------------------
  const rateEl = document.getElementById('rate');
  const radEl = document.getElementById('rad');
  const rateV = document.getElementById('rateV');
  const radV = document.getElementById('radV');
  rateEl.addEventListener('input', () => { rateV.textContent = (+rateEl.value).toFixed(1); });
  radEl.addEventListener('input', () => { radV.textContent = (+radEl.value).toFixed(1); });

  let seed = Math.random() * 100;
  let pendingReset = true;      // generate terrain on first frame
  let pendingClear = false;
  document.getElementById('reset').addEventListener('click', () => {
    seed = Math.random() * 100;
    pendingReset = true;
  });
  document.getElementById('clear').addEventListener('click', () => { pendingClear = true; });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') { seed = Math.random() * 100; pendingReset = true; }
    if (e.key === 'c' || e.key === 'C') { pendingClear = true; }
  });

  // --- per-frame uniform staging ---------------------------------------------------
  const simU = new Float32Array(16);
  const renU = new Float32Array(44);
  const fpsEl = document.getElementById('fps');
  let frames = 0, fpsT = performance.now(), simTime = 0;

  const wgCount = Math.ceil(N / 8);

  function frame(nowMs) {
    resize();
    const aspect = canvas.width / canvas.height;
    const t = nowMs * 0.001;

    // camera basis
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const eye = [
      cam.target[0] + cam.dist * cp * cy,
      cam.target[1] + cam.dist * sp,
      cam.target[2] + cam.dist * cp * sy,
    ];
    const fwd = [cam.target[0] - eye[0], cam.target[1] - eye[1], cam.target[2] - eye[2]];
    const fl = Math.hypot(...fwd);
    fwd[0] /= fl; fwd[1] /= fl; fwd[2] /= fl;
    const right = [-(fwd[2]), 0, fwd[0]];
    const rl = Math.hypot(...right);
    right[0] /= rl; right[1] /= rl; right[2] /= rl;
    const up = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];
    const view = mat4LookAt(eye, cam.target, [0, 1, 0]);
    const proj = mat4Perspective(cam.fov, aspect, 0.5, 3000);
    const viewProj = mat4Mul(proj, view);

    // mouse ray
    const tanF = Math.tan(cam.fov / 2);
    const ndcX = (mouse.x / canvas.clientWidth) * 2 - 1;
    const ndcY = 1 - (mouse.y / canvas.clientHeight) * 2;
    const rd = [
      fwd[0] + right[0] * ndcX * tanF * aspect + up[0] * ndcY * tanF,
      fwd[1] + right[1] * ndcX * tanF * aspect + up[1] * ndcY * tanF,
      fwd[2] + right[2] * ndcX * tanF * aspect + up[2] * ndcY * tanF,
    ];

    simTime += DT * SUBSTEPS;
    simU.set([eye[0], eye[1], eye[2], 0], 0);
    simU.set([rd[0], rd[1], rd[2], mouse.over ? 1 : 0], 4);
    simU[8] = DT;
    simU[9] = simTime;
    simU[10] = +rateEl.value;
    simU[11] = +radEl.value;
    simU[12] = mouse.pouring && !mouse.orbiting ? 1 : 0;
    simU[13] = seed;
    device.queue.writeBuffer(simUBuf, 0, simU);

    renU.set(viewProj, 0);
    renU.set([eye[0], eye[1], eye[2], 0], 16);
    renU.set([right[0], right[1], right[2], 0], 20);
    renU.set([up[0], up[1], up[2], 0], 24);
    renU.set([fwd[0], fwd[1], fwd[2], tanF], 28);
    const sunL = Math.hypot(0.55, 0.62, 0.32);
    renU.set([0.55 / sunL, 0.62 / sunL, 0.32 / sunL, 0], 32);
    renU.set([t, simU[12], +radEl.value, aspect], 36);
    device.queue.writeBuffer(renUBuf, 0, renU);

    const enc = device.createCommandEncoder();

    const runPass = (pass, names) => {
      for (const name of names) {
        pass.setPipeline(simPipe[name]);
        pass.setBindGroup(0, simBG);
        if (name === 'pickCast') pass.dispatchWorkgroups(1);
        else pass.dispatchWorkgroups(wgCount, wgCount);
      }
    };

    if (pendingReset || pendingClear) {
      const pass = enc.beginComputePass();
      runPass(pass, [pendingReset ? 'genTerrain' : 'clearWater']);
      pass.end();
      pendingReset = false;
      pendingClear = false;
    }

    {
      const pass = enc.beginComputePass();
      runPass(pass, ['pickCast']);
      pass.end();
    }
    for (let s = 0; s < SUBSTEPS; s++) {
      const pass = enc.beginComputePass();
      runPass(pass, ['addWater', 'fluxPass', 'depthVel', 'erosion', 'advect']);
      pass.end();
      enc.copyBufferToBuffer(bufScratch, 0, bufSed, 0, cells * 4);
    }
    {
      const pass = enc.beginComputePass();
      runPass(pass, ['thermal']);
      pass.end();
      enc.copyBufferToBuffer(bufScratch, 0, bufTerrain, 0, cells * 4);
    }

    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: msaaTex.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'discard',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
        depthClearValue: 1,
      },
    });
    rp.setBindGroup(0, renBG);
    rp.setPipeline(skyPipe);
    rp.draw(3);
    rp.setIndexBuffer(indexBuf, 'uint32');
    rp.setPipeline(terrainPipe);
    rp.drawIndexed(indices.length);
    rp.setPipeline(waterPipe);
    rp.drawIndexed(indices.length);
    rp.end();

    device.queue.submit([enc.finish()]);

    frames++;
    const now = performance.now();
    if (now - fpsT > 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - fpsT))} fps · ${N}×${N}`;
      frames = 0;
      fpsT = now;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  fail('WebGPU init failed: ' + e.message);
});
