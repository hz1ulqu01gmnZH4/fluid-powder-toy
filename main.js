'use strict';

// ============================================================================
// Hydra Terra — WebGPU fluid & erosion sandbox
//
// Simulation: heightfield shallow-water "virtual pipes" model (Mei et al. 2007)
// with hydraulic erosion (capacity / dissolve / deposit + conservative
// flux-form sediment advection) and thermal erosion (talus slippage).
//
// The underground is a stratigraphic column per cell: up to NK strata in
// arbitrary vertical order, each (material, thickness, stored water), from a
// world-floor datum (BASE) up to the surface. Eight materials (bedrock, clay,
// silt, sand, gravel, loam, regolith, boulders) span ~3 decades of hydraulic
// conductivity, so clay seals, gravel transmits, and a thin gravel bed between
// rock layers acts as a confined aquifer. Groundwater follows a quasi-3D
// MODFLOW-style scheme: per-stratum heads, lateral Darcy exchange over the
// depth overlap of neighbouring strata, gravity percolation down the column
// throttled by the receiving material, and upward seepage of confined
// overpressure — perched water tables and artesian springs emerge from the
// rules rather than being scripted. A tree-density field shields the ground
// from erosion and is rendered as instanced pines.
//
// Scale: 1 cell = 2 m, 1 height unit = 1 m, ~9.9 sim-seconds per wall-second.
//
// Everything lives in GPU storage buffers; the CPU never reads the field
// back. Mouse picking is a single-thread compute raymarch that writes the hit
// point into a small buffer which the tool pass and cursor-ring shader read
// directly.
// ============================================================================

// Grid resolution is selectable via ?n=128|256|384|512 (rebuilding every
// buffer and shader, so it applies through a page reload).
const N_CHOICES = [128, 256, 384, 512];
const urlN = (typeof location !== 'undefined')
  ? parseInt(new URLSearchParams(location.search).get('n') || '', 10)
  : NaN;
const N = N_CHOICES.includes(urlN) ? urlN : 256;   // sim + mesh grid resolution (cells)
const SUBSTEPS = 3;            // sim substeps per frame
const DT = 0.055;              // sim timestep per substep
const MAX_SOURCES = 16;        // persistent water springs (shift+click)
const NK = 10;                 // strata slots per column (arbitrary vertical order)

// Tools (SimU.tool)
const TOOL_WATER = 0, TOOL_ADD = 1, TOOL_DIG = 2, TOOL_REPLACE = 3,
      TOOL_TREE = 4, TOOL_UNTREE = 5;

// ---------------------------------------------------------------------------
// WGSL — shared helper snippets (templated into both modules)
// ---------------------------------------------------------------------------

const WGSL_COMMON = /* wgsl */`
const N : u32 = ${N}u;
const NI : i32 = ${N};
const NF : f32 = ${N}.0;
const PI : f32 = 3.14159265;

// Stratigraphic columns: up to NK strata per cell, stored bottom-up from the
// world-floor datum BASE. Each stratum is vec4f(material, thickness, water, 0);
// occupied slots are contiguous from slot 0.
const NK : i32 = ${NK};
const NKU : u32 = ${NK}u;
const BASE : f32 = -26.0;       // column bottom / world floor elevation

// Materials: 0 bedrock, 1 clay, 2 silt, 3 sand, 4 gravel, 5 loam, 6 regolith,
// 7 boulders. Hydraulic numbers keep the real-world (Freeze & Cherry)
// ordering, compressed to ~3 decades so the aquifer/aquitard contrast is
// visible at sim scale. Clay: high porosity but ~zero yield/conductivity —
// the aquitard. Gravel: the prime aquifer.
const M_ERODE : array<f32, 8> = array<f32, 8>(0.03, 0.40, 1.20, 1.00, 0.70, 1.40, 0.10, 0.10);
const M_TALUS : array<f32, 8> = array<f32, 8>(5.00, 1.60, 0.60, 0.70, 0.90, 0.60, 2.50, 0.95);
const M_CREEP : array<f32, 8> = array<f32, 8>(0.00, 0.15, 1.00, 1.00, 0.80, 1.10, 0.10, 0.65);
const M_SY    : array<f32, 8> = array<f32, 8>(0.02, 0.03, 0.12, 0.25, 0.28, 0.18, 0.06, 0.30);
const M_KSAT  : array<f32, 8> = array<f32, 8>(0.000, 0.002, 0.020, 0.300, 1.000, 0.080, 0.100, 1.200);
const M_INFIL : array<f32, 8> = array<f32, 8>(0.0001, 0.0005, 0.002, 0.020, 0.050, 0.008, 0.002, 0.080);

const SS_EFF : f32 = 0.02;   // confined storativity: small ⇒ stiff overpressure head
const OP_MAX : f32 = 0.5;    // max overpressure volume a stratum can hold
const VPERC  : f32 = 6.0;    // vertical/lateral conductivity ratio (gravity helps)

// Property lookups (const arrays must be copied to a var for runtime indexing).
fn mErode(m : i32) -> f32 { var t = M_ERODE; return t[m]; }
fn mTalus(m : i32) -> f32 { var t = M_TALUS; return t[m]; }
fn mCreep(m : i32) -> f32 { var t = M_CREEP; return t[m]; }
fn mSy   (m : i32) -> f32 { var t = M_SY;    return t[m]; }
fn mK    (m : i32) -> f32 { var t = M_KSAT;  return t[m]; }
fn mInfil(m : i32) -> f32 { var t = M_INFIL; return t[m]; }

fn cIdx(x : i32, y : i32) -> u32 {
  return u32(clamp(y, 0, NI - 1)) * N + u32(clamp(x, 0, NI - 1));
}

alias Col = array<vec4f, ${NK}>;

// Topmost occupied slot (-1 when the column is empty).
fn colTopIdx(c : ptr<function, Col>) -> i32 {
  for (var k = NK - 1; k >= 0; k--) {
    if ((*c)[k].y > 1e-4) { return k; }
  }
  return -1;
}
fn colSumTh(c : ptr<function, Col>) -> f32 {
  var s = 0.0;
  for (var k = 0; k < NK; k++) { s += (*c)[k].y; }
  return s;
}
fn colWater(c : ptr<function, Col>) -> f32 {
  var s = 0.0;
  for (var k = 0; k < NK; k++) { s += (*c)[k].z; }
  return s;
}
fn colCap(c : ptr<function, Col>) -> f32 {
  var s = 0.0;
  for (var k = 0; k < NK; k++) { s += (*c)[k].y * mSy(i32((*c)[k].x + 0.5)); }
  return s;
}
fn topMatC(c : ptr<function, Col>) -> i32 {
  for (var k = NK - 1; k >= 0; k--) {
    if ((*c)[k].y > 0.02) { return i32((*c)[k].x + 0.5); }
  }
  return 0;
}

// Depth of the unconsolidated cover under the surface: contiguous top strata
// of clay / silt / sand / loam (films thinner than 2 cm don't end the run).
// Rock, regolith and boulder armor stop it — nothing roots in those.
fn colSoilTh(c : ptr<function, Col>) -> f32 {
  var s = 0.0;
  for (var k = NK - 1; k >= 0; k--) {
    let th = (*c)[k].y;
    if (th <= 1e-4) { continue; }
    let m = i32((*c)[k].x + 0.5);
    if (m == 1 || m == 2 || m == 3 || m == 5) {
      s += th;
    } else if (th > 0.02) {
      break;
    }
  }
  return s;
}

// Stratum head: unconfined water-table elevation while under capacity, stiff
// confined overpressure above it (a little extra volume ⇒ a big head rise,
// which is what drives artesian flow along a sealed aquifer). The pressure
// term is clamped at OP_MAX so a sealed pocket holds bounded head instead of
// being forced through an impermeable roof.
fn stratHead(zb : f32, th : f32, mat : i32, w : f32) -> f32 {
  let sy = max(mSy(mat), 1e-3);
  let cap = th * sy;
  if (w <= cap) { return zb + min(w / sy, th); }
  return zb + th + min(w - cap, OP_MAX) / SS_EFF;
}

// Merge the thinnest adjacent occupied pair (mass-conserving; the merged
// stratum takes the thicker member's material) and shift the stack down,
// freeing the top slot.
fn colMergeThinnest(c : ptr<function, Col>) {
  var best = 0;
  var bestTh = 1e9;
  for (var k = 0; k + 1 < NK; k++) {
    if ((*c)[k + 1].y <= 1e-4) { continue; }
    let s = (*c)[k].y + (*c)[k + 1].y;
    if (s < bestTh) { bestTh = s; best = k; }
  }
  let a = (*c)[best];
  let b = (*c)[best + 1];
  let mat = select(b.x, a.x, a.y >= b.y);
  (*c)[best] = vec4f(mat, a.y + b.y, a.z + b.z, 0.0);
  for (var k = best + 1; k + 1 < NK; k++) { (*c)[k] = (*c)[k + 1]; }
  (*c)[NK - 1] = vec4f(0.0);
}

// Deposit material on top of the column: merge with a same-material top
// stratum, otherwise start a new one (merging below first if the stack is full).
fn pushTop(c : ptr<function, Col>, mat : i32, amt : f32) {
  if (amt <= 0.0) { return; }
  var t = colTopIdx(c);
  if (t >= 0 && i32((*c)[t].x + 0.5) == mat) {
    (*c)[t].y += amt;
    return;
  }
  if (t == NK - 1) {
    colMergeThinnest(c);
    t = colTopIdx(c);
    if (t >= 0 && i32((*c)[t].x + 0.5) == mat) {
      (*c)[t].y += amt;
      return;
    }
  }
  (*c)[t + 1] = vec4f(f32(mat), amt, 0.0, 0.0);
}

// Remove thickness from the top down. Water that no longer fits in a
// shrunken stratum is squeezed out. Returns vec2(removed, released water).
fn stripTop(c : ptr<function, Col>, amt0 : f32) -> vec2f {
  var rem = amt0;
  var rel = 0.0;
  for (var k = NK - 1; k >= 0; k--) {
    if (rem <= 0.0) { break; }
    var s = (*c)[k];
    if (s.y <= 1e-4) { continue; }
    let take = min(rem, s.y);
    s.y -= take;
    rem -= take;
    if (s.y <= 1e-4) {
      rel += s.z;
      s = vec4f(0.0);
    } else {
      let cap = s.y * mSy(i32(s.x + 0.5)) + OP_MAX;
      if (s.z > cap) {
        rel += s.z - cap;
        s.z = cap;
      }
    }
    (*c)[k] = s;
  }
  return vec2f(amt0 - rem, rel);
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
// relief scales the vertical amplitude, isl scales the island footprint.
fn genHeight(gp : vec2f, seed : f32, relief : f32, isl : f32) -> f32 {
  let uv = gp / NF;
  let c = uv - vec2f(0.5);
  let off = vec2f(seed * 37.71, seed * 91.27);
  let p = uv * 5.0 + off;
  let warp = vec2f(fbm(p + vec2f(0.0, 5.2)), fbm(p + vec2f(3.1, 1.7))) * 1.7;
  var m = ridgedFbm(p * 0.85 + warp * 0.7);
  m = pow(max(m, 0.0), 1.55);
  let rolling = fbm(p * 0.55 + warp) * 0.55;
  let fall = 1.0 - smoothstep(0.30, 0.92, length(c) * 2.0 / max(isl, 0.1));
  var h = ((m * 30.0 + rolling * 9.0) * fall + fall * 2.5) * relief;
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
  rate      : f32,             // tool rate (depth or thickness / sec at center)
  radius    : f32,             // brush radius in cells
  toolActive: f32,
  seed      : f32,
  tool      : f32,             // TOOL_* enum
  material  : f32,             // layer index for add / replace
  depth     : f32,             // replace band: top offset below surface
  band      : f32,             // replace band thickness
  walls     : f32,             // 1 = solid impenetrable map edges
  cutZ      : f32,             // cutaway plane (0 = off) — pick skips hidden cells
  genRelief : f32,             // terrain generation: height amplitude scale
  genIsland : f32,             // terrain generation: island footprint scale
  genSoil   : f32,             // terrain generation: soil mantle depth scale
  genDepth  : f32,             // terrain generation: sediment column depth (m)
};

@group(0) @binding(0) var<uniform> U : SimU;
@group(0) @binding(1) var<storage, read_write> strata  : array<vec4f>; // NK per cell: (mat, thick, water, 0)
@group(0) @binding(2) var<storage, read_write> fields  : array<vec4f>; // water, sed, height cache, trees
@group(0) @binding(3) var<storage, read_write> flux    : array<vec4f>; // x:-X y:+X z:-Y w:+Y
@group(0) @binding(4) var<storage, read_write> vel     : array<vec2f>;
@group(0) @binding(5) var<storage, read_write> scratch : array<vec4f>; // ping target (strata / fields)
@group(0) @binding(6) var<storage, read_write> pick    : array<vec4f>; // [0] = (hitX, hitZ, hit?, t)
@group(0) @binding(7) var<storage, read_write> orig    : array<f32>;   // total height as generated
@group(0) @binding(8) var<storage, read_write> sources : array<vec4f>; // springs: x, z, rate, radius (radius 0 = free slot)

const NSRC : i32 = ${MAX_SOURCES};

const GRAV   : f32 = 9.81;
const VMAX   : f32 = 12.0;
const EVAP   : f32 = 0.0006;  // ≈2 min wall-clock half-life (was seconds — far too thirsty)
const KC     : f32 = 1.0;     // sediment capacity
const KS     : f32 = 0.30;    // dissolve rate
const KD     : f32 = 0.35;    // deposit rate
const KT     : f32 = 0.50;    // thermal creep rate

fn colLoad(i : u32) -> Col {
  var c : Col;
  for (var k = 0u; k < NKU; k++) { c[k] = strata[i * NKU + k]; }
  return c;
}
fn colStore(i : u32, c : ptr<function, Col>) {
  for (var k = 0u; k < NKU; k++) { strata[i * NKU + k] = (*c)[k]; }
}
fn colStoreScratch(i : u32, c : ptr<function, Col>) {
  for (var k = 0u; k < NKU; k++) { scratch[i * NKU + k] = (*c)[k]; }
}

// Terrain height comes from the per-cell cache in fields.z — the hot
// surface-water passes must not walk NK strata per neighbour lookup. The
// cache is maintained by every pass that changes thicknesses (and refreshed
// after the thermal ping-pong by refreshHeight).
fn terrAt(i : u32) -> f32 { return fields[i].z; }

fn totalH(x : i32, y : i32) -> f32 {
  if (x < 0 || y < 0 || x >= NI || y >= NI) {
    if (U.walls > 0.5) { return 1.0e6; }   // solid wall: nothing flows out
    return BASE - 2.0;                     // open boundary: falls off the map
  }
  let i = cIdx(x, y);
  return terrAt(i) + fields[i].x;
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
  let a = terrAt(cIdx(x, y))         + fields[cIdx(x, y)].x;
  let b = terrAt(cIdx(x + 1, y))     + fields[cIdx(x + 1, y)].x;
  let c = terrAt(cIdx(x, y + 1))     + fields[cIdx(x, y + 1)].x;
  let d = terrAt(cIdx(x + 1, y + 1)) + fields[cIdx(x + 1, y + 1)].x;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn bilFields(p : vec2f) -> vec4f {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  let a = fields[cIdx(x, y)];
  let b = fields[cIdx(x + 1, y)];
  let c = fields[cIdx(x, y + 1)];
  let d = fields[cIdx(x + 1, y + 1)];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// --- terrain generation: stratigraphic island columns ------------------------
// Layer-cake with noise-perturbed interfaces: bedrock basement, weathered
// regolith, then clay / gravel-lens / silt / clay / sand beds whose thickness
// fields are low-frequency so every bed stays laterally connected. The gravel
// lens pinches in and out — where present it is a confined aquifer sandwiched
// between aquitards. Soil, beach sand and boulder armor overprint the top.
@compute @workgroup_size(8, 8)
fn genTerrain(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  let gp = vec2f(f32(gid.x), f32(gid.y));
  let h  = genHeight(gp, U.seed, U.genRelief, U.genIsland);
  let hx = genHeight(gp + vec2f(1.5, 0.0), U.seed, U.genRelief, U.genIsland);
  let hy = genHeight(gp + vec2f(0.0, 1.5), U.seed, U.genRelief, U.genIsland);
  let slope = length(vec2f(hx - h, hy - h)) / 1.5;
  let so = vec2f(U.seed * 13.1, U.seed * 7.7);

  // basement top: a sedimentary BASIN — fill is deep under lowlands and the
  // coast (coastal-plain style, after the stacked-aquifer archetype) and
  // pinches out against the rock-cored mountains; a gentle regional tilt
  // makes the beds dip and outcrop on slopes
  let depthP = max(U.genDepth, 1.0);
  let tilt = (gp.x - NF * 0.5) * 0.014 + (gp.y - NF * 0.5) * 0.009;
  var cover = depthP * (0.8 + 1.2 * fbm(gp * 0.016 + so))
            * (1.0 - 0.7 * smoothstep(6.0, 20.0, h)) + tilt;
  cover = clamp(cover, 1.2, h - BASE - 1.0);
  let zBase = h - cover;

  // sedimentary sequence above the basement, bottom-up: TWO aquifer cycles
  // (gravel sealed by clay below the silt, and a second gravel horizon higher
  // up) so deep basins hold stacked confined aquifers. Nominal thicknesses
  // are normalized to exactly fill the cover so the column reaches the surface.
  let n1 = fbm(gp * 0.020 + so + vec2f(11.0, 3.0));
  let n2 = fbm(gp * 0.022 + so + vec2f(29.0, 47.0));
  let n3 = fbm(gp * 0.020 + so + vec2f(5.0, 71.0));
  let n4 = fbm(gp * 0.018 + so + vec2f(53.0, 13.0));
  let n5 = fbm(gp * 0.020 + so + vec2f(91.0, 37.0));
  let n6 = fbm(gp * 0.021 + so + vec2f(43.0, 83.0));
  let lens1 = smoothstep(0.42, 0.56, fbm(gp * 0.014 + so + vec2f(67.0, 19.0)));
  let lens2 = smoothstep(0.44, 0.58, fbm(gp * 0.013 + so + vec2f(7.0, 59.0)));
  var th = array<f32, 8>(
    0.4 + 0.7 * n1,                    // regolith: weathered basement cap
    0.9 + 1.6 * n1,                    // clay aquitard (lower seal)
    lens1 * (0.7 + 1.4 * n2),          // GRAVEL — the deep confined aquifer
    1.0 + 1.8 * n3,                    // silt: leaky aquitard
    0.6 + 1.1 * n4,                    // clay (upper seal)
    lens2 * (0.6 + 1.2 * n6),          // GRAVEL — second aquifer horizon
    0.7 + 1.3 * n2,                    // silt interbed
    0.9 + 1.6 * n5);                   // sand: upper unconfined aquifer
  var mats = array<i32, 8>(6, 1, 4, 2, 1, 4, 2, 3);
  var tot = 0.0;
  for (var k = 0; k < 8; k++) { tot += th[k]; }
  var c : Col;
  for (var k = 0; k < NK; k++) { c[k] = vec4f(0.0); }
  c[0] = vec4f(0.0, zBase - BASE, 0.0, 0.0);          // bedrock floor
  var slot = 1;
  for (var k = 0; k < 8; k++) {
    let t = th[k] * cover / max(tot, 1e-3);
    if (t > 0.05) {
      c[slot] = vec4f(f32(mats[k]), t, 0.0, 0.0);
      slot++;
    } else {
      c[slot - 1].y += t;     // pinched-out bed: fold into the one below
    }
  }

  // soil mantle overprint: loam, thick on gentle ground, scraped off ridges
  let soilN = fbm(gp * 0.05 + so);
  var soil = clamp(2.6 * U.genSoil * (1.0 - clamp(slope * 1.1, 0.0, 0.9)), 0.0, 3.0 * U.genSoil)
           * (0.5 + 0.7 * soilN);
  soil = min(soil, max(h, 0.0) * 0.5);
  if (soil > 0.05) {
    let r = stripTop(&c, soil);
    pushTop(&c, 5, r.x);
  }
  // beach sand at the shoreline
  if (h < 2.0) {
    let r = stripTop(&c, 0.6);
    pushTop(&c, 3, r.x);
  }
  // boulder armor patches on steeper ground
  let bn = fbm(gp * 0.06 + vec2f(U.seed * 3.3, U.seed * 21.7));
  let b = smoothstep(0.60, 0.74, bn) * smoothstep(0.10, 0.30, slope) * 1.6;
  if (b > 0.05) {
    let r = stripTop(&c, b);
    pushTop(&c, 7, r.x);
  }

  // charge the aquifer: saturated near/below sea level, drying with altitude
  var zb = BASE;
  for (var k = 0; k < NK; k++) {
    let thk = c[k].y;
    if (thk > 1e-4) {
      let zMid = zb + 0.5 * thk;
      let sat = clamp(0.9 - 0.07 * max(zMid, 0.0), 0.05, 0.9);
      c[k].z = thk * mSy(i32(c[k].x + 0.5)) * sat;
      zb += thk;
    }
  }
  colStore(i, &c);
  orig[i] = h;

  // forests on gentle mid-altitude soil
  let treeN = fbm(gp * 0.030 + vec2f(U.seed * 5.7, U.seed * 2.9));
  var trees = smoothstep(0.55, 0.70, treeN)
            * smoothstep(2.5, 4.5, h) * (1.0 - smoothstep(14.0, 20.0, h))
            * (1.0 - smoothstep(0.35, 0.70, slope));
  trees = select(0.0, trees, soil > 0.3);
  fields[i] = vec4f(0.0, 0.0, BASE + colSumTh(&c), trees);
  flux[i] = vec4f(0.0);
  vel[i] = vec2f(0.0);
  if (gid.x == 0u && gid.y == 0u) {        // new world: clear all springs
    for (var s = 0; s < NSRC; s++) { sources[s] = vec4f(0.0); }
  }
}

// --- springs: shift+click toggles a persistent water source -------------------
@compute @workgroup_size(1)
fn placeSource() {
  let pk = pick[0];
  if (pk.z < 0.5) { return; }
  // clicking near an existing spring removes it...
  for (var s = 0; s < NSRC; s++) {
    if (sources[s].w > 0.0 &&
        distance(sources[s].xy, pk.xy) < max(sources[s].w, U.radius)) {
      sources[s] = vec4f(0.0);
      return;
    }
  }
  // ...otherwise it occupies the first free slot (full table: click ignored)
  for (var s = 0; s < NSRC; s++) {
    if (sources[s].w <= 0.0) {
      sources[s] = vec4f(pk.x, pk.y, U.rate, U.radius);
      return;
    }
  }
}

@compute @workgroup_size(8, 8)
fn applySources(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let p = vec2f(f32(gid.x), f32(gid.y));
  var add = 0.0;
  for (var s = 0; s < NSRC; s++) {
    let so = sources[s];
    if (so.w <= 0.0) { continue; }
    let d = distance(p, so.xy);
    if (d < so.w) {
      add += so.z * U.dt * (0.5 + 0.5 * cos(d / so.w * PI));
    }
  }
  if (add > 0.0) {
    fields[gid.y * N + gid.x].x += add;
  }
}

@compute @workgroup_size(8, 8)
fn clearWater(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  for (var k = 0u; k < NKU; k++) {                 // drain every stratum
    strata[i * NKU + k].z = 0.0;
  }
  let F = fields[i];
  fields[i] = vec4f(0.0, 0.0, F.z, F.w);           // keep height cache + trees
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
      let inMap = p.x >= 0.0 && p.x <= NF - 1.0 && p.z >= 0.0 && p.z <= NF - 1.0
               && (U.cutZ < 0.5 || p.z >= U.cutZ);   // cutaway hides z < cutZ
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

// --- apply the active tool at the picked point -------------------------------

// Append a segment to a column being rebuilt, merging with the previous
// segment when materials match. Water from zero-thickness scraps and slot
// overflow is never dropped: it goes to spill (→ surface water).
fn emitSeg(o : ptr<function, Col>, n : ptr<function, i32>,
           mat : i32, th : f32, w : f32, spill : ptr<function, f32>) {
  if (th <= 1e-4) {                  // scrap: fold into the segment below so
    if ((*n) > 0) {                  // even sub-0.1mm slivers keep their mass
      (*o)[(*n) - 1].y += th;
      (*o)[(*n) - 1].z += w;
    } else {
      (*spill) += w;                 // no segment yet (can't happen with a
    }                                // bedrock floor): route water to surface
    return;
  }
  if ((*n) > 0 && i32((*o)[(*n) - 1].x + 0.5) == mat) {
    (*o)[(*n) - 1].y += th;
    (*o)[(*n) - 1].z += w;
    return;
  }
  if ((*n) >= NK) {                  // out of slots: fold into the top stratum
    (*o)[NK - 1].y += th;
    (*o)[NK - 1].z += w;
    return;
  }
  (*o)[(*n)] = vec4f(f32(mat), th, w, 0.0);
  (*n) += 1;
}

// Replace the depth band [dA, dB] below the surface with material mat,
// preserving total column height (split strata at the band edges, rebuild the
// stack). Water in the band stays up to the new material's capacity; the
// rest is squeezed out to the surface.
fn replaceBand(c : ptr<function, Col>, dA : f32, dB : f32, mat : i32,
               F : ptr<function, vec4f>) {
  let H = colSumTh(c);
  let bLo = clamp(H - dB, 0.0, H);                 // band in bottom-up coords
  let bHi = clamp(H - dA, 0.0, H);
  if (bHi - bLo <= 1e-4) { return; }
  var out : Col;
  for (var k = 0; k < NK; k++) { out[k] = vec4f(0.0); }
  var n = 0;
  var spill = 0.0;
  var bandW = 0.0;
  var z = 0.0;
  for (var k = 0; k < NK; k++) {                   // parts below the band
    let s = (*c)[k];
    if (s.y <= 1e-4) { continue; }
    let z0 = z;
    let z1 = z + s.y;
    z = z1;
    let keep = max(0.0, min(z1, bLo) - z0);
    let ov = max(0.0, min(z1, bHi) - max(z0, bLo));
    bandW += s.z * ov / s.y;
    emitSeg(&out, &n, i32(s.x + 0.5), keep, s.z * keep / s.y, &spill);
  }
  let bth = bHi - bLo;                             // the band itself
  let bcap = bth * mSy(mat) + OP_MAX;
  if (bandW > bcap) {
    spill += bandW - bcap;
    bandW = bcap;
  }
  emitSeg(&out, &n, mat, bth, bandW, &spill);
  z = 0.0;
  for (var k = 0; k < NK; k++) {                   // parts above the band
    let s = (*c)[k];
    if (s.y <= 1e-4) { continue; }
    let z0 = z;
    let z1 = z + s.y;
    z = z1;
    let keep = max(0.0, z1 - max(z0, bHi));
    emitSeg(&out, &n, i32(s.x + 0.5), keep, s.z * keep / s.y, &spill);
  }
  (*c) = out;
  (*F).x += spill;
}

@compute @workgroup_size(8, 8)
fn toolApply(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let pk = pick[0];
  if (U.toolActive < 0.5 || pk.z < 0.5) { return; }
  let d = distance(vec2f(f32(gid.x), f32(gid.y)), pk.xy);
  if (d >= U.radius) { return; }
  let w = 0.5 + 0.5 * cos(d / U.radius * PI);
  let i = gid.y * N + gid.x;
  let tool = i32(U.tool + 0.5);
  let mi = i32(U.material + 0.5);
  var F = fields[i];

  switch (tool) {
    case 0: {                                       // pour water
      F.x += U.rate * U.dt * w;
    }
    case 1: {                                       // add material on top
      var c = colLoad(i);
      let amt = U.rate * U.dt * w * 0.45;
      pushTop(&c, mi, amt);
      colStore(i, &c);
      F.z += amt;                                   // maintain the height cache
    }
    case 2: {                                       // dig: remove from the top down
      var c = colLoad(i);
      let r = stripTop(&c, U.rate * U.dt * w * 0.45);
      colStore(i, &c);
      F.z -= r.x;
      F.x += r.y;             // squeezed-out groundwater pools in the pit
    }
    case 3: {                                       // replace a band below the surface
      var c = colLoad(i);
      replaceBand(&c, U.depth, U.depth + U.band, mi, &F);
      colStore(i, &c);
    }
    case 4: {                                       // plant trees (need soil, no flood)
      var c = colLoad(i);
      if (F.x < 0.15 && colSoilTh(&c) > 0.3) {
        F.w = min(1.0, F.w + U.rate * U.dt * w * 0.5);
      }
    }
    case 5: {                                       // remove trees
      F.w = max(0.0, F.w - U.rate * U.dt * w * 1.5);
    }
    default: {}
  }
  fields[i] = F;
}

// --- pipe-model outflow flux --------------------------------------------------
@compute @workgroup_size(8, 8)
fn fluxPass(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  let h = terrAt(i) + fields[i].x;
  var f = flux[i];
  let dh = vec4f(
    h - totalH(x - 1, y),
    h - totalH(x + 1, y),
    h - totalH(x, y - 1),
    h - totalH(x, y + 1));
  f = max(vec4f(0.0), f * 0.995 + U.dt * GRAV * dh);
  let total = f.x + f.y + f.z + f.w;
  if (total > 1e-6) {
    let k = min(1.0, fields[i].x / (total * U.dt));
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
  let d0 = fields[i].x;
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
  fields[i].x = d1;
}

// --- column hydrology: infiltration, percolation, artesian seepage ------------
// Column-local (no neighbour access, safe in place). Surface water soaks into
// the topmost stratum at the exposed material's rate; water drains down the
// stack throttled by the *receiving* material's conductivity, so a clay bed
// below ⇒ a perched water table above, for free. Volume past a stratum's
// capacity is confined overpressure: it seeps upward fast through a permeable
// roof, barely through an aquitard, and wells out as a spring when the top
// stratum can't hold it either.
@compute @workgroup_size(8, 8)
fn infiltrate(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  var c = colLoad(i);
  var F = fields[i];
  let t = colTopIdx(&c);
  if (t < 0) { return; }

  // surface water into the top stratum (tree roots help)
  let tm = i32(c[t].x + 0.5);
  var inf = min(F.x, mInfil(tm) * (1.0 + 0.6 * clamp(F.w, 0.0, 1.0)) * U.dt);
  inf = min(inf, max(0.0, c[t].y * mSy(tm) - c[t].z));
  F.x -= inf;
  c[t].z += inf;

  // gravity percolation down the stack, throttled by the receiver
  for (var k = t; k >= 1; k--) {
    let mB = i32(c[k - 1].x + 0.5);
    var drain = min(c[k].z * 0.25, mK(mB) * VPERC * U.dt);
    drain = min(drain, max(0.0, c[k - 1].y * mSy(mB) - c[k - 1].z));
    c[k].z -= drain;
    c[k - 1].z += drain;
  }

  // confined overpressure seeps upward through each stratum's roof. A truly
  // impermeable roof (rock, K = 0) seals completely: the pocket stores the
  // water at the OP_MAX-clamped head — never forced through the seal.
  for (var k = 0; k < t; k++) {                    // bottom-up: pressure climbs
    let ex = c[k].z - c[k].y * mSy(i32(c[k].x + 0.5));
    if (ex <= 0.0) { continue; }
    let roofK = mK(i32(c[k + 1].x + 0.5));
    let up = min(ex * 0.5, roofK * VPERC * U.dt);
    c[k].z -= up;
    c[k + 1].z += up;
  }
  // top stratum: anything past capacity wells out — a spring
  let exT = c[t].z - c[t].y * mSy(i32(c[t].x + 0.5));
  if (exT > 0.0) {
    let out = max(exT * 0.5, exT - OP_MAX);
    c[t].z -= out;
    F.x += out;
  }
  colStore(i, &c);
  fields[i] = F;
}

// --- groundwater: lateral Darcy exchange between neighbouring columns ---------
// Symmetric gather over the 4 neighbours. Strata don't align across columns,
// so the two stacks are swept by depth overlap (two-pointer merge). Flow per
// overlapping pair = harmonic-mean conductivity × overlap × head difference,
// clamped to 20% of the donor stratum's water weighted by the overlap share —
// both sides compute identical transfers, so mass is conserved exactly.
// Writes the new stacks to scratch (copied back after the pass).
@compute @workgroup_size(8, 8)
fn gwLateral(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  var c = colLoad(i);
  var net : array<f32, ${NK}>;
  for (var k = 0; k < NK; k++) { net[k] = 0.0; }

  for (var d = 0; d < 4; d++) {
    var nx = x;
    var ny = y;
    if (d == 0) { nx = x - 1; } else if (d == 1) { nx = x + 1; }
    else if (d == 2) { ny = y - 1; } else { ny = y + 1; }

    if (nx < 0 || ny < 0 || nx >= NI || ny >= NI) {
      if (U.walls > 0.5) { continue; }
      // open edge = the sea, head 0: aquifers above sea level drain out,
      // below it they take seawater in (coastal recharge)
      var zb = BASE;
      for (var a = 0; a < NK; a++) {
        let s = c[a];
        if (s.y <= 1e-4) { continue; }
        let ma = i32(s.x + 0.5);
        let q = mK(ma) * s.y * (0.0 - stratHead(zb, s.y, ma, s.z)) * U.dt;
        net[a] += clamp(q, -0.2 * s.z, max(0.0, 0.2 * (s.y * mSy(ma) - s.z)));
        zb += s.y;
      }
      continue;
    }

    let j = cIdx(nx, ny);
    var a = 0;
    var b = 0;
    var za = BASE;                                  // bottom of our stratum a
    var zbn = BASE;                                 // bottom of neighbour's b
    for (var it = 0; it < 2 * NK; it++) {
      if (a >= NK || b >= NK) { break; }
      let sa = c[a];
      let sb = strata[j * NKU + u32(b)];
      if (sa.y <= 1e-4) { a++; continue; }
      if (sb.y <= 1e-4) { b++; continue; }
      let topA = za + sa.y;
      let topB = zbn + sb.y;
      let ov = min(topA, topB) - max(za, zbn);
      if (ov > 1e-4 && (sa.z > 1e-6 || sb.z > 1e-6)) {
        let ma = i32(sa.x + 0.5);
        let mb = i32(sb.x + 0.5);
        let Ka = mK(ma);
        let Kb = mK(mb);
        let Kf = 2.0 * Ka * Kb / max(Ka + Kb, 1e-6);   // aquitard wins
        let ha = stratHead(za, sa.y, ma, sa.z);
        let hb = stratHead(zbn, sb.y, mb, sb.z);
        var q = Kf * ov * (hb - ha) * U.dt;            // > 0: flows to us
        q = clamp(q, -0.2 * sa.z * ov / sa.y, 0.2 * sb.z * ov / sb.y);
        net[a] += q;
      }
      if (topA < topB) { za = topA; a++; } else { zbn = topB; b++; }
    }
  }

  for (var k = 0; k < NK; k++) {
    c[k].z = max(0.0, c[k].z + net[k]);
  }
  colStoreScratch(i, &c);
}

// --- hydraulic erosion / deposition (per-material) -----------------------------
@compute @workgroup_size(8, 8)
fn erosion(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;

  var F = fields[i];
  let tC = terrAt(i);
  let tL = terrAt(cIdx(x - 1, y));
  let tR = terrAt(cIdx(x + 1, y));
  let tB = terrAt(cIdx(x, y - 1));
  let tT = terrAt(cIdx(x, y + 1));
  let grad = 0.5 * vec2f(tR - tL, tT - tB);
  let sinA = length(grad) / sqrt(1.0 + dot(grad, grad));
  let slope = max(0.05, sinA);

  let sp = length(vel[i]);
  let depthFac = clamp(F.x * 6.0, 0.0, 1.0);   // dry film doesn't carve
  let cap = KC * slope * sp * depthFac;
  let st = F.y;

  // dry settling: when the carrying water is gone (evaporated / drained),
  // suspended sediment must return to the ground instead of lingering as
  // phantom mass that the water-clamped deposit branch can never place
  // NOTE: erosion never writes F.z — neighbours read the height cache during
  // this dispatch, so every invocation must see the same pre-pass snapshot
  // (refreshHeight trues the cache up at the end of the frame; the lag is
  // bounded by the 0.05/substep erosion cap)
  if (F.x < 0.001 && st > 0.0) {
    var c = colLoad(i);
    pushTop(&c, 3, st);                             // settles out as sand
    colStore(i, &c);
    F.y = 0.0;
    fields[i] = F;
    return;
  }

  // relief clamp vs the 4-neighbour mean keeps the erode/deposit feedback from
  // checkerboarding: a local pit stops digging, a local bump stops growing
  let avg4 = 0.25 * (tL + tR + tB + tT);
  if (cap > st) {
    var c = colLoad(i);
    let m = topMatC(&c);
    let shield = 1.0 - 0.8 * clamp(F.w, 0.0, 1.0);  // roots bind the ground
    var a = min(KS * mErode(m) * shield * (cap - st) * U.dt, 0.05);
    a = min(a, max(0.0, tC - (avg4 - 0.9)));
    let r = stripTop(&c, a);                        // strip from the top down
    colStore(i, &c);
    F.y = st + r.x;
    F.x += r.y;       // only the squeezed-out groundwater joins the flow —
                      // solids and water are separate budgets, so the HUD
                      // water total is honest (no minting on dissolution)
  } else {
    var a = min(KD * (st - cap) * U.dt, st);
    a = min(a, max(0.0, (avg4 + 0.3) - tC));   // deposits fill lows, never build bumps
    var c = colLoad(i);
    pushTop(&c, 3, a);                              // settles out as sand
    colStore(i, &c);
    F.y = st - a;
  }
  fields[i] = F;
}

// --- transport: flux-form sediment advection + tree health ---------------------
// Sediment rides the same pipe fluxes as the water: each face moves
// concentration × moved-water-volume, and both cells of a face compute the
// identical transfer from the giver's state — exactly conservative, unlike
// the semi-Lagrangian gather this replaces (which manufactured ~0.5% of the
// suspended load per substep). Writes the next fields vector into scratch.
@compute @workgroup_size(8, 8)
fn transport(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  let F = fields[i];

  // outflow: the share of our (pre-move) water that left this substep takes
  // the same share of the suspended sediment with it
  let fc = flux[i];
  let outV = (fc.x + fc.y + fc.z + fc.w) * U.dt;
  let d0i = F.x + outV;                 // estimate of our water before the move
  var newSed = F.y;
  if (d0i > 1e-6) { newSed -= F.y * outV / d0i; }
  // inflows: each neighbour's flux component pointing at us × its concentration
  for (var k = 0; k < 4; k++) {
    var nx = x;
    var ny = y;
    var comp = 0;
    if (k == 0) { nx = x - 1; comp = 1; }        // left nbr's +X flux
    else if (k == 1) { nx = x + 1; comp = 0; }   // right nbr's -X flux
    else if (k == 2) { ny = y - 1; comp = 3; }   // bottom nbr's +Y flux
    else { ny = y + 1; comp = 2; }               // top nbr's -Y flux
    if (nx < 0 || ny < 0 || nx >= NI || ny >= NI) { continue; }
    let j = cIdx(nx, ny);
    var fj = flux[j];
    let fIn = fj[comp];
    if (fIn <= 0.0) { continue; }
    let Fj = fields[j];
    let dj = Fj.x + (fj.x + fj.y + fj.z + fj.w) * U.dt;
    if (dj > 1e-6) { newSed += Fj.y * fIn * U.dt / dj; }
  }
  newSed = max(newSed, 0.0);

  // trees drown under deep/fast water and need soil to stand in
  var c = colLoad(i);
  let soil = colSoilTh(&c);
  var tr = F.w;
  let drown = smoothstep(0.5, 1.5, F.x) + smoothstep(4.0, 7.0, length(vel[i]));
  tr = max(0.0, tr - (drown * 0.3 + select(0.0, 0.15, soil < 0.2)) * U.dt);

  scratch[i] = vec4f(F.x, newSed, F.z, tr);
}

// --- thermal erosion: per-material talus slippage over 8 neighbours -----------
// Symmetric gather: both cells of a pair compute the same transfer from the
// giver's top material, so mass is conserved. Received material is collected
// per material id and pushed onto the stack; our own give is stripped off the
// top (its squeezed-out water pools on the surface). New strata → scratch.
fn topOf(j : u32) -> vec2f {                       // (mat, thickness) of the top stratum
  for (var k = NK - 1; k >= 0; k--) {
    let s = strata[j * NKU + u32(k)];
    if (s.y > 0.02) { return s.xy; }
  }
  return vec2f(0.0, 0.0);
}

@compute @workgroup_size(8, 8)
fn thermal(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * N + gid.x;
  var c = colLoad(i);
  let h = terrAt(i);
  // both sides of a pair MUST derive (material, thickness) the same way —
  // topOf — or the giver's cap and the receiver's cap disagree and the pass
  // manufactures mass
  let tSelf = topOf(i);
  let mTop = i32(tSelf.x + 0.5);
  let topTh = tSelf.y;
  var give = 0.0;
  var recv : array<f32, 8>;
  for (var m = 0; m < 8; m++) { recv[m] = 0.0; }

  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      if (ox == 0 && oy == 0) { continue; }
      let nx = x + ox;
      let ny = y + oy;
      if (nx < 0 || ny < 0 || nx >= NI || ny >= NI) { continue; }
      let j = cIdx(nx, ny);
      let hj = terrAt(j);
      let dist = length(vec2f(f32(ox), f32(oy)));
      if (h > hj) {                                // we give from our top stratum
        let creep = mCreep(mTop);
        if (creep > 0.0) {
          let lim = mTalus(mTop) * dist;
          let dh = h - hj;
          if (dh > lim) {
            // per-pair cap 12% of the stratum: 8 * 0.12 < 1 keeps it positive
            give += min(KT * creep * (dh - lim) * U.dt / dist * 0.5,
                        topTh * 0.12);
          }
        }
      } else {                                     // neighbour gives from its top
        let tj = topOf(j);
        let m = i32(tj.x + 0.5);
        let creep = mCreep(m);
        if (creep > 0.0) {
          let lim = mTalus(m) * dist;
          let dh = hj - h;
          if (dh > lim) {
            recv[m] += min(KT * creep * (dh - lim) * U.dt / dist * 0.5,
                           tj.y * 0.12);
          }
        }
      }
    }
  }

  let r = stripTop(&c, give);          // give ≤ 0.96 × topTh: stays in the top stratum
  for (var m = 0; m < 8; m++) {
    pushTop(&c, m, recv[m]);
  }
  colStoreScratch(i, &c);
  if (r.y > 0.0) {
    fields[i].x += r.y;                // own cell only: no concurrent writer
  }
}

// --- height cache refresh (after the thermal ping-pong) -----------------------
@compute @workgroup_size(8, 8)
fn refreshHeight(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  var c = colLoad(i);
  fields[i].z = BASE + colSumTh(&c);
}
`;

// ---------------------------------------------------------------------------
// WGSL — mass budget reduction (own module: the sim bind group is at the
// 8-storage-buffer limit). One workgroup sums every cell into
// stats[0] = (surface water, suspended sediment, groundwater, terrain volume).
// ---------------------------------------------------------------------------

const WGSL_MASS = /* wgsl */`
const N : u32 = ${N}u;
const NK : u32 = ${NK}u;
const NSRC : i32 = ${MAX_SOURCES};

@group(0) @binding(0) var<storage, read> fields  : array<vec4f>;
@group(0) @binding(1) var<storage, read> strata  : array<vec4f>;
@group(0) @binding(2) var<storage, read> sources : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> stats : array<vec4f>;
// stats[0] = (surface water, sediment, groundwater, terrain)
// stats[1] = (active spring count, 0, 0, 0)

var<workgroup> partial : array<vec4f, 256>;

@compute @workgroup_size(256)
fn massReduce(@builtin(local_invocation_id) lid : vec3u) {
  var s = vec4f(0.0);
  for (var i = lid.x; i < N * N; i += 256u) {
    let F = fields[i];
    var gw = 0.0;
    var th = 0.0;
    for (var k = 0u; k < NK; k++) {
      let st = strata[i * NK + k];
      th += st.y;
      gw += st.z;
    }
    s += vec4f(F.x, F.y, gw, th);
  }
  partial[lid.x] = s;
  for (var off = 128u; off > 0u; off >>= 1u) {
    workgroupBarrier();
    if (lid.x < off) {
      partial[lid.x] += partial[lid.x + off];
    }
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    stats[0] = partial[0];
    var n = 0.0;
    for (var k = 0; k < NSRC; k++) {
      if (sources[k].w > 0.0) { n += 1.0; }
    }
    stats[1] = vec4f(n, 0.0, 0.0, 0.0);
  }
}
`;

// ---------------------------------------------------------------------------
// WGSL — render module (sky / terrain / trees / cutaway / water, one bind group)
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
  misc     : vec4f,            // x: time, y: toolActive, z: brushRadius, w: aspect
  misc2    : vec4f,            // x: cutZ (0 = off), y: view mode, z: walls, w: unused
};

@group(0) @binding(0) var<uniform> R : RU;
@group(0) @binding(1) var<storage, read> strata : array<vec4f>;  // NK per cell
@group(0) @binding(2) var<storage, read> fields : array<vec4f>;  // water, sed, height, trees
@group(0) @binding(3) var<storage, read> vel    : array<vec2f>;
@group(0) @binding(4) var<storage, read> orig   : array<f32>;
@group(0) @binding(5) var<storage, read> pick   : array<vec4f>;
@group(0) @binding(6) var<storage, read> sources : array<vec4f>;

const NSRC : i32 = ${MAX_SOURCES};

// Per-cell column taps (the stack walk is cheap; bilinear users blend the
// results of the 4 corner cells instead of blending raw strata).
fn topMatAt(i : u32) -> i32 {
  for (var k = NK - 1; k >= 0; k--) {
    let s = strata[i * NKU + u32(k)];
    if (s.y > 0.02) { return i32(s.x + 0.5); }
  }
  return 0;
}
fn satAt(i : u32) -> f32 {              // column groundwater saturation 0..1
  var w = 0.0;
  var cp = 0.0;
  for (var k = 0u; k < NKU; k++) {
    let s = strata[i * NKU + k];
    w += s.z;
    cp += s.y * mSy(i32(s.x + 0.5));
  }
  return clamp(w / max(cp, 0.05), 0.0, 1.0);
}
fn soilAt(i : u32) -> f32 {             // rooting depth (clay/silt/sand/loam top)
  var s = 0.0;
  for (var k = NK - 1; k >= 0; k--) {
    let st = strata[i * NKU + u32(k)];
    if (st.y <= 1e-4) { continue; }
    let m = i32(st.x + 0.5);
    if (m == 1 || m == 2 || m == 3 || m == 5) {
      s += st.y;
    } else if (st.y > 0.02) {
      break;
    }
  }
  return s;
}
fn bilFields(p : vec2f) -> vec4f {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  let a = fields[cIdx(x, y)];
  let b = fields[cIdx(x + 1, y)];
  let c = fields[cIdx(x, y + 1)];
  let d = fields[cIdx(x + 1, y + 1)];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
fn bilO(p : vec2f) -> f32 {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  let a = orig[cIdx(x, y)];
  let b = orig[cIdx(x + 1, y)];
  let c = orig[cIdx(x, y + 1)];
  let d = orig[cIdx(x + 1, y + 1)];
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
fn bilT(p : vec2f) -> f32 { return bilFields(p).z; }   // terrain height cache
fn bilW(p : vec2f) -> f32 { return bilFields(p).x; }
fn bilTW(p : vec2f) -> f32 {
  let F = bilFields(p);
  return F.z + F.x;
}

// 4-corner bilinear weights for blending per-cell column taps.
struct Corners {
  i : vec4u,     // the 4 cell indices
  w : vec4f,     // their bilinear weights
};
fn corners(p : vec2f) -> Corners {
  let q = clamp(p, vec2f(0.0), vec2f(NF - 1.001));
  let ip = floor(q);
  let f = q - ip;
  let x = i32(ip.x);
  let y = i32(ip.y);
  var c : Corners;
  c.i = vec4u(cIdx(x, y), cIdx(x + 1, y), cIdx(x, y + 1), cIdx(x + 1, y + 1));
  c.w = vec4f((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
              (1.0 - f.x) * f.y, f.x * f.y);
  return c;
}
fn bilSat(p : vec2f) -> f32 {
  let c = corners(p);
  return dot(c.w, vec4f(satAt(c.i.x), satAt(c.i.y), satAt(c.i.z), satAt(c.i.w)));
}
fn bilSoil(p : vec2f) -> f32 {
  let c = corners(p);
  return dot(c.w, vec4f(soilAt(c.i.x), soilAt(c.i.y), soilAt(c.i.z), soilAt(c.i.w)));
}

fn cutHidden(wp : vec3f) -> bool {
  return R.misc2.x > 0.5 && wp.z < R.misc2.x;
}

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

// --- material palette ----------------------------------------------------------
fn rockColor(p : vec2f, h : f32) -> vec3f {
  let detail = fbm(p * 0.55 + h * 0.18);
  return mix(vec3f(0.27, 0.24, 0.21), vec3f(0.49, 0.46, 0.42), detail);
}
fn clayColor(p : vec2f) -> vec3f {
  return mix(vec3f(0.47, 0.27, 0.16), vec3f(0.63, 0.41, 0.24), fbm(p * 0.30));
}
fn siltColor(p : vec2f) -> vec3f {
  return mix(vec3f(0.52, 0.48, 0.40), vec3f(0.67, 0.63, 0.54), fbm(p * 0.6));
}
fn sandColor(p : vec2f) -> vec3f {
  return mix(vec3f(0.69, 0.60, 0.41), vec3f(0.79, 0.71, 0.52), vnoise(p * 3.1));
}
fn gravelColor(p : vec2f) -> vec3f {
  // fine pebble speckle, cooler than sand
  let cells = step(0.5, vnoise(p * 4.2));
  let tone = vnoise(p * 7.7 + vec2f(3.1, 8.9));
  return mix(vec3f(0.36, 0.36, 0.36), vec3f(0.62, 0.60, 0.56), cells * 0.55 + tone * 0.45);
}
fn loamColor(p : vec2f) -> vec3f {
  return mix(vec3f(0.23, 0.16, 0.10), vec3f(0.37, 0.27, 0.17), fbm(p * 0.5));
}
fn regolithColor(p : vec2f, h : f32) -> vec3f {
  // weathered, cracked rock — paler than the fresh basement
  let crack = step(0.62, vnoise(p * 2.4));
  return mix(rockColor(p, h) * 1.15 + vec3f(0.06, 0.05, 0.03), vec3f(0.30, 0.27, 0.23), crack * 0.4);
}
fn boulderColor(p : vec2f) -> vec3f {
  // chunky speckle so boulder fields read as rubble, not paint
  let cells = step(0.48, vnoise(p * 1.5));
  let tone = vnoise(p * 3.3 + vec2f(13.7, 7.1));
  return mix(vec3f(0.28, 0.26, 0.24), vec3f(0.56, 0.54, 0.50), cells * 0.7 + tone * 0.3);
}
fn matColor(m : i32, p : vec2f, h : f32) -> vec3f {
  switch (m) {
    case 1: { return clayColor(p); }
    case 2: { return siltColor(p); }
    case 3: { return sandColor(p); }
    case 4: { return gravelColor(p); }
    case 5: { return loamColor(p); }
    case 6: { return regolithColor(p, h); }
    case 7: { return boulderColor(p); }
    default: { return rockColor(p, h); }
  }
}

// --- terrain material ---------------------------------------------------------
fn terrainAlbedo(wp : vec3f, n : vec3f) -> vec3f {
  let p = wp.xz;
  let h = wp.y;
  let mode = i32(R.misc2.y + 0.5);

  if (mode == 1) {        // moisture view: groundwater saturation heat-map
    let sat1 = bilSat(p);
    return mix(vec3f(0.80, 0.78, 0.71), vec3f(0.04, 0.20, 0.80), sat1 * sat1 * 0.85 + sat1 * 0.15);
  }

  // exposed material: blend the 4 corner cells' top-material colors
  let cc = corners(p);
  var a = matColor(topMatAt(cc.i.x), p, h) * cc.w.x
        + matColor(topMatAt(cc.i.y), p, h) * cc.w.y
        + matColor(topMatAt(cc.i.z), p, h) * cc.w.z
        + matColor(topMatAt(cc.i.w), p, h) * cc.w.w;

  if (mode == 2) {        // strata view: bare material colors, no dressing
    return a;
  }

  let F = bilFields(p);
  let sat = bilSat(p);
  let slope = 1.0 - n.y;
  let rnd = fbm(p * 0.16);
  let soil = bilSoil(p);
  let tm = topMatAt(cc.i.x);
  let rubble = f32(tm == 7 || tm == 0 || tm == 6);

  // grass settles on gentle soil above the beach line, but not on rubble
  let grass = mix(vec3f(0.10, 0.24, 0.05), vec3f(0.26, 0.37, 0.10), rnd);
  let grassM = smoothstep(0.15, 0.50, soil)
             * (1.0 - smoothstep(0.12, 0.34, slope + (rnd - 0.5) * 0.1))
             * smoothstep(1.5, 3.0, h + rnd * 1.2)
             * (1.0 - rubble);
  a = mix(a, grass, grassM * 0.85);

  // forest floor darkens under the canopy
  a = mix(a, vec3f(0.05, 0.14, 0.04), smoothstep(0.15, 0.8, F.w) * 0.55);

  // snow caps bare high ground
  a = mix(a, vec3f(0.92, 0.94, 0.98),
          smoothstep(21.0, 25.0, h + rnd * 5.0) * (1.0 - smoothstep(0.18, 0.5, slope)));

  // damp soil reads darker where the water table is high
  a *= mix(1.0, 0.80, sat * smoothstep(0.15, 0.5, soil));

  // erosion scars expose rock; deposition leaves silt (wide tap — the raw
  // per-cell delta is noisy and would dapple the basin)
  let q = wp.xz;
  let delta = 0.25 * ((bilT(q) - bilO(q))
            + (bilT(q + vec2f(1.6, 0.0)) - bilO(q + vec2f(1.6, 0.0)))
            + (bilT(q + vec2f(-0.8, 1.4)) - bilO(q + vec2f(-0.8, 1.4)))
            + (bilT(q + vec2f(-0.8, -1.4)) - bilO(q + vec2f(-0.8, -1.4))));
  a = mix(a, rockColor(p, h) * 0.85, clamp(-delta * 0.7, 0.0, 1.0) * 0.55);
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
  let wp = vec3f(x, fields[vi].z, y);
  out.wp = wp;
  out.pos = R.viewProj * vec4f(wp, 1.0);
  return out;
}
@fragment
fn fsTerrain(in : TOut) -> @location(0) vec4f {
  if (cutHidden(in.wp)) { discard; }
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

  // spring markers: pulsing rings around persistent water sources
  for (var s = 0; s < NSRC; s++) {
    let so = sources[s];
    if (so.w <= 0.0) { continue; }
    let d = distance(p, so.xy);
    if (d > so.w + 1.5) { continue; }
    let pulse = 0.55 + 0.45 * sin(R.misc.x * 3.5 - d * 1.2);
    let ring = (1.0 - smoothstep(0.15, 0.7, abs(d - so.w * 0.55)))
             + (1.0 - smoothstep(0.0, 0.5, d)) * 0.8;
    col += vec3f(0.20, 0.65, 1.0) * ring * pulse * 0.7;
  }

  return vec4f(finish(col), 1.0);
}

// ============================ X-RAY PASS =====================================
// Fullscreen volumetric view: raymarch the column volume accumulating the
// optical depth of groundwater (saturated pores) and surface water, mapped
// onto a shallow→deep color scale, over a ghosted terrain silhouette.
fn xrayScale(od : f32) -> vec3f {
  let w = 1.0 - exp(-od * 0.30);
  var c = mix(vec3f(0.55, 0.97, 1.00), vec3f(0.08, 0.40, 0.95), smoothstep(0.0, 0.5, w));
  c = mix(c, vec3f(0.18, 0.05, 0.50), smoothstep(0.5, 1.0, w));
  return c;
}

@fragment
fn fsXray(in : SkyOut) -> @location(0) vec4f {
  let tf = R.camFwd.w;
  let aspect = R.misc.w;
  let rd0 = R.camFwd.xyz +
    R.camRight.xyz * in.ndc.x * tf * aspect +
    R.camUp.xyz * in.ndc.y * tf;
  let rd = normalize(rd0);
  let ro = R.camPos.xyz;
  var col = skyColor(rd);

  // clip the ray to the map volume (respecting the cutaway plane)
  let zMin = select(0.0, R.misc2.x, R.misc2.x > 0.5);
  let bmin = vec3f(0.0, BASE - 2.0, zMin);
  let bmax = vec3f(NF - 1.0, 52.0, NF - 1.0);
  var safe = rd;
  // axis-aligned rays: avoid /0, keeping the component's sign
  if (abs(safe.x) < 1e-5) { safe.x = select(-1e-5, 1e-5, safe.x >= 0.0); }
  if (abs(safe.y) < 1e-5) { safe.y = select(-1e-5, 1e-5, safe.y >= 0.0); }
  if (abs(safe.z) < 1e-5) { safe.z = select(-1e-5, 1e-5, safe.z >= 0.0); }
  let ta = (bmin - ro) / safe;
  let tb = (bmax - ro) / safe;
  let t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));

  if (t1 > t0) {
    let steps = 140;
    let dt = (t1 - t0) / f32(steps);
    var od = 0.0;        // water optical depth (ground + surface)
    var soil = 0.0;      // ground optical depth — depth cue for the ghost
    var hitT = -1.0;
    for (var i = 0; i < steps; i++) {
      let t = t0 + (f32(i) + 0.5) * dt;
      let p = ro + rd * t;
      let q = p.xz;
      let F = bilFields(q);
      let hT = F.z;
      let surf = hT + F.x;
      if (p.y < surf && hitT < 0.0) { hitT = t; }
      if (p.y < hT) {
        soil += dt;
        // stratum containing this depth contributes its own saturation, so
        // perched and confined aquifers show as separate glowing sheets
        let ci = cIdx(i32(q.x + 0.5), i32(q.y + 0.5));
        var zb = BASE;
        for (var k = 0u; k < NKU; k++) {
          let s = strata[ci * NKU + k];
          if (s.y <= 1e-4) { continue; }
          let zt = zb + s.y;
          if (p.y >= zb && p.y < zt) {
            // weight by water VOLUME fraction, not saturation — saturated
            // bedrock holds a trace (Sy 0.02) and must not read like an
            // aquifer over a 26 m column; full gravel/sand ≈ 1 per metre
            let volFrac = s.z / max(s.y, 1e-3);
            od += dt * clamp((volFrac - 0.03) * 4.5, 0.0, 1.0);
            break;
          }
          zb = zt;
        }
      } else if (p.y < surf) {
        od += dt * 1.2;   // open water reads slightly denser
      }
    }
    if (hitT > 0.0) {     // ghosted terrain silhouette behind the volume
      let hp = ro + rd * hitT;
      let n = terrNormal(hp.xz);
      let ndl = clamp(dot(n, R.sun.xyz), 0.0, 1.0);
      let ghost = vec3f(0.68, 0.72, 0.78) * (0.35 + 0.65 * ndl)
                * exp(-soil * 0.012);
      col = mix(col, ghost, 0.50);
    }
    let alpha = 1.0 - exp(-od * 0.42);
    col = mix(col, xrayScale(od), alpha);
  }
  return vec4f(finish(col), 1.0);
}

// ============================ TREE PASS ======================================
// One instance per cell; low-poly pine (3-sided trunk cone + 5-sided canopy
// cone = 24 vertices) scaled by the cell's tree density, hidden when sparse.
struct TreeOut {
  @builtin(position) pos : vec4f,
  @location(0) wp  : vec3f,
  @location(1) nrm : vec3f,
  @location(2) col : vec3f,
};

fn coneVert(tri : f32, corner : u32, sides : f32,
            yBase : f32, yTip : f32, rBase : f32, phase : f32) -> vec3f {
  if (corner == 0u) { return vec3f(0.0, yTip, 0.0); }
  let k = tri + select(0.0, 1.0, corner == 2u);
  let ang = k / sides * 2.0 * PI + phase;
  return vec3f(cos(ang) * rBase, yBase, sin(ang) * rBase);
}
fn coneNrm(tri : f32, sides : f32, yBase : f32, yTip : f32, rBase : f32, phase : f32) -> vec3f {
  let ang = (tri + 0.5) / sides * 2.0 * PI + phase;
  return normalize(vec3f(cos(ang) * (yTip - yBase), rBase, sin(ang) * (yTip - yBase)));
}

@vertex
fn vsTree(@builtin(vertex_index) vi : u32,
          @builtin(instance_index) inst : u32) -> TreeOut {
  var out : TreeOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);   // default: clipped away
  out.wp = vec3f(0.0);
  out.nrm = vec3f(0.0, 1.0, 0.0);
  out.col = vec3f(0.0);
  let density = fields[inst].w;
  if (density < 0.12) { return out; }

  let gx = f32(inst % N);
  let gy = f32(inst / N);
  let h1 = hash21(vec2f(gx * 1.37 + 0.71, gy * 2.93 + 4.1));
  let h2 = hash21(vec2f(gy * 3.11 + 9.7, gx * 0.83 + 1.9));
  let base2 = vec2f(gx + (h1 - 0.5) * 0.9, gy + (h2 - 0.5) * 0.9);
  if (bilW(base2) > 0.45) { return out; }          // submerged: hide
  let gh = bilT(base2);
  let size = (0.65 + 0.7 * h2) * (0.45 + 0.75 * min(density, 1.0)) * 4.8;
  let phase = h1 * 6.2832;

  var lp : vec3f;
  if (vi < 9u) {                                   // trunk
    let tri = f32(vi / 3u);
    lp = coneVert(tri, vi % 3u, 3.0, 0.0, 0.42 * size, 0.075 * size, phase);
    out.nrm = coneNrm(tri, 3.0, 0.0, 0.42 * size, 0.075 * size, phase);
    out.col = vec3f(0.25, 0.16, 0.10);
  } else {                                         // canopy
    let k = vi - 9u;
    let tri = f32(k / 3u);
    lp = coneVert(tri, k % 3u, 5.0, 0.20 * size, 1.05 * size, 0.42 * size, phase);
    out.nrm = coneNrm(tri, 5.0, 0.20 * size, 1.05 * size, 0.42 * size, phase);
    out.col = mix(vec3f(0.06, 0.22, 0.06), vec3f(0.13, 0.36, 0.11), h1);
  }
  let wp = vec3f(base2.x + lp.x, gh - 0.15 + lp.y, base2.y + lp.z);
  out.wp = wp;
  out.pos = R.viewProj * vec4f(wp, 1.0);
  return out;
}
@fragment
fn fsTree(in : TreeOut) -> @location(0) vec4f {
  if (cutHidden(in.wp)) { discard; }
  let n = normalize(in.nrm);
  let ndl = clamp(dot(n, R.sun.xyz), 0.0, 1.0);
  var col = in.col * (ndl * vec3f(1.30, 1.20, 1.00) * 1.35
                      + vec3f(0.30, 0.38, 0.50) * (0.50 + 0.50 * n.y));
  col = applyFog(col, in.wp);
  return vec4f(finish(col), 1.0);
}

// ============================ CUTAWAY PASS ===================================
// A vertical curtain along X at z = cutZ showing the column cross-section:
// strata colors, the saturated groundwater zone, and any surface water.
struct CutOut {
  @builtin(position) pos : vec4f,
  @location(0) wp : vec3f,
};
@vertex
fn vsCut(@builtin(vertex_index) vi : u32) -> CutOut {
  var out : CutOut;
  let q = vi / 6u;
  let c = vi % 6u;
  var cx = 0.0;
  var cy = 0.0;
  switch (c) {
    case 0u: { cx = 0.0; cy = 0.0; }
    case 1u: { cx = 1.0; cy = 0.0; }
    case 2u: { cx = 0.0; cy = 1.0; }
    case 3u: { cx = 1.0; cy = 0.0; }
    case 4u: { cx = 1.0; cy = 1.0; }
    default: { cx = 0.0; cy = 1.0; }
  }
  let wp = vec3f(f32(q) + cx, mix(BASE - 1.0, 50.0, cy), R.misc2.x);
  out.wp = wp;
  out.pos = R.viewProj * vec4f(wp, 1.0);
  return out;
}
@fragment
fn fsCut(in : CutOut) -> @location(0) vec4f {
  let colP = vec2f(in.wp.x, R.misc2.x);
  let F = bilFields(colP);
  let hTop = F.z;
  let hWat = hTop + F.x;
  let y = in.wp.y;
  if (y > hWat) { discard; }

  let tp = vec2f(in.wp.x, y);
  var c : vec3f;
  if (y > hTop) {                                  // surface water column
    c = mix(vec3f(0.10, 0.30, 0.38), vec3f(0.04, 0.16, 0.24),
            clamp((hWat - y) / max(F.x, 0.05), 0.0, 1.0));
  } else {
    // walk the column to the stratum containing y: material color, its own
    // saturated zone and phreatic line — perched and confined aquifers each
    // get their own blue band in the section
    let ci = cIdx(i32(in.wp.x + 0.5), i32(R.misc2.x + 0.5));
    var zb = BASE;
    var mat = 0;
    var wf = 0.0;
    var tableD = 1e9;
    for (var k = 0u; k < NKU; k++) {
      let s = strata[ci * NKU + k];
      if (s.y <= 1e-4) { continue; }
      let zt = zb + s.y;
      if (y >= zb && y < zt) {
        mat = i32(s.x + 0.5);
        let wt = zb + min(s.z / max(mSy(mat), 1e-3), s.y);
        if (y <= wt) {
          // tint by water volume fraction so a charged gravel bed reads as
          // a solid blue aquifer while damp bedrock stays faint
          wf = clamp((s.z / max(s.y, 1e-3) - 0.015) * 6.0, 0.0, 1.0);
        }
        if (s.z > 0.01) { tableD = abs(y - wt); }
        break;
      }
      zb = zt;
    }
    if (mat == 0) {                                // bedrock with faint veins
      c = mix(vec3f(0.30, 0.28, 0.27), vec3f(0.46, 0.44, 0.42),
              vnoise(vec2f(in.wp.x * 0.33, y * 0.9)));
    } else {
      c = matColor(mat, tp, y);
    }
    c = mix(c, vec3f(0.08, 0.30, 0.58), 0.60 * wf);
    c += vec3f(0.20, 0.50, 0.80) * (1.0 - smoothstep(0.0, 0.22, tableD)) * 0.7;
  }

  // simple depth shading so the face doesn't look flat
  c *= 0.58 + 0.42 * clamp((y - (hTop - 30.0)) / 30.0, 0.0, 1.0);
  c = applyFog(c, in.wp);
  return vec4f(finish(c), 1.0);
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
  let terr = fields[vi].z;
  let d = fields[vi].x;
  var h = terr + d;
  if (d < 0.0012) {
    h = terr - 0.06;               // sink dry verts just below ground
  }
  let wp = vec3f(x, h, y);
  out.wp = wp;
  out.depth = d;
  out.pos = R.viewProj * vec4f(wp, 1.0);
  return out;
}
@fragment
fn fsWater(in : WOut) -> @location(0) vec4f {
  if (cutHidden(in.wp)) { discard; }
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
  let mud = clamp(bilFields(p).y * 0.6, 0.0, 0.8);
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
  let gpuErrorShown = false;
  device.addEventListener('uncapturederror', (e) => {
    console.error(e.error);
    if (!gpuErrorShown) {
      gpuErrorShown = true;
      fail('WebGPU error: ' + e.error.message);
    }
  });
  async function checkModule(mod, name) {
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) {
      throw new Error(`${name} WGSL failed to compile:\n` +
        errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }
  }

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // --- buffers ---------------------------------------------------------------
  const cells = N * N;
  const mk = (bytes) => device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const bufStrata   = mk(cells * NK * 16); // NK vec4 strata per cell (mat, thick, water, 0)
  const bufFields   = mk(cells * 16);   // vec4: water, sed, height cache, trees
  const bufFlux     = mk(cells * 16);
  const bufVel      = mk(cells * 8);
  const bufScratch  = mk(cells * NK * 16); // ping target (strata, or fields in its head)
  const bufPick     = mk(16);
  const bufOrig     = mk(cells * 4);
  const bufSources  = mk(MAX_SOURCES * 16);
  const bufStats    = mk(32);
  const statsStage = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const simUBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
  await checkModule(simModule, 'sim');
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
      { binding: 1, resource: { buffer: bufStrata } },
      { binding: 2, resource: { buffer: bufFields } },
      { binding: 3, resource: { buffer: bufFlux } },
      { binding: 4, resource: { buffer: bufVel } },
      { binding: 5, resource: { buffer: bufScratch } },
      { binding: 6, resource: { buffer: bufPick } },
      { binding: 7, resource: { buffer: bufOrig } },
      { binding: 8, resource: { buffer: bufSources } },
    ],
  });
  const simPipe = {};
  for (const ep of ['genTerrain', 'clearWater', 'pickCast', 'toolApply', 'fluxPass',
                    'depthVel', 'infiltrate', 'gwLateral', 'erosion', 'transport',
                    'thermal', 'refreshHeight', 'placeSource', 'applySources']) {
    simPipe[ep] = device.createComputePipeline({
      layout: simLayout,
      compute: { module: simModule, entryPoint: ep },
    });
  }

  // mass-budget reduction: separate module + bind group (sim group is full)
  const massModule = device.createShaderModule({ code: WGSL_MASS });
  await checkModule(massModule, 'mass');
  const massBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const massPipe = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [massBGL] }),
    compute: { module: massModule, entryPoint: 'massReduce' },
  });
  const massBG = device.createBindGroup({
    layout: massBGL,
    entries: [
      { binding: 0, resource: { buffer: bufFields } },
      { binding: 1, resource: { buffer: bufStrata } },
      { binding: 2, resource: { buffer: bufSources } },
      { binding: 3, resource: { buffer: bufStats } },
    ],
  });

  // --- render pipelines --------------------------------------------------------
  const SAMPLES = 4;
  const renModule = device.createShaderModule({ code: WGSL_RENDER });
  await checkModule(renModule, 'render');
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
      { binding: 1, resource: { buffer: bufStrata } },
      { binding: 2, resource: { buffer: bufFields } },
      { binding: 3, resource: { buffer: bufVel } },
      { binding: 4, resource: { buffer: bufOrig } },
      { binding: 5, resource: { buffer: bufPick } },
      { binding: 6, resource: { buffer: bufSources } },
    ],
  });

  const mkRenderPipe = (vs, fs, opts = {}) => device.createRenderPipeline({
    layout: renLayout,
    vertex: { module: renModule, entryPoint: vs },
    fragment: {
      module: renModule, entryPoint: fs,
      targets: [opts.blend ? {
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      } : { format }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: SAMPLES },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: opts.depthWrite !== false,
      depthCompare: opts.depthAlways ? 'always' : 'less',
    },
  });
  const skyPipe = mkRenderPipe('vsSky', 'fsSky', { depthWrite: false, depthAlways: true });
  const xrayPipe = mkRenderPipe('vsSky', 'fsXray', { depthWrite: false, depthAlways: true });
  const terrainPipe = mkRenderPipe('vsTerrain', 'fsTerrain');
  const treePipe = mkRenderPipe('vsTree', 'fsTree');
  const cutPipe = mkRenderPipe('vsCut', 'fsCut');
  const waterPipe = mkRenderPipe('vsWater', 'fsWater', { blend: true });

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
    yaw: 0.9, pitch: 0.62, dist: N * 1.13,
    target: [N / 2, 6, N / 2],
    fov: (45 * Math.PI) / 180,
  };
  const mouse = { x: 0, y: 0, over: false, pouring: false, orbiting: false, lx: 0, ly: 0 };

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  let placeSourceQueued = false;
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    mouse.x = e.clientX;      // touch/pen may not have sent a pointermove yet;
    mouse.y = e.clientY;      // the pick ray must come from THIS click
    mouse.over = true;
    if (e.button === 0) {
      if (e.shiftKey) {
        placeSourceQueued = true;   // shift+click: toggle a spring, don't pour
      } else {
        mouse.pouring = true;
      }
    }
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
    cam.dist = Math.min(N * 2.75, Math.max(40, cam.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  // --- UI ------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const rateEl = $('rate'), radEl = $('rad'), depthEl = $('depth'), bandEl = $('band'), cutEl = $('cut');
  const reliefEl = $('relief'), islandEl = $('island'), soilEl = $('soil'), gdepthEl = $('gdepth');
  rateEl.addEventListener('input', () => { $('rateV').textContent = (+rateEl.value).toFixed(1); });
  radEl.addEventListener('input', () => { $('radV').textContent = (+radEl.value).toFixed(1); });
  depthEl.addEventListener('input', () => { $('depthV').textContent = (+depthEl.value).toFixed(1); });
  bandEl.addEventListener('input', () => { $('bandV').textContent = (+bandEl.value).toFixed(1); });
  cutEl.max = String(N - 1);          // slider spans the actual grid depth
  cutEl.addEventListener('input', () => {
    $('cutV').textContent = +cutEl.value > 0 ? cutEl.value : 'Off';
  });
  reliefEl.addEventListener('input', () => { $('reliefV').textContent = (+reliefEl.value).toFixed(2); });
  islandEl.addEventListener('input', () => { $('islandV').textContent = (+islandEl.value).toFixed(2); });
  soilEl.addEventListener('input', () => { $('soilV').textContent = (+soilEl.value).toFixed(2); });
  gdepthEl.addEventListener('input', () => { $('gdepthV').textContent = (+gdepthEl.value).toFixed(1); });

  // map resolution applies via reload (every buffer/shader is sized off N)
  const sizeEl = $('mapsize');
  sizeEl.value = String(N);
  sizeEl.addEventListener('change', () => {
    const u = new URL(location.href);
    u.searchParams.set('n', sizeEl.value);
    location.href = u.href;
  });

  let tool = TOOL_WATER;
  let material = 3;        // sand
  let viewMode = 0;

  const toolBtns = [...document.querySelectorAll('#tools button')];
  const matBtns = [...document.querySelectorAll('#mats button')];
  const viewBtns = [...document.querySelectorAll('#views button')];
  function setTool(t) {
    tool = t;
    toolBtns.forEach((b) => b.classList.toggle('active', +b.dataset.tool === t));
    $('matRow').style.display = (t === TOOL_ADD || t === TOOL_REPLACE) ? '' : 'none';
    $('bandRow').style.display = t === TOOL_REPLACE ? '' : 'none';
  }
  function setMaterial(m) {
    material = m;
    matBtns.forEach((b) => b.classList.toggle('active', +b.dataset.mat === m));
  }
  function setView(v) {
    viewMode = v;
    viewBtns.forEach((b) => b.classList.toggle('active', +b.dataset.view === v));
    $('legend').style.display = v === 3 ? '' : 'none';
  }
  toolBtns.forEach((b) => b.addEventListener('click', () => setTool(+b.dataset.tool)));
  matBtns.forEach((b) => b.addEventListener('click', () => setMaterial(+b.dataset.mat)));
  viewBtns.forEach((b) => b.addEventListener('click', () => setView(+b.dataset.view)));
  setTool(TOOL_WATER);
  setMaterial(3);
  setView(0);

  let seed = Math.random() * 100;
  let pendingReset = true;      // generate terrain on first frame
  let pendingClear = false;
  $('reset').addEventListener('click', () => {
    seed = Math.random() * 100;
    pendingReset = true;
  });
  $('clear').addEventListener('click', () => { pendingClear = true; });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') { seed = Math.random() * 100; pendingReset = true; }
    if (e.key === 'c' || e.key === 'C') { pendingClear = true; }
    if (e.key >= '1' && e.key <= '6') setTool(+e.key - 1);
    if (e.key === 'v' || e.key === 'V') setView((viewMode + 1) % 4);
    if (e.key === 'b' || e.key === 'B') $('walls').checked = !$('walls').checked;
  });

  // --- per-frame uniform staging ---------------------------------------------------
  const simU = new Float32Array(24);
  const renU = new Float32Array(44);
  const fpsEl = $('fps');
  const massEl = $('mass');
  let frames = 0, fpsT = performance.now(), simTime = 0;

  // mass budget readback: copy the GPU totals into a staging buffer and map
  // it asynchronously; the HUD lags a frame or two, which is fine
  let statsPending = false;
  const fmtMass = (v) => v >= 10000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(1);
  function readbackStats() {
    statsPending = true;
    statsStage.mapAsync(GPUMapMode.READ).then(() => {
      const [w, sed, gw, terr, nSprings] = new Float32Array(statsStage.getMappedRange());
      massEl.textContent =
        `terrain ${fmtMass(terr)} · water ${fmtMass(w)} · ground ${fmtMass(gw)}` +
        ` · sediment ${fmtMass(sed)}` +
        (nSprings > 0
          ? ` · springs ${nSprings}/${MAX_SOURCES}${nSprings >= MAX_SOURCES ? ' — table full' : ''}`
          : '');
      statsStage.unmap();
      statsPending = false;
    }).catch((e) => {
      // latch statsPending so the HUD stops instead of silently retrying every
      // frame; unmap if the failure happened after a successful map so later
      // frames can't encode a copy into a mapped buffer
      console.error(e);
      if (statsStage.mapState === 'mapped') statsStage.unmap();
      massEl.textContent = 'mass readback failed — see console';
    });
  }

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

    const cut = +cutEl.value;
    const walls = $('walls').checked ? 1 : 0;
    const active = mouse.pouring && !mouse.orbiting ? 1 : 0;

    simTime += DT * SUBSTEPS;
    simU.set([eye[0], eye[1], eye[2], 0], 0);
    simU.set([rd[0], rd[1], rd[2], mouse.over ? 1 : 0], 4);
    simU[8] = DT;
    simU[9] = simTime;
    simU[10] = +rateEl.value;
    simU[11] = +radEl.value;
    simU[12] = active;
    simU[13] = seed;
    simU[14] = tool;
    simU[15] = material;
    simU[16] = +depthEl.value;
    simU[17] = +bandEl.value;
    simU[18] = walls;
    simU[19] = cut;
    simU[20] = +reliefEl.value;
    simU[21] = +islandEl.value;
    simU[22] = +soilEl.value;
    simU[23] = +gdepthEl.value;
    device.queue.writeBuffer(simUBuf, 0, simU);

    renU.set(viewProj, 0);
    renU.set([eye[0], eye[1], eye[2], 0], 16);
    renU.set([right[0], right[1], right[2], 0], 20);
    renU.set([up[0], up[1], up[2], 0], 24);
    renU.set([fwd[0], fwd[1], fwd[2], tanF], 28);
    const sunL = Math.hypot(0.55, 0.62, 0.32);
    renU.set([0.55 / sunL, 0.62 / sunL, 0.32 / sunL, 0], 32);
    renU.set([t, active, +radEl.value, aspect], 36);
    renU.set([cut, viewMode, walls, 0], 40);
    device.queue.writeBuffer(renUBuf, 0, renU);

    const enc = device.createCommandEncoder();

    const runPass = (pass, names) => {
      for (const name of names) {
        pass.setPipeline(simPipe[name]);
        pass.setBindGroup(0, simBG);
        // pickCast and placeSource are single-thread passes; dispatching them
        // per-cell would run the spring toggle 1024 times concurrently
        if (name === 'pickCast' || name === 'placeSource') pass.dispatchWorkgroups(1);
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
      runPass(pass, placeSourceQueued && mouse.over ? ['pickCast', 'placeSource'] : ['pickCast']);
      pass.end();
      placeSourceQueued = false;
    }
    for (let s = 0; s < SUBSTEPS; s++) {
      // surface water + column hydrology, then the lateral groundwater
      // ping-pong (gwLateral writes new strata into scratch)
      let pass = enc.beginComputePass();
      runPass(pass, ['toolApply', 'applySources', 'fluxPass', 'depthVel', 'infiltrate', 'gwLateral']);
      pass.end();
      enc.copyBufferToBuffer(bufScratch, 0, bufStrata, 0, cells * NK * 16);
      // erosion mutates its own column in place; transport ping-pongs fields
      pass = enc.beginComputePass();
      runPass(pass, ['erosion', 'transport']);
      pass.end();
      enc.copyBufferToBuffer(bufScratch, 0, bufFields, 0, cells * 16);
    }
    {
      const pass = enc.beginComputePass();
      runPass(pass, ['thermal']);
      pass.end();
      enc.copyBufferToBuffer(bufScratch, 0, bufStrata, 0, cells * NK * 16);
      // thermal moved material between columns: rebuild the height cache
      const pass2 = enc.beginComputePass();
      runPass(pass2, ['refreshHeight']);
      pass2.end();
    }

    // mass budget totals (HUD); copy out only while the staging buffer is free
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(massPipe);
      pass.setBindGroup(0, massBG);
      pass.dispatchWorkgroups(1);
      pass.end();
      if (!statsPending) enc.copyBufferToBuffer(bufStats, 0, statsStage, 0, 32);
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
    if (viewMode === 3) {
      // volumetric X-ray: one fullscreen raymarch (sky + ghost terrain +
      // groundwater volume); the surface passes would just occlude it
      rp.setPipeline(xrayPipe);
      rp.draw(3);
    } else {
      rp.setPipeline(skyPipe);
      rp.draw(3);
      rp.setIndexBuffer(indexBuf, 'uint32');
      rp.setPipeline(terrainPipe);
      rp.drawIndexed(indices.length);
      rp.setPipeline(treePipe);
      rp.draw(24, cells);
      if (cut > 0) {
        rp.setPipeline(cutPipe);
        rp.draw(6 * (N - 1));
      }
      rp.setPipeline(waterPipe);
      rp.drawIndexed(indices.length);
    }
    rp.end();

    device.queue.submit([enc.finish()]);
    if (!statsPending) readbackStats();

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
