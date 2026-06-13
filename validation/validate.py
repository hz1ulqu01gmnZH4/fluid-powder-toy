"""Headless validation of the Hydra Terra shaders + simulation on lavapipe."""
import json, subprocess, math, struct, sys, os
import numpy as np
import wgpu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

meta = json.loads(subprocess.check_output(["node", os.path.join(HERE, "extract.js")]).decode())
N, SUBSTEPS, DT, NSRC, NK = meta["N"], meta["SUBSTEPS"], meta["DT"], meta["MAX_SOURCES"], meta["NK"]
BASE = -26.0                    # world floor datum, mirrors WGSL_COMMON
sim_src = open(os.path.join(HERE, "sim.wgsl")).read()
ren_src = open(os.path.join(HERE, "render.wgsl")).read()
mass_src = open(os.path.join(HERE, "mass.wgsl")).read()

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
print("adapter:", adapter.info["device"])

cells = N * N
def mkbuf(size):
    return device.create_buffer(
        size=size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

strata   = mkbuf(cells * NK * 16)  # NK vec4 per cell: (mat, thick, water, 0)
fields   = mkbuf(cells * 16)       # vec4: water, sed, height cache, trees
flux     = mkbuf(cells * 16)
vel      = mkbuf(cells * 8)
scratch  = mkbuf(cells * NK * 16)  # ping target (strata, or fields in its head)
pick     = mkbuf(16)
orig     = mkbuf(cells * 4)
sources  = mkbuf(NSRC * 16)        # springs: x, z, rate, radius
stats    = mkbuf(32)               # mass totals (water, sed, gw, terrain) + spring count
simU = device.create_buffer(size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
renU = device.create_buffer(size=176, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

sim_mod = device.create_shader_module(code=sim_src)
ren_mod = device.create_shader_module(code=ren_src)
mass_mod = device.create_shader_module(code=mass_src)
print("shader modules compiled OK")

sim_bgl = device.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
] + [
    {"binding": b, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}
    for b in range(1, 9)
])
sim_layout = device.create_pipeline_layout(bind_group_layouts=[sim_bgl])
sim_bg = device.create_bind_group(layout=sim_bgl, entries=[
    {"binding": 0, "resource": {"buffer": simU, "offset": 0, "size": 96}},
    {"binding": 1, "resource": {"buffer": strata, "offset": 0, "size": cells * NK * 16}},
    {"binding": 2, "resource": {"buffer": fields, "offset": 0, "size": cells * 16}},
    {"binding": 3, "resource": {"buffer": flux, "offset": 0, "size": cells * 16}},
    {"binding": 4, "resource": {"buffer": vel, "offset": 0, "size": cells * 8}},
    {"binding": 5, "resource": {"buffer": scratch, "offset": 0, "size": cells * NK * 16}},
    {"binding": 6, "resource": {"buffer": pick, "offset": 0, "size": 16}},
    {"binding": 7, "resource": {"buffer": orig, "offset": 0, "size": cells * 4}},
    {"binding": 8, "resource": {"buffer": sources, "offset": 0, "size": NSRC * 16}},
])

eps = ["genTerrain", "clearWater", "pickCast", "toolApply", "fluxPass",
       "depthVel", "infiltrate", "gwLateral", "erosion", "transport",
       "thermal", "refreshHeight", "placeSource", "applySources"]
pipes = {}
for ep in eps:
    pipes[ep] = device.create_compute_pipeline(
        layout=sim_layout, compute={"module": sim_mod, "entry_point": ep})
print("compute pipelines OK:", ", ".join(eps))

mass_bgl = device.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
    {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
])
mass_pipe = device.create_compute_pipeline(
    layout=device.create_pipeline_layout(bind_group_layouts=[mass_bgl]),
    compute={"module": mass_mod, "entry_point": "massReduce"})
mass_bg = device.create_bind_group(layout=mass_bgl, entries=[
    {"binding": 0, "resource": {"buffer": fields, "offset": 0, "size": cells * 16}},
    {"binding": 1, "resource": {"buffer": strata, "offset": 0, "size": cells * NK * 16}},
    {"binding": 2, "resource": {"buffer": sources, "offset": 0, "size": NSRC * 16}},
    {"binding": 3, "resource": {"buffer": stats, "offset": 0, "size": 32}},
])
print("mass-reduce pipeline OK")

# --- render pipelines ---------------------------------------------------------
ren_bgl = device.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
     "buffer": {"type": wgpu.BufferBindingType.uniform}},
] + [
    {"binding": b, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
     "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}
    for b in range(1, 7)
])
ren_layout = device.create_pipeline_layout(bind_group_layouts=[ren_bgl])
ren_bg = device.create_bind_group(layout=ren_bgl, entries=[
    {"binding": 0, "resource": {"buffer": renU, "offset": 0, "size": 176}},
    {"binding": 1, "resource": {"buffer": strata, "offset": 0, "size": cells * NK * 16}},
    {"binding": 2, "resource": {"buffer": fields, "offset": 0, "size": cells * 16}},
    {"binding": 3, "resource": {"buffer": vel, "offset": 0, "size": cells * 8}},
    {"binding": 4, "resource": {"buffer": orig, "offset": 0, "size": cells * 4}},
    {"binding": 5, "resource": {"buffer": pick, "offset": 0, "size": 16}},
    {"binding": 6, "resource": {"buffer": sources, "offset": 0, "size": NSRC * 16}},
])

FMT = wgpu.TextureFormat.bgra8unorm
SAMPLES = 4
common_ds = {"format": wgpu.TextureFormat.depth24plus, "depth_write_enabled": True,
             "depth_compare": wgpu.CompareFunction.less,
             "stencil_front": {}, "stencil_back": {}}
def mk_pipe(vs, fs, blend=None, ds=None):
    return device.create_render_pipeline(
        layout=ren_layout,
        vertex={"module": ren_mod, "entry_point": vs, "buffers": []},
        fragment={"module": ren_mod, "entry_point": fs,
                  "targets": [{"format": FMT, **({"blend": blend} if blend else {})}]},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
        multisample={"count": SAMPLES},
        depth_stencil=ds or common_ds,
    )
sky_pipe = mk_pipe("vsSky", "fsSky",
                   ds={**common_ds, "depth_write_enabled": False,
                       "depth_compare": wgpu.CompareFunction.always})
xray_pipe = mk_pipe("vsSky", "fsXray",
                    ds={**common_ds, "depth_write_enabled": False,
                        "depth_compare": wgpu.CompareFunction.always})
terr_pipe = mk_pipe("vsTerrain", "fsTerrain")
tree_pipe = mk_pipe("vsTree", "fsTree")
cut_pipe = mk_pipe("vsCut", "fsCut")
blend = {"color": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add},
         "alpha": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add}}
water_pipe = mk_pipe("vsWater", "fsWater", blend=blend)
print("render pipelines OK: sky, xray, terrain, tree, cut, water")

# --- index buffer ----------------------------------------------------------------
idx = np.zeros(((N - 1) * (N - 1) * 6,), dtype=np.uint32)
k = 0
for y in range(N - 1):
    base = y * N
    for x in range(N - 1):
        i = base + x
        idx[k:k + 6] = [i, i + 1, i + N, i + 1, i + N + 1, i + N]
        k += 6
index_buf = device.create_buffer(size=idx.nbytes, usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST)
device.queue.write_buffer(index_buf, 0, idx.tobytes())

# --- helpers -----------------------------------------------------------------------
def write_simU(dt, time, rate, radius, active, seed, ro, rd, hover,
               tool=0, material=3, depth=0.0, band=2.0, walls=0.0, cut=0.0,
               relief=1.0, island=1.0, soil=1.0, gdepth=7.0):
    data = struct.pack("24f",
        ro[0], ro[1], ro[2], 0.0,
        rd[0], rd[1], rd[2], 1.0 if hover else 0.0,
        dt, time, rate, radius,
        active, seed, float(tool), float(material),
        depth, band, walls, cut,
        relief, island, soil, gdepth)
    device.queue.write_buffer(simU, 0, data)

def gpu_mass():
    """Run massReduce and return (water, sed, gw, terrain, springs, 0, 0, 0)."""
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(mass_pipe)
    p.set_bind_group(0, mass_bg)
    p.dispatch_workgroups(1, 1, 1)
    p.end()
    device.queue.submit([enc.finish()])
    return np.frombuffer(device.queue.read_buffer(stats, 0, size=32), dtype=np.float32).copy()

def dispatch(names):
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    for n in names:
        p.set_pipeline(pipes[n])
        p.set_bind_group(0, sim_bg)
        if n in ("pickCast", "placeSource"):   # single-thread passes
            p.dispatch_workgroups(1, 1, 1)
        else:
            p.dispatch_workgroups(math.ceil(N / 8), math.ceil(N / 8), 1)
    p.end()
    device.queue.submit([enc.finish()])

def read_f32(buf, count):
    data = device.queue.read_buffer(buf, 0, size=count * 4)
    return np.frombuffer(data, dtype=np.float32).copy()

def copy(src, dst, nbytes):
    enc = device.create_command_encoder()
    enc.copy_buffer_to_buffer(src, 0, dst, 0, nbytes)
    device.queue.submit([enc.finish()])

def read_strata():
    return read_f32(strata, cells * NK * 4).reshape(cells, NK, 4)

def read_fields():
    return read_f32(fields, cells * 4).reshape(cells, 4)

def heights(S):
    """Absolute terrain elevation per cell from a strata array."""
    return BASE + S[:, :, 1].sum(axis=1)

def mat_volume(S, m):
    return float(S[:, :, 1][S[:, :, 0].round() == m].sum())

def col_top(S, i):
    """(slot, mat, thickness) of the topmost occupied stratum of cell i."""
    for k in range(NK - 1, -1, -1):
        if S[i, k, 1] > 1e-4:
            return k, int(round(S[i, k, 0])), float(S[i, k, 1])
    return -1, 0, 0.0

def sim_frame():
    dispatch(["pickCast"])
    for s in range(SUBSTEPS):
        dispatch(["toolApply", "applySources", "fluxPass", "depthVel", "infiltrate", "gwLateral"])
        copy(scratch, strata, cells * NK * 16)
        dispatch(["erosion", "transport"])
        copy(scratch, fields, cells * 16)
    dispatch(["thermal"])
    copy(scratch, strata, cells * NK * 16)
    dispatch(["refreshHeight"])

def read_sources():
    return np.frombuffer(device.queue.read_buffer(sources, 0, size=NSRC * 16),
                         dtype=np.float32).reshape(NSRC, 4).copy()

# --- test 1: stratigraphic terrain generation ----------------------------------------
write_simU(DT, 0, 3, 4, 0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["genTerrain"])
S0 = read_strata()
F0 = read_fields()
t0 = heights(S0)
assert np.isfinite(S0).all(), "strata have non-finite values"
assert (S0[:, :, 1] >= -1e-4).all(), "negative stratum thickness after generation"
assert (S0[:, :, 2] >= -1e-5).all(), "negative stratum water after generation"
assert np.abs(F0[:, 2] - t0).max() < 1e-2, "height cache != strata thickness sum"
assert (S0[:, 0, 0].round() == 0).all() and S0[:, 0, 1].min() > 0.5, \
    "column floor is not a bedrock stratum"
nslots = (S0[:, :, 1] > 1e-4).sum(axis=1)
print(f"terrain gen OK  h min={t0.min():.2f} max={t0.max():.2f} mean={t0.mean():.2f} "
      f"slots/col mean={nslots.mean():.1f}")
vols = {m: mat_volume(S0, m) for m in range(8)}
print("  material volumes:", {m: round(v) for m, v in vols.items()})
assert t0.max() > 8, "terrain suspiciously flat"
assert vols[1] > 100 and vols[2] > 100 and vols[3] > 100, "clay/silt/sand beds missing"
assert vols[4] > 100, "no gravel lens generated (the confined aquifer)"
assert vols[5] > 50, "no loam soil mantle"
assert vols[6] > 50, "no regolith"
assert vols[7] > 10, "no boulder patches generated"
assert nslots.mean() > 4, "columns are not actually stratified"
gw0_total = S0[:, :, 2].sum()
assert gw0_total > 1000, "aquifer starts empty"
assert F0[:, 3].max() > 0.5 and (F0[:, 3] > 0.12).sum() > 200, "no forests generated"
print(f"  groundwater initial={gw0_total:.0f}  forest cells={(F0[:,3]>0.12).sum()}")

# --- test 2: pick raymarch ------------------------------------------------------------
dispatch(["pickCast"])
pk = read_f32(pick, 4)
print(f"pick: hit={pk[2]:.0f} at ({pk[0]:.1f},{pk[1]:.1f}) t={pk[3]:.1f}")
assert pk[2] == 1.0, "pick ray straight down at map center should hit"
assert abs(pk[0] - N/2) < 2 and abs(pk[1] - N/2) < 2

# --- test 3: pour + simulate frames -----------------------------------------------------
sim_time = 0.0
for frame in range(240):
    pour = 1.0 if frame < 150 else 0.0
    write_simU(DT, sim_time, 3.0, 4.0, pour, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=0)
    sim_frame()
    sim_time += DT * SUBSTEPS

S1 = read_strata()
F1 = read_fields()
v1 = read_f32(vel, cells * 2)
t1 = heights(S1)
w = F1[:, 0]
assert np.isfinite(F1).all(), "fields non-finite"
assert np.isfinite(S1).all(), "strata non-finite"
assert np.isfinite(v1).all(), "velocity non-finite"
assert (w >= 0).all(), "negative water depth"
assert (S1[:, :, 1] >= -1e-3).all(), "negative stratum thickness after simulation"
assert (S1[:, :, 2] >= -1e-5).all(), "negative stratum water"
assert np.abs(F1[:, 2] - t1).max() < 1e-2, "height cache drifted from strata"
print(f"after 240 frames: water max={w.max():.3f} total={w.sum():.1f} cells wet={(w>1e-3).sum()}")
print(f"  terrain delta: min={(t1-t0).min():.3f} max={(t1-t0).max():.3f}  |vel|max={np.abs(v1).max():.2f}  sed max={F1[:,1].max():.4f}")
print(f"  groundwater: {gw0_total:.0f} -> {S1[:,:,2].sum():.0f}")
assert w.max() > 0.01, "no water accumulated from pouring"
assert (t1 - t0).min() < -0.01, "no erosion happened"
assert np.abs(t1 - t0).max() < 15, "terrain exploded"
assert abs(S1[:, :, 2].sum() - gw0_total) > 1.0, "groundwater never moved (no infiltration/flow)"

# --- test 4: editing tools ---------------------------------------------------------------
# add gravel at the pick point
write_simU(DT, sim_time, 8.0, 5.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True,
           tool=1, material=4)
ci = (N // 2) * N + N // 2
Sb = read_strata()
g_before = Sb[ci][Sb[ci, :, 0].round() == 4][:, 1].sum()
h_add0 = heights(Sb)[ci]
for _ in range(10):
    dispatch(["pickCast", "toolApply"])
Sa = read_strata()
g_after = Sa[ci][Sa[ci, :, 0].round() == 4][:, 1].sum()
_, top_m, _ = col_top(Sa, ci)
assert g_after > g_before + 0.1, f"add-material tool did nothing ({g_before:.3f} -> {g_after:.3f})"
assert top_m == 4, f"added gravel is not the top stratum (top mat {top_m})"
assert read_fields()[ci, 2] > h_add0 + 0.1, "height cache not raised by add tool"
print(f"tool add-gravel OK: {g_before:.2f} -> {g_after:.2f}, now the top stratum")

# replace a band 1..4 below the surface with clay; column height must not change
Sa = read_strata()
h_before = heights(Sa)[ci]
clay_before = Sa[ci][Sa[ci, :, 0].round() == 1][:, 1].sum()
write_simU(DT, sim_time, 8.0, 5.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True,
           tool=3, material=1, depth=1.0, band=3.0)
for _ in range(3):
    dispatch(["pickCast", "toolApply"])
Sb2 = read_strata()
clay_after = Sb2[ci][Sb2[ci, :, 0].round() == 1][:, 1].sum()
h_after = heights(Sb2)[ci]
assert clay_after > clay_before + 1.5, f"replace tool did not add clay ({clay_before:.2f} -> {clay_after:.2f})"
assert abs(h_after - h_before) < 1e-3, f"replace tool changed the column height by {h_after-h_before:.4f}"
# the clay band must sit BELOW the surface, not on top
_, top_m2, top_th2 = col_top(Sb2, ci)
assert not (top_m2 == 1 and top_th2 > 2.5), "replace dumped the clay on top instead of in the band"
print(f"tool replace-with-clay OK: clay {clay_before:.2f} -> {clay_after:.2f}, height drift {abs(h_after-h_before):.2e}")

# dig lowers the column
write_simU(DT, sim_time, 8.0, 5.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=2)
for _ in range(10):
    dispatch(["pickCast", "toolApply"])
h_dug = heights(read_strata())[ci]
assert h_dug < h_before - 0.2, "dig tool did not lower the terrain"
print(f"tool dig OK: height {h_before:.2f} -> {h_dug:.2f}")

# trees: clear-cut first so the plant assertion has a known ~0 baseline (the
# seed may forest the center cell naturally), then plant, then remove again
write_simU(DT, sim_time, 8.0, 6.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["genTerrain"])
write_simU(DT, sim_time, 8.0, 6.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=5)
for _ in range(20):
    dispatch(["pickCast", "toolApply"])
tr_base = read_fields()[ci, 3]
assert tr_base < 0.05, f"clear-cut baseline failed (density {tr_base:.3f})"
write_simU(DT, sim_time, 8.0, 6.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=4)
for _ in range(10):
    dispatch(["pickCast", "toolApply"])
tr_planted = read_fields()[ci, 3]
write_simU(DT, sim_time, 8.0, 6.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=5)
for _ in range(20):
    dispatch(["pickCast", "toolApply"])
tr_removed = read_fields()[ci, 3]
assert tr_planted > max(tr_base + 0.25, 0.3), \
    f"plant tool grew nothing ({tr_base:.3f} -> {tr_planted:.3f})"
assert tr_removed < 0.05, f"remove tool left trees (density {tr_removed:.3f})"
print(f"tool trees OK: baseline {tr_base:.3f}, planted {tr_planted:.2f}, removed to {tr_removed:.3f}")

# --- test 5: stratified groundwater physics ---------------------------------------------

def write_synth(stack, surf_water=0.0):
    """Fill the whole map with one synthetic column. stack = [(mat, th), ...] bottom-up."""
    S = np.zeros((cells, NK, 4), dtype=np.float32)
    for k, (m, th) in enumerate(stack):
        S[:, k, 0] = m
        S[:, k, 1] = th
    F = np.zeros((cells, 4), dtype=np.float32)
    F[:, 0] = surf_water
    F[:, 2] = BASE + sum(th for _, th in stack)
    device.queue.write_buffer(strata, 0, S.tobytes())
    device.queue.write_buffer(fields, 0, F.tobytes())
    device.queue.write_buffer(flux, 0, b"\0" * (cells * 16))
    device.queue.write_buffer(vel, 0, b"\0" * (cells * 8))
    return S, F

SY = {0: 0.02, 1: 0.03, 2: 0.12, 3: 0.25, 4: 0.28, 5: 0.18, 6: 0.06, 7: 0.30}

# (a) confined aquifer: a thin gravel bed sandwiched between sealed rock.
# Overcharge the gravel in a center disc; water must travel ALONG the gravel
# and never enter the rock above or below. Runs the FULL hydrology pair
# (infiltrate = the vertical path that could pump through rock + gwLateral),
# so a percolation regression cannot hide.
write_simU(DT, 0.0, 0.0, 4.0, 0.0, 1.0, (N/2, 120, N/2), (0, -1, 0), False, walls=1.0)
S, F = write_synth([(0, 10.0), (4, 1.0), (0, 12.0)])
cap_g = 1.0 * SY[4]
yy, xx = np.mgrid[0:N, 0:N]
disc = ((xx - N//2) ** 2 + (yy - N//2) ** 2 <= 9).reshape(-1)
S[disc, 1, 2] = cap_g + 0.5            # full + max overpressure
device.queue.write_buffer(strata, 0, S.tobytes())
for _ in range(60):
    dispatch(["infiltrate", "gwLateral"])
    copy(scratch, strata, cells * NK * 16)
S2 = read_strata()
F2 = read_fields()
wet0 = disc.sum()
wet1 = (S2[:, 1, 2] > 0.02).sum()
not_gravel = [0] + list(range(2, NK))
assert S2[:, not_gravel, 2].max() < 1e-6, "water leaked into sealed rock strata"
assert F2[:, 0].max() < 1e-6, "confined water escaped to the surface through rock"
assert wet1 > wet0 * 3, f"confined gravel aquifer did not transmit ({wet0} -> {wet1} wet cells)"
assert abs(S2[:, 1, 2].sum() - S[:, 1, 2].sum()) < 0.5, \
    f"lateral exchange lost mass ({S[:,1,2].sum():.2f} -> {S2[:,1,2].sum():.2f})"
print(f"confined aquifer OK: {wet0} -> {wet1} wet gravel cells, rock dry, mass drift "
      f"{S2[:,1,2].sum()-S[:,1,2].sum():+.4f}")

# (a2) misaligned stacks: gravel beds at DIFFERENT elevations on the two map
# halves. The two-pointer depth-overlap sweep must move water across the seam
# exactly when the intervals overlap — and not at all when they don't.
def synth_split(right_below_gravel, charge_left):
    S = np.zeros((cells, NK, 4), dtype=np.float32)
    left = (np.arange(cells) % N) < N // 2
    S[left, 0, 0], S[left, 0, 1] = 0, 10.0           # gravel spans z[-16,-14]
    S[left, 1, 0], S[left, 1, 1] = 4, 2.0
    S[left, 2, 0], S[left, 2, 1] = 0, 11.0
    S[~left, 0, 0], S[~left, 0, 1] = 0, right_below_gravel
    S[~left, 1, 0], S[~left, 1, 1] = 4, 2.0
    S[~left, 2, 0], S[~left, 2, 1] = 0, 21.0 - right_below_gravel
    S[left, 1, 2] = charge_left
    F = np.zeros((cells, 4), dtype=np.float32)
    F[:, 2] = BASE + 23.0
    device.queue.write_buffer(strata, 0, S.tobytes())
    device.queue.write_buffer(fields, 0, F.tobytes())
    device.queue.write_buffer(flux, 0, b"\0" * (cells * 16))
    device.queue.write_buffer(vel, 0, b"\0" * (cells * 8))
    return S, left

charge = 2.0 * SY[4] + 0.4
for below, label, expects_flow in ((11.0, "1 m overlap", True), (14.0, "no overlap", False)):
    S, left = synth_split(below, charge)
    total0 = S[:, :, 2].astype(np.float64).sum()
    for _ in range(60):
        dispatch(["infiltrate", "gwLateral"])
        copy(scratch, strata, cells * NK * 16)
    Sx = read_strata()
    crossed = Sx[~left, 1, 2].sum()
    assert Sx[:, [0, 2], 2].max() < 1e-6, f"misaligned ({label}): rock got wet"
    assert abs(Sx[:, :, 2].astype(np.float64).sum() - total0) < 0.05, \
        f"misaligned ({label}): sweep lost mass"
    if expects_flow:
        assert crossed > 1.0, f"misaligned ({label}): no water crossed the seam ({crossed:.3f})"
    else:
        assert crossed < 1e-6, f"misaligned ({label}): water crossed without overlap ({crossed:.4f})"
    print(f"misaligned stacks OK ({label}): crossed={crossed:.3f}")

# (b) perched water table: sand over a clay aquitard. Surface water infiltrates
# the sand; the clay fills only to its tiny capacity and the bedrock below
# stays dry — the water table perches on the clay.
write_simU(DT, 0.0, 0.0, 4.0, 0.0, 1.0, (N/2, 120, N/2), (0, -1, 0), False, walls=1.0)
write_synth([(0, 16.0), (1, 2.0), (3, 4.0)], surf_water=1.0)
for _ in range(400):
    dispatch(["infiltrate"])
S3 = read_strata()
F3 = read_fields()
sand_w = S3[:, 2, 2].mean()
clay_w = S3[:, 1, 2].mean()
rock_w = S3[:, 0, 2].mean()
clay_cap = 2.0 * SY[1]
assert sand_w > 0.25, f"sand never charged from the surface ({sand_w:.3f})"
assert clay_w <= clay_cap + 1e-3, f"clay holds more than its capacity ({clay_w:.3f} > {clay_cap:.3f})"
assert rock_w < 1e-6, f"water percolated into sealed bedrock ({rock_w:.2e})"
# per-cell closure: infiltration MOVES water (1.0 was poured per cell; no
# evaporation runs in this loop, so surface + column must still equal 1.0)
closure = F3[:, 0] + S3[:, :, 2].sum(axis=1)
assert np.abs(closure - 1.0).max() < 1e-3, \
    f"infiltration does not conserve water (per-cell total off by {np.abs(closure-1.0).max():.4f})"
print(f"perched table OK: sand {sand_w:.3f}, clay {clay_w:.3f} (cap {clay_cap:.2f}), rock {rock_w:.1e}, closure exact")

# (c) spring overflow: an overcharged top stratum wells out to the surface
S4, F4 = write_synth([(0, 16.0), (3, 4.0)])
cap_s = 4.0 * SY[3]
S4[ci, 1, 2] = cap_s + 5.0
device.queue.write_buffer(strata, 0, S4.tobytes())
dispatch(["infiltrate"])
F5 = read_fields()
S5 = read_strata()
assert F5[ci, 0] > 4.0, f"saturated column did not overflow to surface (got {F5[ci,0]:.2f})"
assert S5[ci, 1, 2] < cap_s + 0.6, f"groundwater left far above capacity ({S5[ci,1,2]:.2f})"
total5 = F5[ci, 0] + S5[ci, :, 2].sum()
assert abs(total5 - (cap_s + 5.0)) < 1e-3, \
    f"spring overflow does not conserve water ({cap_s+5.0:.3f} -> {total5:.3f})"
print(f"spring overflow OK: gw {cap_s+5.0:.2f} -> {S5[ci,1,2]:.2f}, surface +{F5[ci,0]:.2f}, closure exact")

# (d) artesian contrast: identical overpressure under a clay cap vs a sand cap.
# The permeable roof bleeds pressure upward; the aquitard seals it.
S6, F6 = write_synth([(0, 10.0), (4, 2.0), (1, 3.0), (3, 3.0)])
cj = ci + 8
cap_g2 = 2.0 * SY[4]
S6[ci, 1, 2] = cap_g2 + 0.4            # under clay: sealed
S6[cj, 1, 2] = cap_g2 + 0.4
S6[cj, 2, 0] = 3                        # swap the cap at cj: clay -> sand window
device.queue.write_buffer(strata, 0, S6.tobytes())
for _ in range(30):
    dispatch(["infiltrate"])
S7 = read_strata()
sealed_ex = S7[ci, 1, 2] - cap_g2
window_ex = S7[cj, 1, 2] - cap_g2
assert sealed_ex > 0.35, f"clay cap leaked confined overpressure ({sealed_ex:.3f} left)"
assert window_ex < sealed_ex * 0.5, \
    f"permeable window did not vent overpressure (sealed {sealed_ex:.3f} vs window {window_ex:.3f})"
print(f"artesian seal OK: overpressure under clay {sealed_ex:.3f}, under sand window {window_ex:.3f}")

# --- test 5b: persistent springs (shift+click) ----------------------------------------------
write_simU(DT, 0.0, 4.0, 5.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["genTerrain"])
dispatch(["pickCast", "placeSource"])
src = read_sources()
active = src[src[:, 3] > 0]
assert len(active) == 1, f"expected 1 active spring, got {len(active)}"
assert abs(active[0, 0] - N/2) < 3 and abs(active[0, 1] - N/2) < 3, "spring placed off-target"
st = 0.0
for f in range(60):                      # no tool active: only the spring pours
    write_simU(DT, st, 4.0, 5.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), False)
    sim_frame()
    st += DT * SUBSTEPS
w_spring = read_fields()[:, 0].sum()
assert w_spring > 5.0, f"spring produced no water (total {w_spring:.2f})"
assert gpu_mass()[4] == 1.0, "GPU spring count != 1 while a spring is active"
write_simU(DT, st, 4.0, 5.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["pickCast", "placeSource"])    # second shift+click toggles it off
src2 = read_sources()
assert (src2[:, 3] <= 0).all(), "second shift+click did not remove the spring"
dispatch(["pickCast", "placeSource"])    # place again, then regen must clear it
assert read_sources()[:, 3].max() > 0, "re-placing the spring failed"
dispatch(["genTerrain"])
assert (read_sources()[:, 3] <= 0).all(), "genTerrain did not clear active springs"
print(f"springs OK: placed at ({active[0,0]:.0f},{active[0,1]:.0f}), poured {w_spring:.1f}, removed, cleared on regen")

# dry settling: suspended sediment stranded in a dry cell must return to the ground
write_simU(DT, 0.0, 4.0, 5.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["genTerrain"])
Fdry = read_fields()
Sdry = read_strata()
h_dry0 = heights(Sdry)[ci]
Fdry[ci] = [0.0, 0.5, Fdry[ci, 2], Fdry[ci, 3]]      # sediment with no water
device.queue.write_buffer(fields, 0, Fdry.astype(np.float32).tobytes())
dispatch(["erosion"])
F_set = read_fields()
S_set = read_strata()
assert F_set[ci, 1] < 1e-6, f"dry sediment did not settle (sed {F_set[ci,1]:.4f})"
assert heights(S_set)[ci] > h_dry0 + 0.49, "settled sediment did not raise the column"
_, set_m, _ = col_top(S_set, ci)
assert set_m == 3, f"settled sediment is not sand on top (mat {set_m})"
print(f"dry settling OK: 0.5 suspended -> column {h_dry0:.2f} -> {heights(S_set)[ci]:.2f}, top=sand")

# --- test 5c: mass budget (GPU reduction + terrain+sediment conservation) --------------------
write_simU(DT, 0.0, 3.0, 4.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["genTerrain"])
m0 = gpu_mass()
Sm = read_strata()
Fm = read_fields()
terr_np = Sm[:, :, 1].sum()
gw_np = Sm[:, :, 2].sum()
# the GPU f32 tree-reduce of a ~2M total carries round-off — this bound is
# only for the cross-check; conservation below is asserted in float64
assert abs(m0[3] - terr_np) < max(2e-4 * terr_np, 1.0), "GPU terrain total != numpy"
assert abs(m0[2] - gw_np) < max(2e-4 * gw_np, 1.0), "GPU groundwater total != numpy"
def ts64():
    """terrain + suspended sediment, summed in float64 (immune to f32 reduce error)."""
    return (read_strata()[:, :, 1].astype(np.float64).sum()
            + read_fields()[:, 1].astype(np.float64).sum())
def water64():
    return (read_strata()[:, :, 2].astype(np.float64).sum()
            + read_fields()[:, 0].astype(np.float64).sum())
terr_sed0 = ts64()
st = 0.0
for f in range(80):                      # pour inside solid walls so nothing leaves
    write_simU(DT, st, 5.0, 5.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=0, walls=1.0)
    sim_frame()
    st += DT * SUBSTEPS
w_mid = water64()
for f in range(40):                      # settle: no input — water must only shrink
    write_simU(DT, st, 5.0, 5.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=0, walls=1.0)
    sim_frame()
    st += DT * SUBSTEPS
w_end = water64()
drift = ts64() - terr_sed0
print(f"mass budget: terrain+sediment drift {drift:+.4f}; "
      f"water settle {w_mid:.2f} -> {w_end:.2f}")
assert abs(drift) < 5.0, f"terrain+sediment not conserved (drift {drift:+.4f})"
# with walls on and no sources, every water path is move-or-sink: any growth
# means a pass is minting water
assert w_end <= w_mid + 0.5, f"water total grew with no input ({w_mid:.2f} -> {w_end:.2f})"

# --- test 5d: generation options --------------------------------------------------------------
write_simU(DT, 0.0, 3.0, 4.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, relief=0.5)
dispatch(["genTerrain"])
h_low = heights(read_strata())
assert h_low.max() < t0.max() * 0.65, f"relief=0.5 had no effect ({h_low.max():.1f} vs {t0.max():.1f})"
write_simU(DT, 0.0, 3.0, 4.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, island=0.6)
dispatch(["genTerrain"])
land_small = (heights(read_strata()) > 1.0).sum()
land_full = (t0 > 1.0).sum()
assert land_small < land_full * 0.8, f"island=0.6 had no effect ({land_small} vs {land_full} land cells)"
write_simU(DT, 0.0, 3.0, 4.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, soil=0.3)
dispatch(["genTerrain"])
loam_low = mat_volume(read_strata(), 5)
loam_full = vols[5]
assert loam_low < loam_full * 0.55, f"soil=0.3 had no effect ({loam_low:.0f} vs {loam_full:.0f})"
write_simU(DT, 0.0, 3.0, 4.0, 0.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, gdepth=14.0)
dispatch(["genTerrain"])
Sdeep = read_strata()
sed_deep = Sdeep[:, :, 1].sum() - mat_volume(Sdeep, 0)
sed_full = S0[:, :, 1].sum() - vols[0]
assert sed_deep > sed_full * 1.4, f"gdepth=14 did not deepen the sediments ({sed_full:.0f} -> {sed_deep:.0f})"
print(f"gen options OK: relief {t0.max():.1f}->{h_low.max():.1f}, land {land_full}->{land_small}, "
      f"loam {loam_full:.0f}->{loam_low:.0f}, sediments {sed_full:.0f}->{sed_deep:.0f}")

# --- test 6: solid walls keep water on the map ---------------------------------------------
# On a bare-rock world (no pore storage, no sea recharge) the only sinks are
# evaporation (now slow) and the open boundary: pour next to the edge and the
# difference between wall modes is pure edge outflow.
edge_ro = (6.0, 120, N / 2)
totals = {}
for walls in (0.0, 1.0):
    write_simU(DT, 0.0, 6.0, 4.0, 0.0, 42.7, edge_ro, (0, -1, 0), True, tool=0, walls=walls)
    write_synth([(0, 20.0)])
    st = 0.0
    for frame in range(120):
        pour = 1.0 if frame < 60 else 0.0
        write_simU(DT, st, 6.0, 4.0, pour, 42.7, edge_ro, (0, -1, 0), True, tool=0, walls=walls)
        sim_frame()
        st += DT * SUBSTEPS
    totals[walls] = read_fields()[:, 0].sum()
print(f"walls test: open-edge water={totals[0.0]:.1f}, solid-edge water={totals[1.0]:.1f}")
assert totals[1.0] > totals[0.0] * 1.5, "solid walls did not retain more water than open edges"
assert totals[1.0] > 200.0, "solid walls retained almost nothing (absolute floor)"

# --- test 7: render a frame ---------------------------------------------------------------
write_simU(DT, 0.0, 3.0, 4.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=0)
dispatch(["genTerrain"])
for frame in range(120):
    sim_frame()

W, H = 512, 320
msaa = device.create_texture(size=(W, H, 1), sample_count=SAMPLES, format=FMT,
                             usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
resolve = device.create_texture(size=(W, H, 1), format=FMT,
                                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
depth = device.create_texture(size=(W, H, 1), sample_count=SAMPLES,
                              format=wgpu.TextureFormat.depth24plus,
                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

import numpy.linalg as la

def render_frame(yaw, pitch, dist, misc, misc2):
    target = np.array([N/2, 6, N/2])
    eye = target + dist * np.array([math.cos(pitch)*math.cos(yaw), math.sin(pitch), math.cos(pitch)*math.sin(yaw)])
    fwd = (target - eye); fwd /= la.norm(fwd)
    right = np.array([-fwd[2], 0, fwd[0]]); right /= la.norm(right)
    up = np.cross(right, fwd)
    z = -fwd
    x_ = np.cross(np.array([0, 1, 0.0]), z); x_ /= la.norm(x_)
    y_ = np.cross(z, x_)
    V = np.array([
        [x_[0], y_[0], z[0], 0],
        [x_[1], y_[1], z[1], 0],
        [x_[2], y_[2], z[2], 0],
        [-x_.dot(eye), -y_.dot(eye), -z.dot(eye), 1]], dtype=np.float32)
    fov = 45*math.pi/180; aspect = W/H; near, far = 0.5, 3000
    f = 1/math.tan(fov/2)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f/aspect; P[1, 1] = f; P[2, 2] = far/(near-far); P[2, 3] = -1; P[3, 2] = near*far/(near-far)
    # column-major multiply mirroring JS mat4Mul(proj, view)
    vp = np.zeros(16, dtype=np.float32)
    a, b = P.flatten(), V.flatten()
    for c in range(4):
        for r in range(4):
            vp[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3]
    ren = np.zeros(44, dtype=np.float32)
    ren[0:16] = vp
    ren[16:19] = eye
    ren[20:23] = right
    ren[24:27] = up
    ren[28:31] = fwd; ren[31] = math.tan(fov/2)
    sun = np.array([0.55, 0.62, 0.32]); sun /= la.norm(sun)
    ren[32:35] = sun
    ren[36:40] = misc
    ren[40:44] = misc2
    device.queue.write_buffer(renU, 0, ren.tobytes())

    enc = device.create_command_encoder()
    rp = enc.begin_render_pass(
        color_attachments=[{"view": msaa.create_view(), "resolve_target": resolve.create_view(),
                            "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.discard,
                            "clear_value": (0, 0, 0, 1)}],
        depth_stencil_attachment={"view": depth.create_view(), "depth_load_op": wgpu.LoadOp.clear,
                                  "depth_store_op": wgpu.StoreOp.discard, "depth_clear_value": 1.0},
    )
    rp.set_bind_group(0, ren_bg)
    if misc2[1] == 3.0:                  # X-ray view: single fullscreen raymarch
        rp.set_pipeline(xray_pipe); rp.draw(3)
    else:
        rp.set_pipeline(sky_pipe); rp.draw(3)
        rp.set_index_buffer(index_buf, wgpu.IndexFormat.uint32)
        rp.set_pipeline(terr_pipe); rp.draw_indexed(len(idx))
        rp.set_pipeline(tree_pipe); rp.draw(24, cells)
        if misc2[0] > 0.5:
            rp.set_pipeline(cut_pipe); rp.draw(6 * (N - 1))
        rp.set_pipeline(water_pipe); rp.draw_indexed(len(idx))
    rp.end()
    device.queue.submit([enc.finish()])
    img = np.frombuffer(device.queue.read_texture(
        {"texture": resolve, "origin": (0, 0, 0), "mip_level": 0},
        {"bytes_per_row": W*4, "rows_per_image": H},
        (W, H, 1)), dtype=np.uint8).reshape(H, W, 4)
    return img

def save_ppm(img, name):
    with open(os.path.join(OUT, name), "wb") as fh:
        fh.write(f"P6 {W} {H} 255\n".encode())
        fh.write(img[..., [2, 1, 0]].tobytes())   # bgra -> rgb

img = render_frame(0.9, 0.62, 290.0, [1.0, 1.0, 4.0, W/H], [0.0, 0.0, 0.0, 0.0])
uniq = len(np.unique(img[..., :3].reshape(-1, 3), axis=0))
print(f"render OK: {W}x{H}, {uniq} unique colors, mean rgb={img[...,:3].mean(axis=(0,1)).round(1)}")
assert uniq > 500, "render output suspiciously uniform"
save_ppm(img, "frame.ppm")

# cutaway + moisture view exercises the new pipelines and modes
img_cut = render_frame(1.45, 0.35, 200.0, [1.0, 0.0, 4.0, W/H], [128.0, 0.0, 0.0, 0.0])
uniq_cut = len(np.unique(img_cut[..., :3].reshape(-1, 3), axis=0))
save_ppm(img_cut, "frame-cutaway.ppm")
img_diff = np.abs(img_cut.astype(int) - render_frame(1.45, 0.35, 200.0, [1.0, 0.0, 4.0, W/H],
                                                     [0.0, 0.0, 0.0, 0.0]).astype(int)).mean()
print(f"cutaway render OK: {uniq_cut} unique colors, diff vs no-cut={img_diff:.2f}")
assert uniq_cut > 500, "cutaway render suspiciously uniform"
assert img_diff > 0.5, "cutaway plane changed nothing on screen"

img_moist = render_frame(0.9, 0.62, 290.0, [1.0, 0.0, 4.0, W/H], [0.0, 1.0, 0.0, 0.0])
save_ppm(img_moist, "frame-moisture.ppm")
img_layers = render_frame(0.9, 0.62, 290.0, [1.0, 0.0, 4.0, W/H], [0.0, 2.0, 0.0, 0.0])
save_ppm(img_layers, "frame-layers.ppm")
mdiff = np.abs(img_moist.astype(int) - img.astype(int)).mean()
ldiff = np.abs(img_layers.astype(int) - img.astype(int)).mean()
uniq_l = len(np.unique(img_layers[..., :3].reshape(-1, 3), axis=0))
print(f"view modes OK: moisture diff={mdiff:.2f}, strata diff={ldiff:.2f} ({uniq_l} colors)")
assert mdiff > 0.5, "moisture view identical to normal view"
assert ldiff > 0.5, "strata view identical to normal view"
assert uniq_l > 300, "strata view suspiciously uniform"

# volumetric groundwater X-ray
img_xray = render_frame(0.9, 0.62, 290.0, [1.0, 0.0, 4.0, W/H], [0.0, 3.0, 0.0, 0.0])
save_ppm(img_xray, "frame-xray.ppm")
uniq_x = len(np.unique(img_xray[..., :3].reshape(-1, 3), axis=0))
xdiff = np.abs(img_xray.astype(int) - img.astype(int)).mean()
print(f"x-ray render OK: {uniq_x} unique colors, diff vs normal={xdiff:.2f}")
assert uniq_x > 300, "x-ray render suspiciously uniform"
assert xdiff > 2.0, "x-ray view identical to normal view"
# the aquifer should read blue: more blue than red on average inside the volume
bgr = img_xray[..., :3].astype(float)
assert bgr[..., 0].mean() > bgr[..., 2].mean(), "x-ray volume not blue-tinted (BGR check)"

# --- test 8: close-up stress run + frame -----------------------------------------
sim_time = 0.0
write_simU(DT, sim_time, 5.0, 5.0, 1.0, 42.7, (N/2 - 25, 120, N/2 + 18), (0, -1, 0), True, tool=0)
dispatch(["genTerrain"])
for frame in range(300):
    sim_frame()
img2 = render_frame(2.3, 0.42, 130.0, [8.0, 1.0, 5.0, W/H], [0.0, 0.0, 0.0, 0.0])
save_ppm(img2, "frame2.ppm")
uniq2 = len(np.unique(img2[..., :3].reshape(-1, 3), axis=0))
assert uniq2 > 500, "close-up render suspiciously uniform"
print(f"close-up frame written ({uniq2} unique colors)")

# --- diagnostics + stress assertions on the close-up run ---------------------------
Sd = read_strata()
Fd = read_fields()
assert np.isfinite(Sd).all() and np.isfinite(Fd).all(), "non-finite state after stress run"
assert (Sd[:, :, 1] >= -1e-3).all(), "negative stratum thickness after stress run"
assert (Sd[:, :, 2] >= -1e-5).all(), "negative stratum water after stress run"
assert (Fd[:, :2] >= -1e-5).all() and (Fd[:, 3] >= -1e-5).all(), \
    "negative water/sediment/trees after stress run"
# orig was rewritten by the genTerrain dispatch that started this stress run
d_stress = heights(Sd) - read_f32(orig, cells)
assert d_stress.max() > 0.05, "no deposition anywhere after stress run"
assert np.abs(d_stress).max() < 15, "terrain exploded during stress run"
t_now = heights(Sd).reshape(N, N)
w_now = Fd[:, 0].reshape(N, N)
s_now = Fd[:, 1].reshape(N, N)
g_now = Sd[:, :, 2].sum(axis=1).reshape(N, N)
tr_now = Fd[:, 3].reshape(N, N)
o_now = read_f32(orig, cells).reshape(N, N)
delta = t_now - o_now

def save_gray(arr, path, lo=None, hi=None):
    lo = arr.min() if lo is None else lo
    hi = arr.max() if hi is None else hi
    g = np.clip((arr - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    with open(path, "wb") as fh:
        fh.write(f"P6 {N} {N} 255\n".encode())
        fh.write(np.repeat(g[..., None], 3, axis=2).tobytes())

save_gray(t_now, os.path.join(OUT, "diag_terrain.ppm"))
save_gray(w_now, os.path.join(OUT, "diag_water.ppm"), 0, 1.0)
save_gray(delta, os.path.join(OUT, "diag_delta.ppm"), -2, 2)
save_gray(s_now, os.path.join(OUT, "diag_sed.ppm"), 0, 0.5)
save_gray(g_now, os.path.join(OUT, "diag_gw.ppm"), 0, 4.0)
save_gray(tr_now, os.path.join(OUT, "diag_trees.ppm"), 0, 1.0)

# high-frequency energy: mean |laplacian| of terrain — checkerboard detector
lap = np.abs(4*t_now[1:-1,1:-1] - t_now[:-2,1:-1] - t_now[2:,1:-1] - t_now[1:-1,:-2] - t_now[1:-1,2:])
print(f"diag: water max={w_now.max():.3f} wet={np.sum(w_now>0.0012)} sed max={s_now.max():.3f}")
print(f"diag: gw total={g_now.sum():.0f} max={g_now.max():.3f}  trees>0.12={(tr_now>0.12).sum()}")
print(f"diag: delta min={delta.min():.3f} max={delta.max():.3f}")
print(f"diag: terrain |lap| mean={lap.mean():.4f} p99={np.percentile(lap,99):.3f} max={lap.max():.3f}")
# calibrated against healthy runs (mean ~0.14, p99 ~1.7); checkerboarding
# regressions historically push these up by an order of magnitude
assert lap.mean() < 0.6, f"terrain high-frequency energy too high (mean |lap| {lap.mean():.3f})"
assert np.percentile(lap, 99) < 6.0, f"terrain checkerboarding (p99 |lap| {np.percentile(lap,99):.3f})"
wet = w_now > 0.0012
if wet.any():
    print(f"diag: wet-region depth mean={w_now[wet].mean():.4f} p90={np.percentile(w_now[wet],90):.3f}")
print("ALL CHECKS PASSED")
