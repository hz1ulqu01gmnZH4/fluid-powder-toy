"""Headless validation of the Hydra Terra shaders + simulation on lavapipe."""
import json, subprocess, math, struct, sys, os
import numpy as np
import wgpu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

meta = json.loads(subprocess.check_output(["node", os.path.join(HERE, "extract.js")]).decode())
N, SUBSTEPS, DT = meta["N"], meta["SUBSTEPS"], meta["DT"]
sim_src = open(os.path.join(HERE, "sim.wgsl")).read()
ren_src = open(os.path.join(HERE, "render.wgsl")).read()

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
print("adapter:", adapter.info["device"])

cells = N * N
def mkbuf(size):
    return device.create_buffer(
        size=size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

terrain = mkbuf(cells * 4)
water   = mkbuf(cells * 4)
flux    = mkbuf(cells * 16)
vel     = mkbuf(cells * 8)
sed     = mkbuf(cells * 4)
scratch = mkbuf(cells * 4)
pick    = mkbuf(16)
orig    = mkbuf(cells * 4)
simU = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
renU = device.create_buffer(size=176, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

sim_mod = device.create_shader_module(code=sim_src)
ren_mod = device.create_shader_module(code=ren_src)
print("shader modules compiled OK")

sim_bgl = device.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
] + [
    {"binding": b, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}}
    for b in range(1, 9)
])
sim_layout = device.create_pipeline_layout(bind_group_layouts=[sim_bgl])
sim_bg = device.create_bind_group(layout=sim_bgl, entries=[
    {"binding": 0, "resource": {"buffer": simU, "offset": 0, "size": 64}},
    {"binding": 1, "resource": {"buffer": terrain, "offset": 0, "size": cells * 4}},
    {"binding": 2, "resource": {"buffer": water, "offset": 0, "size": cells * 4}},
    {"binding": 3, "resource": {"buffer": flux, "offset": 0, "size": cells * 16}},
    {"binding": 4, "resource": {"buffer": vel, "offset": 0, "size": cells * 8}},
    {"binding": 5, "resource": {"buffer": sed, "offset": 0, "size": cells * 4}},
    {"binding": 6, "resource": {"buffer": scratch, "offset": 0, "size": cells * 4}},
    {"binding": 7, "resource": {"buffer": pick, "offset": 0, "size": 16}},
    {"binding": 8, "resource": {"buffer": orig, "offset": 0, "size": cells * 4}},
])

eps = ["genTerrain", "clearWater", "pickCast", "addWater", "fluxPass", "depthVel", "erosion", "advect", "thermal"]
pipes = {}
for ep in eps:
    pipes[ep] = device.create_compute_pipeline(
        layout=sim_layout, compute={"module": sim_mod, "entry_point": ep})
print("compute pipelines OK:", ", ".join(eps))

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
    {"binding": 1, "resource": {"buffer": terrain, "offset": 0, "size": cells * 4}},
    {"binding": 2, "resource": {"buffer": water, "offset": 0, "size": cells * 4}},
    {"binding": 3, "resource": {"buffer": sed, "offset": 0, "size": cells * 4}},
    {"binding": 4, "resource": {"buffer": vel, "offset": 0, "size": cells * 8}},
    {"binding": 5, "resource": {"buffer": orig, "offset": 0, "size": cells * 4}},
    {"binding": 6, "resource": {"buffer": pick, "offset": 0, "size": 16}},
])

FMT = wgpu.TextureFormat.bgra8unorm
SAMPLES = 4
common_ds = {"format": wgpu.TextureFormat.depth24plus, "depth_write_enabled": True,
             "depth_compare": wgpu.CompareFunction.less,
             "stencil_front": {}, "stencil_back": {}}
sky_pipe = device.create_render_pipeline(
    layout=ren_layout,
    vertex={"module": ren_mod, "entry_point": "vsSky", "buffers": []},
    fragment={"module": ren_mod, "entry_point": "fsSky", "targets": [{"format": FMT}]},
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    multisample={"count": SAMPLES},
    depth_stencil={**common_ds, "depth_write_enabled": False,
                   "depth_compare": wgpu.CompareFunction.always},
)
terr_pipe = device.create_render_pipeline(
    layout=ren_layout,
    vertex={"module": ren_mod, "entry_point": "vsTerrain", "buffers": []},
    fragment={"module": ren_mod, "entry_point": "fsTerrain", "targets": [{"format": FMT}]},
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
    multisample={"count": SAMPLES},
    depth_stencil=common_ds,
)
blend = {"color": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add},
         "alpha": {"src_factor": wgpu.BlendFactor.one, "dst_factor": wgpu.BlendFactor.one_minus_src_alpha, "operation": wgpu.BlendOperation.add}}
water_pipe = device.create_render_pipeline(
    layout=ren_layout,
    vertex={"module": ren_mod, "entry_point": "vsWater", "buffers": []},
    fragment={"module": ren_mod, "entry_point": "fsWater", "targets": [{"format": FMT, "blend": blend}]},
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
    multisample={"count": SAMPLES},
    depth_stencil=common_ds,
)
print("render pipelines OK: sky, terrain, water")

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
def write_simU(dt, time, rate, radius, pour, seed, ro, rd, hover):
    data = struct.pack("16f",
        ro[0], ro[1], ro[2], 0.0,
        rd[0], rd[1], rd[2], 1.0 if hover else 0.0,
        dt, time, rate, radius,
        pour, seed, 0.0, 0.0)
    device.queue.write_buffer(simU, 0, data)

def dispatch(names):
    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    for n in names:
        p.set_pipeline(pipes[n])
        p.set_bind_group(0, sim_bg)
        if n == "pickCast":
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

# --- test 1: terrain generation ------------------------------------------------------
write_simU(DT, 0, 3, 4, 0, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
dispatch(["genTerrain"])
t0 = read_f32(terrain, cells)
assert np.isfinite(t0).all(), "terrain has non-finite values"
print(f"terrain gen OK  min={t0.min():.2f} max={t0.max():.2f} mean={t0.mean():.2f}")
assert t0.max() > 8, "terrain suspiciously flat"

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
    write_simU(DT, sim_time, 3.0, 4.0, pour, 42.7, (N/2, 120, N/2), (0, -1, 0), True)
    dispatch(["pickCast"])
    for s in range(SUBSTEPS):
        dispatch(["addWater", "fluxPass", "depthVel", "erosion", "advect"])
        copy(scratch, sed, cells * 4)
        sim_time += DT
    dispatch(["thermal"])
    copy(scratch, terrain, cells * 4)

w = read_f32(water, cells)
t1 = read_f32(terrain, cells)
s1 = read_f32(sed, cells)
v1 = read_f32(vel, cells * 2)
assert np.isfinite(w).all(), "water non-finite"
assert np.isfinite(t1).all(), "terrain non-finite"
assert np.isfinite(s1).all(), "sediment non-finite"
assert np.isfinite(v1).all(), "velocity non-finite"
assert (w >= 0).all(), "negative water depth"
print(f"after 240 frames: water max={w.max():.3f} total={w.sum():.1f} cells wet={(w>1e-3).sum()}")
print(f"  terrain delta: min={(t1-t0).min():.3f} max={(t1-t0).max():.3f}  |vel|max={np.abs(v1).max():.2f}  sed max={s1.max():.4f}")
assert w.max() > 0.01, "no water accumulated from pouring"
assert (t1 - t0).min() < -0.01, "no erosion happened"
assert np.abs(t1 - t0).max() < 15, "terrain exploded"

# --- test 4: render a frame ---------------------------------------------------------------
W, H = 512, 320
msaa = device.create_texture(size=(W, H, 1), sample_count=SAMPLES, format=FMT,
                             usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
resolve = device.create_texture(size=(W, H, 1), format=FMT,
                                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
depth = device.create_texture(size=(W, H, 1), sample_count=SAMPLES,
                              format=wgpu.TextureFormat.depth24plus,
                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

# camera mirroring main.js
import numpy.linalg as la
yaw, pitch, dist = 0.9, 0.62, 290.0
target = np.array([N/2, 6, N/2])
eye = target + dist * np.array([math.cos(pitch)*math.cos(yaw), math.sin(pitch), math.cos(pitch)*math.sin(yaw)])
fwd = (target - eye); fwd /= la.norm(fwd)
right = np.array([-fwd[2], 0, fwd[0]]); right /= la.norm(right)
up = np.cross(right, fwd)
z = -fwd
x_ = np.cross(np.array([0,1,0.0]), z); x_ /= la.norm(x_)
y_ = np.cross(z, x_)
view = np.identity(4, dtype=np.float32)
view[:3,0], view[:3,1], view[:3,2] = 0,0,0
V = np.array([
    [x_[0], y_[0], z[0], 0],
    [x_[1], y_[1], z[1], 0],
    [x_[2], y_[2], z[2], 0],
    [-x_.dot(eye), -y_.dot(eye), -z.dot(eye), 1]], dtype=np.float32)  # column-major rows
fov = 45*math.pi/180; aspect = W/H; near, far = 0.5, 3000
f = 1/math.tan(fov/2)
P = np.zeros((4,4), dtype=np.float32)
P[0,0]=f/aspect; P[1,1]=f; P[2,2]=far/(near-far); P[2,3]=-1; P[3,2]=near*far/(near-far)
# column-major multiply mirroring JS mat4Mul(proj, view)
def colmul(a, b):
    o = np.zeros(16, dtype=np.float32)
    af, bf = a.flatten(), b.flatten()
    for c in range(4):
        for r in range(4):
            o[c*4+r] = sum(af[k*4+r]*bf[c*4+k] for k in range(4))
    return o
vp = colmul(V.flatten().reshape(4,4), P.flatten().reshape(4,4))  # placeholder
# do it exactly like JS: arrays are already column-major flat lists
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
ren[36:40] = [1.0, 1.0, 4.0, aspect]
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
rp.set_pipeline(sky_pipe); rp.draw(3)
rp.set_index_buffer(index_buf, wgpu.IndexFormat.uint32)
rp.set_pipeline(terr_pipe); rp.draw_indexed(len(idx))
rp.set_pipeline(water_pipe); rp.draw_indexed(len(idx))
rp.end()
device.queue.submit([enc.finish()])

img = np.frombuffer(device.queue.read_texture(
    {"texture": resolve, "origin": (0,0,0), "mip_level": 0},
    {"bytes_per_row": W*4, "rows_per_image": H},
    (W, H, 1)), dtype=np.uint8).reshape(H, W, 4)
uniq = len(np.unique(img[..., :3].reshape(-1, 3), axis=0))
print(f"render OK: {W}x{H}, {uniq} unique colors, mean rgb={img[...,:3].mean(axis=(0,1)).round(1)}")
assert uniq > 500, "render output suspiciously uniform"

# save a PPM preview
with open(os.path.join(OUT, "frame.ppm"), "wb") as fh:
    fh.write(f"P6 {W} {H} 255\n".encode())
    fh.write(img[..., [2,1,0]].tobytes())   # bgra -> rgb
print("ALL CHECKS PASSED")

# --- extra: close-up frame while pouring ----------------------------------------
dispatch(["genTerrain", "clearWater"])  # fresh terrain so we can judge a clean stream
write_simU(DT, sim_time, 5.0, 5.0, 1.0, 42.7, (N/2 - 25, 120, N/2 + 18), (0, -1, 0), True)
for frame in range(300):
    dispatch(["pickCast"])
    for s in range(SUBSTEPS):
        dispatch(["addWater", "fluxPass", "depthVel", "erosion", "advect"])
        copy(scratch, sed, cells * 4)
    dispatch(["thermal"])
    copy(scratch, terrain, cells * 4)

yaw, pitch, dist = 2.3, 0.42, 130.0
eye = target + dist * np.array([math.cos(pitch)*math.cos(yaw), math.sin(pitch), math.cos(pitch)*math.sin(yaw)])
fwd = (target - eye); fwd /= la.norm(fwd)
right = np.array([-fwd[2], 0, fwd[0]]); right /= la.norm(right)
up = np.cross(right, fwd)
z = -fwd
x_ = np.cross(np.array([0,1,0.0]), z); x_ /= la.norm(x_)
y_ = np.cross(z, x_)
V = np.array([
    [x_[0], y_[0], z[0], 0],
    [x_[1], y_[1], z[1], 0],
    [x_[2], y_[2], z[2], 0],
    [-x_.dot(eye), -y_.dot(eye), -z.dot(eye), 1]], dtype=np.float32)
a, b = P.flatten(), V.flatten()
for c in range(4):
    for r in range(4):
        vp[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3]
ren[0:16] = vp
ren[16:19] = eye
ren[20:23] = right
ren[24:27] = up
ren[28:31] = fwd
ren[36:40] = [8.0, 1.0, 5.0, aspect]
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
rp.set_pipeline(sky_pipe); rp.draw(3)
rp.set_index_buffer(index_buf, wgpu.IndexFormat.uint32)
rp.set_pipeline(terr_pipe); rp.draw_indexed(len(idx))
rp.set_pipeline(water_pipe); rp.draw_indexed(len(idx))
rp.end()
device.queue.submit([enc.finish()])
img = np.frombuffer(device.queue.read_texture(
    {"texture": resolve, "origin": (0,0,0), "mip_level": 0},
    {"bytes_per_row": W*4, "rows_per_image": H},
    (W, H, 1)), dtype=np.uint8).reshape(H, W, 4)
with open(os.path.join(OUT, "frame2.ppm"), "wb") as fh:
    fh.write(f"P6 {W} {H} 255\n".encode())
    fh.write(img[..., [2,1,0]].tobytes())
print("close-up frame written")

# --- diagnostics: dump field maps after close-up run ------------------------------
def read_f32(buf):
    return np.frombuffer(device.queue.read_buffer(buf), dtype=np.float32)

t_now = read_f32(terrain)[:cells].reshape(N, N)
w_now = read_f32(water)[:cells].reshape(N, N)
s_now = read_f32(sed)[:cells].reshape(N, N)
o_now = read_f32(orig)[:cells].reshape(N, N)
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

# high-frequency energy: mean |laplacian| of terrain — checkerboard detector
lap = np.abs(4*t_now[1:-1,1:-1] - t_now[:-2,1:-1] - t_now[2:,1:-1] - t_now[1:-1,:-2] - t_now[1:-1,2:])
print(f"diag: water max={w_now.max():.3f} wet={np.sum(w_now>0.0012)} sed max={s_now.max():.3f}")
print(f"diag: delta min={delta.min():.3f} max={delta.max():.3f}")
print(f"diag: terrain |lap| mean={lap.mean():.4f} p99={np.percentile(lap,99):.3f} max={lap.max():.3f}")
wet = w_now > 0.0012
if wet.any():
    print(f"diag: wet-region depth mean={w_now[wet].mean():.4f} p90={np.percentile(w_now[wet],90):.3f}")
