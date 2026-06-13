"""Diagnostic renders: strata cutaway face-on + X-ray, for visual inspection.

Reuses the validate.py setup head (buffers, pipelines) like the other diag
scripts, then renders a handful of camera/view-mode combinations into out/.
"""
import os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "validate.py")) as fh:
    head_lines = fh.read().splitlines()
marks = [k for k, ln in enumerate(head_lines) if ln.startswith("# --- test 1")]
if len(marks) != 1:
    raise RuntimeError("validate.py layout changed: '# --- test 1' marker not found")
head = "\n".join(head_lines[:marks[0]])
g = {"__name__": "diag", "__file__": os.path.join(HERE, "validate.py")}
exec(compile(head, "validate-head", "exec"), g)

import wgpu
N = g["N"]; DT = g["DT"]; cells = g["cells"]
device = g["device"]; renU = g["renU"]; ren_bg = g["ren_bg"]
sky_pipe = g["sky_pipe"]; terr_pipe = g["terr_pipe"]; tree_pipe = g["tree_pipe"]
cut_pipe = g["cut_pipe"]; water_pipe = g["water_pipe"]; xray_pipe = g["xray_pipe"]
index_buf = g["index_buf"]; idx = g["idx"]; FMT = g["FMT"]; SAMPLES = g["SAMPLES"]
write_simU = g["write_simU"]; dispatch = g["dispatch"]; sim_frame = g["sim_frame"]
OUT = g["OUT"]

W, H = 640, 400
msaa = device.create_texture(size=(W, H, 1), sample_count=SAMPLES, format=FMT,
                             usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
resolve = device.create_texture(size=(W, H, 1), format=FMT,
                                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
depth = device.create_texture(size=(W, H, 1), sample_count=SAMPLES,
                              format=wgpu.TextureFormat.depth24plus,
                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

def render(yaw, pitch, dist, misc, misc2, target_y=6.0):
    target = np.array([N / 2, target_y, N / 2])
    eye = target + dist * np.array([math.cos(pitch) * math.cos(yaw), math.sin(pitch),
                                    math.cos(pitch) * math.sin(yaw)])
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.array([-fwd[2], 0, fwd[0]]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    z = -fwd
    x_ = np.cross(np.array([0, 1, 0.0]), z); x_ /= np.linalg.norm(x_)
    y_ = np.cross(z, x_)
    V = np.array([
        [x_[0], y_[0], z[0], 0],
        [x_[1], y_[1], z[1], 0],
        [x_[2], y_[2], z[2], 0],
        [-x_.dot(eye), -y_.dot(eye), -z.dot(eye), 1]], dtype=np.float32)
    fov = 45 * math.pi / 180; aspect = W / H; near, far = 0.5, 3000
    f = 1 / math.tan(fov / 2)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect; P[1, 1] = f; P[2, 2] = far / (near - far); P[2, 3] = -1
    P[3, 2] = near * far / (near - far)
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
    ren[28:31] = fwd; ren[31] = math.tan(fov / 2)
    sun = np.array([0.55, 0.62, 0.32]); sun /= np.linalg.norm(sun)
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
                                  "depth_store_op": wgpu.StoreOp.discard, "depth_clear_value": 1.0})
    rp.set_bind_group(0, ren_bg)
    if misc2[1] == 3.0:
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
        {"bytes_per_row": W * 4, "rows_per_image": H},
        (W, H, 1)), dtype=np.uint8).reshape(H, W, 4)
    return img

def save(img, name):
    with open(os.path.join(OUT, name), "wb") as fh:
        fh.write(f"P6 {W} {H} 255\n".encode())
        fh.write(img[..., [2, 1, 0]].tobytes())

write_simU(DT, 0.0, 3.0, 4.0, 1.0, 42.7, (N/2, 120, N/2), (0, -1, 0), True, tool=0,
           gdepth=10.0)   # the app's default sediment depth
dispatch(["genTerrain"])
for f in range(150):
    sim_frame()

# cutaway viewed face-on from the cut side (negative yaw puts the camera at
# z < cutZ where the terrain is discarded, exposing the curtain)
save(render(-1.35, 0.18, 190.0, [1.0, 0.0, 4.0, W/H], [128.0, 0.0, 0.0, 0.0], target_y=2.0),
     "diag-cutface.ppm")
save(render(0.9, 0.62, 290.0, [1.0, 0.0, 4.0, W/H], [0.0, 3.0, 0.0, 0.0]),
     "diag-xray.ppm")
print("diag renders written: diag-cutface.ppm, diag-xray.ppm")
