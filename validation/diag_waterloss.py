"""Diagnostic: where does surface water go over time?

Reuses the setup portion of validate.py (buffers, pipelines, helpers), then:
  A) normal terrain, solid walls, one big pour -> track water/gw/sed totals
  B) flat pure-bedrock world, solid walls, water column -> does any of it
     enter the ground (through rock)? compare decay against analytic EVAP.
"""
import os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "validate.py")) as fh:
    head_lines = fh.read().splitlines()
marks = [k for k, ln in enumerate(head_lines) if ln.startswith("# --- test 1")]
if len(marks) != 1:
    raise RuntimeError("validate.py layout changed: '# --- test 1' marker not found")
head = "\n".join(head_lines[:marks[0]])  # everything before the first test
g = {"__name__": "diag", "__file__": os.path.join(HERE, "validate.py")}
exec(compile(head, "validate-head", "exec"), g)

N = g["N"]; DT = g["DT"]; SUBSTEPS = g["SUBSTEPS"]; NK = g["NK"]; cells = N * N
write_simU = g["write_simU"]; dispatch = g["dispatch"]; sim_frame = g["sim_frame"]
gpu_mass = g["gpu_mass"]; device = g["device"]
strata = g["strata"]; fields = g["fields"]; flux = g["flux"]; vel = g["vel"]
BASE = g["BASE"]

m_evap = re.search(r"const EVAP\s*:\s*f32\s*=\s*([0-9.]+)", g["sim_src"])
if m_evap is None:
    raise RuntimeError("EVAP constant not found in sim.wgsl")
EVAP = float(m_evap.group(1))
print(f"N={N} DT={DT} SUBSTEPS={SUBSTEPS} EVAP={EVAP}")
SIM_PER_FRAME = DT * SUBSTEPS

def zero(buf, n):
    device.queue.write_buffer(buf, 0, b"\0" * n)

def run(frames, walls=1.0, label=""):
    # tool inactive + rate 0: observe the existing water with no new input
    write_simU(DT, 0, 0, 4, 0, 1.0, (N / 2, 120, N / 2), (0, -1, 0), False, walls=walls)
    log = []
    for f in range(frames + 1):
        if f % 60 == 0:
            w, sed, gw, terr = gpu_mass()[:4]
            log.append((f, w, gw, sed))
            print(f"  {label} frame {f:4d}  water={w:12.2f} gw={gw:12.2f} sed={sed:9.3f}")
        if f < frames:
            sim_frame()
    return log

# --- A: normal terrain, sealed walls, single big pour ------------------------
print("\n[A] normal terrain, walls ON, pour 90 frames then watch 600 frames")
write_simU(DT, 0, 0, 4, 0, 42.7, (N / 2, 120, N / 2), (0, -1, 0), True, walls=1.0)
dispatch(["genTerrain"])
w0 = gpu_mass()
print(f"  after gen: water={w0[0]:.2f} gw={w0[2]:.2f}")
write_simU(DT, 0, 6.0, 10.0, 1.0, 42.7, (N / 2, 120, N / 2), (0, -1, 0), True, tool=0, walls=1.0)
dispatch(["pickCast"])
for f in range(90):
    sim_frame()
runA = run(600, walls=1.0, label="A")

# --- B: flat pure-bedrock world, sealed walls, standing water ----------------
print("\n[B] flat pure-bedrock, walls ON, 2.0 deep water everywhere, 600 frames")
S = np.zeros((cells, NK, 4), dtype=np.float32)
S[:, 0, 1] = 20.0                                                   # one rock stratum
F = np.zeros((cells, 4), dtype=np.float32)
F[:, 0] = 2.0                                                       # water
F[:, 2] = BASE + 20.0                                               # height cache
device.queue.write_buffer(strata, 0, S.tobytes())
device.queue.write_buffer(fields, 0, F.tobytes())
zero(flux, cells * 16); zero(vel, cells * 8)
runB = run(600, walls=1.0, label="B")

w_start = runB[0][1]; w_end = runB[-1][1]; gw_end = runB[-1][2]
t_sim = 600 * SIM_PER_FRAME
# per-substep multiplicative evaporation -> exponent uses (1 - EVAP*DT) per substep
pred = w_start * (1.0 - EVAP * DT) ** (600 * SUBSTEPS)
print(f"\n[B] groundwater in pure rock after 600 frames: {gw_end:.6f} "
      f"(rock now takes a TRACE — M_INFIL 1e-4 into Sy 0.02 pores — by design)")
print(f"[B] water measured {w_end:.2f} vs pure-evaporation prediction {pred:.2f} "
      f"({abs(w_end - pred) / max(pred, 1e-9) * 100:.2f}% off)")
print(f"[B] evaporation half-life: {math.log(2) / (EVAP * SIM_PER_FRAME):.0f} frames "
      f"(~{math.log(2) / (EVAP * SIM_PER_FRAME) / 60:.1f} s at 60 fps)")
