"""Diagnostic: does the semi-Lagrangian sediment gather alone create mass?

Reuses the validate.py setup (same trick as diag_waterloss.py), runs a pour to
build a realistic suspended-sediment + velocity state, then measures the
sediment total before and after a SINGLE isolated transport dispatch — the
delta is pure advection-resampling error, nothing else runs.
"""
import os
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

N = g["N"]; DT = g["DT"]; cells = N * N; NK = g["NK"]
write_simU = g["write_simU"]; dispatch = g["dispatch"]; sim_frame = g["sim_frame"]
device = g["device"]; read_fields = g["read_fields"]; read_f32 = g["read_f32"]
scratch = g["scratch"]

write_simU(DT, 0, 5.0, 5.0, 1.0, 42.7, (N / 2, 120, N / 2), (0, -1, 0), True, tool=0, walls=1.0)
dispatch(["genTerrain"])
for f in range(60):
    sim_frame()

sed0 = read_fields()[:, 1].sum()
dispatch(["transport"])                  # isolated: only advection + tree decay
sed1 = read_f32(scratch, cells * 4).reshape(cells, 4)[:, 1].sum()
print(f"sediment before transport: {sed0:.3f}")
print(f"sediment after  transport: {sed1:.3f}  (delta {sed1 - sed0:+.4f} in ONE substep)")
print(f"per-frame estimate (x3 substeps): {(sed1 - sed0) * 3:+.4f}")
