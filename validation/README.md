# Headless validation harness

Compiles the real WGSL (via naga) and runs the simulation off-browser on
[lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html) (llvmpipe Vulkan), so
shader and sim regressions are caught without a GPU or a browser.

`extract.js` pulls the composed `WGSL_SIM` / `WGSL_RENDER` / `WGSL_MASS`
strings out of `../main.js` (writing `sim.wgsl`, `render.wgsl` and
`mass.wgsl`), then `validate.py`:

- compiles all three shader modules and builds all 21 pipelines (14 sim
  compute + 1 mass-reduce, 6 render) with the exact bind-group layouts the
  app uses,
- runs stratigraphic terrain generation (layer-cake columns with a gravel
  lens, regolith, loam mantle, initial aquifer, forests) and asserts the
  columns are genuinely stratified, the height cache matches the strata, and
  every material is present,
- runs GPU picking and a few hundred pour/erosion frames with finiteness and
  behaviour assertions (erosion happened, groundwater moved, nothing
  exploded, no negative thicknesses or water),
- exercises every editing tool: add material (lands on top), dig (releases
  squeezed groundwater), replace-underground (splits strata at the band
  edges, column height preserved), plant and remove trees,
- tests the stratified groundwater model on synthetic columns (running the
  full infiltrate + gwLateral hydrology pair): a confined gravel aquifer
  transmits laterally between sealed rock beds without leaking a drop into
  them, misaligned gravel beds exchange exactly when their depth intervals
  overlap (exercising the two-pointer sweep) and not at all when they don't,
  a clay aquitard perches the water table (with exact per-cell water
  closure), an overcharged top stratum overflows as a spring (closure
  checked), and a clay cap holds artesian overpressure that a permeable
  window vents,
- places and removes a persistent spring (shift+click feature) and asserts
  it pours; checks dry sediment settles back to ground as sand,
- checks the GPU mass reduction against numpy, asserts terrain+sediment
  conservation over a walled pour run in float64 (the flux-form sediment
  advection is exactly conservative, so the bound is ±5 units on a ~2M
  total), and asserts the total water budget never grows once input stops,
- verifies the generation options (relief / island size / soil depth /
  sediment depth) change the generated world,
- checks that solid map edges retain water that open edges drain,
- renders overview, cutaway, moisture-view, strata-view, volumetric X-ray
  and close-up frames plus terrain/water/sediment/groundwater/trees/delta
  diagnostic maps into `out/`.

Reference renders committed alongside: `render-overview.png`,
`render-closeup.png`.

The `diag_*.py` scripts are scratch diagnostics that reuse the harness setup
(water-loss budget, advection conservation).

## Run

Needs Node (for `extract.js`), a Vulkan driver with lavapipe
(`mesa-vulkan-drivers libvulkan1`), and a Python env with `wgpu` + `numpy`
(`pillow` to convert the `.ppm` output to PNG):

```sh
python validate.py
```

All generated files (`sim.wgsl`, `render.wgsl`, `mass.wgsl`, `out/`) are gitignored.
