# Headless validation harness

Compiles the real WGSL (via naga) and runs the simulation off-browser on
[lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html) (llvmpipe Vulkan), so
shader and sim regressions are caught without a GPU or a browser.

`extract.js` pulls the composed `WGSL_SIM` / `WGSL_RENDER` strings out of
`../main.js` (writing `sim.wgsl` and `render.wgsl`), then `validate.py`:

- compiles both shader modules and builds all 12 pipelines with the exact
  bind-group layouts the app uses,
- runs terrain generation, GPU picking, and a few hundred pour/erosion frames
  with finiteness and behaviour assertions,
- renders an overview and a close-up frame plus terrain/water/sediment/delta
  diagnostic maps into `out/`.

Reference renders committed alongside: `render-overview.png`,
`render-closeup.png`.

## Run

Needs Node (for `extract.js`), a Vulkan driver with lavapipe
(`mesa-vulkan-drivers libvulkan1`), and a Python env with `wgpu` + `numpy`
(`pillow` to convert the `.ppm` output to PNG):

```sh
python validate.py
```

All generated files (`sim.wgsl`, `render.wgsl`, `out/`) are gitignored.
