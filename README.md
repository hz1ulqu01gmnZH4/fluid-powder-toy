# Hydra Terra — WebGPU fluid & erosion sandbox

A 3D "powder toy" style sandbox: procedurally generated island terrain, real-time
shallow-water fluid simulation, and hydraulic + thermal erosion — all running on
the GPU via WebGPU. Click anywhere to pour water and watch it carve channels,
carry sediment, and build deltas.

The underground is a true stratigraphic column per cell: up to 10 strata in
arbitrary vertical order — bedrock, clay, silt, sand, gravel, loam, regolith,
boulders — each with its own erodibility, angle of repose, specific yield and
hydraulic conductivity (~3 decades of contrast, after Freeze & Cherry).
Every stratum stores its own water: rain soaks down the stack, perches on clay,
travels sideways through buried gravel lenses as a confined aquifer, builds
artesian overpressure under a sealed cap, and wells out as springs. Trees bind
the ground against erosion and can be planted or felled with a brush.
Scale: 1 cell ≈ 2 m, 1 height unit ≈ 1 m.

Note: This project was completed as a capability test of Claude Fable 5.

## Run

WebGPU needs a secure context, so serve the folder over localhost:

```sh
cd fluid-powder-toy
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a WebGPU-capable browser
(Chrome/Edge 113+, Safari 18+, Firefox 141+).

## Controls

| Input          | Action                                        |
| -------------- | --------------------------------------------- |
| Left-drag      | Use the active tool at the cursor             |
| Shift+click    | Toggle a persistent spring (water source)     |
| Right-drag     | Orbit the camera                              |
| Mouse wheel    | Zoom                                          |
| `1`–`6`        | Tools: water, add material, dig, replace, plant trees, cut trees |
| `V`            | Cycle view: normal / moisture / strata / X-ray |
| `B`            | Toggle solid map edges                        |
| `R` / button   | Generate a new terrain                        |
| `C` / button   | Drain all water (surface + ground)            |
| Sliders        | Flow rate, brush radius, replace depth/band, cutaway position, generation relief / island size / soil depth |

### Tools

- **Water** — pour water at the cursor.
- **Add** — deposit the selected material (rock / clay / silt / sand / gravel /
  loam / regolith / boulders) on the surface.
- **Dig** — excavate the column from the top down; groundwater squeezed out of
  the removed strata pools in the pit.
- **Replace** — the underground layer editor: swaps whatever sits in a depth
  band below the surface (depth + band sliders) for the selected material
  without changing the terrain height — strata are split at the band edges, so
  you can thread a gravel aquifer or a clay seal through any column. Pair it
  with the cutaway to watch the strata change.
- **Plant / Cut trees** — paint or erase forest. Trees need soil and dry
  ground, strengthen the surface against hydraulic erosion, improve
  infiltration, and drown under deep or fast water.

### Underground

Columns reach from the surface down to a world floor 26 m below sea level.
The basement is a sedimentary basin: rock-cored mountains with a thin
weathered skin, but deep fill under lowlands and the coast — regolith, then
two stacked aquifer cycles (clay / **gravel lens** / silt, then clay /
**gravel lens** / silt / sand) whose gravel members pinch in and out as
buried lenses sealed by the clay beds.
The **cutaway** slider slices the world along a vertical plane and renders the
cross-section: every stratum in its own color, with its own blue saturated
zone and phreatic line — you can watch a perched table sit on a clay bed while
a confined gravel aquifer glows below it. The **moisture** view paints the
surface by column saturation, and the **X-ray** view raymarches the whole
volume — terrain turns into a ghosted silhouette and each water-bearing bed
glows see-through on a shallow-cyan → deep-violet color scale (legend in the
panel). **Solid map edges** turns the open ocean boundary into an impenetrable
wall so water (surface and ground) can't leave; with open edges the aquifers
drain to (and below sea level, recharge from) the sea.

### Springs

Shift+click anywhere to plant a persistent water source that keeps pouring at
the current flow rate and brush radius (marked by a pulsing blue ring);
shift+click it again to remove it. Up to 16 springs; a new terrain clears them.

### Generation

The Generation section sets up the next ⟳ New Terrain: **relief** scales the
height amplitude, **island size** the landmass footprint, **soil depth** the
loam mantle thickness, **sediment depth** how deep the layered sedimentary
fill (and its aquifers) reaches before basement rock takes over. **Map size**
(128–512, also `?n=` in the URL) changes the grid resolution itself and
reloads the page, since every buffer and shader is sized to the grid.

### Mass budget

The HUD under the FPS counter shows live totals (terrain volume, surface
water, groundwater, suspended sediment) from a GPU reduction. Terrain +
sediment is closed: erosion, deposition, thermal slippage and the flux-form
sediment advection all conserve it exactly (validated against float round-off
over a walled pour run). Water is intentionally *not* closed: it slowly
evaporates (half-life ≈ 2 minutes), soaks into the ground, and drains off
open map edges — enable solid edges to keep it on the map.

## How it works

Everything lives in two WGSL modules inside `main.js`; the simulation never
reads back to the CPU.

- **Terrain** — ridged fbm with domain warp and a radial island falloff sets
  the surface; below it a generation pass builds a stratigraphic column per
  cell (up to 10 strata of `(material, thickness, stored water)`, bottom-up
  from a fixed world floor): a basin-shaped bedrock basement (deep sediment
  fill under lowlands, rock-cored peaks) with a gentle regional tilt,
  weathered regolith, then two stacked aquifer cycles — clay / **gravel
  lens** / silt, clay / **gravel lens** / silt / sand — whose thickness
  fields are low-frequency noise so every bed stays laterally connected;
  where a gravel member is present it is a confined aquifer. Loam soil
  mantles gentle slopes, beach sand rims the shore, boulder fields collect
  on steeper ground, forests grow on mid-altitude soil.
- **Water** — the virtual-pipes shallow-water model (Mei et al. 2007):
  per-cell outflow flux in 4 directions, depth integration, and a velocity
  field, run 3 substeps per frame.
- **Groundwater** — a quasi-3D, MODFLOW-flavoured scheme. Each substep,
  surface water infiltrates the topmost stratum at the exposed material's
  rate (boulders ≫ gravel ≫ sand ≫ loam ≫ silt ≫ clay ≫ rock), then drains
  down the stack throttled by the *receiving* material's conductivity — a
  clay bed below means a perched water table above. Laterally, neighbouring
  columns exchange water per overlapping stratum pair (two-pointer sweep over
  both stacks): harmonic-mean conductivity × depth overlap × head difference,
  with symmetric donor caps so the exchange conserves mass exactly. A full
  stratum stores a little extra volume as stiff *overpressure*: it drives
  fast lateral flow along confined gravel (artesian pressure), seeps up
  through permeable roofs, is held by clay or rock seals, and wells out as a
  spring where the top of the column can't hold it.
- **Erosion** — sediment capacity from slope × speed × depth; the *topmost*
  material dissolves at its own erodibility (loam and silt fast, clay slow,
  boulders and bedrock barely), reduced by tree cover, and deposits settle
  out as sand. A relief clamp against the 4-neighbour mean keeps the feedback
  loop from checkerboarding. Suspended sediment rides the same pipe fluxes as
  the water (concentration × moved volume, flux-form, exactly conservative);
  a thermal-erosion pass relaxes slopes past each material's angle of repose
  (sand slumps at ~35°, clay holds steep cuts, bedrock never moves).
- **Picking** — a single-thread compute raymarch writes the cursor hit
  point to a small GPU buffer that the tool pass and the cursor-ring shader
  consume directly (zero CPU readback).
- **Rendering** — heightfield meshes generated from `vertex_index`
  (sky / terrain / trees / cutaway / water passes, 4× MSAA): soft-shadow
  heightfield raymarch, procedural sky with sun and fbm clouds,
  depth-absorbing fresnel water with sediment tinting and flow foam,
  material-aware terrain albedo (grass, boulder speckle, snow, erosion scars,
  damp-soil darkening), ~65k instanced low-poly pines scaled by tree density,
  and a cross-section curtain that draws strata + groundwater at the cutaway
  plane. ACES tonemapping.

## Files

- `index.html` — UI shell (panel, tools, sliders, hints, FPS counter)
- `main.js` — the entire engine: WGSL shaders, pipelines, camera, input, frame loop
- `validation/` — headless shader + simulation validation harness (lavapipe)
