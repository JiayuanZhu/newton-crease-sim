# Newton Crease Plasticity — Paper Airplane Crash Demo

Replicating the ARCSim dart crash demo (Narain et al. 2013, Figure 6) using **NVIDIA Newton** physics engine with VBD solver and crease plasticity.

## Overview

A pre-folded paper airplane flies into a wall and forms new creases on impact. This demonstrates:

- **VBD solver** maintaining folded rest shape during flight
- **Crease plasticity** (Narain et al. 2013) creating permanent fold lines on collision
- **ViewerRTX** interactive ray-traced rendering
- **USD-first scene** with composed deformable and collision schemas

## Key Insight

Setting `edge_rest_angle` to match the dihedral angles of the pre-folded mesh allows VBD to treat the folded airplane as its equilibrium state. Shape deviation during flight is < 0.00001 RMS.

## Files

| File | Description |
|------|-------------|
| `dart_crash_rtx.py` | **Main demo** — dart crash with ViewerRTX rendering |
| `crease_plasticity.py` | Crease plasticity module (Narain et al. 2013) |
| `example_cloth_crease.py` | Reference: cloth dropping on wedge |
| `create_dart_usd.py` | Tool: convert dart.obj to USD |
| `meshes/dart.usda` | Paper airplane mesh, cloth schemas, paper thickness and material |
| `meshes/scene.usda` | Unified scene referencing `dart.usda`, with wall/ground colliders |

## Results

- `newton_dart_crash.mp4` — Latest Newton VBD dart crash result

## Environment Setup

**Requirements:**
- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with CUDA (tested on L40, CUDA 12.9)
- Python 3.10+

**Install Warp:**
```bash
pip install warp-lang
```

**Install Newton (from source, dev build):**
```bash
git clone https://github.com/NVIDIA/newton.git
cd newton
pip install -e .
```

Newton is currently in early access / dev. If you have a released wheel:
```bash
pip install newton-physics
```

**Verify GPU:**
```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

## Usage

```bash
python dart_crash_rtx.py
```

The left UI panel is hidden initially. Press `H` to show/hide it, `Tab` to
switch between the fixed and follow cameras, or `R` to reset the full dart
state and relaunch it. The UI's **Reset** button performs the same reset.

This will:
1. Compose `meshes/scene.usda`, which references `meshes/dart.usda`
2. Import the deformable dart and static wall/ground through `ModelBuilder.add_usd`
3. Refine the authored dart cage in memory to 369 particles / 640 triangles
4. Compute folded-geometry dihedral angles as VBD rest angles
5. Simulate flight at 10 m/s and render it with ViewerRTX

USD stores gravity, cloth thickness/density/stretch/bend values, collision
schemas, contact stiffness/damping/friction, transforms and display materials.
Newton currently does not import cloth initial velocity, area stiffness,
damping, subdivision, or rest bend angles from USD, so those values remain
small, explicit runtime overrides in `dart_crash_rtx.py`.

USD paper thickness is 2 mm (mass/stiffness). Particle collision radius is
overridden at runtime to 10 mm with a 50 mm soft-contact margin so the dart
does not tunnel the wall at 10 m/s. Particle self-contact stays off: folded
layers sit closer than 2×radius and would explode if enabled.

## References

- Narain, R., Samii, A., O'Brien, J.F. (2012). "Adaptive Anisotropic Remeshing for Cloth Simulation." ACM TOG (SIGGRAPH Asia)
- Narain, R., Pfaff, T., O'Brien, J.F. (2013). "Folding and Crumpling Adaptive Sheets." ACM TOG (SIGGRAPH)
