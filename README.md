# Newton Crease Plasticity — Paper Airplane Crash Demo

Replicating the ARCSim dart crash demo (Narain et al. 2013, Figure 6) using **NVIDIA Newton** physics engine with VBD solver and crease plasticity.

## Overview

A pre-folded paper airplane flies into a wall and forms new creases on impact. This demonstrates:

- **VBD solver** maintaining folded rest shape during flight
- **Crease plasticity** (Narain et al. 2013) creating permanent fold lines on collision
- **ViewerRTX** headless ray-traced rendering

## Key Insight

Setting `edge_rest_angle` to match the dihedral angles of the pre-folded mesh allows VBD to treat the folded airplane as its equilibrium state. Shape deviation during flight is < 0.00001 RMS.

## Files

| File | Description |
|------|-------------|
| `dart_crash_rtx.py` | **Main demo** — dart crash with ViewerRTX rendering |
| `crease_plasticity.py` | Crease plasticity module (Narain et al. 2013) |
| `example_cloth_crease.py` | Reference: cloth dropping on wedge |
| `create_dart_usd.py` | Tool: convert dart.obj to USD |
| `dart.usda` | Paper airplane mesh in USD format |

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

This will:
1. Load `dart.obj` (from ARCSim) and subdivide to 369 verts / 640 faces
2. Compute dihedral angles of the folded shape → set as VBD rest angles
3. Simulate flight at 10 m/s into a wall
4. Render with ViewerRTX (headless ray tracing)
5. Output `newton_dart_crash.mp4`

The `dart.obj` mesh is included in `meshes/`.

## References

- Narain, R., Samii, A., O'Brien, J.F. (2012). "Adaptive Anisotropic Remeshing for Cloth Simulation." ACM TOG (SIGGRAPH Asia)
- Narain, R., Pfaff, T., O'Brien, J.F. (2013). "Folding and Crumpling Adaptive Sheets." ACM TOG (SIGGRAPH)
