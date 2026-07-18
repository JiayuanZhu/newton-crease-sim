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

## Dependencies

- NVIDIA Newton (with Warp)
- CUDA GPU (tested on NVIDIA L40)

## Usage

```bash
python dart_crash_rtx.py
```

## References

- Narain, R., Samii, A., O'Brien, J.F. (2012). "Adaptive Anisotropic Remeshing for Cloth Simulation." ACM TOG (SIGGRAPH Asia)
- Narain, R., Pfaff, T., O'Brien, J.F. (2013). "Folding and Crumpling Adaptive Sheets." ACM TOG (SIGGRAPH)
