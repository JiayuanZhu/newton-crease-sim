# Newton Crease Simulation

Integrating the **Folding and Crumpling** plasticity algorithm from [Narain et al. 2013](https://objf.ai/papers/Narain-FCA-2013-07/) into [NVIDIA Newton](https://github.com/newton-physics/newton)'s VBD (Vertex Block Descent) thin-shell solver.

## Concept

Newton's VBD solver simulates thin shells (cloth) using dihedral-angle-based bending energy:

```
E_bend = k * (θ - θ_rest)²
```

where `θ` is the current dihedral angle and `θ_rest` is the rest angle per edge.

**The key insight**: By dynamically updating `θ_rest` when the elastic bending exceeds a yield threshold, we can simulate **permanent creases and folds** — exactly what Narain et al. achieve with their plastic curvature tensor `S_plastic`.

## Algorithm

After each simulation substep:

1. **Measure elastic bending** for each edge: `Δθ = θ_current - θ_rest`
2. **Check yield criterion**: if `|Δθ| > θ_yield`
3. **Plastic flow**: update `θ_rest += (|Δθ| - θ_yield) * sign(Δθ)`
4. **Track damage**: accumulate plastic strain for visualization
5. **(Optional) Adaptive refinement**: split edges near sharp creases to better resolve fold geometry

This mirrors ARCSim's `plastic_update()` which tracks face-level `S_plastic` and maps it back to edge rest angles.

## Why it works

- Newton already stores `edge_rest_angle` per edge — this is the natural target for plasticity
- VBD is iterative and positional — injecting rest angle changes between substeps is stable
- The yield criterion prevents noise from causing spurious creases
- Damage accumulation allows material weakening at fold lines (lower yield threshold with repeated folding)

## Files

- `example_cloth_crease.py` — Main sample: cloth dropping onto a wedge, forming permanent creases
- `crease_plasticity.py` — The plasticity module (yield detection + rest angle update)
- `requirements.txt` — Dependencies

## Usage

```bash
pip install newton[examples]
python example_cloth_crease.py
```

## References

- Narain, R., Pfaff, T., O'Brien, J.F. "Folding and Crumpling Adaptive Sheets". ACM TOG 32(4), 2013.
- Newton Physics Engine: https://github.com/newton-physics/newton
- ARCSim source: http://graphics.berkeley.edu/resources/ARCSim/
