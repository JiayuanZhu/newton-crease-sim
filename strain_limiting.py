# SPDX-License-Identifier: Apache-2.0
"""
Edge-based Strain Limiting for Newton VBD Cloth Solver

Post-step position projection that enforces maximum edge stretch ratio.
This provides "rigid-like" in-plane behavior without requiring VBD to
converge fully, while leaving bending (dihedral angles) unconstrained.

Algorithm:
  For each edge (i, j) with rest length L0:
    L = ||x_i - x_j||
    if L / L0 > max_stretch:
        overshoot = L - max_stretch * L0
        correction = overshoot / 2 (split by inverse mass)
        move i and j toward each other along edge direction

Multiple Jacobi iterations allow corrections to propagate through the mesh.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def edge_strain_limit_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=wp.float32),
    spring_indices: wp.array(dtype=wp.int32),
    spring_rest_length: wp.array(dtype=wp.float32),
    max_stretch: float,
    dt_inv: float,
    corrections: wp.array(dtype=wp.vec3),
    correction_counts: wp.array(dtype=wp.int32),
):
    """Compute position corrections for edges exceeding max stretch ratio.

    Uses a gather/scatter pattern: each edge computes corrections for its two
    endpoints, atomically adds them to a shared buffer. A second pass averages
    and applies.
    """
    edge_idx = wp.tid()

    i = spring_indices[edge_idx * 2 + 0]
    j = spring_indices[edge_idx * 2 + 1]

    xi = particle_q[i]
    xj = particle_q[j]

    rest_len = spring_rest_length[edge_idx]
    max_len = max_stretch * rest_len

    diff = xi - xj
    current_len = wp.length(diff)

    # Skip degenerate or non-violated edges
    if current_len <= max_len or current_len < 1.0e-8:
        return

    # Edge direction (i -> j normalized)
    direction = diff / current_len

    # Overshoot beyond allowed stretch
    overshoot = current_len - max_len

    # Inverse mass weighting
    wi = particle_inv_mass[i]
    wj = particle_inv_mass[j]
    w_sum = wi + wj

    if w_sum < 1.0e-10:
        return  # Both pinned, cannot correct

    # Correction magnitudes (mass-weighted)
    ci = -(wi / w_sum) * overshoot
    cj = (wj / w_sum) * overshoot

    # Accumulate corrections
    wp.atomic_add(corrections, i, ci * direction)
    wp.atomic_add(corrections, j, cj * direction)
    wp.atomic_add(correction_counts, i, 1)
    wp.atomic_add(correction_counts, j, 1)


@wp.kernel
def apply_corrections_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    correction_counts: wp.array(dtype=wp.int32),
    dt_inv: float,
    damping: float,
):
    """Apply averaged corrections to positions and update velocities."""
    i = wp.tid()
    count = correction_counts[i]
    if count == 0:
        return

    # Average the accumulated corrections from all incident edges
    avg_correction = corrections[i] / wp.float32(count)

    # Apply position correction
    particle_q[i] = particle_q[i] + avg_correction

    # Update velocity to reflect the position change (implicit velocity update)
    # This prevents the solver from undoing the correction next step
    particle_qd[i] = particle_qd[i] + avg_correction * dt_inv * damping


class StrainLimiter:
    """
    Post-step strain limiting for Newton VBD cloth.

    Enforces maximum edge elongation by iteratively projecting particle
    positions. Only constrains in-plane stretch (edge lengths); bending
    is left free for crease formation.

    Parameters:
        model: Newton Model instance with spring_indices and spring_rest_length.
        max_stretch: Maximum allowed stretch ratio (1.0 = no stretch at all,
                     1.005 = 0.5% elongation allowed). Default 1.005.
        iterations: Number of Jacobi projection passes per call. More iterations
                    allow corrections to propagate further through the mesh.
        velocity_damping: Fraction of position correction applied to velocity.
                         1.0 = full velocity update, 0.0 = position-only.
    """

    def __init__(
        self,
        model,
        max_stretch: float = 1.005,
        iterations: int = 3,
        velocity_damping: float = 0.8,
    ):
        self.model = model
        self.max_stretch = max_stretch
        self.iterations = iterations
        self.velocity_damping = velocity_damping

        self.num_particles = model.particle_count
        self.num_springs = model.spring_rest_length.shape[0]

        # Pre-allocate correction buffers
        self.corrections = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.correction_counts = wp.zeros(self.num_particles, dtype=wp.int32)

    def limit(self, state, dt: float, max_stretch: float | None = None, iterations: int | None = None):
        """
        Apply strain limiting to the given state.

        Call this AFTER solver.step() and BEFORE plasticity.step().

        Args:
            state: Newton State with particle_q and particle_qd.
            dt: Simulation timestep (for velocity correction).
            max_stretch: Override max stretch ratio (or use instance default).
            iterations: Override iteration count (or use instance default).
        """
        max_s = max_stretch if max_stretch is not None else self.max_stretch
        iters = iterations if iterations is not None else self.iterations
        dt_inv = 1.0 / dt if dt > 1.0e-10 else 0.0

        for _ in range(iters):
            # Clear buffers
            self.corrections.zero_()
            self.correction_counts.zero_()

            # Compute corrections for all violated edges
            wp.launch(
                edge_strain_limit_kernel,
                dim=self.num_springs,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.model.particle_inv_mass,
                    self.model.spring_indices,
                    self.model.spring_rest_length,
                    max_s,
                    dt_inv,
                    self.corrections,
                    self.correction_counts,
                ],
            )

            # Apply averaged corrections
            wp.launch(
                apply_corrections_kernel,
                dim=self.num_particles,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.corrections,
                    self.correction_counts,
                    dt_inv,
                    self.velocity_damping,
                ],
            )

    def get_max_stretch_ratio(self, state) -> float:
        """Diagnostic: compute the current maximum stretch ratio across all edges."""
        pos = state.particle_q.numpy()
        indices = self.model.spring_indices.numpy().reshape(-1, 2)
        rest = self.model.spring_rest_length.numpy()

        edge_vecs = pos[indices[:, 0]] - pos[indices[:, 1]]
        lengths = np.linalg.norm(edge_vecs, axis=1)
        ratios = lengths / np.maximum(rest, 1e-8)
        return float(np.max(ratios))
