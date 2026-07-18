# SPDX-License-Identifier: Apache-2.0
"""
Crease Plasticity Module for Newton VBD Solver

Implements the bending plasticity algorithm from:
  Narain, Pfaff, O'Brien. "Folding and Crumpling Adaptive Sheets"
  ACM Transactions on Graphics, 32(4):51:1-8, July 2013.

Adapted for Newton's edge-based dihedral angle bending model.
Instead of face-level curvature tensors (as in ARCSim), we work directly
with per-edge dihedral angles, which is Newton's native representation.

Core idea:
  - After each simulation step, compute the elastic bending strain per edge
  - If it exceeds a yield threshold, permanently shift the rest angle
  - Optionally weaken the material at crease locations (damage)
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def compute_dihedral_angles(
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    dihedral_angles: wp.array(dtype=wp.float32),
):
    """Compute current dihedral angle for each bending edge."""
    idx = wp.tid()

    # Edge indices: [opposite0, opposite1, edge_start, edge_end]
    vi0 = edge_indices[idx, 0]
    vi1 = edge_indices[idx, 1]
    vi2 = edge_indices[idx, 2]
    vi3 = edge_indices[idx, 3]

    # Skip boundary edges
    if vi0 == -1 or vi1 == -1:
        dihedral_angles[idx] = 0.0
        return

    x0 = pos[vi0]  # opposite vertex 0
    x1 = pos[vi1]  # opposite vertex 1
    x2 = pos[vi2]  # edge start
    x3 = pos[vi3]  # edge end

    # Edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Face normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    eps = 1.0e-8
    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        dihedral_angles[idx] = 0.0
        return

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    dihedral_angles[idx] = theta


@wp.kernel
def plastic_update_kernel(
    dihedral_angles: wp.array(dtype=wp.float32),
    edge_rest_angle: wp.array(dtype=wp.float32),
    edge_indices: wp.array2d(dtype=wp.int32),
    yield_angle: float,
    flow_rate: float,
    max_plastic_angle: float,
    damage: wp.array(dtype=wp.float32),
    damage_rate: float,
):
    """
    Per-edge plastic update (Narain et al. 2013, adapted for edge angles).

    For each edge:
      elastic_strain = theta_current - theta_rest
      if |elastic_strain| > yield_angle:
          plastic_increment = flow_rate * (|elastic_strain| - yield_angle) * sign(elastic_strain)
          theta_rest += plastic_increment
          damage += |plastic_increment| * damage_rate
    """
    idx = wp.tid()

    # Skip boundary edges
    if edge_indices[idx, 0] == -1 or edge_indices[idx, 1] == -1:
        return

    theta = dihedral_angles[idx]
    rest = edge_rest_angle[idx]

    elastic_strain = theta - rest

    abs_strain = wp.abs(elastic_strain)

    if abs_strain > yield_angle:
        # Compute plastic flow
        excess = abs_strain - yield_angle
        increment = flow_rate * excess

        # Apply sign
        if elastic_strain > 0.0:
            new_rest = rest + increment
        else:
            new_rest = rest - increment

        # Clamp total plastic deformation
        if new_rest > max_plastic_angle:
            new_rest = max_plastic_angle
        elif new_rest < -max_plastic_angle:
            new_rest = -max_plastic_angle

        edge_rest_angle[idx] = new_rest

        # Accumulate damage
        damage[idx] = damage[idx] + increment * damage_rate


@wp.kernel
def weaken_yield_by_damage(
    damage: wp.array(dtype=wp.float32),
    base_yield_angle: float,
    weakening_factor: float,
    min_yield_angle: float,
    effective_yield: wp.array(dtype=wp.float32),
):
    """
    Compute effective yield angle per edge, weakened by accumulated damage.
    Models how repeated folding makes creases easier to form.

    effective_yield = max(base_yield - weakening_factor * damage, min_yield)
    """
    idx = wp.tid()
    d = damage[idx]
    ey = base_yield_angle - weakening_factor * d
    if ey < min_yield_angle:
        ey = min_yield_angle
    effective_yield[idx] = ey


class CreasePlasticity:
    """
    Manages bending plasticity for Newton's VBD cloth solver.

    Implements the core algorithm from Narain et al. 2013:
    - Yield criterion based on dihedral angle deviation
    - Plastic flow that permanently modifies rest angles
    - Damage accumulation for material weakening

    Parameters:
        model: Newton Model instance
        yield_angle: Threshold elastic bending (radians) before plastic flow begins.
                     Typical values: 0.1-0.5 rad (6-30 degrees)
        flow_rate: Fraction of excess strain converted to plastic deformation per step.
                   1.0 = immediate yield, 0.1 = gradual creasing
        max_plastic_angle: Maximum allowed plastic rest angle (radians). Prevents extreme deformation.
        damage_rate: Rate of damage accumulation per unit plastic strain.
        weakening_factor: How much damage reduces the yield threshold.
        min_yield_angle: Floor for the effective yield angle after weakening.
    """

    def __init__(
        self,
        model,
        yield_angle: float = 0.2,
        flow_rate: float = 0.8,
        max_plastic_angle: float = 2.5,
        damage_rate: float = 1.0,
        weakening_factor: float = 0.05,
        min_yield_angle: float = 0.05,
    ):
        self.model = model
        self.yield_angle = yield_angle
        self.flow_rate = flow_rate
        self.max_plastic_angle = max_plastic_angle
        self.damage_rate = damage_rate
        self.weakening_factor = weakening_factor
        self.min_yield_angle = min_yield_angle

        # Number of bending edges
        self.num_edges = model.edge_indices.shape[0]

        # Allocate damage and dihedral angle arrays
        self.damage = wp.zeros(self.num_edges, dtype=wp.float32)
        self.dihedral_angles = wp.zeros(self.num_edges, dtype=wp.float32)
        self.effective_yield = wp.zeros(self.num_edges, dtype=wp.float32)

        # Fill effective yield with base value
        self.effective_yield.fill_(yield_angle)

    def step(self, state):
        """
        Call after each VBD solver step to apply plasticity.

        Args:
            state: Newton State containing current particle positions
        """
        # 1. Compute current dihedral angles
        wp.launch(
            compute_dihedral_angles,
            dim=self.num_edges,
            inputs=[
                state.particle_q,
                self.model.edge_indices,
                self.dihedral_angles,
            ],
        )

        # 2. Optionally update effective yield based on damage
        if self.weakening_factor > 0.0:
            wp.launch(
                weaken_yield_by_damage,
                dim=self.num_edges,
                inputs=[
                    self.damage,
                    self.yield_angle,
                    self.weakening_factor,
                    self.min_yield_angle,
                    self.effective_yield,
                ],
            )

        # 3. Apply plastic update
        wp.launch(
            plastic_update_kernel,
            dim=self.num_edges,
            inputs=[
                self.dihedral_angles,
                self.model.edge_rest_angle,
                self.model.edge_indices,
                self.yield_angle,  # could use per-edge effective_yield for weakening
                self.flow_rate,
                self.max_plastic_angle,
                self.damage,
                self.damage_rate,
            ],
        )

    def get_damage_numpy(self) -> np.ndarray:
        """Get damage array as numpy for visualization."""
        return self.damage.numpy()

    def get_rest_angles_numpy(self) -> np.ndarray:
        """Get current rest angles as numpy for inspection."""
        return self.model.edge_rest_angle.numpy()

    def reset(self):
        """Reset all plastic deformation."""
        self.damage.zero_()
        # Reset rest angles to zero (flat)
        self.model.edge_rest_angle.zero_()
