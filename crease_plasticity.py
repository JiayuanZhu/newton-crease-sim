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
  - Authored folds can be protected so impact crumples without erasing them
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
    yield_angles: wp.array(dtype=wp.float32),
    flow_rate: float,
    max_plastic_angle: float,
    damage: wp.array(dtype=wp.float32),
    damage_rate: float,
):
    """
    Per-edge plastic update (Narain et al. 2013, adapted for edge angles).

    For each edge:
      elastic_strain = theta_current - theta_rest
      if |elastic_strain| > yield_angle[edge]:
          plastic_increment = flow_rate * (|elastic_strain| - yield_angle) * sign(elastic_strain)
          theta_rest += plastic_increment
          damage += |plastic_increment| * damage_rate

    Authored folds may use a higher yield_angle but still plasticize under large strain.
    """
    idx = wp.tid()

    # Skip boundary edges. Authored folds may use a higher yield, but they are
    # never hard-locked: large impact strain must plasticize or the dart springs back.
    if edge_indices[idx, 0] == -1 or edge_indices[idx, 1] == -1:
        return

    theta = dihedral_angles[idx]
    rest = edge_rest_angle[idx]
    yield_angle = yield_angles[idx]

    elastic_strain = theta - rest
    abs_strain = wp.abs(elastic_strain)

    if abs_strain > yield_angle:
        excess = abs_strain - yield_angle
        increment = flow_rate * excess

        if elastic_strain > 0.0:
            new_rest = rest + increment
        else:
            new_rest = rest - increment

        if new_rest > max_plastic_angle:
            new_rest = max_plastic_angle
        elif new_rest < -max_plastic_angle:
            new_rest = -max_plastic_angle

        edge_rest_angle[idx] = new_rest
        damage[idx] = damage[idx] + increment * damage_rate


@wp.kernel
def weaken_yield_by_damage(
    damage: wp.array(dtype=wp.float32),
    base_yield: wp.array(dtype=wp.float32),
    weakening_factor: float,
    min_yield_angle: float,
    effective_yield: wp.array(dtype=wp.float32),
):
    """
    Compute effective yield angle per edge, weakened by accumulated damage.

    effective_yield = max(base_yield - weakening_factor * damage, min_yield)
    """
    idx = wp.tid()
    ey = base_yield[idx] - weakening_factor * damage[idx]
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
    - Optional protection of authored fold edges

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

        self.num_edges = model.edge_indices.shape[0]

        self.damage = wp.zeros(self.num_edges, dtype=wp.float32)
        self.dihedral_angles = wp.zeros(self.num_edges, dtype=wp.float32)
        self.base_yield = wp.zeros(self.num_edges, dtype=wp.float32)
        self.effective_yield = wp.zeros(self.num_edges, dtype=wp.float32)
        self.fold_edges = wp.zeros(self.num_edges, dtype=wp.int32)

        self.base_yield.fill_(yield_angle)
        self.effective_yield.fill_(yield_angle)

    def protect_initial_folds(
        self,
        rest_angles,
        threshold: float = 0.4,
        protected_yield: float = 0.55,
    ) -> int:
        """Raise yield on authored folds so mild flight noise does not erase them.

        Large impact strain still exceeds the elevated yield and plasticizes the
        fold permanently — otherwise the dart elastically springs back to shape
        and self-propels after the crash.

        Args:
            rest_angles: Current rest dihedral angles (wp.array or numpy).
            threshold: |rest| above this marks an authored fold (radians).
            protected_yield: Elevated yield for fold edges (still finite).

        Returns:
            Number of fold edges with elevated yield.
        """
        rest = rest_angles.numpy() if hasattr(rest_angles, "numpy") else np.asarray(rest_angles)
        base = np.full(self.num_edges, self.yield_angle, dtype=np.float32)
        folds = (np.abs(rest) >= threshold).astype(np.int32)
        base[folds.astype(bool)] = protected_yield
        self.base_yield.assign(base)
        self.effective_yield.assign(base)
        self.fold_edges.assign(folds)
        return int(folds.sum())

    def step(self, state):
        """
        Call after each VBD solver step to apply plasticity.

        Args:
            state: Newton State containing current particle positions
        """
        wp.launch(
            compute_dihedral_angles,
            dim=self.num_edges,
            inputs=[
                state.particle_q,
                self.model.edge_indices,
                self.dihedral_angles,
            ],
        )

        if self.weakening_factor > 0.0:
            wp.launch(
                weaken_yield_by_damage,
                dim=self.num_edges,
                inputs=[
                    self.damage,
                    self.base_yield,
                    self.weakening_factor,
                    self.min_yield_angle,
                    self.effective_yield,
                ],
            )
        else:
            self.effective_yield.assign(self.base_yield)

        wp.launch(
            plastic_update_kernel,
            dim=self.num_edges,
            inputs=[
                self.dihedral_angles,
                self.model.edge_rest_angle,
                self.model.edge_indices,
                self.effective_yield,
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
        """Reset damage and restore per-edge yield from the protected base map."""
        self.damage.zero_()
        self.effective_yield.assign(self.base_yield)
