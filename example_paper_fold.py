# SPDX-License-Identifier: Apache-2.0
"""
Example: Paper Folding with Permanent Creases

A sheet of paper is folded in half and then released. The crease formed
during folding persists permanently due to bending plasticity.

This demonstrates the integration of Narain et al. 2013's plastic bending
model into Newton's VBD solver:
  1. Paper starts flat
  2. One half is kinematically driven to fold over the other half (0-3 seconds)
  3. The fold is held briefly to let plasticity settle (3-4 seconds)
  4. The kinematic constraints are released (4+ seconds)
  5. The paper retains a permanent crease at the fold line

The key physics: when dihedral angles exceed the yield threshold during
folding, the edge rest angles are permanently updated. When released,
the paper "remembers" the fold.

Command: python example_paper_fold.py
"""

import math
import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

from crease_plasticity import CreasePlasticity


class Example:
    def __init__(self, viewer=None, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # --- Paper dimensions (A5-ish: 15cm x 10cm) ---
        self.paper_width = 0.15   # m (along x, fold axis will be at x=0)
        self.paper_height = 0.10  # m (along y, fold direction)
        self.grid_x = 30          # cells along width
        self.grid_y = 20          # cells along height
        self.cell_x = self.paper_width / self.grid_x
        self.cell_y = self.paper_height / self.grid_y

        # Paper particle count
        self.num_verts_x = self.grid_x + 1  # 31
        self.num_verts_y = self.grid_y + 1  # 21
        self.num_particles = self.num_verts_x * self.num_verts_y  # 651

        # --- Simulation phases ---
        self.phase_fold_start = 0.5    # Start folding at t=0.5s
        self.phase_fold_end = 3.0      # Complete fold by t=3.0s
        self.phase_hold_end = 4.0      # Hold fold until t=4.0s
        self.phase_release = 4.0       # Release at t=4.0s

        # --- Build model ---
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

        # Paper physical properties:
        # Real paper: ~80 g/m², area=0.15*0.10=0.015 m² → mass=1.2g
        # Per particle: 1.2e-3 / 651 ≈ 1.8e-6 kg (very light)
        # We use slightly heavier for numerical stability
        paper_mass_per_particle = 5.0e-5  # kg

        # Paper material properties:
        # - Very stiff in-plane (paper doesn't stretch)
        # - Moderate bending stiffness (paper resists bending but can fold)
        tri_ke = 1.0e5   # High stretch stiffness (paper doesn't stretch)
        tri_ka = 1.0e5   # High shear stiffness
        tri_kd = 5.0e0   # Some in-plane damping
        edge_ke = 5.0e1  # Bending stiffness (paper is moderately stiff in bending)
        edge_kd = 2.0e0  # Bending damping

        builder.add_cloth_grid(
            pos=wp.vec3(-self.paper_width / 2, -self.paper_height / 2, 0.3),
            rot=wp.quat_identity(),  # Paper lies in XY plane, Z up
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.grid_x,
            dim_y=self.grid_y,
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            mass=paper_mass_per_particle,
            fix_left=False,
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            particle_radius=0.002,
        )

        # Ground plane for the paper to rest on
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = 1.0e5
        ground_cfg.kd = 1.0e2
        ground_cfg.mu = 0.3
        builder.add_ground_plane(cfg=ground_cfg)

        builder.color(include_bending=True)
        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 50.0
        self.model.soft_contact_mu = 0.4

        # --- Identify particle indices for folding ---
        # Paper grid: vertex (i, j) = j * num_verts_x + i
        # Fold axis: the center column (y = paper_height/2)
        # "Bottom half" (y < center): these will be folded over
        # "Top half" (y > center): stays put (kinematically fixed during fold)
        self.fold_row = self.num_verts_y // 2  # Row 10

        # Bottom half particles (rows 0 to fold_row-1) will be driven
        self.driven_particles = []
        for j in range(self.fold_row):
            for i in range(self.num_verts_x):
                self.driven_particles.append(j * self.num_verts_x + i)
        self.driven_particles = np.array(self.driven_particles)

        # Top row near fold (rows fold_row to fold_row+1) will be pinned during fold
        self.pinned_particles = []
        for j in range(self.fold_row, min(self.fold_row + 2, self.num_verts_y)):
            for i in range(self.num_verts_x):
                self.pinned_particles.append(j * self.num_verts_x + i)
        self.pinned_particles = np.array(self.pinned_particles)

        # Store initial positions for computing fold targets
        self.initial_positions = self.model.particle_q.numpy().copy()

        # --- Solver ---
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=15,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.003,
            particle_self_contact_margin=0.005,
        )

        # --- Plasticity Module ---
        self.plasticity = CreasePlasticity(
            model=self.model,
            yield_angle=0.12,         # ~7 degrees - paper yields relatively easily
            flow_rate=0.85,           # High flow rate - paper creases sharply
            max_plastic_angle=2.8,    # ~160 degrees max fold
            damage_rate=1.0,
            weakening_factor=0.03,
            min_yield_angle=0.02,
        )

        # --- States ---
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Initially pin the top half so paper doesn't fall
        self._set_particle_flags(self.pinned_particles, active=False)

        if self.viewer:
            self.viewer.set_model(self.model)

        self.released = False
        self.frame_count = 0

    def _set_particle_flags(self, indices, active=True):
        """Set particles as active or kinematic (fixed)."""
        flags = self.model.particle_flags.numpy()
        for idx in indices:
            if active:
                flags[idx] = flags[idx] | ParticleFlags.ACTIVE
            else:
                flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags, dtype=wp.int32)

    def _compute_fold_targets(self, fold_progress):
        """
        Compute target positions for the driven particles during folding.
        Fold is a rotation about the fold axis (x-axis at y=fold_center).

        fold_progress: 0.0 (flat) to 1.0 (fully folded over)
        """
        fold_center_y = self.initial_positions[self.fold_row * self.num_verts_x, 1]
        fold_center_z = self.initial_positions[self.fold_row * self.num_verts_x, 2]

        angle = fold_progress * math.pi  # 0 to π (180 degrees)

        targets = self.initial_positions.copy()
        for idx in self.driven_particles:
            pos = self.initial_positions[idx]
            # Rotate around fold axis (x-axis at fold_center_y, fold_center_z)
            dy = pos[1] - fold_center_y
            dz = pos[2] - fold_center_z
            # Apply rotation
            new_dy = dy * math.cos(angle) - dz * math.sin(angle)
            new_dz = dy * math.sin(angle) + dz * math.cos(angle)
            targets[idx, 1] = fold_center_y + new_dy
            targets[idx, 2] = fold_center_z + new_dz

        return targets

    def _apply_kinematic_fold(self):
        """Drive particles toward fold targets."""
        if self.sim_time < self.phase_fold_start:
            return
        if self.sim_time > self.phase_release:
            return

        # Compute fold progress
        if self.sim_time < self.phase_fold_end:
            t = (self.sim_time - self.phase_fold_start) / (self.phase_fold_end - self.phase_fold_start)
            # Smooth easing
            progress = 0.5 * (1.0 - math.cos(t * math.pi))
        else:
            progress = 1.0  # Fully folded, holding

        targets = self._compute_fold_targets(progress)

        # Move driven particles toward targets (soft kinematic)
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        blend = 0.3  # How strongly we drive toward target each substep

        for idx in self.driven_particles:
            diff = targets[idx] - q[idx]
            q[idx] += diff * blend
            qd[idx] = diff * blend / self.sim_dt * 0.1  # Gentle velocity

        self.state_0.particle_q = wp.array(q, dtype=wp.vec3)
        self.state_0.particle_qd = wp.array(qd, dtype=wp.vec3)

    def _check_release(self):
        """Release all constraints after hold phase."""
        if self.sim_time >= self.phase_release and not self.released:
            print(f"\n{'='*50}")
            print(f"[t={self.sim_time:.1f}s] RELEASING PAPER - fold complete!")
            print(f"{'='*50}")

            # Release all particles
            self._set_particle_flags(self.pinned_particles, active=True)
            self.released = True

            # Print plasticity stats at release
            damage = self.plasticity.get_damage_numpy()
            rest_angles = self.plasticity.get_rest_angles_numpy()
            num_creased = np.sum(np.abs(rest_angles) > 0.01)
            max_rest = np.max(np.abs(rest_angles))
            print(f"  Creased edges: {num_creased}/{len(rest_angles)}")
            print(f"  Max rest angle: {math.degrees(max_rest):.1f}°")
            print(f"  Total damage: {np.sum(damage):.2f}")
            print(f"  → The paper should retain its crease!\n")

    def simulate(self):
        """Run one frame of simulation."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            if self.viewer:
                self.viewer.apply_forces(self.state_0)

            # Apply folding motion (kinematic drive)
            if not self.released:
                self._apply_kinematic_fold()

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )

            # === PLASTICITY: detect yield and update rest angles ===
            self.plasticity.step(self.state_1)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance one frame."""
        self._check_release()
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1

        # Print periodic stats
        if self.frame_count % 120 == 0:
            damage = self.plasticity.get_damage_numpy()
            rest_angles = self.plasticity.get_rest_angles_numpy()
            max_rest = np.max(np.abs(rest_angles))
            num_creased = np.sum(np.abs(rest_angles) > 0.01)
            phase = "FOLDING" if self.sim_time < self.phase_fold_end else \
                    "HOLDING" if self.sim_time < self.phase_release else "FREE"
            print(
                f"[t={self.sim_time:.1f}s | {phase}] "
                f"Creased: {num_creased} edges | "
                f"Max fold: {math.degrees(max_rest):.1f}° | "
                f"Damage: {np.sum(damage):.2f}"
            )

    def render(self):
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer=viewer, args=args)

    newton.examples.run(example, args)
