# SPDX-License-Identifier: Apache-2.0
"""
Example: Cloth with Crease Plasticity

A cloth sheet drops onto a wedge obstacle, forming permanent creases.
Uses Newton's VBD solver with the crease plasticity module that implements
Narain et al. 2013's folding algorithm adapted for Newton's edge-based
dihedral angle representation.

The key difference from standard Newton cloth: after the cloth forms a fold
over the wedge, the crease is PERMANENT — it persists even after the obstacle
is removed or the cloth is lifted.

Command: python example_cloth_crease.py
"""

import os
import math

import numpy as np
import warp as wp

import newton
import newton.examples

from crease_plasticity import CreasePlasticity


def create_wedge_body(builder, pos, half_width=0.3, height=0.15, depth=0.5):
    """Create a wedge-shaped rigid body as an obstacle to induce creases."""
    # Approximate wedge with a thin box (tilted) - for collision purposes
    # In practice you'd use a proper mesh shape
    b = builder.add_body(
        origin=wp.transform(pos, wp.quat_identity()),
    )
    # Use a box shape as simple wedge approximation
    builder.add_shape_box(
        body=b,
        hx=half_width,
        hy=height * 0.3,
        hz=depth,
        ke=1.0e4,
        kd=100.0,
        kf=0.5,
        mu=0.3,
    )
    return b


class Example:
    """
    Cloth Crease Formation Demo

    A square cloth drops onto a wedge obstacle. The VBD solver simulates
    the bending, and our CreasePlasticity module detects when bending exceeds
    the yield threshold, permanently modifying the edge rest angles.

    After the cloth drapes over the wedge and the simulation settles,
    you can observe that the fold line has become a permanent crease in the
    rest shape of the cloth.
    """

    def __init__(self, viewer=None, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # --- Build the model ---
        builder = newton.ModelBuilder()

        # Ground plane
        builder.add_ground_plane()

        # Create a cloth grid (40x40)
        cloth_size = 40
        cloth_scale = 0.01  # Each grid cell is 1cm, total 40cm
        vertices = []
        indices = []

        for j in range(cloth_size):
            for i in range(cloth_size):
                x = (i - cloth_size / 2) * cloth_scale
                z = (j - cloth_size / 2) * cloth_scale
                vertices.append(wp.vec3(x, 0.0, z))

        for j in range(cloth_size - 1):
            for i in range(cloth_size - 1):
                v0 = j * cloth_size + i
                v1 = v0 + 1
                v2 = v0 + cloth_size
                v3 = v2 + 1
                indices.extend([v0, v1, v2])
                indices.extend([v1, v3, v2])

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.5, 0.0),  # Start above the wedge
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vertices,
            indices=indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.1,  # Light cloth (like paper/thin fabric)
            tri_ke=5.0e5,  # High in-plane stiffness (paper-like)
            tri_ka=5.0e5,
            tri_kd=10.0,
            edge_ke=5.0e0,  # Moderate bending stiffness
            edge_kd=0.1,
        )

        # Add wedge obstacle
        # Positioned so the cloth falls onto it
        create_wedge_body(builder, pos=wp.vec3(0.0, 0.2, 0.0))

        builder.color(include_bending=True)
        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 10.0
        self.model.soft_contact_mu = 0.5

        # --- Create VBD Solver ---
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=15,
            particle_enable_self_contact=False,
        )

        # --- Create Plasticity Module ---
        # This is the Narain et al. 2013 adaptation
        self.plasticity = CreasePlasticity(
            model=self.model,
            yield_angle=0.15,         # ~8.6 degrees yield threshold
            flow_rate=0.7,            # 70% of excess converts to plastic
            max_plastic_angle=2.0,    # Max ~115 degrees permanent fold
            damage_rate=0.5,          # Damage accumulation rate
            weakening_factor=0.02,    # Slight weakening with repeated folding
            min_yield_angle=0.03,     # Minimum yield (~1.7 degrees)
        )

        # --- States ---
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        if self.viewer:
            self.viewer.set_model(self.model)

        # Stats tracking
        self.total_damage = 0.0
        self.max_damage_edge = 0
        self.frame_count = 0

    def simulate(self):
        """Run one frame of simulation with plasticity."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            if self.viewer:
                self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )

            # === PLASTICITY STEP ===
            # After each VBD substep, check for yield and update rest angles
            self.plasticity.step(self.state_1)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance one frame."""
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1

        # Print stats every 60 frames (1 second)
        if self.frame_count % 60 == 0:
            damage = self.plasticity.get_damage_numpy()
            total = np.sum(damage)
            max_d = np.max(damage)
            num_creased = np.sum(damage > 0.01)
            rest_angles = self.plasticity.get_rest_angles_numpy()
            max_rest = np.max(np.abs(rest_angles))
            print(
                f"[t={self.sim_time:.1f}s] "
                f"Total damage: {total:.2f} | "
                f"Max edge damage: {max_d:.3f} | "
                f"Creased edges: {num_creased}/{len(damage)} | "
                f"Max rest angle: {math.degrees(max_rest):.1f}°"
            )

    def render(self):
        """Render current state."""
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
