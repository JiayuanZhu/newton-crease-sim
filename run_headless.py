#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Headless runner for the paper folding demo.
Runs the simulation without a graphical viewer and prints stats.
"""

import math
import sys
import numpy as np
import warp as wp
import newton
from newton import ParticleFlags
from crease_plasticity import CreasePlasticity


class NullViewer:
    """Minimal no-op viewer for headless runs."""
    def set_model(self, model): pass
    def begin_frame(self, t): pass
    def end_frame(self): pass
    def log_state(self, state): pass
    def log_contacts(self, contacts, state): pass
    def apply_forces(self, state): pass


def run_paper_fold(num_frames=600):
    """
    Simulate a paper sheet being folded in half and released.
    
    Timeline:
      0.0 - 0.5s: Paper settles (falls slightly under gravity, then rests)
      0.5 - 3.0s: Bottom half folds over top half (kinematic drive)
      3.0 - 4.0s: Hold fold in place (plasticity accumulates)
      4.0+      : Release — paper should retain crease
    """
    print("=" * 60)
    print("  Newton Crease Simulation — Paper Fold Demo")
    print("  Based on: Narain et al. 2013 'Folding and Crumpling'")
    print("=" * 60)
    
    # --- Parameters ---
    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = 20
    sim_dt = frame_dt / sim_substeps
    
    # Paper: 10cm x 6cm, grid 20x12
    grid_x = 20
    grid_y = 12
    cell_x = 0.005  # 5mm per cell → 10cm total width
    cell_y = 0.005  # 5mm per cell → 6cm total height
    num_verts_x = grid_x + 1  # 21
    num_verts_y = grid_y + 1  # 13
    
    paper_width = grid_x * cell_x
    paper_height = grid_y * cell_y
    
    print(f"\nPaper: {paper_width*100:.0f}cm × {paper_height*100:.0f}cm")
    print(f"Grid: {grid_x}×{grid_y} cells ({num_verts_x * num_verts_y} particles)")
    
    # --- Build Model ---
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    
    builder.add_cloth_grid(
        pos=wp.vec3(-paper_width / 2, -paper_height / 2, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(0, 0, 0),
        dim_x=grid_x,
        dim_y=grid_y,
        cell_x=cell_x,
        cell_y=cell_y,
        mass=5e-5,
        tri_ke=1e5,     # Very stiff in-plane
        tri_ka=1e5,
        tri_kd=5.0,
        edge_ke=30.0,   # Moderate bending stiffness
        edge_kd=1.5,
        particle_radius=0.002,
    )
    
    # Ground
    ground_cfg = newton.ModelBuilder.ShapeConfig()
    ground_cfg.ke = 1e5
    ground_cfg.kd = 100.0
    ground_cfg.mu = 0.5
    builder.add_ground_plane(cfg=ground_cfg)
    
    builder.color(include_bending=True)
    model = builder.finalize()
    model.soft_contact_ke = 5e4
    model.soft_contact_kd = 50.0
    model.soft_contact_mu = 0.4
    
    num_edges = model.edge_indices.shape[0]
    print(f"Bending edges: {num_edges}")
    
    # --- Solver ---
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=12,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.003,
        particle_self_contact_margin=0.005,
    )
    
    # --- Plasticity ---
    plasticity = CreasePlasticity(
        model=model,
        yield_angle=0.12,
        flow_rate=0.85,
        max_plastic_angle=2.8,
        damage_rate=1.0,
        weakening_factor=0.03,
        min_yield_angle=0.02,
    )
    
    # --- States ---
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    
    # --- Identify fold axis particles ---
    # Fold axis at y = 0 (center of paper in y direction)
    # Bottom half: rows j < fold_row
    fold_row = num_verts_y // 2  # row 6
    
    driven_particles = []
    for j in range(fold_row):
        for i in range(num_verts_x):
            driven_particles.append(j * num_verts_x + i)
    driven_particles = np.array(driven_particles)
    
    # Pin top rows near fold
    pinned_particles = []
    for j in range(fold_row, min(fold_row + 2, num_verts_y)):
        for i in range(num_verts_x):
            pinned_particles.append(j * num_verts_x + i)
    pinned_particles = np.array(pinned_particles)
    
    initial_positions = model.particle_q.numpy().copy()
    
    # Pin the top half initially
    flags = model.particle_flags.numpy()
    for idx in pinned_particles:
        flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
    model.particle_flags = wp.array(flags, dtype=wp.int32)
    
    # --- Timing phases ---
    phase_fold_start = 0.5
    phase_fold_end = 3.0
    phase_release = 4.0
    
    released = False
    sim_time = 0.0
    
    print(f"\nPhases:")
    print(f"  0.0 - 0.5s: Settle")
    print(f"  0.5 - 3.0s: Fold")
    print(f"  3.0 - 4.0s: Hold")
    print(f"  4.0+      : Release")
    print(f"\nRunning {num_frames} frames ({num_frames/fps:.1f}s)...")
    print("-" * 60)
    
    for frame in range(num_frames):
        # --- Check release ---
        if sim_time >= phase_release and not released:
            print(f"\n{'*'*60}")
            print(f"  [t={sim_time:.1f}s] *** RELEASING PAPER ***")
            flags = model.particle_flags.numpy()
            for idx in pinned_particles:
                flags[idx] = flags[idx] | ParticleFlags.ACTIVE
            model.particle_flags = wp.array(flags, dtype=wp.int32)
            released = True
            
            rest_angles = model.edge_rest_angle.numpy()
            damage_arr = plasticity.get_damage_numpy()
            num_creased = np.sum(np.abs(rest_angles) > 0.01)
            max_rest = np.max(np.abs(rest_angles))
            print(f"  Creased edges at release: {num_creased}/{num_edges}")
            print(f"  Max fold angle: {math.degrees(max_rest):.1f}°")
            print(f"{'*'*60}\n")
        
        # --- Simulate substeps ---
        for sub in range(sim_substeps):
            state_0.clear_forces()
            
            # Apply folding motion
            if not released and sim_time >= phase_fold_start:
                if sim_time < phase_fold_end:
                    t = (sim_time - phase_fold_start) / (phase_fold_end - phase_fold_start)
                    progress = 0.5 * (1.0 - math.cos(t * math.pi))
                else:
                    progress = 1.0
                
                fold_center_y = initial_positions[fold_row * num_verts_x, 1]
                fold_center_z = initial_positions[fold_row * num_verts_x, 2]
                angle = progress * math.pi
                
                q = state_0.particle_q.numpy()
                for idx in driven_particles:
                    pos = initial_positions[idx]
                    dy = pos[1] - fold_center_y
                    dz = pos[2] - fold_center_z
                    new_y = fold_center_y + dy * math.cos(angle) - dz * math.sin(angle)
                    new_z = fold_center_z + dy * math.sin(angle) + dz * math.cos(angle)
                    blend = 0.4
                    q[idx, 1] += (new_y - q[idx, 1]) * blend
                    q[idx, 2] += (new_z - q[idx, 2]) * blend
                state_0.particle_q = wp.array(q, dtype=wp.vec3)
            
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            
            # Plasticity step
            plasticity.step(state_1)
            
            state_0, state_1 = state_1, state_0
        
        sim_time += frame_dt
        
        # Print progress
        if frame % 60 == 0 and frame > 0:
            rest_angles = model.edge_rest_angle.numpy()
            damage_arr = plasticity.get_damage_numpy()
            num_creased = np.sum(np.abs(rest_angles) > 0.01)
            max_rest = np.max(np.abs(rest_angles))
            phase = "SETTLE" if sim_time < phase_fold_start else \
                    "FOLD" if sim_time < phase_fold_end else \
                    "HOLD" if sim_time < phase_release else "FREE"
            print(
                f"  [t={sim_time:.1f}s | {phase:6s}] "
                f"Creased: {num_creased:4d} | "
                f"Max fold: {math.degrees(max_rest):6.1f}° | "
                f"Damage: {np.sum(damage_arr):7.2f}"
            )
    
    # --- Final report ---
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    rest_angles = model.edge_rest_angle.numpy()
    damage_arr = plasticity.get_damage_numpy()
    num_creased = np.sum(np.abs(rest_angles) > 0.01)
    max_rest = np.max(np.abs(rest_angles))
    
    print(f"  Simulation time: {sim_time:.1f}s")
    print(f"  Total creased edges: {num_creased}/{num_edges}")
    print(f"  Max permanent fold angle: {math.degrees(max_rest):.1f}°")
    print(f"  Total damage: {np.sum(damage_arr):.2f}")
    
    if max_rest > 0.1:
        print(f"\n  ✅ SUCCESS: Paper retains visible crease ({math.degrees(max_rest):.1f}°)!")
    else:
        print(f"\n  ⚠️  Crease is small — may need parameter tuning")
    
    # Save rest angle data for analysis
    np.save("rest_angles_final.npy", rest_angles)
    np.save("damage_final.npy", damage_arr)
    print(f"\n  Saved: rest_angles_final.npy, damage_final.npy")
    print("=" * 60)


if __name__ == "__main__":
    frames = 600
    if len(sys.argv) > 1:
        frames = int(sys.argv[1])
    run_paper_fold(frames)
