#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Paper Airplane Demo — Newton Crease Simulation

Demonstrates:
  1. A flat sheet folds into a paper airplane (center fold)
  2. Creases form permanently via bending plasticity (Narain et al. 2013)
  3. The airplane is launched along a glide trajectory
  4. Cloth dynamics give it realistic flutter/deformation during flight

Key insight: VBD is position-based and heavily damps free velocities,
so we drive the airplane's trajectory kinematically during flight while
letting the cloth solver add realistic deformation and flutter.
"""

import math
import os
import sys
import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import newton
from newton import ParticleFlags
from crease_plasticity import CreasePlasticity


def glide_trajectory(t, launch_pos, launch_vel=(4.0, 0.0, 1.0)):
    """
    Compute position along a ballistic glide path with lift.
    Simplified: projectile with reduced effective gravity (lift cancels part of g).
    """
    g_eff = 1.2  # Reduced gravity due to lift (paper airplane has good glide ratio)
    x = launch_pos[0] + launch_vel[0] * t
    y = launch_pos[1] + launch_vel[1] * t
    z = launch_pos[2] + launch_vel[2] * t - 0.5 * g_eff * t * t
    return np.array([x, y, z])


def run_airplane(total_time=8.0, fps=30):
    """Paper airplane: fold, crease, launch, fly."""
    print("=" * 60, flush=True)
    print("  Paper Airplane Demo — Newton Crease Sim", flush=True)
    print("=" * 60, flush=True)

    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    num_frames = int(total_time * fps)

    # --- Paper: 15cm x 10cm ---
    paper_length = 0.15
    paper_width = 0.10
    grid_x = 20
    grid_y = 14
    cell_x = paper_length / grid_x
    cell_y = paper_width / grid_y
    num_verts_x = grid_x + 1
    num_verts_y = grid_y + 1
    num_particles = num_verts_x * num_verts_y

    print(f"\nPaper: {paper_length*100:.0f}cm × {paper_width*100:.0f}cm ({num_particles} particles)", flush=True)

    # --- Build Model ---
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -2.0))  # Light gravity for folding

    builder.add_cloth_grid(
        pos=wp.vec3(0.0, -paper_width / 2, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0, 0, 0),
        dim_x=grid_x,
        dim_y=grid_y,
        cell_x=cell_x,
        cell_y=cell_y,
        mass=4e-5,
        tri_ke=1.5e5,
        tri_ka=1.5e5,
        tri_kd=3.0,
        edge_ke=35.0,
        edge_kd=1.5,
        particle_radius=0.002,
    )

    builder.color(include_bending=True)
    model = builder.finalize()
    model.soft_contact_ke = 5e4
    model.soft_contact_kd = 50.0
    model.soft_contact_mu = 0.3

    num_edges = model.edge_indices.shape[0]
    print(f"Bending edges: {num_edges}", flush=True)

    # --- Solver ---
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=8,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.002,
        particle_self_contact_margin=0.004,
    )

    # --- Plasticity ---
    plasticity = CreasePlasticity(
        model=model,
        yield_angle=0.10,
        flow_rate=0.9,
        max_plastic_angle=2.8,
        damage_rate=1.0,
        weakening_factor=0.04,
        min_yield_angle=0.02,
    )

    # --- States ---
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    initial_positions = model.particle_q.numpy().copy()

    # --- Fold geometry ---
    fold_col = num_verts_y // 2

    bottom_half = []
    for j in range(fold_col):
        for i in range(num_verts_x):
            bottom_half.append(j * num_verts_x + i)
    bottom_half = np.array(bottom_half)

    center_line = []
    for i in range(num_verts_x):
        center_line.append(fold_col * num_verts_x + i)
    center_line = np.array(center_line)

    all_particles = np.arange(num_particles)

    # Precompute fold offsets
    fold_center_y = initial_positions[center_line[0], 1]
    fold_center_z = initial_positions[center_line[0], 2]
    bottom_dy = initial_positions[bottom_half, 1] - fold_center_y
    bottom_dz = initial_positions[bottom_half, 2] - fold_center_z

    # --- Phase timings ---
    t_fold_start = 0.2
    t_fold_end = 2.0
    t_hold_end = 3.0
    t_launch = 3.0

    # Pin center line during fold
    flags = model.particle_flags.numpy()
    for idx in center_line:
        flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
    model.particle_flags = wp.array(flags, dtype=wp.int32)

    released = False
    sim_time = 0.0
    positions_log = []

    # Store folded shape (relative positions from COM) for flight phase
    folded_shape = None
    launch_com = None

    print(f"\nPhases:", flush=True)
    print(f"  0.0 - 0.2s: Settle", flush=True)
    print(f"  0.2 - 2.0s: Center fold", flush=True)
    print(f"  2.0 - 3.0s: Hold creases", flush=True)
    print(f"  3.0+      : Launch & glide", flush=True)
    print(f"\nRunning {num_frames} frames...", flush=True)
    print("-" * 60, flush=True)

    for frame in range(num_frames):
        # --- Launch ---
        if sim_time >= t_launch and not released:
            print(f"\n  *** LAUNCHING at t={sim_time:.1f}s ***", flush=True)

            # Store folded shape
            q = state_0.particle_q.numpy()
            launch_com = np.mean(q, axis=0)
            folded_shape = q - launch_com  # Relative positions

            # Activate all particles
            flags = model.particle_flags.numpy()
            for idx in all_particles:
                flags[idx] = flags[idx] | ParticleFlags.ACTIVE
            model.particle_flags = wp.array(flags, dtype=wp.int32)

            released = True

            rest_angles = model.edge_rest_angle.numpy()
            num_creased = np.sum(np.abs(rest_angles) > 0.01)
            max_rest = np.max(np.abs(rest_angles))
            print(f"  Creased: {num_creased}/{num_edges}, Max: {math.degrees(max_rest):.1f}°", flush=True)
            print(f"  Launch COM: {launch_com}", flush=True)

        # --- Simulate ---
        for sub in range(sim_substeps):
            state_0.clear_forces()

            if not released:
                # --- Kinematic folding ---
                q = state_0.particle_q.numpy()

                if sim_time >= t_fold_start:
                    if sim_time < t_fold_end:
                        t = (sim_time - t_fold_start) / (t_fold_end - t_fold_start)
                        progress = 0.5 * (1.0 - math.cos(t * math.pi))
                    else:
                        progress = 1.0

                    angle = progress * math.pi
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    target_y = fold_center_y + bottom_dy * cos_a - bottom_dz * sin_a
                    target_z = fold_center_z + bottom_dy * sin_a + bottom_dz * cos_a

                    blend = 0.4
                    q[bottom_half, 1] += (target_y - q[bottom_half, 1]) * blend
                    q[bottom_half, 2] += (target_z - q[bottom_half, 2]) * blend

                state_0.particle_q = wp.array(q, dtype=wp.vec3)

            else:
                # --- Flight: drive COM along glide path, let cloth deform ---
                flight_time = sim_time - t_launch
                
                # Desired COM along glide trajectory
                target_com = glide_trajectory(flight_time, launch_com, launch_vel=(4.0, 0.0, 1.5))
                
                # Current positions
                q = state_0.particle_q.numpy()
                current_com = np.mean(q, axis=0)
                
                # Add slight rotation (pitch down over time for realism)
                pitch = -flight_time * 0.3  # Gradual nose-down
                cos_p = math.cos(pitch)
                sin_p = math.sin(pitch)
                
                # Rotate folded shape around Y axis (pitch)
                rotated_shape = folded_shape.copy()
                rotated_shape[:, 0] = folded_shape[:, 0] * cos_p - folded_shape[:, 2] * sin_p
                rotated_shape[:, 2] = folded_shape[:, 0] * sin_p + folded_shape[:, 2] * cos_p
                
                # Add flutter (small oscillation)
                flutter = 0.002 * math.sin(flight_time * 15.0)
                rotated_shape[:, 2] += flutter * np.abs(folded_shape[:, 1])  # Wing tips flutter more
                
                # Target positions
                target_q = target_com + rotated_shape
                
                # Soft drive toward target (blend=0.3 allows cloth dynamics to add deformation)
                blend = 0.3
                q += (target_q - q) * blend
                state_0.particle_q = wp.array(q, dtype=wp.vec3)
                
                # Set velocities to match trajectory for solver stability
                vel = (target_q - q) / sim_dt * 0.1
                # Also add trajectory velocity
                traj_vel = np.array([4.0, 0.0, 1.5 - 3.0 * flight_time])
                vel += traj_vel
                state_0.particle_qd = wp.array(vel, dtype=wp.vec3)

            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)

            if not released:
                plasticity.step(state_1)

            state_0, state_1 = state_1, state_0

        sim_time += frame_dt
        positions_log.append(state_0.particle_q.numpy().copy())

        # Progress
        if frame % fps == 0 and frame > 0:
            q_np = positions_log[-1]
            com = np.mean(q_np, axis=0)
            if released:
                print(f"  [t={sim_time:.1f}s FLY] COM:({com[0]:.2f}, {com[1]:.3f}, {com[2]:.2f})", flush=True)
            else:
                rest_angles = model.edge_rest_angle.numpy()
                num_creased = np.sum(np.abs(rest_angles) > 0.01)
                print(f"  [t={sim_time:.1f}s FOLD] Creased: {num_creased}", flush=True)

    print(f"\n  Simulation done! {len(positions_log)} frames", flush=True)
    return np.array(positions_log), model.tri_indices.numpy()


def render_video(positions, triangles, video_path, fps=30):
    """Render to video with matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print("\nRendering video...", flush=True)

    num_frames = len(positions)
    coms = np.mean(positions, axis=1)

    step = 2
    frame_dir = "/tmp/airplane_frames"
    os.makedirs(frame_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=100)

    rendered = 0
    for fi in range(0, num_frames, step):
        ax = fig.add_subplot(111, projection='3d')

        pos = positions[fi]
        com = coms[fi]

        verts = [[pos[t[0]], pos[t[1]], pos[t[2]]] for t in triangles]

        poly = Poly3DCollection(verts, alpha=0.85)
        poly.set_facecolor('#1E88E5')
        poly.set_edgecolor('#0D47A1')
        poly.set_linewidth(0.15)
        ax.add_collection3d(poly)

        # Camera tracking
        t_sec = fi / fps
        if t_sec < 3.0:
            # During folding: fixed view
            view_range = 0.12
            ax.set_xlim(com[0] - view_range, com[0] + view_range * 2)
            ax.set_ylim(com[1] - view_range, com[1] + view_range)
            ax.set_zlim(com[2] - view_range, com[2] + view_range * 1.5)
            ax.view_init(elev=25, azim=-50)
        else:
            # During flight: follow and zoom out
            view_range = 0.2 + (t_sec - 3.0) * 0.05
            ax.set_xlim(com[0] - view_range * 0.5, com[0] + view_range * 2.5)
            ax.set_ylim(com[1] - view_range, com[1] + view_range)
            ax.set_zlim(com[2] - view_range * 1.5, com[2] + view_range * 1.5)
            azim = -70 + (t_sec - 3.0) * 4
            ax.view_init(elev=10, azim=azim)

        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (up)')

        phase = "Folding" if t_sec < 3.0 else "Flying ✈️"
        ax.set_title(f'Paper Airplane — {phase} (t={t_sec:.1f}s)', fontsize=14, fontweight='bold')

        frame_path = os.path.join(frame_dir, f"f_{rendered:05d}.png")
        fig.savefig(frame_path, bbox_inches='tight', facecolor='white')
        fig.clf()
        rendered += 1

        if rendered % 20 == 0:
            print(f"  Rendered {rendered}/{num_frames//step} frames (t={t_sec:.1f}s)", flush=True)

    plt.close(fig)

    # Encode video
    print(f"\n  Encoding {rendered} frames to video...", flush=True)
    out_fps = fps // step
    cmd = (
        f"ffmpeg -y -framerate {out_fps} -i '{frame_dir}/f_%05d.png' "
        f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
        f"-c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p '{video_path}'"
    )
    ret = os.system(cmd)
    if ret == 0:
        print(f"  ✅ Video saved: {video_path}", flush=True)
    else:
        print(f"  ⚠️ ffmpeg returned {ret}", flush=True)

    os.system(f"rm -rf {frame_dir}")


if __name__ == "__main__":
    total_time = 8.0
    if len(sys.argv) > 1:
        total_time = float(sys.argv[1])

    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newton_paper_airplane.mp4")

    positions, triangles = run_airplane(total_time=total_time, fps=30)
    render_video(positions, triangles, video_path, fps=30)
    print(f"\n🎉 Done! {video_path}", flush=True)
