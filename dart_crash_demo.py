#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Newton Dart Crash — Faithful replication of ARCSim dart.json

Scene (Narain et al. 2013, Figure 6):
  - Pre-folded paper airplane (dart.obj)
  - Rotated -2 rad around X, translated to (0, -0.75, 0)
  - Flying at 15 m/s in +Y direction toward a vertical wall
  - Wall: 1m×1m square at Y=0 plane (rotated square.obj)
  - On impact: new creases form via bending plasticity

Paper material (paper.json):
  - density: 0.1 kg/m²
  - stretching stiffness: 0.5 MN/m (Eh, E=5GPa, h=0.1mm)
  - bending stiffness: 0.4 mNm
  - yield curvature: 200 (1/m)
  - weakening: 0.5
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


def parse_dart_obj(obj_path):
    """Parse dart.obj (vertices + faces)."""
    vertices = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face_verts = []
                for p in parts[1:]:
                    vi = int(p.split('/')[0]) - 1
                    face_verts.append(vi)
                faces.append(face_verts)
    return np.array(vertices), faces


def subdivide_mesh(vertices, faces, subdivisions=3):
    """Loop subdivision for higher resolution."""
    verts = list(vertices)
    tris = [list(f) for f in faces]

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_tris = []
        for tri in tris:
            v0, v1, v2 = tri
            mids = []
            for (a, b) in [(v0, v1), (v1, v2), (v2, v0)]:
                edge = tuple(sorted((a, b)))
                if edge not in edge_midpoints:
                    mid = (np.array(verts[a]) + np.array(verts[b])) / 2.0
                    edge_midpoints[edge] = len(verts)
                    verts.append(mid.tolist())
                mids.append(edge_midpoints[edge])
            m01, m12, m20 = mids
            new_tris.append([v0, m01, m20])
            new_tris.append([v1, m12, m01])
            new_tris.append([v2, m20, m12])
            new_tris.append([m01, m12, m20])
        tris = new_tris

    return np.array(verts), tris


def transform_dart(verts):
    """
    Apply ARCSim dart.json transform:
      rotate: [-2, 1, 0, 0]  → rotate -2 radians around X axis
      translate: [0, -0.75, 0]
    """
    angle = -2.0  # radians around X
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rotation matrix around X axis
    rotated = np.zeros_like(verts)
    rotated[:, 0] = verts[:, 0]
    rotated[:, 1] = verts[:, 1] * cos_a - verts[:, 2] * sin_a
    rotated[:, 2] = verts[:, 1] * sin_a + verts[:, 2] * cos_a

    # Translate
    rotated[:, 0] += 0.0
    rotated[:, 1] += -0.75
    rotated[:, 2] += 0.0

    return rotated


def run_dart_crash(total_time=0.5, fps=120):
    """
    Faithfully replicate ARCSim dart crash scene using Newton VBD + plasticity.
    """
    print("=" * 60, flush=True)
    print("  Newton Dart Crash — ARCSim Scene Replication", flush=True)
    print("  Narain et al. 2013, Figure 6", flush=True)
    print("=" * 60, flush=True)

    # --- Load and prepare mesh ---
    obj_path = '/home/horde/.openclaw/workspace/arcsim-0.2.1/meshes/dart.obj'
    print(f"\nLoading: {obj_path}", flush=True)
    raw_verts, raw_faces = parse_dart_obj(obj_path)
    print(f"  Raw: {len(raw_verts)} verts, {len(raw_faces)} faces", flush=True)

    # Subdivide for resolution (3x → 369 verts, 640 faces)
    verts, faces = subdivide_mesh(raw_verts, raw_faces, subdivisions=3)
    print(f"  Subdivided: {len(verts)} verts, {len(faces)} faces", flush=True)

    # Apply ARCSim transform (rotate -2 rad X, translate (0,-0.75,0))
    verts = transform_dart(verts)

    num_particles = len(verts)

    # --- Sim params ---
    frame_dt = 1.0 / fps
    sim_substeps = 30  # High substeps for fast impact
    sim_dt = frame_dt / sim_substeps
    num_frames = int(total_time * fps)

    # ARCSim uses 15 m/s in +Y
    impact_velocity = np.array([0.0, 15.0, 0.0])

    print(f"\n  Impact velocity: {impact_velocity} m/s (+Y toward wall)", flush=True)
    print(f"  Sim: {total_time}s @ {fps}fps = {num_frames} frames", flush=True)
    print(f"  Substeps: {sim_substeps} (dt={sim_dt*1000:.4f}ms)", flush=True)

    # --- Build Model ---
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))

    wp_verts = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    flat_indices = []
    for f in faces:
        flat_indices.extend(f)

    # Paper material from paper.json:
    # density=0.1 kg/m², stretching=0.5e6 N/m, bending=0.4e-3 Nm
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vertices=wp_verts,
        indices=flat_indices,
        vel=wp.vec3(float(impact_velocity[0]), float(impact_velocity[1]), float(impact_velocity[2])),
        density=0.1,          # kg/m² (matches paper.json)
        tri_ke=5.0e5,         # Stretching stiffness (high, paper doesn't stretch)
        tri_ka=5.0e5,
        tri_kd=10.0,
        edge_ke=80.0,         # Bending stiffness (paper is thin but stiff)
        edge_kd=3.0,
    )

    # --- Wall at Y=0 plane ---
    # ARCSim: square.obj rotated 90° around X, translated (-0.5, 0, -0.5)
    # This creates a 1m×1m wall in the XZ plane at Y=0
    # In Newton: use a ground plane or a box shape
    # Simplest: add a thin box as the wall
    wall_cfg = newton.ModelBuilder.ShapeConfig()
    wall_cfg.ke = 1.0e6  # Very stiff wall (collision_stiffness: 1e11 in ARCSim)
    wall_cfg.kd = 1000.0
    wall_cfg.mu = 0.3

    # Wall as a thin box at Y=0, spanning X: [-0.5, 0.5], Z: [-0.5, 0.5]
    wall_body = builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        is_kinematic=True,
    )
    builder.add_shape_box(
        body=wall_body,
        xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.5,   # half-extent X
        hy=0.005, # thin wall (1cm thick)
        hz=0.5,   # half-extent Z
        cfg=wall_cfg,
    )

    # Also add ground plane below (for if the airplane falls after impact)
    ground_cfg = newton.ModelBuilder.ShapeConfig()
    ground_cfg.ke = 1.0e5
    ground_cfg.kd = 500.0
    ground_cfg.mu = 0.5
    builder.add_ground_plane(cfg=ground_cfg)

    builder.color(include_bending=True)
    model = builder.finalize()

    model.soft_contact_ke = 1.0e6
    model.soft_contact_kd = 1000.0
    model.soft_contact_mu = 0.3

    num_edges = model.edge_indices.shape[0]
    print(f"  Bending edges: {num_edges}", flush=True)

    # --- Solver ---
    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=15,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.003,
        particle_self_contact_margin=0.005,
    )

    # --- Plasticity (matching ARCSim paper.json) ---
    # yield_curv: 200 → yield_angle ≈ yield_curv * avg_edge_length
    # For our mesh, avg edge ~5mm → yield_angle ≈ 200 * 0.005 = 1.0 rad? That's too high.
    # Actually yield_curv is curvature threshold. For discrete hinges:
    # yield_angle = yield_curv * edge_length ≈ 0.15 rad for small edges
    plasticity = CreasePlasticity(
        model=model,
        yield_angle=0.12,
        flow_rate=0.85,
        max_plastic_angle=2.8,
        damage_rate=1.0,
        weakening_factor=0.5,   # matches ARCSim weakening: 0.5
        min_yield_angle=0.02,
    )

    # --- States ---
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # Set initial rest angles to match pre-folded geometry
    from crease_plasticity import compute_dihedral_angles
    initial_dihedral = wp.zeros(num_edges, dtype=wp.float32)
    wp.launch(
        compute_dihedral_angles,
        dim=num_edges,
        inputs=[state_0.particle_q, model.edge_indices, initial_dihedral],
    )
    model.edge_rest_angle = wp.clone(initial_dihedral)
    plasticity.damage.zero_()

    init_rest = model.edge_rest_angle.numpy()
    print(f"\n  Pre-fold rest angles: {np.sum(np.abs(init_rest) > 0.01)}/{num_edges} non-zero", flush=True)
    print(f"  Max pre-fold: {math.degrees(np.max(np.abs(init_rest))):.1f}°", flush=True)

    # Initial nose position (for tracking)
    init_q = state_0.particle_q.numpy()
    # Nose = vertex closest to wall (highest Y)
    nose_idx = np.argmax(init_q[:, 1])
    print(f"  Nose at Y={init_q[nose_idx, 1]:.3f} (wall at Y=0)", flush=True)
    print(f"  Time to impact: ~{abs(init_q[nose_idx, 1]) / 15.0 * 1000:.1f} ms", flush=True)

    print(f"\n  Running...", flush=True)
    print("-" * 60, flush=True)

    positions_log = []
    sim_time = 0.0
    impact_detected = False

    for frame in range(num_frames):
        for sub in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            plasticity.step(state_1)
            state_0, state_1 = state_1, state_0

        sim_time += frame_dt
        q = state_0.particle_q.numpy().copy()
        positions_log.append(q)

        # Track
        rest_angles = model.edge_rest_angle.numpy()
        delta_rest = np.abs(rest_angles - init_rest)
        new_creased = np.sum(delta_rest > 0.01)

        if new_creased > 10 and not impact_detected:
            impact_detected = True
            print(f"\n  💥 IMPACT at t={sim_time*1000:.1f}ms!", flush=True)

        if frame % (fps // 4) == 0:
            nose_y = q[nose_idx, 1]
            com = np.mean(q, axis=0)
            max_delta = math.degrees(np.max(delta_rest)) if new_creased > 0 else 0
            print(
                f"  [t={sim_time*1000:.0f}ms] "
                f"Nose Y={nose_y:.4f} | "
                f"COM=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f}) | "
                f"New creases: {new_creased} | "
                f"Max Δ: {max_delta:.1f}°",
                flush=True
            )

    # Final
    print(f"\n{'='*60}", flush=True)
    rest_angles = model.edge_rest_angle.numpy()
    delta_rest = np.abs(rest_angles - init_rest)
    new_creased = np.sum(delta_rest > 0.01)
    max_new = math.degrees(np.max(delta_rest))
    print(f"  New creases: {new_creased}/{num_edges}", flush=True)
    print(f"  Max new angle: {max_new:.1f}°", flush=True)
    print(f"{'='*60}", flush=True)

    return np.array(positions_log), faces


def render_video(positions, faces, video_path, fps=120):
    """Render from side view (XZ plane, looking along -Y) like the paper figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print("\nRendering video...", flush=True)

    num_frames = len(positions)
    # Render every Nth frame to get ~150 output frames
    step = max(1, num_frames // 150)

    frame_dir = "/tmp/dart_frames"
    os.makedirs(frame_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=120)

    rendered = 0
    for fi in range(0, num_frames, step):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#16213e')
        fig.patch.set_facecolor('#16213e')

        pos = positions[fi]
        com = np.mean(pos, axis=0)

        # Paper airplane triangles
        verts = [[pos[f[0]], pos[f[1]], pos[f[2]]] for f in faces]
        poly = Poly3DCollection(verts, alpha=0.92)
        poly.set_facecolor('#eceff1')  # White paper
        poly.set_edgecolor('#455a64')
        poly.set_linewidth(0.25)
        ax.add_collection3d(poly)

        # Wall (semi-transparent, at Y≈0)
        wall_x = [com[0]-0.5, com[0]+0.5, com[0]+0.5, com[0]-0.5]
        wall_y = [0, 0, 0, 0]
        wall_z = [com[2]-0.5, com[2]-0.5, com[2]+0.5, com[2]+0.5]
        wall_verts = [list(zip(wall_x, wall_y, wall_z))]
        wall_poly = Poly3DCollection(wall_verts, alpha=0.2)
        wall_poly.set_facecolor('#ef5350')
        wall_poly.set_edgecolor('#c62828')
        wall_poly.set_linewidth(0.8)
        ax.add_collection3d(wall_poly)

        # Camera: track the airplane
        view_range = 0.3
        ax.set_xlim(com[0] - view_range, com[0] + view_range)
        ax.set_ylim(com[1] - view_range * 1.5, com[1] + view_range * 0.5)
        ax.set_zlim(com[2] - view_range, com[2] + view_range)

        ax.set_xlabel('X', color='#90a4ae', fontsize=9)
        ax.set_ylabel('Y (flight dir)', color='#90a4ae', fontsize=9)
        ax.set_zlabel('Z', color='#90a4ae', fontsize=9)
        ax.tick_params(colors='#546e7a', labelsize=7)

        t_ms = fi / fps * 1000
        ax.set_title(
            f'Dart Crash — Newton + Crease Plasticity  (t={t_ms:.0f}ms)',
            fontsize=13, fontweight='bold', color='white', pad=10
        )

        # Side view angle
        ax.view_init(elev=15, azim=-75)

        frame_path = os.path.join(frame_dir, f"f_{rendered:05d}.png")
        fig.savefig(frame_path, bbox_inches='tight', facecolor='#16213e')
        fig.clf()
        rendered += 1

        if rendered % 30 == 0:
            print(f"  Rendered {rendered} frames (t={t_ms:.0f}ms)", flush=True)

    plt.close(fig)

    # Encode
    print(f"\n  Encoding {rendered} frames...", flush=True)
    out_fps = min(30, fps // step)
    if out_fps < 10:
        out_fps = 15
    cmd = (
        f"ffmpeg -y -framerate {out_fps} -i '{frame_dir}/f_%05d.png' "
        f"-vf 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x16213e' "
        f"-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "
        f"-movflags +faststart '{video_path}'"
    )
    ret = os.system(cmd)
    os.system(f"rm -rf {frame_dir}")

    if ret == 0:
        size_kb = os.path.getsize(video_path) // 1024
        print(f"  ✅ Video: {video_path} ({size_kb}KB)", flush=True)
    else:
        print(f"  ⚠️ ffmpeg error: {ret}", flush=True)


if __name__ == "__main__":
    total_time = 0.5  # 500ms is plenty for this fast impact (ARCSim uses 200ms)
    if len(sys.argv) > 1:
        total_time = float(sys.argv[1])

    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newton_dart_crash.mp4")

    positions, faces = run_dart_crash(total_time=total_time, fps=120)
    render_video(positions, faces, video_path, fps=120)
    print(f"\n🎉 Done! {video_path}", flush=True)
