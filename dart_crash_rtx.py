#!/usr/bin/env python3
"""
Newton Dart Crash — v8 (USD unified scene)

scene.usda references dart.usda and supplies the wall, ground, collision
schemas, contact materials, gravity, and transforms. dart.usda supplies the
deformable cloth schemas, physical paper material, and display material.

Newton does not yet import cloth subdivision, initial velocity, damping,
area stiffness, or rest bend angles from USD; those remain runtime overrides.
"""

import argparse
import ctypes
import math
import os
import sys
from collections import deque
from time import perf_counter

# ViewerRTX uses pyglet/GLX via XWayland. On Wayland, PyOpenGL defaults to EGL
# and then imgui init fails with "no valid context". Must set before OpenGL import.
if "PYOPENGL_PLATFORM" not in os.environ:
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ["PYOPENGL_PLATFORM"] = "glx"


def patch_imgui_for_newton():
    """Newton 1.4 expects imgui-bundle>=1.92; Python 3.10 only has 1.5.x Linux wheels."""
    try:
        from imgui_bundle import imgui
    except ImportError:
        return
    if not hasattr(imgui.Col_, "nav_cursor") and hasattr(imgui.Col_, "nav_highlight"):
        imgui.Col_.nav_cursor = imgui.Col_.nav_highlight

    try:
        from newton._src.viewer.gl import gui as newton_gui
    except ImportError:
        return

    original_apply = newton_gui.UI._apply_dpi_scaling

    def _apply_dpi_scaling_compat(self):
        if not self.is_available:
            return
        self._setup_dark_style()
        style = self.imgui.get_style()
        if self.dpi_scale != 1.0:
            style.scale_all_sizes(self.dpi_scale)
        if hasattr(style, "font_scale_dpi"):
            style.font_scale_dpi = self.dpi_scale
        elif hasattr(self, "io") and hasattr(self.io, "font_global_scale"):
            self.io.font_global_scale = self.dpi_scale

    # Always install the compat wrapper so reinstalls of newton are covered by this script.
    if not getattr(newton_gui.UI, "_newton_dpi_patched", False):
        newton_gui.UI._apply_dpi_scaling = _apply_dpi_scaling_compat
        newton_gui.UI._newton_dpi_patched = True
        newton_gui.UI._newton_dpi_original = original_apply


import numpy as np
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import newton
import newton.viewer

patch_imgui_for_newton()

from crease_plasticity import CreasePlasticity, compute_dihedral_angles
from strain_limiting import StrainLimiter
from viewer_rtx_cuda import ViewerRTXCUDA

DART_ROOT_PATH = "/World/Dart"
DART_MESH_PATH = f"{DART_ROOT_PATH}/dart"
GROUND_PATH = "/World/StaticGeometry/Ground"
WALL_PATH = "/World/StaticGeometry/Wall"
# USD thickness drives mass/stiffness (areal density, membrane KE). Collision radius is
# overridden separately: thickness/2 = 1 mm tunnels through the wall at 10 m/s.
PAPER_THICKNESS = 0.002
PARTICLE_COLLISION_RADIUS = 0.008  # 8 mm: enough anti-tunnel, less stored spring energy
# Broadphase/narrowphase look-ahead. Default 1 cm misses most contacts at 10 m/s.
SOFT_CONTACT_MARGIN = 0.05
SIM_SUBSTEPS = 32  # ~5 mm travel per substep at 10 m/s
# Self-contact radius must be large enough that VBD's conservative bound allows
# flight speed: max_v ≈ 0.85 * radius / sim_dt. With 32 substeps, radius>=6.1 mm.
SELF_CONTACT_RADIUS = 0.01
SELF_CONTACT_MARGIN = 0.016
# Ignore pairs already this close in the folded rest pose (intentional layers).
SELF_CONTACT_REST_EXCLUSION = 0.03
VBD_ITERATIONS = 16
# Kill wall-exit velocity so the dart crushes instead of springing back.
WALL_SEPARATION_DAMP = 0.15
WALL_DAMP_ZONE_Y = -0.35  # particles with y > this and vy < 0 are leaving the wall


@wp.kernel
def damp_wall_separation(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    zone_y: float,
    damp: float,
):
    """After wall impact, dissipate velocity that carries the paper back away."""
    i = wp.tid()
    q = particle_q[i]
    v = particle_qd[i]
    if q[1] > zone_y and v[1] < 0.0:
        particle_qd[i] = wp.vec3(v[0] * 0.85, v[1] * damp, v[2] * 0.85)


def make_collision_pipeline(model):
    """Build a contact pipeline with enough look-ahead for high-speed impact."""
    return newton.CollisionPipeline(
        model,
        soft_contact_margin=SOFT_CONTACT_MARGIN,
    )


def load_follow_camera(path):
    """Load the follow-camera local position and target authored in dart.usda."""
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    camera = UsdGeom.Camera.Get(stage, "/root/dart/FollowCamera")
    if not camera:
        raise RuntimeError("Camera /root/dart/FollowCamera not found in dart.usda")

    matrix = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
    target = np.asarray(camera.GetPrim().GetAttribute("target").Get(), dtype=np.float64)
    return position, target


def camera_angles(position, target):
    """Return ViewerRTX Z-up pitch/yaw angles for a look-at camera."""
    direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-8:
        return 0.0, 0.0
    direction /= norm
    pitch = float(math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))))
    yaw = float(math.degrees(math.atan2(float(direction[1]), float(direction[0]))))
    return pitch, yaw


def apply_camera(viewer, position, pitch, yaw):
    """Apply a camera pose using plain Python floats (pyglet rejects numpy scalars)."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    viewer.set_camera(
        pos=wp.vec3(float(position[0]), float(position[1]), float(position[2])),
        pitch=float(pitch),
        yaw=float(yaw),
    )


def _is_transient_input_type_error(exc):
    """True for known pyglet/imgui typing glitches (often ctypes.ArgumentError)."""
    if isinstance(exc, (TypeError, ctypes.ArgumentError)):
        return True
    message = str(exc)
    return "wrong type" in message or "TypeError" in message


def harden_viewer_input(viewer):
    """Keep the RTX window alive through transient pyglet/imgui typing errors.

    ViewerRTX treats any exception from ``dispatch_events()`` as fatal and sets
    ``_should_close``. On Wayland/X11 those events often raise
    ``ctypes.ArgumentError: argument 2: TypeError: wrong type`` (not a Python
    TypeError), so we swallow only that class of glitch here.
    """
    window = getattr(viewer, "_window", None)
    if window is None or getattr(window, "_newton_safe_dispatch", False):
        return

    original_dispatch = window.dispatch_events

    def safe_dispatch_events():
        try:
            original_dispatch()
        except Exception as exc:
            if not _is_transient_input_type_error(exc):
                raise
            # Keep the window open; do not let ViewerRTX mark itself closed.

    window.dispatch_events = safe_dispatch_events
    window._newton_safe_dispatch = True

    gui = getattr(viewer, "gui", None)
    if gui is not None and not getattr(gui, "_newton_safe_camera_keys", False):
        original_update = gui.update_camera_from_keys

        def safe_update_camera_from_keys(dt, is_key_down):
            camera = getattr(viewer, "camera", None)
            if camera is not None:
                camera.pitch = float(camera.pitch)
                camera.yaw = float(camera.yaw)
            original_update(dt, is_key_down)
            if camera is not None:
                camera.pos = type(camera.pos)(
                    float(camera.pos.x),
                    float(camera.pos.y),
                    float(camera.pos.z),
                )
                camera.pitch = float(camera.pitch)
                camera.yaw = float(camera.yaw)

        gui.update_camera_from_keys = safe_update_camera_from_keys
        gui._newton_safe_camera_keys = True


def install_viewer_shortcuts(viewer, request_camera_toggle, request_reset, request_upload_toggle):
    """Register edge-triggered camera, reset, and mesh-upload shortcuts."""
    window = getattr(viewer, "_window", None)
    pyglet = getattr(viewer, "_pyglet", None)
    if window is None or pyglet is None or getattr(viewer, "_dart_shortcut_handlers", None):
        return False

    key_state = {"tab": False, "r": False, "u": False}

    def on_key_press(symbol, modifiers):
        del modifiers
        if symbol == pyglet.window.key.TAB and not key_state["tab"]:
            key_state["tab"] = True
            request_camera_toggle()
        elif symbol == pyglet.window.key.R and not key_state["r"]:
            key_state["r"] = True
            request_reset()
        elif symbol == pyglet.window.key.U and not key_state["u"]:
            key_state["u"] = True
            request_upload_toggle()

    def on_key_release(symbol, modifiers):
        del modifiers
        if symbol == pyglet.window.key.TAB:
            key_state["tab"] = False
        elif symbol == pyglet.window.key.R:
            key_state["r"] = False
        elif symbol == pyglet.window.key.U:
            key_state["u"] = False

    window.push_handlers(on_key_press=on_key_press, on_key_release=on_key_release)
    viewer._dart_shortcut_handlers = (on_key_press, on_key_release)
    return True


def bind_paper_material(viewer, dart_usd_path):
    """Recreate the Paper material from dart.usda on ViewerRTX's dynamic mesh."""
    source_stage = Usd.Stage.Open(dart_usd_path)
    source_shader = UsdShade.Shader.Get(source_stage, "/root/Looks/Paper/PreviewSurface")
    if not source_shader:
        raise RuntimeError("Paper material not found in dart.usda")

    def get_input(name, default):
        shader_input = source_shader.GetInput(name)
        value = shader_input.Get() if shader_input else None
        return default if value is None else value

    diffuse = get_input("diffuseColor", (0.88, 0.86, 0.80))
    roughness = float(get_input("roughness", 1.0))
    metallic = float(get_input("metallic", 0.0))
    specular = get_input("specularColor", (0.01, 0.01, 0.01))

    viewer.stage.DefinePrim("/root/Looks", "Scope")
    material = UsdShade.Material.Define(viewer.stage, "/root/Looks/Paper")
    surface = UsdShade.Shader.Define(viewer.stage, "/root/Looks/Paper/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(value) for value in diffuse])
    )
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    surface.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(value) for value in specular])
    )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    target_mesh = viewer.stage.GetPrimAtPath("/root/dart")
    UsdShade.MaterialBindingAPI.Apply(target_mesh).Bind(material)


def triangulate_usd_mesh(mesh):
    """Return the mesh's local points and fan-triangulated faces."""
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    offset = 0
    for count in counts:
        polygon = indices[offset : offset + count]
        faces.extend(
            [polygon[0], polygon[index], polygon[index + 1]]
            for index in range(1, count - 1)
        )
        offset += count
    return vertices, faces


def subdivide(vertices, faces, n=3):
    verts = list(vertices)
    tris = [list(f) for f in faces]
    for _ in range(n):
        edge_mp = {}
        new_tris = []
        for tri in tris:
            v0, v1, v2 = tri
            mids = []
            for (a, b) in [(v0, v1), (v1, v2), (v2, v0)]:
                e = tuple(sorted((a, b)))
                if e not in edge_mp:
                    edge_mp[e] = len(verts)
                    verts.append(((np.array(verts[a]) + np.array(verts[b])) / 2).tolist())
                mids.append(edge_mp[e])
            m01, m12, m20 = mids
            new_tris += [[v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20]]
        tris = new_tris
    return np.array(verts), tris


def prepare_scene_stage(scene_path, subdivisions=3):
    """Compose scene.usda and refine its referenced cloth cage in memory."""
    # Reload the root layer so repeated imports in one process do not keep
    # subdividing an already-refined in-memory USD stage.
    layer = Sdf.Layer.FindOrOpen(scene_path)
    if layer is None:
        raise RuntimeError(f"Failed to open USD layer: {scene_path}")
    layer.Reload()
    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {scene_path}")

    mesh = UsdGeom.Mesh.Get(stage, DART_MESH_PATH)
    if not mesh:
        raise RuntimeError(f"Cloth mesh not found at {DART_MESH_PATH}")

    raw_vertices, raw_faces = triangulate_usd_mesh(mesh)
    vertices, faces = subdivide(raw_vertices, raw_faces, subdivisions)

    # Newton imports the authored polygon cage and does not evaluate subdivision.
    # Author an in-memory override so both rigid geometry and cloth still originate
    # from scene.usda/dart.usda, while retaining the 369-particle simulation mesh.
    mesh.GetPointsAttr().Set([Gf.Vec3f(*vertex) for vertex in vertices])
    mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.GetFaceVertexIndicesAttr().Set([index for face in faces for index in face])
    return stage, raw_vertices, vertices, faces


def build_model_from_scene(scene_path, flight_velocity):
    """Import static colliders and the deformable dart from one composed USD stage."""
    stage, raw_vertices, local_vertices, faces = prepare_scene_stage(scene_path)
    builder = newton.ModelBuilder()
    import_result = builder.add_usd(stage, return_deformable_results=True)

    cloth = import_result["path_cloth_map"].get(DART_MESH_PATH)
    if cloth is None:
        raise RuntimeError(f"Newton did not import cloth prim {DART_MESH_PATH}")
    if GROUND_PATH not in import_result["path_shape_map"]:
        raise RuntimeError(f"Newton did not import ground collider {GROUND_PATH}")
    if WALL_PATH not in import_result["path_shape_map"]:
        raise RuntimeError(f"Newton did not import wall collider {WALL_PATH}")

    particle_start, particle_end = cloth["particle"]
    tri_start, tri_end = cloth["tri"]
    edge_start, edge_end = cloth["edge"]

    # USD deformable import currently drops initial velocity, area stiffness,
    # and damping. Keep only those unsupported solver-specific values here.
    for index in range(particle_start, particle_end):
        builder.particle_qd[index] = wp.vec3(0.0, flight_velocity, 0.0)
        # Collision spheres are enlarged for contact reliability. Keep USD thickness
        # for mass/stiffness conversion (do not change builder areal density).
        builder.particle_radius[index] = PARTICLE_COLLISION_RADIUS
    for index in range(tri_start, tri_end):
        tri_ke, _, _, drag, lift = builder.tri_materials[index]
        builder.tri_materials[index] = (tri_ke, 5.0e5, 10.0, drag, lift)
    for index in range(edge_start, edge_end):
        edge_ke, _ = builder.edge_bending_properties[index]
        # Keep authored fold stiffness; bend damping kills post-impact ringing.
        builder.edge_bending_properties[index] = (edge_ke, 35.0)

    builder.color(include_bending=True)
    model = builder.finalize()
    # Hard, highly dissipative wall/ground contact: crush and stick, don't spring off.
    model.soft_contact_ke = 4.0e5
    model.soft_contact_kd = 6.0e5
    model.soft_contact_kf = 5.0e5
    model.soft_contact_mu = 2.0
    model.soft_contact_restitution = 0.0
    model.particle_mu = 2.0

    radii = model.particle_radius.numpy()[particle_start:particle_end]
    if not np.allclose(radii, PARTICLE_COLLISION_RADIUS, atol=1.0e-7):
        raise RuntimeError(
            f"Particle collision radius is {radii.min():g}..{radii.max():g}; "
            f"expected {PARTICLE_COLLISION_RADIUS:g}"
        )

    tri_flat = np.asarray([index for face in faces for index in face], dtype=np.int32)
    return model, stage, raw_vertices, local_vertices, faces, tri_flat, import_result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Newton deformable dart RTX demo.")
    parser.add_argument(
        "--mesh-upload",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Deforming-mesh upload path (press U at runtime to toggle).",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Disable per-frame PNG readback and video encoding for meaningful FPS measurements.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph simulation capture for A/B performance comparison.",
    )
    parser.add_argument(
        "--max-stretch",
        type=float,
        default=1.005,
        help="Maximum allowed edge stretch ratio (1.005 = 0.5%% elongation). Lower = stiffer.",
    )
    parser.add_argument(
        "--strain-iters",
        type=int,
        default=3,
        help="Number of strain limiting Jacobi iterations per substep.",
    )
    parser.add_argument(
        "--no-strain-limit",
        action="store_true",
        help="Disable strain limiting (revert to pure VBD behavior).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60, flush=True)
    print("  Newton Dart Crash — v8 (USD unified scene)", flush=True)
    print("=" * 60, flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.join(script_dir, "meshes")
    scene_path = os.path.join(meshes_dir, "scene.usda")
    dart_usd_path = os.path.join(meshes_dir, "dart.usda")
    follow_camera_local, follow_target_local = load_follow_camera(dart_usd_path)

    # --- Sim ---
    total_time = 4.0
    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = SIM_SUBSTEPS
    sim_dt = frame_dt / sim_substeps
    num_frames = int(total_time * fps)

    # Flight velocity: 10 m/s in +Y toward wall at Y=0
    flight_vel = 10.0

    (
        model,
        scene_stage,
        raw_v,
        _local_vertices,
        faces,
        tri_flat,
        import_result,
    ) = build_model_from_scene(scene_path, flight_vel)
    state_0 = model.state()
    verts = state_0.particle_q.numpy()
    num_particles = len(verts)
    com0 = np.mean(verts, axis=0)
    cloth_range = import_result["path_cloth_map"][DART_MESH_PATH]["particle"]
    particle_radius = model.particle_radius.numpy()[cloth_range[0]]
    travel_per_substep = flight_vel * sim_dt

    print(f"  Scene: {scene_path}", flush=True)
    print(f"  Mesh: {num_particles} verts, {len(faces)} faces", flush=True)
    print(
        f"  Collision radius: {particle_radius*1000:.1f} mm, "
        f"soft margin: {SOFT_CONTACT_MARGIN*1000:.0f} mm "
        f"(USD thickness {PAPER_THICKNESS*1000:.1f} mm for mass/stiffness only)",
        flush=True,
    )
    print(
        f"  Substeps: {sim_substeps} → travel/substep {travel_per_substep*1000:.1f} mm "
        f"(keep ≤ ~radius to limit tunneling)",
        flush=True,
    )
    print(
        f"  Contact: ke={model.soft_contact_ke:g} kd={model.soft_contact_kd:g} "
        f"kf={model.soft_contact_kf:g} mu={model.soft_contact_mu:g}",
        flush=True,
    )
    print(f"  COM: ({com0[0]:.3f}, {com0[1]:.3f}, {com0[2]:.3f})", flush=True)
    print(f"  Flight: {flight_vel} m/s in +Y, wall at Y=0", flush=True)
    print(f"  Time to wall: ~{abs(com0[1])/flight_vel:.2f}s", flush=True)
    print(f"  Frames: {num_frames} ({total_time}s)", flush=True)

    num_edges = model.edge_indices.shape[0]
    print(f"  Edges: {num_edges}", flush=True)

    # --- KEY: Set rest angles to match folded dart geometry ---
    state_1 = model.state()
    control = model.control()
    collision_pipeline = make_collision_pipeline(model)
    contacts = model.contacts(collision_pipeline=collision_pipeline)

    initial_dihedral = wp.zeros(num_edges, dtype=wp.float32)
    wp.launch(compute_dihedral_angles, dim=num_edges,
              inputs=[state_0.particle_q, model.edge_indices, initial_dihedral])
    model.edge_rest_angle = wp.clone(initial_dihedral)
    initial_rest_angles = wp.clone(initial_dihedral)
    initial_state = model.state()
    initial_state.assign(state_0)

    init_rest = initial_dihedral.numpy()
    non_zero = np.sum(np.abs(init_rest) > 0.01)
    max_angle = math.degrees(np.max(np.abs(init_rest)))
    print(f"  Rest angles set from folded geometry: {non_zero}/{num_edges} non-zero", flush=True)
    print(f"  Max fold angle: {max_angle:.1f}°", flush=True)
    print(f"  ✅ VBD will maintain dart shape as equilibrium!", flush=True)

    # --- Solver ---
    def make_solver():
        # Self-contact with rest-pose exclusion: intentional folded layers stay free,
        # but crumple-time sheet-through-sheet contacts are blocked.
        return newton.solvers.SolverVBD(
            model=model,
            iterations=VBD_ITERATIONS,
            particle_enable_self_contact=True,
            particle_self_contact_radius=SELF_CONTACT_RADIUS,
            particle_self_contact_margin=SELF_CONTACT_MARGIN,
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=SELF_CONTACT_REST_EXCLUSION,
            particle_collision_detection_interval=0,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=128,
        )

    solver = make_solver()

    # --- Plasticity: impact permanently creases and absorbs rebound energy.
    plasticity = CreasePlasticity(
        model=model,
        yield_angle=0.12,
        flow_rate=0.85,
        max_plastic_angle=2.5,
        damage_rate=0.6,
        weakening_factor=0.25,
        min_yield_angle=0.05,
    )
    n_protected = plasticity.protect_initial_folds(
        initial_rest_angles, threshold=0.25, protected_yield=0.45
    )
    print(
        f"  Elevated yield on {n_protected}/{num_edges} fold edges "
        f"(hard impact still plasticizes them)",
        flush=True,
    )

    # --- Strain Limiter (post-step edge projection) ---
    strain_limiter = None
    if not args.no_strain_limit:
        strain_limiter = StrainLimiter(
            model=model,
            max_stretch=args.max_stretch,
            iterations=args.strain_iters,
            velocity_damping=0.8,
        )
        print(
            f"  Strain limiting: ON (max_stretch={args.max_stretch}, "
            f"iters={args.strain_iters})",
            flush=True,
        )
    else:
        print("  Strain limiting: OFF", flush=True)

    print(
        f"  Self-contact: ON (r={SELF_CONTACT_RADIUS*1000:.1f} mm, "
        f"rest-exclude<{SELF_CONTACT_REST_EXCLUSION*1000:.0f} mm)",
        flush=True,
    )
    reset_state = {"requested": False}
    sim_graph = None

    def simulate_frame():
        """Advance one display frame; captured as one CUDA Graph when available."""
        nonlocal state_0, state_1
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts, collision_pipeline=collision_pipeline)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            # --- Strain limiting: enforce max edge stretch ---
            if strain_limiter is not None:
                strain_limiter.limit(state_1, dt=sim_dt)
            # Dissipate wall-exit velocity before plasticity (inelastic crush).
            wp.launch(
                damp_wall_separation,
                dim=model.particle_count,
                inputs=[
                    state_1.particle_q,
                    state_1.particle_qd,
                    WALL_DAMP_ZONE_Y,
                    WALL_SEPARATION_DAMP,
                ],
            )
            plasticity.step(state_1)
            state_0, state_1 = state_1, state_0

    def restore_simulation_arrays():
        state_0.assign(initial_state)
        state_1.assign(initial_state)
        model.edge_rest_angle.assign(initial_rest_angles)
        plasticity.reset()
        plasticity.dihedral_angles.zero_()

    def capture_simulation_graph():
        """Warm lazy buffers, restore initial data, then capture one full frame."""
        if args.no_cuda_graph or not model.device.is_cuda:
            restore_simulation_arrays()
            return None
        if sim_substeps % 2:
            raise RuntimeError("CUDA Graph capture currently requires an even number of simulation substeps")

        # VBD and collision lazily initialize buffers on their first use. Warm them
        # before capture, then restore all externally visible simulation state.
        restore_simulation_arrays()
        wp.synchronize_device(model.device)
        simulate_frame()
        wp.synchronize_device(model.device)
        restore_simulation_arrays()
        wp.synchronize_device(model.device)

        with wp.ScopedCapture(device=model.device) as capture:
            simulate_frame()

        # Capturing records work without advancing the restored simulation state.
        restore_simulation_arrays()
        wp.synchronize_device(model.device)
        return capture.graph

    def request_reset():
        reset_state["requested"] = True

    def reset_simulation():
        """Restore all dynamic and plastic state for a clean replay."""
        nonlocal solver, contacts, sim_graph
        contacts = model.contacts(collision_pipeline=collision_pipeline)
        solver = make_solver()
        sim_graph = capture_simulation_graph()
        reset_state["requested"] = False

    sim_graph = capture_simulation_graph()
    print(
        f"  CUDA Graph: {'ON' if sim_graph is not None else 'OFF'}"
        + (" (use --no-cuda-graph for A/B)" if sim_graph is not None else ""),
        flush=True,
    )

    # --- ViewerRTX ---
    print(f"\n  Init RTX...", flush=True)
    viewer = ViewerRTXCUDA(
        width=1280, height=720, headless=False, fps=fps,
        up_axis='Z', environment='studio', mesh_upload_mode=args.mesh_upload,
    )
    viewer.set_model(model)
    viewer.show_triangles = False
    viewer.show_ui = False
    viewer.set_reset_callback(request_reset)
    print(
        f"  Mesh upload: {viewer.mesh_upload_mode.upper()} "
        f"({'direct GPU' if viewer.mesh_upload_mode == 'cuda' else 'GPU→CPU→GPU'})"
        " (press U to toggle)",
        flush=True,
    )
    print(f"  Frame capture: {'OFF' if args.no_capture else 'ON'}", flush=True)

    # Camera 0: right-rear elevated overview. Camera 1: dart.usda follow camera.
    fixed_position = np.array((0.9, -5.8, 1.8), dtype=np.float64)
    fixed_target = np.array((0.0, -2.3, 1.0), dtype=np.float64)
    fixed_pitch, fixed_yaw = camera_angles(fixed_position, fixed_target)
    fixed_camera = {
        "position": fixed_position,
        "pitch": fixed_pitch,
        "yaw": fixed_yaw,
    }
    dart_matrix = UsdGeom.Xformable(
        scene_stage.GetPrimAtPath(DART_ROOT_PATH)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    follow_camera_world = np.asarray(
        dart_matrix.Transform(Gf.Vec3d(*follow_camera_local)), dtype=np.float64
    )
    follow_offset = follow_camera_world - com0
    nose_vertex = int(np.argmin(np.linalg.norm(raw_v - follow_target_local, axis=1)))
    camera_state = {"active": 0, "toggle_requested": False}
    upload_state = {"toggle_requested": False}

    def request_camera_toggle():
        camera_state["toggle_requested"] = True

    def request_upload_toggle():
        upload_state["toggle_requested"] = True

    apply_camera(
        viewer,
        fixed_camera["position"],
        fixed_camera["pitch"],
        fixed_camera["yaw"],
    )
    print("  Camera: Fixed (press Tab to switch)", flush=True)

    # --- Run ---
    print(f"\n  Running...", flush=True)
    print("-" * 60, flush=True)

    frame_dir = "/tmp/dart_rtx_frames"
    os.makedirs(frame_dir, exist_ok=True)
    wp_tri_indices = wp.array(tri_flat, dtype=wp.int32)

    sim_time = 0.0
    initial_rel = verts - com0  # For shape deviation check
    stopped_early = False
    first_render = True
    frame = 0
    frame_times = deque(maxlen=30)

    while frame < num_frames:
        # ESC / window close sets ViewerRTX._should_close; honor it immediately.
        if not viewer.is_running():
            stopped_early = True
            print("  Window closed — stopping simulation.", flush=True)
            break

        if reset_state["requested"]:
            reset_simulation()
            sim_time = 0.0
            frame = 0
            print("  Reset: dart relaunched from its initial state.", flush=True)

        frame_start = perf_counter()
        if sim_graph is not None:
            wp.capture_launch(sim_graph)
        else:
            simulate_frame()

        sim_time += frame_dt

        # Render
        viewer.begin_frame(sim_time)
        if not viewer.is_running():
            stopped_early = True
            print("  Window closed — stopping simulation.", flush=True)
            break

        reset_during_events = reset_state["requested"]
        if reset_during_events:
            reset_simulation()
            sim_time = 0.0
            frame = 0
            print("  Reset: dart relaunched from its initial state.", flush=True)

        if camera_state["toggle_requested"]:
            camera_state["toggle_requested"] = False
            camera_state["active"] = 1 - camera_state["active"]
            print(
                f"  Camera: {'Follow' if camera_state['active'] else 'Fixed'}",
                flush=True,
            )
            if camera_state["active"] == 0:
                apply_camera(
                    viewer,
                    fixed_camera["position"],
                    fixed_camera["pitch"],
                    fixed_camera["yaw"],
                )

        if upload_state["toggle_requested"]:
            upload_state["toggle_requested"] = False
            mode = viewer.toggle_mesh_upload_mode()
            frame_times.clear()
            print(
                f"  Mesh upload: {mode.upper()} "
                f"({'direct GPU' if mode == 'cuda' else 'GPU→CPU→GPU'})",
                flush=True,
            )

        if camera_state["active"] == 1:
            positions = state_0.particle_q.numpy()
            current_com = np.mean(positions, axis=0)
            follow_position = current_com + follow_offset
            follow_target = positions[nose_vertex]
            pitch, yaw = camera_angles(follow_position, follow_target)
            apply_camera(viewer, follow_position, pitch, yaw)

        viewer.log_state(state_0)
        viewer.log_mesh("dart", points=state_0.particle_q, indices=wp_tri_indices,
                       backface_culling=False)
        if first_render:
            bind_paper_material(viewer, dart_usd_path)
        viewer.end_frame()
        if first_render:
            first_render = False
            harden_viewer_input(viewer)
            if not install_viewer_shortcuts(
                viewer,
                request_camera_toggle,
                request_reset,
                request_upload_toggle,
            ):
                print("  Warning: could not install viewer keyboard shortcuts", flush=True)
        if not args.no_capture:
            viewer.save_screenshot(os.path.join(frame_dir, f"frame_{frame:04d}.png"))

        frame_times.append(perf_counter() - frame_start)

        # Progress
        if frame % 15 == 0:
            q = state_0.particle_q.numpy()
            com = np.mean(q, axis=0)
            # Shape deviation
            rel = q - com
            rms = np.sqrt(np.mean(np.sum((rel - initial_rel)**2, axis=1)))

            rest_angles = model.edge_rest_angle.numpy()
            delta = np.abs(rest_angles - init_rest)
            new_creased = np.sum(delta > 0.01)
            soft_n = int(contacts.soft_contact_count.numpy()[0])
            measured_fps = len(frame_times) / sum(frame_times)
            max_s_info = ""
            if strain_limiter is not None:
                max_s_info = f" max_stretch:{strain_limiter.get_max_stretch_ratio(state_0):.4f}"

            print(
                f"  [t={sim_time:.2f}s] COM:({com[0]:.2f},{com[1]:.2f},{com[2]:.2f}) "
                f"shape_rms:{rms:.5f} soft:{soft_n} new_crease:{new_creased}{max_s_info} "
                f"FPS:{measured_fps:.1f} upload:{viewer.mesh_upload_ms:.3f}ms "
                f"[{viewer.mesh_upload_mode.upper()}]",
                flush=True,
            )

        if not reset_during_events:
            frame += 1

    viewer.close()

    if stopped_early:
        os.system(f"rm -rf {frame_dir}")
        print("\n  Stopped by user (ESC / window close).", flush=True)
        return

    if not args.no_capture:
        # Encode
        print(f"\n  Encoding...", flush=True)
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newton_dart_crash.mp4")
        cmd = (
            f"ffmpeg -y -framerate {fps} -i '{frame_dir}/frame_%04d.png' "
            f"-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "
            f"-movflags +faststart '{video_path}'"
        )
        os.system(cmd)
        os.system(f"rm -rf {frame_dir}")
        size_kb = os.path.getsize(video_path) // 1024
        print(f"  ✅ {video_path} ({size_kb}KB)", flush=True)

    rest_angles = model.edge_rest_angle.numpy()
    delta = np.abs(rest_angles - init_rest)
    new_creased = np.sum(delta > 0.01)
    print(f"\n  Final: {new_creased}/{num_edges} new creases", flush=True)
    print(f"🎉 Done!", flush=True)


if __name__ == "__main__":
    main()
