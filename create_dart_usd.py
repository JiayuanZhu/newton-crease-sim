#!/usr/bin/env python3
"""
Convert dart.obj to USD with crease edges defined.
The dart mesh has predefined crease edges (the 'e' lines in the OBJ).
In USD, these become subdivision creases that make the fold lines sharp.
"""

import numpy as np
from pxr import Usd, UsdGeom, Sdf, Vt, Gf

def parse_dart_obj(obj_path):
    """Parse dart.obj with crease edges."""
    vertices = []
    faces = []
    crease_edges = []
    crease_sharpness = []

    with open(obj_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if not parts:
            i += 1
            continue
        if parts[0] == 'v':
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == 'f':
            face_verts = []
            for p in parts[1:]:
                vi = int(p.split('/')[0]) - 1
                face_verts.append(vi)
            faces.append(face_verts)
        elif parts[0] == 'e':
            # Crease edge: e v1 v2
            v1 = int(parts[1]) - 1
            v2 = int(parts[2]) - 1
            crease_edges.append((v1, v2))
            # Next line should be 'el' (edge sharpness)
            if i + 1 < len(lines):
                next_parts = lines[i+1].strip().split()
                if next_parts and next_parts[0] == 'el':
                    crease_sharpness.append(float(next_parts[1]))
                    i += 1
                else:
                    crease_sharpness.append(10.0)  # Default high sharpness
        i += 1

    return np.array(vertices), faces, crease_edges, crease_sharpness


def create_dart_usd(obj_path, usd_path):
    """Create USD file from dart.obj with creases."""
    vertices, faces, crease_edges, crease_sharpness = parse_dart_obj(obj_path)

    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"Crease edges: {len(crease_edges)}")
    print(f"Crease sharpness: {crease_sharpness}")

    # Create USD stage
    stage = Usd.Stage.CreateNew(usd_path)
    stage.SetMetadata('upAxis', 'Z')
    stage.SetMetadata('metersPerUnit', 1.0)

    # Create root xform
    root = UsdGeom.Xform.Define(stage, '/root')

    # Create the dart mesh
    mesh = UsdGeom.Mesh.Define(stage, '/root/dart')

    # Set vertices (points)
    points = Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices])
    mesh.GetPointsAttr().Set(points)

    # Set face topology
    face_vertex_counts = Vt.IntArray([len(f) for f in faces])
    face_vertex_indices = Vt.IntArray([vi for f in faces for vi in f])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Set crease edges (USD subdivision creases)
    # creaseIndices: flat array of vertex pairs
    # creaseLengths: number of vertices per crease (2 for edges)
    # creaseSharpnesses: sharpness per crease
    if crease_edges:
        ci = []
        cl = []
        cs = []

        for (v1, v2), sharp in zip(crease_edges, crease_sharpness):
            ci.append(v1)
            ci.append(v2)
            cl.append(2)
            cs.append(min(sharp * 10.0, 10.0))

        mesh.GetCreaseIndicesAttr().Set(Vt.IntArray(ci))
        mesh.GetCreaseLengthsAttr().Set(Vt.IntArray(cl))
        mesh.GetCreaseSharpnessesAttr().Set(Vt.FloatArray(cs))

    # Set subdivision scheme to catmullClark so creases are respected
    mesh.GetSubdivisionSchemeAttr().Set('catmullClark')

    # Set display color (white paper)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.95, 0.95, 0.92)]))

    stage.GetRootLayer().Save()
    print(f"\nSaved: {usd_path}")
    print(f"  Subdivision: catmullClark")
    print(f"  Creases: {len(crease_edges)} edges with sharpness")


if __name__ == "__main__":
    obj_path = '/home/horde/.openclaw/workspace/arcsim-0.2.1/meshes/dart.obj'
    usd_path = '/home/horde/.openclaw/workspace/newton-crease-sim/dart.usda'
    create_dart_usd(obj_path, usd_path)
