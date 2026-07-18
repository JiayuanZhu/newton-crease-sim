# SPDX-License-Identifier: Apache-2.0
"""
Adaptive Remeshing Module for Crease Resolution

This module implements a simplified version of the adaptive remeshing from
Narain et al. 2013, adapted for Newton's particle-based representation.

The key idea: when a crease forms (high bending curvature concentrated on
a single edge), split that edge to better resolve the fold geometry. This
avoids "bend locking" where a coarse mesh cannot represent sharp folds.

Strategy:
  1. After plasticity update, identify edges with high curvature
  2. Split edges where the dihedral angle change is too sharp for the current resolution
  3. Interpolate positions, velocities, and rest angles for new vertices
  4. Update the Newton model's particle and triangle arrays

NOTE: Dynamic remeshing at runtime is complex in Newton's GPU-optimized pipeline.
This module provides the algorithmic framework. Full integration would require
Newton API support for dynamic topology changes (adding particles/triangles mid-sim).

For now, this serves as a reference implementation and can be used for:
  - Offline remeshing between simulation segments
  - Pre-refinement of meshes near expected crease locations
  - Analysis of where creases would benefit from refinement
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def compute_edge_curvature_metric(
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    edge_rest_angle: wp.array(dtype=wp.float32),
    edge_rest_length: wp.array(dtype=wp.float32),
    curvature_metric: wp.array(dtype=wp.float32),
):
    """
    Compute a sizing metric per edge based on curvature.
    High values indicate edges that should be split.

    Metric = |theta - theta_rest| / edge_length
    This approximates the curvature concentrated at this edge.
    """
    idx = wp.tid()

    vi0 = edge_indices[idx, 0]
    vi1 = edge_indices[idx, 1]
    vi2 = edge_indices[idx, 2]
    vi3 = edge_indices[idx, 3]

    if vi0 == -1 or vi1 == -1:
        curvature_metric[idx] = 0.0
        return

    x2 = pos[vi2]
    x3 = pos[vi3]
    edge_len = wp.length(x3 - x2)

    if edge_len < 1.0e-8:
        curvature_metric[idx] = 0.0
        return

    # Current dihedral angle
    x0 = pos[vi0]
    x1 = pos[vi1]
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    if n1_norm < 1.0e-8 or n2_norm < 1.0e-8 or e_norm < 1.0e-8:
        curvature_metric[idx] = 0.0
        return

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    rest = edge_rest_angle[idx]
    curvature = wp.abs(theta - rest) / edge_len
    curvature_metric[idx] = curvature


class AdaptiveCreaseRefinement:
    """
    Identifies edges that need refinement due to high crease curvature.

    This is a simplified version of ARCSim's dynamic_remesh() that focuses
    specifically on crease resolution. Full adaptive remeshing (with coarsening,
    aspect ratio optimization, etc.) is left for future work.

    Parameters:
        model: Newton Model instance
        max_curvature: Curvature threshold above which edges should be split
        min_edge_length: Minimum allowed edge length (prevents infinite refinement)
        max_edges: Maximum total number of edges (budget)
    """

    def __init__(
        self,
        model,
        max_curvature: float = 10.0,
        min_edge_length: float = 0.005,
        max_edges: int = 50000,
    ):
        self.model = model
        self.max_curvature = max_curvature
        self.min_edge_length = min_edge_length
        self.max_edges = max_edges

        self.num_edges = model.edge_indices.shape[0]
        self.curvature_metric = wp.zeros(self.num_edges, dtype=wp.float32)

    def compute_refinement_field(self, state) -> np.ndarray:
        """
        Compute per-edge curvature metric and return edges that need splitting.

        Returns:
            numpy array of edge indices that exceed the curvature threshold
        """
        wp.launch(
            compute_edge_curvature_metric,
            dim=self.num_edges,
            inputs=[
                state.particle_q,
                self.model.edge_indices,
                self.model.edge_rest_angle,
                self.model.edge_rest_length,
                self.curvature_metric,
            ],
        )

        metrics = self.curvature_metric.numpy()
        edges_to_split = np.where(metrics > self.max_curvature)[0]

        # Filter by minimum edge length
        if len(edges_to_split) > 0:
            edge_indices = self.model.edge_indices.numpy()
            positions = state.particle_q.numpy()

            valid = []
            for eidx in edges_to_split:
                vi2, vi3 = edge_indices[eidx, 2], edge_indices[eidx, 3]
                edge_len = np.linalg.norm(positions[vi3] - positions[vi2])
                if edge_len > self.min_edge_length * 2.0:  # Must be splittable
                    valid.append(eidx)
            edges_to_split = np.array(valid)

        return edges_to_split

    def get_curvature_numpy(self) -> np.ndarray:
        """Get the curvature metric array for visualization."""
        return self.curvature_metric.numpy()

    def suggest_refinement_report(self, state) -> str:
        """Generate a human-readable report of where refinement is needed."""
        edges = self.compute_refinement_field(state)
        metrics = self.curvature_metric.numpy()

        report = f"Adaptive Refinement Report\n"
        report += f"{'='*40}\n"
        report += f"Total edges: {self.num_edges}\n"
        report += f"Edges needing split: {len(edges)}\n"
        report += f"Max curvature: {np.max(metrics):.2f}\n"
        report += f"Mean curvature: {np.mean(metrics):.4f}\n"

        if len(edges) > 0:
            report += f"\nTop 10 edges to refine:\n"
            sorted_edges = edges[np.argsort(metrics[edges])[::-1]][:10]
            for i, eidx in enumerate(sorted_edges):
                report += f"  {i+1}. Edge {eidx}: curvature = {metrics[eidx]:.3f}\n"

        return report
