"""CUDA deforming-mesh updates for Newton's OVRTX viewer."""

from __future__ import annotations

import numpy as np
import warp as wp
from time import perf_counter

from newton.viewer import ViewerRTX


class ViewerRTXCUDA(ViewerRTX):
    """ViewerRTX variant that sends deforming mesh vertices directly from CUDA."""

    def __init__(self, *args, mesh_upload_mode: str = "cuda", **kwargs):
        self._cuda_mesh_point_bindings = {}
        self._cuda_mesh_normal_bindings = {}
        self._cuda_meshes_with_normals = set()
        self.mesh_upload_ms = 0.0
        self.mesh_upload_mode = "cuda"
        self.set_mesh_upload_mode(mesh_upload_mode)
        super().__init__(*args, **kwargs)

    def set_mesh_upload_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in {"cpu", "cuda"}:
            raise ValueError(f"mesh upload mode must be 'cpu' or 'cuda', got {mode!r}")
        self.mesh_upload_mode = mode

    def toggle_mesh_upload_mode(self) -> str:
        self.set_mesh_upload_mode("cpu" if self.mesh_upload_mode == "cuda" else "cuda")
        return self.mesh_upload_mode

    def _release_cuda_mesh_bindings(self):
        for binding in (
            *self._cuda_mesh_point_bindings.values(),
            *self._cuda_mesh_normal_bindings.values(),
        ):
            binding.unbind()
        self._cuda_mesh_point_bindings.clear()
        self._cuda_mesh_normal_bindings.clear()

    def _init_ovrtx(self):
        super()._init_ovrtx()

        from ovrtx import BindingFlag, PrimMode

        for name, prim_path in self._mesh_prim_paths.items():
            self._cuda_mesh_point_bindings[name] = self._rtx.bind_array_attribute(
                prim_paths=[prim_path],
                attribute_name="points",
                dtype="float32",
                shape=(3,),
                prim_mode=PrimMode.MUST_EXIST,
                flags=BindingFlag.OPTIMIZE,
            )
            if name in self._cuda_meshes_with_normals:
                self._cuda_mesh_normal_bindings[name] = self._rtx.bind_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="normals",
                    dtype="float32",
                    shape=(3,),
                    prim_mode=PrimMode.MUST_EXIST,
                    flags=BindingFlag.OPTIMIZE,
                )

    def log_mesh(
        self,
        name: str,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ) -> None:
        if self._phase == self._PHASE_BUILD:
            if normals is not None:
                self._cuda_meshes_with_normals.add(self._qualify(name))
            super().log_mesh(
                name,
                points,
                indices,
                normals,
                uvs,
                texture,
                hidden,
                backface_culling,
                color=color,
                roughness=roughness,
                metallic=metallic,
            )
            return

        name = self._qualify(name)
        if name in self._mesh_prim_paths:
            self._pending_mesh_points[name] = points
            if normals is not None:
                self._pending_mesh_normals[name] = normals

    @staticmethod
    def _cuda_stream(array: wp.array) -> int | None:
        if isinstance(array, wp.array) and array.device.is_cuda:
            return array.device.stream.cuda_stream
        return None

    def _update_ovrtx_mesh_points(self):
        if self._rtx is None:
            return

        from ovrtx import DataAccess

        start = perf_counter()
        if self.mesh_upload_mode == "cuda":
            for name, points in self._pending_mesh_points.items():
                binding = self._cuda_mesh_point_bindings.get(name)
                if binding is None:
                    continue
                binding.write(
                    [points],
                    data_access=DataAccess.ASYNC,
                    cuda_stream=self._cuda_stream(points),
                )
            for name, normals in self._pending_mesh_normals.items():
                binding = self._cuda_mesh_normal_bindings.get(name)
                if binding is not None:
                    binding.write(
                        [normals],
                        data_access=DataAccess.ASYNC,
                        cuda_stream=self._cuda_stream(normals),
                    )
        else:
            for name, points in self._pending_mesh_points.items():
                prim_path = self._mesh_prim_paths.get(name)
                if prim_path is None:
                    continue
                points_np = (
                    points.numpy().astype(np.float32)
                    if isinstance(points, wp.array)
                    else np.asarray(points, dtype=np.float32)
                )
                self._rtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="points",
                    tensors=[self._make_point3f_dltensor(points_np)],
                )
            for name, normals in self._pending_mesh_normals.items():
                prim_path = self._mesh_prim_paths.get(name)
                if prim_path is None:
                    continue
                normals_np = (
                    normals.numpy().astype(np.float32)
                    if isinstance(normals, wp.array)
                    else np.asarray(normals, dtype=np.float32)
                )
                self._rtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="normals",
                    tensors=[self._make_point3f_dltensor(normals_np)],
                )
        self.mesh_upload_ms = (perf_counter() - start) * 1000.0

    def clear_model(self) -> None:
        self._release_cuda_mesh_bindings()
        self._cuda_meshes_with_normals.clear()
        super().clear_model()

    def close(self) -> None:
        self._release_cuda_mesh_bindings()
        super().close()
