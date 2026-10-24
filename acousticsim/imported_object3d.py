"""
Imported Object 3D Module
Extends Object3D to support externally loaded mesh geometry.

Provides two ray intersection modes:
    'aabb'  — Axis-Aligned Bounding Box (fast, approximate)
    'mesh'  — Per-triangle Möller–Trumbore (accurate, slower on large meshes)
"""

import numpy as np
from acousticsim.object3d import Object3D
from acousticsim.mesh_importer import MeshData


def _rotation_matrix_xyz(roll_deg, pitch_deg, yaw_deg):
    """
    Build a 3×3 rotation matrix from Euler angles (XYZ extrinsic convention).

    Args:
        roll_deg:  Rotation around X-axis in degrees
        pitch_deg: Rotation around Y-axis in degrees
        yaw_deg:   Rotation around Z-axis in degrees

    Returns:
        3×3 numpy rotation matrix R = Rz @ Ry @ Rx
    """
    rx, ry, rz = np.radians([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ])
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ])
    return Rz @ Ry @ Rx


class ImportedObject3D(Object3D):
    """
    Acoustic simulation object loaded from an external 3D mesh file.

    Inherits AABB intersection from Object3D.
    Adds optional per-triangle Möller–Trumbore intersection for accuracy.

    Args:
        mesh_data:           MeshData from MeshImporter.load()
        position:            (x, y, z) world-space center position [meters]
        rotation_deg:        (roll, pitch, yaw) Euler angles [degrees, XYZ convention]
        scale:               Additional uniform scale applied to mesh extents
        intersection_mode:  'aabb' (fast) or 'mesh' (accurate)
        reflection_coefficient: Acoustic energy fraction reflected (0–1)
        frequency_absorption:  {(f_min_khz, f_max_khz): absorption_fraction, ...}
        name:                Override for object name (defaults to mesh filename stem)
    """

    def __init__(
        self,
        mesh_data: MeshData,
        position,
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        intersection_mode='aabb',
        reflection_coefficient=0.85,
        frequency_absorption=None,
        name=None,
    ):
        if intersection_mode not in ('aabb', 'mesh'):
            raise ValueError("intersection_mode must be 'aabb' or 'mesh'")

        self.mesh_data = mesh_data
        self.rotation_deg = np.array(rotation_deg, dtype=float)
        self.intersection_mode = intersection_mode
        self._scale = float(scale)

        # AABB size is the scaled mesh extents
        aabb_size = mesh_data.extents * scale

        super().__init__(
            name=name or mesh_data.name,
            position=position,
            size=aabb_size,
            reflection_coefficient=reflection_coefficient,
            frequency_absorption=frequency_absorption or {},
        )

        # Rotation matrix (and its inverse = transpose for orthogonal R)
        self._R = _rotation_matrix_xyz(*rotation_deg)

        # Precompute world-space triangles for mesh mode
        self._world_triangles = None  # lazy or eager
        if intersection_mode == 'mesh':
            self._world_triangles = self._build_world_triangles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_rotation(self, roll_deg, pitch_deg, yaw_deg):
        """
        Update rotation and rebuild triangle cache if in mesh mode.

        Args:
            roll_deg, pitch_deg, yaw_deg: New Euler angles [degrees]
        """
        self.rotation_deg = np.array([roll_deg, pitch_deg, yaw_deg])
        self._R = _rotation_matrix_xyz(roll_deg, pitch_deg, yaw_deg)
        if self.intersection_mode == 'mesh':
            self._world_triangles = self._build_world_triangles()

    def set_position(self, position):
        """
        Move the object to a new world-space position and update bounds.

        Args:
            position: (x, y, z) new position [meters]
        """
        self.position = np.array(position, dtype=float)
        # Recompute AABB bounds (kept axis-aligned even for rotated mesh)
        self.min_bounds = self.position - self.size / 2
        self.max_bounds = self.position + self.size / 2
        if self.intersection_mode == 'mesh':
            self._world_triangles = self._build_world_triangles()

    def get_pose(self):
        """
        Return the object's pose as a flat array for ML labeling.

        Returns:
            (6,) array: [x, y, z, roll_deg, pitch_deg, yaw_deg]
        """
        return np.concatenate([self.position, self.rotation_deg])

    # ------------------------------------------------------------------
    # Ray intersection
    # ------------------------------------------------------------------

    def ray_intersect(self, ray_origin, ray_direction, max_distance=np.inf):
        """
        Ray intersection dispatch: AABB or per-triangle mesh.

        Returns:
            (hit, distance, hit_point, normal)
        """
        if self.intersection_mode == 'aabb':
            return super().ray_intersect(ray_origin, ray_direction, max_distance)
        return self._intersect_mesh(ray_origin, ray_direction, max_distance)

    # ------------------------------------------------------------------
    # Internal: mesh transformation
    # ------------------------------------------------------------------

    def _build_world_triangles(self):
        """
        Transform mesh triangles from local→world space.

        Pipeline:
            1. Center mesh at local origin
            2. Scale
            3. Rotate (R)
            4. Translate to world position

        Returns:
            (M, 3, 3) float64 array of world-space triangle vertices
        """
        local_tris = self.mesh_data.get_triangles()  # (M, 3, 3)

        # Center mesh at local origin
        centered = local_tris - self.mesh_data.center  # broadcast over (M,3)

        # Scale
        scaled = centered * self._scale  # (M, 3, 3)

        # Rotate each vertex: v_rot = R @ v  → einsum over vertices/coords
        # 'ij, mnj -> mni' : R[i,j] * tri[m,n,j] → tri_rot[m,n,i]
        rotated = np.einsum('ij,mnj->mni', self._R, scaled)

        # Translate to world position (broadcast: position is (3,))
        world = rotated + self.position  # (M, 3, 3)

        return world

    # ------------------------------------------------------------------
    # Internal: Möller–Trumbore intersection (vectorized)
    # ------------------------------------------------------------------

    def _intersect_mesh(self, ray_origin, ray_direction, max_distance):
        """
        Vectorized Möller–Trumbore algorithm against all triangles.

        Returns the closest triangle hit.

        Args:
            ray_origin:    (3,) array
            ray_direction: (3,) array (will be normalized)
            max_distance:  float

        Returns:
            (hit: bool, distance: float, hit_point: array or None, normal: array or None)
        """
        if self._world_triangles is None:
            self._world_triangles = self._build_world_triangles()

        origin = np.array(ray_origin, dtype=np.float64)
        d = np.array(ray_direction, dtype=np.float64)
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-12:
            return False, np.inf, None, None
        d = d / d_norm

        tris = self._world_triangles          # (M, 3, 3)
        v0 = tris[:, 0]                       # (M, 3)
        v1 = tris[:, 1]
        v2 = tris[:, 2]

        EPSILON = 1e-8

        edge1 = v1 - v0                       # (M, 3)
        edge2 = v2 - v0                       # (M, 3)

        # h = d × edge2
        h = np.cross(d[np.newaxis, :], edge2)  # (M, 3)
        a = np.einsum('ij,ij->i', edge1, h)    # (M,) dot products

        # Mask parallel rays
        valid = np.abs(a) > EPSILON
        f = np.where(valid, 1.0 / a, 0.0)

        # Barycentric u
        s = origin[np.newaxis, :] - v0         # (M, 3)
        u = f * np.einsum('ij,ij->i', s, h)    # (M,)
        valid &= (u >= 0.0) & (u <= 1.0)

        # Barycentric v
        q = np.cross(s, edge1)                 # (M, 3)
        v_bary = f * np.einsum('ij,ij->i', d[np.newaxis, :], q)  # (M,)
        valid &= (v_bary >= 0.0) & (u + v_bary <= 1.0)

        # Ray parameter t
        t = f * np.einsum('ij,ij->i', edge2, q)  # (M,)
        valid &= (t > EPSILON) & (t < max_distance)

        if not np.any(valid):
            return False, np.inf, None, None

        # Closest valid hit
        t_masked = np.where(valid, t, np.inf)
        idx = int(np.argmin(t_masked))
        t_hit = t_masked[idx]

        hit_point = origin + d * t_hit

        # Face normal
        e1_hit = edge1[idx]
        e2_hit = edge2[idx]
        normal = np.cross(e1_hit, e2_hit)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-12:
            normal /= n_len
        else:
            normal = np.array([0.0, 0.0, 1.0])

        # Ensure normal opposes the ray
        if np.dot(normal, d) > 0.0:
            normal = -normal

        return True, float(t_hit), hit_point, normal

    def __repr__(self):
        return (
            f"ImportedObject3D('{self.name}', pos={self.position.round(2)}, "
            f"rot={self.rotation_deg}, mode='{self.intersection_mode}', "
            f"faces={self.mesh_data.num_faces})"
        )
