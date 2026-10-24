"""
Mesh Importer Module
Loads 3D objects from STL, OBJ, and GLTF/GLB files for use in acoustic simulation.

Requires: pip install trimesh[easy]
"""

import numpy as np
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


class MeshImporter:
    """
    Loads 3D mesh files and prepares them for acoustic ray simulation.

    Supported formats: .stl, .obj, .gltf, .glb
    """

    SUPPORTED_FORMATS = {'.stl', '.obj', '.gltf', '.glb'}

    @staticmethod
    def load(filepath, scale=1.0):
        """
        Load a 3D mesh from a STL, OBJ, or GLTF file.

        Args:
            filepath: Path to mesh file (str or Path)
            scale: Scale factor applied uniformly to the mesh (meters per unit).
                   e.g., if your model is in mm, set scale=0.001

        Returns:
            MeshData object

        Raises:
            ImportError: If trimesh is not installed
            FileNotFoundError: If the file doesn't exist
            ValueError: If format is unsupported or file has no geometry
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError(
                "trimesh is required to import 3D objects.\n"
                "Install with: pip install trimesh[easy]"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        suffix = filepath.suffix.lower()
        if suffix not in MeshImporter.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {sorted(MeshImporter.SUPPORTED_FORMATS)}"
            )

        # Load the file — trimesh handles STL, OBJ, GLTF/GLB
        loaded = trimesh.load(str(filepath), force='mesh')

        # GLTF/GLB may return a Scene with multiple meshes
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                raise ValueError(f"No geometry found in: {filepath}")
            loaded = trimesh.util.concatenate(meshes)

        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Could not load a valid mesh from: {filepath}")

        # Apply scale
        if scale != 1.0:
            loaded.apply_scale(scale)

        return MeshData(loaded, name=filepath.stem)

    @staticmethod
    def supported_formats():
        """Return list of supported file format extensions."""
        return sorted(MeshImporter.SUPPORTED_FORMATS)


class MeshData:
    """
    Holds loaded mesh geometry and precomputed data for ray tracing.

    Attributes:
        name: Mesh name (derived from filename stem)
        vertices: (N, 3) array of vertex positions in mesh-local space
        faces: (M, 3) array of vertex index triplets
        bounds_min: (3,) minimum vertex coordinates (mesh-local)
        bounds_max: (3,) maximum vertex coordinates (mesh-local)
        center: (3,) centroid of bounding box (mesh-local)
        extents: (3,) width/depth/height of AABB
    """

    def __init__(self, trimesh_mesh, name="imported"):
        self.name = name
        self._mesh = trimesh_mesh

        self.vertices = np.array(trimesh_mesh.vertices, dtype=np.float64)
        self.faces = np.array(trimesh_mesh.faces, dtype=np.int64)
        self._face_normals = np.array(trimesh_mesh.face_normals, dtype=np.float64)

        # Bounding box in mesh-local space
        self.bounds_min = self.vertices.min(axis=0)
        self.bounds_max = self.vertices.max(axis=0)
        self.center = (self.bounds_min + self.bounds_max) / 2.0
        self.extents = self.bounds_max - self.bounds_min  # (lx, ly, lz)

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.faces)

    def get_triangles(self):
        """
        Return (M, 3, 3) array of triangle vertex positions in mesh-local space.
        Axis 0 = triangle index, Axis 1 = vertex (0/1/2), Axis 2 = coordinate (x/y/z).
        """
        return self.vertices[self.faces]

    def is_watertight(self):
        """True if the mesh is a closed, watertight volume."""
        return self._mesh.is_watertight

    def volume(self):
        """Approximate volume in cube of the mesh's units (only valid if watertight)."""
        if self._mesh.is_watertight:
            return float(self._mesh.volume)
        return None

    def __repr__(self):
        return (
            f"MeshData('{self.name}', vertices={self.num_vertices}, "
            f"faces={self.num_faces}, extents={self.extents.round(3)})"
        )
