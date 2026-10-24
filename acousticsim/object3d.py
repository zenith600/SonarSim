"""
Object3D Module
Represents walls, reflectors, seabed, surface, and obstacles in 3D space.
"""

import numpy as np


class Object3D:
    """
    Represents a 3D object (wall, reflector, obstacle) in the water.
    
    Attributes:
        name: Object identifier
        position: (x, y, z) center position in meters
        size: (length, width, height) in meters
        reflection_coefficient: 0.0 (absorb) to 1.0 (perfect reflection)
        is_rigid: True for rigid surfaces (simple reflection)
    """
    
    def __init__(self, name, position, size, reflection_coefficient=0.9, is_rigid=True,
                 frequency_absorption=None):
        """
        Initialize 3D object.
        
        Args:
            name: Object name/identifier
            position: (x, y, z) center position
            size: (length, width, height) dimensions
            reflection_coefficient: Fraction of energy reflected (0-1) at reference freq
            is_rigid: If True, simple reflection; if False, can model refraction
            frequency_absorption: Dict mapping frequency ranges to absorption coefficients
                                 e.g., {(30, 50): 0.8, (50, 80): 0.5}
                                 Higher value = more absorption (less reflection)
        """
        self.name = name
        self.position = np.array(position, dtype=float)
        self.size = np.array(size, dtype=float)
        self.reflection_coefficient = reflection_coefficient
        self.is_rigid = is_rigid
        self.frequency_absorption = frequency_absorption or {}
        
        # Compute AABB (Axis-Aligned Bounding Box)
        self.min_bounds = self.position - self.size / 2
        self.max_bounds = self.position + self.size / 2
    
    def ray_intersect(self, ray_origin, ray_direction, max_distance=np.inf):
        """
        AABB ray-box intersection using slab method.
        
        Args:
            ray_origin: (x, y, z) starting point of ray
            ray_direction: (dx, dy, dz) normalized direction vector
            max_distance: Maximum distance to check
            
        Returns:
            (hit, distance, hit_point, normal) where:
                hit: True if intersection occurs
                distance: Distance to intersection
                hit_point: (x, y, z) intersection point
                normal: (nx, ny, nz) surface normal at intersection
        """
        # Normalize direction
        direction = np.array(ray_direction, dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        else:
            return False, np.inf, None, None
        
        origin = np.array(ray_origin, dtype=float)
        
        # Compute intersection with each slab
        t_min = -np.inf
        t_max = np.inf
        normal = np.zeros(3)
        
        for i in range(3):
            if abs(direction[i]) < 1e-8:
                # Ray parallel to slab
                if origin[i] < self.min_bounds[i] or origin[i] > self.max_bounds[i]:
                    return False, np.inf, None, None
            else:
                # Compute intersection distances
                t1 = (self.min_bounds[i] - origin[i]) / direction[i]
                t2 = (self.max_bounds[i] - origin[i]) / direction[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update t_min
                if t1 > t_min:
                    t_min = t1
                    # Determine which face we hit
                    normal = np.zeros(3)
                    if direction[i] > 0:
                        normal[i] = -1  # Hit min face from outside
                    else:
                        normal[i] = 1   # Hit max face from outside
                
                # Update t_max
                if t2 < t_max:
                    t_max = t2
                
                # Early exit if no intersection
                if t_min > t_max:
                    return False, np.inf, None, None
        
        # Check if intersection is in valid range
        if t_min < 0 or t_min > max_distance:
            return False, np.inf, None, None
        
        # Compute intersection point
        hit_point = origin + t_min * direction
        
        return True, t_min, hit_point, normal
    
    def get_reflection_coefficient(self, frequency_khz):
        """
        Get frequency-dependent reflection coefficient.
        
        Args:
            frequency_khz: Frequency in kHz
            
        Returns:
            Reflection coefficient (0-1) for this frequency
        """
        if not self.frequency_absorption:
            return self.reflection_coefficient
        
        # Check which frequency range this falls into
        for (f_min, f_max), absorption in self.frequency_absorption.items():
            if f_min <= frequency_khz <= f_max:
                # More absorption = less reflection
                # absorption is fraction absorbed (0-1)
                return self.reflection_coefficient * (1.0 - absorption)
        
        # Default if frequency not in any defined range
        return self.reflection_coefficient
    
    def reflect_ray(self, ray_direction, surface_normal):
        """
        Compute reflected ray direction using law of reflection.
        
        Args:
            ray_direction: Incident ray direction (normalized)
            surface_normal: Surface normal (normalized, pointing outward)
            
        Returns:
            Reflected ray direction (normalized)
        """
        # Normalize inputs
        d = np.array(ray_direction, dtype=float)
        d = d / np.linalg.norm(d)
        
        n = np.array(surface_normal, dtype=float)
        n = n / np.linalg.norm(n)
        
        # Reflection formula: r = d - 2(d·n)n
        reflected = d - 2 * np.dot(d, n) * n
        
        return reflected / np.linalg.norm(reflected)
    
    def get_bounds(self):
        """
        Get bounding box.
        
        Returns:
            (min_bounds, max_bounds) as numpy arrays
        """
        return self.min_bounds.copy(), self.max_bounds.copy()
    
    def __repr__(self):
        return (f"Object3D(name='{self.name}', pos={self.position}, "
                f"size={self.size}, refl_coeff={self.reflection_coefficient})")


class Surface(Object3D):
    """
    Specialized class for water surface (with phase inversion on reflection).
    """
    
    def __init__(self, size):
        """
        Create water surface at z=0.
        
        Args:
            size: (length, width) of water surface
        """
        # Surface is a thin plane at z=0
        super().__init__(
            name="Surface",
            position=[size[0]/2, size[1]/2, 0],
            size=[size[0], size[1], 0.01],  # Very thin
            reflection_coefficient=0.98,  # Nearly perfect reflection
            is_rigid=True
        )
        self.phase_inversion = True  # Surface causes phase inversion


class Seabed(Object3D):
    """
    Specialized class for seabed.
    """
    
    def __init__(self, size, depth, hardness=0.3):
        """
        Create seabed at bottom of water volume.
        
        Args:
            size: (length, width) of water
            depth: Depth of water
            hardness: Reflection coefficient (0=soft mud, 1=hard rock)
        """
        super().__init__(
            name="Seabed",
            position=[size[0]/2, size[1]/2, depth],
            size=[size[0], size[1], 0.01],  # Very thin
            reflection_coefficient=hardness,
            is_rigid=True
        )
        self.hardness = hardness
