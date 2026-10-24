"""
Ray Module
Represents acoustic rays propagating through water.
"""

import numpy as np


class Ray:
    """
    Represents a single acoustic ray path.
    
    Tracks position, direction, amplitude, phase, and path history.
    """
    
    def __init__(self, origin, direction, frequency_khz, initial_amplitude=1.0,
                 wave_type='continuous', frequency_range=None):
        """
        Initialize acoustic ray.
        
        Args:
            origin: (x, y, z) starting position
            direction: (dx, dy, dz) initial direction (will be normalized)
            frequency_khz: Center frequency in kHz
            initial_amplitude: Starting amplitude (arbitrary units)
            wave_type: 'continuous' or 'chirp'
            frequency_range: (f_min, f_max) for chirp signals
        """
        self.origin = np.array(origin, dtype=float)
        self.position = self.origin.copy()
        
        # Normalize direction
        direction = np.array(direction, dtype=float)
        self.direction = direction / np.linalg.norm(direction)
        
        self.frequency_khz = frequency_khz
        self.wave_type = wave_type
        self.frequency_range = frequency_range or (frequency_khz, frequency_khz)
        self.amplitude = initial_amplitude
        self.phase = 0.0  # Phase in radians
        
        # Path tracking
        self.path_length = 0.0  # Total distance traveled
        self.bounces = 0  # Number of reflections
        self.is_active = True
        
        # Store path for visualization
        self.path_points = [self.origin.copy()]
        
        # Store reflection points and which object caused each bounce
        self.reflection_points = []          # list of (x, y, z) positions
        self.reflection_objects = []         # parallel list of object names (str)
        self.reflection_amplitudes = []      # parallel list of incident amplitude at each bounce
        
    def propagate(self, distance, water):
        """
        Propagate ray through water for given distance with realistic energy loss.
        
        Implements:
        - Geometric spreading loss (spherical or cylindrical)
        - Frequency-dependent absorption (Thorp equation)
        - Scattering loss (particle scattering)
        
        Args:
            distance: Distance to propagate in meters
            water: Water environment object
            
        Returns:
            New position after propagation
        """
        if not self.is_active:
            return self.position
        
        # Update position
        new_position = self.position + self.direction * distance
        
        # Compute average sound speed along path
        c_start = water.sound_speed_at(*self.position)
        c_end = water.sound_speed_at(*new_position)
        c_avg = (c_start + c_end) / 2.0
        
        # Compute travel time
        travel_time = distance / c_avg
        
        # Update phase (2π * f * t)
        self.phase += 2 * np.pi * (self.frequency_khz * 1000) * travel_time
        self.phase = self.phase % (2 * np.pi)  # Keep in [0, 2π)
        
        # === GEOMETRIC SPREADING LOSS ===
        # Spherical spreading: TL = 20*log10(r) (intensity ∝ 1/r²)
        # After first meter, use spherical spreading
        if self.path_length < 1.0:
            # Near field: minimal loss
            spreading_loss_db = 0.0
        else:
            # Far field: spherical spreading
            r1 = max(1.0, self.path_length)  # Reference at 1m or current distance
            r2 = self.path_length + distance
            spreading_loss_db = 20 * np.log10(r2 / r1)
        
        # Convert dB to linear amplitude factor
        spreading_factor = 10 ** (-spreading_loss_db / 20.0)
        self.amplitude *= spreading_factor
        
        # === ABSORPTION LOSS ===
        # Frequency-dependent absorption using Thorp equation (dB/km)
        absorption_db_per_km = water.absorption_at(*self.position, self.frequency_khz)
        absorption_db = absorption_db_per_km * (distance / 1000.0)
        absorption_factor = 10 ** (-absorption_db / 20.0)
        self.amplitude *= absorption_factor
        
        # === SCATTERING LOSS ===
        # Volume scattering from particles, bubbles, turbulence
        # Higher frequencies scatter more (Rayleigh scattering ∝ f⁴)
        scattering_coefficient = 0.001 * (self.frequency_khz / 100.0) ** 2  # dB/km
        scattering_db = scattering_coefficient * (distance / 1000.0)
        scattering_factor = 10 ** (-scattering_db / 20.0)
        self.amplitude *= scattering_factor
        
        # Update position and path
        self.position = new_position
        self.path_length += distance
        self.path_points.append(self.position.copy())
        
        return self.position
    
    def reflect(self, surface_normal, reflection_coefficient,
                phase_inversion=False, object_name=None):
        """
        Reflect ray off a surface with angle-dependent losses.
        
        Implements:
        - Specular reflection (law of reflection)
        - Angle-dependent reflection loss (grazing angles lose more energy)
        - Material-dependent absorption
        - Phase inversion for pressure-release surfaces
        
        Args:
            surface_normal: Normal vector of surface (pointing outward)
            reflection_coefficient: Fraction of energy reflected at normal incidence
            phase_inversion: If True, phase inverted (π shift) for surface reflection
            object_name: Name of the object this ray bounced off (for filtering)
        """
        # Normalize normal
        normal = np.array(surface_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate incidence angle (angle between ray and normal)
        # cos(θ) = |d · n| where θ is angle from normal
        cos_theta = abs(np.dot(self.direction, normal))
        theta_deg = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
        
        # Reflection formula: r = d - 2(d·n)n
        self.direction = self.direction - 2 * np.dot(self.direction, normal) * normal
        self.direction = self.direction / np.linalg.norm(self.direction)
        
        # === ANGLE-DEPENDENT REFLECTION LOSS ===
        # At grazing incidence (θ → 90°), reflection is stronger (less absorption)
        # At normal incidence (θ = 0°), more energy is absorbed
        # Rayleigh roughness: reduces reflection at high angles and frequencies
        
        # Base reflection from material properties
        base_reflection = reflection_coefficient
        
        # Angle-dependent factor (Lambert's cosine law for rough surfaces)
        # For smooth surfaces: reflection increases at grazing angles
        # For rough surfaces: reflection decreases at grazing angles
        
        # Roughness parameter (wavelength-dependent)
        wavelength = 1500.0 / self.frequency_khz  # meters (c ≈ 1500 m/s)
        roughness_factor = np.exp(-((theta_deg / 45.0) ** 2) * 0.1)  # Peaks at normal
        
        # Combined reflection coefficient
        effective_reflection = base_reflection * (0.7 + 0.3 * roughness_factor)
        
        # Additional loss at grazing angles (> 60°) due to surface waves
        if theta_deg > 60:
            grazing_loss = 0.9 ** ((theta_deg - 60) / 10.0)
            effective_reflection *= grazing_loss
        
        # Apply reflection loss (energy loss = 1 - reflection coefficient)
        incident_amplitude = float(self.amplitude)   # capture BEFORE loss
        self.amplitude *= effective_reflection
        
        # === TRANSMISSION LOSS INTO MATERIAL ===
        # Some energy is transmitted into the reflector (converted to heat)
        # This is already accounted for in (1 - reflection_coefficient)
        transmission_loss = 1.0 - effective_reflection
        # Energy transmitted is lost from the ray
        
        # Phase inversion at pressure-release surface (water-air interface)
        if phase_inversion:
            self.phase += np.pi
            self.phase = self.phase % (2 * np.pi)
        
        # Random phase shift due to surface roughness (small effect)
        if theta_deg > 45:  # More phase randomization at high angles
            phase_noise = np.random.uniform(-0.1, 0.1) * (theta_deg - 45) / 45
            self.phase += phase_noise
            self.phase = self.phase % (2 * np.pi)
        
        # Increment bounce counter
        self.bounces += 1
        
        # Record reflection point and which object caused it
        self.reflection_points.append(self.position.copy())
        self.reflection_objects.append(object_name or 'unknown')
        self.reflection_amplitudes.append(incident_amplitude)  # incident amplitude at surface
    
    def refract(self, sound_speed_gradient, ds=0.1):
        """
        Apply refraction based on sound speed gradient (Snell's law).
        
        Args:
            sound_speed_gradient: Gradient vector (dc/dx, dc/dy, dc/dz)
            ds: Step size for refraction calculation
        """
        # Simple ray bending based on gradient
        # In stratified medium, rays bend toward lower sound speed
        
        grad = np.array(sound_speed_gradient, dtype=float)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm > 1e-6:
            # Bend direction slightly toward gradient
            # This is a simplified model; full ray tracing would use eikonal equation
            bend_strength = 0.001  # Tunable parameter
            bend = -grad * bend_strength * ds  # Negative: bend toward lower c
            
            self.direction = self.direction + bend
            self.direction = self.direction / np.linalg.norm(self.direction)
    
    def terminate(self):
        """Mark ray as inactive."""
        self.is_active = False
    
    def get_path(self):
        """
        Get ray path for visualization.
        
        Returns:
            List of (x, y, z) points along path
        """
        return self.path_points
    
    def get_reflection_points(self, object_name=None):
        """
        Get reflection points for visualization.
        
        Args:
            object_name: If given, only return reflections caused by this object.
                         If None, return all reflection points.
        Returns:
            List of (x, y, z) reflection points
        """
        if object_name is None:
            return self.reflection_points
        return [
            pt for pt, name in zip(self.reflection_points, self.reflection_objects)
            if name == object_name
        ]
    
    def __repr__(self):
        return (f"Ray(pos={self.position}, dir={self.direction}, "
                f"amp={self.amplitude:.4f}, bounces={self.bounces}, "
                f"active={self.is_active})")
