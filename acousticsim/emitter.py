"""
Wave Emitter Module
Emits acoustic rays with various beam patterns (80-90 kHz range).
"""

import numpy as np
from acousticsim.ray import Ray


class WaveEmitter:
    """
    Emits acoustic waves as rays from a point source.
    
    Supports:
        - Spherical (omnidirectional) emission
        - Cone beam
        - Custom directional patterns
    """
    
    def __init__(self, position, frequency_khz=85.0, beam_pattern='spherical',
                 beam_direction=None, beam_angle=30.0, power_db=180.0,
                 wave_type='continuous', frequency_range=None, chirp_duration=0.01):
        """
        Initialize wave emitter.
        
        Args:
            position: (x, y, z) emitter location
            frequency_khz: Center frequency in kHz (30-200 kHz range)
            beam_pattern: 'spherical', 'cone', or 'custom'
            beam_direction: (dx, dy, dz) main beam direction for cone pattern
            beam_angle: Half-angle of cone beam in degrees
            power_db: Source level in dB re 1 μPa @ 1m
            wave_type: 'continuous' or 'chirp'
            frequency_range: (f_min, f_max) in kHz for chirp, None for continuous
            chirp_duration: Duration of chirp in seconds
        """
        self.position = np.array(position, dtype=float)
        self.frequency_khz = np.clip(frequency_khz, 30.0, 200.0)  # Expanded range
        self.beam_pattern = beam_pattern
        self.power_db = power_db
        self.wave_type = wave_type
        
        # Frequency range for chirp signals
        if wave_type == 'chirp':
            if frequency_range is None:
                # Default chirp range: ±10% of center frequency
                f_min = frequency_khz * 0.9
                f_max = frequency_khz * 1.1
                self.frequency_range = (f_min, f_max)
            else:
                self.frequency_range = tuple(frequency_range)
            self.chirp_duration = chirp_duration
            # For chirp, use center of range as main frequency
            self.frequency_khz = (self.frequency_range[0] + self.frequency_range[1]) / 2
        else:
            self.frequency_range = (frequency_khz, frequency_khz)
            self.chirp_duration = None
        
        # Convert power to linear amplitude
        # Reference: 1 μPa @ 1m = 0 dB
        # Amplitude ∝ 10^(dB/20)
        self.initial_amplitude = 10 ** (power_db / 20.0)
        
        # Beam parameters
        if beam_direction is not None:
            beam_direction = np.array(beam_direction, dtype=float)
            self.beam_direction = beam_direction / np.linalg.norm(beam_direction)
        else:
            self.beam_direction = np.array([1.0, 0.0, 0.0])  # Default: +x direction
        
        self.beam_angle = beam_angle  # In degrees
    
    def emit_rays(self, num_rays=1000):
        """
        Emit rays according to beam pattern.
        
        Args:
            num_rays: Number of rays to emit
            
        Returns:
            List of Ray objects
        """
        rays = []
        
        if self.beam_pattern == 'spherical':
            rays = self._emit_spherical(num_rays)
        elif self.beam_pattern == 'cone':
            rays = self._emit_cone(num_rays)
        else:
            # Default to spherical
            rays = self._emit_spherical(num_rays)
        
        return rays
    
    def _emit_spherical(self, num_rays):
        """
        Emit rays uniformly in all directions (spherical pattern).
        
        Args:
            num_rays: Number of rays to emit
            
        Returns:
            List of Ray objects
        """
        rays = []
        
        # Fibonacci sphere algorithm for uniform distribution
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(num_rays):
            # Fibonacci lattice on sphere
            theta = 2 * np.pi * i / phi
            phi_angle = np.arccos(1 - 2 * (i + 0.5) / num_rays)
            
            # Convert to Cartesian
            x = np.sin(phi_angle) * np.cos(theta)
            y = np.sin(phi_angle) * np.sin(theta)
            z = np.cos(phi_angle)
            
            direction = np.array([x, y, z])
            
            # Create ray with amplitude adjusted for number of rays
            # Total power is conserved, so each ray gets 1/num_rays of power
            amplitude = self.initial_amplitude / np.sqrt(num_rays)
            
            ray = Ray(self.position, direction, self.frequency_khz, amplitude,
                     wave_type=self.wave_type, frequency_range=self.frequency_range)
            rays.append(ray)
        
        return rays
    
    def _emit_cone(self, num_rays):
        """
        Emit rays in a cone pattern.
        
        Args:
            num_rays: Number of rays to emit
            
        Returns:
            List of Ray objects
        """
        rays = []
        
        # Convert beam angle to radians
        max_angle = np.radians(self.beam_angle)
        
        # Create rays within cone
        for i in range(num_rays):
            # Random angle within cone
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
            
            # Polar angle (0 to max_angle)
            # Use sqrt for uniform distribution on cone
            cos_alpha = np.random.uniform(np.cos(max_angle), 1.0)
            alpha = np.arccos(cos_alpha)
            
            # Local coordinates (beam direction is z-axis)
            x_local = np.sin(alpha) * np.cos(theta)
            y_local = np.sin(alpha) * np.sin(theta)
            z_local = np.cos(alpha)
            
            # Rotate to align with beam_direction
            direction = self._rotate_to_beam_direction([x_local, y_local, z_local])
            
            # Create ray
            amplitude = self.initial_amplitude / np.sqrt(num_rays)
            ray = Ray(self.position, direction, self.frequency_khz, amplitude,
                     wave_type=self.wave_type, frequency_range=self.frequency_range)
            rays.append(ray)
        
        return rays
    
    def _rotate_to_beam_direction(self, local_vector):
        """
        Rotate a vector from local coordinates to global coordinates.
        Local z-axis aligns with beam_direction.
        
        Args:
            local_vector: (x, y, z) in local coordinates
            
        Returns:
            Vector in global coordinates
        """
        # Find rotation matrix to align [0, 0, 1] with beam_direction
        z_axis = np.array([0, 0, 1])
        
        if np.allclose(self.beam_direction, z_axis):
            return local_vector
        elif np.allclose(self.beam_direction, -z_axis):
            # Flip z
            return np.array([local_vector[0], local_vector[1], -local_vector[2]])
        
        # Rodrigues' rotation formula
        v = np.cross(z_axis, self.beam_direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, self.beam_direction)
        
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        
        return R @ local_vector
    
    def __repr__(self):
        if self.wave_type == 'chirp':
            return (f"WaveEmitter(pos={self.position}, "
                    f"chirp={self.frequency_range[0]:.1f}-{self.frequency_range[1]:.1f} kHz, "
                    f"pattern='{self.beam_pattern}', power={self.power_db} dB)")
        else:
            return (f"WaveEmitter(pos={self.position}, freq={self.frequency_khz} kHz, "
                    f"pattern='{self.beam_pattern}', power={self.power_db} dB)")
