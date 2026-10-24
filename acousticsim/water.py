import numpy as np


class Water:
    """
    Represents the underwater environment with physical properties.
    
    Attributes:
        size: (length, width, depth) in meters
        temperature_profile: function(depth) -> temperature in Celsius
        salinity: parts per thousand (ppt)
        ambient_noise_level: dB re 1 μPa
    """
    
    def __init__(self, size=(100, 100, 50), temperature_surface=20.0, 
                 temperature_bottom=10.0, salinity=35.0, ambient_noise_level=50.0):
        """
        Initialize water environment.
        
        Args:
            size: (length_x, width_y, depth_z) in meters
            temperature_surface: Surface temperature in Celsius
            temperature_bottom: Bottom temperature in Celsius
            salinity: Salinity in ppt
            ambient_noise_level: Ambient noise in dB re 1 μPa
        """
        self.size = np.array(size, dtype=float)
        self.temperature_surface = temperature_surface
        self.temperature_bottom = temperature_bottom
        self.salinity = salinity
        self.ambient_noise_level = ambient_noise_level
        
    def temperature_at(self, x, y, z):
        """
        Get temperature at position (x, y, z).
        Linear gradient from surface to bottom.
        
        Args:
            x, y, z: Position in meters (z=0 is surface, positive down)
            
        Returns:
            Temperature in Celsius
        """
        # Clamp z to valid range
        z_clamped = np.clip(z, 0, self.size[2])
        
        # Linear interpolation between surface and bottom
        ratio = z_clamped / self.size[2]
        temp = self.temperature_surface + (self.temperature_bottom - self.temperature_surface) * ratio
        return temp
    
    def sound_speed_mackenzie(self, temperature, depth):
        """
        Mackenzie equation for sound speed in seawater.
        
        Args:
            temperature: Temperature in Celsius
            depth: Depth in meters
            
        Returns:
            Sound speed in m/s
        """
        T = temperature
        D = depth
        S = self.salinity
        
        # Mackenzie (1981) equation
        c = (1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
             + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D**2
             - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D**3)
        
        return c
    
    def sound_speed_at(self, x, y, z):
        """
        Get sound speed at position (x, y, z).
        
        Args:
            x, y, z: Position in meters
            
        Returns:
            Sound speed in m/s
        """
        temp = self.temperature_at(x, y, z)
        depth = z
        return self.sound_speed_mackenzie(temp, depth)
    
    def absorption_thorp(self, frequency_khz, depth=None):
        """
        Thorp equation for acoustic absorption.
        Extended to support 30-200 kHz range.
        
        Args:
            frequency_khz: Frequency in kHz (30-200 kHz)
            depth: Depth in meters (not used in Thorp, but included for API consistency)
            
        Returns:
            Absorption coefficient in dB/km
        """
        f = frequency_khz
        
        # Thorp equation (valid for wide frequency range)
        alpha = (0.11 * f**2 / (1 + f**2) + 
                 44 * f**2 / (4100 + f**2) + 
                 2.75e-4 * f**2 + 
                 0.003)
        
        return alpha
    
    def absorption_at(self, x, y, z, frequency_khz):
        """
        Get absorption coefficient at position (x, y, z) for given frequency.
        
        Args:
            x, y, z: Position in meters
            frequency_khz: Frequency in kHz
            
        Returns:
            Absorption coefficient in dB/km
        """
        return self.absorption_thorp(frequency_khz, z)
    
    def is_inside(self, x, y, z):
        """
        Check if position is inside water volume.
        
        Args:
            x, y, z: Position in meters
            
        Returns:
            True if inside water volume
        """
        return (0 <= x <= self.size[0] and 
                0 <= y <= self.size[1] and 
                0 <= z <= self.size[2])
    
    def sound_speed_gradient(self, x, y, z, dx=0.1):
        """
        Compute gradient of sound speed at position (for refraction).
        
        Args:
            x, y, z: Position in meters
            dx: Step size for numerical gradient
            
        Returns:
            Gradient vector (dc/dx, dc/dy, dc/dz)
        """
        c0 = self.sound_speed_at(x, y, z)
        
        # Numerical gradient
        grad = np.zeros(3)
        
        # x-component
        if x + dx <= self.size[0]:
            c_plus = self.sound_speed_at(x + dx, y, z)
            grad[0] = (c_plus - c0) / dx
        
        # y-component (usually negligible in layered ocean)
        if y + dx <= self.size[1]:
            c_plus = self.sound_speed_at(x, y + dx, z)
            grad[1] = (c_plus - c0) / dx
        
        # z-component (dominant in stratified water)
        if z + dx <= self.size[2]:
            c_plus = self.sound_speed_at(x, y, z + dx)
            grad[2] = (c_plus - c0) / dx
        
        return grad
    
    def get_ambient_noise(self):
        """
        Get ambient noise level.
        
        Returns:
            Noise level in dB re 1 μPa
        """
        return self.ambient_noise_level
    
    def __repr__(self):
        return (f"Water(size={self.size}, T_surf={self.temperature_surface}°C, "
                f"T_bottom={self.temperature_bottom}°C, salinity={self.salinity} ppt)")
