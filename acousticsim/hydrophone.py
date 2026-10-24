"""
Hydrophone Module
Represents acoustic sensors that receive multipath signals.
"""

import numpy as np


# Optional noise model — imported lazily to avoid circular deps.
def _check_noise_model(noise_model):
    if noise_model is not None:
        from acousticsim.noise import NoiseModel  # noqa: F401 (just validates type)
        if not isinstance(noise_model, NoiseModel):
            raise TypeError("noise_model must be a NoiseModel instance or None.")


class Hydrophone:
    """
    Represents a hydrophone sensor in 3D space.
    
    Receives acoustic signals from multiple ray paths and computes:
        - Time of arrivals
        - Amplitudes
        - Phases
        - Combined signal
    """
    
    def __init__(self, position, name=None, sensitivity_db=-170.0):
        """
        Initialize hydrophone.
        
        Args:
            position: (x, y, z) sensor location
            name: Identifier for this hydrophone
            sensitivity_db: Sensor sensitivity in dB re 1V/μPa
        """
        self.position = np.array(position, dtype=float)
        self.name = name or f"Hydrophone_{id(self)}"
        self.sensitivity_db = sensitivity_db
        
        # Storage for received signals
        self.arrivals = []  # List of (time, amplitude, phase, ray_path)
        
        # Detection threshold (acoustic noise floor)
        # In real underwater environments: -170 to -200 dB re 1 μPa
        # Corresponds to amplitude ~ 1e-9 to 1e-6 relative to source
        self.min_amplitude = 1e-7  # Minimum detectable amplitude (noise floor)
        self.max_detection_distance = 50.0  # Maximum perpendicular distance to detect (meters)
    
    def detect_ray(self, ray, sound_speed=1500.0):
        """
        Detect acoustic wave from ray path using physically accurate amplitude calculation.
        
        Unlike geometric ray tracing, acoustic waves spread out from the ray path.
        We calculate the amplitude at the hydrophone position based on:
        1. Ray amplitude (accounts for propagation losses along path)
        2. Perpendicular distance from ray path to hydrophone
        3. Additional spreading loss from ray to hydrophone
        
        Args:
            ray: Ray object
            sound_speed: Average sound speed for time calculation
            
        Returns:
            True if ray detected (amplitude above threshold)
        """
        # Calculate closest point on ray path to hydrophone
        # Ray current position and direction define the ray line
        ray_to_hydro = self.position - ray.position
        
        # Project onto ray direction to find closest point
        # Only consider points behind current ray position (already traveled)
        along_ray = np.dot(ray_to_hydro, ray.direction)
        
        # If projection is negative, ray hasn't reached closest approach yet
        # Still calculate distance from current position
        if along_ray < 0:
            closest_point = ray.position
        else:
            # Don't project beyond reasonable distance (wave spreading limit)
            along_ray = min(along_ray, self.max_detection_distance * 2)
            closest_point = ray.position + ray.direction * along_ray
        
        # Perpendicular distance from ray path to hydrophone
        perpendicular_distance = np.linalg.norm(self.position - closest_point)
        
        # === ACOUSTIC WAVE AMPLITUDE CALCULATION ===
        # The ray carries amplitude along its path, but the acoustic wave spreads
        # cylindrically perpendicular to the ray direction.
        
        # For short perpendicular distances (near the ray), use cylindrical spreading:
        # Amplitude decreases as 1/sqrt(r_perp) for cylindrical waves
        
        if perpendicular_distance < 0.01:  # Very close to ray path
            perpendicular_distance = 0.01  # Avoid division by zero
        
        # Cylindrical spreading factor (amplitude ∝ 1/√r for perpendicular distance)
        # Reference distance: 1 meter
        spreading_factor = 1.0 / np.sqrt(max(1.0, perpendicular_distance))
        
        # Calculate effective amplitude at hydrophone position
        amplitude_at_hydrophone = ray.amplitude * spreading_factor
        
        # Additional path length from ray to hydrophone (for time calculation)
        additional_distance = perpendicular_distance
        total_distance = ray.path_length + additional_distance
        
        # Check if amplitude is above detection threshold
        if amplitude_at_hydrophone >= self.min_amplitude:
            # Compute time of arrival
            time_of_arrival = total_distance / sound_speed
            
            # Calculate additional phase shift from ray path to hydrophone
            wavelength = sound_speed / (ray.frequency_khz * 1000.0)
            phase_shift = 2 * np.pi * (additional_distance / wavelength)
            total_phase = (ray.phase + phase_shift) % (2 * np.pi)
            
            # Record arrival
            arrival = {
                'time': time_of_arrival,
                'amplitude': amplitude_at_hydrophone,
                'phase': total_phase,
                'path_length': total_distance,
                'bounces': ray.bounces,
                'frequency_khz': ray.frequency_khz,
                'wave_type': ray.wave_type,
                'frequency_range': ray.frequency_range,
                'position': closest_point.copy(),
                'perpendicular_distance': perpendicular_distance
            }
            
            self.arrivals.append(arrival)
            return True
        
        return False
    
    def get_time_of_arrivals(self):
        """
        Get all arrival times.
        
        Returns:
            List of arrival times in seconds
        """
        return [arrival['time'] for arrival in self.arrivals]
    
    def get_amplitudes(self, noise_model=None):
        """
        Get all arrival amplitudes.

        Args:
            noise_model: Optional :class:`~acousticsim.noise.NoiseModel`.
                         When provided, random noise is added to each
                         amplitude value before returning.

        Returns:
            List of amplitudes (with noise if *noise_model* is set).
        """
        amps = [arrival['amplitude'] for arrival in self.arrivals]
        if noise_model is not None and len(amps) > 0:
            _check_noise_model(noise_model)
            # Generate at least 64 samples for stable spectral shaping,
            # then trim to the number of arrivals.
            n_gen = max(len(amps), 64)
            noise = noise_model.generate(n_gen)[: len(amps)]
            amps = [max(0.0, a + float(n)) for a, n in zip(amps, noise)]
        return amps
    
    def get_phases(self):
        """
        Get all arrival phases.
        
        Returns:
            List of phases in radians
        """
        return [arrival['phase'] for arrival in self.arrivals]
    
    def get_multipath_signal(self, time_samples=None, duration=None,
                             noise_model=None, sample_rate=None):
        """
        Construct combined multipath signal.

        Args:
            time_samples:  Time points to sample (seconds). Auto-computed
                           if *None*.
            duration:      Total duration to simulate (seconds).
            noise_model:   Optional :class:`~acousticsim.noise.NoiseModel`.
                           When provided, shaped acoustic noise is added to
                           the returned signal array.
            sample_rate:   Sample rate in Hz passed to the noise generator
                           for spectral shaping (overrides the auto value).

        Returns:
            ``(time_array, signal_array)`` — time and combined (noisy) signal.
        """
        if noise_model is not None:
            _check_noise_model(noise_model)
        if len(self.arrivals) == 0:
            return np.array([]), np.array([])
        
        # Determine time range
        if time_samples is None:
            min_time = min(arrival['time'] for arrival in self.arrivals)
            max_time = max(arrival['time'] for arrival in self.arrivals)
            
            if duration is None:
                duration = max_time - min_time + 0.01  # Add 10ms buffer
                time_start = min_time - 0.005
            else:
                time_start = 0
            
            # Sample at ~10x carrier frequency (only if caller didn't specify)
            if sample_rate is None:
                max_freq = max(arrival['frequency_khz'] for arrival in self.arrivals)
                sample_rate = max_freq * 1000 * 10  # Hz

            time_samples = np.arange(time_start, time_start + duration, 1/sample_rate)

        # ── keep auto sample_rate for noise shaping ──────────────────────────
        if sample_rate is None and len(self.arrivals) > 0:
            max_freq = max(a['frequency_khz'] for a in self.arrivals)
            sample_rate = max_freq * 1000 * 10
        elif sample_rate is None:
            sample_rate = 44100.0

        # Construct signal as sum of all arrivals
        signal = np.zeros_like(time_samples)
        
        for arrival in self.arrivals:
            t_arrival = arrival['time']
            amplitude = arrival['amplitude']
            phase = arrival['phase']
            freq_hz = arrival['frequency_khz'] * 1000
            
            # Add sinusoidal component for this arrival
            # s(t) = A * sin(2πft + φ) for t >= t_arrival
            mask = time_samples >= t_arrival
            signal[mask] += amplitude * np.sin(
                2 * np.pi * freq_hz * (time_samples[mask] - t_arrival) + phase
            )
        
        # ── optional noise injection ─────────────────────────────────────────
        if noise_model is not None and len(signal) > 0:
            noise = noise_model.generate(
                len(signal),
                sample_rate=sample_rate,
                reference_signal=signal,
            )
            signal = signal + noise

        return time_samples, signal

    def get_strongest_arrival(self):
        """
        Get the arrival with strongest amplitude.
        
        Returns:
            Dictionary with arrival info, or None if no arrivals
        """
        if len(self.arrivals) == 0:
            return None
        
        return max(self.arrivals, key=lambda a: a['amplitude'])
    
    def get_first_arrival(self):
        """
        Get the earliest arrival (direct path or first reflection).
        
        Returns:
            Dictionary with arrival info, or None if no arrivals
        """
        if len(self.arrivals) == 0:
            return None
        
        return min(self.arrivals, key=lambda a: a['time'])
    
    def clear_arrivals(self):
        """Clear all recorded arrivals."""
        self.arrivals = []
    
    def get_arrival_count(self):
        """Get number of arrivals detected."""
        return len(self.arrivals)
    
    def __repr__(self):
        return (f"Hydrophone(name='{self.name}', pos={self.position}, "
                f"arrivals={len(self.arrivals)})")
