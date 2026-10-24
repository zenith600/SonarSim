"""
Core Simulation Module
Implements ray tracing engine for acoustic propagation.
"""

import numpy as np
import asyncio
from typing import Optional, Callable
from acousticsim.water import Water
from acousticsim.object3d import Object3D, Surface, Seabed
from acousticsim.emitter import WaveEmitter
from acousticsim.hydrophone import Hydrophone
from acousticsim.ray import Ray


class Simulation:
    """
    Main simulation engine for underwater acoustic propagation.
    
    Uses ray tracing to simulate:
        - Reflections off objects
        - Refraction through varying sound speed
        - Multipath propagation
        - Absorption and spreading loss
    """
    
    def __init__(self, water=None):
        """
        Initialize simulation.
        
        Args:
            water: Water environment object (or None to create default)
        """
        self.water = water or Water()
        
        self.objects = []
        self.emitters = []
        self.hydrophones = []
        self.rays = []
        
        # Simulation parameters
        self.max_bounces = 15  # Allow more bounces with realistic loss
        self.min_amplitude = 1e-6  # Threshold for ray termination (realistic acoustic floor)
        self.step_size = 0.5  # meters per ray step
        self.max_steps = 20000  # Increased to allow longer paths with multiple reflections
        self.max_path_length = 1000.0  # Maximum total ray path length in meters
        
        # Real-time callback
        self.realtime_callback: Optional[Callable] = None
        self.ray_batch_size = 50  # Send updates every N rays
        
        # Add default surface and seabed
        self.add_surface()
        self.add_seabed()
    
    def add_surface(self):
        """Add water surface."""
        surface = Surface(self.water.size[:2])
        self.objects.append(surface)
        return surface
    
    def add_seabed(self, hardness=0.3):
        """Add seabed."""
        seabed = Seabed(self.water.size[:2], self.water.size[2], hardness)
        self.objects.append(seabed)
        return seabed
    
    def add_object(self, obj):
        """Add an object to the simulation."""
        self.objects.append(obj)
    
    def add_emitter(self, emitter):
        """Add a wave emitter."""
        self.emitters.append(emitter)
    
    def add_hydrophone(self, hydrophone):
        """Add a hydrophone."""
        self.hydrophones.append(hydrophone)
    
    def run(self, num_rays_per_emitter=1000, verbose=True):
        """
        Run the simulation.
        
        Args:
            num_rays_per_emitter: Number of rays to emit from each emitter
            verbose: Print progress information
            
        Returns:
            Dictionary with simulation results
        """
        if verbose:
            print(f"Starting simulation with {len(self.emitters)} emitter(s)")
            print(f"Water: {self.water}")
            print(f"Objects: {len(self.objects)}")
            print(f"Hydrophones: {len(self.hydrophones)}")
        
        # Clear previous results
        self.rays = []
        for hydro in self.hydrophones:
            hydro.clear_arrivals()
        
        # Emit rays from all emitters
        for emitter_idx, emitter in enumerate(self.emitters):
            if verbose:
                print(f"\nEmitter {emitter_idx}: {emitter}")
            
            rays = emitter.emit_rays(num_rays_per_emitter)
            
            if verbose:
                print(f"  Emitted {len(rays)} rays")
            
            # Trace each ray
            for ray_idx, ray in enumerate(rays):
                self._trace_ray(ray)
                
                if verbose and ray_idx % (num_rays_per_emitter // 10) == 0:
                    print(f"  Progress: {ray_idx}/{len(rays)} rays traced")
            
            self.rays.extend(rays)
        
        # Compile results
        results = {
            'num_rays': len(self.rays),
            'hydrophones': {},
            'water': self.water,
            'objects': self.objects,
            'emitters': self.emitters
        }
        
        for hydro in self.hydrophones:
            results['hydrophones'][hydro.name] = {
                'position': hydro.position,
                'arrivals': hydro.arrivals,
                'count': hydro.get_arrival_count(),
                'first_arrival': hydro.get_first_arrival(),
                'strongest_arrival': hydro.get_strongest_arrival()
            }
        
        if verbose:
            print("\n=== Simulation Results ===")
            for hydro in self.hydrophones:
                print(f"{hydro.name}: {hydro.get_arrival_count()} arrivals")
        
        return results
    
    def _trace_ray(self, ray):
        """
        Trace a single ray through the environment.
        
        Args:
            ray: Ray object to trace
        """
        for step in range(self.max_steps):
            if not ray.is_active:
                break
            
            # Check termination conditions
            # 1. Amplitude below detection threshold
            if ray.amplitude < self.min_amplitude:
                ray.terminate()
                break
            
            # 2. Too many reflections (energy significantly degraded)
            if ray.bounces >= self.max_bounces:
                ray.terminate()
                break
            
            # 3. Path too long (time dispersion too large)
            if ray.path_length >= self.max_path_length:
                ray.terminate()
                break
            
            # 4. Ray escaped the water volume
            if not self.water.is_inside(*ray.position):
                ray.terminate()
                break
            
            # Find nearest intersection with any object
            nearest_distance = self.step_size
            nearest_object = None
            nearest_hit_point = None
            nearest_normal = None
            
            for obj in self.objects:
                hit, distance, hit_point, normal = obj.ray_intersect(
                    ray.position, ray.direction, nearest_distance
                )
                
                if hit and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_object = obj
                    nearest_hit_point = hit_point
                    nearest_normal = normal
            
            # Propagate ray
            if nearest_object is not None:
                # Propagate to intersection point
                ray.propagate(nearest_distance, self.water)
                
                # Get frequency-dependent reflection coefficient
                refl_coeff = nearest_object.get_reflection_coefficient(ray.frequency_khz)
                
                # Reflect off object
                phase_inversion = isinstance(nearest_object, Surface)
                ray.reflect(
                    nearest_normal,
                    refl_coeff,
                    phase_inversion,
                    object_name=nearest_object.name,
                )
                
                # Small step away from surface to avoid re-collision
                ray.position += ray.direction * 0.01
            else:
                # No collision, propagate one step
                ray.propagate(self.step_size, self.water)
                
                # Apply refraction based on sound speed gradient
                gradient = self.water.sound_speed_gradient(*ray.position)
                ray.refract(gradient, self.step_size)
            
            # Check if acoustic wave from this ray is detected by any hydrophone
            # Physical: acoustic waves spread from the ray path, so hydrophones
            # can detect even if the ray doesn't geometrically intersect them
            for hydro in self.hydrophones:
                # Quick distance check to avoid expensive calculations for far rays
                approx_distance = np.linalg.norm(ray.position - hydro.position)
                if approx_distance < hydro.max_detection_distance:
                    c_avg = self.water.sound_speed_at(*ray.position)
                    hydro.detect_ray(ray, c_avg)
    
    def get_rays_for_visualization(self, max_rays=1000):
        """
        Get ray paths for visualization (limit number for performance).
        
        Args:
            max_rays: Maximum number of rays to return
            
        Returns:
            List of ray path lists
        """
        # Sample rays uniformly if too many
        if len(self.rays) <= max_rays:
            selected_rays = self.rays
        else:
            indices = np.linspace(0, len(self.rays) - 1, max_rays, dtype=int)
            selected_rays = [self.rays[i] for i in indices]
        
        return [ray.get_path() for ray in selected_rays]
    
    def get_reflection_points(self, object_name=None):
        """
        Get all reflection points from all rays.
        
        Args:
            object_name: If given, only return points from that object's surface.
        Returns:
            List of (x, y, z) reflection points
        """
        points = []
        for ray in self.rays:
            points.extend(ray.get_reflection_points(object_name=object_name))
        return points
    
    def export_results(self, filename='simulation_results.json'):
        """
        Export simulation results to JSON.
        
        Args:
            filename: Output filename
        """
        import json
        
        data = {
            'water': {
                'size': self.water.size.tolist(),
                'temperature_surface': self.water.temperature_surface,
                'temperature_bottom': self.water.temperature_bottom,
                'salinity': self.water.salinity
            },
            'emitters': [
                {
                    'position': em.position.tolist(),
                    'frequency_khz': em.frequency_khz,
                    'beam_pattern': em.beam_pattern
                }
                for em in self.emitters
            ],
            'hydrophones': [
                {
                    'name': hydro.name,
                    'position': hydro.position.tolist(),
                    'arrivals': [
                        {
                            'time': arr['time'],
                            'amplitude': arr['amplitude'],
                            'phase': arr['phase'],
                            'bounces': arr['bounces']
                        }
                        for arr in hydro.arrivals
                    ]
                }
                for hydro in self.hydrophones
            ],
            'num_rays': len(self.rays),
            'parameters': {
                'max_bounces': self.max_bounces,
                'min_amplitude': self.min_amplitude,
                'step_size': self.step_size
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to {filename}")
    
    async def run_realtime(self, num_rays_per_emitter=1000, verbose=True):
        """
        Run simulation with real-time updates via WebSocket.
        
        Args:
            num_rays_per_emitter: Number of rays to emit per emitter
            verbose: Print progress information
            
        Returns:
            Dictionary containing simulation results
        """
        import time
        from realtime_server import get_server
        
        server = get_server()
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING REAL-TIME ACOUSTIC SIMULATION")
            print(f"{'='*60}")
            print(f"Emitters: {len(self.emitters)}")
            print(f"Hydrophones: {len(self.hydrophones)}")
            print(f"Objects: {len(self.objects)}")
            print(f"Rays per emitter: {num_rays_per_emitter}")
        
        # Clear previous results
        self.rays = []
        for hydro in self.hydrophones:
            hydro.clear_arrivals()
        
        # Send initial scene setup
        await server.send_initial_scene({
            'pool_size': self.water.size,
            'objects': [
                {
                    'name': obj.name,
                    'position': obj.position.tolist(),
                    'size': obj.size.tolist()
                }
                for obj in self.objects if not isinstance(obj, (Surface, Seabed))
            ],
            'emitters': [
                {
                    'position': em.position.tolist(),
                    'frequency': em.frequency_khz
                }
                for em in self.emitters
            ],
            'hydrophones': [
                {
                    'name': hydro.name,
                    'position': hydro.position.tolist()
                }
                for hydro in self.hydrophones
            ],
            'total_rays': num_rays_per_emitter * len(self.emitters)
        })
        
        total_rays = num_rays_per_emitter * len(self.emitters)
        traced_rays = 0
        
        # Emit and trace rays from each emitter
        for emitter_idx, emitter in enumerate(self.emitters):
            if verbose:
                print(f"\nEmitter {emitter_idx+1}/{len(self.emitters)}:")
            
            rays = emitter.emit_rays(num_rays_per_emitter)
            
            if verbose:
                print(f"  Emitted {len(rays)} rays")
            
            ray_batch = []
            
            # Trace each ray
            for ray_idx, ray in enumerate(rays):
                # Store initial arrivals count
                initial_arrivals = {hydro.name: len(hydro.arrivals) for hydro in self.hydrophones}
                
                # Trace the ray
                self._trace_ray(ray)
                traced_rays += 1
                
                # Check if this ray was detected by any hydrophone
                was_detected = False
                
                # Check for new arrivals and send them
                for hydro in self.hydrophones:
                    new_arrival_count = len(hydro.arrivals) - initial_arrivals[hydro.name]
                    if new_arrival_count > 0:
                        was_detected = True
                        # Get the latest arrival
                        latest_arrival = hydro.arrivals[-1]
                        await server.send_arrival(hydro.name, {
                            'position': hydro.position.tolist(),
                            'time': latest_arrival['time'],
                            'amplitude': latest_arrival['amplitude']
                        })
                
                # Collect ray path data with color based on detection
                if len(ray.path_points) > 1:
                    # Color coding:
                    # Green (0x00ff00): Detected by hydrophone(s)
                    # Cyan (0x00ffff): Not detected, no bounces (direct path missed)
                    # Blue (0x0088ff): Not detected, with bounces (multipath missed)
                    if was_detected:
                        color = 0x00ff00  # Green - successfully detected
                    elif ray.bounces == 0:
                        color = 0x00ffff  # Cyan - direct path, not detected
                    else:
                        color = 0x0088ff  # Blue - multipath, not detected
                    
                    ray_batch.append({
                        'path': [p.tolist() for p in ray.path_points],
                        'color': color
                    })
                
                # Send batch of rays periodically
                if len(ray_batch) >= self.ray_batch_size:
                    await server.send_ray_batch(ray_batch)
                    ray_batch = []
                
                # Send progress update
                if ray_idx % (num_rays_per_emitter // 20) == 0:
                    await server.send_progress(traced_rays, total_rays)
                    if verbose:
                        print(f"  Progress: {ray_idx}/{len(rays)} rays traced")
                    
                    # Small delay to allow visualization to update
                    await asyncio.sleep(0.01)
            
            # Send any remaining rays in batch
            if ray_batch:
                await server.send_ray_batch(ray_batch)
            
            self.rays.extend(rays)
        
        # Final progress update
        await server.send_progress(total_rays, total_rays)
        
        # Compile results
        results = {
            'num_rays': len(self.rays),
            'hydrophones': {},
            'water': self.water,
            'objects': self.objects,
            'emitters': self.emitters
        }
        
        total_arrivals = 0
        for hydro in self.hydrophones:
            arrivals = hydro.get_arrival_count()
            total_arrivals += arrivals
            results['hydrophones'][hydro.name] = {
                'position': hydro.position,
                'arrivals': hydro.arrivals,
                'count': arrivals,
                'first_arrival': hydro.get_first_arrival(),
                'strongest_arrival': hydro.get_strongest_arrival()
            }
        
        duration = time.time() - start_time
        
        # Send completion message
        await server.send_complete({
            'total_rays': total_rays,
            'total_arrivals': total_arrivals,
            'hydrophone_count': len(self.hydrophones),
            'duration': duration
        })
        
        if verbose:
            print("\n=== Real-Time Simulation Complete ===")
            print(f"Duration: {duration:.2f}s")
            for hydro in self.hydrophones:
                print(f"{hydro.name}: {hydro.get_arrival_count()} arrivals")
        
        return results
    
    def __repr__(self):
        return (f"Simulation(water={self.water}, objects={len(self.objects)}, "
                f"emitters={len(self.emitters)}, hydrophones={len(self.hydrophones)})")
