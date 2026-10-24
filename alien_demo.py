"""
AlienAnimal Acoustic Point Cloud Demo
======================================
Bistatic sonar simulation: an emitter fires acoustic rays that bounce off
the AlienAnimal.obj mesh. Reflections are collected and mapped as a 3D point
cloud, giving an "acoustic image" of the object's surface.

Pipeline:
    1. Load AlienAnimal.obj via trimesh
    2. Run acoustic ray-trace simulation with AABB intersection (fast physics)
    3. Collect reflection points tagged to the object, weighted by amplitude
    4. Use trimesh BVH raycasting (same ray directions) for exact surface hits
    5. Render and save:
           alien_pointcloud_scene.png   — full 3D scene overview
           alien_pointcloud_surface.png — acoustic surface point cloud

Run:
    cd frequency_examples
    python alien_demo.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── acousticsim imports (package in the acousticsim/ subfolder) ────────────
sys.path.insert(0, str(Path(__file__).parent))
from acousticsim.water import Water
from acousticsim.emitter import WaveEmitter
from acousticsim.hydrophone import Hydrophone
from acousticsim.simulation import Simulation
from acousticsim.mesh_importer import MeshImporter
from acousticsim.imported_object3d import ImportedObject3D

# ── Try trimesh for BVH-accurate surface hit points ────────────────────────
try:
    import trimesh
    TRIMESH_OK = True
except ImportError:
    TRIMESH_OK = False
    print("[warn] trimesh not found – skipping exact surface point cloud.")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT  = Path(__file__).parent
OBJ_FILE   = REPO_ROOT / "AlienAnimal.obj"
OUT_SCENE  = REPO_ROOT / "alien_pointcloud_scene.png"
OUT_CLOUD  = REPO_ROOT / "alien_pointcloud_surface.png"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SIMULATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Water volume — large coastal bay
WATER_SIZE = (100.0, 100.0, 50.0)   # lx, ly, lz  [meters]

# AlienAnimal placement
# At scale=1.0, extents = 17.9 × 19.6 × 39.3 m  (a large creature / sub-scale object)
OBJECT_SCALE    = 1.0
OBJECT_POSITION = [50.0, 50.0, 25.0]
OBJECT_ROTATION = (0.0, 0.0, 0.0)   # roll, pitch, yaw [deg]
OBJECT_REFLECT  = 0.80              # acoustic reflection coefficient

# Bistatic sonar geometry
#   Emitter  → object (left wall)
#   Hydro array ← reflected energy (right wall)
EMITTER_POS       = [5.0,  50.0, 25.0]
EMITTER_FREQ_KHZ  = 85.0
EMITTER_POWER_DB  = 185.0
EMITTER_BEAM_DIR  = [1.0, 0.0, 0.0]   # → +X toward object
EMITTER_BEAM_ANG  = 25.0              # half-angle [deg] — tighter cone on large object

# 2 × 2 hydrophone array at x=95  (opposite side from emitter)
HYDRO_X  = 95.0
HYDRO_YS = np.linspace(35.0, 65.0, 3)
HYDRO_ZS = np.linspace(15.0, 35.0, 3)

NUM_RAYS = 1200    # rays fired per emitter


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def build_simulation(mesh_data):
    """Construct the Water environment, object, emitter and hydrophones."""
    water = Water(
        size=WATER_SIZE,
        temperature_surface=18.0,
        temperature_bottom=8.0,
        salinity=35.0,
    )

    sim = Simulation(water)
    sim.max_bounces = 8
    sim.step_size   = 0.5

    # --- Imported object ---
    alien = ImportedObject3D(
        mesh_data=mesh_data,
        position=OBJECT_POSITION,
        rotation_deg=OBJECT_ROTATION,
        scale=OBJECT_SCALE,
        intersection_mode='aabb',       # fast for physics pass
        reflection_coefficient=OBJECT_REFLECT,
        name='AlienAnimal',
    )
    sim.add_object(alien)

    # --- Emitter ---
    emitter = WaveEmitter(
        position=EMITTER_POS,
        frequency_khz=EMITTER_FREQ_KHZ,
        beam_pattern='cone',
        beam_direction=EMITTER_BEAM_DIR,
        beam_angle=EMITTER_BEAM_ANG,
        power_db=EMITTER_POWER_DB,
    )
    sim.add_emitter(emitter)

    # --- Hydrophone array (2×2 grid on opposite wall) ---
    hydrophones = []
    for hy in HYDRO_YS:
        for hz in HYDRO_ZS:
            h = Hydrophone(
                position=[HYDRO_X, hy, hz],
                name=f"H_{hy:.0f}_{hz:.0f}",
            )
            h.max_detection_distance = 18.0   # only detect nearby rays (faster)
            sim.add_hydrophone(h)
            hydrophones.append(h)

    return sim, alien, hydrophones


# ══════════════════════════════════════════════════════════════════════════════
# 3.  COLLECT ACOUSTIC REFLECTION POINTS
# ══════════════════════════════════════════════════════════════════════════════

def collect_alien_hits(sim, object_name='AlienAnimal'):
    """
    Gather every reflection point that bounced off the target object,
    together with the incident acoustic amplitude at each hit.

    Returns:
        pts  : (N, 3) numpy array of hit positions
        amps : (N,)  numpy array of incident amplitudes
    """
    pts  = []
    amps = []

    for ray in sim.rays:
        for pt, name, amp in zip(
            ray.reflection_points,
            ray.reflection_objects,
            ray.reflection_amplitudes,
        ):
            if name == object_name:
                pts.append(pt)
                amps.append(amp)

    if not pts:
        return np.empty((0, 3)), np.empty(0)

    return np.array(pts, dtype=np.float64), np.array(amps, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRIMESH BVH SURFACE MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def trimesh_surface_hits(mesh_data, alien_obj, num_rays=8000):
    """
    Fire rays from the emitter position through a dense cone using
    trimesh's BVH-accelerated raycaster to get EXACT triangle-level
    surface hit points on the AlienAnimal mesh.

    Returns:
        pts      : (M, 3) world-space hit positions
        angles   : (M,)  angle of incidence in degrees
    """
    if not TRIMESH_OK:
        return None, None

    # Build a transformed trimesh: centered, scaled, positioned in world space
    tm = mesh_data._mesh.copy()

    # Center at local origin
    tm.apply_translation(-mesh_data.center)
    # Scale
    tm.apply_scale(alien_obj._scale)
    # Translate to world position
    tm.apply_translation(alien_obj.position)

    ray_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(tm)

    # Sample ray directions in a cone (same geometry as emitter)
    emitter_origin = np.array(EMITTER_POS, dtype=np.float64)
    beam_dir       = np.array(EMITTER_BEAM_DIR, dtype=np.float64)
    half_angle_rad = np.radians(EMITTER_BEAM_ANG)

    # Random directions within the cone using uniform spherical cap sampling
    rng  = np.random.default_rng(42)
    u, v = rng.random(num_rays), rng.random(num_rays)
    cos_a  = 1 - u * (1 - np.cos(half_angle_rad))
    theta  = np.arccos(np.clip(cos_a, -1, 1))
    phi    = 2 * np.pi * v
    sin_t  = np.sin(theta)
    dirs_local = np.column_stack([
        np.cos(theta),           # along beam axis
        sin_t * np.cos(phi),
        sin_t * np.sin(phi),
    ])  # (N, 3) in a frame where x=beam_dir

    # Rotate to beam direction frame (beam_dir = [1,0,0], so trivial here)
    # General case rotation handled below
    # Build rotation that maps [1,0,0] → beam_dir
    bx = beam_dir / np.linalg.norm(beam_dir)
    arb = np.array([0., 1., 0.]) if abs(bx[1]) < 0.9 else np.array([1., 0., 0.])
    by  = np.cross(bx, arb); by /= np.linalg.norm(by)
    bz  = np.cross(bx, by)
    R   = np.stack([bx, by, bz], axis=1)            # (3,3): columns are basis vecs
    ray_dirs = dirs_local @ R.T                      # (N, 3) in world space
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)

    ray_origins = np.tile(emitter_origin, (num_rays, 1))

    # Cast rays
    hit_locations, ray_indices, tri_indices = ray_mesh.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=False,         # closest hit only
    )

    if len(hit_locations) == 0:
        return np.empty((0, 3)), np.empty(0)

    # Compute angle of incidence at each hit (for coloring)
    hit_normals = tm.face_normals[tri_indices]
    hit_rays    = ray_dirs[ray_indices]
    cos_inc     = np.abs(np.einsum('ij,ij->i', hit_rays, hit_normals))
    angles      = np.degrees(np.arccos(np.clip(cos_inc, 0, 1)))

    return hit_locations, angles


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _aabb_wireframe(ax, center, extents, color='cyan', lw=1.2, alpha=0.5):
    """Draw a 3D wireframe AABB box."""
    c = np.asarray(center, dtype=float)
    h = np.asarray(extents, dtype=float) / 2.0
    corners = np.array([
        c + np.array([sx, sy, sz]) * h
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    edges = [
        (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color=color, lw=lw, alpha=alpha)


def plot_scene(sim, alien, acou_pts, acou_amps, hydrophones,
               mesh_pts=None, mesh_angles=None):
    """Render and save both output figures."""

    # — Normalise amplitudes for colour mapping —
    def _norm(arr):
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - a_min) / (a_max - a_min)

    # ─── Figure 1: Full 3D scene ──────────────────────────────────────────
    fig1 = plt.figure(figsize=(14, 10), facecolor='#0a0a1a')
    ax = fig1.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a1a')

    # coordinate axes labels
    ax.set_xlabel('X  [m]', color='white', labelpad=8)
    ax.set_ylabel('Y  [m]', color='white', labelpad=8)
    ax.set_zlabel('Z  [m]', color='white', labelpad=8)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    lx, ly, lz = sim.water.size
    ax.set_xlim(0, lx); ax.set_ylim(0, ly); ax.set_zlim(0, lz)
    ax.set_title("Acoustic Bistatic Sonar — AlienAnimal Scene", color='white', fontsize=13)

    # ┄ Water volume outline ┄
    _aabb_wireframe(ax, [lx/2, ly/2, lz/2], [lx, ly, lz], color='#1a4a6a', lw=0.6, alpha=0.3)

    # ┄ Object AABB ┄
    _aabb_wireframe(ax, alien.position, alien.size, color='cyan', lw=1.5, alpha=0.8)

    # ┄ Sample ray paths (first 200 rays that hit the object) ┄
    hit_rays = [r for r in sim.rays if 'AlienAnimal' in r.reflection_objects][:120]
    other_rays = [r for r in sim.rays if 'AlienAnimal' not in r.reflection_objects][:80]
    for ray in other_rays:
        pts_r = np.array(ray.path_points)
        if len(pts_r) > 1:
            ax.plot3D(pts_r[:,0], pts_r[:,1], pts_r[:,2],
                      color='#1a3a5a', lw=0.4, alpha=0.25)
    for ray in hit_rays:
        pts_r = np.array(ray.path_points)
        if len(pts_r) > 1:
            ax.plot3D(pts_r[:,0], pts_r[:,1], pts_r[:,2],
                      color='#00aaff', lw=0.6, alpha=0.5)

    # ┄ Acoustic reflection points on the object ┄
    if len(acou_pts) > 0:
        nc   = _norm(acou_amps)
        cols = cm.plasma(nc)
        ax.scatter(acou_pts[:,0], acou_pts[:,1], acou_pts[:,2],
                   c=cols, s=6, alpha=0.9, zorder=5, label=f'Acoustic hits ({len(acou_pts)})')

    # ┄ Emitter ┄
    ep = np.array(EMITTER_POS)
    ax.scatter(*ep, c='yellow', s=120, marker='*', zorder=10, label='Emitter')
    ax.text(ep[0]+1, ep[1], ep[2]+1, 'EMITTER', color='yellow', fontsize=7)

    # ┄ Hydrophone array ┄
    hp_arr = np.array([h.position for h in hydrophones])
    ax.scatter(hp_arr[:,0], hp_arr[:,1], hp_arr[:,2],
               c='lime', s=40, marker='^', zorder=10, label=f'Hydrophones ({len(hydrophones)})')

    ax.legend(facecolor='#111111', labelcolor='white', fontsize=8, loc='upper left')
    fig1.tight_layout()
    fig1.savefig(OUT_SCENE, dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
    print(f"  Saved: {OUT_SCENE}")

    # ─── Figure 2: Acoustic point-cloud ──────────────────────────────────
    ncols  = 3 if (mesh_pts is not None and len(mesh_pts) > 0) else 2
    fig2, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6),
                              facecolor='#0a0a1a',
                              subplot_kw={'projection': '3d'})
    fig2.suptitle("Acoustic Surface Point Cloud — AlienAnimal",
                  color='white', fontsize=14, y=0.98)

    panel_cfg = [
        ("Acoustic Hits\n(AABB physics)", acou_pts, acou_amps, 'plasma', 'amplitude'),
    ]
    if mesh_pts is not None and len(mesh_pts) > 0:
        panel_cfg.append(("Surface Hits\n(trimesh BVH)", mesh_pts, mesh_angles,
                          'inferno', 'incidence angle °'))

    # Third panel: combined Y-Z projection (front view)
    panel_cfg.append(None)  # placeholder for projection

    for col, cfg in enumerate(panel_cfg):
        ax2 = axes[col] if ncols > 1 else axes
        ax2.set_facecolor('#0a0a1a')
        ax2.tick_params(colors='#555555')

        if cfg is None:
            # Y-Z projection (front view) of AABB acoustic hits
            ax2 = axes[col]
            ax2.set_facecolor('#0a0a1a')
            ax2.remove()
            ax2_2d = fig2.add_subplot(1, ncols, col+1)
            ax2_2d.set_facecolor('#0a0a1a')
            ax2_2d.tick_params(colors='#555555')
            if len(acou_pts) > 0:
                nc2 = _norm(acou_amps)
                sc = ax2_2d.scatter(acou_pts[:,2], acou_pts[:,1],
                                    c=cm.plasma(nc2), s=8, alpha=0.8)
            if mesh_pts is not None and len(mesh_pts) > 0:
                ax2_2d.scatter(mesh_pts[:,2], mesh_pts[:,1],
                               c='#00aaff', s=1, alpha=0.15, label='mesh surface')
            ax2_2d.set_xlabel('Z  [m]', color='white')
            ax2_2d.set_ylabel('Y  [m]', color='white')
            ax2_2d.set_title('Front View\n(Y–Z projection)', color='white', fontsize=10)
            ax2_2d.set_aspect('equal', 'box')
            ax2_2d.spines[['bottom','left']].set_color('#555555')
            ax2_2d.spines[['top','right']].set_visible(False)
            for lbl in ax2_2d.get_xticklabels() + ax2_2d.get_yticklabels():
                lbl.set_color('#888888')
            continue

        title_txt, pts, vals, cmap, cbar_label = cfg
        if len(pts) == 0:
            ax2.set_title(f"{title_txt}\n(no hits)", color='#888888', fontsize=10)
            continue

        nc3 = _norm(vals)
        cols3 = plt.colormaps[cmap](nc3)
        ax2.scatter(pts[:,0], pts[:,1], pts[:,2], c=cols3, s=4, alpha=0.8)
        _aabb_wireframe(ax2, OBJECT_POSITION,
                        [alien.size[0], alien.size[1], alien.size[2]],
                        color='cyan', lw=1.0, alpha=0.5)
        ax2.set_xlabel('X [m]', color='white', fontsize=8, labelpad=4)
        ax2.set_ylabel('Y [m]', color='white', fontsize=8, labelpad=4)
        ax2.set_zlabel('Z [m]', color='white', fontsize=8, labelpad=4)
        ax2.set_title(title_txt, color='white', fontsize=10)
        ax2.tick_params(colors='#555555')
        # Small colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vals.min(), vals.max()))
        sm.set_array([])
        cb = fig2.colorbar(sm, ax=ax2, shrink=0.5, pad=0.12)
        cb.set_label(cbar_label, color='white', fontsize=7)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=6)

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(OUT_CLOUD, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"  Saved: {OUT_CLOUD}")

    # Only show interactively if not running with a non-GUI backend (e.g. Agg)
    import matplotlib
    if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  AlienAnimal Acoustic Point Cloud Demo")
    print("=" * 65)

    # ── Load mesh ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading mesh: {OBJ_FILE.name}")
    mesh_data = MeshImporter.load(str(OBJ_FILE))
    world_extents = mesh_data.extents * OBJECT_SCALE
    print(f"       Raw mesh    : {mesh_data}")
    print(f"       World size  : {world_extents.round(2)} m  (scale={OBJECT_SCALE})")
    print(f"       Placed at   : {OBJECT_POSITION} m")

    # ── Build & run simulation ────────────────────────────────────────────
    print(f"\n[2/5] Building simulation  ({NUM_RAYS} rays, AABB mode)")
    sim, alien, hydrophones = build_simulation(mesh_data)
    total_hydros = len(hydrophones)
    print(f"       Hydrophone array : {total_hydros} sensors "
          f"(3×3 grid at x={HYDRO_X:.0f} m)")

    print(f"\n[3/5] Running acoustic ray trace ...")
    sim.run(num_rays_per_emitter=NUM_RAYS, verbose=False)

    # ── Collect hits ──────────────────────────────────────────────────────
    print(f"\n[4/5] Collecting acoustic reflection data ...")
    acou_pts, acou_amps = collect_alien_hits(sim, object_name='AlienAnimal')

    total_arrivals = sum(h.get_arrival_count() for h in hydrophones)
    hit_rays_count = sum(1 for r in sim.rays if 'AlienAnimal' in r.reflection_objects)

    print(f"       Total rays traced      : {NUM_RAYS}")
    print(f"       Rays hitting AlienAnimal: {hit_rays_count}")
    print(f"       Acoustic hit points    : {len(acou_pts)}")
    print(f"       Hydrophone arrivals    : {total_arrivals}")

    if len(acou_amps) > 0:
        print(f"       Amplitude range        : "
              f"{acou_amps.min():.3e}  →  {acou_amps.max():.3e}")

    # Hydrophone summary
    print(f"\n       Hydrophone signal summary:")
    for h in sorted(hydrophones, key=lambda h: -h.get_arrival_count())[:5]:
        fa = h.get_first_arrival()
        print(f"         {h.name:20s}  arrivals={h.get_arrival_count():4d}  "
              f"first={fa['time']*1000:.3f} ms" if fa else
              f"         {h.name:20s}  arrivals=0")

    # ── Trimesh exact surface mapping ─────────────────────────────────────
    mesh_pts = mesh_angles = None
    if TRIMESH_OK:
        print(f"\n[5/5] Trimesh BVH surface mapping (8,000 rays) ...")
        mesh_pts, mesh_angles = trimesh_surface_hits(mesh_data, alien, num_rays=8000)
        if mesh_pts is not None:
            print(f"       Exact surface hits : {len(mesh_pts)}")
        else:
            print(f"       No surface hits found.")
    else:
        print(f"\n[5/5] Skipping trimesh surface mapping (not installed).")

    # ── Visualise ─────────────────────────────────────────────────────────
    print(f"\n[6/5] Rendering figures ...")
    plot_scene(sim, alien, acou_pts, acou_amps, hydrophones, mesh_pts, mesh_angles)

    print("\nDone.")
    print(f"  {OUT_SCENE.name}")
    print(f"  {OUT_CLOUD.name}")


if __name__ == '__main__':
    main()
