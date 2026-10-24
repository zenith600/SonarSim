"""
build_docs.py
=============
Generates  acousticsim_documentation.docx  — the full technical reference for
the acousticsim package, including physics derivations, module API, usage
examples and the AlienAnimal demo screenshots.

Run:
    python build_docs.py
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
IMG_SCENE   = ROOT / "alien_pointcloud_scene.png"
IMG_SURFACE = ROOT / "alien_pointcloud_surface.png"
OUT_PATH    = ROOT / "acousticsim_documentation.docx"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def h1(doc, text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1A, 0x5C, 0x9B)
    return p

def h2(doc, text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = RGBColor(0x1A, 0x7A, 0x5C)
    return p

def h3(doc, text):
    return doc.add_heading(text, level=3)

def body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    return p

def bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    return p

def code_block(doc, text):
    for line in text.strip().split('\n'):
        p = doc.add_paragraph(line)
        p.style = doc.styles['Normal']
        run = p.runs[0] if p.runs else p.add_run()
        run.font.name  = 'Courier New'
        run.font.size  = Pt(9)
        run.font.color.rgb = RGBColor(0x10, 0x10, 0x60)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(0)
        shading = OxmlElement('w:shd')
        shading.set(qn('w:val'),   'clear')
        shading.set(qn('w:color'), 'auto')
        shading.set(qn('w:fill'),  'F0F4FF')
        p._p.get_or_add_pPr().append(shading)

def kv(doc, key, value):
    p = doc.add_paragraph()
    r = p.add_run(f"{key}:  ")
    r.bold = True
    p.add_run(value)
    p.paragraph_format.space_after = Pt(3)

def add_image(doc, path, caption, width=Inches(5.8)):
    if path.exists():
        doc.add_picture(str(path), width=width)
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)
    else:
        body(doc, f"[Image not found: {path.name}]")

def divider(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bottom)
    pPr.append(pBdr)

def page_break(doc):
    doc.add_page_break()


# ─────────────────────────────────────────────────────────────────────────────
# Document
# ─────────────────────────────────────────────────────────────────────────────

doc = Document()

# Global styling
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

sections = doc.sections
for section in sections:
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)

# ═══════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════════════════

title = doc.add_heading('acousticsim', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.runs[0].font.color.rgb = RGBColor(0x1A, 0x5C, 0x9B)

sub = doc.add_paragraph('Underwater Acoustic Ray-Tracing Simulator')
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(16)
sub.runs[0].bold = True

sub2 = doc.add_paragraph('Technical Documentation  •  v0.1.0')
sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub2.runs[0].font.size = Pt(12)
sub2.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

date_p = doc.add_paragraph(datetime.datetime.now().strftime('%B %Y'))
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.runs[0].font.size = Pt(11)
date_p.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)

doc.add_paragraph('')
doc.add_paragraph('')

if IMG_SCENE.exists():
    add_image(doc, IMG_SCENE,
              'Figure 1 — AlienAnimal bistatic sonar demo: 3D scene overview')

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS  (manual)
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, 'Table of Contents')
toc_entries = [
    ('1', 'Overview'),
    ('2', 'System Architecture'),
    ('3', 'Coordinate System'),
    ('4', 'Physics Engine'),
    ('  4.1', 'Acoustic Wave Fundamentals'),
    ('  4.2', 'Sound Speed — Mackenzie Equation'),
    ('  4.3', 'Absorption — Thorp Equation'),
    ('  4.4', 'Ray Theory'),
    ('  4.5', 'Reflection Physics'),
    ('  4.6', 'Refraction and Snell\'s Law'),
    ('  4.7', 'Phase and Interference'),
    ('  4.8', 'Multipath Propagation'),
    ('5', 'Module Reference'),
    ('  5.1', 'Water'),
    ('  5.2', 'Ray'),
    ('  5.3', 'WaveEmitter'),
    ('  5.4', 'Hydrophone'),
    ('  5.5', 'Object3D, Surface, Seabed'),
    ('  5.6', 'Simulation'),
    ('  5.7', 'MeshImporter & ImportedObject3D'),
    ('  5.8', 'NoiseModel'),
    ('  5.9', 'Feature Extractor'),
    ('  5.10', 'DatasetGenerator & AcousticDataset'),
    ('  5.11', 'SklearnDetector & PyTorchDetector'),
    ('  5.12', 'AcousticDetector (inference)'),
    ('6', 'AlienAnimal Demo Walkthrough'),
    ('7', 'Data Persistence — Save & Load'),
    ('  7.1', 'File Format Overview'),
    ('  7.2', 'Training Dataset (.npz)'),
    ('  7.3', 'sklearn Detector (.pkl)'),
    ('  7.4', 'PyTorch Detector (.pt)'),
    ('  7.5', 'Full File Lifecycle'),
    ('  7.6', 'Inspecting a Dataset Without acousticsim'),
    ('8', 'Machine Learning Pipeline'),
    ('9', 'Quick Usage Guide'),
    ('10', 'Installation'),
]
for num, title_text in toc_entries:
    p = doc.add_paragraph()
    r = p.add_run(f'{num}')
    r.bold = True
    r.font.size = Pt(10)
    p.add_run(f'    {title_text}')
    p.paragraph_format.space_after = Pt(2)

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '1. Overview')
body(doc,
    'acousticsim is a Python package for 3-D underwater acoustic ray-tracing simulation. '
    'It models the propagation of high-frequency sound (30 – 200 kHz) through a '
    'stratified water column with realistic physics: depth- and temperature-dependent '
    'sound speed, frequency-dependent absorption, angle-dependent reflection and '
    'refraction, multipath arrival detection at hydrophone arrays, and full '
    'per-bounce metadata for building acoustic point clouds.')

body(doc,
    'Beyond raw simulation the package includes a complete machine-learning pipeline: '
    '3-D mesh import (STL / OBJ / GLTF), randomised dataset generation, sklearn '
    'RandomForest baseline and a PyTorch multi-task MLP for simultaneous object '
    'classification and 6-DoF pose estimation.')

h2(doc, 'Key Capabilities')
bullets = [
    'Physics-accurate 3-D ray tracing (Mackenzie sound speed, Thorp absorption)',
    'AABB slab-method intersection (fast) and Möller–Trumbore triangle mesh intersection (accurate)',
    'Spherical and cone beam emission; continuous-wave and chirp waveforms',
    'Per-hydrophone multipath detection with SNR-aware noise floor',
    'Object-tagged reflection points — acoustic point cloud output',
    '3-D mesh import from STL, OBJ, GLTF/GLB via trimesh',
    'Four underwater noise models: white, thermal, ambient (Wenz), shipping',
    'ML dataset generation → sklearn RF → PyTorch MLP (class + 6-DoF pose)',
    'Full PyPI-ready packaging (pyproject.toml, PEP 517)',
]
for b in bullets:
    bullet(doc, b)

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '2. System Architecture')
body(doc,
    'The package is organised as a single importable namespace acousticsim with '
    'twelve sub-modules. The dependency graph flows from environment (Water) through '
    'emitter → rays → objects → hydrophones → simulation, and then optionally into '
    'the ML stack.')

h2(doc, 'Package Structure')
code_block(doc, """\
acousticsim/
├── __init__.py          # public API, re-exports all symbols
├── water.py             # Water environment — sound speed, absorption
├── ray.py               # Ray  — propagation state, per-bounce metadata
├── emitter.py           # WaveEmitter — spherical / cone beam, chirp
├── hydrophone.py        # Hydrophone — multipath detection, noise injection
├── object3d.py          # Object3D, Surface, Seabed — AABB primitives
├── simulation.py        # Simulation — ray-trace engine, BVH loop
├── mesh_importer.py     # MeshImporter — STL/OBJ/GLTF loader via trimesh
├── imported_object3d.py # ImportedObject3D — AABB or Möller-Trumbore mesh
├── noise.py             # NoiseModel — white/thermal/ambient/shipping
├── feature_extractor.py # extract_features() — hydrophone → ML vector
├── dataset_generator.py # DatasetGenerator — randomised simulation runs
├── train_detector.py    # SklearnDetector, PyTorchDetector, AcousticMLP
└── inference.py         # AcousticDetector — deploy / infer from mesh""")

h2(doc, 'Data Flow')
code_block(doc, """\
Water (environment)
    │
    ▼
WaveEmitter  ──emits──►  [Ray, Ray, ...]
                              │
                    Simulation._trace_ray()
                              │   intersect every Object3D / ImportedObject3D
                              │   ray.reflect(object_name=...)  ← logs hit
                              │   ray.refract() / propagation step
                              ▼
                         Ray arrives near Hydrophone
                              │  detect_ray()  →  arrival dict stored
                              ▼
                         get_multipath_signal(noise_model=...) → signal array
                         get_amplitudes(noise_model=...)       → amplitude list
                              │
                    feature_extractor.extract_features()  → 1-D float32 vector
                              │
                    DatasetGenerator  →  AcousticDataset (.npz)
                              │
                    train_sklearn / train_pytorch  →  detector checkpoint
                              │
                    AcousticDetector.detect_from_hydrophones()  →  prediction""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 3. COORDINATE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '3. Coordinate System')
body(doc,
    'All positions and dimensions are in metres. The simulator uses a right-handed '
    '3-D Cartesian coordinate system with the following convention:')

kv(doc, 'X-axis', 'Width — left to right')
kv(doc, 'Y-axis', 'Length — front to back')
kv(doc, 'Z-axis', 'Depth — positive downward (oceanographic convention)')
kv(doc, 'Origin', 'Front-left corner of the water volume, at the surface (Z = 0)')

body(doc, '')
code_block(doc, """\
          Y (Length)
          ↑
          |
          |________→ X (Width)
         /
        ↓
       Z (Depth, positive = deeper)

Z = 0    → Water surface
Z = 25   → 25 m below surface
Z = 50   → Seabed (for a 50 m deep environment)""")

body(doc, '')
body(doc,
    'Object positions refer to the geometric centre of the bounding box. '
    'A block placed at [30, 50, 10] with size [4, 6, 3] spans '
    'X: 28–32, Y: 47–53, Z: 8.5–11.5 m.')

divider(doc)

h2(doc, 'Direction Vectors')
body(doc, 'Normalised unit vectors describe ray and beam directions:')
code_block(doc, """\
[1, 0, 0]   →  +X direction (right)
[-1, 0, 0]  →  -X direction (left)
[0, 1, 0]   →  +Y direction (forward)
[0, 0, 1]   →  +Z direction (downward)
[0.577, 0.577, 0.577]  →  diagonal (normalise before use)""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 4. PHYSICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '4. Physics Engine')

# ─── 4.1 ──────────────────────────────────────────────────────────────────
h2(doc, '4.1 Acoustic Wave Fundamentals')
body(doc,
    'Sound in water is a longitudinal pressure wave. The simulator models '
    'acoustic energy propagation in the 30 – 200 kHz sonar band.')

body(doc, 'Core relationships:')
code_block(doc, """\
λ = c / f          wavelength [m] = sound speed [m/s] / frequency [Hz]
T = 1 / f          period [s]
φ = 2π × d / λ    phase shift [rad] over distance d [m]
E ∝ A²             energy proportional to amplitude squared""")

body(doc, '')
body(doc,
    'Amplitude is tracked linearly throughout propagation. dB values are used '
    'only at the emitter source level and in the noise floor comparison:')
code_block(doc, """\
A = 10^(SL_dB / 20)      linear amplitude from source level dB
dB = 20 × log10(A / A_ref)""")

# ─── 4.2 ──────────────────────────────────────────────────────────────────
h2(doc, '4.2 Sound Speed — Mackenzie Equation')
body(doc, 'The Mackenzie (1981) nine-term equation gives sound speed as a function of temperature T (°C), depth D (m) and salinity S (ppt):')
code_block(doc, """\
c = 1448.96
  + 4.591  T
  - 5.304e-2 T²
  + 2.374e-4 T³
  + 1.340 (S - 35)
  + 1.630e-2 D
  + 1.675e-7 D²
  - 1.025e-2 T (S - 35)
  - 7.139e-13 T D³       [m/s]""")

body(doc,
    'Temperature is modelled as a linear vertical gradient between the user-supplied '
    'surface and bottom temperatures. A horizontal sound-speed gradient is computed '
    'numerically for refraction calculations.')

body(doc, 'Typical values for this simulator\'s default environment (T_surf=18°C, T_bottom=8°C, S=35 ppt):')
code_block(doc, """\
Surface (Z=0,  T=18°C):  c ≈ 1516 m/s
Mid     (Z=25, T=13°C):  c ≈ 1505 m/s
Bottom  (Z=50, T=8°C):   c ≈ 1496 m/s""")

# ─── 4.3 ──────────────────────────────────────────────────────────────────
h2(doc, '4.3 Absorption — Thorp Equation')
body(doc, 'Acoustic energy is absorbed by the water column according to the Thorp model, valid across 30 – 200 kHz:')
code_block(doc, """\
α(f) = 0.11 f² / (1 + f²)        ← ionic relaxation
     + 44 f² / (4100 + f²)       ← MgSO₄ relaxation
     + 2.75×10⁻⁴ f²              ← viscosity
     + 0.003                      ← structural

Result in dB/km, where f is in kHz.""")

body(doc, 'Worked examples:')
code_block(doc, """\
f = 30 kHz  →  α ≈  8.7 dB/km
f = 85 kHz  →  α ≈ 27.6 dB/km
f = 200 kHz →  α ≈ 54.8 dB/km""")

body(doc,
    'Absorption loss is applied to the ray amplitude every simulation step: '
    'A_new = A × 10^(−α × Δd / 20 000) where Δd is the step distance in metres.')

# ─── 4.4 ──────────────────────────────────────────────────────────────────
h2(doc, '4.4 Ray Theory')
body(doc,
    'Ray theory is valid when the acoustic wavelength is much smaller than '
    'object dimensions (λ << L). For the 30–200 kHz band, λ = 7.5 mm – 50 mm, '
    'which satisfies ray conditions for objects larger than ~5 cm.')

body(doc, 'Each Ray object carries:')
code_block(doc, """\
position          current 3-D location [m]
direction         unit vector of travel
amplitude         linear pressure amplitude
phase             accumulated phase [rad]
path_length       total distance travelled [m]
frequency_khz     carrier frequency
bounces           number of reflections so far
reflection_points list of (x,y,z) hit locations
reflection_objects parallel list of object names hit
reflection_amplitudes incident amplitude at each bounce""")

body(doc,
    'Geometric spreading is not applied as an explicit 1/r² factor between steps; '
    'instead, each emitter fires N rays and the total source power is distributed '
    'uniformly: A_per_ray = A_source / √N. Cylindrical spreading from the ray to '
    'a hydrophone is applied at detection time.')

# ─── 4.5 ──────────────────────────────────────────────────────────────────
h2(doc, '4.5 Reflection Physics')
body(doc, 'When a ray strikes an Object3D or ImportedObject3D surface:')
body(doc, '1.  The incident amplitude before energy loss is captured.')
body(doc, '2.  Effective reflection coefficient is computed from the base coefficient '
          'and optional frequency-dependent absorption tables.')
body(doc, '3.  The reflected amplitude is applied:')
code_block(doc, """\
A_reflected = A_incident × R_eff

where R_eff ∈ [0, 1]  (1 = perfect reflection, 0 = full absorption)""")
body(doc, '4.  The reflected direction is computed using the geometric law of reflection:')
code_block(doc, """\
d_reflected = d_incident - 2 (d_incident · n̂) n̂

where n̂ is the outward surface normal.""")
body(doc, '5.  Optional phase inversion (rigid boundary): φ += π')
body(doc, '6.  The hit point, object name and incident amplitude are appended to the '
          'ray\'s parallel metadata lists.')

body(doc, 'Typical reflection coefficients:')
code_block(doc, """\
Steel / aluminium hull  : 0.95 – 0.98
Rock seabed             : 0.85 – 0.92
Sandy seabed            : 0.70 – 0.85
Soft tissue / flesh     : 0.30 – 0.60
Open water surface      : 0.90 – 0.99  (angle-dependent)""")

# ─── 4.6 ──────────────────────────────────────────────────────────────────
h2(doc, '4.6 Refraction and Snell\'s Law')
body(doc,
    'When the sound speed changes with depth, rays bend according to Snell\'s law. '
    'The simulator applies a small incremental refraction correction at each step '
    'by projecting the local sound-speed gradient perpendicular to the ray direction:')
code_block(doc, """\
∇c = (∂c/∂x, ∂c/∂y,  ∂c/∂z)   [computed numerically, Δ=0.1 m]

Bending correction per step Δd:
  d_new = d + (∇c - (∇c · d̂) d̂) × (Δd / c)
  d_new = d_new / |d_new|        [re-normalize]""")

body(doc,
    'For the default linear temperature gradient, the dominant component is ∂c/∂z '
    '≈ −0.2 m/s per metre of depth, causing rays to bend toward the deeper, '
    'slower region (downward refraction for the warm-surface case).')

# ─── 4.7 ──────────────────────────────────────────────────────────────────
h2(doc, '4.7 Phase and Interference')
body(doc, 'Phase accumulates continuously along each ray:')
code_block(doc, """\
Δφ = 2π × Δd / λ     per step Δd
λ  = c_local / f

Additionally:
  +π   phase shift on reflection from a rigid boundary (optional)
  +2π×(r_perp / λ)  extra phase when spreading to hydrophone position""")

body(doc,
    'Constructive interference occurs when two arrivals satisfy '
    '|Δφ| = 2πk (k integer). '
    'Destructive interference occurs at |Δφ| = (2k+1)π. '
    'get_multipath_signal() superposes all arrivals coherently.')

# ─── 4.8 ──────────────────────────────────────────────────────────────────
h2(doc, '4.8 Multipath Propagation')
body(doc,
    'Multiple ray paths connect the emitter to a hydrophone: direct path, '
    'surface-reflected, seabed-reflected, object-reflected, and combinations. '
    'Each path arrives at a different time and amplitude, creating the '
    'characteristic spread of arrivals measured in real sonar:')
code_block(doc, """\
ToA_i = path_length_i / c_avg + r_perp_i / c_avg    [seconds]

The multipath signal is:
  s(t) = Σ_i  A_i × sin(2π f (t - ToA_i)  +  φ_i)""")

body(doc,
    'The Delay Spread (std of ToA values) and Time-of-Arrival Differences (TDOA) '
    'between hydrophone pairs are the primary spatial features used by the '
    'ML detector.')

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 5. MODULE REFERENCE
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '5. Module Reference')

# ── 5.1 Water ─────────────────────────────────────────────────────────────
h2(doc, '5.1 Water')
body(doc, 'Defines the water volume and its physical properties.')

code_block(doc, """\
Water(
    size                = (lx, ly, lz),   # volume dimensions [m]
    temperature_surface = 20.0,           # surface temperature [°C]
    temperature_bottom  = 10.0,           # bottom temperature [°C]
    salinity            = 35.0,           # salinity [ppt]
    ambient_noise_level = 50.0,           # dB re 1 μPa
)""")

h3(doc, 'Key Methods')
kv(doc, 'sound_speed_at(x, y, z)', 'Returns c [m/s] via Mackenzie equation at position')
kv(doc, 'absorption_thorp(frequency_khz)', 'Returns α [dB/km] via Thorp model')
kv(doc, 'temperature_at(x, y, z)', 'Linear interpolated temperature [°C]')
kv(doc, 'sound_speed_gradient(x, y, z)', 'Numerical ∇c for refraction [m/s per m]')
kv(doc, 'is_inside(x, y, z)', 'True if point is within the water volume')

# ── 5.2 Ray ───────────────────────────────────────────────────────────────
h2(doc, '5.2 Ray')
body(doc, 'Tracks a single acoustic ray: position, direction, amplitude, phase, bounce history and object-tagged reflection metadata.')

code_block(doc, """\
Ray(
    position,              # (x, y, z) starting point
    direction,             # (dx, dy, dz) unit vector
    frequency_khz = 85.0,
    amplitude     = 1.0,
    wave_type     = 'continuous',   # or 'chirp'
    frequency_range = (f_min, f_max),
)""")

h3(doc, 'Key Methods')
kv(doc, 'propagate(distance, water)', 'Move the ray, apply absorption and refraction')
kv(doc, 'reflect(normal, R, phase_inversion, object_name)', 'Bounce off a surface; records hit to metadata lists')
kv(doc, 'get_reflection_points(object_name=None)', 'All bounce hit points, optionally filtered by object name')
kv(doc, 'get_reflection_objects()', 'List of object names struck at each bounce')
kv(doc, 'get_reflection_amplitudes()', 'Incident amplitude at each bounce (before loss)')

# ── 5.3 WaveEmitter ───────────────────────────────────────────────────────
h2(doc, '5.3 WaveEmitter')
body(doc, 'Generates a list of Ray objects according to the selected beam pattern.')

code_block(doc, """\
WaveEmitter(
    position,
    frequency_khz  = 85.0,
    beam_pattern   = 'spherical',   # or 'cone'
    beam_direction = [1, 0, 0],
    beam_angle     = 30.0,          # half-angle [deg] for cone
    power_db       = 180.0,         # source level [dB re 1 μPa @ 1 m]
    wave_type      = 'continuous',  # or 'chirp'
    frequency_range = None,         # (f_min, f_max) kHz for chirp
    chirp_duration  = 0.01,         # seconds
)

emitter.emit_rays(num_rays=1000)  →  list[Ray]""")

body(doc,
    'Spherical pattern: Fibonacci lattice ensures uniform solid-angle coverage. '
    'Cone pattern: uniform distribution within a specified half-angle using the '
    'cos(α) = Uniform(cos(θ_max), 1) inverse-CDF method.')

# ── 5.4 Hydrophone ────────────────────────────────────────────────────────
h2(doc, '5.4 Hydrophone')
body(doc, 'A point sensor that detects acoustic arrivals from nearby rays.')

code_block(doc, """\
Hydrophone(
    position,
    name             = None,
    sensitivity_db   = -170.0,   # [dB re 1V/μPa]
)

# Key attributes
hydrophone.min_amplitude        = 1e-7   # noise floor
hydrophone.max_detection_distance = 50.0 # perpendicular detection radius [m]""")

h3(doc, 'Key Methods')
kv(doc, 'detect_ray(ray, sound_speed)', 'Test one Ray; stores arrival if amplitude > min_amplitude')
kv(doc, 'get_time_of_arrivals()', 'List of ToA values [s]')
kv(doc, 'get_amplitudes(noise_model=None)', 'List of amplitudes; optional NoiseModel perturbs each value')
kv(doc, 'get_multipath_signal(time_samples, duration, noise_model, sample_rate)',
        'Construct coherent multipath waveform; optional NoiseModel adds shaped background noise')
kv(doc, 'inject_arrivals(noise_model, n, frequency_khz, time_range)', '(via NoiseModel) Inject synthetic noise arrivals')
kv(doc, 'get_first_arrival() / get_strongest_arrival()', 'Convenience accessors')

body(doc,
    'Detection model: cylindrical spreading from the ray path to the sensor. '
    'Amplitude at hydrophone = ray.amplitude × 1/√(max(1, r_perp)). '
    'Time of arrival = (path_length + r_perp) / c.')

# ── 5.5 Object3D ──────────────────────────────────────────────────────────
h2(doc, '5.5 Object3D, Surface, Seabed')
body(doc, 'AABB (Axis-Aligned Bounding Box) primitives for walls, reflectors and the environment boundary.')

code_block(doc, """\
Object3D(
    name, position, size,
    reflection_coefficient = 0.9,
    is_rigid               = True,
    frequency_absorption   = {(30,50): 0.8, (50,80): 0.5},
)

Surface(water)   # water–air interface at Z=0 (R ≈ 0.99)
Seabed(water, reflection_coefficient=0.85)  # bottom boundary""")

body(doc, 'Intersection: slab method — six half-space tests. O(1) per ray per object.')
body(doc, 'Surface normal is determined by which AABB face is hit; direction sign is chosen so the normal points toward the incoming ray.')

# ── 5.6 Simulation ────────────────────────────────────────────────────────
h2(doc, '5.6 Simulation')
body(doc, 'The main ray-tracing engine. Manages the world of objects, emitters and hydrophones and drives the propagation loop.')

code_block(doc, """\
sim = Simulation(water)
sim.max_bounces = 8
sim.step_size   = 0.5   # m per propagation step

sim.add_object(object3d)
sim.add_emitter(emitter)
sim.add_hydrophone(hydrophone)

results = sim.run(num_rays=1000)

# After run():
sim.get_reflection_points(object_name='AlienAnimal')  # filtered hit list
sim.rays                                               # all Ray objects""")

h3(doc, 'Trace Loop (per ray)')
body(doc, 'For each step, the engine:')
code_block(doc, """\
1. ray.propagate(step_size, water)         ← move, absorb, refract
2. For every Object3D: test ray_intersect()
3. Pick nearest hit within the step
4. If hit: ray.reflect(..., object_name=name)
           ray.position = hit_point + ε × normal   ← push off surface
5. Test all Hydrophones: detect_ray()
6. Stop if ray.amplitude < threshold or
           ray.bounces  > max_bounces  or
           ray outside water volume""")

# ── 5.7 MeshImporter / ImportedObject3D ───────────────────────────────────
h2(doc, '5.7 MeshImporter & ImportedObject3D')
body(doc, 'Load arbitrary 3-D meshes and use them as acoustic reflectors.')

code_block(doc, """\
from acousticsim import MeshImporter, ImportedObject3D

mesh = MeshImporter.load('AlienAnimal.obj', scale=1.0)
# mesh.vertices, mesh.faces, mesh.normals, mesh.bounds

obj = ImportedObject3D(
    mesh_data         = mesh,
    position          = [50, 50, 25],
    rotation_deg      = (0, 0, 0),      # roll, pitch, yaw
    scale             = 1.0,
    intersection_mode = 'aabb',         # or 'mesh'
    reflection_coefficient = 0.80,
    name              = 'AlienAnimal',
)""")

body(doc, '')
body(doc, 'Two intersection modes:')
code_block(doc, """\
'aabb'   — axis-aligned bounding box slab test.
           Very fast; suitable for physics pass with many rays.

'mesh'   — vectorised Möller–Trumbore algorithm against every triangle.
           Exact per-facet hit; suitable for small ray counts or
           high-fidelity surface mapping.""")

body(doc, 'ImportedObject3D.get_pose() returns a (6,) array: [x, y, z, roll°, pitch°, yaw°].')

# ── 5.8 NoiseModel ────────────────────────────────────────────────────────
h2(doc, '5.8 NoiseModel')
body(doc, 'Generates spectrally-shaped underwater ambient noise for signal corruption or arrival injection.')

code_block(doc, """\
from acousticsim import NoiseModel, add_noise_to_signal

nm = NoiseModel(
    noise_type      = 'ambient',  # 'white' | 'thermal' | 'ambient' | 'shipping'
    snr_db          = 15.0,       # dB SNR relative to input signal; or None
    noise_amplitude = 1e-6,       # absolute RMS when snr_db is None
    seed            = 42,         # reproducibility
)

# Add to a signal array
noisy = add_noise_to_signal(clean_signal, nm, sample_rate=250_000)

# Use via Hydrophone
t, noisy_sig = hydrophone.get_multipath_signal(noise_model=nm)
noisy_amps   = hydrophone.get_amplitudes(noise_model=nm)

# Inject fake noise arrivals into arrival list
nm.inject_arrivals(hydrophone, n_noise_arrivals=5)""")

h3(doc, 'Noise Spectra')
code_block(doc, """\
'white'    flat Gaussian, all frequencies equal power.
'thermal'  rises ∝ √f above 1 kHz — models hydrophone electronics.
'ambient'  Wenz model: 1/f below 300 Hz, flat above — ocean background.
'shipping' steep low-frequency content below 200 Hz — shipping / industrial.""")

body(doc,
    'When snr_db is provided and a reference_signal is passed to generate(), '
    'the noise amplitude is scaled so that SNR = snr_db dB relative to the '
    'RMS of that reference signal.')

# ── 5.9 Feature Extractor ──────────────────────────────────────────────────
h2(doc, '5.9 Feature Extractor')
body(doc, 'Converts raw hydrophone arrival lists into a fixed-size float32 ML feature vector.')

code_block(doc, """\
from acousticsim import extract_features

vec, names = extract_features(hydrophones, sound_speed=1500.0)
# vec.shape = (H*8 + C(H,2),)  e.g. H=9 → 72+36 = 108 features""")

h3(doc, 'Feature Layout')
code_block(doc, """\
Per hydrophone h (×8):
    h_n_arrivals          total arrivals
    h_first_time_ms       earliest ToA [ms]
    h_last_time_ms        latest ToA [ms]
    h_delay_spread_ms     std of ToA values
    h_max_amplitude       peak amplitude
    h_total_energy        Σ A²
    h_amplitude_mean      mean amplitude
    h_amplitude_std       std of amplitudes

Cross-hydrophone TDOA pairs i<j (×C(H,2)):
    tdoa_ij_ms = first_time[i] - first_time[j]""")

# ── 5.10 DatasetGenerator ─────────────────────────────────────────────────
h2(doc, '5.10 DatasetGenerator & AcousticDataset')
body(doc, 'Runs many simulation episodes with randomised object positions and rotations to build a labelled training dataset.')

code_block(doc, """\
from acousticsim import DatasetGenerator, DatasetConfig

config = DatasetConfig(
    num_samples    = 500,
    num_rays       = 300,
    num_hydrophones= 5,
    frequency_khz  = 85.0,
    noise_model    = None,   # optional NoiseModel
)

gen     = DatasetGenerator(mesh_dict={'target': mesh_data}, config=config)
dataset = gen.generate()

dataset.save('training_data.npz')
dataset = AcousticDataset.load('training_data.npz')
# dataset.X  — feature matrix  (N × F)
# dataset.y_class — class labels (N,)
# dataset.y_pose  — pose labels  (N, 6)""")

# ── 5.11 Detectors ────────────────────────────────────────────────────────
h2(doc, '5.11 SklearnDetector & PyTorchDetector')
body(doc, 'Two ready-to-train detector backends.')

code_block(doc, """\
from acousticsim import SklearnDetector, PyTorchDetector
from acousticsim import train_sklearn, train_pytorch, evaluate

# sklearn Random Forest
sk_det = train_sklearn(dataset)
metrics = evaluate(sk_det, test_dataset)

# PyTorch multi-task MLP
pt_det = train_pytorch(
    dataset,
    hidden_dims     = [256, 128, 64],
    lr              = 1e-3,
    epochs          = 100,
    patience        = 10,
)""")

h3(doc, 'AcousticMLP Architecture')
code_block(doc, """\
Input layer  (F features)
     │
     ▼
Shared backbone: Linear → BatchNorm → ReLU → Dropout  (×N layers)
     │
     ├──► Classification head  → softmax  → class probabilities
     │
     └──► Pose regression head → linear   → (x,y,z,roll,pitch,yaw)

Loss = CrossEntropy(class) + λ × MSE(pose)   [λ default = 0.1]""")

# ── 5.12 AcousticDetector ─────────────────────────────────────────────────
h2(doc, '5.12 AcousticDetector (Inference)')
body(doc, 'High-level deployment wrapper for either backend.')

code_block(doc, """\
from acousticsim import AcousticDetector

det = AcousticDetector.from_sklearn(sk_det, class_names=['target'])
# or
det = AcousticDetector.from_pytorch(pt_det, class_names=['target'])

# Simulate from mesh file and predict
pred = det.simulate_and_detect(
    mesh_filepath = 'AlienAnimal.obj',
    position      = [50, 50, 25],
    rotation_deg  = [0, 0, 15],
)
print(pred['class'], pred['pose_xyz'], pred['confidence'])

# Predict from already-populated hydrophones
pred = det.detect_from_hydrophones(hydrophones)

# Batch mode
preds = det.detect_batch(mesh_filepaths, positions)""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 6. ALIANANIMAL DEMO
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '6. AlienAnimal Demo Walkthrough')
body(doc,
    'alien_demo.py demonstrates a bistatic sonar configuration where an emitter '
    'fires acoustic rays at the AlienAnimal.obj mesh and a hydrophone array on the '
    'opposite side captures reflections. The demo produces two output images '
    'and prints detailed statistics.')

h2(doc, '6.1 Scene Configuration')
code_block(doc, """\
Water volume      : 100 × 100 × 50 m  (coastal bay)
AlienAnimal.obj   : trimesh import, scale=1.0
                    extents ≈ 17.9 × 19.6 × 39.3 m
                    position [50, 50, 25]

Emitter           : [5, 50, 25]  (left wall)
                    85 kHz  185 dB  cone beam  ± 25°  pointing +X
                    1 200 rays

Hydrophone array  : 3 × 3 grid at X=95 (right wall)
                    Y: 35, 50, 65 m
                    Z: 15, 25, 35 m
                    max_detection_distance = 18.0 m

Simulation        : step_size=0.5 m, max_bounces=8""")

h2(doc, '6.2 Pipeline Steps')
code_block(doc, """\
1. MeshImporter.load('AlienAnimal.obj', scale=1.0)
   → trimesh mesh with 25 695 vertices, 46 588 faces

2. ImportedObject3D(mesh_data, mode='aabb', R=0.80, name='AlienAnimal')
   Added to Simulation alongside Surface and Seabed.

3. sim.run(num_rays=1200)
   → rays traced; reflection metadata stored per bounce

4. collect_alien_hits(sim, 'AlienAnimal')
   → filter reflection_points where object_name == 'AlienAnimal'
   → 870 AABB acoustic hit points

5. trimesh_surface_hits(mesh_data, alien_obj, num_rays=8000)
   → 8000 directions sampled from hit points toward object centre
   → BVH RayMeshIntersector fires each ray
   → 3197 exact triangle surface intersection points
   → incidence angle computed for colour mapping

6. plot_scene() — saves two PNG images""")

h2(doc, '6.3 Output — Scene Overview')
if IMG_SCENE.exists():
    add_image(doc, IMG_SCENE,
              'Figure 2 — Full 3D scene: emitter (green star), AABB hit cloud (cyan), '
              'hydrophone array (yellow), water volume bounding box')
else:
    body(doc, '[alien_pointcloud_scene.png — run alien_demo.py to generate]')

body(doc, '')
body(doc,
    'The scene plot shows: the 3-D water volume wireframe, the emitter position, '
    'the 9-hydrophone array, AABB acoustic reflection hits on the AlienAnimal '
    'coloured by reflected amplitude, and the object bounding box.')

h2(doc, '6.4 Output — Surface Point Cloud')
if IMG_SURFACE.exists():
    add_image(doc, IMG_SURFACE,
              'Figure 3 — Acoustic surface point cloud: exact trimesh BVH hits '
              'coloured by incidence angle (blue = glancing, yellow = normal)')
else:
    body(doc, '[alien_pointcloud_surface.png — run alien_demo.py to generate]')

body(doc, '')
body(doc,
    'The surface point cloud uses trimesh\'s BVH RayMeshIntersector for exact '
    'per-triangle hit locations. Points are coloured by incidence angle: '
    'blue = glancing (large angle from normal), yellow = near-normal incidence '
    '(strongest return). This gives a acoustic "image" of the object\'s visible '
    'surface from the emitter side.')

h2(doc, '6.5 Running the Demo')
code_block(doc, """\
# From the repo root:
python alien_demo.py

# Or double-click:
run_demo.bat

# Expected terminal output:
Loading AlienAnimal.obj ...
  vertices : 25695
  faces    : 46588
  extents  : 17.94 × 19.56 × 39.30 m   scale=1.0
Running simulation (1200 rays) ...
  AABB acoustic hits on AlienAnimal : 870
  Running trimesh BVH surface mapping (8000 rays) ...
  Exact surface hits                : 3197
  Total hydrophone arrivals         : 10893
Saving alien_pointcloud_scene.png
Saving alien_pointcloud_surface.png
Done.""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 7. DATA PERSISTENCE — SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '7. Data Persistence — Save & Load')
body(doc,
    'acousticsim uses three different file formats depending on what is being '
    'persisted: the generated training dataset, the sklearn model, and the '
    'PyTorch model checkpoint. Each format is self-contained — you can close '
    'Python, share the file with a colleague, and resume without any '
    're-simulation.')

# ─── File overview ────────────────────────────────────────────────────────
h2(doc, '7.1 File Format Overview')
code_block(doc, """\
dataset.npz          ← NumPy compressed archive (AcousticDataset)
sklearn_det.pkl      ← Python pickle (SklearnDetector + StandardScaler)
pytorch_det.pt       ← PyTorch checkpoint dict (weights + normalization stats)""")

# ─── NPZ ─────────────────────────────────────────────────────────────────
h2(doc, '7.2 Training Dataset — .npz (NumPy Compressed Archive)')
body(doc,
    'AcousticDataset.save() calls numpy.savez_compressed(), which stores '
    'all arrays in a single zipped .npz file. This is cross-platform, '
    'readable without acousticsim installed, and typically 5–20× smaller '
    'than the uncompressed equivalent.')

h3(doc, 'What is stored')
code_block(doc, """\
┌─────────────────┬──────────────────────────────┬──────────────────────────┐
│  Key            │  Shape / dtype               │  Description             │
├─────────────────┼──────────────────────────────┼──────────────────────────┤
│  X              │  (N, F)  float32             │  Feature matrix          │
│                 │  e.g. (1000, 108) for H=9    │  one row per simulation  │
│  y_class        │  (N,)    int32               │  Class ID per sample     │
│  y_pose         │  (N, 6)  float32             │  x, y, z, roll, pitch,  │
│                 │                              │  yaw in metres / degrees │
│  class_names    │  (C,)    object (str)        │  Name for each class ID  │
│  feature_names  │  (F,)    object (str)        │  Name for each column    │
└─────────────────┴──────────────────────────────┴──────────────────────────┘

N = number of simulation runs
F = H×8 + C(H,2)   (H = number of hydrophones)
C = number of object classes""")

h3(doc, 'Feature matrix layout (X)')
body(doc, 'Each row is the output of extract_features() for one simulation run. '
         'Columns are arranged as follows for H hydrophones:')
code_block(doc, """\
Columns 0 … H×8-1     — per-hydrophone statistics (8 per sensor)
                         [ n_arrivals, first_time_ms, last_time_ms,
                           delay_spread_ms, max_amplitude, total_energy,
                           amplitude_mean, amplitude_std ]

Columns H×8 … F-1     — cross-hydrophone TDOA pairs, all i<j ordered
                         tdoa_ij = first_arrival_time[i] - first_arrival_time[j]
                         In milliseconds.""")

h3(doc, 'Save & load')
code_block(doc, """\
# Save
dataset.save('my_dataset.npz')
# Prints: Dataset saved: my_dataset.npz  (1000 samples, 108 features)

# Load — returns AcousticDataset (config field will be None)
from acousticsim import AcousticDataset
dataset = AcousticDataset.load('my_dataset.npz')

print(dataset.X.shape)          # (1000, 108)
print(dataset.y_class[:5])      # [0 1 0 0 1]
print(dataset.class_names)      # ['alien', 'submarine']
print(dataset.class_distribution())  # {'alien': 512, 'submarine': 488}

# Read directly with NumPy (no acousticsim needed)
import numpy as np
raw = np.load('my_dataset.npz', allow_pickle=True)
print(list(raw.keys()))   # ['X', 'y_class', 'y_pose', 'class_names', 'feature_names']
print(raw['X'].shape)     # (1000, 108)""")

# ─── sklearn PKL ─────────────────────────────────────────────────────────
h2(doc, '7.3 sklearn Detector — .pkl (Python Pickle)')
body(doc,
    'SklearnDetector.save() serialises the entire object — including the fitted '
    'RandomForestClassifier, MultiOutputRegressor, and StandardScaler — into a '
    'single .pkl file using Python\'s standard pickle module. '
    'The scaler\'s fitted mean and std are embedded inside the same object, '
    'so no separate normalisation file is needed.')

h3(doc, 'What is stored')
code_block(doc, """\
SklearnDetector object containing:
  .classifier        RandomForestClassifier  (n_estimators trees, all node data)
  .regressor         MultiOutputRegressor wrapping 6 RandomForestRegressors
  .scaler            StandardScaler  (.mean_, .var_, .scale_)
  .class_names       list of str
  .n_estimators      int
  .n_jobs            int""")

h3(doc, 'Save & load')
code_block(doc, """\
# Save
sklearn_det.save('sklearn_det.pkl')

# Load
from acousticsim.train_detector import SklearnDetector
sklearn_det = SklearnDetector.load('sklearn_det.pkl')

# Predict immediately — scaler is embedded
pred_class, pred_pose = sklearn_det.predict(X_new)
result = sklearn_det.predict_single(feature_vector)
# → {'class_id': 0, 'class_name': 'alien', 'x': 49.3, 'y': 51.2, ...}""")

body(doc,
    'Warning: pickle files are Python-version and library-version sensitive. '
    'Use the same major version of scikit-learn to load a saved file. '
    'For long-term storage prefer re-training from the .npz dataset.')

# ─── PyTorch PT ───────────────────────────────────────────────────────────
h2(doc, '7.4 PyTorch Detector — .pt (Torch Checkpoint)')
body(doc,
    'PyTorchDetector.save() calls torch.save() on a plain Python dict '
    'that contains the model\'s state_dict plus all normalisation statistics. '
    'This is the standard PyTorch checkpoint pattern and is more robust '
    'than pickling the whole object.')

h3(doc, 'What is stored inside the .pt file')
code_block(doc, """\
{
  'model_state'      : OrderedDict  — all layer weights & biases
  'class_names'      : list[str]
  'feature_means'    : (F,) float32 — per-feature training mean
  'feature_stds'     : (F,) float32 — per-feature training std
  'pose_mean'        : (6,) float32 — per-DoF training mean
  'pose_std'         : (6,) float32 — per-DoF training std
  'pose_loss_weight' : float
}""")

body(doc,
    'During training, both the feature vectors (X) and pose targets (y_pose) '
    'are normalised to zero-mean unit-variance before being passed to the network. '
    'At inference, the same statistics are applied to incoming feature vectors '
    'and the pose output is denormalised back to physical units before returning.')

h3(doc, 'Normalisation flow')
code_block(doc, """\
TRAINING:
  X_norm  = (X  − feature_means) / feature_stds       → fed to network
  yp_norm = (yp − pose_mean)     / pose_std            → target for pose head

INFERENCE:
  X_norm   = (X_new − feature_means) / feature_stds   → network input
  pose_raw = network(X_norm).pose_head                 → network output (normalised)
  pose_out = pose_raw × pose_std + pose_mean           → physical [m, deg]""")

h3(doc, 'Save & load')
code_block(doc, """\
# Save
pytorch_det.save('pytorch_det.pt')

# Load — must supply architecture parameters
from acousticsim.train_detector import PyTorchDetector
pytorch_det = PyTorchDetector.load(
    'pytorch_det.pt',
    input_dim   = 108,          # must match saved model
    num_classes = 2,
    hidden_dims = (256, 128, 64),
    dropout     = 0.3,
)

# Predict
pred_class, pred_pose = pytorch_det.predict(X_new)
result = pytorch_det.predict_single(feature_vector)
# → {'class_id': 0, 'class_name': 'alien',
#    'x': 49.7, 'y': 50.8, 'z': 24.1,
#    'roll_deg': 0.2, 'pitch_deg': -1.4, 'yaw_deg': 178.3,
#    'class_proba': {'alien': 0.94, 'submarine': 0.06}}""")

# ─── Full lifecycle ───────────────────────────────────────────────────────
h2(doc, '7.5 Full File Lifecycle')
code_block(doc, """\
[Meshes]  AlienAnimal.obj, submarine.stl
    │
    ▼  MeshImporter.load()
[MeshData objects in memory]
    │
    ▼  DatasetGenerator.generate(num_samples=1000)
    │   runs 1000 simulations, extracts feature vectors
    ▼
[AcousticDataset in memory]
    │
    ├──► dataset.save('dataset.npz')      ← persist raw training data
    │                                        (arrays only, no Python objects)
    ▼
    ├── train_sklearn(dataset)
    │       SklearnDetector.save('sklearn_det.pkl')   ← RF trees + scaler
    │
    └── train_pytorch(dataset)
            PyTorchDetector.save('pytorch_det.pt')   ← weights + norm stats

─────────── Later / different machine ───────────────────────────────────

AcousticDataset.load('dataset.npz')      ← reload to re-train or inspect
SklearnDetector.load('sklearn_det.pkl')  ← deploy sklearn model
PyTorchDetector.load('pytorch_det.pt',   ← deploy pytorch model
    input_dim=108, num_classes=2)
    │
    ▼  AcousticDetector.from_sklearn() / .from_pytorch()
        .simulate_and_detect('AlienAnimal.obj', [50,50,25], [0,0,0])
        .detect_from_hydrophones(hydrophones)
        .detect_batch(filepaths, positions)""")

h2(doc, '7.6 Inspecting a Saved Dataset Without acousticsim')
code_block(doc, """\
import numpy as np

raw = np.load('dataset.npz', allow_pickle=True)

# Arrays
X            = raw['X']            # (N, F) float32
y_class      = raw['y_class']      # (N,)   int32
y_pose       = raw['y_pose']       # (N, 6) float32
class_names  = list(raw['class_names'])    # ['alien', ...]
feature_names= list(raw['feature_names'])  # ['h0_n_arrivals', ...]

# Inspect
print(f"Samples   : {len(X)}")
print(f"Features  : {X.shape[1]}")
print(f"Classes   : {class_names}")

for i, name in enumerate(class_names):
    count = (y_class == i).sum()
    print(f"  {name}: {count} samples")

print(f"Pose range: x={y_pose[:,0].min():.1f}–{y_pose[:,0].max():.1f} m")""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 8. MACHINE LEARNING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '8. Machine Learning Pipeline')
body(doc,
    'The ML pipeline converts acoustic simulation data into a trained detector '
    'that predicts object class and 6-DoF pose from live hydrophone arrays.')

h2(doc, '8.1 End-to-End Workflow')
code_block(doc, """\
from acousticsim import (
    MeshImporter, DatasetGenerator, DatasetConfig,
    NoiseModel, train_pytorch, AcousticDetector
)

# 1. Load meshes
meshes = {
    'alien'     : MeshImporter.load('AlienAnimal.obj'),
    'submarine' : MeshImporter.load('sub.stl'),
}

# 2. Configure and generate dataset
nm = NoiseModel('ambient', snr_db=15, seed=0)
cfg = DatasetConfig(
    num_samples     = 1000,
    num_rays        = 400,
    num_hydrophones = 9,
    frequency_khz   = 85.0,
    noise_model     = nm,
)
dataset = DatasetGenerator(meshes, cfg).generate()
dataset.save('dataset.npz')

# 3. Train
detector = train_pytorch(dataset, epochs=150, patience=15)

# 4. Deploy
det = AcousticDetector.from_pytorch(detector,
                                    class_names=['alien','submarine'])
pred = det.simulate_and_detect('AlienAnimal.obj', [50,50,25], [0,0,0])
print(pred)  # → {'class': 'alien', 'confidence': 0.94, 'pose_xyz': [...]}""")

h2(doc, '8.2 Feature Vector Size')
body(doc, 'For a 9-hydrophone array (3×3 grid):')
code_block(doc, """\
Per-hydrophone features : 9 × 8  = 72
TDOA pairs              : C(9,2) = 36    (all unique sensor pairs)
Total features          : 108""")

h2(doc, '8.3 Model Outputs')
code_block(doc, """\
Classification  →  softmax probability over N object classes
Pose regression →  [x, y, z, roll°, pitch°, yaw°]  6-DoF continuous values

Loss function:
L = CrossEntropy(ŷ_class, y_class)  +  0.1 × MSE(ŷ_pose, y_pose)""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 8. QUICK USAGE GUIDE
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '9. Quick Usage Guide')

h2(doc, '9.1 Minimal Simulation')
code_block(doc, """\
from acousticsim import Water, WaveEmitter, Hydrophone, Simulation, Surface, Seabed

water = Water(size=(100,100,50), temperature_surface=18, temperature_bottom=8)
sim   = Simulation(water)
sim.add_object(Surface(water))
sim.add_object(Seabed(water))

emitter = WaveEmitter([5,50,25], frequency_khz=85, beam_pattern='cone',
                       beam_direction=[1,0,0], beam_angle=30, power_db=185)
sim.add_emitter(emitter)

hydro = Hydrophone([95,50,25], name='H1')
sim.add_hydrophone(hydro)

sim.run(num_rays=500)

print('Arrivals:', hydro.get_arrival_count())
t, sig = hydro.get_multipath_signal()""")

h2(doc, '9.2 Import a 3-D Mesh Object')
code_block(doc, """\
from acousticsim import MeshImporter, ImportedObject3D, Simulation

mesh = MeshImporter.load('my_object.obj', scale=0.5)
obj  = ImportedObject3D(mesh, position=[50,50,25],
                         intersection_mode='mesh',
                         reflection_coefficient=0.85,
                         name='target')
sim.add_object(obj)""")

h2(doc, '9.3 Add Noise to Signals')
code_block(doc, """\
from acousticsim import NoiseModel

nm = NoiseModel('ambient', snr_db=12, seed=42)

# Noisy waveform
t, noisy = hydrophone.get_multipath_signal(noise_model=nm)

# Noisy amplitudes (for feature extraction)
amps = hydrophone.get_amplitudes(noise_model=nm)

# Inject fake arrivals (simulates reverberation tail / bio noise)
nm.inject_arrivals(hydrophone, n_noise_arrivals=6)""")

h2(doc, '9.4 Get Acoustic Point Cloud')
code_block(doc, """\
# After sim.run():
all_hits       = sim.get_reflection_points()                      # all objects
alien_hits     = sim.get_reflection_points(object_name='AlienAnimal')  # filtered
alien_amps     = [r for r, o in zip(all_amps, all_objects) if o == 'AlienAnimal']

# Per-ray access
for ray in sim.rays:
    pts   = ray.get_reflection_points(object_name='AlienAnimal')
    names = ray.get_reflection_objects()
    amps  = ray.get_reflection_amplitudes()""")

h2(doc, '9.5 Chirp Waveform')
code_block(doc, """\
emitter = WaveEmitter(
    position       = [5, 50, 25],
    wave_type      = 'chirp',
    frequency_range= (70, 100),   # kHz
    chirp_duration = 0.005,       # 5 ms
    power_db       = 185,
)""")

page_break(doc)

# ═══════════════════════════════════════════════════════════════════════════
# 9. INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════

h1(doc, '10. Installation')

h2(doc, '10.1 From Source (development)')
code_block(doc, """\
git clone https://github.com/yourname/acousticsim.git
cd acousticsim
pip install -e .[all]""")

h2(doc, '10.2 From PyPI (once published)')
code_block(doc, """\
pip install acousticsim            # core physics only
pip install acousticsim[mesh]      # + trimesh mesh import
pip install acousticsim[ml]        # + scikit-learn + torch
pip install acousticsim[all]       # everything""")

h2(doc, '10.3 Optional Dependencies')
code_block(doc, """\
trimesh[easy] >= 3.23   # STL, OBJ, GLTF import + BVH raycasting
scikit-learn  >= 1.3    # Random Forest detector
torch         >= 2.0    # PyTorch MLP detector
aiohttp       >= 3.8    # real-time WebSocket server (future)
matplotlib    >= 3.7    # visualisation""")

h2(doc, '10.4 Running the Demo')
code_block(doc, """\
# Ensure AlienAnimal.obj is in the repo root, then:
python alien_demo.py
# or
run_demo.bat

# Outputs:
#   alien_pointcloud_scene.png
#   alien_pointcloud_surface.png""")

divider(doc)
body(doc, '')
p = doc.add_paragraph(
    f'acousticsim  v0.1.0  —  MIT License  —  Generated {datetime.datetime.now().strftime("%B %d, %Y")}'
)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.runs[0].font.size = Pt(9)
p.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

doc.save(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
