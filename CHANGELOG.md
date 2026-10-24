# Changelog

## 0.1.0 (2026-10-24): rebuilt from the paper

The SonarSim version used for the paper was lost and no copy survived (not
on PyPI or TestPyPI, not installed anywhere, no backup). The only surviving
code was an earlier package, `acousticsim` (sources dated 2026-03-01): a
step-marching ray tracer with Thorp absorption and a Random-Forest/MLP
detector. It is preserved unchanged in this repository's first commit.

This release is a rebuild of the paper's pipeline, written from the paper's
methods, tables and figures:

- New package `sonarsim` with a `src/` layout, a working `pyproject.toml`
  (the old one named a non-existent build backend and a missing entry point),
  tests and a dataset CLI.
- Physics as listed in the paper's Table I: Mackenzie sound speed,
  Francois-Garrison absorption, Eckart surface and Rayleigh bottom
  reflection, Wenz ambient noise, Kirchhoff facet scattering with
  closed-form triangle integrals, delay-and-sum beamforming, matched
  filtering, CW and LFM pulses.
- Configurations `A_HF_imaging` and `B_LF_search` from the paper's Table II.
- AutoPlacer with the paper's five seabed types; placements now stay inside
  the beamformed sector by default.
- Labels now record every predicted multipath arrival, a measured SNR and an
  image-space bounding box, and ranges in labels and range axes use the same
  sound speed.
- The old ray tracer and ML detector were dropped.

Because the lost code cannot be compared line by line, figures and numbers
regenerated with this version will differ in detail from the published ones.
Modelling choices the paper did not state were made here and are documented
in the code: sediment parameters, Lambert backscatter strengths, sea-state
wind speeds, source levels, pulse lengths, array spacing (half a wavelength
at the top of the band), sampling rate (2.5 x bandwidth, complex baseband),
sonar depth (7 m) and vertical beam (60 deg, untilted).
