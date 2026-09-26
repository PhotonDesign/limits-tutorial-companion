# Companion codes: *Fundamental limits in photonics and electromagnetics: a tutorial*

Numerical checks and full derivations accompanying the tutorial by O. D. Miller and F. Monticone. The tutorial's scope is broad, so it states many results without derivations. This repository provides:

1. **`derivations/derivations.pdf`**: complete derivations for Secs. 3.1 (free-space focusing: plane-wave, 2π, 4π, super-oscillatory and space-time focusing) and 3.2 (focusing in complex media: reciprocity, time reversal). Each closed-form result is annotated with the numerical check that verifies it.
2. **Independent numerical checks** of key stated results in all five chapters (most extensively Secs. 3.1–3.2). Each script prints computed values next to the claimed ones. A check is independent of the derivation it tests, e.g. brute-force quadrature with explicit polarization vectors, or a full multiple-scattering model.
3. **Regression tests** (`pytest`) that pin the key numbers.

All codes are small (seconds each on a laptop) and depend only on NumPy, SciPy (≥ 1.15) and Matplotlib; Python ≥ 3.11.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py            # runs every check in registry.toml; logs to outputs/
python -m pytest tests -q    # ~10 s
```

Individual scripts can be run directly, e.g. `python sec3_1_free_space_focusing/focal_plane_fields.py`. Figures are written to `figures/`.

To rebuild the derivations PDF:

```bash
cd derivations && latexmk -pdf derivations.tex
```

## Contents

| Section | Script | What is checked |
|:--|:--|:--|
| 1.1–1.7 | `sec1_fundamental_principles/*.py` | equivalence principle, Kramers–Kronig relations, passivity/Herglotz representations and sum rules, Rayleigh-quotient theorems |
| 2.1–2.7 | `sec2_material_bounds/*.py` | maximum refractive index, dispersion/bandwidth bounds on ε, material sum rules and estimates |
| 3.1.2 | `sec3_1_free_space_focusing/power_normalization_check.py` | plane-wave power formula, by direct Poynting integration |
| 3.1.2 | `sec3_1_free_space_focusing/focal_intensity_bounds.py` | focal-intensity maxima for power/amplitude normalization; optimal polarization; reference beams |
| 3.1.2–3 | `sec3_1_free_space_focusing/focal_plane_fields.py` | optimal fields across the focal plane (closed forms, anisotropy); FWHM and first zeros |
| 3.1.3 | `sec3_1_free_space_focusing/fourpi_spherical_waves.py` | vector-spherical-wave derivation of the 4π bound; Im G; the doubling argument |
| 3.1.4 | `sec3_1_free_space_focusing/superoscillation.py` | exact energy penalties for super-oscillatory spots (1/δ², 1/δ⁴, G⁴ laws with constants) |
| 3.1.5 | `sec3_1_free_space_focusing/spacetime_focusing.py` | space-time-optimal fields in 1D and 3D |
| 3.1 | `sec3_1_free_space_focusing/tutorial_figure_panels.py` | label-light figure panels for the tutorial |
| 3.2 | `sec3_2_complex_media/complex_media_focusing.py`, `spacetime_complex_media.py` | focusing bound, reciprocity vs. time-reversal recipes, partial time reversal, and space-time focusing, in a 3D coupled-dipole medium with tunable symmetries |
| 3.4 | `sec3_4_communication_channels/communication_modes.py` | communication modes between two square arrays (SVD) |
| 4.1 | `sec4_global_symmetries/extinction_symmetry_{slab,dipoles}.py` | extinction symmetry of reciprocal scatterers (graded slab; 3D dipole clusters) |
| 4.2 | `sec4_global_symmetries/tr_field_ratio_bounds.py` | time-reversal-symmetry bounds on field ratios |
| 4.3 | `sec4_global_symmetries/change_of_basis_kirchhoff.py` | scattering-matrix changes of basis; Kirchhoff's law |
| 4.4 | `sec4_global_symmetries/brightness_theorem.py` | brightness theorems for passive linear systems |
| 5.1 | `sec5_local_constraints/vie_conservation_laws.py` | global and local (D-matrix) conservation laws of the volume integral equation |
| 5.1 | `sec5_local_constraints/material_loss_bounds.py` | material-loss, per-channel, optical-theorem and dual bounds vs. Mie theory and ellipsoids |
| 5.2.1 | `sec5_local_constraints/extinction_sum_rules.py` | extinction sum rules (static and high-frequency) |
| 5.2.3 | `sec5_local_constraints/tmatrix_oscillators.py` | oscillator representation of the T-matrix and its sum rules |
| 5.4–5.9 | `sec5_4_9_scattering_limits/*.py` | polarizability and Q limits, spherical-wave sum rules, Bode–Fano/Rozanov, cloaking bandwidth, sensing limits, delay–bandwidth |

`registry.toml` lists every check with the tutorial equations it tests; add a new check by adding an entry there. `VALIDATION.md` summarizes the numerical outcomes.

Shared utilities live in `tutorial_tools/`: plane-wave/angular-spectrum quadrature, closed-form focal fields, vector spherical waves, and a coupled-dipole (Foldy–Lax) medium.

## Conventions

Time dependence is e^{−iωt}. Codes use natural units ε₀ = μ₀ = c = Z₀ = 1 (so ω = k) and report dimensionless ratios such as (2πc/k²) ε₀|E(0)|²/P. Scripts cite tutorial equations by their LaTeX labels (e.g. `[eq:OptPowerNorm]`), which remain stable when equation numbers change.

## Citation

If you use these codes, please cite the tutorial: O. D. Miller and F. Monticone, "Fundamental limits in photonics and electromagnetics: a tutorial" (2026).
