# Validation summary

> **Pre-publication draft (2026-09-26).** Rows marked ⚠ are cases where the numerics disagree with the *submitted* (first-round) text of the tutorial. They are listed so the authors can revise the text; once the text is revised, the ⚠ notes should be rewritten as ordinary confirmations. Tutorial equations are cited by LaTeX label. Every number below is printed by the listed script (`python run_all.py` regenerates all logs in `outputs/`).

## Sec. 3.1 Free-space focusing (`sec3_1_free_space_focusing/`)

| Claim | Outcome | Script |
|:--|:--|:--|
| Plane-wave power P = (2π²/Z₀)Σ∫(k_z/k)\|c_p\|² | Direct Poynting integration of random complex spectra: rel. err ≤ 7×10⁻¹⁶ | `power_normalization_check.py` |
| [eq:OptPowerNorm], [eq:OptAmpNorm]; in-plane polarization globally optimal | The 3×3 focal matrix built with explicit TE/TM vectors is diagonal (off-diagonals < 10⁻¹⁶) and reproduces all closed forms to 3×10⁻¹³. The gap x−z equals cos θ sin²θ ≥ 0 | `focal_intensity_bounds.py` |
| [eq:TwoPiBound] U/P ≤ k²/(6πc) | 4/3 × ¼ confirmed | same |
| "all polarizations equal at 2π" | True for power normalization only; amplitude normalization gives 3/4 (in-plane) vs 1/2 (z) | same |
| ⚠ "power-normalized maximum exceeds the amplitude-normalized one by 1.77X" | 16/9 = 1.778, but the two maxima have different denominators (P vs Σ∫\|c\|²). At equal *physical* power the amplitude-optimal beam gives 135/128, so the advantage is 1.264 | same |
| "exceeds the standard diffraction-limited Gaussian beam by 1.43X [Dias2020]" | 1.435 against Dias's reference beam c ∝ √(k_z/k); 1.10 against an aplanatic lens with uniform pupil | same |
| ⚠ FocusingFourPi table, paraxial row "0.66 sin²θ" | Eq. [eq:OptPowerNorm] gives exactly sin²θ paraxially (0.661 is its value at sin θ = 0.8) | same |
| ⚠ [eq:E2TwoPi], 2π field intensity | Correct along the polarization axis only. The full focal-plane field is E_x = 2πk²[sin²φ j₀ + (3cos²φ−1) j₁/a], E_y = 2πk² sinφ cosφ j₂, E_z = −2πik² cosφ J₂/a, confirmed three independent ways to 2×10⁻¹⁴. The spot is elliptical; perpendicular to the polarization, \|E\|² ∝ (j₀ − j₁/a)², identical to the 4π profile | `focal_plane_fields.py` |
| ⚠ Fig. PWFocusing(d): E_z map | E_z ∝ cos φ is antisymmetric in x (the published map is rotationally symmetric); \|E\|² is elliptical. Replacement panels are in `figures/tutorial_panels/` | same, `tutorial_figure_panels.py` |
| Fig. PWFocusing(e) FWHM vs NA | Reproduced along the polarization axis, including the maximum of 0.651 near sin θ = 0.975. ⚠ Perpendicular to the polarization the FWHM *decreases* (0.514 → 0.402 λ/sin θ) | same |
| ⚠ FWHM / first-zero table (2π: 0.63λ / 0.77λ; 4π: 0.40λ / 0.44λ) | Values reproduced (0.634, 0.765, 0.402, 0.437). The 2π "first zero" is a minimum with I/I₀ = 2.8×10⁻³. Along y the 2π spot is also 0.402/0.437 λ. Axial widths: 2π 0.887λ, 4π 0.402λ | same |
| [eq:FourPiBound] U/P_in ≤ k²/(3πc); \|N₁₀(0)\| = (2/3)√(3/8π); P_in = 1/(8Z₀k²); only N₁ₘ nonzero at the origin; [eq:N10xy] | All confirmed (10⁻¹⁰–10⁻¹⁵); P_in flux-integrated at R = 3, 30, 300 | `fourpi_spherical_waves.py` |
| 4π optimum ∝ Im G₀; 4π = 2 × 2π | Herglotz wave = 16π²k Im G₀ (8×10⁻¹⁵); ratio 8/3 vs 4/3 | same |
| E ≳ 1/δ², E ≳ 1/δ⁴ | Exact constants: 3/(2k_m³δ²) and 45/(2k_m⁵δ⁴) (penalty 45/(k_mδ)⁴) | `superoscillation.py` |
| Fig. Superres G⁴ law | S_max(u) = 1 − 4J₁²/[u²(J₀²+J₁²)] → u⁴/192 (paraxial scalar), brute force 10⁻⁸ | same |
| ⚠ "a factor of 2 reduction and a corresponding 16X increase" | Exact: 9.3× (2D zero ring), 5.3× (1D line). The tutorial's own S_max curve gives ≈ 10× at G = 0.5 | same |
| [eq:ST1d] | Confirmed vs direct quadrature (1.5×10⁻¹⁴); first zero doubles in 1D | `spacetime_focusing.py` |
| ⚠ 3D space-time profile: "closed form in terms of Si"; side lobes and first zero | Closed form ∝ [Si(U) + j₁(U)]/U. It has no zeros and a non-oscillatory 1/ρ tail, so the two highlighted features hold in 1D only (amplitude normalization) | same |

## Sec. 3.2 Complex media (`sec3_2_complex_media/complex_media_focusing.py`)

Medium: a random 3D cluster of electric point scatterers (Foldy–Lax). The four symmetry classes are set by the bare polarizability α₀: lossless-reciprocal (real symmetric), lossy-reciprocal (complex symmetric), TRI-nonreciprocal (real non-symmetric), and gyrotropic (Hermitian non-symmetric).

| Claim | Outcome |
|:--|:--|
| [eq:CMFocusBound0] rank-one maximum ‖Γ₀ₐ†φ‖² | Iterative eigensolver on the full operator agrees to ≤ 2×10⁻¹⁵ |
| [eq:CMTRRecipe] conjugated dipole field is optimal in reciprocal media, lossy or not, open or closed aperture | Efficiency 1.00000000 in all reciprocal cases |
| Not optimal in non-reciprocal media; optimal after transposing the medium (Sec. 3.2.5) | Recipe efficiency 0.08–0.26; transposed-medium recipe 1.00000000 |
| Far-field bound ∝ LDOS; near-field apertures larger | bound/(2P_rad) = 1.00012 at R = 30λ; 94.8 at R = 0.06λ |
| [eq:ConjFields] ψ_TR = −2i Im ψ, with **only TRI** required (PEC construction) and no far-field requirement | Scatterer 0.05λ from S: 7×10⁻⁸ (lossless reciprocal) and 1×10⁻⁶ (**TRI non-reciprocal**), converging with quadrature. Lossy: 0.45; gyrotropic: 1.24 |
| Validity region of [eq:ConjFields] | Holds inside S; outside S the TR field is −Γs₀* (1.7×10⁻⁷) |
| [eq:CMSTBound] space-time bound (`spacetime_complex_media.py`) | Time-reversed dipole response attains the maximum (1.0000000000) in a dispersive lossy medium; the bound is ∫ over [t₀−T, t₀] (the printed [0,T] holds for t₀ = T; for t₀ = 1.3T it would give 0.61) |

## Sec. 3.4 Communication channels (`sec3_4_communication_channels/communication_modes.py`)

| Claim | Outcome |
|:--|:--|
| ⚠ Fig. CommChannels: 16λ arrays at 50λ with the listed percentages | The 12 percentages are reproduced to 0.002 pp by 17×17-point arrays at **100λ** (best fit 99.99λ). At 50λ mode 1 carries 3.2% and about 25–33 channels are nearly equal (N_H = 26) |
| Right singular vectors = inputs, left = measurements; Ky Fan chain argument | Confirmed. ⚠ The text's notation sentence swaps "left/right" |

## Sec. 4 Global symmetries (`sec4_global_symmetries/`)

| Claim | Outcome | Script |
|:--|:--|:--|
| Extinction symmetry for reciprocal scatterers | Graded slab: equal to 1.8×10⁻¹⁵ at 0/30/60°, s/p, while A and R differ by 0.08–0.53. 3D reciprocal dipole clusters: ≤ 3×10⁻¹⁶. Gyrotropic: violated by up to 18%, but σ_ext[α](k,e) = σ_ext[αᵀ](−k,e*) | `extinction_symmetry_{slab,dipoles}.py` |
| ⚠ Fig. ExtSym(c) curves | Plotted "extinction" uses t referenced to the exit face (Yeh convention); physical extinction is 2Re(1 − t e^{−ikL}). The equality still holds, but the curves differ for f ≲ 2c/L. An empty slab would show up to 4. | `extinction_symmetry_slab.py` |
| Two-port TR envelope (1∓R)/(1±R) | Confirmed to 5 digits, on a physical 10-layer stack as well | `tr_field_ratio_bounds.py` |
| ⚠ [eq:fTRRatio] n-port generalized eigenproblem | As written (complex Hermitian W′, V′) it returns the trivial bound; the real pencil (Re W′, Re V′) gives the tight bound (n = 4: 0.69625, matching BFGS) | same |
| S₂, A₂, E₂; symmetries preserved under change of basis | Exact (≤ 2×10⁻¹⁶) | `change_of_basis_kirchhoff.py` |
| Termwise brightness theorem λ_k(ρ_out) ≤ λ_k(ρ_in) | Max excess +7×10⁻¹⁷ in 14,000 trials | `brightness_theorem.py` |
| ⚠ Doubly-stochastic / entropy-increase statement | Valid for unitary S only; with loss the normalized entropy dropped in 98.2% of random 4×4 cases | same |
| Concentrator 1/N; TCMT ≤ M/N | Confirmed (1/11 = 9.09%) | same |

## Sec. 5 Local constraints (`sec5_local_constraints/`)

| Claim | Outcome | Script |
|:--|:--|:--|
| [eq:OpticalThm], local D-matrix constraints, domain obliviousness, constraint count | Hold to ≤ 1.4×10⁻¹⁶; 432 local constraints; rank 162/162 | `vie_conservation_laws.py` |
| ⚠ [eq:ReacOpticalThm] and the second of [eq:TwoConstraints] | Sign error: φ†(ReΓ₀+Reξ)φ = **−**Re(ψ†φ). As printed the residual is 0.12; corrected, 10⁻¹⁶ | same |
| ⚠ footnote on P_scat for φ_max | Im[ψ†φ − φ†(Imξ)φ] takes Im of a real term; should be Im(ψ†φ) − φ†(Imξ)φ | same |
| [eq:CSBoundML], per-channel Mie bounds, [eq:CSOTBound], [eq:DualBounds] | Never exceeded; saturated where predicted | `material_loss_bounds.py` |
| [eq:SRExt0] | Quasi-static spheroids 1.0000000000; Mie 0.999993 | `extinction_sum_rules.py` |
| ⚠ [eq:SRExtInf] | Missing 1/c: ∫σ_ext dω = πω_p²V/(2c) (ratio to the printed form = 1/c) | same |
| ⚠ [eq:GenDisp], [eq:GenSumRule] | Missing ω′ in the numerator (KK check 1.401 vs 1.743) and dω | same |
| ⚠ [eq:Tosc] sum rules | With the 2/π prefactor, ∫X dω = (π/2)ω_p² I_V (a single Lorentz oscillator gives 1.5708), violating the printed ≤ ω_p² I_V | `tmatrix_oscillators.py` |
