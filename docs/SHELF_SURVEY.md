# astro_GPU - free shelf survey (J0), 2026-10-10

Purpose: per-component "reuse or develop" inventory. Every candidate listed as **bench baseline** or **component** is to be
measured on the common bench (PLAN.md section 5) against our own implementation. Free tools only (open source, or free
tools with closed weights). Nothing was installed and no project file was modified for this survey.

## How the facts were verified

- Stars, licence detected by GitHub, default-branch last commit and latest release: read from the github.com page
  (`stargazerCount`), the `commits/<branch>.atom` and `releases.atom` feeds, on 2026-10-10 (GitHub REST API was
  rate-limited). GitLab: GitLab API. Python packages: PyPI JSON.
- Behaviour, inputs, licences of weights: read in the repositories (source code, LICENSE files, READMEs), official docs,
  paper pages (arXiv abstract, availability statements). Anything not confirmed from such a source is marked
  **unverified**.
- Some numbers were measured locally on synthetic data in the project venv (CPU): marked "measured locally".
- Where the survey found something that contradicts PLAN.md section 4, it is flagged **[plan update]**.

Verdict vocabulary: **bench baseline** (run as-is on the bench, score its output), **component** (integrate code or
method into our chain; says how and what to modify), **skip** (with reason).

Data facts used for the verdicts (PLAN.md): Heart per-frame FWHM p10-p90 3.81-4.33 px, Iris 4.30-5.03 px; Sony A7S
frames are 12 MP (about 4240 x 2830 px); stacks may be drizzled to larger sizes (up to ~24 MP); two half-stacks.

---

## 0. Key findings (read this first)

1. **deepinv is confirmed as the backbone** (BSD-3, PyTorch, Windows + CUDA). It has exactly the operators we need:
   `SpaceVaryingBlur` (product convolution, filters and multipliers supplied as plain tensors, so our measured
   eigen-PSFs plug in) and `TiledSpaceVaryingBlur` (one measured PSF per overlapping tile). Plus `StackedLinearPhysics`
   (multi-frame), PnP / RED / DPIR with any `forward(x, sigma)` denoiser (our N2N), and self-supervised losses (SURE
   Poisson-Gaussian, R2R, Neighbor2Neighbor, splitting, EI). Gaps we must write: Poisson-Gaussian data fidelity,
   RGGB mask (its `Demosaicing` is BGGR), warp physics, streaming over hundreds of frames, PSF-field builder.
2. **torchmfbd cannot impose a measured PSF through its API** (PSF modes are always optimised; `infer_modes=False`
   removes the PSF instead of fixing it). Non-blind only by a source hack (`lr_modes = 0` + injected coefficients).
   Licence inconsistent (LICENSE MIT, pyproject GPLv3). **[plan update]**: ideas only, no dependency.
3. **New multi-frame baseline found: SASpro MFDeconv** (GPL-3, PyTorch GPU, Windows): multi-frame RL on registered
   frames with a PSF measured per frame (stationary within a frame), Huber/variance weights, optional SR. Only runnable
   free multi-frame astro deconvolution found. **[plan update]**
4. **ImageMM and AstroClearNet: no public code** (papers only). ImageMM's majorisation-minimisation is the algorithm
   template for our joint solver.
5. **ASTERIS code and weights are public** (MIT code, CC-BY-4.0 weights on Zenodo). Pretrained models are JWST/Subaru
   (not transferable); it is the multi-frame generalisation of our half-stack N2N. **[plan update]**
6. **StarNet2 2.6 (2026-09) adds `--linear`**; our local install is 2.5.4. Licence forbids using its outputs to train
   networks for commercial software and sharing full-resolution inputs demonstrating its performance.
7. **GraXpert**: deconvolution exists only in 3.1.0 release candidates; weights are **CC BY-NC-SA 4.0**; main branch
   idle since 2025-01. Its AI background model works on a 240 x 240 downsample of the whole image.
8. **No free BlurXTerminator equivalent with open weights** exists; Cosmic Clarity models are licensed for use inside
   SASpro only.
9. **Windows constraints**: JAX has no NVIDIA GPU on native Windows (SCICO, scarlet2, STARRED, J-UBIK: CPU or WSL2);
   Piff, PSFEx, Gnuastro, GalSim need WSL2.
10. **Siril SPCC sensor database has no Sony A7S**, and DSLR curves include the stock IR-cut: a custom A7S-mod JSON is
    required for a fair SPCC baseline.

---

## 1. Inverse-problem frameworks for non-blind, space-varying, multi-frame deconvolution

| Candidate | Licence | Activity (verified) | Stars | Framework, GPU, Windows | Space-varying / multi-frame / noise | Verdict |
|---|---|---|---|---|---|---|
| **deepinv** | BSD-3 | v0.4.2 2026-08-30; commit 2026-10-10; JOSS 2025 | 827 | PyTorch, GPU, Windows OK (pure Python) | `SpaceVaryingBlur`, `TiledSpaceVaryingBlur`; `StackedLinearPhysics`; Poisson, no Poisson-Gaussian fidelity | **component (backbone)** |
| **SASpro MFDeconv** | GPL-3.0 | V1.22.5 2026-10-10 | 112 | PyTorch CUDA / DirectML / CPU, Windows | one stationary PSF per frame (median of detected stars); multi-frame RL; Huber + variance weights | **bench baseline** (multi-frame reference to beat) |
| **PyLops** `NonStationaryConvolve2D` (+ `TorchOperator`) | LGPL-3.0 | v2.8.0 2026-07-25; commit 2026-10-08 | 540 | NumPy / CuPy / numba-cuda; Windows feasible (untested) | exact scatter of a regular grid of PSFs with bilinear blending; adjoint = gather | **component (reference operator to validate ours)** |
| **DeepLens** `splat_psf_per_pixel` | Apache-2.0 | v2.5.5 2026-09-30 | 740 | PyTorch | exact per-pixel PSF splatting (no adjoint) | **component (ground truth on crops)** |
| **ImageMM** (Sukurdeep et al., AJ 170, 233, 2025; arXiv 2501.03002) | no public code (TensorFlow per paper) | paper v3 2025-09-12 | - | - | joint multi-frame restoration + SR, per-exposure PSF refined, Huber, non-negativity; no space variation | **component (algorithm to re-implement)** |
| **STARRED** (+ Firedec) | GPL-3.0+ | 1.7.1 2026-08-13 | 6 (GitLab) | JAX; GPU only via WSL2 (experimental) | joint multi-epoch, point sources + pixel background, one PSF per epoch, finite target PSF | **crop-level bench** (optional) |
| **AstroPhot** | GPL-3.0 | v0.18.0 2026-06-25 | 106 | PyTorch (JAX optional via caskade), Windows | joint multi-image fits, per-image PSF, Poisson likelihood in LM | **crop-level validation bench** (optional) |
| **torchmfbd** | MIT (LICENSE) vs GPLv3 (pyproject): inconsistent | PyPI 0.9.2 2025-08-18; commit 2026-08-11 | 11 | PyTorch; classifiers Unix/macOS only | blind MOMFBD, PSF modes always optimised, space-variant by coefficient grid + apodised patches, Gaussian likelihood | **ideas only** (see key finding 2) |
| PyProximal | LGPL-3.0 | v0.13.0 2026-10-03 | 85 | NumPy / CuPy | `PlugAndPlay`, `ConsensusADMM` (one term per frame); no Poisson prox | optional alternate solver |
| J-UBIK `linpatch_convolve` | BSD-2 | v0.4 2026-08-13 | 8 | JAX, float64 | overlapping tiled convolution with exact adjoint | skip (idea = deepinv tiled operator) |
| rdmpy (Ring Deconvolution Microscopy, Nature Methods 2025) | MIT | commit 2025-10-07 | 60 | PyTorch | rotationally symmetric space-varying PSF, Seidel aberration fit | study only (Newton coma model); `pip install rdmpy` is an unrelated package |
| CUQIpy (+ CUQIpy-PyTorch) | Apache-2.0 | v1.5.1 2026-04-07 | 53 | NumPy (CPU) | stationary PSF only, MCMC UQ | **skip** (toy scale) |
| ODL 1.0 | MPL-2.0 | 1.0.0 2026-03-11 | 434 | Array API incl. PyTorch | no space-variant operator, no PnP | **skip** |
| SCICO | BSD-3 | v0.0.7 2025-12-09 | 174 | JAX, "not supported on Windows" | stationary convolution, PnP (BM3D, DnCNN) | **skip** (examples useful as references) |
| Pyxu | MIT | v2.0.3 2025-03-26 (dormant) | 142 | NumPy | stationary only | **skip** |
| ProxImaL | MIT | v0.2.0 2024-10-22 | 122 | Halide, numpy 1.26 pin | conv + warp + subsample + Poisson (conceptually our model) | **skip** (obsolete pins) |
| pylops-gpu | LGPL-3.0 | last commit 2022-05-23, README "unmaintained" | 12 | | | **skip** (use PyLops CuPy backend) |
| scarlet2 / scarlet v1 / the Tractor | MIT / MIT / GPL-2.0 | 2026-09-29 / 2024-08 / dr11.1 2026-04-22 | 32 / - / 94 | JAX / numpy / C+Python (Linux) | galaxy deblending or catalogue forced photometry | **skip** |
| AstroClearNet (arXiv 2504.06463) | no code | paper 2025 | - | - | deep-image-prior fusion per scene, known PSFs | **skip** (no code, does not scale) |
| MultiWienerNet, UABC, DegradationTransfer, DMBSR, pycudadecon, RedLionfish, sdeconv, DeconvolutionLab2, CIL, SigPy, SPORCO, MIRTorch, NIFTy | various | | | | stationary only, no licence, no weights, or other domain | **skip** (some are algorithm references) |

Notes.

- **deepinv** (https://github.com/deepinv/deepinv; source `deepinv/physics/blur.py`,
  `physics/functional/product_convolution.py`; class signatures verified in source on 2026-10-10).
  - `SpaceVaryingBlur(filters (B,C,K,h,w), multipliers (B,C,K,H,W), padding, use_fft, mask_first=True)`:
    y = sum_k h_k * (w_k . x) when `mask_first=True` (source-indexed form, right for PSFs measured at star positions);
    `mask_first=False` (gather form) added in v0.4.2. Explicit adjoint. Implementation: Python loop over K full-frame
    convolutions, filter FFTs recomputed at each call. Memory (float32, multipliers, C = 3, K = 10): about 1.4 GB at
    12 MP, 2.9 GB at 24 MP; fits in 16 GB.
  - `ProductConvolutionBlurGenerator` only wraps a synthetic PSF generator; its recipe (SVD of the PSF grid into
    eigen-PSFs + thin-plate-spline coefficient maps) is about 20 lines to copy, but the full-resolution TPS evaluation is
    too heavy at 12-24 MP: evaluate on a coarse grid and upsample.
  - `TiledSpaceVaryingBlur(filters, patch_size, stride=patch//2, blending_mode="bump"|"linear", use_fft=True)` (v0.4.0):
    one PSF per overlapping tile, `valid` padding only. Note: the docstring states the mask-first formula, but the code
    blends on the output (gather form); check with `adjointness_test` and against PyLops on crops.
  - `LinearPhysics(A, A_adjoint=None)` builds the adjoint by autograd (`torch.func.vjp`); `adjointness_test()`,
    `compute_norm()`; composition with `*`. `Demosaicing(pattern="bayer")` puts R at (1,1) and B at (0,0), i.e. **BGGR**
    (verified in `physics/inpainting.py`): for RGGB use `Inpainting(mask=custom)`. No warp physics (a `grid_sample`
    LinearPhysics is short to write).
  - Multi-frame: `StackedLinearPhysics`, `StackedPhysicsDataFidelity`, `StackedPhysicsLoss`; `deepinv.distributed`
    (beta) can split operators and tile a denoiser. Everything assumes all frames in memory (721 x 12 MP x 4 B = 35 GB):
    hundreds of frames need our own streaming loop (ordered subsets / stochastic passes).
  - Noise / fidelity: `PoissonGaussianNoise` (simulation only); fidelities `L2`, `PoissonLikelihood(gain, bkg)`,
    `LogPoissonLikelihood`, `L1`. **No Poisson-Gaussian likelihood**: write a variance-weighted L2 or use shifted
    Poisson (`PoissonLikelihood(bkg = sigma^2 / g)` on y + sigma^2 / g).
  - Solvers: ADMM, DRS, HQS, PGD, FISTA, PDCP, MLEM (= RL), OSEM, BSREM, unfolded / DEQ; `PnP(denoiser)`, `RED`,
    `DPIR(denoiser=...)` accept any module with `forward(x, sigma)`. `OSEM` accepts frames as subsets but is documented
    for tomography only (untested on blur).
  - Pretrained `RAM` model (Poisson-Gaussian aware, natural images; weights on Hugging Face, licence not stated) could be
    a zero-shot baseline with our physics.
  - Evidence: only a radio-interferometry astro example; no optical-astro benchmark of these operators. Effort: operator
    from Gaia PSF field 1-2 days; single-image PnP on the stack 2-3 days; joint multi-frame solver 1-2 weeks.
- **torchmfbd** (https://github.com/aasensio/torchmfbd, arXiv 2505.10639). Verified in source: `Deconvolution.deconvolve(..., modes_in=...)`
  uses `modes_in` only as initial value, then `modes.requires_grad_(True)` and an optimiser group with `lr_modes`
  (default 0.01). `DeconvolutionSV.deconvolve(..., infer_modes=False)` removes the PSF from the model rather than fixing
  it. PSF basis = Zernike / KL wavefront, or NMF / PCA basis loaded from an npz. Square patches, mono, Gaussian
  likelihood, no Bayer, no Poisson. Worth copying: Fourier-domain einsum over PSF modes, per-frame warp fields inside the
  joint model, IUWT sparsity regulariser.
- **PyLops `NonStationaryConvolve2D(dims, hs, ihx, ihz, engine="cuda", dtype="float32")`**: regular grid of odd-sized
  PSFs, each pixel blends the 4 nearest PSFs bilinearly; direct (not FFT) cost; CUDA kernel via numba. Use float32
  (float64 is very slow on GeForce cards). Role: exact reference to measure the approximation error of our fast
  eigen-PSF / tiled operator on crops.
- **ImageMM**: per-frame forward model with PSF measured from stars and refined variationally, multiplicative MM
  updates, about 0.12 s per iteration at 1000^2 on a V100 (33 HSC frames). Re-implement in PyTorch on deepinv operators,
  adding space-varying PSF, RGGB sampling, warps inside the operator, Poisson-Gaussian weights and N2N PnP steps.
- **STARRED** (https://gitlab.com/cosmograil/starred, Michalewicz et al. 2023 JOSS; Millon et al. 2024 AJ 168, 55):
  deconvolves to a finite "narrow" target PSF (about 2 px FWHM) rather than to a delta, which avoids ringing: an idea to
  adopt for our deconvolution target. Stamp-scale (32^2 -> 64^2).
- **AstroPhot** (https://github.com/Autostronomy/AstroPhot): still PyTorch; LM fitter memory makes a full-frame
  pixel nebula infeasible, but a crop with a pixel nebula + Gaia point sources and per-frame PSFs is expressible: an
  independent check of our forward model and photometry.
- No candidate provides: a Poisson-Gaussian likelihood with a space-varying operator; Bayer RGGB sampling of
  non-interpolated frames inside the forward model; streaming of hundreds of 12-24 MP frames; a PSF-field builder from
  Gaia stars per frame and per colour. All are small additions on top of deepinv.

---

## 2. Deconvolution tools usable as baselines on a stack

| Candidate | Licence code / weights | Activity (verified) | Stars | Framework, GPU, Windows | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **Siril 1.4.4** RL / Wiener / split Bregman (+ `makepsf`) | GPL-3.0+ / none | 1.4.4 released 2026-06-16 (1.4.0 2025-12-04) | 194 (GitLab) | C/C++, CPU only (threads + FFTW, no GPU code in `src/filters/deconvolution`), native Windows, `siril-cli` headless | linear (RL gradient-descent mode) or stretched (multiplicative RL); mono or RGB; one **space-invariant** PSF, can load ours | **bench baseline** (classical reference) |
| **GraXpert 3.1.0rc2** `deconv-obj` / `deconv-stellar` | GPL-3.0 / **CC BY-NC-SA 4.0** | main 2025-01-01; develop 2025-10-03; rc3 2025-10-03 (macOS-only fix); last activity branch 2026-01-12 | 325 | ONNX Runtime: DirectML on Windows, CUDA on Linux | linear accepted (log-domain tile normalisation); each channel separately; 512 px tiles | **bench baseline** |
| **Seti Astro Suite Pro (SASpro) MFDeconv** | GPL-3.0 / no weights (algorithmic) | v1.22.5 2026-10-10 (very active) | 112 | PyTorch CUDA / DirectML / CPU, Windows | registered frames, PSF measured per frame from stars, luma or per channel, optional SR x2/x3 | **bench baseline + code reference** (closest free tool to our target) |
| Cosmic Clarity AI4 sharpen (inside SASpro) | app GPL-3.0 / **proprietary, SASpro-only** | 2026-10 | - | NAFNet, PyTorch/ONNX, CUDA/DirectML | auto-stretch if median < 0.08, then un-stretch | **bench baseline only, inside SASpro**; never a component |
| Aberration Remover 2.0.1 (R. Alberghi) | Siril script GPL-3 / weights CC BY-NC-SA 4.0 | model 2.0.1 2026-02-11 | 5 | ONNX via Siril `pyscript`, GPU | linear, after background + colour calibration, before deconvolution | **bench baseline for Iris (Newton coma)** |
| SyQon Parallax Nano | Siril script MIT / closed weights, terms PDF unverified | unverified | - | unverified | unverified | optional bench (login required) |
| BM3D deblurring (`pip bm3d` 4.0.3) | "free for non-commercial use" | unverified | - | CPU | known PSF, Gaussian noise | optional classical bench |
| AstroSharp, Galaxy-Deconv, Astraios, PI-AstroDeconv, AstroClearNet | see notes | | | | | **skip** |

Notes.

- **Siril deconvolution** (docs https://siril.readthedocs.io/en/stable/processing/deconvolution.html, source
  https://gitlab.com/free-astro/siril). Richardson-Lucy in two flavours: multiplicative (for stretched data) and gradient
  descent (default, recommended for linear data), regularised by TV or Frobenius norm of the Hessian; Wiener (Gaussian
  noise, weak on deep sky); split Bregman TV (starless or lunar images). Blind PSF: l0 descent (Anger et al., IPOL) or
  spectral irregularities. PSF from stars: averaged Gaussian/Moffat fit of selected or auto-detected stars (linear
  only). PSF file: square, odd size, mono (then only L of LAB is deconvolved) or 3-channel (one PSF per channel). Commands:
  `makepsf {load f | stars [-sym] | blind ... | manual ...}`, `rl [-loadpsf=] [-alpha=] [-iters=] [-tv|-fh] [-mul]`,
  `sb`, `wiener`, sequence versions `seqrl`, `seqsb`, `seqwiener` (verified in the command reference). For sequences
  the docs state "the PSF will be calculated only for the first image" and re-used for all frames, so a per-frame PSF
  needs one script call per frame. Python: `sirilpy` drives a *running* Siril (`cmd()`, `get/set_image_pixeldata`), ships
  with Siril, not on PyPI. Limitation: no space-varying PSF, CPU only. Integration: `siril-cli -s script.ssf` with
  `makepsf load our_psf.fits` then `rl`, and also `makepsf stars` (its own PSF) as a second baseline. Effort 0.5-1 day.
- **GraXpert deconvolution** (https://github.com/Steffenhir/GraXpert). Object model added in 3.1.0rc1 (2024-11-10),
  stellar model in 3.1.0rc2 (2025-01-01); last Windows binary with deconvolution is 3.1.0rc2
  (`graxpert-windows-amd64.msi`); the "latest" stable is still 3.0.2 (no deconvolution). Weights licence CC BY-NC-SA 4.0
  (`licenses/Deconvolution-Model-LICENSE.html`, develop branch). Models are `model.onnx` zips fetched from a MinIO/S3
  bucket with credentials injected at build time (a plain source checkout cannot download them; the official binary or
  the PyPI test build 3.2.0a3 from the fork geeksville/GraXpert can). CLI:
  `GraXpert-win64.exe img.fits -cli -cmd deconv-obj|deconv-stellar -strength 0..1 -psfsize <FWHM px> -batch_size 4 -gpu true -ai_version x.y.z -output name`.
  From `deconvolution.py`: 512 px tiles (stride 448), each channel processed as 1-channel input, log + z-score
  normalisation, residual predicted in log space, output clipped to [0, 1]. **psfsize is clamped** (computed from the
  code): stellar model effective FWHM about 3.9-10.2 px, object model about 1.8-13.5 px (v1.0.1). Object and stellar
  models are designed to be chained. No published evaluation (forum comparisons only). Effort 0.5 day (CLI) or 1 day
  (calling `model.onnx` with onnxruntime-gpu in our harness, copying the tiling).
- **SASpro MFDeconv** (https://github.com/setiastro/setiastrosuitepro, `src/setiastro/saspro/mfdeconv.py`, function
  `multiframe_deconv_normal_rebuild(paths, out_path, iters, kappa, color_mode="luma"|"PerChannel", seed_mode, rho="huber", use_star_masks, use_variance_maps, super_res_factor)`;
  module structure verified). Multi-frame multiplicative RL on registered frames; PSF measured per frame from stars
  (SEP-based `compute_psf_kernel_for_image`, Gaussian fallback); Huber-robust weights x inverse-variance maps x star
  masks; update ratio clamped to [1/kappa, kappa]; optional super-resolution with a super-sampled PSF; frames streamed
  through memmaps. Limitations: one PSF per frame (space-invariant inside a frame), works on registered (already
  interpolated) frames, no Bayer model, no prior beyond the clamp/early stop; the module imports PyQt6, so a small
  wrapper is needed to run it headless. No published evaluation. Use: feed it 20-40
  registered sub-stacks (not 721 single frames) of the same batch, compare with our joint deconvolution; read the code as
  a reference implementation. Effort 1-2 days.
- **Cosmic Clarity AI4** (in SASpro). Licence file `COSMIC_CLARITY_LICENSE.txt` (SASpro release assets): personal,
  non-commercial use **only inside SASpro**; no use in another pipeline, no weight extraction, distillation or
  fine-tuning. The old standalone repo setiastro/cosmicclarity (MIT code, archived 2026-09-16) still contains old `.pth`
  weights; Siril's `CosmicClarity_Native.py` runs AI4 models outside SASpro, which seems to contradict that licence: do
  not use that path. Bench only via SASpro (GUI or its headless entry point). Effort 0.5 day.
- **Aberration Remover** (https://github.com/riccardoalberghi/abberation_models): corrects defocus, astigmatism, coma,
  trefoil, spherical (not seeing, not lateral chromatism) on linear data. Relevant to the Newton 130/900 coma. Effort
  0.5 day through Siril.
- **Skips.** AstroSharp (deepskydetail/AstroSharp, MIT, R Shiny app on stretched TIFF, sharpening not deconvolution);
  Galaxy-Deconv (Lukeli0425, weak-lensing stamps on simulations); Astraios (alpha, 2 stars); PI-AstroDeconv
  (arXiv 2403.01692) and AstroClearNet (arXiv 2504.06463): no code found (see section 1).
- No free tool trained like BlurXTerminator (space-varying, PSF-conditioned, linear) with redistributable weights was
  found. The closest free ones are GraXpert (CC BY-NC-SA weights) and Cosmic Clarity (SASpro-only).

---

## 3. Denoising

| Candidate | Licence code / weights | Activity (verified) | Stars | Framework, GPU, Windows | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **ASTERIS** (Science 2026, arXiv 2602.17205) | **MIT / CC-BY-4.0 (Zenodo)** | last commit 2026-06-23 | 55 | PyTorch 2.5, CUDA 12.1, tested on Ubuntu; Windows unverified | 4 or 8 aligned exposures (3D stack), mono, linear | **component (ideas) + retrained bench baseline** |
| **GraXpert denoise** | GPL-3.0 / CC BY-NC-SA 4.0 | since 3.0.0 (2024-04-22) | 325 | ONNX Runtime, DirectML/CUDA | linear, 3-channel (mono replicated) | **bench baseline** |
| **Siril `denoise`** (NL-Bayes, `-vst`, `-da3d`, `-sos`) | GPL-3.0+ | 1.4.4 | 194 | C, CPU, Windows | linear (with VST) or stretched | **bench baseline** (classical floor) |
| **BM3D with noise PSD** (`pip bm3d`) | free non-commercial | unverified | - | CPU | Gaussian noise with **correlated** PSD | **bench baseline** (matches our correlated stack noise) |
| **CAREamics** (N2V, N2V2, structN2V, N2N, CARE, PN2V, HDN, MicroSplit) | BSD-3 | v0.3.4 2026-09-23; commit 2026-10-06 | 154 | PyTorch Lightning, Python >= 3.11, GPU, Windows | single image or pairs, mono/multichannel | **bench baseline** (N2V2 / structN2V) + reference N2N |
| **deepinv self-supervised losses** | BSD-3 | v0.4.2 2026-08-30 | 827 | PyTorch | any physics | **component** (extra loss terms for our N2N) |
| ZS-N2N (Mansour & Heckel, CVPR 2023) | Colab notebook only | - | - | PyTorch, tiny net | single image | bench baseline (30-line re-implementation) |
| Cosmic Clarity AI4 denoise, DeepSNR (Misiura), SyQon Prism mini | closed weights; terms SASpro-only / unverified / unverified | 2026 | - | ONNX / PyTorch, Windows | stretched internally | bench only |
| NVlabs/noise2noise, juglab/n2v, czbiohub/noise2self, Neighbor2Neighbor, Blind2Unblind, R2R, Noise2Score repos | see notes | 2019-2024 | | | | **skip** (use deepinv / CAREamics versions) |
| Noise2Astro, AstroSURE, Poisson2Gaussian, TDR, AstroDenoisePy | see notes | | | | | **skip** (no code, or same idea as ours) |

Notes.

- **ASTERIS** (paper https://arxiv.org/abs/2602.17205, DOI 10.1126/science.ady9404; code
  https://github.com/freemercury/ASTERIS_THU, also Zenodo 10.5281/zenodo.17115038; weights Zenodo 17114980 and 18358472,
  about 296 MB, models `ASTERIS{4,8}_nrc{short,long}` and `ASTERIS8_SUBARU_ks`; demo data Zenodo 17105027, 2.4 GB). The
  code is public, confirmed by the paper's availability statement. Method: self-supervised "spatiotemporal" denoising of a
  stack of 4 or 8 aligned exposures; training samples of 16 exposures split into two sets of 8 with independent noise;
  loss = MSE between co-added input and target sets + MAE between paired single exposures; 3D U-Net with Restormer-style
  3D attention; a 3-sigma threshold across exposures separates bright pixels (median co-added, not processed) from the
  faint part (z-scored, denoised). Reported: about +1.0 mag deeper detection at 90 % completeness and purity on mocks,
  PSF and photometry preserved, about 3x more z >~ 9 candidates on JWST. Cost: inference 18 s for 8 x 1650^2 px on
  4 x 40 GB GPUs; training about 26 h. Limitations for us: pretrained weights are JWST NIRCam / Subaru (not
  transferable); mono; 16 GB VRAM needs smaller patches; Windows untested (depends on swanlab). It is the multi-frame
  generalisation of our half-stack Noise2Noise. Use: (a) borrow multi-sub-stack input, the "average + frame" loss, the
  bright/faint split and the completeness/purity metric; (b) retrain on our sub-stacks per CFA plane as a baseline.
  Effort 1-2 weeks (retraining).
- **GraXpert denoise**: CLI `-cmd denoising -strength 0..1 -batch_size -ai_version -gpu`. From the code: 256 px tiles,
  per-channel (x - median)/MAD x 0.04 normalisation, pixels above about median + 250 MAD (model v2+) not denoised,
  strength = linear blend with the input. Evidence: blog comparisons only. Effort 0.5 day.
- **Siril `denoise`**: NL-Bayes with optional generalised Anscombe VST, DA3D, SOS iterations. Effort 0.5 day.
- **BM3D**: accepts a noise power spectral density, which matters because Bayer drizzle, registration interpolation and
  debayering spatially correlate the noise of our stacks. Effort 0.5-1 day.
- **CAREamics** (https://github.com/CAREamics/careamics): the maintained PyTorch successor of juglab/n2v; algorithms
  verified in source: N2V, N2V2, structN2V, N2N, CARE, PN2V, HDN, MicroSplit, DenoiSplit. Effort 1 day.
- **deepinv losses** (`deepinv/loss`): Neighbor2Neighbor, R2RLoss (generalised R2R, Gaussian / Poisson / Gamma),
  ScoreLoss (Noise2Score), SURE (Gaussian, Poisson, Poisson-Gaussian), SplittingLoss (measurement splitting; with
  pixelwise masks a Noise2Self-type loss), EquivariantSplittingLoss, EI and MC losses (self-supervised deblurring). No
  Blind2Unblind, no ZS-N2N, no named Noise2Noise loss (`SupLoss` with a noisy target is equivalent). Use: add a
  Poisson-Gaussian SURE or GR2R term to our N2N training, or self-supervised fine-tuning on the full stack. Effort 2-3 days.
- **Correlated-noise warning**: blind-spot / single-image methods (N2V, Noise2Self, Neighbor2Neighbor, Blind2Unblind,
  ZS-N2N) assume pixel-independent noise; on our drizzled stacks they will under-denoise or leave artefacts unless
  structN2V masks are used or they run on CFA-domain data. Half-stack N2N avoids that, but keeps any pattern common to
  both halves (master calibration frames, common registration grid) as "signal".
- **Skips**: NVlabs/noise2noise (TensorFlow 1, CC BY-NC 4.0, last commit 2021); juglab/n2v (TF2, being replaced by
  CAREamics); czbiohub/noise2self (2019 notebooks); TaoHuang2018/Neighbor2Neighbor (BSD-3, 2021) and R2R / Noise2Score
  repos (use deepinv); zejinwang/Blind2Unblind (no licence file); Noise2Astro (arXiv 2209.07071), AstroSURE
  (arXiv 2604.16793, verified: no code; but its detection-metric protocol is worth copying), Poisson2Gaussian
  (arXiv 2606.23098, "code will be available"); TDR (zimugh/Denoising-TDR, notebook, no clear licence);
  AstroDenoisePy (p7ayfu77/astro-csbdeep, BSD-3, TF2, CARE trained with N2N, same idea as ours);
  Learning2Recorrupt (watch); SCUNet-based scripts (natural-image priors on stretched data).

---

## 4. Star removal

| Candidate | Licence code / weights | Activity (verified) | Stars | Framework, GPU, Windows | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **StarNet2 CLI 2.6.2** (installed: 2.5.4) | closed, free; licence restricts use (see notes) | 2.6.2 package 2026-09-19 (2.6.0 2026-09-04) | - | ONNX Runtime: DirectML bundled, CUDA by DLL swap; Windows/Linux/macOS | 8/16-bit TIFF/PNG or FITS in [0, 1]; **`--linear` since 2.6.0**; RGB recommended; min 512 x 512 | **bench baseline**; component only as gated residual cleaner |
| **nox** (charvey2718) | **MIT code and MIT weights** | `nox.exe` v1.1.0 2025-12-25 | 8 | TensorFlow/Keras; exe; GPU on Windows untested by author | TIFF 8/16/32-float (no FITS), `-l` temporary stretch for linear | **bench baseline + component template** (synthetic-star training) |
| **compact-source-removal** (Madarasz et al. 2025, A&A 696 A37) | MIT | last commit 2025-11-21 | 2 | PyTorch partial-conv U-Net | linear FITS (Herschel), masks given | **component** (architecture + training recipe; retrain) |
| SyQon Zenith (Siril script) / SyQon Studio CLI | script by Siril/SyQon; weights proprietary, licence PDF unreachable (unverified) | 2026 | - | PyTorch NAFNet, DirectML | temporary IHS autostretch for linear | bench baseline only |
| Cosmic Clarity Dark Star | SASpro-only proprietary models | 2026 | - | PyTorch | auto-stretch | **skip** (licence); optional manual comparison |
| StarNet v1 (nekitmm/starnet) | MIT code / CC BY-NC-SA weights | 2022-09-12 | 313 | TF1/TF2 | stretched 8-bit | **skip** (superseded) |
| starrem2k13 | MIT | v7.0 2025-06-16 | 11 | ONNX CPU | 8-bit | **skip** (trained on 3 images) |
| MRF (AstroJacobLi/mrf), elderflower | MIT | 2021 / 2025-06-25 | - / 6 | numpy, GalSim | high-res reference / bright stars | **component ideas** (halo and wing subtraction, see section 7) |
| GraXpert | - | - | - | - | - | no star removal (verified: CLI commands are only background-extraction, denoising, deconv-obj, deconv-stellar) |

Notes.

- **StarNet2** (https://starnetastro.com/cli-tools/starnet/, CLI reference
  https://starnetastro.com/documentation/starnet/command-line-tool/). Local install
  `F:\photos\heart_nebula\tools\starnet2_2.5.4\...` (2.5.4, ORT backend, no `--linear`). Its `StarNet2_weights.onnx`
  (inspected read-only): pix2pix-style U-Net, about 32.9 M parameters, input NHWC [N, 512, 512, 3] float32 in [0, 1],
  output clipped to [0, 1]. Version 2.6.x adds `--linear` (verified on the official page): per-channel MTF stretch,
  network, inverse stretch, restoration of clipped shadows; negative values accepted; **values above 1 clipped with a
  warning, no automatic normalisation** (scale float ADU to [0, 1] first); options `--shadows-clipping`,
  `--target-background`, `-u` (2x upsample, for tight stars), `-s` stride, `-m` mask, `-n` unscreen star layer;
  `-e/--eight` removed in 2.6.0. Licence (LICENSE.txt of the local install): use "solely for astrophotography image
  processing"; outputs may not be used to create commercial software "including for training neural networks"; no
  sharing of full-resolution inputs that demonstrate its performance (relevant if bench comparisons are published).
  Evidence: qualitative amateur comparisons only; our own finding: it eats faint diffuse emission. Use: upgrade to 2.6.2,
  run `--linear` on [0, 1]-scaled linear FITS (test with and without `-u`: our stack FWHM is ~4 px or more, so `-u` may
  not be needed), and in the chain accept its star layer only inside Gaia/PSF footprints. Effort 0.5-1 day.
- **nox** (https://github.com/charvey2718/nox; LICENSE and README verified: "My code and trained weights ... are made
  available under the MIT License"; GitHub's licence detector shows NOASSERTION because of a preamble). Same generator
  architecture as StarNet v1, trained on a **fully synthetic** dataset (`GenerateStars.py`: Moffat/Gaussian stars with
  random FWHM, Poisson noise, gain, MTF stretch, spikes, colour fringing, asymmetry, on photo-denoising backgrounds). The
  author says generalisation "remains to be proved". Use: bench `nox.exe -l`; real value is as a template to train our
  own linear-domain PyTorch star remover with our measured PSFs (spikes, coma), our noise model and real starless
  backgrounds from our data. Effort 0.5 day (bench), 1-2 weeks (own model).
- **compact-source-removal** (https://github.com/xMediMadix/compact-source-removal, arXiv 2502.18345): partial
  convolution U-Net, S/N-adaptive masks, trained by injecting sources into real linear backgrounds; the paper quantifies
  photometric accuracy and noise / power-spectrum preservation. Inpainting acts only inside the masks, so outside stars
  the background is untouched by construction. Use: retrain with Gaia footprint masks on our star-subtracted
  backgrounds. Effort about 1 week.
- **Siril integration**: since 1.4 Siril calls StarNet through the Python script `StarNetAstro/StarNet.py`
  (`pyscript StarNet.py --linear --masks descreen`, supports CLI 2.5.0-2.6.2); the legacy C `starnet` command remains
  for old versions.

---

## 5. Background extraction

| Candidate | Licence code / weights | Activity (verified) | Stars | Framework, GPU, Windows | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **GraXpert AI background** | GPL-3.0 / **CC BY-NC-SA 4.0** | stable 3.0.2 (2024-05-03); 3.1.0rc2/rc3; PyPI test build 3.2.0a3 2026-09-22 (fork) | 325 | ONNX Runtime; CLI + importable Python | linear OK; whole image downsampled to 240 x 240 | **bench baseline**; optional prior only |
| **Siril 1.4 `subsky` / `seqsubsky`** (poly <= 4, RBF) | GPL-3.0+ | 1.4.4 | 194 | C, CPU, Windows | linear; `seqsubsky` works per frame on **CFA** subplanes | **bench baseline + idea** (per-frame CFA gradient) |
| VeraLux Nox (Siril script) | GPL-3.0 | v1.0.1 (2025) | - | Python in Siril | linear | bench baseline |
| AutoBGE (Siril script, AutoDBE port) | GPL-3.0 | unverified | - | Python in Siril | linear | optional bench |
| photutils `Background2D` 3.0.0, SEP 1.4.1 | BSD-3 / LGPL-3 | 2026-04-17 / 2025-02-18 | 310 / - | CPU, Windows (in venv) | any | component for local sky / RMS in forced photometry only |
| Gnuastro NoiseChisel 0.24 | GPL-3.0 | 2025-11-26 | - | C, WSL only | | **skip** (dense Milky Way field leaves no undetected tiles) |

Notes.

- **GraXpert** (`graxpert/background_extraction.py`): the whole image is resized to 240 x 240 (padded 256), each
  channel normalised (x - median)/MAD x 0.04 and clipped, the network output smoothed (sigma = 20 x smoothing, then 3)
  and upscaled. It can only model scales >= image/240 (about 18 px for a 4240 px wide frame, more on a drizzled stack), and in a field filled with
  H-alpha (Heart & Soul) it can absorb real large-scale emission / IFN. CLI:
  `GraXpert-win64.exe img.fits -cli -cmd background-extraction [-correction Subtraction|Division] [-smoothing 0..1] [-bg] [-ai_version] [-gpu true]`.
  Python: `from graxpert.background_extraction import extract_background(...)`. Effort 0.5-1 day.
- **Siril**: `subsky {-rbf | degree} [-samples] [-tolerance] [-smooth] [-existing]`; `seqsubsky` on CFA frames (docs
  recommend order 1 before Bayer drizzle); samples settable from Python (`set_bgsamples`). Idea to test in our chain
  (data-driven, no instrument rule): per-frame low-order gradient fitted on (registered frame - reference stack) over
  star-masked pixels, where the nebula cancels; effort about 1 day.

---

## 6. Stacking / Bayer multi-frame super-resolution

| Candidate | Licence | Activity (verified) | Stars | Framework, GPU, Windows | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **Siril 1.4.4 drizzle** (HST algorithm port, Bayer drizzle) | GPL-3.0+ | 1.4.0 2025-12-05, 1.4.4 2026-06-16/17 | 194 | C, CPU, Windows; `siril-cli`; sirilpy | raw CFA frames; SIP distortion | **bench baseline** |
| **STScI drizzle 3.0.0** | BSD-3 (AURA) | release 2026-09-24, commit 2026-09-28 | 59 | C core + Python, CPU | any frames + **explicit pixmap** + weight map | **component (validator of our GPU Bayer drizzle)** |
| Handheld Multi-Frame SR (Jamy-L) | MIT | release 1.4 2026-09-03, commit 2026-09-30 | 214 | numba-cuda + torch; WSL recommended on Windows | raw Bayer burst of ~13-20 frames | **skip** as baseline; possible kernel idea |
| DeepSkyStacker 6.2.3 beta | BSD-3 | 2026-10-02 | 1140 | C++, Windows | | optional bench (no pixfrac drizzle) |
| SyQon Stacking Studio | closed, account | 2026 | - | | CFA drizzle | optional bench |
| astroalign, reproject, drizzlepac, burst-SR networks | | | | | | **skip** |

Notes.

- **Siril drizzle** (https://siril.readthedocs.io/en/stable/preprocessing/drizzle.html; source `src/drizzle/cdrizzlebox.c`
  carries the AURA copyright, i.e. a port of STScI cdrizzle): pixmap-based, CFA colour selected per input pixel; kernels
  square (flux-conserving), point, turbo, gaussian, lanczos2/3 (only at scale = pixfrac = 1); `register | seqapplyreg ...
  -drizzle -pixfrac= -kernel= -scale= -disto=master`; master flat usable as pixel weight. Frames are drizzled into an
  **intermediate sequence on disk** and then stacked with rejection (whether per-pixel drizzle weights are carried into
  the stack is unverified). Disk: about 100 GB for 721 frames at scale 1 (use C: or E:, not F:). Effort about 1 day plus
  run time.
- **STScI drizzle** (https://github.com/spacetelescope/drizzle): `Drizzle(kernel=...)`,
  `add_image(data, exptime, pixmap, weight_map, pixfrac, ...)`, outputs `out_img`, `out_wht`, `out_ctx`; 2.2.0 added
  squared-weight co-adds for variance. Bayer drizzle = three Drizzle objects with weight maps equal to 1 only on each
  channel's CFA sites. Our per-frame polynomial maps plug in directly as pixmaps. Use as the CPU reference to check flux,
  weights, PSF and noise of our GPU drizzle on a 50-100 frame subset. Effort about 1 day.
- **Handheld MFSR** (https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution, Wronski et al. 2019,
  re-implementation): joint demosaic + SR + denoise from a raw burst with block matching + optical flow per tile and
  anisotropic kernel regression. Built for short in-memory bursts of bright scenes; flow on noisy 15 s star fields and
  non-flux-conserving kernels make it a poor fit. At most, test its anisotropic merge kernel inside our accumulator.

---

## 7. PSF modelling (spatial, colour, bright-star wings)

| Candidate | Licence | Activity (verified) | Stars | Framework, GPU, Windows | Colour / wings | Verdict |
|---|---|---|---|---|---|---|
| **photutils 3.0.0** EPSFBuilder, GriddedPSFModel, PSFPhotometry | BSD-3 | 3.0.0 2026-04-17; commit 2026-10-10 | 310 | numpy/scipy, CPU, native Windows (in venv) | no colour axis; wings only inside stamp | **bench baseline + validation oracle** |
| **Piff 1.6.0** | BSD-like | 2025-11-24 | 69 | numpy/GalSim, CPU; **no Windows wheels** (verified on PyPI) -> WSL2 | **colour as interpolation key** (DES Y6) | **bench baseline (WSL2) + design reference** |
| **Gnuastro 0.24 `astscript-psf-*`** | GPL-3.0+ | 2025-11-26 | - | C/shell, WSL only | extended PSF to arcminutes, no spatial/colour variation | **component (re-implement recipe)**, optional WSL bench |
| **elderflower** | MIT | commit 2025-06-25 | 6 | numpy, dynesty, SExtractor needed (WSL in practice) | Moffat core + multi-power-law aureole | **component (port functional form)** |
| PSFEx 3.24.2 | GPL-3.0 | release 2023-04-13; commit 2025-03-27 | 44 | C, no Windows build | polynomial in any catalogue key (colour unverified) | **skip** (covered by Piff / photutils) |
| STARRED 1.7.1 | GPL-3.0+ | 2026-08-13 | 6 (GitLab) | JAX; no native-Windows GPU | Moffat + regularised pixel grid | skip for PSF production; pattern to copy (see section 1) |
| psfr 0.1.1, psfmachine 1.1.4, AstroPhot 0.18.0, crowdsource 0.5.9 | BSD-3 / MIT / GPL-3.0 / MIT | 2026-09 / 2023-12 / 2026-06-25 / 2026-08 | 34 / - / 106 / - | | | **skip** (see notes) |
| dLux, poppy, hcipy, prysm, WaveDiff | BSD-3 / BSD-3 / MIT / MIT / MIT | 2026 | | | physical chromatic PSF | component idea (small torch pupil model), not a dependency |

Notes.

- **photutils** (https://github.com/astropy/photutils): Anderson & King ePSF building; `GriddedPSFModel` interpolates
  ePSFs built per region; `PSFPhotometry` / `IterativePSFPhotometry` with fixed x_0/y_0 = forced photometry; masked
  pixels and NaNs ignored, so saturated cores can be treated as missing data. Measured locally (CPU, synthetic
  2000 x 2000, FWHM 2.5 px): EPSFBuilder on 500 stars 11.5 s (oversampling 2) / 15.2 s (oversampling 4); forced
  PSFPhotometry 5.9 s for 3000 stars (about 2 ms per star). One ePSF per tile x colour x frame on CPU therefore costs
  days on 721-1268 frames: use as reference and validator on a subset, not as the production engine. Untested idea: run
  EPSFBuilder directly on raw CFA frames with other-channel pixels masked. Effort about 1 day.
- **Piff** (https://github.com/rmjarvis/Piff): `PixelGrid` model + `BasisPolynomial` interpolation with arbitrary
  `keys`; colour enters via `property_cols` in the input and `keys: [u, v, colour]`. DES Y6 (Schutt et al. 2025,
  arXiv 2501.05781): PixelGrid jointly fitted with 2nd order in position and **linear in colour**, size offsets between
  colour groups reduced from about 3 % to sub-percent. Saturated pixels: use a bad-pixel weight map, not `satur` (which
  drops the star). Use: run on a few frames in WSL2 as an independent colour-PSF reference; copy the (x, y, lambda_eff)
  polynomial design into our PyTorch model (with thousands of stars per frame, more than linear in lambda is
  affordable). Effort 2-3 days.
- **Gnuastro PSF scripts**: stamps normalised in a ring and stacked per magnitude range (faint stars for the core,
  saturated bright stars with masked cores for the wings), parts joined with a scale factor, then per-star subtraction.
  Evidence: Infante-Sainz et al. 2020 (MNRAS 491, 5317), SDSS PSF to about 8 arcmin; Eskandarlou et al. 2025
  (arXiv 2510.12940, J-PLUS) find the PSF ring radius grows in redder filters (wings are wavelength dependent).
  Re-implementation in PyTorch with Gaia magnitudes and per-channel synthetic fluxes: 2-3 days.
- **elderflower** (https://github.com/NGC4676/elderflower, Liu et al. 2022 ApJ 925, 219): Bayesian fit of the
  wide-angle PSF from many bright stars plus sky; Moffat core + multi-power-law aureole (verified in `modeling.py`);
  developed for Dragonfly (lens-based telephoto array, closest in spirit to the TS61). Port the functional form per
  channel in PyTorch with Gaia fluxes and masked cores: 2-4 days.
- **Skips**: PSFEx (old toolchain, no Windows, no wing model); psfr (always refits centroids, no spatial/colour
  variation); psfmachine (Kepler/TESS; the concept - fixed Gaia scene, linear flux solve - is what we already do);
  AstroPhot (pixelated PSF per star does not scale to thousands of stars per frame); crowdsource (TensorFlow
  dependency, PSF variation listed as future work); MRF (needs a matching high-resolution image; possible later with
  Pan-STARRS).
- **Colour-dependent PSF for OSC / refractors**: no open tool or paper specific to OSC cameras or refractors was found.
  Precedents: DES Y6 Piff (linear colour term), Eskandarlou 2025 (wavelength-dependent wings), WaveDiff (SED-weighted
  polychromatic PSF, space optics). Suggested design (not a cited source): per Bayer channel, per star
  lambda_eff = integral(lambda F R_c) / integral(F R_c) from Gaia XP x channel response; a small physical pupil model
  (defocus / spherochromatism polynomial in lambda) fitted to the per-channel ePSFs as a prior for the H-alpha
  extrapolation.

---

## 8. Masked-data filling

| Candidate | Licence | Activity (verified) | Stars | Framework | Verdict |
|---|---|---|---|---|---|
| **maskfill 1.1.1** (van Dokkum & Pasha 2024, PASP 136, 034503) | MIT | release 2024-02-02; commit 2024-07-23 | 25 | numpy/scipy/astropy, Windows | **bench baseline + trivial GPU port (component)** |
| OpenCV `cv2.inpaint` (NS / Telea) | Apache-2.0 | (cv2 5.0 in venv) | - | C++, float32 single channel accepted | fast fallback / bench |
| deepinv `Inpainting` physics + PnP (DRUNet, RAM, DiffPIR...) | BSD-3 / natural-image weights (licences unverified) | v0.4.2 | 827 | PyTorch GPU | optional cosmetic bench only |
| astropy `interpolate_replace_nans`, skimage `inpaint_biharmonic`, Gnuastro `interpolate-*` | BSD / BSD / GPL | | | | **skip** |
| LaMa, astroDDPM, Euclid / radio diffusion inpainting, CMB nets, deepCR | Apache / AGPL / no code / ... | | | | **skip** (domain mismatch or no code) |

Notes.

- **maskfill** (https://github.com/dokkum/maskfill): fills each masked pixel bordering good pixels with the median of the
  good pixels in a 3 x 3 window, iterates inward, then smooths the filled region with a 3 x 3 boxcar. About 30 lines,
  linear in the number of masked pixels. Paper: best overall RMSE versus biharmonic / linear / cubic / nearest,
  especially with noise; fill is noiseless (sigma_mask = sigma (2d + 1)^-0.5 at distance d from the mask edge); cannot
  recover structure that does not cross the mask edge. Measured locally (synthetic 2000 x 2000, 3000 circular masks,
  8.5 % of pixels; RMSE versus noiseless truth inside masks): cv2 NS 0.53 s / 2.05, cv2 Telea 0.85 s / 2.43, astropy
  interpolate_replace_nans 2.68 s / 2.42 (687 px left NaN), skimage biharmonic 3.15 s / 3.60, maskfill algorithm
  (numpy re-implementation) 3.08 s / 2.71. Smooth synthetic backgrounds favour low-order methods; all fills are smoother
  than the noise. Use: torch port + noise injection with variance sigma^2 - sigma_mask^2; effort 0.5-1 day.
- For **saturated cores** of Gaia stars, the fill is not an inpainting problem: model = measured PSF x Gaia-calibrated
  flux + background (sections 1 and 7); keep the pixels flagged as missing in any likelihood.
- No production-ready astronomical inpainting network with code and weights for optical star masks was found
  (astroDDPM is a 2022 galaxy-cutout demo, AGPL; Euclid Q1 diffusion inpainting arXiv 2503.15321 has no code found).

---

## 9. Colour calibration

| Candidate | Licence | Activity (verified) | Windows | Verdict |
|---|---|---|---|---|
| **Siril 1.4.4 SPCC** | GPL-3.0+; offline Gaia XP catalogue CC BY 4.0 (Zenodo 10.5281/zenodo.14697692) | 1.4.4 2026-06-16; sensor DB commit 2026-10-02 | native, `siril-cli` | **bench baseline** (with a custom A7S-mod sensor JSON) |
| **GaiaXPy 2.1.4** | BSD-3 | 2025-12-15 | yes (in venv) | **component** (XP spectra -> our own synthetic photometry and lambda_eff) |
| GaiaXPcorrection (Huang et al. 2024) | not detected | 2026-05 | yes | optional component, inside its validity range (E(B-V) < 0.8) |
| colour-science 0.4.7 `matrix_idt` | BSD-3 | | yes | component for the final camera -> display transform |
| SASpro SFCC | GPL-3.0 | 2026-10-10 | yes | skip as tool; read `sfcc.py` ("blind sensor characterisation" idea) |
| Siril PCC, ASTAP, GraXpert | | | | **skip** (no sensor response, or no colour calibration) |

Notes.

- **Siril SPCC** (https://siril.readthedocs.io/en/latest/processing/color-calibration/spcc.html): plate-solved linear
  image; stars matched to Gaia DR3 XP sampled spectra; per-channel synthetic flux = spectrum x sensor QE x filter / CFA
  (x optional low-pass filter), same for a white reference; robust linear fit of image R/G, B/G versus catalogue ratios;
  optional Rayleigh atmosphere. Command `spcc [-limitmag=] [-catalog=gaia|localgaia] [-oscsensor=] [-oscfilter=] [-osclpf=] [-whiteref=] [-atmos ...]`.
  The offline catalogue is capped at 127 brightest XP sources per level-8 HEALPixel (about 13.7 arcmin), enough for a
  white balance, far fewer stars than our Gaia forced photometry.
- **Sensor database** (https://gitlab.com/free-astro/siril-spcc-database, GPL-3; verified by listing `osc_sensors`):
  **no Sony A7S** (22 Sony entries, IMX071 ... IMX715, ICX285AK). DSLR curves include the stock IR-cut (Canon 600D R at
  656 nm = 0.35 relative), so a DSLR entry is wrong for an astro-modified camera; use a custom JSON = bare Sony CFA QE x
  measured mod-filter curve. No public measured A7S curve was found. Our Gaia XP-derived sensor response can be
  exported in that JSON format for the bench. Effort 0.5-1 day.
- **GaiaXPy** (https://github.com/gaia-dpci/GaiaXPy): `calibrate()` gives absolute sampled spectra (336-1020 nm, 2 nm);
  `generate()` synthetic photometry for built-in or SVO systems; our own curves are better integrated ourselves
  (photon-weighted), which also gives lambda_eff per channel for the colour-dependent PSF (section 7).
- Method details worth copying from PixInsight SPCC (paid, literature only, https://pixinsight.com/doc/docs/SPCC/SPCC.html):
  repeated-median regression (breakdown 0.5), fine integration grid.
- Gaia DR4 is announced for December 2026 (XP epoch spectra): keep the XP loader version-agnostic.

---

## 10. What to add to the benchmark, prioritised

Every item needs an installation, so each needs the user's go (PLAN principle 8). Efforts are integration estimates
(wrapper + run on level-1 batch + scoring), not including run time on the full data.

### Priority 1: shelf baselines for J0 (cheap, needed to fill "who wins what")

| # | Item | Component(s) / milestone | What exactly | Effort |
|---|---|---|---|---|
| 1 | **Siril 1.4.4 headless chain** | stacking J0, deconvolution J2, background J5, denoise J3, colour J6 | `siril-cli -s` scripts: calibrate, register `-disto master -drizzle` (Bayer drizzle, scale 1 / pixfrac 1, then 1.5 / 0.7), stack with rejection; `rl` with our PSF (`makepsf load`) and with `makepsf stars`; `subsky -rbf`; `denoise -vst`; `spcc -catalog=localgaia` with a custom A7S-mod sensor JSON | 2-3 days (one script set); ~100 GB intermediate on C:/E: |
| 2 | **StarNet2 2.6.2 `--linear`** (upgrade from 2.5.4) + **nox.exe `-l`** | stars J4 | on [0, 1]-scaled linear stack, with / without `-u`; score inside and outside Gaia footprints | 1 day |
| 3 | **GraXpert 3.1.0rc2 CLI** | background J5, denoise J3, deconvolution J2 | `background-extraction`, `denoising`, `deconv-obj` then `deconv-stellar` (psfsize in its valid range) | 1 day |
| 4 | **SASpro MFDeconv** | deconvolution J2 (multi-frame) | headless wrapper; 20-40 registered sub-stacks of the batch; per-frame star PSF; luma and per-channel | 1-2 days |
| 5 | **photutils 3.0 GriddedPSFModel + forced PSFPhotometry** | PSF J1, photometry J4 | per tile and per Bayer channel on a frame subset; saturated pixels masked; reference for our PSF field | 1 day |
| 6 | **STScI drizzle 3.0** | stacking J0 (validator) | three Drizzle objects with CFA weight maps and our pixmaps; compare flux / weights / PSF / noise with our GPU drizzle on 50-100 frames | 1 day |
| 7 | **maskfill + cv2 NS** | masked data J4 | fill of residual masks; noise-matched variant | 0.5 day |

### Priority 2: components to build on (the "develop with shelf parts" path)

| # | Item | What to do | Effort |
|---|---|---|---|
| 8 | **deepinv operator from our Gaia PSF field** | eigen-PSF + coefficient maps per frame / colour -> `SpaceVaryingBlur`; per-tile PSFs -> `TiledSpaceVaryingBlur`; RGGB `Inpainting` mask; cached FFTs; validate with `adjointness_test`, PyLops `NonStationaryConvolve2D` and DeepLens `splat_psf_per_pixel` on crops | 2-3 days |
| 9 | **deepinv single-image non-blind PnP on the stack** | Poisson-Gaussian fidelity (variance-weighted L2 or shifted Poisson) + our N2N as `PnP` / `DPIR` / HQS prior; tiled denoiser; also RAM / DRUNet as off-the-shelf priors for comparison | 2-3 days |
| 10 | **Joint multi-frame solver** | ImageMM-style MM or ordered subsets on A_k = M_RGGB o Warp_k o H_k, streaming frames from NVMe memmaps, N2N PnP steps; finite target PSF (STARRED idea) | 1-2 weeks |
| 11 | **Denoising baselines and loss terms** | BM3D with noise PSD, CAREamics N2V2 / structN2V, ZS-N2N; deepinv `SurePGLoss` / `R2RLoss` added to our N2N training | 2-3 days (baselines) + 2-3 days (losses) |
| 12 | **Colour-dependent PSF design** | Piff/DES-Y6 style polynomial in (x, y, lambda_eff) per Bayer channel, lambda_eff from Gaia XP x channel response (GaiaXPy already in the venv); optional small physical pupil model for the H-alpha extrapolation | part of J1 (1-2 weeks total with item 5) |
| 13 | **Bright-star wings** | port elderflower functional form (Moffat core + multi-power-law aureole) and Gnuastro ring-normalised stacking per channel, Gaia fluxes, masked cores | 2-4 days |

### Priority 3: larger or optional efforts (decide after J1 / J2 results)

| # | Item | Why / when | Effort |
|---|---|---|---|
| 14 | **ASTERIS retrained** on our sub-stacks (per CFA plane) | multi-frame self-supervised denoising SOTA; compare with half-stack N2N; borrow its loss and bright/faint split even if not retrained | 1-2 weeks |
| 15 | **Own linear-domain star remover** (nox MIT template or compact-source-removal partial-conv U-Net) trained with our measured PSFs and noise | only if Gaia subtraction + StarNet2 gating fails J4 | 1-2 weeks |
| 16 | **Piff 1.6 in WSL2** | independent colour-PSF reference | 2-3 days |
| 17 | **STARRED / AstroPhot crop benches** | independent joint-fit references on 64-256 px crops | 1-2 days each |
| 18 | **Closed-weight AI baselines**: Cosmic Clarity AI4 (inside SASpro only), SyQon Zenith / Prism mini / Parallax Nano, DeepSNR, Aberration Remover (Iris) | completeness of the "free shelf" comparison; scoring only, never components | 0.5 day each |
| 19 | Gnuastro / elderflower originals in WSL, DeepSkyStacker, Siril VeraLux Nox / AutoBGE | optional cross-checks | 0.5-2 days each |

### Skipped (with reason, details in sections 1-9)

CUQIpy, ODL, SCICO, Pyxu, ProxImaL, pylops-gpu, SigPy, MIRTorch, NIFTy (no space-varying operator, toy scale, JAX
without Windows GPU, or obsolete); scarlet / scarlet2 / Tractor (galaxy deblending or catalogue photometry); AstroClearNet,
ImageMM-as-code, Firedec, PI-AstroDeconv, Noise2Astro, AstroSURE, Poisson2Gaussian (no code); torchmfbd as a dependency
(blind only, licence inconsistent); Handheld MFSR and burst-SR networks (daylight bursts, not flux-conserving);
StarNet v1, starrem2k13, Cosmic Clarity Dark Star as component (superseded, toy, or licence); PSFEx, psfr, psfmachine;
LaMa, astroDDPM, biharmonic, `interpolate_replace_nans`; Siril PCC, ASTAP, NoiseChisel; original N2N / N2V / N2S repos
(use deepinv / CAREamics).

---

## 11. Licence consequences for a public GitHub release

- deepinv (BSD-3), PyLops / PyProximal (LGPL-3: keep replaceable), photutils, GaiaXPy, STScI drizzle (BSD): no
  constraint beyond attribution.
- Copying code from SASpro, GraXpert, Siril scripts, STARRED, AstroPhot or elderflower-derived GPL files makes the
  project GPL-3 (acceptable per PLAN; to be decided once).
- GraXpert and Aberration Remover weights are CC BY-NC-SA 4.0: non-commercial, ShareAlike on any derivative model.
- StarNet2 licence: outputs may not be used to train networks for commercial software; full-resolution inputs that
  demonstrate its performance may not be shared. Keep StarNet2 outputs out of any training set we publish, and publish
  bench comparisons as crops / metrics.
- Cosmic Clarity AI4: SASpro-only; no extraction, distillation or fine-tuning. Bench scoring only.
- SyQon and DeepSNR terms: unverified (SyQon licence PDF unreachable on 2026-10-10).

## 12. Items still unverified

GraXpert remote model version list (needs its credentials); BM3D package details; SyQon Parallax / Prism / Zenith
licence terms; DeepSNR licence; ASTERIS on native Windows; deepinv `distributed` on Windows multi-GPU; RAM weights
licence; whether Siril propagates per-pixel drizzle weights into the stack; PSFEx acceptance of non-SExtractor colour
columns; GaiaXPcorrection validity versus the reddening of the Heart field.
