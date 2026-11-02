# astro_GPU - survey of free existing tools (Step 1: test bench), 2026-10-10

Survey of free, existing software for each processing step (stacking, stars, noise, sharpening, background, colour),
with a verdict per tool: test it as-is against our own code, reuse its code or ideas, or skip it.

**Bottom line.** Test as-is first: Siril 1.4.4 run from scripts (stacking, sharpening, background, noise, SPCC colour
with a custom A7S-mod sensor file); StarNet2 2.6.2 `--linear` and nox (stars); GraXpert 3.1.0rc2 (background, noise,
sharpening); SASpro MFDeconv (sharpening from many frames, the closest free tool to our goal); photutils (star shapes);
STScI drizzle (check of our stacking); maskfill (masked pixels). Build on: deepinv, the main toolbox for our own
sharpening and noise reduction, with ideas from ImageMM, STARRED, ASTERIS, Piff, elderflower and Gnuastro. There is no
free BlurXTerminator equivalent with open weights.

## How to read this document

| Verdict | Meaning |
|---|---|
| **Test as-is** | Run the tool unchanged on our test bench (PLAN.md section 5) and score its output against ours. PLAN.md calls this a "bench baseline". |
| **Reuse** | Take its code or its method into our pipeline; the notes say how and what to change. PLAN.md calls this a "component". |
| **Skip** | Not worth it; the reason is given. |

- Free tools only: open source, or free programs whose trained networks are closed.
- Nothing was installed and no project file was modified for this survey.
- **[changes PLAN.md]** marks a finding that contradicts PLAN.md section 4.
- **unverified** marks anything not confirmed from the source code, licence files, official docs or the paper.
- "Measured locally" means timed on synthetic test images on our PC (CPU, in the project's Python environment, the venv).

How the facts were checked (2026-10-10):
- GitHub stars, licence, last commit and latest release: read from the GitHub pages and their commit and release feeds
  (the GitHub API was rate-limited). GitLab projects: GitLab API. Python packages: PyPI (the Python package repository).
- What each tool does, its inputs and the licence of its trained networks: read in the source code, LICENSE files,
  READMEs, official docs and paper pages (arXiv abstracts, availability statements).

Our data, as used for the verdicts (from PLAN.md): star size (FWHM) on single frames is 3.81-4.33 px for the Heart and
4.30-5.03 px for the Iris (10th to 90th percentile of frames); Sony A7S frames are 12 MP (about 4240 x 2830 px); stacks
may be drizzled to larger sizes (up to about 24 MP); two half-stacks are available.

## Words used in this document

| Term | Meaning |
|---|---|
| PSF, star shape | The shape a point-like star takes on the image. |
| FWHM | Star size: width of the star profile at half its peak brightness. |
| Deconvolution, sharpening | Undoing the blur caused by the optics and the atmosphere, using the PSF. Richardson-Lucy (RL) is the classic method (the one in Siril). |
| Non-blind / blind | Non-blind: the blur is measured on the stars and given to the tool. Blind: the tool guesses the blur itself. |
| Varying across the field ("space-varying") | The star shape changes across the image (for example coma in the corners). The opposite, "one shape for the whole image", is called space-invariant or stationary. |
| Many-frame ("multi-frame") | Working on all the individual registered frames together, instead of on the finished stack. |
| Super-resolution (SR) | Rebuilding a finer pixel grid from many slightly shifted frames, as drizzle does. |
| Linear data | Unstretched data, as it comes out of stacking. |
| Bayer, CFA, RGGB | The colour filter pattern on the sensor (CFA = colour filter array). RGGB or BGGR = order of the colour pixels; ours is RGGB. |
| Gaia, Gaia XP | Star catalogue with precise positions and brightnesses; XP = its low-resolution spectrum of each star. |
| Forced photometry | Measuring a star's brightness at its known (Gaia) position. |
| Camera noise ("Poisson-Gaussian") | Photon noise, which grows with the signal, plus constant read noise. "Poisson" alone = photon noise only; "Gaussian" alone = constant noise only. |
| Correlated noise | Noise shared between neighbouring pixels (created by drizzle, registration and debayering). Many denoisers assume there is none. |
| Noise2Noise (N2N) | Our noise-reduction network, trained only on our own images, by comparing two half-stacks whose noise is independent. |
| Learning without clean examples ("self-supervised") | Training a network from noisy images only, with no clean target image. |
| Plug-and-play (PnP) | A sharpening loop that alternates "match the data" and "clean with a denoiser". Any noise-reduction network (ours included) can supply the idea of what a clean image looks like (the "prior"). |
| Fit term ("likelihood", "data fidelity") | The part of a solver that measures how well a candidate image matches the real data, given the noise. |
| Model of the frame ("forward model") | A simulation of how the true sky becomes a recorded frame: blur, alignment, Bayer sampling, noise. |
| Reverse operation ("adjoint") | A mathematical counterpart of the blur that solvers need; a blur model is only usable if it has one. |
| Weights | The trained parameters of a neural network. "Closed weights": the program is free but the trained network is not open. |
| PyTorch, JAX | Python deep-learning libraries. PyTorch uses NVIDIA GPUs on Windows; JAX cannot (CPU only, or WSL2). |
| ONNX Runtime, DirectML, CUDA | ONNX Runtime runs pretrained networks; DirectML = any GPU on Windows; CUDA = NVIDIA GPUs. |
| WSL2 | Windows Subsystem for Linux: runs Linux programs inside Windows. |
| CLI, headless | Run from the command line, without the graphical interface. |
| Licences | MIT, BSD, Apache: free use, credit required. LGPL-3: usable as a separate library that stays replaceable. GPL-3: copying its code makes our project GPL-3 too. CC BY-NC-SA 4.0: non-commercial use only, and any model derived from it must keep that licence. CC-BY-4.0: free use with credit. |

---

## 0. Key findings (read this first)

1. **deepinv is confirmed as the main toolbox** (BSD-3, PyTorch, Windows + NVIDIA GPU). It already has exactly the
   pieces we need:
   - `SpaceVaryingBlur`: a blur whose star shape changes across the image, described as a few basic star shapes plus
     maps saying how much of each applies where. Both are plain arrays, so our own measured star shapes plug in.
   - `TiledSpaceVaryingBlur`: one measured star shape per overlapping tile.
   - `StackedLinearPhysics`: handles many frames together.
   - Plug-and-play solvers (`PnP`, `RED`, `DPIR`) that accept any denoiser with a `forward(x, sigma)` function, i.e. our
     Noise2Noise network.
   - Several training recipes that learn without clean examples (listed in section 3).

   Missing pieces we must write: a fit term for camera noise; the RGGB Bayer pattern (its `Demosaicing` assumes BGGR);
   frame alignment inside the model; reading hundreds of frames from disk one after another; a tool that builds the map
   of star shapes from Gaia stars.
2. **torchmfbd cannot be given a measured star shape through its normal interface.** It always re-estimates the shape
   itself; its option `infer_modes=False` removes the star shape from the model instead of fixing it. Using a measured
   shape (non-blind) would need a hack in its source (`lr_modes = 0` plus injected coefficients). Its licence is
   inconsistent (LICENSE file: MIT; pyproject: GPLv3). **[changes PLAN.md]**: use its ideas only, not the library.
3. **New many-frame tool to compare against: SASpro MFDeconv** (GPL-3, PyTorch, GPU, Windows). It runs Richardson-Lucy
   on all registered frames together, with a star shape measured on each frame (one shape for the whole frame), robust
   weighting (Huber: pixels that disagree strongly count less), noise-based weighting, and optional super-resolution. It
   is the only free, runnable many-frame deconvolution for astro images found. **[changes PLAN.md]**
4. **ImageMM and AstroClearNet: papers only, no public code.** ImageMM's method (majorisation-minimisation: an
   iterative scheme where every step is guaranteed to improve the fit) is the template for our joint many-frame solver.
5. **ASTERIS code and trained networks are public** (code MIT, weights CC-BY-4.0 on Zenodo, a public research data
   archive). The pretrained networks are for JWST and Subaru data and do not transfer to ours. ASTERIS is the
   many-frame generalisation of our half-stack Noise2Noise. **[changes PLAN.md]**
6. **StarNet2 2.6 (2026-09) adds `--linear`** (works on unstretched data); our local install is 2.5.4. Its licence
   forbids using its outputs to train networks for commercial software, and sharing full-resolution input images that
   demonstrate its performance.
7. **GraXpert**: deconvolution exists only in the 3.1.0 release candidates (test versions); its trained networks are
   **CC BY-NC-SA 4.0**; the main branch has been idle since 2025-01. Its AI background model works on a 240 x 240
   shrunk copy of the whole image.
8. **No free BlurXTerminator equivalent with open weights exists.** Cosmic Clarity's networks are licensed for use
   inside SASpro only.
9. **Windows limits**: JAX cannot use an NVIDIA GPU on native Windows (so SCICO, scarlet2, STARRED and J-UBIK run on
   CPU or in WSL2); Piff, PSFEx, Gnuastro and GalSim need WSL2.
10. **Siril's SPCC sensor database has no Sony A7S**, and its DSLR curves include the stock IR-cut filter: a custom
    A7S-mod sensor file (JSON) is needed for a fair SPCC comparison.

---

## 1. Toolboxes to build our own sharpening (measured star shape, varying across the field, many frames)

These are libraries for writing our own deconvolution where the star shape is measured on the stars (non-blind),
changes across the field, and all frames are used together.

| Candidate | Licence | Activity (checked) | GitHub stars | Runs on | What it offers | Verdict |
|---|---|---|---|---|---|---|
| **deepinv** | BSD-3 | v0.4.2 2026-08-30; commit 2026-10-10; published in the Journal of Open Source Software 2025 | 827 | PyTorch, GPU, Windows OK (pure Python) | star shape varying across the field (`SpaceVaryingBlur`, `TiledSpaceVaryingBlur`); many frames (`StackedLinearPhysics`); photon-noise fit term, but none for camera noise | **Reuse (main toolbox)** |
| **SASpro MFDeconv** | GPL-3.0 | V1.22.5 2026-10-10 | 112 | PyTorch on CUDA / DirectML / CPU, Windows | one star shape per frame (median of detected stars); Richardson-Lucy on many frames; robust (Huber) + noise weights | **Test as-is** (the many-frame reference to beat) |
| **PyLops** `NonStationaryConvolve2D` (+ `TorchOperator`) | LGPL-3.0 | v2.8.0 2026-07-25; commit 2026-10-08 | 540 | NumPy / CuPy / numba-cuda (Python GPU libraries); Windows feasible (untested) | exact blur from a regular grid of star shapes, blended between grid points; reverse operation included | **Reuse** (exact reference to check our blur model) |
| **DeepLens** `splat_psf_per_pixel` | Apache-2.0 | v2.5.5 2026-09-30 | 740 | PyTorch | exact blur with one star shape per pixel (no reverse operation) | **Reuse** ("ground truth" on crops) |
| **ImageMM** (Sukurdeep et al., AJ 170, 233, 2025; arXiv 2501.03002) | no public code (TensorFlow according to the paper) | paper v3 2025-09-12 | - | - | joint restoration of many frames + super-resolution; star shape of each exposure refined during the fit; robust (Huber) weights; no negative values; star shape constant across a frame | **Reuse** (algorithm to re-implement) |
| **STARRED** (+ Firedec) | GPL-3.0+ | 1.7.1 2026-08-13 | 6 (GitLab) | JAX; GPU only through WSL2 (experimental) | joint fit of many epochs: point sources + pixel background, one star shape per epoch, sharpens to a small target star size | **Test on crops** (optional) |
| **AstroPhot** | GPL-3.0 | v0.18.0 2026-06-25 | 106 | PyTorch (JAX optional via caskade), Windows | joint fits over several images, one star shape per image, photon-noise fit term in its least-squares (Levenberg-Marquardt) fitter | **Test on crops, for validation** (optional) |
| **torchmfbd** | MIT (LICENSE) vs GPLv3 (pyproject): inconsistent | PyPI 0.9.2 2025-08-18; commit 2026-08-11 | 11 | PyTorch; declared for Unix/macOS only | blind (a solar-astronomy method, MOMFBD, that guesses the blur from many short frames); star shape always re-estimated; variation across the field through a grid of coefficients + soft-edged patches; Gaussian noise fit | **Ideas only** (key finding 2) |
| PyProximal | LGPL-3.0 | v0.13.0 2026-10-03 | 85 | NumPy / CuPy | `PlugAndPlay` (plug-and-play solver), `ConsensusADMM` (a solver with one term per frame); no photon-noise step | optional alternative solver |
| J-UBIK `linpatch_convolve` | BSD-2 | v0.4 2026-08-13 | 8 | JAX, double precision | blur by overlapping tiles with an exact reverse operation | Skip (same idea as the deepinv tiled blur) |
| rdmpy (Ring Deconvolution Microscopy, Nature Methods 2025) | MIT | commit 2025-10-07 | 60 | PyTorch | star shape that varies only with distance from the centre, fitted with classic optical aberrations (Seidel) | Study only (model for the Newton coma); `pip install rdmpy` installs an unrelated package |
| CUQIpy (+ CUQIpy-PyTorch) | Apache-2.0 | v1.5.1 2026-04-07 | 53 | NumPy (CPU) | one star shape only; error bars by random sampling (MCMC) | **Skip** (toy scale) |
| ODL 1.0 | MPL-2.0 | 1.0.0 2026-03-11 | 434 | works with several array libraries, including PyTorch | no blur varying across the field, no plug-and-play | **Skip** |
| SCICO | BSD-3 | v0.0.7 2025-12-09 | 174 | JAX, "not supported on Windows" | one star shape only; plug-and-play with BM3D and DnCNN denoisers | **Skip** (its examples are useful references) |
| Pyxu | MIT | v2.0.3 2025-03-26 (dormant) | 142 | NumPy | one star shape only | **Skip** |
| ProxImaL | MIT | v0.2.0 2024-10-22 | 122 | Halide (an image-processing language), requires numpy 1.26 | blur + alignment + subsampling + photon noise (in concept our model) | **Skip** (obsolete fixed versions) |
| pylops-gpu | LGPL-3.0 | last commit 2022-05-23, README says "unmaintained" | 12 | | | **Skip** (use PyLops with CuPy instead) |
| scarlet2 / scarlet v1 / the Tractor | MIT / MIT / GPL-2.0 | 2026-09-29 / 2024-08 / dr11.1 2026-04-22 | 32 / - / 94 | JAX / numpy / C + Python (Linux) | separating overlapping galaxies, or catalogue photometry | **Skip** |
| AstroClearNet (arXiv 2504.06463) | no code | paper 2025 | - | - | trains a network from scratch on each scene to merge frames ("deep image prior"), star shapes known | **Skip** (no code, does not scale) |
| MultiWienerNet, UABC, DegradationTransfer, DMBSR, pycudadecon, RedLionfish, sdeconv, DeconvolutionLab2, CIL, SigPy, SPORCO, MIRTorch, NIFTy | various | | | | one star shape only, no licence, no weights, or another field of science | **Skip** (some are useful algorithm references) |

Notes.

- **deepinv** (https://github.com/deepinv/deepinv; source files `deepinv/physics/blur.py` and
  `physics/functional/product_convolution.py`; function signatures checked in the source on 2026-10-10).
  - `SpaceVaryingBlur(filters (B,C,K,h,w), multipliers (B,C,K,H,W), padding, use_fft, mask_first=True)`. The blurred
    image is the sum of K blurs, each with one basic star shape (`filters`) applied to the image weighted by its map
    (`multipliers`): y = sum_k h_k * (w_k . x). With `mask_first=True` the star shape is tied to where the light comes
    from, which is right for star shapes measured at star positions. `mask_first=False` (tied to where the light lands)
    was added in v0.4.2. The reverse operation is provided.
  - Implementation: a Python loop of K full-frame blurs; the Fourier transforms of the star shapes are recomputed at
    every call. Memory (single precision, maps, 3 colours, K = 10): about 1.4 GB at 12 MP, 2.9 GB at 24 MP; fits in
    16 GB.
  - `ProductConvolutionBlurGenerator` only wraps a generator of synthetic star shapes. Its recipe (split the grid of
    measured star shapes into a few basic shapes plus smooth maps, using a standard matrix decomposition, SVD, and
    thin-plate splines) is about 20 lines to copy. But computing the smooth maps at full resolution is too heavy at
    12-24 MP: compute them on a coarse grid and enlarge.
  - `TiledSpaceVaryingBlur(filters, patch_size, stride=patch//2, blending_mode="bump"|"linear", use_fft=True)` (v0.4.0):
    one star shape per overlapping tile; only `valid` edge handling. Caution: its documentation gives the "where the
    light comes from" formula, but the code blends on the output ("where it lands"). Check with `adjointness_test` and
    against PyLops on crops.
  - `LinearPhysics(A, A_adjoint=None)` builds the reverse operation automatically (`torch.func.vjp`); also
    `adjointness_test()`, `compute_norm()`, and chaining with `*`. `Demosaicing(pattern="bayer")` puts R at (1,1) and B
    at (0,0), i.e. **BGGR** (checked in `physics/inpainting.py`): for our RGGB use `Inpainting(mask=custom)`. No frame
    alignment model (one based on `grid_sample` is short to write).
  - Many frames: `StackedLinearPhysics`, `StackedPhysicsDataFidelity`, `StackedPhysicsLoss`; `deepinv.distributed`
    (beta) can split the work and run a denoiser tile by tile. All of it assumes every frame is in memory
    (721 x 12 MP x 4 bytes = 35 GB): hundreds of frames need our own loop reading frames from disk (in groups, or in
    random passes).
  - Noise and fit terms: `PoissonGaussianNoise` (simulation only); fit terms `L2`, `PoissonLikelihood(gain, bkg)`,
    `LogPoissonLikelihood`, `L1`. **No fit term for camera noise**: write a noise-weighted least squares, or use the
    "shifted Poisson" trick (`PoissonLikelihood(bkg = sigma^2 / g)` applied to y + sigma^2 / g).
  - Solvers: about ten standard ones, including HQS (a common plug-and-play solver), MLEM (= Richardson-Lucy) and OSEM
    (Richardson-Lucy on subsets of the data). `PnP(denoiser)`, `RED` and `DPIR(denoiser=...)` (plug-and-play variants)
    accept any network with `forward(x, sigma)`. `OSEM` accepts frames as subsets but is documented for tomography only
    (untested on blur).
  - Pretrained `RAM` network (handles camera noise; trained on everyday photos; weights on Hugging Face, licence not
    stated): could be a no-training baseline combined with our blur model.
  - Evidence: only one astro example (radio interferometry); no test of these blur models on optical astro images.
    Effort: blur model from our Gaia star-shape map 1-2 days; plug-and-play on the single stack 2-3 days; joint
    many-frame solver 1-2 weeks.
- **torchmfbd** (https://github.com/aasensio/torchmfbd, arXiv 2505.10639). Checked in the source:
  `Deconvolution.deconvolve(..., modes_in=...)` uses `modes_in` only as a starting value, then re-optimises it
  (`modes.requires_grad_(True)`, learning rate `lr_modes`, default 0.01). `DeconvolutionSV.deconvolve(...,
  infer_modes=False)` removes the star shape from the model instead of fixing it. The star shape is described either
  from optical aberrations (Zernike / Karhunen-Loeve wavefront modes) or from statistical shape bases (NMF / PCA) loaded
  from a file. Square patches, mono, Gaussian noise fit, no Bayer, no photon noise. Worth copying: the fast Fourier-space
  computation over the shape components, the per-frame distortion fields inside the joint model, and the wavelet-based
  clean-up term (IUWT, the "a trous" wavelets used in astro software).
- **PyLops `NonStationaryConvolve2D(dims, hs, ihx, ihz, engine="cuda", dtype="float32")`**: a regular grid of odd-sized
  star shapes; each pixel blends the 4 nearest shapes. Computed directly, not with Fourier transforms (cost of a direct
  convolution); GPU code through numba. Use single precision (float32): double precision (float64) is very slow on
  GeForce cards. Role: exact reference to measure the error of our fast approximate blur (basic shapes + maps, or
  tiles) on crops.
- **ImageMM**: models each frame with a star shape measured on the stars and refined during the fit; multiplicative
  majorisation-minimisation updates (key finding 4); about 0.12 s per iteration at 1000 x 1000 px on a V100 GPU (33
  frames from Subaru's HSC camera). Re-implement in PyTorch on deepinv, adding: star shape varying across the field,
  RGGB sampling, alignment inside the model, camera-noise weights and Noise2Noise denoising steps.
- **STARRED** (https://gitlab.com/cosmograil/starred, Michalewicz et al. 2023, Journal of Open Source Software; Millon
  et al. 2024 AJ 168, 55): sharpens towards a small "target" star (about 2 px FWHM) instead of a single-pixel point,
  which avoids ringing. An idea to adopt for our own sharpening target. Works on small cut-outs (32 x 32 -> 64 x 64 px).
- **AstroPhot** (https://github.com/Autostronomy/AstroPhot): still PyTorch. Its fitter's memory use rules out a
  full-frame nebula modelled pixel by pixel, but a crop with a pixel-by-pixel nebula + Gaia point sources + one star
  shape per frame works: an independent check of our model of the frame and of our photometry.
- None of these tools provides: a camera-noise fit term combined with a blur varying across the field; Bayer RGGB
  sampling of non-debayered frames inside the model; reading hundreds of 12-24 MP frames from disk in turn; a builder
  of star-shape maps from Gaia stars per frame and per colour. All are small additions on top of deepinv.

---

## 2. Sharpening (deconvolution) tools to test on a finished stack

| Candidate | Licence (code / weights) | Activity (checked) | GitHub stars | Runs on | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **Siril 1.4.4** RL / Wiener / split Bregman (+ `makepsf`) | GPL-3.0+ / none | 1.4.4 released 2026-06-16 (1.4.0 2025-12-04) | 194 (GitLab) | C/C++, CPU only (multi-threaded, FFTW library; no GPU code in `src/filters/deconvolution`), native Windows, `siril-cli` without interface | linear (RL "gradient descent" mode) or stretched (multiplicative RL); mono or RGB; one star shape for the whole image, can load ours | **Test as-is** (classic reference) |
| **GraXpert 3.1.0rc2** `deconv-obj` / `deconv-stellar` | GPL-3.0 / **CC BY-NC-SA 4.0** | main 2025-01-01; develop 2025-10-03; rc3 2025-10-03 (macOS-only fix); last active branch 2026-01-12 | 325 | ONNX Runtime: DirectML on Windows, CUDA on Linux | accepts linear data (each tile normalised in log scale); each colour channel separately; 512 px tiles | **Test as-is** |
| **Seti Astro Suite Pro (SASpro) MFDeconv** | GPL-3.0 / no network (pure algorithm) | v1.22.5 2026-10-10 (very active) | 112 | PyTorch CUDA / DirectML / CPU, Windows | registered frames; star shape measured per frame on the stars; lightness (luma) or per channel; optional super-resolution x2 / x3 | **Test as-is + code reference** (closest free tool to our goal) |
| Cosmic Clarity AI4 sharpen (inside SASpro) | app GPL-3.0 / **proprietary, SASpro only** | 2026-10 | - | NAFNet (a published restoration network design), PyTorch/ONNX, CUDA/DirectML | auto-stretches if median < 0.08, then un-stretches | **Test as-is, inside SASpro only**; never reused |
| Aberration Remover 2.0.1 (R. Alberghi) | Siril script GPL-3 / weights CC BY-NC-SA 4.0 | model 2.0.1 2026-02-11 | 5 | ONNX through Siril `pyscript`, GPU | linear data, after background and colour calibration, before deconvolution | **Test as-is for the Iris (Newton coma)** |
| SyQon Parallax Nano | Siril script MIT / closed weights, terms PDF unverified | unverified | - | unverified | unverified | optional test (login required) |
| BM3D deblurring (`pip bm3d` 4.0.3) | "free for non-commercial use" | unverified | - | CPU | known star shape, constant (Gaussian) noise | optional classic test |
| AstroSharp, Galaxy-Deconv, Astraios, PI-AstroDeconv, AstroClearNet | see notes | | | | | **Skip** |

Notes.

- **Siril deconvolution** (docs https://siril.readthedocs.io/en/stable/processing/deconvolution.html, source
  https://gitlab.com/free-astro/siril).
  - Richardson-Lucy in two flavours: multiplicative (for stretched data) and gradient descent (default, recommended
    for linear data). Anti-noise regularisation: total variation (TV) or the Frobenius norm of the Hessian (a
    smoothness measure). Wiener (assumes constant noise; weak on deep sky). Split Bregman TV (for starless or lunar
    images).
  - Star shape guessed automatically (blind): l0 descent (Anger et al., IPOL) or spectral irregularities. Star shape
    from stars: average Gaussian/Moffat fit of selected or auto-detected stars (linear data only). PSF file: square, odd
    size; mono (then only the L of LAB is deconvolved) or 3 channels (one PSF per channel).
  - Commands (checked in the command reference): `makepsf {load f | stars [-sym] | blind ... | manual ...}`,
    `rl [-loadpsf=] [-alpha=] [-iters=] [-tv|-fh] [-mul]`, `sb`, `wiener`, and sequence versions `seqrl`, `seqsb`,
    `seqwiener`. For sequences the docs say "the PSF will be calculated only for the first image" and re-used for all
    frames, so a PSF per frame needs one script call per frame.
  - Python: `sirilpy` drives a *running* Siril (`cmd()`, `get/set_image_pixeldata`); it ships with Siril and is not on
    PyPI.
  - Limits: one star shape for the whole image; CPU only.
  - How to plug in: `siril-cli -s script.ssf` with `makepsf load our_psf.fits` then `rl`; and `makepsf stars` (Siril's
    own PSF) as a second baseline. Effort 0.5-1 day.
- **GraXpert deconvolution** (https://github.com/Steffenhir/GraXpert).
  - Versions: object model added in 3.1.0rc1 (2024-11-10), stellar model in 3.1.0rc2 (2025-01-01). The last Windows
    installer with deconvolution is 3.1.0rc2 (`graxpert-windows-amd64.msi`); the "latest" stable version is still 3.0.2
    (no deconvolution).
  - Network licence: CC BY-NC-SA 4.0 (`licenses/Deconvolution-Model-LICENSE.html`, develop branch). The networks are
    `model.onnx` zip files downloaded from online storage (MinIO/S3) with credentials added when the official program is
    built: a plain copy of the source code cannot download them; the official program, or the PyPI test build 3.2.0a3
    from the fork geeksville/GraXpert, can.
  - CLI: `GraXpert-win64.exe img.fits -cli -cmd deconv-obj|deconv-stellar -strength 0..1 -psfsize <FWHM px> -batch_size 4 -gpu true -ai_version x.y.z -output name`.
  - How it works (from `deconvolution.py`): 512 px tiles moved in steps of 448 px (overlapping); each channel processed
    on its own; values put in log scale then normalised (minus mean, divided by standard deviation); the network
    predicts the correction to apply, in log scale; output clipped to [0, 1]. **psfsize is clamped** (computed from the
    code): stellar model effective FWHM about 3.9-10.2 px, object model about 1.8-13.5 px (v1.0.1). The object and
    stellar models are designed to be run one after the other.
  - No published evaluation (forum comparisons only). Effort 0.5 day (CLI) or 1 day (calling `model.onnx` with
    onnxruntime-gpu from our test code, copying the tiling).
- **SASpro MFDeconv** (https://github.com/setiastro/setiastrosuitepro, file `src/setiastro/saspro/mfdeconv.py`, function
  `multiframe_deconv_normal_rebuild(paths, out_path, iters, kappa, color_mode="luma"|"PerChannel", seed_mode, rho="huber", use_star_masks, use_variance_maps, super_res_factor)`;
  module structure checked).
  - How it works: multiplicative Richardson-Lucy on registered frames. Star shape measured per frame on the stars
    (`compute_psf_kernel_for_image`, based on SEP, the Python version of Source Extractor; Gaussian if that fails).
    Weights = robust (Huber) x inverse noise maps x star masks. Each update is limited to [1/kappa, kappa]. Optional
    super-resolution with a finer-sampled PSF. Frames are read from disk as needed (memory-mapped files).
  - Limits: one star shape per frame (the same over the whole frame); works on registered, already interpolated frames;
    no Bayer model; no "clean image" constraint other than the update limit and stopping early. The module loads the
    graphical-interface library PyQt6, so a small wrapper is needed to run it without the interface.
  - No published evaluation. Use: feed it 20-40 registered sub-stacks (not 721 single frames) of the same batch and
    compare with our joint deconvolution; read the code as a reference implementation. Effort 1-2 days.
- **Cosmic Clarity AI4** (in SASpro). Licence file `COSMIC_CLARITY_LICENSE.txt` (SASpro release files): personal,
  non-commercial use **only inside SASpro**; no use in another pipeline; no extracting the networks, no training a copy
  from them (distillation), no retraining (fine-tuning). The old standalone repository setiastro/cosmicclarity (MIT
  code, archived 2026-09-16) still contains old `.pth` network files. Siril's `CosmicClarity_Native.py` runs AI4 models
  outside SASpro, which seems to contradict that licence: do not use that path. Test only through SASpro (interface or
  its headless entry point). Effort 0.5 day.
- **Aberration Remover** (https://github.com/riccardoalberghi/abberation_models): corrects defocus, astigmatism, coma,
  trefoil and spherical aberration (not seeing, not lateral chromatic aberration) on linear data. Relevant to the
  Newton 130/900 coma. Effort 0.5 day through Siril.
- **Skipped.** AstroSharp (deepskydetail/AstroSharp, MIT; a web app written in R, works on stretched TIFF; sharpening,
  not deconvolution); Galaxy-Deconv (Lukeli0425; small galaxy cut-outs from weak-lensing simulations); Astraios (alpha,
  2 GitHub stars); PI-AstroDeconv (arXiv 2403.01692) and AstroClearNet (arXiv 2504.06463): no code found (section 1).
- No free tool trained like BlurXTerminator (star shape varying across the field, network told the measured star
  shape, works on linear data) with weights that may be redistributed was found. The closest free ones are GraXpert
  (CC BY-NC-SA weights) and Cosmic Clarity (SASpro only).

---

## 3. Noise reduction

| Candidate | Licence (code / weights) | Activity (checked) | GitHub stars | Runs on | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **ASTERIS** (Science 2026, arXiv 2602.17205) | **MIT / CC-BY-4.0 (Zenodo)** | last commit 2026-06-23 | 55 | PyTorch 2.5, CUDA 12.1, tested on Ubuntu Linux; Windows unverified | 4 or 8 aligned exposures (stacked as a 3D block), mono, linear | **Reuse ideas + test after retraining on our data** |
| **GraXpert denoise** | GPL-3.0 / CC BY-NC-SA 4.0 | since 3.0.0 (2024-04-22) | 325 | ONNX Runtime, DirectML/CUDA | linear; 3 channels (mono is copied to 3) | **Test as-is** |
| **Siril `denoise`** (NL-Bayes, `-vst`, `-da3d`, `-sos`) | GPL-3.0+ | 1.4.4 | 194 | C, CPU, Windows | linear (with VST) or stretched | **Test as-is** (classic level to beat) |
| **BM3D with noise spectrum** (`pip bm3d`) | free non-commercial | unverified | - | CPU | Gaussian noise with a **correlated** noise spectrum | **Test as-is** (matches the correlated noise of our stacks) |
| **CAREamics** (N2V, N2V2, structN2V, N2N, CARE, PN2V, HDN, MicroSplit) | BSD-3 | v0.3.4 2026-09-23; commit 2026-10-06 | 154 | PyTorch Lightning, Python >= 3.11, GPU, Windows | single image or pairs, mono or multi-channel | **Test as-is** (N2V2 / structN2V) + reference Noise2Noise code |
| **deepinv training recipes ("losses")** | BSD-3 | v0.4.2 2026-08-30 | 827 | PyTorch | any image model | **Reuse** (extra training terms for our Noise2Noise) |
| ZS-N2N (zero-shot Noise2Noise: trains a tiny network on the single image itself; Mansour & Heckel, CVPR 2023) | Colab notebook only | - | - | PyTorch, tiny network | single image | Test as-is (30-line re-implementation) |
| Cosmic Clarity AI4 denoise, DeepSNR (Misiura), SyQon Prism mini | closed weights; terms: SASpro only / unverified / unverified | 2026 | - | ONNX / PyTorch, Windows | stretch the data internally | test only |
| NVlabs/noise2noise, juglab/n2v, czbiohub/noise2self, Neighbor2Neighbor, Blind2Unblind, R2R, Noise2Score repositories | see notes | 2019-2024 | | | | **Skip** (use the deepinv / CAREamics versions) |
| Noise2Astro, AstroSURE, Poisson2Gaussian, TDR, AstroDenoisePy | see notes | | | | | **Skip** (no code, or same idea as ours) |

Notes.

- **ASTERIS** (paper https://arxiv.org/abs/2602.17205, DOI 10.1126/science.ady9404; code
  https://github.com/freemercury/ASTERIS_THU, also Zenodo 10.5281/zenodo.17115038; weights Zenodo 17114980 and 18358472,
  about 296 MB, models `ASTERIS{4,8}_nrc{short,long}` and `ASTERIS8_SUBARU_ks`; demo data Zenodo 17105027, 2.4 GB). The
  code is public, confirmed by the paper's availability statement.
  - Method: noise reduction learned without clean examples, from a stack of 4 or 8 aligned exposures (it uses both the
    image and the exposure-to-exposure dimension). Training samples of 16 exposures are split into two sets of 8 with
    independent noise. Training loss = squared difference between the co-added input set and the co-added target set,
    plus absolute difference between paired single exposures. Network: a 3D U-Net (the standard image-to-image network
    layout) with Restormer-style 3D attention layers. A 3-sigma threshold across the exposures separates bright pixels
    (median-combined, not processed) from the faint part (normalised, then denoised).
  - Reported results: about 1.0 magnitude deeper detection at 90 % completeness and purity on simulated images (90 % of
    real sources found, 90 % of detections real); star shapes and brightness measurements preserved; about 3x more
    very-high-redshift (z above about 9) candidates in JWST data.
  - Cost: 18 s to process 8 x 1650 x 1650 px on 4 GPUs of 40 GB; training about 26 h.
  - Limits for us: the pretrained networks are for JWST NIRCam / Subaru (do not transfer); mono only; with 16 GB of
    GPU memory, smaller patches are needed; Windows untested (depends on the swanlab package).
  - It is the many-frame generalisation of our half-stack Noise2Noise. Use: (a) borrow its input made of several
    sub-stacks, its "average + frame" loss, its bright/faint split and its completeness/purity metric; (b) retrain it
    on our sub-stacks, per Bayer colour plane, as a comparison. Effort 1-2 weeks (retraining).
- **GraXpert denoise**: CLI `-cmd denoising -strength 0..1 -batch_size -ai_version -gpu`. From the code: 256 px tiles;
  each channel normalised as (x - median) / MAD x 0.04 (MAD = median absolute deviation, a robust noise measure);
  pixels above about median + 250 MAD (model v2 and later) are not denoised; strength = linear blend with the input.
  Evidence: blog comparisons only. Effort 0.5 day.
- **Siril `denoise`**: NL-Bayes (a classic patch-based denoiser), with options: generalised Anscombe VST
  (variance-stabilising transform: makes photon noise behave like constant noise, for linear data), DA3D and SOS
  (extra refinement passes). Effort 0.5 day.
- **BM3D** (a classic patch-based denoiser): accepts a noise power spectrum (PSD: describes how the noise is correlated
  between neighbouring pixels). This matters because Bayer drizzle, registration interpolation and debayering
  correlate the noise of our stacks. Effort 0.5-1 day.
- **CAREamics** (https://github.com/CAREamics/careamics): the maintained PyTorch successor of juglab/n2v (microscopy
  tools). Algorithms checked in the source: N2V (Noise2Void: learns from a single noisy image by hiding each pixel and
  predicting it from its neighbours), N2V2 (improved N2V), structN2V (hides a pattern of neighbouring pixels, to cope
  with correlated noise), N2N, CARE (needs clean targets), and the microscopy variants PN2V, HDN, MicroSplit,
  DenoiSplit. Effort 1 day.
- **deepinv training recipes** (`deepinv/loss`), all usable to learn without clean examples:
  - Neighbor2Neighbor (compares two sub-sampled copies of the same noisy image);
  - R2RLoss (generalised "Recorrupted-to-Recorrupted": makes two noisier copies of the image and learns to map one to
    the other; Gaussian, Poisson or Gamma noise);
  - ScoreLoss (Noise2Score);
  - SURE (an estimate of the denoising error computed from the noisy image alone, using the known noise model;
    Gaussian, Poisson and camera-noise versions);
  - SplittingLoss (hides part of the data and predicts it; with pixel masks it is a Noise2Self-type loss),
    EquivariantSplittingLoss;
  - EI and MC losses (equivariant imaging and measurement consistency: learn sharpening without clean examples, using
    the fact that a shifted or rotated sky must be restored the same way).
  - Not included: Blind2Unblind, ZS-N2N, and no loss named Noise2Noise (`SupLoss` with a noisy target is equivalent).
  - Use: add a camera-noise SURE term or a GR2R (generalised R2R) term to our Noise2Noise training, or fine-tune on the
    full stack without clean examples. Effort 2-3 days.
- **Warning about correlated noise**: single-image methods that learn from neighbouring pixels (N2V, Noise2Self,
  Neighbor2Neighbor, Blind2Unblind, ZS-N2N) assume each pixel's noise is independent of its neighbours. On our drizzled
  stacks they will under-denoise or leave artefacts, unless structN2V masks are used or they run on raw Bayer (CFA)
  data. Half-stack Noise2Noise avoids that, but keeps as "signal" any pattern common to both halves (master calibration
  frames, common registration grid).
- **Skipped**: NVlabs/noise2noise (TensorFlow 1, CC BY-NC 4.0, last commit 2021); juglab/n2v (TensorFlow 2, being
  replaced by CAREamics); czbiohub/noise2self (2019 notebooks); TaoHuang2018/Neighbor2Neighbor (BSD-3, 2021) and the R2R
  / Noise2Score repositories (use deepinv); zejinwang/Blind2Unblind (no licence file); Noise2Astro (arXiv 2209.07071);
  AstroSURE (arXiv 2604.16793, checked: no code, but its detection-based test protocol is worth copying);
  Poisson2Gaussian (arXiv 2606.23098, "code will be available"); TDR (zimugh/Denoising-TDR, notebook, no clear
  licence); AstroDenoisePy (p7ayfu77/astro-csbdeep, BSD-3, TensorFlow 2, CARE trained Noise2Noise-style: same idea as
  ours); Learning2Recorrupt (to watch); SCUNet-based scripts (networks trained on everyday photos, used on stretched
  data).

---

## 4. Star removal

| Candidate | Licence (code / weights) | Activity (checked) | GitHub stars | Runs on | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **StarNet2 CLI 2.6.2** (installed: 2.5.4) | closed, free; licence restricts use (see notes) | 2.6.2 package 2026-09-19 (2.6.0 2026-09-04) | - | ONNX Runtime: DirectML included, CUDA by swapping a DLL; Windows/Linux/macOS | 8/16-bit TIFF/PNG or FITS in [0, 1]; **`--linear` since 2.6.0**; RGB recommended; minimum 512 x 512 | **Test as-is**; in our chain only to clean leftovers, and only where a known star is |
| **nox** (charvey2718) | **MIT code and MIT weights** | `nox.exe` v1.1.0 2025-12-25 | 8 | TensorFlow/Keras; exe; GPU on Windows untested by the author | TIFF 8/16/32-bit float (no FITS); `-l` = temporary stretch for linear data | **Test as-is + template for our own** (trained on synthetic stars) |
| **compact-source-removal** (Madarasz et al. 2025, A&A 696 A37) | MIT | last commit 2025-11-21 | 2 | PyTorch, partial-convolution U-Net | linear FITS (Herschel), masks supplied | **Reuse** (network design + training recipe; retrain) |
| SyQon Zenith (Siril script) / SyQon Studio CLI | script by Siril/SyQon; weights proprietary, licence PDF unreachable (unverified) | 2026 | - | PyTorch NAFNet, DirectML | temporary auto-stretch (in intensity-hue-saturation space) for linear data | test only |
| Cosmic Clarity Dark Star | proprietary networks, SASpro only | 2026 | - | PyTorch | auto-stretch | **Skip** (licence); optional manual comparison |
| StarNet v1 (nekitmm/starnet) | MIT code / CC BY-NC-SA weights | 2022-09-12 | 313 | TensorFlow 1/2 | stretched 8-bit | **Skip** (superseded) |
| starrem2k13 | MIT | v7.0 2025-06-16 | 11 | ONNX, CPU | 8-bit | **Skip** (trained on 3 images) |
| MRF (AstroJacobLi/mrf), elderflower | MIT | 2021 / 2025-06-25 | - / 6 | numpy, GalSim | high-resolution reference image / bright stars | **Reuse ideas** (subtracting bright-star halos and wings, section 7) |
| GraXpert | - | - | - | - | - | no star removal (checked: its CLI commands are only background-extraction, denoising, deconv-obj, deconv-stellar) |

Notes.

- **StarNet2** (https://starnetastro.com/cli-tools/starnet/, command-line reference
  https://starnetastro.com/documentation/starnet/command-line-tool/).
  - Local install `F:\photos\heart_nebula\tools\starnet2_2.5.4\...` (2.5.4, ONNX Runtime engine, no `--linear`). Its
    network file `StarNet2_weights.onnx` (inspected read-only): a U-Net in pix2pix style (a classic image-to-image
    design), about 32.9 million parameters; input 512 x 512 RGB tiles, single-precision values in [0, 1]; output
    clipped to [0, 1].
  - Version 2.6.x adds `--linear` (checked on the official page): per-channel midtones (MTF) stretch, network, inverse
    stretch, restoration of clipped shadows. Negative values accepted; **values above 1 are clipped with a warning, no
    automatic scaling** (scale raw values (ADU) to [0, 1] first). Options: `--shadows-clipping`, `--target-background`,
    `-u` (2x upsampling, for tight stars), `-s` stride, `-m` mask, `-n` unscreen star layer; `-e/--eight` removed in
    2.6.0.
  - Licence (LICENSE.txt of the local install): use "solely for astrophotography image processing"; outputs may not be
    used to create commercial software "including for training neural networks"; no sharing of full-resolution inputs
    that demonstrate its performance (relevant if test comparisons are published).
  - Evidence: only qualitative amateur comparisons. Our own finding: it eats faint diffuse emission.
  - Use: upgrade to 2.6.2; run `--linear` on linear FITS scaled to [0, 1] (test with and without `-u`: our stack FWHM is
    about 4 px or more, so `-u` may not be needed). In our chain, accept its star layer only inside the footprint of
    known Gaia stars (the area covered by their measured star shape). Effort 0.5-1 day.
- **nox** (https://github.com/charvey2718/nox; LICENSE and README checked: "My code and trained weights ... are made
  available under the MIT License"; GitHub's licence detector shows NOASSERTION because of a preamble). Same network
  design as StarNet v1, trained on a **fully synthetic** dataset (`GenerateStars.py`: Moffat/Gaussian stars with random
  FWHM, Poisson noise, gain, MTF stretch, spikes, colour fringing, asymmetry, on photo-denoising backgrounds). The author
  says generalisation "remains to be proved". Use: test `nox.exe -l`. Its real value is as a template to train our own
  PyTorch star remover for linear data, with our measured star shapes (spikes, coma), our noise model and real starless
  backgrounds from our data. Effort 0.5 day (test), 1-2 weeks (own model).
- **compact-source-removal** (https://github.com/xMediMadix/compact-source-removal, arXiv 2502.18345): partial
  convolution U-Net (a network that fills masked holes while ignoring the masked pixels), masks sized by signal-to-noise,
  trained by injecting sources into real linear backgrounds. The paper measures photometric accuracy and how well the
  noise and its texture (power spectrum) are preserved. Filling acts only inside the masks, so the background outside
  the stars is untouched by design. Use: retrain with Gaia star masks on our star-subtracted backgrounds. Effort about
  1 week.
- **Siril integration**: since 1.4, Siril calls StarNet through the Python script `StarNetAstro/StarNet.py`
  (`pyscript StarNet.py --linear --masks descreen`, supports CLI 2.5.0-2.6.2); the old built-in `starnet` command
  remains for old versions.

---

## 5. Sky background extraction

| Candidate | Licence (code / weights) | Activity (checked) | GitHub stars | Runs on | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **GraXpert AI background** | GPL-3.0 / **CC BY-NC-SA 4.0** | stable 3.0.2 (2024-05-03); 3.1.0rc2/rc3; PyPI test build 3.2.0a3 2026-09-22 (fork) | 325 | ONNX Runtime; CLI + usable from Python | linear OK; whole image shrunk to 240 x 240 | **Test as-is**; at most an optional hint (prior) for our own fit |
| **Siril 1.4 `subsky` / `seqsubsky`** (polynomial up to degree 4, or RBF = a flexible smooth surface) | GPL-3.0+ | 1.4.4 | 194 | C, CPU, Windows | linear; `seqsubsky` works per frame on the raw **Bayer (CFA)** colour planes | **Test as-is + idea** (gradient removed per frame, on Bayer data) |
| VeraLux Nox (Siril script) | GPL-3.0 | v1.0.1 (2025) | - | Python in Siril | linear | Test as-is |
| AutoBGE (Siril script, port of AutoDBE) | GPL-3.0 | unverified | - | Python in Siril | linear | optional test |
| photutils `Background2D` 3.0.0, SEP 1.4.1 | BSD-3 / LGPL-3 | 2026-04-17 / 2025-02-18 | 310 / - | CPU, Windows (in our venv) | any | **Reuse** only for the local sky level and noise in forced photometry |
| Gnuastro NoiseChisel 0.24 | GPL-3.0 | 2025-11-26 | - | C, WSL only | | **Skip** (needs sky tiles with no detected objects; a dense Milky Way field leaves none) |

Notes.

- **GraXpert** (`graxpert/background_extraction.py`): the whole image is resized to 240 x 240 (padded to 256); each
  channel normalised as (x - median) / MAD x 0.04 and clipped; the network output is smoothed (sigma = 20 x the
  smoothing setting, then 3) and enlarged back. It can only model structures larger than image width / 240 (about
  18 px for a 4240 px wide frame, more on a drizzled stack). In a field filled with H-alpha (Heart & Soul) it can absorb
  real large-scale emission and IFN (faint galactic dust clouds). CLI:
  `GraXpert-win64.exe img.fits -cli -cmd background-extraction [-correction Subtraction|Division] [-smoothing 0..1] [-bg] [-ai_version] [-gpu true]`.
  Python: `from graxpert.background_extraction import extract_background(...)`. Effort 0.5-1 day.
- **Siril**: `subsky {-rbf | degree} [-samples] [-tolerance] [-smooth] [-existing]`; `seqsubsky` on Bayer frames (the
  docs recommend degree 1 before Bayer drizzle); sample points can be set from Python (`set_bgsamples`). Idea to test in
  our chain (driven by the data, no instrument-specific rule): for each frame, fit a simple low-degree gradient to
  (registered frame - reference stack) on star-masked pixels, where the nebula cancels out. Effort about 1 day.

---

## 6. Stacking and Bayer drizzle / super-resolution

| Candidate | Licence | Activity (checked) | GitHub stars | Runs on | Inputs | Verdict |
|---|---|---|---|---|---|---|
| **Siril 1.4.4 drizzle** (port of the Hubble drizzle code, Bayer drizzle) | GPL-3.0+ | 1.4.0 2025-12-05, 1.4.4 2026-06-16/17 | 194 | C, CPU, Windows; `siril-cli`; sirilpy | raw Bayer (CFA) frames; SIP distortion (the standard FITS description of optical distortion) | **Test as-is** |
| **STScI drizzle 3.0.0** (from Hubble's Space Telescope Science Institute) | BSD-3 (AURA) | release 2026-09-24, commit 2026-09-28 | 59 | C core + Python, CPU | any frames + **explicit pixel map** (where each input pixel lands) + weight map | **Reuse** (to check our GPU Bayer drizzle) |
| Handheld Multi-Frame SR (Jamy-L) | MIT | release 1.4 2026-09-03, commit 2026-09-30 | 214 | numba-cuda + torch; WSL recommended on Windows | raw Bayer burst of about 13-20 frames | **Skip** as a test; possible idea for the merging step |
| DeepSkyStacker 6.2.3 beta | BSD-3 | 2026-10-02 | 1140 | C++, Windows | | optional test (no pixfrac drizzle) |
| SyQon Stacking Studio | closed, account needed | 2026 | - | | Bayer (CFA) drizzle | optional test |
| astroalign, reproject, drizzlepac, burst super-resolution networks | | | | | | **Skip** |

Notes.

- **Siril drizzle** (https://siril.readthedocs.io/en/stable/preprocessing/drizzle.html; the source file
  `src/drizzle/cdrizzlebox.c` carries the AURA copyright, i.e. it is a port of STScI's cdrizzle). Works from a pixel
  map; for Bayer drizzle the colour is chosen per input pixel from the CFA. Kernels: square (keeps total flux), point,
  turbo, gaussian, lanczos2/3 (only at scale = pixfrac = 1). Command:
  `register | seqapplyreg ... -drizzle -pixfrac= -kernel= -scale= -disto=master`; the master flat can be used as pixel
  weight. Frames are drizzled into an **intermediate sequence on disk** and then stacked with rejection (whether the
  per-pixel drizzle weights are carried into the stack is unverified). Disk: about 100 GB for 721 frames at scale 1
  (use C: or E:, not F:). Effort about 1 day plus run time.
- **STScI drizzle** (https://github.com/spacetelescope/drizzle): `Drizzle(kernel=...)`,
  `add_image(data, exptime, pixmap, weight_map, pixfrac, ...)`; outputs `out_img`, `out_wht`, `out_ctx` (image,
  weights, context). Since 2.2.0 it can also add up squared weights, to compute the noise (variance). Bayer drizzle =
  three Drizzle objects, each with a weight map equal to 1 only on that colour's Bayer pixels. Our per-frame polynomial
  maps plug in directly as pixel maps. Use as the CPU reference to check flux, weights, star shape and noise of our GPU
  drizzle on a subset of 50-100 frames. Effort about 1 day.
- **Handheld MFSR** (https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution, a re-implementation of Wronski
  et al. 2019, a method for handheld camera bursts): debayering + super-resolution + noise reduction in one go from a
  raw burst, aligning by block matching and motion estimation per tile, then merging with direction-dependent kernels.
  Built for short bursts of bright scenes held in memory. Motion estimation on noisy 15 s star-field frames, and
  kernels that do not conserve flux, make it a poor fit. At most, test its direction-dependent merging kernel inside
  our own stacking.

---

## 7. Star-shape (PSF) modelling: across the field, by colour, bright-star halos

| Candidate | Licence | Activity (checked) | GitHub stars | Runs on | Colour / halos | Verdict |
|---|---|---|---|---|---|---|
| **photutils 3.0.0** EPSFBuilder, GriddedPSFModel, PSFPhotometry | BSD-3 | 3.0.0 2026-04-17; commit 2026-10-10 | 310 | numpy/scipy, CPU, native Windows (in our venv) | no colour dimension; halo modelled only inside the star cut-out | **Test as-is + reference to validate ours** |
| **Piff 1.6.0** | BSD-like | 2025-11-24 | 69 | numpy/GalSim, CPU; **no Windows packages** (checked on PyPI), so WSL2 | **colour can be a variable of the model** (used in the Dark Energy Survey, DES Y6) | **Test as-is (in WSL2) + design reference** |
| **Gnuastro 0.24 `astscript-psf-*`** | GPL-3.0+ | 2025-11-26 | - | C/shell, WSL only | extended star shape out to arcminutes; no variation across the field or with colour | **Reuse (re-implement the recipe)**, optional test in WSL |
| **elderflower** | MIT | commit 2025-06-25 | 6 | numpy, dynesty, needs SExtractor (WSL in practice) | Moffat core + halo made of several power laws ("multi-power-law aureole") | **Reuse (port the profile formula)** |
| PSFEx 3.24.2 | GPL-3.0 | release 2023-04-13; commit 2025-03-27 | 44 | C, no Windows build | shape varies as a polynomial of any catalogue column (colour unverified) | **Skip** (Piff / photutils cover it) |
| STARRED 1.7.1 | GPL-3.0+ | 2026-08-13 | 6 (GitLab) | JAX; no GPU on native Windows | Moffat + regularised pixel grid | Skip for producing star shapes; pattern to copy (section 1) |
| psfr 0.1.1, psfmachine 1.1.4, AstroPhot 0.18.0, crowdsource 0.5.9 | BSD-3 / MIT / GPL-3.0 / MIT | 2026-09 / 2023-12 / 2026-06-25 / 2026-08 | 34 / - / 106 / - | | | **Skip** (see notes) |
| dLux, poppy, hcipy, prysm, WaveDiff | BSD-3 / BSD-3 / MIT / MIT / MIT | 2026 | | | star shape computed from physical optics, per wavelength | Reuse the idea (a small PyTorch model of the telescope aperture), not as a dependency |

Notes.

- **photutils** (https://github.com/astropy/photutils): builds an "effective PSF" (ePSF: the average star shape from
  many stars, sampled finer than the pixels; Anderson & King method). `GriddedPSFModel` interpolates between ePSFs
  built per region of the image. `PSFPhotometry` / `IterativePSFPhotometry` with fixed positions (x_0/y_0) = forced
  photometry. Masked pixels and empty (NaN) pixels are ignored, so saturated star cores can be treated as missing data.
  Measured locally (CPU, synthetic 2000 x 2000 image, FWHM 2.5 px): EPSFBuilder on 500 stars 11.5 s (sampled 2x finer
  than the pixels) / 15.2 s (4x finer); forced PSFPhotometry 5.9 s for 3000 stars (about 2 ms per star). So one ePSF
  per tile x colour x frame on CPU would take days on 721-1268 frames: use it as a reference and a check on a subset,
  not as the production engine. Untested idea: run EPSFBuilder directly on raw Bayer frames with the other colours'
  pixels masked. Effort about 1 day.
- **Piff** (https://github.com/rmjarvis/Piff): the star shape is stored as a pixel grid (`PixelGrid`) that varies as a
  polynomial (`BasisPolynomial`) of any chosen quantities (`keys`); colour enters through `property_cols` in the input
  and `keys: [u, v, colour]`. In DES Y6 (Dark Energy Survey, year-6 analysis; Schutt et al. 2025, arXiv 2501.05781) the
  pixel grid is fitted jointly, 2nd order in position and **linear in colour**; star-size differences between colour
  groups dropped from about 3 % to below 1 %. Saturated pixels: use a bad-pixel weight map, not `satur` (which drops
  the whole star). Use: run on a few frames in WSL2 as an independent colour-dependent star-shape reference; copy the
  design (polynomial in x, y and lambda_eff, the star's effective wavelength in a given channel) into our PyTorch model.
  With thousands of stars per frame, more than a linear term in wavelength is affordable. Effort 2-3 days.
- **Gnuastro PSF scripts**: star cut-outs are normalised in a ring and stacked by magnitude range (faint stars for the
  core, saturated bright stars with masked cores for the halo); the parts are joined with a scale factor; then each
  star is subtracted. Evidence: Infante-Sainz et al. 2020 (MNRAS 491, 5317), SDSS star shape out to about 8 arcmin;
  Eskandarlou et al. 2025 (arXiv 2510.12940, J-PLUS survey) find the PSF ring radius grows in redder filters (halos
  depend on wavelength). Re-implementation in PyTorch with Gaia magnitudes and per-channel synthetic fluxes: 2-3 days.
- **elderflower** (https://github.com/NGC4676/elderflower, Liu et al. 2022 ApJ 925, 219): statistical (Bayesian) fit
  of the wide-angle star shape from many bright stars plus sky; Moffat core + halo made of several power laws (checked
  in `modeling.py`); developed for Dragonfly (an array of telephoto lenses, closest in spirit to the TS61). Port the
  profile formula per channel in PyTorch, with Gaia fluxes and masked cores: 2-4 days.
- **Skipped**: PSFEx (old toolchain, no Windows, no halo model); psfr (always re-fits star centres; no variation across
  the field or with colour); psfmachine (made for Kepler/TESS; its concept - fixed Gaia star positions, solve for the
  fluxes - is what we already do); AstroPhot (one pixelated star shape per star does not scale to thousands of stars per
  frame); crowdsource (depends on TensorFlow; star-shape variation listed as future work); MRF (needs a matching
  high-resolution image; possible later with Pan-STARRS).
- **Colour-dependent star shape for one-shot-colour (OSC) cameras and refractors**: no open tool or paper specific to
  them was found. Precedents: DES Y6 Piff (linear colour term); Eskandarlou 2025 (halos depend on wavelength); WaveDiff
  (star shape weighted by each star's spectrum, for space optics). Suggested design (ours, not from a cited source): for
  each Bayer channel and each star, lambda_eff = integral(lambda F R_c) / integral(F R_c), where F is the star's Gaia XP
  spectrum and R_c the channel response; plus a small physical optics model (defocus / spherochromatism as a polynomial
  in wavelength) fitted to the per-channel ePSFs, used as a guide (prior) to extrapolate the star shape to H-alpha.

---

## 8. Filling masked pixels (star masks, saturated star cores)

| Candidate | Licence | Activity (checked) | GitHub stars | Runs on | Verdict |
|---|---|---|---|---|---|
| **maskfill 1.1.1** (van Dokkum & Pasha 2024, PASP 136, 034503) | MIT | release 2024-02-02; commit 2024-07-23 | 25 | numpy/scipy/astropy, Windows | **Test as-is + easy GPU port (reuse)** |
| OpenCV `cv2.inpaint` (NS (Navier-Stokes) / Telea methods) | Apache-2.0 | (cv2 5.0 in venv) | - | C++, accepts single-channel float32 | fast fallback / test |
| deepinv `Inpainting` model + plug-and-play (DRUNet = general denoising network, RAM, DiffPIR = diffusion-based...) | BSD-3 / networks trained on everyday photos (licences unverified) | v0.4.2 | 827 | PyTorch GPU | optional cosmetic test only |
| astropy `interpolate_replace_nans`, skimage `inpaint_biharmonic`, Gnuastro `interpolate-*` | BSD / BSD / GPL | | | | **Skip** |
| LaMa (photo inpainting), astroDDPM, Euclid / radio diffusion inpainting, CMB networks (cosmic microwave background), deepCR (cosmic-ray removal) | Apache / AGPL / no code / ... | | | | **Skip** (wrong domain or no code) |

Notes.

- **maskfill** (https://github.com/dokkum/maskfill): fills each masked pixel that touches good pixels with the median of
  the good pixels in a 3 x 3 window, works inward step by step, then smooths the filled area with a 3 x 3 average. About
  30 lines; time proportional to the number of masked pixels. Paper: best overall error versus biharmonic / linear /
  cubic / nearest filling, especially with noise. The fill is noiseless compared with real data: its noise is
  sigma_mask = sigma x (2d + 1)^-0.5 at distance d from the mask edge. It cannot recover structure that does not cross
  the mask edge.

  Measured locally (synthetic 2000 x 2000 image, 3000 round masks covering 8.5 % of the pixels; error = RMSE versus the
  noiseless truth inside the masks):

  | Method | Time | Error (RMSE) |
  |---|---|---|
  | cv2 NS | 0.53 s | 2.05 |
  | cv2 Telea | 0.85 s | 2.43 |
  | astropy interpolate_replace_nans | 2.68 s | 2.42 (687 px left empty) |
  | skimage biharmonic | 3.15 s | 3.60 |
  | maskfill algorithm (numpy re-implementation) | 3.08 s | 2.71 |

  Smooth synthetic backgrounds favour simple methods; all fills are smoother than the noise. Use: port to PyTorch and
  add back the missing noise (variance sigma^2 - sigma_mask^2, where sigma_mask is the fill's own noise). Effort
  0.5-1 day.

- For **saturated cores** of Gaia stars, filling is not the right approach: model = measured star shape x
  Gaia-calibrated flux + background (sections 1 and 7); keep those pixels flagged as missing in any fit.
- No ready-to-use astronomical filling network with code and weights for optical star masks was found (astroDDPM is a
  2022 galaxy cut-out demo, AGPL; the Euclid Q1 diffusion inpainting, arXiv 2503.15321, has no code found).

---

## 9. Colour calibration

| Candidate | Licence | Activity (checked) | Windows | Verdict |
|---|---|---|---|---|
| **Siril 1.4.4 SPCC** | GPL-3.0+; offline Gaia XP catalogue CC BY 4.0 (Zenodo 10.5281/zenodo.14697692) | 1.4.4 2026-06-16; sensor database commit 2026-10-02 | native, `siril-cli` | **Test as-is** (with a custom A7S-mod sensor JSON) |
| **GaiaXPy 2.1.4** | BSD-3 | 2025-12-15 | yes (in venv) | **Reuse** (Gaia XP spectra -> our own synthetic star brightness per channel and effective wavelengths) |
| GaiaXPcorrection (Huang et al. 2024) | not detected | 2026-05 | yes | optional reuse, within its validity range (interstellar reddening E(B-V) < 0.8) |
| colour-science 0.4.7 `matrix_idt` | BSD-3 | | yes | Reuse for the final camera -> display colour matrix |
| SASpro SFCC | GPL-3.0 | 2026-10-10 | yes | Skip as a tool; read `sfcc.py` (its "blind sensor characterisation" idea) |
| Siril PCC, ASTAP, GraXpert | | | | **Skip** (no sensor response, or no colour calibration) |

Notes.

- **Siril SPCC** (https://siril.readthedocs.io/en/latest/processing/color-calibration/spcc.html): needs a plate-solved
  linear image. Stars are matched to Gaia DR3 (data release 3) XP spectra. For each channel, synthetic flux = spectrum x
  sensor QE (quantum efficiency curve) x filter / Bayer filter (x optional low-pass filter); same for a white reference.
  Robust straight-line fit of the image's R/G and B/G against the catalogue ratios; optional correction for the
  atmosphere (Rayleigh scattering). Command
  `spcc [-limitmag=] [-catalog=gaia|localgaia] [-oscsensor=] [-oscfilter=] [-osclpf=] [-whiteref=] [-atmos ...]`.
  The offline catalogue keeps at most the 127 brightest XP stars per sky cell of about 13.7 arcmin (level-8
  HEALPixel): enough for a white balance, but far fewer stars than our Gaia forced photometry.
- **Sensor database** (https://gitlab.com/free-astro/siril-spcc-database, GPL-3; checked by listing `osc_sensors`):
  **no Sony A7S** (22 Sony entries, IMX071 ... IMX715, ICX285AK). DSLR curves include the stock IR-cut filter (Canon
  600D red at 656 nm = 0.35 relative), so a DSLR entry is wrong for an astro-modified camera. Use a custom JSON = bare
  Sony Bayer-filter QE x measured mod-filter curve. No public measured A7S curve was found. Our sensor response derived
  from Gaia XP can be exported in that JSON format for the test. Effort 0.5-1 day.
- **GaiaXPy** (https://github.com/gaia-dpci/GaiaXPy): `calibrate()` gives spectra in absolute units (336-1020 nm, every
  2 nm); `generate()` gives synthetic brightness for built-in filter systems or those of the SVO online filter
  database. For our own curves it is better to integrate ourselves (counting photons), which also gives the effective
  wavelength per channel for the colour-dependent star shape (section 7).
- Method details worth copying from PixInsight SPCC (paid; read from its documentation only,
  https://pixinsight.com/doc/docs/SPCC/SPCC.html): repeated-median regression (a robust line fit that tolerates up to
  50 % outliers), fine integration grid.
- Gaia DR4 is announced for December 2026 (XP spectra per observation date): keep our XP loader independent of the
  data-release version.

---

## 10. What to add to the test bench, by priority

Each item needs something installed, so each needs your go-ahead (PLAN.md, principle 8). Efforts are integration
estimates (wrapper + a run on the 100-frame quick test + scoring), not counting run time on the full data.

### Priority 1: existing tools to test as-is, for Step 1: test bench (cheap; needed to know "who wins what")

| # | Item | Step(s) | What exactly | Effort |
|---|---|---|---|---|
| 1 | **Siril 1.4.4 run from scripts** | Step 1: test bench (stacking); Step 3: sharpening (deconvolution); Step 4: noise reduction; Step 6: sky background; Step 7: colour | `siril-cli -s` scripts: calibrate; register `-disto master -drizzle` (Bayer drizzle, scale 1 / pixfrac 1, then 1.5 / 0.7); stack with rejection; `rl` with our PSF (`makepsf load`) and with `makepsf stars`; `subsky -rbf`; `denoise -vst`; `spcc -catalog=localgaia` with a custom A7S-mod sensor JSON | 2-3 days (one set of scripts); about 100 GB of intermediate files on C: or E: |
| 2 | **StarNet2 2.6.2 `--linear`** (upgrade from 2.5.4) + **nox.exe `-l`** | Step 5: stars | on the linear stack scaled to [0, 1], with and without `-u`; score inside and outside the footprints of known Gaia stars | 1 day |
| 3 | **GraXpert 3.1.0rc2 CLI** | Step 6: sky background; Step 4: noise reduction; Step 3: sharpening (deconvolution) | `background-extraction`, `denoising`, `deconv-obj` then `deconv-stellar` (psfsize within its valid range) | 1 day |
| 4 | **SASpro MFDeconv** | Step 3: sharpening (deconvolution), many frames | wrapper to run it without the interface; 20-40 registered sub-stacks of the batch; star shape measured per frame; lightness and per channel | 1-2 days |
| 5 | **photutils 3.0 GriddedPSFModel + forced PSFPhotometry** | Step 2: star-shape measurement; Step 5: stars (photometry) | per tile and per Bayer channel on a subset of frames; saturated pixels masked; reference for our star-shape map | 1 day |
| 6 | **STScI drizzle 3.0** | Step 1: test bench (check of our stacking) | three Drizzle objects with Bayer weight maps and our pixel maps; compare flux / weights / star shape / noise with our GPU drizzle on 50-100 frames | 1 day |
| 7 | **maskfill + cv2 NS** | Step 5: stars (masked pixels) | fill leftover masks; variant with matching noise added | 0.5 day |

### Priority 2: building blocks for our own tools

| # | Item | What to do | Effort |
|---|---|---|---|
| 8 | **deepinv blur model built from our Gaia star-shape map** | basic star shapes + maps per frame / colour -> `SpaceVaryingBlur`; one shape per tile -> `TiledSpaceVaryingBlur`; RGGB `Inpainting` mask; Fourier transforms computed once and cached; validate with `adjointness_test`, PyLops `NonStationaryConvolve2D` and DeepLens `splat_psf_per_pixel` on crops | 2-3 days |
| 9 | **deepinv non-blind plug-and-play on the single stack** | camera-noise fit term (noise-weighted least squares, or the shifted-Poisson trick) + our Noise2Noise network as the clean-image ingredient (deepinv `PnP` / `DPIR` / HQS solvers); denoiser run tile by tile; also the pretrained RAM and DRUNet networks as ready-made alternatives, for comparison | 2-3 days |
| 10 | **Joint many-frame solver** | ImageMM-style iterative solver, or processing the frames in groups, on a model of each frame (sky -> blur with that frame's star shape -> alignment of that frame -> RGGB Bayer sampling); frames read from the NVMe disk as needed; Noise2Noise denoising steps; sharpen to a small target star size (STARRED idea) | 1-2 weeks |
| 11 | **Noise-reduction comparisons and training terms** | BM3D with noise spectrum, CAREamics N2V2 / structN2V, ZS-N2N; deepinv `SurePGLoss` (SURE for camera noise) / `R2RLoss` added to our Noise2Noise training | 2-3 days (comparisons) + 2-3 days (training terms) |
| 12 | **Colour-dependent star-shape design** | Piff / DES Y6 style polynomial in (x, y, effective wavelength) per Bayer channel, effective wavelength from Gaia XP x channel response (GaiaXPy already in the venv); optional small physical optics model to extrapolate to H-alpha | part of Step 2: star-shape measurement (1-2 weeks in total with item 5) |
| 13 | **Bright-star halos (wings)** | port elderflower's profile formula (Moffat core + multi-power-law halo) and Gnuastro's ring-normalised stacking, per channel, with Gaia fluxes and masked cores | 2-4 days |

### Priority 3: larger or optional efforts (decide after the results of Step 2: star-shape measurement and Step 3: sharpening (deconvolution))

| # | Item | Why / when | Effort |
|---|---|---|---|
| 14 | **ASTERIS retrained** on our sub-stacks (per Bayer colour plane) | state of the art for many-frame noise reduction without clean examples; compare with half-stack Noise2Noise; borrow its training loss and bright/faint split even without retraining | 1-2 weeks |
| 15 | **Our own star remover for linear data** (nox MIT template, or compact-source-removal partial-convolution U-Net) trained with our measured star shapes and noise | only if Gaia star subtraction + StarNet2 limited to known stars fails Step 5: stars | 1-2 weeks |
| 16 | **Piff 1.6 in WSL2** | independent colour-dependent star-shape reference | 2-3 days |
| 17 | **STARRED / AstroPhot tests on crops** | independent joint-fit references on 64-256 px crops | 1-2 days each |
| 18 | **AI tools with closed weights**: Cosmic Clarity AI4 (inside SASpro only), SyQon Zenith / Prism mini / Parallax Nano, DeepSNR, Aberration Remover (Iris) | complete the comparison of free tools; scoring only, never reused | 0.5 day each |
| 19 | Gnuastro / elderflower originals in WSL, DeepSkyStacker, Siril VeraLux Nox / AutoBGE | optional cross-checks | 0.5-2 days each |

### Skipped (details in sections 1-9)

- CUQIpy, ODL, SCICO, Pyxu, ProxImaL, pylops-gpu, SigPy, MIRTorch, NIFTy: no blur varying across the field, toy scale,
  JAX without Windows GPU, or obsolete.
- scarlet / scarlet2 / Tractor: separating galaxies, or catalogue photometry.
- AstroClearNet, ImageMM as code, Firedec, PI-AstroDeconv, Noise2Astro, AstroSURE, Poisson2Gaussian: no code.
- torchmfbd as a dependency: blind only, inconsistent licence.
- Handheld MFSR and burst super-resolution networks: made for daylight bursts, do not conserve flux.
- StarNet v1, starrem2k13, Cosmic Clarity Dark Star as a reusable part: superseded, toy, or licence.
- PSFEx, psfr, psfmachine.
- LaMa, astroDDPM, biharmonic filling, `interpolate_replace_nans`.
- Siril PCC, ASTAP, NoiseChisel.
- The original Noise2Noise / Noise2Void / Noise2Self repositories (use deepinv / CAREamics).

---

## 11. Licence consequences if the project is published on GitHub

- deepinv (BSD-3), PyLops / PyProximal (LGPL-3: keep them replaceable), photutils, GaiaXPy, STScI drizzle (BSD): no
  constraint beyond credit.
- Copying code from SASpro, GraXpert, Siril scripts, STARRED, AstroPhot or elderflower-derived GPL files makes the
  project GPL-3 (acceptable according to PLAN.md; to be decided once).
- GraXpert and Aberration Remover weights are CC BY-NC-SA 4.0: non-commercial, and any derived model must keep the same
  licence.
- StarNet2 licence: outputs may not be used to train networks for commercial software; full-resolution inputs that
  demonstrate its performance may not be shared. Keep StarNet2 outputs out of any training set we publish, and publish
  test comparisons as crops and scores only.
- Cosmic Clarity AI4: SASpro only; no extraction, distillation or retraining. Test scoring only.
- SyQon and DeepSNR terms: unverified (SyQon licence PDF unreachable on 2026-10-10).

## 12. Still unverified

- GraXpert's online list of model versions (needs its credentials).
- BM3D package details.
- Licence terms of SyQon Parallax / Prism / Zenith; DeepSNR licence; licence of the RAM weights.
- ASTERIS on native Windows; deepinv `distributed` on Windows with several GPUs.
- Whether Siril carries the per-pixel drizzle weights into the stack.
- Whether PSFEx accepts colour columns not produced by SExtractor.
- Whether GaiaXPcorrection is valid for the reddening of the Heart field.
