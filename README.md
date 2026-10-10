# astro_GPU

GPU astrophotography processing for colour (one-shot colour / DSLR / mirrorless) deep-sky data: from the raw frames
to a calibrated, physically anchored image, with an interactive final adjustment tab.

What makes it different from the usual stack → stretch → star removal → denoise chain:

* **Bayer drizzle on the GPU** directly from the raw colour pixels (no debayering interpolation), polynomial
  sub-pixel registration, transparency weighting, predictive outlier rejection, two independent half-stacks.
* **Astrometry and colour from Gaia DR3**: plate solving on Gaia, spectral response of *your* sensor reconstructed
  from Gaia XP spectra, physical colour matrix (G2V white).
* **Noise2Noise denoiser trained on your own half-stacks**: it cannot invent detail that is not in both halves.
* **Stars are measured, not erased**: empirical variable chromatic PSF, forced photometry of every Gaia star with a
  colour prior, faint-star model. The starless image is obtained by subtraction (no neural star removal), so faint
  nebulosity under the stars is preserved, and stars are redrawn with the real profile of your optics.
* **Physical sky background**: star flat (flat-field errors measured on the stars), sky gradient separated from the
  emission with a professional H-alpha survey (NSNS / Finkbeiner) and from the unresolved starlight predicted by Gaia.
* **Interactive rendering on the GPU**: tone curve with soft toe, edge-preserving HDR, logarithmic highlight shoulder
  (no burnt cores), OKLab colour handling, star size / density / colour controls, separate layers, automatic
  quality checks (clipping, burnt highlights, dark dips at stars).
* **Resumable**: every stage stores a manifest; an interrupted run continues where it stopped, a parameter change
  only recomputes what depends on it.
* **Every parameter is documented** in the interface, with a basic and an advanced mode, and data-driven automatic
  defaults that always say why they were chosen.

## Install

Requires Windows or Linux, Python 3.11-3.12, an NVIDIA GPU (8 GB+ recommended) and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/n1c1-1/astro_GPU.git
cd astro_GPU
uv sync
```

## Use

Graphical interface:

```
uv run astro-gpu-gui
```

Command line:

```
uv run astro-gpu new D:/astro/NGC7822 --lights D:/raw/NGC7822/lights --darks D:/raw/darks --flats D:/raw/flats --target "NGC 7822" --scratch E:/scratch/NGC7822
uv run astro-gpu run D:/astro/NGC7822          # run / resume
uv run astro-gpu status D:/astro/NGC7822
uv run astro-gpu params D:/astro/NGC7822 stack # effective parameters, with the reason of every automatic value
uv run astro-gpu set D:/astro/NGC7822 stack.scale=2.0
uv run astro-gpu render D:/astro/NGC7822 --preset "H-alpha emphasised" --name ha
```

## Pipeline

| Stage | What it does |
|---|---|
| inventory | lists and checks the frames, reads metadata (exposure, ISO/gain, time, camera, focal length, pointing) |
| calibrate | master dark / bias / flat, bad pixel map, calibrated Bayer frames cached on a fast disk |
| register | per-frame stars, reference frame, 3rd-order polynomial registration, FWHM and transparency |
| stack | weighted Bayer drizzle on the GPU, frame selection, rejection, half-stacks A/B |
| astrometry | plate solve on Gaia DR3 (TAN-SIP) |
| reference | optional: aligns a previous processing of the data for comparison and framing |
| catalogs | Gaia stars of the field (positions at the date of the images) and XP spectra |
| surveys | H-alpha reference survey reprojected on the stack |
| color | sensor spectral response from Gaia XP, camera → sRGB matrix |
| denoise | Noise2Noise network trained on the half-stacks |
| optics | lateral colour correction, saturation magnitude of the setup, empirical PSF |
| stars | forced PSF photometry of every Gaia star, position refinement, faint-star model |
| starless | star subtraction, cleaning, multi-scale Wiener denoising driven by the half-stacks |
| background | star flat, sky model anchored on H-alpha and unresolved starlight |
| render_prep | layers and star tables for the interactive rendering |

## Status

Version 0.1 - colour sensors only (monochrome + filters planned).
