# SideralForge

GPU astrophotography processing for colour (one-shot colour / DSLR / mirrorless) deep-sky data: from the raw frames
to a calibrated, physically anchored image, with an interactive final adjustment tab.

What makes it different from the usual stack → stretch → star removal → denoise chain:

* **Bayer drizzle on the GPU** directly from the raw colour pixels (no debayering interpolation), polynomial
  sub-pixel registration, transparency weighting, predictive outlier rejection, two independent half-stacks.
* **Astrometry and colour from Gaia DR3**: plate solving on Gaia, with an all-sky blind solver (geometric hashing
  of star quads on Hipparcos / Tycho-2) when the target is unknown; spectral response of *your* sensor
  reconstructed from Gaia XP spectra, physical colour matrix (G2V white).
* **Noise2Noise denoiser**: a network pretrained once on many pairs of half-stacks, applied as it is in seconds
  (on the processor too), or first adapted to the image on its own two half-stacks. It never sees a clean image:
  it learned to predict one half-stack from the other, so it cannot invent detail that is not in both halves.
* **Stars are measured, not erased**: empirical variable chromatic PSF, forced photometry of every Gaia star with a
  colour prior, faint-star model. The starless image is obtained by subtracting that model, so faint nebulosity under
  the stars is preserved, and stars are redrawn with the real profile of your optics. A second starless image can come
  from a star-removal network trained on the project's own stack; both are kept and chosen at rendering.
* **Sky gradient removed right after stacking**, before the colour calibration: a polynomial, a surface through
  the sky samples (thin-plate splines) or an automatic fit; the samples and the excluded areas are edited on the
  image, each edit a setting with its undo.
* **Physical sky background**: star flat (flat-field errors measured on the stars), sky gradient separated from the
  emission with a professional H-alpha survey (NSNS / Finkbeiner) and from the unresolved starlight predicted by Gaia.
* **Interactive rendering on the GPU**: tone curve with soft toe, edge-preserving HDR, logarithmic highlight shoulder
  (no burnt cores), OKLab colour handling, star size / density / colour controls, separate layers, automatic
  quality checks (clipping, burnt highlights, dark dips at stars). Detail in faint areas, Noise2Noise fine detail,
  fine colour confirmed by both half-stacks and blue-green / reflection tones are live sliders too: the nebula is
  mixed on the fly from layers prepared once.
* **Resumable**: every stage stores a manifest; an interrupted run continues where it stopped, a parameter change
  only recomputes what depends on it.
* **Every parameter is documented** in the interface, with a basic and an advanced mode, and data-driven automatic
  defaults that always say why they were chosen.

## Install

Requires Windows or Linux, Python 3.11-3.12, an NVIDIA GPU (8 GB+ recommended) and [uv](https://docs.astral.sh/uv/).
From the repository ([github.com/n1c1-1/SideralForge](https://github.com/n1c1-1/SideralForge)):

```
git clone https://github.com/n1c1-1/SideralForge.git
cd SideralForge
uv sync
```

On the development PC the package is installed in editable mode in the Python environment of
`C:\Users\Nicolas\sandbox\ComfyUI\.venv`; `client.bat` starts the client from it.

## Use

The client (projects, sessions, workflows as a graph of blocks, Inspect, Post, export):

```
uv run sideralforge             # or: python -m astro_gpu.client
```

Command line (projects of the older format, one chain per project):

```
uv run astro-gpu new D:/astro/NGC7822 --lights D:/raw/NGC7822/lights --darks D:/raw/darks --flats D:/raw/flats --target "NGC 7822" --scratch E:/scratch/NGC7822
uv run astro-gpu run D:/astro/NGC7822          # run / resume
uv run astro-gpu status D:/astro/NGC7822
uv run astro-gpu params D:/astro/NGC7822 stack # effective parameters, with the reason of every automatic value
uv run astro-gpu set D:/astro/NGC7822 stack.scale=2.0
uv run astro-gpu render D:/astro/NGC7822 --preset "H-alpha emphasised" --name ha
uv run astro-gpu render D:/astro/NGC7822 --preset faithful     # -> results/faithful/ (only what the data prove)
uv run astro-gpu render D:/astro/NGC7822 --preset artistic     # -> results/artistic/ (sharpening, stronger stretch)
```

Every project is meant to end with these two renders: **Faithful** (measured colours, no display sharpening, detail
as measured) and **Artistic** (non-blind sharpening with the star shape measured on the stack, stronger stretch and
local contrast). Both are presets of the final adjustment tab.

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
| gradient | sky gradient model (polynomial, thin-plate splines or automatic) removed from the stack and its halves |
| color | sensor spectral response from Gaia XP, camera → sRGB matrix |
| denoise | pretrained Noise2Noise network, optionally adapted to the image on its half-stacks |
| optics | lateral colour correction, saturation magnitude of the setup, empirical PSF |
| stars | forced PSF photometry of every Gaia star, position refinement, faint-star model |
| star_net | optional: star-removal network trained on the project's own stack; its star layer feeds the learned starless image (default at rendering when it exists) |
| starless | star subtraction, cleaning, multi-scale Wiener denoising driven by the half-stacks |
| background | star flat, sky model anchored on H-alpha and unresolved starlight |
| render_prep | layers and star tables for the interactive rendering |

## Validation

The whole chain was run automatically from the raw frames and compared with two careful earlier processings of the
same data (Sony A7S, TS 61 EDPH with reducer, 7.5 x 5 deg fields):

| Data | Frames | Result |
|---|---|---|
| NGC 7822 (2026-08) | 602 x 30 s, 446 kept | same image as the reference processing (mean difference 5.6/255); blind astrometry without focal length; 1 dark dip on 152 000 stars |
| Heart & Soul (2026-08) | 721 x 15 s | same image (mean difference 4.1/255), on an 8 GB GPU |

Since then the whole chain has run from the raw frames on seven more sessions, each judged blind against the
previous version and, when there is one, the user's own finished image:

| Data | Setup | Notes |
|---|---|---|
| Iris nebula (2025-03) | Newton 130/900, Sony A7S, 1242 x 15 s | reference for the Newtonian (diffraction spikes kept) |
| M33 (2025-08) | Newton 130/900, Sony A7S, no darks, no dithering | strong vignetting, walking noise |
| Horsehead, Flaming Star, Rosette (2024-12-28) | Newton 130/900, Sony A7S, 15 s, no flats that night | darks chosen by uniformity, flats of another night for the vignetting |
| Wizard nebula NGC 7380 | Newton 130/900, Sony A7S | blind astrometry; variable transparency |
| M31 (2023-08) | Newton 130/900, Sony A7S, no darks | large smooth object: sky fit still open |
| M31 (2026-09-12) | Newton 150/750, 6000 x 4000 14-bit colour camera (FITS), 30 x 60 s | strongly defocused stars (rings) |

Typical durations on an RTX 4060 Ti: about 1 h 15 for 600 frames of 12 Mpx when the denoiser was trained on each
image (35 min of it); the pretrained denoiser now takes under a minute;
interactive rendering: 0.15-0.35 s per adjustment in preview, 2 s at full resolution.

## Status

Version 0.1 - colour sensors only (monochrome + filters planned). Tested on Windows 11 with RTX 4060 Ti (16 GB)
and RTX 3070 (8 GB).

Interfaces: the client (`astro_gpu/client`, `client.bat`) and the command line. The Python package keeps its
working name, `astro_gpu`. How the code is organised:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Tests: [tests/README.md](tests/README.md). History (plans, the
interface mockups, the ComfyUI blocks, the notes of past work): [docs/archive/](docs/archive/) and the git tag
`archive/avant-nettoyage-2026-11-03`.
