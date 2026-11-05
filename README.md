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
* **Star removal by a network trained on our own stacks**: NAFNet (Chen et al. 2022), trained on StarNet's starless
  images corrected by hand and on stars drawn on top of them (a Moffat core, an r^-3 aureole, diffraction spikes,
  saturation). It works on the linear image and gives two images, the starless image and the stars alone (they add
  up to the image, to the pixel). On the bench it leaves less of the faint stars than StarNet and takes the spikes and
  halos of the bright stars off, the background unchanged.
* **Sky gradient removed right after stacking**, before the colour calibration: a polynomial, a surface through
  the sky samples (thin-plate splines) or an automatic fit; the samples and the excluded areas are edited on the
  image, each edit a setting with its undo.
* **Post with layers, as in the image editors**: each workflow's image, the stage results sent from Inspect, filters
  (stretches, curves, local contrast, multiscale sharpening and noise reduction, colour), masks and blend modes. The
  Stars block's two images come as a Stars and background layer (each stretched, the stars put back in Screen, with
  their own sliders: strength, size, colour), or each as a layer of its own; the graph shows these wires and the
  Stars and background blocks of its Composition section, with the same settings as in Post.
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
| catalogs | the bright Gaia stars of the field (positions at the date of the images) and XP spectra |
| surveys | H-alpha reference survey reprojected on the stack |
| gradient | sky gradient model (polynomial, thin-plate splines or automatic) removed from the stack and its halves |
| color | sensor spectral response from Gaia XP, camera → sRGB matrix |
| denoise | pretrained Noise2Noise network, optionally adapted to the image on its half-stacks |
| render_simple | the Output block: the linear image in screen colours, its sky removed, and the statistics of its stretch |
| star_removal | the Stars block: the trained network takes the stars off the Output block's image (the starless image and the stars alone) |

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
image (35 min of it); the pretrained denoiser now takes under a minute, the star removal about 10 s for 12 Mpx;
Post: a fraction of a second per adjustment in preview.

## Status

Version 0.1 - colour sensors only (monochrome + filters planned). Tested on Windows 11 with RTX 4060 Ti (16 GB)
and RTX 3070 (8 GB).

Interfaces: the client (`astro_gpu/client`, `client.bat`) and the command line. The Python package keeps its
working name, `astro_gpu`. How the code is organised:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Tests: [tests/README.md](tests/README.md). History (plans, the
interface mockups, the ComfyUI blocks, the notes of past work): [docs/archive/](docs/archive/) and the git tag
`archive/avant-nettoyage-2026-11-03`.
