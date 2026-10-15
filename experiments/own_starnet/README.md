# Own-data star removal network (experiment)

Question: can a star-removal network trained only on one project's own stack remove the bright stars
(cores, diffraction spikes, rays, halos) more cleanly than the model-based star removal (`sl_final`)?

Test project: Iris nebula (Newtonian, 4 spikes), `C:/Users/Nicolas/astro_gpu_scratch/projects/Iris/work`.
Outputs: `D:/astro_gpu_diag/own_starnet/` (results and verdict in `REPORT.md` there).

## How the training pairs are made

* **Target** = `starless/sl_final.npy` (the model-based starless image, denoised), used only where it is reliable:
  the loss is switched off within max(150 px, 1.2 R) of every star brighter than G 10, within 1.2 R of the
  G 10-10.5 stars, and in two hold-out boxes (the dark-lane part of the Iris core and a 512 px faint-nebula box,
  chosen automatically). Patch centres are at least 150 px from every star brighter than G 10.
* **Input background** = the real stack `optics/lat_full.npy` at the same place. Choice made for simplicity and
  faithfulness: it has exactly the right noise (level, correlation, colour) and the real faint stars, which the
  target has removed, so the network learns faint-star removal from the model-based result, which is good there.
  The half-stack noise sample (A - B) / 2 is only used to measure noise (normalisation, photon gain, evaluation).
* **Pasted stars** = real stamps of this stack. Source = the Noise2Noise-denoised stack (`denoise/n2n_full.npy`),
  colour-realigned exactly like `lat_full` (same `apply_lateral`; it matches `lat_full` to 0.1 % in flux at the
  saturated cores), minus `sl_final`. Using the denoised stack avoids pasting a second copy of the noise.
  - bright library: every star G < 11 with no brighter star within 60 px and its window inside the frame
    (93 stamps, 18 saturated); radius R = 40 px at G 11 to 250 px at G 6 (log-linear, capped at 400 px);
    the median of the outer ring (0.85-1 R) is removed (glare of the other stars, background offsets), then a
    cosine taper (flat to 0.6 R, zero at R), negatives clipped. Spikes are included. No rotation, no flip.
  - faint library: 1740 isolated stars G 11-15 (radius from the same law, min 12 px), 0-24 per patch.
  - 1-6 bright stamps per patch, drawn with weight 10^(-0.15 (G - 11)) (bright ones more often), centres anywhere
    from -0.3 R to P + 0.3 R (so partial stars and lone spikes occur), scale in [0.3, 2] (saturated stamps:
    [1, 2], because a dimmed saturated core is a flat top that never occurs), colour jitter of 8 %.
  - re-saturation: a saturated stamp is clipped at its own measured plateau (plateaus differ from star to star in
    this drizzled stack: 13.9k to 21k ADU); an unsaturated stamp scaled above saturation is clipped at the
    99.999th percentile of the channel times U[1, 1.35]. Light never lowers a pixel that is already brighter.
  - photon noise of the pasted light: Gaussian, variance = b * light with b measured on the half-stacks
    (var = a + b s + c s^2 fitted around G 10-13 stars; b = 0.030 / 0.030 / 0.046 per ADU in R / G / B);
    no noise is added on clipped (saturated) pixels.

## Network and loss

* `astro_gpu.stages.denoise.UNet`, widths (32, 64, 128, 192, 256, 320): 5 down-samplings, ~400 px receptive field,
  9.8 M parameters, bf16 autocast.
* Inputs normalised per channel by n = asinh((x - sky) / (4 sigma)) (sky = median of sl_final, sigma = pixel noise
  of the stack): faint nebula stays linear (1 sigma = 0.25), saturated cores end near 9.
* Output = starless in the same space, as a residual: n_out = n_in + net(n_in).
* Loss = L1 in that space, weight 0 where the target is not trusted, x5 on the pasted stars' footprints
  (light > 1 sigma, dilated 3 px).
* AdamW 3e-4, warm-up 300 steps, cosine decay over a wall-clock budget (32 min), patches 384 px, batch 6.

## Inference and evaluation

* Full frame in 512 px tiles with 128 px overlap, reflect-padded edges, linear feathering -> `starless_net.npy`.
* Per evaluation star (the 4 bright ones given + 4 medium stars G 9.3-10.9 picked automatically: no brighter star
  within 250 px), ring 20-150 px: a quadratic background is fitted robustly on the 150-250 px ring of each starless
  image; residue / hole = pixels of the RGB sum above / below 3 local noise sigmas (sigma from (A - B) / 2 in the
  150-250 px ring); also the mean deviation, and the negative part of the 2 px-smoothed star layer
  (input - starless).
* Nebula regions (hold-out boxes, never seen as targets): rms and mean of (net - sl_final) in noise sigmas, and the
  ratio of fine detail (rms of image minus its 3 px Gaussian blur) net / sl_final.
* Known-truth test: real stamps pasted at the centre of the hold-out boxes, output with minus output without.
* Sheets (100 %, same strong asinh stretch for every panel of a row, 20 and 150 px rings drawn):
  `sheet_bright_stars.jpg`, `sheet_medium_stars.jpg`, `sheet_bright_wide.jpg` (50 %), `sheet_nebula_regions.jpg`.

## Runs

* `runA`: all 93 bright stamps (as specified; the evaluation stars are in the library).
* `runB` (`--hold-eval`): every stamp that holds an evaluation star is left out (85 stamps, brightest G 8.0),
  to see what the network does on bright stars it has never seen.
* `runC` (`--minus-halo --noisy-target`): two fixes for what run A showed.
  - `--minus-halo`: the measured far halo of the stars (`stars/stars_model_halo.npy`, the glare model of the
    pipeline) is subtracted from the input and from the stamp source first (stamps from `library_nohalo.pt`,
    built by `prep --minus-halo`); the network only handles cores, spikes, rays and the inner halo.
  - `--noisy-target`: target = sl_final + the input's own noise (lat_full - denoised stack), so the network learns
    star removal only, not denoising (run A's denoised target made it a strong smoother); its output is noisy
    and meant to go through the pipeline's own denoising afterwards.
* `runD` (`--minus-halo --noisy-target --bright-aug`): run C + nebula brightness augmentation. Run A showed that the
  training targets hold almost no bright nebula (0.003 % of the trained pixels are > 10 sigma above the sky, against
  53 % of the Iris core within 200 px of its star: the only bright nebula of the frame surrounds a bright star,
  inside its masked zone), so the network took the Iris core for a halo. In half of the patches the patch's own
  nebula is amplified k = 1..40 times (log-uniform), same structure: luminance above the sky in noise units,
  6 px Gaussian, interpolated over the filled star cores of sl_final (so star-sized defects are not amplified),
  coloured like the frame's brightest nebula (100-300 px from the Iris star) +-15 %, plus its photon noise, added to
  the input and to the target alike.
* `runE` (`--minus-halo --noisy-target --bright-aug --core-mask`): run D + no loss on the cores of the real stars
  (sl_mask dilated 3 px, and discs of max(4, 0.12 R) px around every catalogue star G < 16; 4.9 % of the frame).
  Run C showed two defects learnt from the targets at real star cores: (1) lat_full - n2n is not pure noise there
  (systematic colour residue of about -3 / +3 / -2 sigma in R / G / B at the cores of G 9-15 stars: the denoiser
  makes star cores colour-consistent), which taught the network to leave green blobs at faint stars; (2) the dark
  discs that sl_final leaves at some G 10-11 stars, which the network learnt to reproduce. With the mask, core
  removal is learnt only from the pasted stamps, whose target is the true sky; no noise is added to the target there.
  For noise-preserving runs the region check adds an identity test: rms of (net - (sl_final + input noise)) on
  star-free pixels, i.e. how much the network changes pixels that hold no star.

All star metrics are computed on images smoothed by a 1.5 px Gaussian (so a noisy output is comparable with a
denoised one; the sheets show the unsmoothed images). A core metric (r < 20 px, deviation from the same background
fit) is reported too; for the star inside the Iris core it is not meaningful (the nebula peaks at the star).

```
set CUDA_VISIBLE_DEVICES=1
python experiments/own_starnet/own_starnet.py prep
python experiments/own_starnet/own_starnet.py prep --minus-halo        (only for runC)
python experiments/own_starnet/own_starnet.py train --run runA --minutes 32 --batch 6
python experiments/own_starnet/own_starnet.py train --run runB --hold-eval --minutes 32 --batch 6
python experiments/own_starnet/own_starnet.py train --run runC --minus-halo --noisy-target --minutes 32 --batch 6
python experiments/own_starnet/own_starnet.py train --run runD --minus-halo --noisy-target --bright-aug --minutes 32 --batch 6
python experiments/own_starnet/own_starnet.py train --run runE --minus-halo --noisy-target --bright-aug --core-mask --minutes 32 --batch 6
python experiments/own_starnet/own_starnet.py infer --run runA          (same for runB ... runE)
python experiments/own_starnet/own_starnet.py eval  --run runA
python experiments/own_starnet/own_starnet.py table --run runA,runB,runC,runD,runE
```

## Other projects (generalised, 2026-10-15)

Every command takes `--project <folder holding work/>` (read only; default: the Iris) and `--out <dir>` (default
`D:/astro_gpu_diag/own_starnet/<project folder name>`; for the Iris it stays `D:/astro_gpu_diag/own_starnet`).
Nothing in the code refers to a particular object or instrument; `prep` measures on the project's own data:

* **Star extent law.** On the 60 brightest stars G < 11 with no brighter star within 100 px, the star layer
  (denoised stack - far halo - sl_final) is measured in 4 px rings: the halo radius is where the ring median (taken
  relative to its level in the outer third of the 1000 px window) falls below 0.25 noise sigma; the spike radius is
  where the median along the spikes (directions measured by the starless stage) minus the median 6-14 px beside them
  falls below 0.25 sigma (RGB sum, 2 px Gaussian, 5-ring running median). R = 1.15 x the larger one; a robust
  log-linear fit R(G) gives the law (12..600 px) used for the faint stamps, the bright-star zones and the core mask;
  a measured bright star takes the larger of its own R and the law. (The first Iris prep used a fixed law: 40 px at
  G 11 to 250 px at G 6; its meta has no `radius_fit` and keeps it.)
* **Hold-out boxes.** At 1/4 scale, the luminance of sl_final in noise units: the structured box (384 px) is the one
  with the largest structure (std of the 8 px-smoothed luminance) outside every bright-star zone; the faint box
  (512 px) is the most structured among the faintest quarter of the admissible boxes, 256 px away from the first.
* **Nebula colour** for the brightness augmentation: median colour (noise units) of the brightest 0.5 % of the
  16 px-smoothed luminance, away from the filled star cores and from the inner half of the bright stars' extent.
* **Evaluation stars.** The 4 brightest stars at least 300 px from the edges with no brighter star within 150 px,
  and 4 medium stars G 9-11 (closest to 9.3, 9.9, 10.4, 10.9) with no brighter star within 250 px; none in a box.

`infer` also writes `star_layer_net.npy` = network input (lat_full minus the far halo for `--minus-halo` runs) -
starless_net, float32 (3, H, W), in stack units; `layer --run R` writes it from an existing inference.
`prep_overview.py` draws what prep chose (`<out>/prep/overview.jpg`); `compare_runs.py runE --at x,y,half,label`
makes a sheet at arbitrary places. `run_projects.sh <project> ...` runs the run E recipe on several projects one
after another (prep if missing, train 32 min, infer, eval, sheets), GPU 1 only:

```
bash experiments/own_starnet/run_projects.sh E:/astro_gpu/projects/Rosette E:/astro_gpu/projects/FlamingStar E:/astro_gpu/projects/Horsehead
```

## Bright-star stamps (runs F and G, 2026-10-15, Flaming Star and Horsehead)

The learned removal dug more light than the brightest stars have (Alnitak, sigma Ori, Flaming Star G 4.1 / 5.3).
Run inputs: read-only snapshots of the project files (`D:/astro_gpu_diag/own_starnet/<name>/snapshot`).

* `runF` (`stamps`, `train --stamps membrane`): bright stamps against a background interpolated under the star
  (membrane fill of sl_final from the ring just outside its measured extent), window 25 % beyond it. The holes of
  sl_final under the star no longer become "star light" (known-truth test: pasted stars removed without holes on
  faint sky, fewer on bright nebula) but the craters at the real brightest stars stayed or deepened, and a
  reflection nebula centred on its star (NGC 2023) was flattened: the smooth background runs below it, so the stamp
  holds the nebula's glow.
* Diagnosis (`crater_check.py`): the network removes its own Alnitak stamp pasted on the real surroundings almost
  perfectly, whatever the tiles, noise or core; with the negative parts of the stamp (clipped at 0 in every stamp so
  far) put back, the full crater appears (-24 sigma). The input (stack minus the far-halo model) has dark rings and
  lobes around the brightest stars (the far-halo model is stronger than the star's light there); no stamp ever
  contained them and the network amplified them about six times.
* `runG` (`stamps --stamp-variant agreed`, `train --stamps agreed`): a bright stamp is the star as the input shows
  it, without the nebula. Bright part: light above the higher of the interpolated background and sl_final (a
  model-based nebula estimate) outside sl_final's filled cores, above the interpolated background inside them.
  Dark part: point-symmetric darkness below the interpolated background (a pixel and its mirror through the star
  both dark: the smaller darkness), inside the star's extent. The loss footprint counts the dark parts.
* `faint_in_glare.py`: faint stars inside the glare stay point-like in the star layer (elongation, radial tail,
  flux) in runs E, F and G alike.

## Compact light only (run H, 2026-10-15, four projects)

The rendering shows a saturated star by its own light inside a footprint only, faded out below 1.5 sigma; the
learned starless image lacks the whole layer. What the network removed beyond what the rendering shows was lost:
grey discs with an edge around saturated stars (Rosette G 5.6 north). Run H: the network removes only the compact
light of the stars; their diffuse glare stays with the far halo model or with the nebula.

* Far halo first (`experiments/halo2/halo_lab.py`, stars stage 13): no halo above the light the data show beyond
  the nebula (see `astro_gpu/stages/far_halo.py`, `_envelope`). Outputs in `D:/astro_gpu_diag/halo2/<project>`.
* `make_snap_halo.py`: a project copy with that halo (hard links to the rest) and the model starless image with it
  (sl_final + old halo - new halo). `same_holdouts.py`: the hold-out boxes and evaluation stars of the earlier runs.
* `stamps --stamp-variant compact`, `train --stamps compact`: each bright stamp is split into compact light (removed)
  and diffuse light (pasted into the input and the target alike). Compact: inside a radius - for a saturated star,
  where its broad light (median of each ring) falls below the pixel noise, at least its saturated core; else the
  star model's radius (60 px) - and, beyond it, what is narrow across (spikes, rays, stars: the stamp minus its
  median over half a ring: over 45 degrees, the glare of sigma Orionis' companions passed for diffuse light).
* `lost_light.py`: the light the rendering loses around each saturated star (footprint rule of
  render_prep.saturated_star_light, the 1.5 sigma fade, the starless stage's 3 sigma cut of the layer).
* `compare_GH.py`, `table_GH.py`: run H against the layer a project used (sheets, ring levels, tables).
* `run_H.sh <project> ...`: the whole chain, GPU 1 only.

Results (runs H against the layers the projects used: run G for the Horsehead and the Flaming Star, run E for the
Rosette; `D:/astro_gpu_diag/own_starnet/<project>/H`):

* light lost at rendering around the saturated stars, against the light shown: Horsehead 0.18 -> 0.17 %, Flaming
  Star 0.42 -> 0.39 %, Rosette 1.92 -> 1.28 % (its cluster); Rosette G 5.6 north 0.17 -> 0.16 %. Most of it is not
  diffuse light beyond the footprints: the G 5.6's is 91 % inside its footprint, where the rendering fades the
  star's light out below 1.5 sigma per pixel (on the noisy image) - any layer that falls smoothly to zero loses its
  faint part there; beyond the footprints the losses are rays (Alnitak) and the glow of the Rosette's cluster.
* the split left ring glows in the starless image around the saturated stars (`ring_glows.py`: stars whose nebula
  profile rises again by more than 0.5 sigma between 40 and 160 px): Horsehead 5 -> 12 of 40, Flaming Star 2 -> 4 of
  45, Rosette 5 -> 13 of 61. Not adopted.
* evaluation stars, hold-outs: as before (residue at Alnitak 0.6 -> 2.7 %, holes 0 in both).
* NGC 2023 (around HD 37903, G 7.8, drawn at rendering): runs G and H both remove its bright core as star light
  (layer 109 sigma at 10-30 px from the star, of 117 there; the star model holds 5). Run H2 tests a remedy.

## Run H2 (2026-10-15): the corrected far halo, full stamps, a glow on the pasted stars

`run_H2.sh`: run H's snapshot and prep, the stamps of run G not split (`stamps --stamp-variant full`, saturated as
the rendering sees it), and a smooth glow centred on half of the pasted bright stars, in the input and the target
alike (`train --glow`: 2-300 noise sigmas at the centre, scale 15-150 px, the nebula's colour or the star's).
`post_H2.sh`-style evaluation: `lost_light.py`, `compare_GH.py`, `table_GH.py`, `ring_glows.py`.

Horsehead (run G / run H / run H2):

* NGC 2023: the layer around HD 37903 holds 109 / 110 / 48 sigma at 10-30 px from the star, 51 / 51 / 10 at 30-50,
  26 / 26 / 2 at 50-70 (the stack minus the far halo: 117, 58, 33; the star model: 5, 1, 0) - the nebula's core is
  kept.
* ring glows around the saturated stars: 5 / 12 / 3 stars above 0.5 sigma (median 0.16 / 0.40 / 0.01).
* sigma Orionis: nebula level at 20-40 px -4.7 / -1.6 / -0.3 sigma (its surroundings at 250-350 px: -4.0 / -3.2 /
  -3.2): the dark disc of run G is gone, a glow of about 3 sigma (its glare above the lowered far halo) is left.
* Alnitak: residue in the 20-150 px ring 0.6 / 2.7 / 15 % (the glare above the far halo, +7 sigma at 20-80 px,
  stays with the nebula); light lost at rendering 0.18 / 0.17 / 0.26 % of the saturated stars' light (Alnitak's
  rays beyond its footprint).
* hold-outs: unchanged within 0.1 sigma.

Rosette (run E / run H / run H2): light lost 1.92 / 1.28 / 1.84 % (G 5.6 north 0.17 / 0.16 / 0.19 %, 84-91 % of it
inside its footprint: the rendering's fade); ring glows 5 / 13 / 3 stars above 0.5 sigma; hold-outs unchanged.

Where the light is lost: around the Rosette's G 5.6 north, at 12-84 px from the star, where its light stands 1.5-3.5
sigma per pixel - the rendering fades the star light out per pixel there, on the noisy image (stack minus learned
starless image). Any layer that falls smoothly to zero loses that part; a cut placed higher leaves a ring in the
starless image instead (run H). Beyond the footprints: the rays of Alnitak, the glow of the Rosette's cluster.
`post_H.sh`, `post_H2.sh`: the evaluation after a run (paths of D:/astro_gpu_diag).

Iris (run E / run H2): light lost 0.11 / 0.10 %, no ring glow, hold-outs unchanged; but at the Iris's central star
(HD 200775, G 7.2) run H2 leaves a round glow of about 40 px in the starless image (nebula level 113 sigma at the
star against 35 with run E; 51 against 35 at 20 px): the glow taught on the pasted stars also keeps the broad core
of a bright star sitting in its nebula. The stage makes the glow a setting ('Nebula peaked on its star').
