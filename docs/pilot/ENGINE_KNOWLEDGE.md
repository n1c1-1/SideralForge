# Engine knowledge for the pilot

Gathered 2026-10-31 from the memory notes (heart_nebula session 2026-10-08..28, sandbox/astro sessions 2026-10-28..31),
PLAN.md, BACKLOG.md, the engine session's user messages (07466723), git log, and read-only checks of the project
folders and of the G: archive done the same day.

## 1. The user's working rules

1. **Plain French** in reports and logbooks, with no internal codes (J2, n2nd2, runH2…). Result first, then numbers,
   then the decision needed. Why: he reads to decide, and jargon cost trust (2026-10-10).
2. **Short documents**: a spec is about one page of bullets ("n'écris pas un roman", 2026-10-31).
3. **No patch stacking.** When a rule needs a 2nd or 3rd per-case variant, stop, say so, and propose a design-level fix.
   After two failed attempts, stop and report with numbers (PLAN principle 5). Why: stacked repairs made rings and
   drift ("bricolage de patch", 2026-10-11).
4. **Generalise from the data.** No rule keyed on focal length, instrument or camera; a percentile-only threshold is
   suspect. Why: the refractor-tuned chain broke on the Newton (2026-10-10).
5. **Visual QC at 100 %, by yourself, before presenting**: faint stars on plain sky, stars in bright nebula, a dense
   field, every bright star with spikes. Any ring, halo, disc or bar means it cannot be shown. Why: he judges at 100 %,
   and averages hid the halos (2026-10-11).
6. **Independent blind QC agents** judge every render before he sees it, without knowing the change. His images are
   guides, not targets. Why: my own QC passed a "horrible" Horsehead (2026-10-13).
7. **Non-regression = "not worse anywhere"**, not identical. It runs on 4 datasets: Heart (refractor), Iris, Rosette,
   Horsehead (Newton) since 2026-10-28 (earlier NGC 7822, Heart, Iris, M33). Subsets first, then full sets. **The
   Rosette's faint-extension detail must be kept absolutely.**
8. **Real projects are never modified.** Work on copies (nonreg.py) or a separate state (ASTRO_GPU_HOME), and never
   re-run one without his go ("1. non", 2026-10-28). Pilot: inputs copied into new clean folders on F:.
9. **Nothing deleted without his OK** (recursive delete is denied). Archive under `E:\astro_gpu\archive\<what>_<date>`.
   A deletion list (~330 GB, plus ~140 GB of nonreg copies) still awaits him.
10. **Live logbook for long runs**: step, progress, honest ETA, usable intermediate results viewable remotely
    (artifact). "je ne veux pas être aveugle sur ce que tu fais et sur l'ETA" (2026-10-08).
11. **Options kept at render**: variants become layers + render settings, the blind-QC winner as default, plausible
    losers kept. "une solution ne couvrira pas tous les cas" (2026-10-15).
12. **Views, not rules**: when a fix helps some zones and hurts others, offer views (to combine later with masks),
    not automatic arbitration. Compute main + learned removal up front (2026-10-16).
13. **Keep the Newton diffraction spikes**, in the star layer; broken or half-subtracted spikes are a defect. "c'est
    tout le charme de l'image au Newton" (2026-10-11).
14. **Results where he looks**: in the GUI and the project results, comparisons on one page. Never point him to
    scratch files. No test copies in the GUI list; results folders hold only the current faithful and artistic
    deliverables. Why: he once judged the wrong images (2026-10-10, 2026-10-14).
15. **Two deliverables per project**: Faithful (detail proven by the half-stacks, measured colour) and Artistic
    (invention allowed). Not scientific imaging (2026-10-11).
16. **Do what he asked before arguing.** The per-frame deconvolution was dodged 3 times ("invraisemblable"). Run it,
    show the result and the next variant, let him decide (2026-10-15).
17. **Measure before claiming.** Never write an expected outcome in the logbook ("Heart identical" was wrong,
    2026-10-28). Run `date` before writing times.
18. **No permission stalls**: he leaves the screen, every tool is allowed (only recursive delete is denied). Never
    add exact commands to allow lists; fix a prompt's cause in the settings ("tu as les manettes", 2026-10-31).
19. **Git and tools**: commits authored by Nicolas (local config; global identity is Kat) with Co-Authored-By. No
    GitHub, never push. Branches in `sandbox\astro`, never worktrees or copies in sandbox (2026-10-25). Free tools
    only; licences do not matter for internal test data (2026-10-28).
20. **No full auto, no magic**: suggest parameters and show what the engine does ("les astrophotographes aiment
    rarement la magie", 2026-10-31).
21. **Delegate mechanical work** (runs, sheets, logbook) to cheaper agents; keep image inspection and decisions. Use
    stronger agents when stuck (2026-10-11, 2026-10-30).

## 2. The projects (state on 2026-10-31)

**Shared facts**
- Camera: Sony A7S mark I, astro-modified, lossy ARW, 15 s subs unless stated.
- Instruments: TS61EDPH refractor with 0.75x reducer (272 mm, 6.37"/px); Newton 130/900 (~893 mm, 1.93"/px, strong
  vignetting, spider spikes).
- GUI registry: `C:\Users\Nicolas\.astro_gpu\projects.json`.

**What a re-run changes.** All real projects are out of date for star_net 4, background 5, stars 15, render_prep 44.
- The Heart retrains its star network (old layer time-trained, so the result will differ); its framing moves a few px.
- M31 2023 and M33 switch to the aperture star flat (maps 2-3 % apart); 12 M33 frames may get the rotation search.
- The runner fingerprints input path lists: new input folders force a full recompute.

| Project | Folder | Raws now (checked 2026-10-31) | Settings of note |
|---|---|---|---|
| Heart & Soul IC 1805/1848, refractor | `F:\photos\heart_nebula\astro_gpu` | `F:\photos\heart_nebula\raw\{signal 721 (+281 cloudy in discard), dark 28, flat 46}` | ref `full.png`; star_flat off; in-project star_net; cuda:1 |
| NGC 7822, refractor | `F:\photos\NGC7822\astro_gpu` (not in the GUI registry) | `F:\photos\NGC7822\{signal 602 x 30 s, dark 19, flat 82}` | ref `NGC7822.png` |
| Iris NGC 7023, Newton | `C:\Users\Nicolas\astro_gpu_scratch\projects\Iris` | `D:\photos\2.3.25_Nebuleuse_iris\raw\{signal 1268, dark 48, flat 62}` | ref `reference\iris_user_manual_full.png`; layer `D:\astro_gpu_diag\learned\Iris_star_layer.npy`; star_net off; cuda:0 |
| M33, Newton | `C:\Users\Nicolas\astro_gpu_scratch\projects\M33` | `D:\astro_photo\22_Aout_25_Triangle\{Raw2 1096, Flat}`, no darks | ref `crop.png`; EXIF date 2014 (real 2025-08-22) |
| Rosette NGC 2237, Newton | `E:\astro_gpu\projects\Rosette` | lights `G:\Astro\2024-12-28_divers_astro\Rosette\signal` (358 ARW) | layer `own_starnet\Rosette\runE`; halo_bound false |
| Horsehead B33, Newton | `E:\astro_gpu\projects\Horsehead` | `G:\Astro\2024-12-28_divers_astro\HorseHead\signal` (783) | layer `own_starnet\Horsehead\H\runH2`; halo_bound **true** |
| Flaming Star IC 405, Newton | `E:\astro_gpu\projects\FlamingStar` | 724 lights split: `G:\Astro\2024-12-28_divers_astro\Etoile_flamboyante\signal` (467) + `G:\Astro\2024-12-27_Etoile-Flamboyante\raw` (257) | layer `own_starnet\FlamingStar\runG`; halo_bound false |
| M31 2023-08-24, Newton | `E:\astro_gpu\projects\M31` | `G:\Astro\2023-08-24_M31_top\{signal 508, flat 30}`, no darks, daylight flats | ref `...\traitement\m31_flat.jpg` (now on G:) |
| Wizard NGC 7380 (folder "Sorciere", not the Witch Head), Newton | `E:\astro_gpu\projects\Wizard` | `E:\Astro-photo\Sorciere\raw\{signal 1181, dark 10, flat 57}` | no reference |
| M31 2026-09-12, defocused | no project.json: ComfyUI graph `D:\astro_gpu_m31\m31_api.json`, blocks folder `D:/astro_gpu/blocks` | `F:\astro-photo\M31_bruts_2026-09-12.zip`, extracted to `D:\astro_gpu_m31\raw` | delivered in `F:\astro-photo` |

**The three 28.12.24 objects** (Rosette, Horsehead, Flaming Star; ISO 4000, camera clock wrong)
- Darks: 100 uniform frames from `G:\Astro\2024-12-28_divers_astro\957MSDCF`, listed one by one in project.json. Later
  frames hold a growing central light; keep only centre minus corners <= 4 ADU.
  - **WRONG (found 2026-11-01, docs/pilot/DARKS_28_12_24.md): none of them is a dark.** All 100 are frames of the
    Rosette field shot with the telescope open (26–410 stars each; 520–524 ADU against 508–512 for real darks of this
    camera). The master dark carries star trails, which show as diagonal streaks in Rosette's results. That is so
    since v1 (2026-10-13). None of the 1,233 raws of 957MSDCF is a real dark.
  - Options: the 48 real darks of the Iris night (same camera, 15 s, ISO 4000), or no darks with cosmetic
    correction; compared on the bench (BENCH.md, "Rosette et ses darks"). The check in core/diagnostics.py now flags a
    calibration frame with stars.
- No flats that night: the Iris flats give the vignetting only, smoothed (calibrate.flat_scale_px 40).
- The project.json input paths (`F:\backup_pc\...`) are dead. Second copy on `I:\backup_pc_LENOVO\photos` (bad zone).
  His Siril stacks: `D:\astro_gpu_diag\siril_ref`.
- The 4 learned layers (327 MB each, under `D:\astro_gpu_diag\`) are referenced by absolute path in
  `starless.learned_stars`.

**Known weaknesses**
- **Heart**: brown sky, single-colour knots, dark holes beside some stars. His daughter prefers the original (blue-green
  tones lost, faint detail softer). His favourite render: `results_archive/natural`, 2026-10-10 10:34.
- **NGC 7822**: results cluttered with old test folders (v7…v31_natural, nonreg_*): archive, don't delete. Transparency
  drops after 02:45.
- **Iris**: far below his manual image in depth, colour and stars; salmon streak through the central star; colour
  mosaic in the faint NE dust; T Cephei dark disc gone only with halo_bound. The G6.9 blob is real dust (his view).
- **M33**: the transparency rule dropped 36 % of the frames; the galaxy's smooth wings bias the sky fit.
- **Rosette**: OIII teal drowned though in the data; diagonal stripes (walking noise, no dithering); olive discs if
  halo_bound is on. Its faint extensions beat his own image.
- **Horsehead**: milky veil, because NGC 2023's core sets the white point (`white_ignore_pct` 0.05-0.2 % helps);
  Alnitak's blue disc; grey bars beside medium stars; faint dust smoothed; grey reflection nebulae.
- **Flaming Star**: red wash; core residue or dips at the bottom bright stars (G 4.1 / 5.3 / 4.9); a lavender oval.
- **M31 2023**: dark ring around the galaxy, pink edges; the sky fit sees 27 % of the field; the core needs large-scale
  compression off (saved view "Faithful - galaxy core").
- **Wizard**: poor transparency; the right edge (drift zone) needs cropping.
- **M31 2026-09-12**: another camera (6000x4000, 3.72 µm, 14-bit FITS, ISO 800), Newton 150/750 at 1.02"/px, 30 x 60 s,
  9 darks, no flats, donut stars (radius ~6 px). Left: grain and dark specks in the faint disc, a slight green tint.

**More of his sessions for new use cases**, probably under `G:\Astro\<date>_...` (not verified): Pleiades 2023-08-21,
Bubble NGC 7635 (2023-08-20/23), IC 1396 (2023-08-22), NGC 6946 (2023-08-19), M101 (2023-08-12), NGC 7331 (2024-09),
Soul nebula (2025-03, darks only), M16 (2025-06). Ignore solar, Moon, comet, star-trail and Milky Way folders.

## 3. The test data sets

**`D:\astro_donnees_test`** (night of 2026-10-28, ~80 GB, 15 public sessions). Notes: `CATALOGUE.md`, `catalogue.csv`,
`LECTURE_PAR_LA_CHAINE.md` (what the chain reads), `PROGRESS.md`.
- Colour: stellina_ic1318, stellina_ngc891, seestar_s50_s30_m31_m42, canon6d_m51_n200, canon40da_rosette_72ed,
  canon_ra_rokinon135_northamerica, siril_asi2600mc_m8. nikon_d810a_wang, fuji_gfx50r_wang, pentax_645z_wang have
  unverified targets and unsorted calibration.
- Mono (not processed by the chain): bath_m51_lrgb, moana_m51_mono, moana_ic405_sho, lco_m101_bvrha,
  apogee_f16m_clusters_wang. Still missing: Fuji X-Trans, RASA, SCT.

**`F:\astro_essais`** (2026-10-30): 9 sets of 15-60 frames chosen evenly over each session: canon6d_m51,
canon40d_rosette, asi2600mc_m8, seestar_m31, canonra_northamerica, stellina_ic1318, stellina_ngc891, iris_a7s,
heart_a7s.
- Layout: `donnees/<set>/{lights,darks,flats,bias}`, `projets/<set>`, `scratch/<set>`, `journaux/<set>_<label>.log`,
  `runs.json`. `etat/` is ASTRO_GPU_HOME (own registry, timings, camera profiles; the Gaia cache stays in
  `~/.astro_gpu/cache`). Env ASTRO_TESTS moves the root.
- Run with `C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe`:
  - `tools/qc/jeux.py build|run|list [NAME…] [--until STAGE] [--force STAGE…] [--label L]`. `run` processes, then
    renders the 'Natural' preset as `natural_<label>`. TEST_PARAMS: star_net off, denoise 6000 iterations (24 000
    took ~33 min per set). Small-frame sets run on cuda:1 via project.json `devices`.
  - `tools/qc/bulletin.py PROJ… --render NAME [--sheet] [--out]`: stage metrics, export checks, sky flatness (8x8
    floor spread) and cast, star FWHM and elongation on the render. `--sheet` writes `<render>_planche.jpg`.
- Last report cards (2026-10-31): canon6d_m51 v4 sky spread 85 -> 27, coma stars remain; canon40d_rosette v3 centre
  too dark; seestar_m31 v3 red blotches (ha_boost); stellina_ic1318 v4 orange stars, nebulosity lost; stellina_ngc891
  v4 magenta sky; m8 and canonra blocked at the catalogues; iris_a7s OK but noisy; heart_a7s OK. Journal "Nuit du 30
  octobre": https://claude.ai/artifact/TUGeg7BYv3HAxQjFArs1DN

## 4. The QC procedure actually used

1. **Expectation and numeric success criterion written before testing** (PLAN principle 5).
2. **Quick loop** on a subset: ~100-160 frames over the night (`inventory.max_lights`), denoise 5000-6000, star_net off.
   Reliable for systematic defects (rings, holes, halos, colour shifts), not for faint detail. Other cameras: jeux.py,
   then bulletin.py.
3. **Fast tests**: `pytest` (markers gpu / network / slow); 84 fast tests passed on 2026-10-31.
4. **Read-only replay on the real projects** for every changed stage: saved values must not move (done for every change
   on 2026-10-30). Raise the stage version when its outputs change.
5. **`tools/qc/nonreg.py SOURCE COPY [--name faithful --name artistic] [--set stage.key=value]`**, on the Heart, Iris,
   Rosette and Horsehead. It robocopies the project (scratch inside COPY, never registered), renders "before" with the
   saved settings, recomputes out-of-date stages and renders "after". `nonreg.json` holds the mean and p99.9 of
   |after - before| in 8-bit levels, the share of pixels > 2 levels, and a x20 diff map. The four copies took ~115-140
   GB on 2026-10-28; put them on C: or F:.
6. **Blind sheets, if the renders differ.** `changed_zones.py BEFORE.tif AFTER.tif [6]` finds zones (>= 600 px apart).
   `blind.py PROJECT OUT SHORT label=render.png … --zones x:y …` shuffles letters A-H, key in `OUT/_keys/SHORT_key.json`.
   Sheets show his reference first, bright stars x2.5, G 8-12 stars, zones x1.5.
7. **Judges**: subagents that don't know the change, briefed with `tools/qc/qc_prompt.md`. Six criteria scored 1-5:
   bright stars, star field, colour, structure and contrast, faint detail, artefacts. PASS only if no score < 3 and
   the image would not embarrass him. They name the better candidate, where it is worse, and the top 3 defects.
8. **Your own 100 % look** (rule 5). Sheets use a noise-relative stretch: per-crop percentiles turned 0.5 sigma
   residuals into "black crosses".
9. **Decide.** Keep only if not worse anywhere; a failed change is removed, not tuned. A change that wins on some
   objects and loses on others becomes an option, off by default (precedent: halo_bound, on only for the Horsehead).
   Record the evidence in `astro_gpu/core/methods.py` (DECISIONS) and in PLAN 11.x.
10. **Before delivering after a re-run**, judge blind against the current deliverables, and archive the old ones
    (`E:\astro_gpu\archive\deliverables_<date>_<topic>`).

## 5. Engine pitfalls and lessons

**Windows and shell**
- Bash heredocs, sed and echo eat backslashes (`\\` -> `\`, `\a` = BEL, `\U` breaks Python). A .cmd became
  `C:SERSNICOLAS…` (2026-10-28) and BACKLOG got a BEL (2026-10-30). Write path-bearing files with Write/Edit; use
  forward slashes or `chr(92)` in scripts.
- Jobs started inside a session die with it (00:45-06:10 lost on 2026-10-14): detach long runs (PowerShell
  Start-Process).
- Kill by PID, never by name pattern ("canonra" also killed the jeux.py wrapper). Anchor monitor patterns ("all
  done" matched "M31 all done").
- core.autocrlf is true. E: is exFAT, so git needs `-c safe.directory=…`.
- Spawn worker pools re-run the host's `__main__` (it hung ComfyUI): guard `__main__` when blocks run inside an app.
- GUI and mockup tests set ALT3_SETTINGS to a scratch file; a test once wrote "fr" into the real user_settings.json.

**Disks** (free on 2026-10-31: C 207 GB, D 825, E 535, F 2 500, G 1 138)
- F: (T9, USB3 UASP) writes 708 MB/s and reads 368 MB/s, healthy; the note "F: slow 40-60 MB/s" is obsolete. D: is a
  USB HDD that saturates with parallel copies and downloads. I: has a bad zone; J: is failing (rescue only).
- **G: was reorganised after the 2026-10-30 memory note**: `G:\Anciennes_sauvegardes\F_backup_pc\…` and
  `I_backup_pc_LENOVO\photos` no longer exist; use section 2. Lookup: `G:\_manifeste\manifest.tsv` (a "doublon" row's
  destination is the kept copy). He says the archive is now stable. Copy from G:, never point a project at G:.
- Sizes: a variant takes ~8.6 GB from the stars stage on (2.5 GB from render_prep); render_prep of 12 Mpx ~1 GB.

**GPU and runs**
- GPUs: cuda:0 RTX 4060 Ti 16 GB; cuda:1 RTX 3070 8 GB, which drives the display. Ryzen 7 5800X, 64 GB. The GPU is
  set per project in `devices.main`.
- Never two chains on one project: the stage folder is cleared at a fresh start, and run.lock holds the PID. A running
  GUI keeps the old engine: restart it after code changes.
- `astro-gpu render --set a=1 --set b=2`: the second `--set` replaces the first.
- An npz decompresses at every access: load arrays once (variants_sheet took 20+ min on the Heart).
- On 24-frame subsets, regular coloured patches in the starless layer are drizzle coverage holes, not a bug.
- Stage times on a full project: calibrate 5-13 min, register 1-4, stack 1.5-18, catalogs seconds (cached) to 12,
  denoise ~33 (24 000 iterations), optics 2-5, stars 4-34, star_net ~40 (11 000 steps), starless ~10, render_prep
  2-4. About 2.5-3.5 h per project; Iris on 300 frames took 1 h 30.

**Reproducibility**: since 2026-10-16 two runs are bit-identical (optics 2, deterministic GPU, seeded networks, smooth
tone fit). star_net 4 (2026-10-30) trains a fixed 11 000 steps; the BACKLOG section calling it "trained for a time" is
stale. The sky model stays sensitive to larger input changes, so re-judge after every re-run.

**Catalogue services and fallbacks** (VizieR, SIMBAD and CDS were all down on 2026-10-30)
- Gaia stars: VizieR -> GAVO Heidelberg TAP (sync then async; 2 slots; a failed cone is split into 7 half-radius
  cones) -> ESA. XP spectra: GAVO (400-800 nm every 10 nm; XP sampled for G < 17.65) -> ESA.
- Names: SIMBAD -> NED. H-alpha survey: CDS hips2fits -> NSNS tiles read from simg.de (correlation 0.994-0.998).
  reproject's HiPS reader is unusable on Windows (`\` in URLs, wrong tile orientation).
- Cache `~/.astro_gpu/cache` holds 2.6 GB. Without VizieR, dense fields (M8: 10 min per 1-degree cell) and wide
  fields (135 mm: 234 cells) are impractical. An unknown scale costs 22 trials at ~1 min each: enter the focal length.
- Offline references (~40-50 GB in `~/.astro_gpu`, with a settable location) are only proposed: "Ne fait rien"
  (2026-10-31).

**Method lessons**
- A per-object fit on noisy data is the warning sign; the cure is a shared measurement cross-checked between halves.
  The spike shape is accepted only if one half's shape is found in the other at > 5 standard errors.
- Decide on measured evidence per star, never on model brightness. Never clip between tone steps. Star brightness
  across the field is not the response to diffuse light (aperture star flat for the sky rejected, 2026-10-17).

## 6. Open problems (PLAN 11.14-11.18, BACKLOG)

1. **Brightest-star glare**: Alnitak's disc, discs and shards, grey-olive discs with the learned removal (the network
   removes diffuse glare beyond the far-halo model). Leads: keep that glare with the nebula in the learned path, or
   model the far halo of dominant stars.
2. **Far-halo bound** (stars 15): fixes Iris T Cep and the Horsehead, olive discs in the Rosette: off by default.
3. **White point** set by a compact bright core (NGC 2023); exclusion rules rejected, `white_ignore_pct` exists.
4. **Colour**: Rosette OIII drowned, Heart blue-green lost, Iris dust single-hue; colour matrix poorly constrained.
5. **Faint detail**: faint dust smoothed; the denoiser biases faint structures.
6. **Star defects**: dark holes beside stars with catalogue-predicted fluxes, grey bars beside medium stars, the Iris
   salmon streak, coma stars (Canon 6D) leaving white or black blobs in the starless layer.
7. **Smooth object filling the frame** (M31, M33) biases the sky; proposed: mask an ellipse of the catalogued size.
8. **Frame selection**: the transparency rule dropped 36 % of M33; refine it, flag large exclusions in the GUI.
9. **Walking noise** (no dithering): estimate the sensor pattern per time block.
10. **Smart telescopes**: star-model residual 23-55 % in red and blue, undersampled stars (FWHM 1.7 px), not the
    drizzle. Stellina NGC 891: magenta sky.
11. **ha_boost** (R - alpha G) takes a red sky excess for emission (Seestar M31); needs the blue channel or survey.
12. **Catalogue depth** (G 18.5 / 20.5) should follow field and exposure; blind solver should give the scale first.
13. **Monochrome**: not processed (one stack per filter, then colour combination): the largest piece of work.
14. **Common tuning of the render defaults** across the sets: not done, it needs his eye.
15. **Founding principle**: per-frame deconvolution (v1, v2) no better than an unsharp mask; per-tile deconvolution of
    the stack with the field-dependent star shape proposed, not started.
16. **Background colour reference** (Mellinger; MARS is proprietary): proposed as a diagnostic on test sets first.
17. **Waiting for him**: deletion list, moving the 4 learned layers out of `D:\astro_gpu_diag`, NGC 7822 clutter.

## 7. What the user said recently

**Engine and data**
- 2026-10-28: "comfi n'était pas une bonne idée, on va revenir a un client lourd … les blocs sont là ils vont servir
  mais plus dans comfi". No GitHub for now; do not re-run the real projects.
- 2026-10-30: the test sets are for "créer des jeux de test et tuner les paramètres avec les échantillons". "On verra
  comment améliorer le fond plus tard". The ambition: "un concurrent crédible de PixInsight".
- 2026-10-30, the night's go: "prend les décisions qui te semblent les meilleures. Si tu butes utilise des agents avec
  des perfs supérieures".
- 2026-10-31 afternoon: questions on PixInsight MGC/MARS, on the H-alpha survey alone, on Mellinger and on local
  copies of the references. Then: "idéalement il faudrait le mettre dans le répertoire de l'app non ? Ne fait rien".

**GUI** (it becomes the engine's specification: "on peut utiliser ce GUI pour spécifier le moteur")
- No full auto ("il faut suggérer les paramètres"); familiar phases; options per workflow (denoise, H-alpha, star
  resynthesis, plain stack); n images become Post layers combined with masks, like Photoshop.
- "les astrophotographes aiment rarement la magie"; graph at the centre to design, image at the centre to tune.
- "on peut faire mieux que les ondelettes pour le débruitage … cf GraXpert"; "utilisable par des personnes n'ayant pas
  de GPU" (CPU alternatives, a Siril equivalent). English UI with a language option, short standard words. "qualité
  pro, sans compromis". Multi-night sessions. Missing-vs-Siril features: "note-les dans un fichier to do".

**The pilot's mandate**
- 20:54: "implémenter en autonome, déplacer les traitements dans des dossiers clean sur le T9 et tester les résultats,
  éventuellement récupérer d'autres use cases".
- 21:08: "go tu es autonome … tu es l'orchestrateur … À la fin je veux les projets prêts que je puisse réévaluer et un
  gui de qualité pro."
