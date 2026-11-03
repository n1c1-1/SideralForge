# L'empilement sur le jeu de données varié (rapport automatique)

Généré le 03/11/2026 à 23:00 par `tools/bench/denoise_dataset.py report`, à partir de `stack.json`, des journaux de la nuit et des mesures de chaque projet (`F:\astro_essais\dataset_projets`).

| Jeu | Appareil | Poses | Niveau | État | Minutes | aligned | align. RMS px | FWHM px | frames used | rejected % | grid | bad px % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 05_dslr_refractor_canon40da_m42 | Canon EOS 40Da | 134 | toutes | fait | 4.7 | 134 | 0.246 | 4.45 | 134 | 0.238 | 3908, 2602 | 0.33 |
| 08_dslr_refractor_canon60da_cocoon | Canon EOS 60Da | 25 | toutes | fait | 3.2 | 25 | 0.142 | 6.2 | 25 | 0.006 | 5202, 3464 | 0.84 |
| 18_osc_newton_asi533mc_dobson12_m42 | ZWO ASI533MC Pro (IMX533, co | 15 | toutes | **échec** : fewer than 2 usable frames after selection; relax the selection thresholds | 1.4 | 1 | 0.0 | 10.42 |  |  |  | 0.06 |
| 30_smart_seestar_s50_m51 | ZWO Seestar S50 (IMX462) | 150 | toutes | **échec** : too few stars detected per frame (median 28); lower the detection threshold or check the focus / exposure | 0.3 |  |  |  |  |  |  | 0.0 |
| 31_smart_seestar_s30_m51 | ZWO Seestar S30 (IMX662) | 150 | toutes | fait | 1.4 | 146 | 0.386 | 2.94 | 146 | 0.002 | 1620, 2880 | 0.0 |
| 32_smart_seestar_s50_m16 | ZWO Seestar S50 (IMX462) | 116 | toutes | fait | 1.2 | 116 | 0.232 | 3.66 | 116 | 0.014 | 1620, 2880 | 0.0 |
| 37_smart_celestron_origin_m51 | Celestron Origin built-in ca | 41 | toutes | fait | 1.3 | 40 | 0.939 | 5.49 | 40 | 0.002 | 3056, 2048 | 0.88 |
| 38_smart_dwarfmini_ngc869 | DWARF Mini tele camera (Sony | 45 | toutes | fait | 0.7 | 45 | 0.328 | 3.89 | 45 | 0.005 | 1920, 1080 | 9.12 |
| 44_dslr_refractor_canon40da_horsehead | Canon EOS 40Da | 63 | toutes | fait | 1.7 | 63 | 0.3 | 6.21 | 63 | 0.023 | 3908, 2602 | 0.52 |

## Avertissements et erreurs, par jeu

**30_smart_seestar_s50_m51**
- warning (calibrate): no dark frames: using the black level as dark (hot pixels will not be removed by calibration)
- warning (calibrate): no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of the vignetting)
- error (register): too few stars detected per frame (median 20); lower the detection threshold or check the focus / exposure
- error (register): too few stars detected per frame (median 28); lower the detection threshold or check the focus / exposure

**31_smart_seestar_s30_m51**
- warning (calibrate): no dark frames: using the black level as dark (hot pixels will not be removed by calibration)
- warning (calibrate): no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of the vignetting)
- warning (register): 4 frame(s) could not be aligned and will be ignored, e.g. F:\dataset_astro\data\31_smart_seestar_s30_m51\lights\Light_M 51_10.0s_IRCUT_20250327-215224.fit

**32_smart_seestar_s50_m16**
- warning (calibrate): no dark frames: using the black level as dark (hot pixels will not be removed by calibration)
- warning (calibrate): no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of the vignetting)

**37_smart_celestron_origin_m51**
- warning (register): 1 frame(s) could not be aligned and will be ignored, e.g. F:\dataset_astro\data\37_smart_celestron_origin_m51\lights\Light0008.fits

