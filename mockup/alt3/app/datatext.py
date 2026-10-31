"""English texts of the engine data (alt2/data/data.json), which alt2/build/prepare.py wrote in French from
alt2/build/fr.py: stage titles, parameter labels, help and reasons, the post sliders, the masks, the layers and the
object labels. Keyed exactly like fr.py; englishize() puts them in place of the French ones when the data is loaded,
and locale/fr/data.json brings the French back through _() at display time.
"""
from __future__ import annotations

import re

STAGES = {
    "inventory": "Inventory", "calibrate": "Calibration", "register": "Registration", "stack": "Stacking",
    "astrometry": "Astrometry", "catalogs": "Gaia catalogues", "surveys": "H-alpha survey", "color": "Color",
    "denoise": "Denoise", "optics": "Optics", "stars": "Star measurement", "star_net": "Neural star removal",
    "starless": "Starless image", "background": "Background", "render_prep": "Output prep",
    "render_prep_learned": "Output prep (neural removal)", "reference": "Reference image",
}

# (label, help); help only for the basic parameters
PARAMS = {
    "inventory.max_lights": ("Limit the number of frames", ""),
    "calibrate.hot_sigma": ("Hot pixel threshold", ""),
    "calibrate.flat_defect": ("Flat defect threshold", ""),
    "calibrate.hot_from_lights": ("Auto-detect hot pixels in the lights", ""),
    "calibrate.flat_scale_px": ("Flat: large scales only", ""),
    "calibrate.saturation": ("Saturation level", ""),
    "calibrate.workers": ("Parallel workers", ""),
    "register.detect_sigma": ("Star detection threshold", ""),
    "register.max_stars": ("Stars per frame", ""),
    "register.workers": ("Parallel workers", ""),
    "stack.scale": ("Drizzle scale", "Output resolution relative to the sensor. Each raw pixel is dropped at its exact "
                    "position on a finer grid: undersampled stars become rounder and smaller."),
    "stack.min_transparency": ("Minimum transparency", "Frames whose star flux falls below this fraction of the best "
                               "frames (clouds, haze, dew) are excluded."),
    "stack.max_fwhm_ratio": ("Maximum star size", "Frames whose stars exceed this multiple of the median size (wind, "
                             "focus, tracking) are excluded."),
    "stack.pixfrac": ("Pixfrac", ""),
    "stack.kappa": ("Rejection threshold", ""),
    "stack.core_tolerance": ("Star core tolerance", ""),
    "stack.half_block": ("Half-stack block", ""),
    "stack.gain": ("Sensor gain", ""),
    "stack.read_noise": ("Read noise", ""),
    "stack.pattern_removal": ("Fixed pattern removal", ""),
    "stack.pattern_frames": ("Frames for the reference pattern", ""),
    "astrometry.target": ("Object name", "The target, to find the field on the sky."),
    "astrometry.ra": ("Center: right ascension", "Empty: found from the object name, or blind."),
    "astrometry.dec": ("Center: declination", "Empty: found from the object name, or blind."),
    "astrometry.focal_mm": ("Focal length", "Focal length of the setup, to estimate the image scale."),
    "astrometry.pixel_um": ("Pixel size", ""),
    "astrometry.scale": ("Image scale", ""),
    "astrometry.blind": ("Blind solve", ""),
    "astrometry.search_offsets": ("Center search radius", ""),
    "astrometry.sip_order": ("Distortion order", ""),
    "catalogs.g_deep": ("Measured star limit", ""),
    "catalogs.g_faint": ("Modelled star limit", ""),
    "catalogs.max_faint": ("Maximum number of modelled stars", ""),
    "catalogs.xp_gmin": ("Color stars: brightest", ""),
    "catalogs.xp_gmax": ("Color stars: faintest", ""),
    "catalogs.xp_max": ("Color stars: number", ""),
    "catalogs.cell_deg": ("Query cell size", ""),
    "surveys.survey": ("Reference H-alpha survey", "Professional survey used to separate the sky background from the "
                       "nebula's emission."),
    "surveys.min_coverage": ("Minimum survey coverage", ""),
    "color.white_temp": ("White reference", "Temperature of the white reference (5778 K: the Sun)."),
    "color.aperture": ("Photometric aperture", ""),
    "color.hii_weight": ("H-alpha anchor weight", ""),
    "color.camera_profile": ("Camera profile", ""),
    "color.synthetic_weight": ("Synthetic spectra weight", ""),
    "color.profile_tolerance": ("Profile agreement", ""),
    "denoise.iterations": ("Training steps", "Number of training steps of the network on the half-stacks of this "
                           "project. Beyond 20,000, the gain becomes small."),
    "denoise.patch": ("Patch size", ""),
    "denoise.batch": ("Batch size", ""),
    "denoise.learning_rate": ("Learning rate", ""),
    "denoise.seed": ("Random seed", ""),
    "denoise.reuse_model": ("Reuse a trained network", ""),
    "optics.lateral": ("Lateral chromatic aberration", "Corrects the color shift toward the edges of the field."),
    "optics.lateral_grid": ("Color fringe grid", ""),
    "optics.psf_iterations": ("Star shape iterations", ""),
    "optics.psf_stars_per_cell": ("Stars per area for the shape", ""),
    "stars.faint_model": ("Model the faint stars", "Adds the light of the stars too faint to be measured one by one, "
                          "from the catalogue."),
    "stars.local_psf": ("Locally corrected star shape", "The star shape is measured area by area (coma, sensor "
                        "tilt)."),
    "stars.angular_shape": ("Measured star shape (spikes)", "Measures the spikes and the non-round shape of the "
                            "bright stars."),
    "stars.far_halo": ("Extended star halo", "Models the very extended glow around the bright stars."),
    "stars.color_prior": ("Color prior strength", ""),
    "stars.refine_passes": ("Position refinement passes", ""),
    "stars.solver_passes": ("Background / flux iterations", ""),
    "stars.cg_iterations": ("Solver iterations", ""),
    "stars.halo_bound": ("Far halo bounded by the data", ""),
    "star_net.enabled": ("Train the star removal network", "Trains a star removal network on the stack of this "
                         "project."),
    "star_net.steps": ("Training steps", "A fixed number: does not depend on the image size."),
    "star_net.device": ("GPU", ""),
    "star_net.batch": ("Batch size", ""),
    "star_net.seed": ("Random seed", ""),
    "star_net.glow": ("Keep a nebula centred on its star", ""),
    "starless.wiener_k": ("Denoise strength", "Denoising of the starless image, guided by the difference between the "
                          "two half-stacks."),
    "starless.sharpen": ("Sharpening network", "Trains the sharpening network used in the render."),
    "starless.mask_k": ("Unreliable core threshold", ""),
    "starless.residual_k": ("Star residual removal", ""),
    "starless.residual_scales": ("Residual scales", ""),
    "starless.nebula_gate": ("Nebula protection", ""),
    "starless.chroma_min_scale": ("Finest color scale", ""),
    "starless.lift_gmax": ("Dip repair limit", ""),
    "starless.learned_stars": ("Neural star layer", ""),
    "starless.halo_gmax": ("Optical halo limit", ""),
    "starless.sharpen_steps": ("Sharpening training steps", ""),
    "background.sky_order": ("Polynomial degree", "1: a plane; 3: a more flexible surface."),
    "background.star_flat": ("Residual flat correction (from the stars)", "Corrects the flat defects measured on the stars."),
    "background.bin_px": ("Sky sampling", ""),
    "background.large_scale_px": ("Large-scale correction", ""),
    "background.object_clip": ("Object rejection", ""),
}

# automatic reasons of the Heart project, by stage.parameter
REASONS = {
    "stack.scale": "4.0 px stars and 721 frames: undersampled image, enough frames for drizzle",
    "stack.min_transparency": "the transparency is stable: no frame falls below 60 %",
    "stack.half_block": "721 frames: blocks of 20 give two half-stacks of about 360 frames",
    "stack.gain": "sky level compared with its noise on the frames, read noise removed",
    "stack.read_noise": "noise of a single dark",
    "calibrate.workers": "16 CPU cores detected",
    "register.workers": "16 CPU cores detected",
    "astrometry.target": "object name of the project",
    "astrometry.pixel_um": "ILCE-7S camera",
    "color.aperture": "4.0 px stars × drizzle 1.5",
    "denoise.iterations": "27 Mpx stack",
    "starless.learned_stars": "star layer of the neural star removal",
    "background.large_scale_px": "14 arcmin at 4.25″ per pixel",
}
# the same reasons by their French text, for a reason found in the data under another key
REASON_TEXTS = {
    "étoiles de 4,0 px et 721 poses : image sous-échantillonnée, assez de poses pour le drizzle":
        REASONS["stack.scale"],
    "la transparence est stable : aucune pose ne tombe sous 60 %": REASONS["stack.min_transparency"],
    "721 poses : des blocs de 20 donnent deux demi-piles d'environ 360 poses": REASONS["stack.half_block"],
    "niveau du ciel comparé à son bruit sur les poses, bruit de lecture retiré": REASONS["stack.gain"],
    "bruit d'un dark seul": REASONS["stack.read_noise"],
    "16 cœurs de processeur détectés": REASONS["calibrate.workers"],
    "nom de l'objet du projet": REASONS["astrometry.target"],
    "appareil ILCE-7S": REASONS["astrometry.pixel_um"],
    "étoiles de 4,0 px × drizzle 1,5": REASONS["color.aperture"],
    "pile de 27 Mpx": REASONS["denoise.iterations"],
    "couche d'étoiles du retrait appris": REASONS["starless.learned_stars"],
    "14 minutes d'arc à 4,25″ par pixel": REASONS["background.large_scale_px"],
}

# rendering settings moved by the simplified sliders (for the expert view)
RENDER = {
    "sky_level": "Sky brightness", "lc_fine": "Fine contrast", "lc_medium": "Medium contrast",
    "lc_large": "Large-scale contrast", "chroma": "Nebula saturation", "neb_sharp": "Sharpening",
    "fine_k": "Fine detail", "faint_smooth": "Faint area smoothing", "g_lim": "Faintest star drawn",
    "d10": "Size of a G = 10 star", "d7": "Size of a G = 7 star", "star_sat": "Star color",
    "star_glow": "Bright star glow", "learned_stars": "Neural star removal",
}

# the simplified post sliders, by key: (name, group, [low end, high end])
SLIDERS = {
    "ciel": ("Sky", "Nebula and background", ["darker", "brighter"]),
    "contraste": ("Contrast", "Nebula and background", ["softer", "stronger"]),
    "saturation": ("Saturation", "Nebula and background", ["paler", "more vivid"]),
    "detail": ("Detail", "Nebula and background", ["smoother", "sharper"]),
    "bruit": ("Noise smoothing", "Nebula and background", ["less", "more"]),
    "nombre": ("Number of stars", "Stars", ["fewer", "more"]),
    "taille": ("Star size", "Stars", ["smaller", "larger"]),
    "couleur": ("Star color", "Stars", ["whiter", "more colorful"]),
    "halo": ("Bright star halo", "Stars", ["none", "wider"]),
}

# the masks made from the data, by key: (name, note); {n} is read from the French note
MASKS = {
    "etoiles": ("Stars", "Each drawn star, and the halo of the {n} stars brighter than magnitude 7.5 (radius measured "
                         "by the neural star removal)."),
    "halpha": ("H-alpha emission", "The H-alpha emission layer of the output prep, smoothed: 0 at the median, "
                                   "1 at the brightest 3 %."),
}

# the post layers of the data, by key: (name, note)
LAYERS = {
    "complet": ("Full", "Stars measured against Gaia, subtracted then redrawn; background anchored on H-alpha; "
                        "denoising on the half-stacks."),
    "complet_appris": ("Full + neural star removal", "Like Full, but the starless image comes from the network "
                                                      "trained on this stack."),
    "classique": ("Classic stack", "The stack, with the real color matrix, a simple background (smooth surface) and "
                                   "an automatic stretch. An approximation of what this workflow would give."),
    "classique_dn": ("Classic + denoise", "The stack denoised on the half-stacks, with the same simple processing. "
                                          "An approximation of what this workflow would give."),
}

# object labels of the astrometric solution, by designation (the text before " · ")
LABELS = {"IC 1805": "IC 1805 · Heart Nebula", "IC 1848": "IC 1848 · Soul Nebula",
          "NGC 896": "NGC 896 · Fish Head Nebula"}


def englishize(data: dict) -> None:
    """Replace in place the French texts of the engine data by the English ones, matched by key. A text without an
    English version is left as it is."""
    for s, st in data.get("stages", {}).items():
        st["title"] = STAGES.get(s, st.get("title", s))
        for q in st.get("params", []):
            k = f"{s}.{q['name']}"
            if k in PARAMS:
                q["label"], q["help"] = PARAMS[k]
            if q.get("reason"):
                q["reason"] = (REASONS[k] if k in REASONS and q.get("origin") == "auto"
                               else REASON_TEXTS.get(q["reason"], q["reason"]))
    for sl in data.get("sliders", []):
        if sl.get("key") in SLIDERS:
            sl["name"], sl["group"], ends = SLIDERS[sl["key"]]
            sl["ends"] = list(ends)
        sl["labels"] = {k: RENDER.get(k, v) for k, v in sl.get("labels", {}).items()}
    for k, m in data.get("masks", {}).items():
        if k in MASKS:
            name, note = MASKS[k]
            n = re.search(r"(\d+) étoiles", m.get("note", ""))
            m["name"] = name
            if "{n}" not in note or n:
                m["note"] = note.format(n=n.group(1)) if n else note
    for lay in data.get("layers", []):
        if lay.get("key") in LAYERS:
            lay["name"], lay["note"] = LAYERS[lay["key"]]
    for lb in data.get("labels", []):
        ref = lb.get("name", "").split(" · ")[0]
        if ref in LABELS:
            lb["name"] = LABELS[ref]
