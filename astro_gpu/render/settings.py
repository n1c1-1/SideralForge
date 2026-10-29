"""Rendering settings: every slider of the final adjustment tab, with its help text, section and level."""
from __future__ import annotations

from ..core.params import ADVANCED, P, StageParams

SECTIONS = ["Colour", "Sky background", "Nebula", "Highlights", "Stars"]


def S(default, *, section: str, **kw):
    f = P(default, **kw)
    f.json_schema_extra["section"] = section
    return f


class RenderSettings(StageParams):
    @classmethod
    def from_saved(cls, values: dict) -> RenderSettings:
        """Settings saved by another version: unknown (removed) settings are ignored."""
        return cls(**{k: v for k, v in (values or {}).items() if k in cls.model_fields})

    # ------------------------------------------------------------------ colour
    white_temp: float = S(5778.0, section="Colour", label="White balance", ge=3500, le=9000, step=50, unit="K",
                          help="Colour temperature rendered as neutral. The calibrated default (5778 K, the Sun) shows "
                               "stars and nebulae in their physical colours.",
                          tip="Lower = cooler (bluer) image, higher = warmer. Departing from 5778 K is an aesthetic "
                              "choice, not a correction.")
    tint: float = S(0.0, section="Colour", label="Tint", ge=-0.2, le=0.2, step=0.005,
                    help="Green / magenta balance. 0 is the calibrated value.")
    chroma: float = S(0.88, section="Colour", label="Nebula saturation", ge=0.0, le=2.0, step=0.02,
                      help="Saturation of the nebula and background layer (perceptual OKLab chroma), applied after "
                           "the tone curve. 1 = calibrated colours.",
                      tip="Values slightly below 1 look natural on screen; above 1.2 the colours become graphic.")
    chroma_protect: float = S(0.08, section="Colour", label="Shadow colour protection", level=ADVANCED, ge=0, le=0.4,
                              step=0.01, help="Darkest tones (OKLab lightness) whose saturation is not increased, "
                                              "so that colour noise in the sky stays neutral.")
    ha_boost: float = S(0.8, section="Colour", label="H-alpha boost", ge=0.0, le=4.0, step=0.05,
                        help="Adds the H-alpha emission extracted from the image (red minus scaled green) with the "
                             "colour of a real HII spectrum. Brings out faint emission without changing star colours.",
                        tip="0 = pure calibrated colour; 0.5-1 natural; 1.5+ emphasised emission.")
    ha_sigma: float = S(1.5, section="Colour", label="H-alpha boost smoothing", level=ADVANCED, ge=0, le=10,
                        step=0.1, unit="px", help="Smoothing of the extracted H-alpha before it is added.")
    oiii_boost: float = S(0.0, section="Colour", label="OIII (teal) tones", ge=0.0, le=4.0, step=0.05,
                          help="Amplifies the OIII emission (496/501 nm, teal), separated from H-alpha and starlight "
                               "with the spectral response of your camera. In emission nebulae it fills the hot "
                               "inner regions around the ionising stars; next to H-alpha it is easily drowned.",
                          tip="0 = calibrated; 1 = OIII doubled; 2-3 = HOO-like rendering. Only OIII above the noise "
                              "threshold is amplified. With a colour camera, blue reflection nebulae cannot be told "
                              "from OIII: keep 0 on reflection nebulae (Iris, Pleiades) and galaxies.")
    oiii_smooth: float = S(16.0, section="Colour", label="OIII smoothing", ge=2, le=60, step=1, unit="px",
                           help="Scale of the OIII map. Its separation from H-alpha is noisy with a colour camera: "
                                "large scales are reliable, fine ones turn into blotches.",
                           tip="12-24 px for most data; lower only with very deep integrations.")
    oiii_threshold: float = S(2.0, section="Colour", label="OIII noise threshold", level=ADVANCED, ge=0, le=6, step=0.1,
                              unit="sigma", help="OIII below this many noise sigmas (measured live on the sky at the "
                                                 "chosen smoothing) is not amplified, so that the background does not "
                                                 "turn teal.")
    resid_boost: float = S(0.0, section="Colour", label="Blue / reflection tones", ge=-1.0, le=3.0, step=0.05,
                           help="Amplifies the colour component that is neither H-alpha nor starlight nor OIII: bluer "
                                "than the stars (reflection nebulae, H-beta) or redder (dust). With a colour camera "
                                "these tones are weak and easily drowned.",
                           tip="0 = calibrated; 0.5-1.5 brings out blue reflection; it also amplifies the colour noise "
                               "of these tones, watch faint areas.")
    colour_detail: float = S(0.0, section="Colour", label="Fine colour", ge=0.0, le=1.5, step=0.05,
                             help="Restores the colour variations finer than 8 px. They are kept aside by the "
                                  "denoising and only where both half-stacks agree (real structure, not noise).",
                             tip="0 = smooth colour (cleanest); 1 = all confirmed fine colour.")
    hue_target: float = S(33.0, section="Colour", label="H-alpha hue", level=ADVANCED, ge=0, le=60, step=1, unit="deg",
                          help="OKLab hue towards which red emission is pulled (33 = deep red, lower = magenta, "
                               "higher = orange-red).")
    hue_shift: float = S(0.6, section="Colour", label="H-alpha hue strength", level=ADVANCED, ge=0, le=1, step=0.05,
                         help="How strongly reds are pulled towards the H-alpha hue (0 = untouched).")
    chroma_sigma: float = S(6.0, section="Colour", label="Faint colour smoothing", level=ADVANCED, ge=1, le=30,
                            unit="px", help="Colour of low signal-to-noise areas is taken from a smoothed version at "
                                            "this scale (luminance keeps full resolution).")
    snr_lo: float = S(3.0, section="Colour", label="Colour S/N low", level=ADVANCED, ge=0, le=50,
                      help="Below this signal-to-noise, colour comes entirely from the smoothed version.")
    snr_hi: float = S(30.0, section="Colour", label="Colour S/N high", level=ADVANCED, ge=1, le=500,
                      help="Above this signal-to-noise, colour keeps full resolution.")
    # ------------------------------------------------------------------ sky background
    sky_level: float = S(24.0, section="Sky background", label="Sky brightness", ge=0, le=80, step=0.5, unit="/255",
                         help="Display level of the typical sky (30th percentile of the field). Real skies are never "
                              "black: unresolved stars and faint emission glow.",
                         tip="18-30 looks natural on screen; lower = darker, more contrasted; higher = airy.")
    dark_level: float = S(10.0, section="Sky background", label="Dark cloud level", ge=0, le=60, step=0.5, unit="/255",
                          help="Display level of the darkest areas (1st percentile): dark nebulae, dust lanes.")
    toe: float = S(0.015, section="Sky background", label="Shadow softness", ge=0, le=0.08, step=0.001,
                   help="Below the black point the curve fades out smoothly instead of clipping, so the cores of dark "
                        "globules keep their gradation.", tip="0 = hard clip.")
    glare_nebula: float = S(1.0, section="Stars", label="Glare with the nebula", level=ADVANCED, ge=0, le=1, step=0.05,
                            help="Part of the bright stars' glare drawn with the nebula (its curve: wide, as "
                                 "photographed; it covers what the star removal leaves around them) rather than with the "
                                 "stars (their gentler curve: the glare fades out close to the star).")
    diffuse_cut: float = S(0.0, section="Sky background", label="Diffuse glow removed", ge=0, le=1, step=0.05,
                           help="The diffuse glow of the field (large-scale H-alpha, galactic light: the level on which "
                                "the structures stand, measured over a fifth of the frame) taken as background, as a "
                                "gradient removal does.",
                           tip="0 = kept with its true brightness (faithful); 1 = removed: darker backgrounds, the "
                               "nebula structures stand out.")
    sky_neutral: float = S(0.0, section="Sky background", label="Neutral sky", ge=0, le=1, step=0.05,
                           help="Makes the typical sky grey (same brightness) by an offset per colour, as a background "
                                "neutralisation does: in a field filled with faint emission, only its brighter parts "
                                "keep their colour.",
                           tip="0 = the sky keeps its measured colour (faithful); 1 = neutral grey sky.")
    sky_pct: float = S(30.0, section="Sky background", label="Sky reference", level=ADVANCED, ge=5, le=80, unit="percentile",
                       help="Percentile of the (smoothed) image defining the 'typical sky'.")
    dark_pct: float = S(1.0, section="Sky background", label="Dark reference", level=ADVANCED, ge=0.01, le=20,
                        unit="percentile", help="Percentile of the (smoothed) image defining the 'dark areas'.")
    glow_k: float = S(0.55, section="Sky background", label="Milky Way glow", ge=0.0, le=2.0, step=0.05,
                      help="Light of the stars too faint to be drawn individually, added back as a smooth glow (with "
                           "the Gaia colours). Without it, dense star fields look unnaturally empty.")
    glow_sigma: float = S(30.0, section="Sky background", label="Glow smoothness", level=ADVANCED, ge=1, le=100,
                          unit="px", help="Spatial smoothing of the faint star glow.")
    g_glow: float = S(16.0, section="Sky background", label="Glow / drawn star limit", level=ADVANCED, ge=10, le=21,
                      step=0.5, unit="G mag", help="Stars fainter than this become glow; brighter ones are drawn.")
    speck_k: float = S(2.0, section="Sky background", label="Dark speck removal", level=ADVANCED, ge=0, le=10, step=0.5,
                       help="Isolated dark pixels deeper than this many noise sigmas below their 5x5 median are lifted.")
    # ------------------------------------------------------------------ nebula
    white_mult: float = S(0.62, section="Nebula", label="White point", ge=0.1, le=2.0, step=0.01,
                          help="White point of the tone curve relative to the brightest nebula. Lower = brighter "
                               "nebula (highlights are then compressed, never clipped).")
    white_ignore_pct: float = S(0.005, section="Nebula", label="Brightest area ignored", level=ADVANCED, ge=0.001,
                                le=0.5, step=0.001, unit="%",
                                help="The white point is measured on the nebula once this share of the image, the very "
                                     "brightest, is left out. Raise it when a small, very bright object (a compact "
                                     "core, a reflection nebula around its star) sets the white point and darkens the "
                                     "whole image.")
    hdr_strength: float = S(0.8, section="Nebula", label="Large-scale compression", ge=0, le=1.0, step=0.05,
                            help="Compresses the brightness of large bright areas (edge-preserving) so that the core "
                                 "of the nebula keeps its detail next to faint extensions.")
    hdr_knee: float = S(0.48, section="Nebula", label="Compression start", level=ADVANCED, ge=0.1, le=1.0, step=0.01,
                        help="Brightness above which the large-scale compression starts.")
    hdr_r: int = S(40, section="Nebula", label="Compression scale", level=ADVANCED, ge=4, le=200, unit="px",
                   help="Radius of the edge-preserving filter of the large-scale compression.")
    hdr_eps: float = S(0.02, section="Nebula", label="Compression edge sensitivity", level=ADVANCED, ge=0.0001, le=0.5,
                       help="Lower values follow edges more closely.")
    lc_fine: float = S(1.05, section="Nebula", label="Fine contrast", ge=0.5, le=2.5, step=0.05,
                       help="Local contrast at the 2 px scale (texture).")
    lc_medium: float = S(1.3, section="Nebula", label="Medium contrast", ge=0.5, le=2.5, step=0.05,
                         help="Local contrast at the 8 px scale (filaments).")
    lc_large: float = S(1.25, section="Nebula", label="Large contrast", ge=0.5, le=2.5, step=0.05,
                        help="Local contrast at the 32 px scale (structures, depth).")
    neb_sharp: float = S(0.9, section="Nebula", label="Sharpening", ge=0, le=3, step=0.05,
                         help="Sharpening of the small scales, applied only in bright emission (high S/N) and never "
                              "near white.")
    sharp_white: float = S(1.0, section="Nebula", label="Sharpening limit", level=ADVANCED, ge=0.3, le=1.0, step=0.01,
                           help="Brightness above which sharpening fades out.")
    sharpen: float = S(0.0, section="Nebula", label="Sharpening (measured star shape)", ge=0.0, le=2.0, step=0.05,
                       help="Amount of the non-blind sharpening: the star shape measured on your stack, undone towards "
                            "a sharper target by a network trained on your own half-stacks. Real detail in bright "
                            "zones; in faint zones it can also sharpen the noise, so it starts above a brightness.",
                       tip="0 = none (faithful); 1 = as learned; above 1 = accentuated.")
    sharpen_lo_pct: float = S(50.0, section="Nebula", label="Sharpening start", level=ADVANCED, ge=0, le=99, step=1,
                              unit="percentile", help="Emission brightness (percentile) where the sharpening starts.")
    fine_k: float = S(1.0, section="Nebula", label="Fine detail", ge=0.0, le=3.0, step=0.05,
                      help="Strength of the finest scales of the denoised nebula (filaments, edges, measured by the "
                           "Noise2Noise network on your data), above the brightness set by 'Fine detail start'.",
                      tip="1 = as measured; above 1 = accentuated; 0 = smoothest.")
    fine_lo_pct: float = S(80.0, section="Nebula", label="Fine detail start", ge=0, le=99, step=1, unit="percentile",
                           help="Emission brightness (percentile of the field) where the detail injection starts. "
                                "Lower it to bring the fine detail into fainter nebulosity.",
                           tip="80 = bright emission only; 40-60 = most of the nebula, with more denoiser texture.")
    fine_hi_pct: float = S(95.0, section="Nebula", label="Fine detail full", level=ADVANCED, ge=1, le=100, step=1,
                           unit="percentile", help="Emission brightness where the detail injection is complete.")
    faint_smooth: float = S(32.0, section="Nebula", label="Faint area smoothing", ge=0, le=100, step=1, unit="px",
                            help="In faint areas, structures smaller than this are not confirmed by the data (noise "
                                 "and star residuals) and are smoothed; bright nebulosity keeps full resolution.",
                            tip="0 = no smoothing (sharpest, more mottle); 32 = clean faint background.")
    faint_lo_pct: float = S(55.0, section="Nebula", label="Faint area limit", level=ADVANCED, ge=0, le=99, step=1,
                            unit="percentile", help="Below this emission percentile the faint-area smoothing is complete.")
    faint_hi_pct: float = S(85.0, section="Nebula", label="Bright area limit", level=ADVANCED, ge=1, le=100, step=1,
                            unit="percentile", help="Above this emission percentile the image keeps full resolution.")
    # ------------------------------------------------------------------ highlights
    hl_knee: float = S(0.7, section="Highlights", label="Highlight knee", ge=0.3, le=0.95, step=0.01,
                       help="Brightness above which highlights are compressed by a logarithmic shoulder (with local "
                            "detail re-injected): bright cores keep their gradations instead of burning out.")
    hl_max: float = S(0.9, section="Highlights", label="Highlight ceiling", ge=0.6, le=1.0, step=0.01,
                      help="Brightness reached by the brightest nebula area.")
    hl_detail: float = S(0.3, section="Highlights", label="Highlight detail", ge=0, le=1.5, step=0.05,
                         help="Extra local detail restored where the highlights are compressed.")
    hl_dmin: float = S(0.5, section="Highlights", label="Minimum detail", level=ADVANCED, ge=0, le=1, step=0.05,
                       help="Floor of the local detail kept in the most compressed areas.")
    hl_r1: int = S(12, section="Highlights", label="Detail scale 1", level=ADVANCED, ge=2, le=100, unit="px",
                   help="First radius of the edge-preserving base layer.")
    hl_r2: int = S(40, section="Highlights", label="Detail scale 2", level=ADVANCED, ge=2, le=300, unit="px",
                   help="Second radius of the edge-preserving base layer.")
    hl_eps: float = S(0.01, section="Highlights", label="Detail edge sensitivity", level=ADVANCED, ge=0.0001, le=0.5,
                      help="Edge sensitivity of the base layer.")
    hl_shoulder: float = S(0.92, section="Highlights", label="Final shoulder", level=ADVANCED, ge=0.5, le=1.0, step=0.01,
                           help="Above this value a soft (tanh) roll-off replaces any hard clip.")
    hl_keep_chroma: float = S(0.0, section="Highlights", label="Keep colour in highlights", level=ADVANCED, ge=0, le=1,
                              step=0.05, help="Out-of-gamut bright reds: 0 = desaturate towards white, 1 = darken "
                                              "and keep the hue.")
    # ------------------------------------------------------------------ stars
    stars: bool = S(True, section="Stars", label="Show stars",
                    help="Combine the star layer with the nebula (screen blend). Off = starless image.")
    real_stars: bool = S(True, section="Stars", label="Saturated stars as photographed",
                         help="Stars with a saturated core are shown with their real light - halo and diffraction "
                              "spikes included - instead of being drawn; every other star gets its measured shape "
                              "(diamond, spikes) on top of its drawn round core. Off = every star drawn round, no "
                              "spikes.")
    learned_stars: bool = S(True, section="Stars", label="Learned star removal",
                            help="The stars removed by the network trained on this project's own stack instead of the "
                                 "star model: no rings, dark discs, residual rays or smooth discs around the bright "
                                 "stars, and the nebula under them as photographed. Needs the preparation 'Rendering "
                                 "preparation, learned star removal' (when the project has a learned star layer; "
                                 "without one the star model is used). On by default: first in blind checks on the "
                                 "Rosette, the Iris and the Flaming Star.",
                            tip="Switching loads the other set of layers (a few seconds the first time).")
    stars_photographed: float = S(0.0, section="Stars", label="Stars as photographed", ge=0, le=1, step=0.05,
                                  help="With the learned star removal: every star shown by its own light (the image "
                                       "minus the nebula), as photographed, instead of drawn from the catalogue. In "
                                       "between, a blend. The star size settings then no longer apply.",
                                  tip="Your stars as you shot them: real sizes, colours and spikes, with the softness "
                                      "and noise of the data. Needs 'Learned star removal'.")
    spike_photo: float = S(1.0, section="Stars", label="Photographed spikes", ge=0, le=1, step=0.05,
                           help="Diffraction spikes of the bright stars: 0 = drawn with the spike shape measured once "
                                "for all the stars; 1 = each star's own spikes, as photographed (measured on the image, "
                                "the nebula interpolated across them; the stars along them stay drawn). In between, a "
                                "blend. Needs a rendering preparation of version 41 or later.",
                           tip="The shared shape never matches each star's own spikes: beads, light bars across the "
                               "nebula and dark blobs along the spikes of the brightest stars. Photographed spikes "
                               "avoid them; on very faint spikes they show as faint dashed fragments (see the next "
                               "setting).")
    spike_g_max: float = S(12.0, section="Stars", label="Photographed spikes down to", ge=4, le=12, step=0.25,
                           unit="G mag", help="Only the stars brighter than this get their photographed spikes; the "
                                              "fainter ones keep the drawn shape.",
                           tip="12 = every spike the preparation measured; 10 = only the clearly visible ones (the "
                               "spikes of G 10-12 stars are faint: photographed, they show as faint dashed fragments "
                               "in a dark sky).")
    spike_smooth: float = S(1.0, section="Stars", label="Spike smoothing", level=ADVANCED, ge=0, le=1, step=0.05,
                            help="The light of the photographed spikes averaged along each spike (sharp across it): "
                                 "takes the noise of the image out of the faint spikes. 0 = raw light.")
    g_lim: float = S(15.0, section="Stars", label="Faintest drawn star", ge=8, le=20, step=0.25, unit="G mag",
                     help="Gaia magnitude of the faintest star drawn with a clearly visible peak.",
                     tip="Lower = fewer, more prominent stars; higher = denser star field.")
    t_lim: float = S(0.2, section="Stars", label="Faint star brightness", ge=0.02, le=0.8, step=0.01,
                     help="Display peak of a star at the faintest drawn magnitude.")
    d10: float = S(4.4, section="Stars", label="Size of a G=10 star", ge=1.5, le=15, step=0.1, unit="px",
                   help="Visible diameter of a magnitude-10 star (pixels at sensor scale).")
    d7: float = S(9.0, section="Stars", label="Size of a G=7 star", ge=2, le=40, step=0.1, unit="px",
                  help="Visible diameter of a magnitude-7 star; together with the G=10 size it sets how fast star "
                       "size grows with brightness.")
    star_glow: float = S(0.3, section="Stars", label="Bright star glow", ge=0, le=1, step=0.05,
                         help="The glare the optics spread around the stars (measured on the image and taken out of "
                              "the nebula) is drawn back with the stars as a glow: the fraction of it shown.",
                         tip="0 = stars on a clean sky; 1 = the whole glare, as photographed (around a very bright "
                             "star it veils a large part of the field).")
    star_sat: float = S(0.7, section="Stars", label="Star colour", ge=0, le=2, step=0.05,
                        help="Saturation of star colours (measured, or from Gaia for saturated and faint stars).")
    sat_min: float = S(0.3, section="Stars", label="Faint star colour", level=ADVANCED, ge=0, le=1, step=0.05,
                       help="Relative saturation kept by the faintest drawn stars.")
    core_white: float = S(0.6, section="Stars", label="White cores", ge=0, le=1, step=0.05,
                          help="Bright star cores turn white like on film (0 = keep the colour up to the core).")
    t9: float = S(0.95, section="Stars", label="Bright star peak", level=ADVANCED, ge=0.5, le=1.0, step=0.01,
                  help="Display peak of a magnitude-9 star.")
    bright_halo_desat: float = S(0.0, section="Stars", label="Halo colour removal", level=ADVANCED, ge=0, le=1, step=0.05,
                                 help="Removes the colour of the background around the brightest stars (G < 9), where "
                                      "residual optical halos tint the sky.")


PRESETS: dict[str, dict] = {
    "Natural": {},
    # The two deliverables of every project. Faithful = the rendering of the reference image of 2026-10-10 10:34 (the
    # defaults: the user's preferred rendering; measured colours without the H-alpha hue convention came out magenta
    # and were not what the data look like to the eye). Artistic = the same base with a stronger stretch, local
    # contrast and colour, judged by eye. Neither desaturates the halos (that produced grey discs around bright stars).
    # (Tried 2026-10-14: the user's own slider settings on the Rosette as Artistic - the independent judge found dark
    # rings around the stars, olive reflection nebulae and crushed faint dust on the Horsehead.)
    "Faithful": {},
    "Artistic": {"sharpen": 1.0, "sharpen_lo_pct": 60.0, "fine_k": 1.3, "fine_lo_pct": 60.0, "colour_detail": 1.0,
                 "resid_boost": 0.3, "chroma": 1.1, "neb_sharp": 1.2, "lc_fine": 1.15, "lc_medium": 1.4,
                 "lc_large": 1.5, "sky_level": 34.0, "dark_level": 11.0, "white_mult": 0.8, "star_sat": 0.8,
                 "sat_min": 0.15, "d10": 5.0, "d7": 12.0},
    "H-alpha emphasised": {"ha_boost": 1.6, "white_mult": 0.58},
    "Prominent stars": {"g_lim": 14.0, "t_lim": 0.28, "d10": 5.4, "star_sat": 0.6, "d7": 11.5},
    "Dense Milky Way": {"g_glow": 17.0, "g_lim": 16.0, "t_lim": 0.18, "d10": 3.8, "d7": 8.0},
    "Starless": {"stars": False},
    "Rich detail & colour": {"fine_k": 1.3, "fine_lo_pct": 50.0, "faint_smooth": 12.0,
                             "colour_detail": 0.8, "resid_boost": 0.5, "chroma": 0.95},
    "Emission nebula, OIII": {"fine_k": 1.3, "fine_lo_pct": 50.0, "faint_smooth": 12.0,
                              "colour_detail": 0.8, "oiii_boost": 1.5, "chroma": 0.95},
}
