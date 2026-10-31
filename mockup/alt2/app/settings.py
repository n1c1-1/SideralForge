"""The settings of each step, sorted by use rather than by engine module, and how one setting is shown.

Two kinds of settings:
* "engine": an existing setting of the engine (101 in all), with its value on Heart, its suggestion and reason;
* "new": a setting the engine does not have yet, shown marked "à créer dans le moteur" so that the mockup doubles as
  the list of the engine work. Its value is kept like the others, so that changing it marks the step to recompute.

Levels: "normal" (always shown), "expert" (expert mode), "system" (speed, memory, reproducibility: kept apart).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from model import TRUNK, PathDef, Session
from widgets import ACCENT, WARN, label

NEW = "#c9b37e"                     # colour of the settings to create in the engine


def E(stage, name, level="normal", label=None, help=None, unit=None):
    return dict(kind="engine", stage=stage, name=name, level=level, label=label, help=help, unit=unit)


def N(stage, key, label, typ, default, help, level="normal", choices=None, lo=None, hi=None, step=None, unit="",
      reason=""):
    return dict(kind="new", stage=stage, name="new_" + key, level=level, label=label, help=help, type=typ,
                default=default, choices=choices, min=lo, max=hi, step=step, unit=unit, reason=reason)


AS_TODAY = "comme le moteur aujourd'hui"
CATALOG = {
    "calibration": [
        ("Masters", [
            N("calibrate", "darks_combine", "Combinaison des darks", "choice", "Moyenne avec rejet",
              "Comment les darks sont fusionnés en un master.",
              choices=["Moyenne avec rejet", "Médiane", "Moyenne"], reason=AS_TODAY),
            N("calibrate", "dark_scaling", "Mise à l'échelle des darks", "bool", False,
              "Ajuste le signal thermique du master à chaque pose : utile quand la température a changé pendant la "
              "nuit ou que les darks ont été faits à une autre température."),
        ]),
        ("Pixels défectueux", [
            E("calibrate", "hot_sigma", help="Un pixel plus chaud que ce seuil (en écarts au bruit) est exclu ; le "
                                             "drizzle comble le trou avec les autres poses."),
            E("calibrate", "hot_from_lights", help="Cherche aussi les pixels chauds dans les poses elles-mêmes, pas "
                                                   "seulement dans les darks."),
            N("calibrate", "cold_pixels", "Corriger aussi les pixels froids", "bool", False,
              "Exclut les pixels trop sombres (morts) de la même façon que les pixels chauds."),
        ]),
        ("Flats", [
            E("calibrate", "flat_defect", help="Un défaut du flat plus profond que ce seuil (poussière, rayure) est "
                                               "corrigé."),
            E("calibrate", "flat_scale_px", "expert", help="Ne garde du flat que les grandes échelles (le "
                                                           "vignetage)."),
        ]),
        ("Capteur", [
            E("calibrate", "saturation", "expert", help="Fraction du blanc au-dessus de laquelle un pixel est "
                                                        "considéré saturé."),
            E("inventory", "max_lights", "expert", help="Ne prend que les N premières poses ; 0 = toutes."),
        ]),
        ("Système", [E("calibrate", "workers", "system")]),
    ],
    "alignement": [
        ("Référence", [
            N("register", "reference", "Pose de référence", "choice", "Automatique : la plus nette",
              "La pose sur laquelle toutes les autres sont alignées. Choisie dans les sources : la pose affichée "
              "dans la visionneuse.", choices=["Automatique : la plus nette", "Choisie dans les sources"],
              reason=AS_TODAY),
        ]),
        ("Étoiles détectées", [
            E("register", "detect_sigma", help="Une étoile doit dépasser ce seuil (en écarts au bruit du fond) pour "
                                               "servir à l'alignement."),
            E("register", "max_stars", "expert", help="Les plus brillantes, au plus ce nombre par pose."),
        ]),
        ("Déformation", [
            N("register", "transform", "Modèle de déformation", "choice", "Automatique",
              "Comment une pose est déformée pour se superposer à la référence. Les polynômes suivent la distorsion "
              "d'un objectif grand champ.",
              choices=["Automatique", "Rotation et translation", "Homographie", "Polynôme de degré 2",
                       "Polynôme de degré 3"], reason=AS_TODAY),
        ]),
        ("Système", [E("register", "workers", "system")]),
    ],
    "selection": [
        ("Autres critères", [
            N("stack", "max_background", "Fond de ciel maximal", "float", 0.0,
              "Exclut les poses au fond trop clair (Lune, aube, phares), en multiple du fond médian ; 0 = pas de "
              "seuil.", lo=0.0, hi=5.0, step=0.1, unit="× médiane"),
            N("stack", "min_roundness", "Rondeur minimale des étoiles", "float", 0.0,
              "Exclut les poses aux étoiles allongées (vent, suivi) ; 1 = parfaitement rondes, 0 = pas de seuil.",
              lo=0.0, hi=1.0, step=0.05),
            N("stack", "keep_best", "Garder les meilleures", "int", 100,
              "Après les seuils, ne garde que ce pourcentage des meilleures poses.", lo=10, hi=100, step=5,
              unit="% des poses"),
        ]),
        ("Pondération", [
            N("stack", "weighting", "Poids des poses", "choice", "Transparence et taille des étoiles",
              "Combien chaque pose compte dans la pile.",
              choices=["Transparence et taille des étoiles", "Rapport signal sur bruit", "Égal pour toutes"],
              reason=AS_TODAY),
        ]),
    ],
    "empilement": [
        ("Résolution", [
            E("stack", "scale"),
            E("stack", "pixfrac", help="La taille de la goutte que chaque pixel brut dépose sur la grille fine."),
        ]),
        ("Rejet des pixels aberrants", [
            N("stack", "rejection", "Méthode de rejet", "choice", "Automatique : autour de la médiane",
              "Comment les pixels aberrants (satellites, avions, rayons cosmiques) sont écartés.",
              choices=["Automatique : autour de la médiane", "Sigma clipping", "Sigma clipping winsorisé",
                       "Ajustement linéaire", "ESD généralisé"], reason=AS_TODAY),
            E("stack", "kappa", help="Un pixel plus loin de la médiane que ce seuil (en écarts au bruit) est rejeté."),
            N("stack", "split_thresholds", "Seuils bas et haut séparés", "bool", False,
              "Par défaut le même seuil vaut pour les pixels trop clairs et trop sombres. Séparés, on peut rejeter "
              "plus fort les traînées claires."),
            E("stack", "core_tolerance", "expert", help="Marge laissée au cœur des étoiles brillantes, qui varie "
                                                        "d'une pose à l'autre avec la turbulence."),
        ]),
        ("Normalisation", [
            N("stack", "normalization", "Normalisation des poses", "choice", "Automatique : niveau et échelle",
              "Remet toutes les poses au même niveau de fond et à la même échelle avant de les combiner, malgré la "
              "Lune ou la brume qui changent.",
              choices=["Automatique : niveau et échelle", "Niveau seulement", "Aucune"], reason=AS_TODAY),
        ]),
        ("Capteur", [
            E("stack", "gain", "expert", help="Électrons par unité numérique : sert au modèle de bruit."),
            E("stack", "read_noise", "expert", help="Bruit de lecture du capteur, en unités numériques."),
        ]),
        ("Motif fixe du capteur", [
            E("stack", "pattern_removal", "expert", help="Retire un motif fixe estimé sur les poses (bandes, "
                                                         "stries)."),
            E("stack", "pattern_frames", "expert", help="Nombre de poses utilisées pour estimer ce motif."),
        ]),
    ],
    "demi_piles": [
        ("Demi-piles", [
            E("stack", "half_block", "expert", help="Les poses sont réparties entre les deux demi-piles par blocs de "
                                                    "ce nombre, pour que chacune couvre toute la nuit."),
        ]),
    ],
    "astrometrie": [
        ("Où regarde l'image", [
            E("astrometry", "target"), E("astrometry", "ra"), E("astrometry", "dec"), E("astrometry", "focal_mm"),
        ]),
        ("Résolution du champ", [
            E("astrometry", "sip_order", "expert", help="Degré du polynôme qui décrit la distorsion de l'optique."),
            E("astrometry", "blind", "expert", help="Trouve le champ sans connaître le centre ni l'objet."),
            E("astrometry", "search_offsets", "expert", help="Jusqu'où chercher autour du centre annoncé, en "
                                                             "fraction du champ."),
            E("astrometry", "pixel_um", "expert", help="Taille des pixels du capteur ; vide = lue dans les "
                                                       "fichiers."),
            E("astrometry", "scale", "expert", help="Secondes d'arc par pixel ; vide = calculée."),
        ]),
        ("Système", [E("catalogs", "cell_deg", "system", help="Taille des morceaux de ciel demandés au catalogue "
                                                              "Gaia.")]),
    ],
    "couleur": [
        ("Référence du blanc", [
            N("color", "white_ref", "Référence du blanc", "choice", "Le Soleil",
              "Ce qui doit apparaître blanc.",
              choices=["Le Soleil", "Moyenne des étoiles du champ", "Galaxie spirale moyenne",
                       "Température au choix"], reason=AS_TODAY),
            E("color", "white_temp", label="Température du blanc",
              help="La température de cette référence ; 5778 K est celle du Soleil."),
            N("color", "neutral_zone", "Zone de neutralisation du fond", "choice", "Automatique : sans nébuleuse",
              "Où le fond du ciel est rendu gris neutre.",
              choices=["Automatique : sans nébuleuse", "Dessinée sur l'image"], reason=AS_TODAY),
        ]),
        ("Mesure de la couleur", [
            E("color", "aperture", "expert", help="Rayon de la mesure de la lumière de chaque étoile."),
            E("color", "camera_profile", "expert", help="Utilise la réponse mesurée de ton capteur plutôt qu'une "
                                                        "réponse type."),
            E("color", "synthetic_weight", "expert", help="Poids des spectres Gaia reconstruits par rapport aux "
                                                          "spectres mesurés."),
            E("color", "profile_tolerance", "expert", help="Écart admis entre le profil mesuré et le profil type."),
            E("color", "hii_weight", "expert", help="Poids donné à la raie H-alpha dans l'ajustement de la "
                                                    "couleur."),
        ]),
        ("Étoiles de couleur (Gaia)", [
            E("catalogs", "xp_gmin", "expert", help="Magnitude de la plus brillante étoile utilisée pour la "
                                                    "couleur."),
            E("catalogs", "xp_gmax", "expert", help="Magnitude de la plus faible."),
            E("catalogs", "xp_max", "expert", help="Nombre maximal d'étoiles utilisées."),
        ]),
    ],
    "debruitage": [
        ("Débruitage", [
            N("denoise", "strength", "Force", "int", 100,
              "Mélange avec la pile d'origine : 100 % garde tout le débruitage, 70 % laisse un peu de grain "
              "naturel.", lo=0, hi=100, step=5, unit="%", reason=AS_TODAY),
            E("denoise", "iterations"),
        ]),
        ("Apprentissage", [
            E("denoise", "patch", "expert", help="Taille des morceaux d'image sur lesquels le réseau apprend."),
            E("denoise", "learning_rate", "expert", help="Pas de l'apprentissage : trop grand, il diverge ; trop "
                                                         "petit, il n'avance pas."),
            E("denoise", "reuse_model", "expert", help="Partir d'un réseau déjà entraîné (fichier) au lieu de "
                                                       "repartir de zéro."),
        ]),
        ("Système", [E("denoise", "batch", "system"), E("denoise", "seed", "system")]),
    ],
    "etoiles": [
        ("Quelles étoiles", [
            E("catalogs", "g_deep", label="Étoiles mesurées jusqu'à la magnitude",
              help="Plus la magnitude est grande, plus il y a d'étoiles mesurées une à une, et plus c'est long.",
              unit="G"),
            E("stars", "faint_model"),
            E("catalogs", "g_faint", "expert", help="Les étoiles plus faibles que la limite des mesurées, jusqu'à "
                                                    "celle-ci, sont modélisées d'après le catalogue.", unit="G"),
            E("catalogs", "max_faint", "expert", help="Au plus ce nombre d'étoiles modélisées."),
        ]),
        ("Forme des étoiles", [
            E("stars", "local_psf"), E("stars", "angular_shape"), E("stars", "far_halo"), E("optics", "lateral"),
            E("stars", "halo_bound", "expert", help="Le halo lointain ne peut pas dépasser ce que les données "
                                                    "montrent."),
            E("optics", "lateral_grid", "expert", help="Finesse de la carte des franges colorées."),
            E("optics", "psf_iterations", "expert", help="Itérations de l'ajustement de la forme des étoiles."),
            E("optics", "psf_stars_per_cell", "expert", help="Étoiles utilisées par zone pour mesurer la forme."),
        ]),
        ("Ajustement", [
            E("stars", "color_prior", "expert", help="Combien la couleur de chaque étoile doit suivre Gaia."),
            E("stars", "refine_passes", "expert", help="Passes d'affinage de la position des étoiles."),
            E("stars", "solver_passes", "expert", help="Allers-retours entre l'estimation du fond et celle du flux "
                                                       "des étoiles."),
            E("stars", "cg_iterations", "expert", help="Itérations du solveur à chaque passe."),
        ]),
    ],
    "appris": [
        ("Apprentissage", [
            E("star_net", "steps"),
            E("star_net", "glow", help="Garde la nébuleuse qui entoure une étoile au lieu de la prendre pour sa "
                                       "lumière."),
        ]),
        ("Système", [E("star_net", "device", "system", help="Sur quelle carte graphique apprendre."),
                     E("star_net", "batch", "system"), E("star_net", "seed", "system")]),
    ],
    "sans_etoiles": [
        ("Débruitage", [
            E("starless", "wiener_k"),
            E("starless", "chroma_min_scale", "expert", help="La couleur n'est pas débruitée en dessous de cette "
                                                             "échelle."),
        ]),
        ("Résidus d'étoiles", [
            E("starless", "residual_k", help="Force de la suppression de ce qui reste des étoiles après "
                                             "soustraction."),
            E("starless", "residual_scales", "expert", help="Nombre d'échelles où les résidus sont cherchés."),
            E("starless", "mask_k", "expert", help="Au-dessus de ce seuil, le cœur d'une étoile est jugé peu "
                                                   "fiable et reconstruit."),
            E("starless", "halo_gmax", "expert", help="Les étoiles plus brillantes que cette magnitude ont leur "
                                                      "halo optique retiré."),
            E("starless", "lift_gmax", "expert", help="Les creux laissés sous les étoiles plus brillantes que cette "
                                                      "magnitude sont réparés."),
        ]),
        ("Nébuleuse", [
            E("starless", "nebula_gate", help="Protège la nébuleuse des corrections faites autour des étoiles ; 1 = "
                                              "protection maximale."),
        ]),
        ("Accentuation", [
            E("starless", "sharpen"),
            E("starless", "sharpen_steps", "expert", help="Pas d'apprentissage du réseau d'accentuation."),
        ]),
    ],
    "fond:halpha": [
        ("Relevé de référence", [
            E("surveys", "survey"),
            E("surveys", "min_coverage", "expert", help="Part du champ que le relevé doit couvrir pour être "
                                                        "utilisé."),
        ]),
        ("Dégradé du ciel", [
            E("background", "sky_order"), E("background", "star_flat"),
            E("background", "bin_px", "expert", help="Taille des cases où le fond est mesuré."),
            E("background", "large_scale_px", "expert", help="Échelle au-delà de laquelle le fond est corrigé."),
            E("background", "object_clip", "expert", help="Les zones plus claires que ce seuil ne servent pas à "
                                                          "mesurer le fond."),
        ]),
        ("Échantillons", [
            N("background", "exclusions", "Zones à exclure", "zone", "aucune",
              "Les zones dessinées ne servent pas à mesurer le fond : nébuleuse faible, galaxie, reflet."),
            N("background", "show_samples", "Points d'échantillonnage visibles", "bool", False,
              "Montre sur l'image les points où le fond est mesuré, et permet d'en déplacer ou d'en retirer."),
        ]),
    ],
    "fond:simple": [
        ("Dégradé du ciel", [
            N("fond_simple", "degree", "Degré de la surface", "int", 2,
              "1 : un plan incliné ; 2 ou 3 : une surface plus souple, pour les grands champs.", lo=1, hi=5, step=1),
            N("fond_simple", "samples", "Échantillons par côté", "int", 16,
              "Nombre de points de mesure du fond sur chaque côté de l'image.", lo=4, hi=64, step=2),
        ]),
        ("Échantillons", [
            N("fond_simple", "exclusions", "Zones à exclure", "zone", "aucune",
              "Les zones dessinées ne servent pas à mesurer le fond : nébuleuse faible, galaxie, reflet."),
            N("fond_simple", "show_samples", "Points d'échantillonnage visibles", "bool", False,
              "Montre sur l'image les points où le fond est mesuré, et permet d'en déplacer ou d'en retirer."),
        ]),
    ],
    "post": [
        ("Étoiles au rendu", [
            E("@render", "star_guard", label="Garde contre les artefacts d'étoiles",
              help="Repère les étoiles redessinées qui ne collent pas à l'image et les corrige."),
            E("@render", "guard_sigma", "expert", label="Tolérance de la garde"),
            E("@render", "real_g", "expert", label="Aigrettes mesurées jusqu'à la magnitude", unit="G"),
            E("@render", "spike_snr", "expert", label="Aigrettes gardées à partir d'un signal sur bruit de"),
        ]),
        ("Bords de l'image", [
            E("@render", "cover_min", "expert", label="Couverture minimale des bords",
              help="Les bords couverts par moins de cette part des poses sont recadrés."),
            E("@render", "edge_frac", "expert", label="Propreté des bords"),
            E("@render", "min_exposure", "expert", label="Exposition effective minimale"),
        ]),
        ("Détails et couleur", [
            E("@render", "fine_sigma", "expert", label="Échelle des détails fins", unit="px"),
            E("@render", "cont_smooth", "expert", label="Lissage du fond sous la lumière des étoiles", unit="px"),
            E("@render", "resid_smooth", "expert", label="Lissage de la couleur résiduelle", unit="px"),
            E("@render", "n2n_base", "expert", label="Débruitage appris dans la base"),
            E("@render", "n2n_scale", "expert", label="Débruitage appris jusqu'à l'échelle", unit="px"),
            E("@render", "diffuse_scale", "expert", label="Échelle de la lueur diffuse"),
            E("@render", "oiii_scale", "expert", label="Lissage de base de la carte OIII", unit="px"),
        ]),
    ],
}
# the 32 usage tips of the engine, in French
TIPS = {
    ("astrometry", "target"): "Inutile si le centre est donné ou si les fichiers FITS contiennent le pointage.",
    ("astrometry", "focal_mm"): "Lue dans les fichiers quand l'appareil l'enregistre ; les télescopes ne la donnent "
                                "en général pas. Vide, la résolution cherche l'échelle (quelques minutes).",
    ("astrometry", "scale"): "206,265 × taille des pixels (µm) / focale (mm), divisé par l'échelle du drizzle.",
    ("astrometry", "blind"): "Marche pour les champs de plus d'environ 1° ; la première fois, deux catalogues "
                             "d'étoiles brillantes sont téléchargés.",
    ("astrometry", "sip_order"): "3 pour les lunettes avec réducteur et les grands champs ; 2 pour les petits champs "
                                 "; 0 = plan tangent pur.",
    ("background", "sky_order"): "2 ou 3 pour la plupart des champs ; plus pour les très grands champs aux dégradés "
                                 "complexes. Trop haut, il peut absorber une grande nébuleuse faible quand aucun "
                                 "relevé H-alpha ne sert de référence.",
    ("calibrate", "hot_sigma"): "Les valeurs basses (0,5 à 1) excluent plus de pixels ; avec beaucoup de poses cela "
                                "ne coûte rien. Monte-le si plus de 5 % environ des pixels sont exclus.",
    ("calibrate", "flat_defect"): "0,3 = 30 %. Baisse-le si des poussières restent visibles après la calibration.",
    ("calibrate", "hot_from_lights"): "Ignoré tout seul quand le ciel n'a pas bougé sur le capteur pendant la nuit.",
    ("calibrate", "flat_scale_px"): "Environ 40 px retire les poussières et garde le vignetage.",
    ("calibrate", "saturation"): "Garde la valeur par défaut, sauf si le cœur des étoiles paraît écrêté ou si des "
                                 "anneaux apparaissent autour des étoiles brillantes.",
    ("calibrate", "workers"): "Par défaut, le nombre de cœurs du processeur moins deux (12 au plus).",
    ("catalogs", "g_faint"): "Égale à la limite des étoiles mesurées, elle désactive la modélisation des étoiles "
                             "faibles.",
    ("color", "aperture"): "Environ deux fois la taille des étoiles ; plus grande, elle est plus sûre par mauvaise "
                           "turbulence, mais plus bruitée.",
    ("color", "white_temp"): "C'est la calibration physique ; le rendu peut encore réchauffer ou refroidir l'image "
                             "sans y toucher.",
    ("denoise", "iterations"): "Plus de pas donnent un débruitage un peu meilleur ; le gain plafonne vers 20 000. "
                               "L'apprentissage est sauvegardé tous les 1000 pas et reprend après une interruption.",
    ("inventory", "max_lights"): "Utile pour un essai rapide de toute la chaîne ; garde 0 pour le vrai calcul.",
    ("register", "detect_sigma"): "Baisse-le (5) pour des poses très courtes ou très faibles ; monte-le dans les "
                                  "champs très denses de la Voie lactée pour ne garder que des étoiles propres.",
    ("stack", "scale"): "1,5 pour la plupart des montages avec des dizaines de poses ; 2 seulement avec beaucoup de "
                        "poses bien décalées et de petites étoiles ; 1 quand les étoiles sont grosses (plus de 5 px) "
                        "ou les poses peu nombreuses. Plus haut coûte de la mémoire et ajoute du bruit.",
    ("stack", "pixfrac"): "Plus petite, l'image est plus fine mais plus bruitée, avec un risque de trous ; 0,6 à 0,8 "
                          "est la plage habituelle.",
    ("stack", "min_transparency"): "0 garde tout (les poses voilées reçoivent quand même un poids faible). Monte-le "
                                   "si les poses voilées apportent des dégradés ou des halos gonflés.",
    ("stack", "kappa"): "Plus bas (3 à 4), il retire plus de traînées, mais peut entamer le cœur des étoiles par "
                        "mauvaise turbulence.",
    ("stack", "pattern_removal"): "Désactivé par défaut : sur la Rosette, l'estimation contenait aussi les traces "
                                  "des étoiles faibles, et la soustraire dessinait les stries à retirer. Neutre sur "
                                  "Heart.",
    ("star_net", "steps"): "11 000 pas prennent environ 32 minutes sur une RTX 4060 Ti. Beaucoup moins laisse des "
                           "halos faibles.",
    ("star_net", "glow"): "Activé, le cœur de NGC 2023 est gardé (désactivé, le réseau le prenait pour la lumière "
                          "de son étoile). Garde aussi une lueur ronde d'environ 40 px autour de l'étoile centrale "
                          "de l'Iris.",
    ("starless", "wiener_k"): "1 est la valeur neutre au sens statistique. Plus haut = plus lisse, plus bas = plus "
                              "de grain.",
    ("stars", "color_prior"): "Petit, les couleurs suivent Gaia de près et évitent les ajustements ratés (points "
                              "verts, trous noirs) ; grand, on fait plus confiance à la mesure.",
    ("stars", "local_psf"): "Nécessaire quand la forme des étoiles change vite dans le champ (coma d'un Newton, "
                            "courbure de champ) : sinon chaque étoile laisse un anneau dans l'image sans étoiles.",
    ("stars", "angular_shape"): "Les aigrettes quittent alors la nébuleuse avec les étoiles et sont redessinées "
                                "avec elles.",
    ("stars", "far_halo"): "La lueur quitte alors la nébuleuse avec les étoiles ; les étoiles redessinées ne la "
                           "portent pas.",
    ("surveys", "survey"): "« auto » prend NSNS (plus fin) quand il couvre le champ, sinon la carte de tout le ciel "
                           "de Finkbeiner. « aucun » pour les champs sans émission (galaxies, amas) : le fond est "
                           "alors un polynôme lisse.",
}
CHOICES = {("surveys", "survey"): [("auto", "auto"), ("nsns", "NSNS"), ("finkbeiner", "Finkbeiner"),
                                   ("none", "aucun")],
           ("star_net", "device"): [("auto", "auto"), ("main", "principale"), ("aux", "secondaire")]}
UNITS = {"x median": "× médiane", "frames": "poses", "deg": "°", "sigma": "σ", "x": "×", "G mag": "G",
         "arcsec/px": "″/px", "e-/ADU": "e⁻/ADU", "um": "µm", "x field": "× champ", "x noise": "× bruit",
         "x sky noise": "× bruit du ciel", "x white level": "× niveau du blanc"}
# where each stage is computed, as the chain names it
STAGE_STEP = {"inventory": "Calibration", "calibrate": "Calibration", "register": "Alignement", "stack": "Empilement",
              "astrometry": "Astrométrie", "catalogs": "Astrométrie", "surveys": "Astrométrie", "color": "Couleur",
              "denoise": "Débruitage", "optics": "Mesure des étoiles", "stars": "Mesure des étoiles",
              "star_net": "Retrait appris des étoiles", "starless": "Image sans étoiles", "background": "Fond",
              "fond_simple": "Fond"}
LEVEL_TEXT = {"normal": "normal", "expert": "expert", "system": "système"}


# ======================================================================== resolving a setting for a workflow
def catalog_key(p: PathDef, step: str) -> str:
    return f"fond:{p.opts['fond']}" if step == "fond" else step


def render_stage(s: Session, p: PathDef) -> str | None:
    return next((x for x in s.path_stages(p) if x in ("render_prep", "render_prep_learned")), None)


def resolve(s: Session, p: PathDef, step: str, it: dict) -> dict | None:
    """The setting as shown: label, help, tip, value, suggestion, reason, bounds; None when not in this workflow."""
    stage = it["stage"]
    if stage == "@render":
        stage = render_stage(s, p)
        if stage is None:
            return None
    if stage not in TRUNK and stage not in s.path_stages(p):
        return None
    target = s.trunk_params if stage in TRUNK else p.params
    stored = target.get(stage, {})
    r = dict(it, stage=stage, cat_stage=it["stage"], step_key=step, key=f"{stage}.{it['name']}")
    if it["kind"] == "engine":
        q = next((x for x in s.stages[stage]["params"] if x["name"] == it["name"]), None)
        if q is None:
            return None
        r.update(label=it["label"] or q["label"], help=it["help"] or q["help"], suggested=q["value"],
                 reason=q["reason"] if q["origin"] == "auto" else ("réglé par toi dans le projet" if q["origin"] ==
                                                                    "user" else "valeur par défaut du moteur"),
                 min=q["min"], max=q["max"], step=q["step"], tip=TIPS.get((stage, it["name"]), ""),
                 unit=it["unit"] if it["unit"] is not None else UNITS.get(q["unit"], q["unit"]),
                 choices=CHOICES.get((stage, it["name"])),
                 type=_type_of(q["value"], q, CHOICES.get((stage, it["name"]))))
    else:
        r.update(suggested=it["default"], tip="", reason=it.get("reason") or "valeur proposée",
                 choices=[(c, c) for c in it["choices"]] if it["choices"] else None)
    r["value"] = stored.get(it["name"], r["suggested"])
    r["changed"] = it["name"] in stored
    where = STAGE_STEP.get(stage)
    r["where"] = where if where and where not in ("Fond",) and _step_title(step) != where and step != "post" else ""
    return r


def _type_of(v, q, choices):
    if choices:
        return "choice"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if q["name"] in ("ra", "dec", "focal_mm", "pixel_um", "scale"):
        return "optfloat"
    return "str"


def _step_title(step):
    from steps import BY_KEY
    return BY_KEY[step]["title"] if step in BY_KEY else step


def step_settings(s: Session, p: PathDef, step: str):
    """[(group title, [resolved settings])] for one step of the workflow."""
    out = []
    for title, items in CATALOG.get(catalog_key(p, step), []):
        rs = [r for r in (resolve(s, p, step, it) for it in items) if r is not None]
        if rs:
            out.append((title, rs))
    return out


def all_settings(s: Session, p: PathDef, steps_obj):
    """Every setting of the workflow, in chain order: [(step title, group, resolved)]."""
    rows = []
    for b in steps_obj.steps_of(p):
        for g, rs in step_settings(s, p, b["key"]):
            for r in rs:
                rows.append((steps_obj.title(p, b), g, r))
    return rows


def fmt(v, typ=None, choices=None) -> str:
    if choices:
        return dict(choices).get(v, str(v))
    if isinstance(v, bool):
        return "oui" if v else "non"
    if v is None or v == "":
        return "auto" if typ == "optfloat" else "—"
    if isinstance(v, float):
        if abs(v) >= 1000 or v == int(v) and abs(v) >= 100:
            return f"{v:,.0f}".replace(",", " ")
        t = f"{v:.4g}"
        return t.replace(".", ",")
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ") if abs(v) >= 10000 else str(v)
    return str(v)


def parse(text: str, r: dict):
    """A value typed as text (table or text view), or raises ValueError."""
    t = text.strip()
    typ = r["type"]
    if r.get("choices"):
        for val, shown in r["choices"]:
            if t.lower() in (str(val).lower(), shown.lower()):
                return val
        raise ValueError(f"choix possibles : {', '.join(s for _, s in r['choices'])}")
    if typ == "bool":
        if t.lower() in ("oui", "true", "1", "vrai"):
            return True
        if t.lower() in ("non", "false", "0", "faux"):
            return False
        raise ValueError("oui ou non")
    if typ == "zone":
        raise ValueError("se dessine sur l'image")
    if typ in ("int", "float", "optfloat"):
        if typ == "optfloat" and t.lower() in ("", "auto", "none"):
            return None
        x = float(t.replace(" ", "").replace(",", "."))
        if r.get("min") is not None and x < r["min"] or r.get("max") is not None and x > r["max"]:
            raise ValueError(f"hors de la plage {fmt(r.get('min'))} à {fmt(r.get('max'))}")
        return int(round(x)) if typ == "int" else x
    return "" if t == "—" else t


# ======================================================================== how one setting is shown
class RangeBar(QWidget):
    """The range of a numeric setting: the suggestion (tick) and the current value (dot)."""

    def __init__(self, lo, hi, sugg, cur, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.sugg, self.cur = float(lo), float(hi), sugg, cur
        self.log = self.lo > 0 and self.hi / self.lo > 100
        self.setFixedHeight(22)
        self.setMinimumWidth(240)
        self.setMaximumWidth(380)

    def _x(self, v, x0, x1):
        if v is None:
            return None
        v = min(max(float(v), self.lo), self.hi)
        if self.log:
            t = (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        else:
            t = (v - self.lo) / ((self.hi - self.lo) or 1)
        return x0 + t * (x1 - x0)

    def paintEvent(self, e):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            f = QFont(self.font())
            f.setPointSizeF(8)
            p.setFont(f)
            m = p.fontMetrics()
            a, b = fmt(self.lo), fmt(self.hi)
            x0 = m.horizontalAdvance(a) + 8
            x1 = self.width() - m.horizontalAdvance(b) - 8
            y = 11
            p.setPen(QColor("#6b7280"))
            p.drawText(QRectF(0, 0, x0 - 4, 22), Qt.AlignRight | Qt.AlignVCenter, a)
            p.drawText(QRectF(x1 + 4, 0, self.width() - x1, 22), Qt.AlignLeft | Qt.AlignVCenter, b)
            p.setPen(QPen(QColor("#353b47"), 3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            xs = self._x(self.sugg, x0, x1)
            if xs is not None:
                p.setPen(QPen(QColor(ACCENT), 2))
                p.drawLine(QPointF(xs, y - 7), QPointF(xs, y + 7))
            xc = self._x(self.cur, x0, x1)
            if xc is not None:
                changed = self.cur != self.sugg
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(WARN if changed else "#e8ebf0"))
                p.drawEllipse(QPointF(xc, y), 4.5, 4.5)
        finally:
            p.end()


class SettingRow(QFrame):
    """One setting, complete: editor, range with suggestion and value, reason, use, tip, where it is computed."""
    changed = Signal(dict, object)                  # resolved setting, new value

    def __init__(self, r: dict, parent=None):
        super().__init__(parent)
        self.r = r
        self.setObjectName("set")
        self.setStyleSheet("QFrame#set { background: transparent; } QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 8)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        name = QLabel(r["label"])
        name.setStyleSheet("font-weight: 600;")
        top.addWidget(name)
        if r["kind"] == "new":
            badge = QLabel("à créer dans le moteur")
            badge.setStyleSheet(f"color: {NEW}; border: 1px dashed {NEW}; border-radius: 8px; padding: 0 6px; "
                                "font-size: 8.5pt;")
            badge.setToolTip("Ce réglage n'existe pas encore dans le moteur : la maquette le montre pour le "
                             "proposer.")
            top.addWidget(badge)
        if r["level"] != "normal":
            lv = QLabel(LEVEL_TEXT[r["level"]])
            lv.setStyleSheet("color: #8b93a1; border: 1px solid #3a4150; border-radius: 8px; padding: 0 6px; "
                             "font-size: 8.5pt;")
            top.addWidget(lv)
        top.addStretch(1)
        top.addWidget(self._editor())
        if r.get("unit"):
            top.addWidget(label(r["unit"], "small"))
        if r["changed"]:
            rb = QPushButton("↺")
            rb.setFixedSize(24, 22)
            rb.setStyleSheet("padding: 0;")
            rb.setToolTip(f"Revenir à la suggestion : {fmt(r['suggested'], r['type'], r.get('choices'))}")
            rb.clicked.connect(lambda: self.changed.emit(self.r, self.r["suggested"]))
            top.addWidget(rb)
        v.addLayout(top)
        numeric = r["type"] in ("int", "float") and r.get("min") is not None and r.get("max") is not None
        if numeric:
            row = QHBoxLayout()
            row.addWidget(RangeBar(r["min"], r["max"], r["suggested"], r["value"]))
            row.addStretch(1)
            v.addLayout(row)
        sug = fmt(r["suggested"], r["type"], r.get("choices"))
        if r["changed"]:
            v.addWidget(label(f"modifié · suggéré : {sug} ({r['reason']})", "warn", wrap=True))
        else:
            v.addWidget(label(f"suggéré : {sug} · {r['reason']}", "reason", wrap=True))
        if r.get("help"):
            v.addWidget(label(r["help"], "small", wrap=True))
        if r.get("tip"):
            t = label(f"Conseil : {r['tip']}", "small", wrap=True)
            t.setStyleSheet("color: #b9c2cf; font-size: 9pt;")
            v.addWidget(t)
        if r.get("where"):
            v.addWidget(label(f"Calculé à l'étape {r['where']} : le changer la recalcule aussi.", "small",
                              wrap=True))

    def _editor(self):
        r = self.r
        typ, cur = r["type"], r["value"]
        if r.get("choices"):
            e = QComboBox()
            for val, shown in r["choices"]:
                e.addItem(shown, val)
            i = e.findData(cur)
            e.setCurrentIndex(max(i, 0))
            e.setMinimumWidth(220)
            e.currentIndexChanged.connect(lambda i: self.changed.emit(self.r, e.itemData(i)))
            return e
        if typ == "bool":
            e = QCheckBox()
            e.setChecked(bool(cur))
            e.toggled.connect(lambda x: self.changed.emit(self.r, x))
            return e
        if typ == "zone":
            e = QPushButton("Dessiner sur l'image…")
            e.clicked.connect(lambda: QMessageBox.information(
                self.window(), "Maquette", "Dans le vrai client : la pile s'ouvre en grand, on dessine les zones à "
                                           "exclure au pinceau ou au lasso, et le fond est recalculé sans elles."))
            return e
        if typ == "int":
            e = QSpinBox()
            e.setRange(int(r["min"]) if r.get("min") is not None else -10 ** 9,
                       int(r["max"]) if r.get("max") is not None else 10 ** 9)
            if r.get("step"):
                e.setSingleStep(int(r["step"]))
            e.setValue(int(cur))
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda: e.value() != self.r["value"] and self.changed.emit(self.r, e.value()))
            return e
        if typ == "float":
            e = QDoubleSpinBox()
            v = float(r["suggested"] or 0)
            step = r.get("step") or (10 ** (math.floor(math.log10(abs(v))) - 1) if v else 0.1)
            dec = max(0, int(-math.floor(math.log10(step)))) if step < 1 else 0
            e.setDecimals(min(dec + 1, 5))
            e.setRange(r["min"] if r.get("min") is not None else -1e9, r["max"] if r.get("max") is not None else 1e9)
            e.setSingleStep(step)
            e.setValue(float(cur))
            e.setFixedWidth(110)
            e.editingFinished.connect(lambda: abs(e.value() - float(self.r["value"])) > 1e-12
                                      and self.changed.emit(self.r, round(e.value(), 6)))
            return e
        e = QLineEdit("" if cur is None else str(cur))
        e.setFixedWidth(160)
        e.setPlaceholderText("auto")

        def done():
            t = e.text().strip()
            try:
                val = parse(t, self.r) if typ == "optfloat" else (t or None)
            except ValueError:
                return
            if val != self.r["value"]:
                self.changed.emit(self.r, val)
        e.editingFinished.connect(done)
        return e
