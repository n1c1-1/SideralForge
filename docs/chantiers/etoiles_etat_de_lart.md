# Chantier étoiles : l'état de l'art (retrait des étoiles, deux calques, recombinaison)

Recherche du 4 novembre 2026. Chaque affirmation est suivie de sa source. Les sources **informelles** (forums, blogs,
critiques d'amateurs) sont marquées *(informel)*. Ce que je n'ai pas pu lire moi-même est signalé.

**Ce qui n'a pas pu être ouvert :** les sous-pages du site officiel de StarNet (documentation, FAQ, référence de la
ligne de commande) renvoient toutes la page d'accueil ([starnetastro.com](https://www.starnetastro.com/)) ; les fils
CloudyNights et Stargazers Lounge refusent la lecture automatique ; les forums AstroBin ne s'affichent pas sans
navigateur (je n'en ai que les extraits du moteur de recherche) ; le fil de Russell Croman sur le « unscreen » (forum
PixInsight, 1er juin 2022) n'est connu que par les messages qui le citent. Pour combler, j'ai lu le **code source** des
scripts Siril qui pilotent ces outils, qui dit précisément ce qu'ils font.

Mots utilisés : **étirer** = passer l'image linéaire à une image visible (courbe non linéaire) ; **profil d'étoile**
(PSF) = la forme d'une étoile ponctuelle dans l'image ; **réseau** = réseau de neurones entraîné ; **remplir** = inventer
le fond sous un masque (« inpainting »).

## 1. Le retrait des étoiles : les outils

### StarNet (Nikita / Mikita Misiura)

- **Version 1 (2019), code ouvert.** Réseau convolutif « encodeur-décodeur à résidus », entraîné avec trois critères à
  la fois : écart pixel à pixel, un second réseau qui juge si le résultat a l'air vrai (« adversaire »), et une
  ressemblance de texture (« perceptuelle ») ; il s'inspire de pix2pix
  ([GitHub nekitmm/starnet](https://github.com/nekitmm/starnet), [pix2pix, arXiv:1611.07004](https://arxiv.org/abs/1611.07004)).
- **Données d'entraînement (v1) :** des paires image / même image sans étoiles, faites par l'auteur sur une lunette
  (FSQ106 + QSI 683) ; le jeu est privé ; l'auteur prévient que la qualité ne dépassera jamais celle des images sans
  étoiles du jeu ([GitHub](https://github.com/nekitmm/starnet)). **Pour la v2, je n'ai trouvé aucune description des
  données d'entraînement.**
- **Entrée (v1) :** image **étirée**, TIFF 8 bits par canal ; les images très retouchées (réduction d'étoiles,
  déconvolution, redimensionnement) marchent moins bien ([GitHub](https://github.com/nekitmm/starnet)).
- **Défauts connus (v1, dits par l'auteur) :** longues aigrettes de télescope à miroir mal traitées, très grosses
  étoiles laissées, petites étoiles laissées dans les champs chargés (repasser le résultat aide), artefacts à retoucher
  à la main ([GitHub](https://github.com/nekitmm/starnet)).
- **Version 2 (24 janvier 2022), code fermé, gratuite**
  ([Galactic Hunter](https://www.galactic-hunter.com/post/starnet2) *(informel)*,
  [doc Siril 1.2](https://siril.readthedocs.io/en/1.2/processing/stars/starnet.html)). Versions actuelles : ligne de
  commande 2.6.2 (Linux, Windows, Mac), module PixInsight 2.6.1, greffons Photoshop 1.0.4 ; la 2.6.2 améliore le
  traitement des FITS linéaires et les calques d'étoiles « différence » et « unscreen »
  ([starnetastro.com](https://www.starnetastro.com/)).
- **Linéaire ou étiré (v2) :** le réseau lui-même ne travaille pas sur du linéaire ; l'option « linéaire » étire
  provisoirement l'image, retire les étoiles, puis défait l'étirement (message de l'auteur, lu via extrait de recherche :
  [CloudyNights](https://www.cloudynights.com/forums/topic/808556-starnet-v2/) *(informel)* ;
  [Astronomy, avril 2023](https://www.astronomy.com/observing/how-to-remove-stars-from-images-with-ai-tools/), qui
  déconseille l'option). Siril fait de même (étirement automatique puis inverse)
  ([doc Siril 1.2](https://siril.readthedocs.io/en/1.2/processing/stars/starnet.html)).
- **Réglages :** « stride » (taille des tuiles), 256 par défaut, à ne pas changer selon l'auteur ; « agrandir ×2 » pour
  les étoiles très fines, qui multiplie le temps par 4 et peut gêner sur les très grosses étoiles
  ([doc Siril 1.2](https://siril.readthedocs.io/en/1.2/processing/stars/starnet.html)).
- **Défauts rapportés (v2) :** encore des artefacts là où étaient les étoiles brillantes
  ([Galactic Hunter](https://www.galactic-hunter.com/post/starnet2) *(informel)*) ; « artefacts fréquents, halos
  occasionnels », peut grignoter la nébulosité faible
  ([Formastro](https://formastro.com/en/blog/syqon-starless-retrait-etoiles-ia) *(informel)*) ; meilleur que
  StarXTerminator sur les petites étoiles, moins bon sur les grosses et les galaxies
  ([Evan Tsai, notes de 2021](https://www.astroimagetw.com/en/tutorials/starless-tools-comparison/) *(informel)*).
- **Ligne de commande : oui.** Le script Siril appelle `starnet2 -i entrée.tif -o sortie.tif [-s stride] [-u]
  [-m masque.tif] [-n etoiles_unscreen.tif] [-d]` (`-d` coupe la protection des hautes lumières, 2.5.3+) ; moteur ONNX
  Runtime ([StarNet.py, siril-scripts](https://gitlab.com/free-astro/siril-scripts/-/blob/main/StarNetAstro/StarNet.py)).
- **Vitesse :** 55 s en v2 contre 1 min 55 en v1 sur une même image (PixInsight, matériel inconnu)
  ([Stargazers Lounge](https://stargazerslounge.com/topic/390040-starnet-v2/) *(informel, lu via extrait)*) ;
  3 à 11 min pour 30 Mpx ([Formastro](https://formastro.com/en/blog/syqon-starless-retrait-etoiles-ia) *(informel)*).

### StarXTerminator (Russell Croman, RC Astro, commercial)

- **Principe :** réseau convolutif d'architecture propre, entraîné sur des images « des objectifs photo jusqu'au
  télescope James Webb » ; aucun réglage ([page produit](https://www.rc-astro.com/software/sxt/)) ; la version 11 de
  l'IA a été entraînée en partie sur Hubble et Webb
  ([Astronomy](https://www.astronomy.com/observing/how-to-remove-stars-from-images-with-ai-tools/)).
- **Linéaire :** à passer le plus tôt possible, idéalement juste après l'empilement ; le réseau est entraîné sur des
  images étirées par une courbe des tons moyens (MTF) ; sur du linéaire il étire en interne puis défait ; les
  étirements arcsinh, GHS, masqués ou HDR faits **avant** le dégradent (les étoiles ressemblent alors à de petites
  galaxies) ([notes d'usage RC Astro](https://rc-astro.com/starxterminator-usage-notes)).
- **Limites dites par l'éditeur :** fortes aberrations (coma, courbure de champ) ; étoiles de plus de 8 px de largeur à
  mi-hauteur (FWHM) — conseil : diviser l'image par 2 au-delà de 6 px
  ([FAQ RC Astro](https://www.rc-astro.com/faq/why-does-starxterminator-fail-to-remove-stars)).
- **Historique utile :** l'IA 6 ajoute une « imitation du bruit » pour que la zone remplie ne soit pas lisse ; les IA 9
  et 10 corrigent des carrés noirs ou blancs au bord ; l'IA 11 garde mieux le détail près des étoiles brillantes
  ([page produit](https://www.rc-astro.com/software/sxt/)).
- **Défauts rapportés :** grosses étoiles saturées et halo pris pour de la nébulosité (Dentelles du Cygne)
  ([AstroBin](https://app.astrobin.com/forum/topic/236748/starxterminator-struggles-with-saturated-stars-in-ngc-6960-processing)
  *(informel)*) ; filaments et petits nœuds effacés
  ([AstroBin](https://ssr.app.astrobin.com/forum/topic/188236/russell-croman-astrophotography-starxterminator/starxterminator-mistakenly-removing-nebulosity)
  *(informel)*).
- **Sorties :** image sans étoiles + image d'étoiles (« différence » = original − sans étoiles, option « unscreen »)
  ; hôtes : PixInsight, Photoshop, Affinity, **ligne de commande** `rc-astro sxt -o … --overlap 0.2 --difference
  [--unscreen]`, calcul sur GPU ou processeur
  ([StarXTerminator.py, siril-scripts](https://gitlab.com/free-astro/siril-scripts/-/blob/main/RC-Astro/StarXTerminator.py)).
- **Usage scientifique :** un article A&A 2026 l'a utilisé pour retirer les étoiles d'images Spitzer, avec un contrôle
  de biais (voir § 3) ([Casassus et al., arXiv:2602.11356](https://arxiv.org/abs/2602.11356)).

### Cosmic Clarity « Dark Star » (Seti Astro)

- **Existe bien :** « Dark Star V2.1c » daté du 29 juin 2025, annoncé comme retrait d'étoiles
  ([page Cosmic Clarity](https://www.setiastro.com/cosmic-clarity)). Le dépôt séparé Cosmic Clarity est archivé depuis
  le 16 septembre 2026 ([GitHub cosmicclarity](https://github.com/setiastro/cosmicclarity)) ; l'outil vit dans Seti
  Astro Suite Pro ([README SASpro](https://github.com/setiastro/setiastrosuitepro/blob/main/README.md)).
- **Ce que dit le code :** réseau de type NAFNet (une architecture publiée de restauration d'image,
  [arXiv:2204.04676](https://arxiv.org/abs/2204.04676)), un modèle mono et un modèle couleur ; si la médiane de
  l'image est sous 0,125, elle est étirée vers une médiane de 0,25 puis désétirée après ; calque d'étoiles optionnel,
  en mode « additif » (original − sans étoiles) ou « unscreen » ((original − sans étoiles) / (1 − sans étoiles))
  ([darkstar_engine.py](https://github.com/setiastro/setiastrosuitepro/blob/main/src/setiastro/saspro/cosmicclarity_engines/darkstar_engine.py)).
- **Ligne de commande : oui** (`--star_removal_mode additive|unscreen`, `--show_extracted_stars`, `--chunk_size 512`,
  `--overlap`, `--disable_gpu`)
  ([CosmicClarity_Darkstar.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/processing/CosmicClarity_Darkstar.py)).
- **Non trouvé :** données d'entraînement, défauts connus, comparaisons publiées.

### SyQon (Zenith, gratuit dans Siril ; Axiom, commercial)

- Zenith est dans les scripts officiels de Siril depuis janvier 2026 ; il travaille « largement à la résolution native »
  et prend environ deux fois plus de temps que les modèles rapides
  ([annonce siril.org](https://siril.org/2026/01/a-brand-new-star-removal-script-comes-to-siril-zenith/)). Le code
  montre lui aussi une variante NAFNet et un étirement provisoire pour les données linéaires
  ([Starless.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/SyQon/Starless.py)).
- Débruite dans la même passe ; peut prendre la nébulosité très faible pour du bruit
  ([Formastro](https://formastro.com/en/blog/syqon-starless-retrait-etoiles-ia) *(informel)*).

### DeepStarRemoval (DeepSkyColors, PixInsight)

- Réseau entraîné à **remplir** le fond sous les étoiles, présenté comme non génératif ; marche en linéaire et en
  étiré, couleur et gris ; le calque d'étoiles contient « exactement la lumière retirée » ; entraîné sur les images de
  l'auteur et des images du domaine public ; GPU si disponible ; pas de ligne de commande mentionnée
  ([DeepSkyColors](https://www.deepskycolors.com/pixinsight/deepstarremoval/)).

### Siril, PixInsight, GraXpert

- **Siril** : depuis Siril 1.4, StarNet passe par un script Python ; l'ancienne interface intégrée est retirée
  ([doc Siril](https://siril.readthedocs.io/en/latest/processing/stars/starnet.html)). Le script étire avec l'étirement
  automatique de Siril, canal par canal, puis applique l'inverse approché ; il fournit le masque de StarNet, l'unscreen
  de StarNet, ou deux calques calculés par Siril : soustraction et « descreen »
  ([StarNet.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/StarNetAstro/StarNet.py)). Siril pilote aussi
  StarXTerminator, SyQon et Dark Star par scripts (mêmes liens ci-dessus).
- **Siril « Resynthèse complète »** (`synthstar`) : refait chaque étoile détectée en profil rond (Moffat ou gaussien)
  aux mêmes intensité, largeur, teinte et saturation ; le résultat est un calque d'étoiles **de synthèse** à recombiner
  avec une image sans étoiles ([doc Siril](https://siril.readthedocs.io/en/latest/processing/stars/resynthesis.html)).
- **PixInsight** : StarNet (v1 portée en module natif, puis StarNet2), StarXTerminator, DeepStarRemoval
  ([Astronomy](https://www.astronomy.com/observing/how-to-remove-stars-from-images-with-ai-tools/)) ;
  **PhotometricGlowSubtraction** (DeepSkyColors) : mesure le profil des étoiles sur l'image, utilise les magnitudes et
  couleurs Gaia DR3 pour retirer le **halo** de chaque étoile ; image linéaire et astrométrée ; le cœur des étoiles est
  laissé ; anneaux saturés exclus de la mesure ; sortie optionnelle de ce qui a été retiré
  ([doc](https://www.deepskycolors.com/astro/docs/tools/PhotometricGlowSubtraction/PhotometricGlowSubtraction.html)).
- **GraXpert** : rien sur les étoiles ; ses fonctions sont l'extraction du fond et le débruitage
  ([GitHub GraXpert](https://github.com/Steffenhir/GraXpert)).

### Les méthodes publiées en astronomie

- **Photométrie par profil d'étoile, puis soustraction.** DAOPHOT (Stetson 1987) est la référence
  ([PASP 99, 191](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S)) ; photutils en est une mise en œuvre :
  les étoiles ajustées sont retirées, de nouvelles étoiles plus faibles sont cherchées dans le reste, jusqu'à ce qu'il
  n'y en ait plus ; une bonne image de reste « ressemble à du bruit »
  ([photutils, PSF](https://docs.astropy.org/projects/photutils/en/stable/user_guide/psf.html)).
- **Ses limites, dites par des pros :** sur JWST, « une image sans étoiles est le reste naturel d'une photométrie par
  profil », mais le reste avait de forts résidus, dus à un profil imparfait et à une **sur-soustraction des étoiles sur
  fond étendu** ; ils ont fini par masquer et interpoler à la main, et n'ont utilisé l'image qu'en qualitatif
  ([Ginsburg et al. 2023, ApJ, arXiv:2308.16050](https://arxiv.org/abs/2308.16050), § 3.5).
- **Profil étendu + Gaia.** J-PLUS mesure le profil jusqu'à 4′ et le soustrait de chaque pose ; le centre des étoiles
  saturées est pris sur les pixels débordants (Gaia est incomplet pour les plus brillantes) ; une émission diffuse
  autour de l'étoile qui sert à mesurer le profil fausse ce profil
  ([Eskandarlou et al. 2026, A&A 705 A67, arXiv:2510.12940](https://arxiv.org/abs/2510.12940) ; détails lus via
  extrait). Méthode « MAHDI » : étoiles Gaia G < 16, position fixée sur Gaia, flux ajusté, de la plus brillante à la
  plus faible ([arXiv:2605.18976](https://arxiv.org/pdf/2605.18976), lu via extrait).
- **Filtrage multi-résolution (MRF).** Modéliser les sources compactes (sur une image plus fine), les flouter au profil
  de l'image, les soustraire ; les étoiles brillantes à part ; variante sans image fine (« self-MRF »)
  ([van Dokkum et al. 2020, PASP, arXiv:1910.12867](https://arxiv.org/abs/1910.12867)).
- **Remplir les cœurs masqués.** « maskfill » : on remplit de l'extérieur vers l'intérieur par médianes 3×3 répétées ;
  meilleur que les interpolations classiques ; limite : un détail entièrement caché sous le masque ne peut pas être
  retrouvé ([van Dokkum & Pasha 2024, PASP, arXiv:2312.03064](https://arxiv.org/abs/2312.03064)).
- **Réseau qui remplit.** U-Net à convolutions partielles, entraîné sur des sources injectées dans de vraies cartes
  Herschel ; garde les propriétés du bruit
  ([Madarász et al. 2025, A&A 696 A37, arXiv:2502.18345](https://arxiv.org/abs/2502.18345)).
- **Diffusion (au sens équation de la chaleur, pas réseau génératif).** Retire étoiles et bruit sur une seule image
  ([Cohen & Lu 2021, Astronomy and Computing 37](https://researchportal.hw.ac.uk/en/publications/a-diffusion-based-method-for-removing-background-stars-from-astro/)).
- **Non trouvé :** aucun article évalué par des pairs sur les outils d'amateurs (StarNet, StarXTerminator) ni de
  banc d'essai public.

### Tableau comparatif

| Outil | Principe | Entrée | Sorties | Ligne de commande | Points faibles connus |
|---|---|---|---|---|---|
| StarNet v1 | réseau (pix2pix), code ouvert | étirée, TIFF 8 bits | sans étoiles + masque | oui (Python) | aigrettes, grosses étoiles, artefacts |
| StarNet v2 | réseau, code fermé | étirée (linéaire = étirement provisoire) | sans étoiles, masque, unscreen | oui (`starnet2`) | artefacts près des étoiles brillantes *(informel)* |
| StarXTerminator | réseau, commercial | linéaire ou étirée MTF ; pas GHS/arcsinh | sans étoiles + différence/unscreen | oui (`rc-astro sxt`) | aberrations, FWHM > 8 px, halos saturés |
| Dark Star | réseau NAFNet | auto-étirée si médiane < 0,125 ; mono/couleur | sans étoiles + additif/unscreen | oui | non documentés |
| SyQon Zenith | réseau NAFNet | étirement provisoire | sans étoiles + masque | via Siril | nébulosité faible *(informel)* |
| DeepStarRemoval | réseau de remplissage | linéaire ou étirée | sans étoiles + étoiles exactes | non | non documentés |
| Siril synthstar | profils ajustés | linéaire conseillé | étoiles de synthèse | oui | étoiles non détectées |
| PSF + Gaia (DAOPHOT, J-PLUS) | ajustement et soustraction | linéaire | reste + modèle | oui (photutils) | sur-soustraction sur fond étendu, cœurs saturés |

## 2. Deux calques et leur recombinaison

**Comment le calque d'étoiles est fait.**
- **En linéaire : soustraction.** étoiles = original − sans étoiles ; StarXTerminator la dit « la meilleure » en
  linéaire, avec la meilleure fidélité de couleur des étoiles, et réserve l'unscreen aux images étirées
  ([notes d'usage](https://rc-astro.com/starxterminator-usage-notes)).
- **En étiré : « unscreen ».** étoiles = (original − sans étoiles) / (1 − sans étoiles) : c'est l'inverse exact du
  mode Écran, si bien que sans étoiles « écran » étoiles redonne l'original
  ([StarNet.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/StarNetAstro/StarNet.py),
  [darkstar_engine.py](https://github.com/setiastro/setiastrosuitepro/blob/main/src/setiastro/saspro/cosmicclarity_engines/darkstar_engine.py)) ;
  en PixelMath : `rescale(starry, starless, 1)` (forum, lu via extrait,
  [AstroBin](https://app.astrobin.com/forum/topic/178601/adding-stars-to-starless-image-causes-them-to-bloat-what-am-i-doing-wrong)
  *(informel)*). D'après l'aide de StarXTerminator citée sur les forums, des étoiles « unscreenées » remises en Écran
  sur un sans-étoiles retravaillé donnent un bien meilleur résultat qu'une simple addition
  ([AstroBin](https://app.astrobin.com/forum/topic/64948/processing/which-formula-do-you-recommend-to-recombine-stars-with-starless-nebula-with-pixelmath)
  *(informel, de seconde main)*).
- **Le piège de l'étirement.** Une courbe non linéaire ne se distribue pas sur une somme : étirer (fond + étoiles)
  n'est pas étirer le fond plus étirer les étoiles. Le script StarNet de Siril le dit : on ne peut pas désétirer un
  calque d'étoiles à part ; en mode linéaire, il désétire d'abord l'image sans étoiles, puis calcule le calque
  d'étoiles sur l'original linéaire ([StarNet.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/StarNetAstro/StarNet.py),
  lignes 1065-1087).
- **Un détail de code qui compte.** Siril et Dark Star **coupent à 0** les valeurs négatives du calque d'étoiles
  (`clip(original − sans étoiles, 0, 1)`) ([StarNet.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/StarNetAstro/StarNet.py),
  ligne 432). Là où le sans-étoiles est un peu au-dessus de l'original (le bruit), cette coupe ajoute une petite
  lumière positive : la somme ne redonne plus exactement l'original. *(Lecture du code, pas une source publiée.)*

**Comment on recombine.**
- **Écran** : 1 − (1 − a)(1 − b). C'est ce que fait l'outil « Recomposition d'étoiles » de Siril, après un étirement
  GHS du fond et un étirement arcsinh modifié des étoiles, chacun réglé à part
  ([remixer.c, ligne 517](https://gitlab.com/free-astro/siril/-/blob/master/src/gui-gtk4/remixer.c),
  [doc Siril 1.2](https://siril.readthedocs.io/en/1.2/processing/stars/star-recomp.html)). Dans PixelMath :
  `~((~starless)*(~stars))`, identique à `combine(stars, starless, op_screen())`
  ([AstroBin](https://app.astrobin.com/forum/topic/178601/adding-stars-to-starless-image-causes-them-to-bloat-what-am-i-doing-wrong)
  *(informel)*). RC Astro recommande d'étirer les deux images puis de les recombiner en Écran
  ([notes d'usage](https://rc-astro.com/starxterminator-usage-notes)).
- **Addition linéaire** (Densité linéaire / « Linear Dodge » dans Photoshop) : sans étoiles + étoiles
  ([Astronomy](https://www.astronomy.com/observing/how-to-remove-stars-from-images-with-ai-tools/)) ; en PixelMath
  `$T+stars`, ou `$T+0.7*stars` pour les atténuer
  ([forum PixInsight](https://pixinsight.com/forum/index.php?threads/after-starxterminator.18773/) *(informel)*).
  En linéaire, Écran et addition se confondent presque, car les valeurs sont petites
  ([AstroBin](https://app.astrobin.com/forum/topic/64948/processing/which-formula-do-you-recommend-to-recombine-stars-with-starless-nebula-with-pixelmath)
  *(informel)*).
- **Photoshop** : RC Astro donne une recette en calques (Diviser, Écran, inversions) qui aboutit à trois calques :
  Étoiles (Écran), Sans étoiles, Original ([notes d'usage](https://rc-astro.com/starxterminator-usage-notes)).
- **Les deux modes côte à côte** : le script VeraLux StarComposer propose « Écran (sûr) : évite l'écrêtage, mais peut
  baisser le contraste local » et « Addition (physique) : garde le contraste, risque d'écrêter les cœurs sur fond
  brillant » ([VeraLux_StarComposer.py](https://gitlab.com/free-astro/siril-scripts/-/blob/main/VeraLux/VeraLux_StarComposer.py)).

**Ce qui tourne mal.**
- **Étoiles gonflées, écrêtées** : l'addition de deux images étirées dépasse 1 ; l'Écran reste sous 1
  ([AstroBin](https://app.astrobin.com/forum/topic/178601/adding-stars-to-starless-image-causes-them-to-bloat-what-am-i-doing-wrong)
  *(informel)*).
- **Anneaux et halos** : certains font toutes les combinaisons en linéaire pour les éviter ; un ajustement linéaire
  préalable évite les anneaux autour des étoiles
  ([AstroBin, extraits](https://app.astrobin.com/forum/topic/64948/processing/which-formula-do-you-recommend-to-recombine-stars-with-starless-nebula-with-pixelmath)
  *(informel)*). Les restes de halo laissés par l'outil sont un défaut à part (§ 1).
- **Couleurs** : calibrer les couleurs **avant** la séparation, sinon les étoiles détonnent (« sapin de Noël ») quand
  la balance du fond change beaucoup ([forum PixInsight](https://pixinsight.com/forum/index.php?threads/after-starxterminator.18773/)
  *(informel)*) ; ne pas étirer automatiquement le calque d'étoiles dans PixInsight, il porte les réglages d'affichage
  de l'original ([notes d'usage](https://rc-astro.com/starxterminator-usage-notes)).
- **Étoiles doubles** : rien de publié trouvé. *(Raisonnement :)* elles apparaissent si le calque d'étoiles ne vient
  pas exactement de la même géométrie que le fond (autre empilement, autre recadrage), ou si on recombine un original
  au lieu d'un sans-étoiles.

**Régler la taille et la couleur des étoiles sur leur calque.**
- **Intensité** : multiplier le calque par un facteur < 1 avant la recombinaison
  ([Astronomy](https://www.astronomy.com/observing/how-to-remove-stars-from-images-with-ai-tools/)).
- **Étirement propre aux étoiles** : arcsinh modifié dans Siril, choisi pour mieux garder les couleurs
  ([doc Siril 1.2](https://siril.readthedocs.io/en/1.2/processing/stars/star-recomp.html)).
- **Réduction** : formules PixelMath dites « Transfer » et « Halo » (paramètre K entre 0 et 1, attribuées à Bill
  Blanshan) ([gist](https://gist.github.com/Xophe92/7df23b1d0c48709266739ae9597117fb) *(informel)*) ; érosion par
  MorphologicalTransformation sous masque d'étoiles dans PixInsight
  ([AstroBin](https://ssr.app.astrobin.com/forum/topic/14724/processing/what-star-reduction-techniques) *(informel)*).
- **Saturation** : pas dans l'outil de recomposition de Siril ; à faire après la recombinaison
  ([doc Siril 1.2](https://siril.readthedocs.io/en/1.2/processing/stars/star-recomp.html)).

## 3. Comment on évalue la qualité

Chez les amateurs, les comparaisons sont **visuelles** ([Evan Tsai](https://www.astroimagetw.com/en/tutorials/starless-tools-comparison/),
[Astronomy](https://www.astronomy.com/observing/how-to-remove-stars-from-images-with-ai-tools/) *(informels)*) ; je
n'ai trouvé **aucun banc d'essai chiffré** de StarNet ou StarXTerminator. Les méthodes chiffrées publiées :

1. **Vérité connue, image de synthèse** : étoiles posées sur un fond connu, scores écart moyen, écart quadratique,
   PSNR, SSIM ([Cohen & Lu 2021](https://researchportal.hw.ac.uk/en/publications/a-diffusion-based-method-for-removing-background-stars-from-astro/)).
2. **Injection puis récupération** : sources de flux connu injectées dans de **vraies** cartes jamais vues à
   l'entraînement ; on mesure flux retrouvé / flux injecté (idéal 1), puis on compare le **spectre de puissance** (la
   texture du bruit à chaque échelle) et l'**histogramme des pixels** sous les masques à ceux de l'original
   ([Madarász et al. 2025](https://arxiv.org/abs/2502.18345)).
3. **Faux masques sur des zones propres** : masquer des zones sans étoiles, remplir, comparer à ce qu'il y avait
   ([maskfill](https://arxiv.org/abs/2312.03064)).
4. **Contrôle de biais hors étoiles** : étoiles masquées, la pente entre image sans étoiles et original doit valoir 1
   ([Casassus et al. 2026](https://arxiv.org/abs/2602.11356), annexe C).
5. **Le reste doit ressembler à du bruit** après soustraction des profils
   ([photutils](https://docs.astropy.org/projects/photutils/en/stable/user_guide/psf.html)).

Pour nos 6 images de banc, ces cinq épreuves se transposent directement : (a) injecter des étoiles au profil mesuré sur
la même image, à plusieurs magnitudes, sur fond de nébuleuse et sur fond de ciel, puis mesurer ce qui reste, par classe
de magnitude ; (b) pente 1 hors étoiles ; (c) spectre de puissance et histogramme dans les zones remplies contre les
zones voisines ; (d) en linéaire, vérifier que sans étoiles + étoiles = original au pixel près ; (e) compter les
étoiles restantes et vérifier que les objets compacts connus (galaxies, nébuleuses planétaires) sont toujours là.

## 4. Ce que ça dit pour nous

- **Notre modèle par profil d'étoile + Gaia est une voie publiée** (DAOPHOT, photutils, J-PLUS, MAHDI,
  PhotometricGlowSubtraction). Ses échecs connus sont précis : profil imparfait et **sur-soustraction sur fond de
  nébuleuse** (Ginsburg 2023), cœurs saturés à masquer puis remplir (J-PLUS, maskfill), profil faussé par la
  nébulosité autour de l'étoile de mesure (J-PLUS). C'est exactement ce que les épreuves (a) et (b) doivent viser.
- **Tous les réseaux du marché travaillent sur une image étirée** (StarNet, StarXTerminator, Dark Star, SyQon) et
  ceux qui acceptent du linéaire l'étirent en interne puis le désétirent. Pour un moteur linéaire, le schéma établi est
  celui du script Siril : étirer, retirer, désétirer le sans-étoiles, **puis** calculer le calque d'étoiles par
  soustraction sur l'original linéaire.
- **Deux calques linéaires, la somme exacte** : sans étoiles + étoiles = original, en flottant, **sans couper les
  négatifs** (contrairement aux scripts Siril). C'est la seule forme où rien ne se perd.
- **Post-traitement avec les modes de fusion ordinaires** : chaque calque étiré à part, puis **Écran** (Siril, RC
  Astro, PixInsight) par défaut, **Addition** en option (VeraLux) ; « unscreen » seulement si les deux calques sont
  extraits d'images étirées par la même courbe. Le facteur d'intensité, l'étirement propre des étoiles et la saturation
  se règlent sur le calque d'étoiles.
- **StarNet en option** : la ligne de commande `starnet2` (TIFF en entrée et sortie, `-s`, `-u`, `-m`, `-n`) suffit ;
  StarXTerminator a aussi la sienne (`rc-astro sxt`). Aucun des deux n'est nécessaire au reste de la chaîne.
- **Pour la comparaison** : aucun chiffre publié n'existe pour StarNet ; nos épreuves (a) à (e) seront les premières
  mesures, à présenter comme telles.
