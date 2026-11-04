# Chantier : le débruitage

Objectif : garder la qualité de notre débruitage actuel (Noise2Noise entraîné sur chaque image), en une fraction du
temps, et le rendre utilisable sur processeur. Ouvert le 4 novembre.

## 1. Ce que fait notre débruitage aujourd'hui

- **La méthode : Noise2Noise** (Lehtinen et al., 2018). L'empilement produit deux demi-empilements indépendants (A et
  B) ; un réseau apprend à prédire l'un à partir de l'autre. Le bruit étant indépendant d'une moitié à l'autre, le
  réseau ne peut apprendre que le signal : il retire le bruit **sans rien inventer**.
- **Le réseau** : un U-Net (4 niveaux, 48 à 320 canaux), dont l'entrée est l'image normalisée (fond et bruit de chaque
  canal) **et la carte du bruit** tirée des poids du drizzle ; il prédit une correction (résidu) ; la perte est un écart
  quadratique pondéré par le bruit attendu.
- **Le coût** : il repart de zéro sur chaque image, 6 000 à 30 000 pas : environ 30 minutes sur GPU ; inutilisable sur
  processeur.

## 2. L'état de l'art (recherches du 4 novembre, sources en lien)

### En astronomie

- **ASTERIS** (Guo et al., *Science*, 2026 ; [arXiv 2602.17205](https://arxiv.org/abs/2602.17205)) : Noise2Noise
  étendu aux poses multiples d'un même champ ; les poses sont réparties en deux groupes indépendants (entrée et cible),
  exactement le principe de nos demi-empilements. Sur des données JWST et Subaru : limite de détection plus profonde
  d'une magnitude, photométrie et forme des étoiles préservées. **C'est la validation la plus forte de notre méthode.**
  Leur variante « spatio-temporelle » donne au réseau plusieurs poses en entrée, et non un empilement.
- **AstroSURE** ([arXiv 2604.16793](https://arxiv.org/abs/2604.16793), 2026) : sur des images HST et CFHT,
  Noise2Noise fait à peu près aussi bien qu'un apprentissage sur des images sans bruit, et améliore la détection des
  sources faibles.
- **Noise2Astro** ([arXiv 2209.07071](https://arxiv.org/abs/2209.07071), 2022) : premier essai de Noise2Noise en
  astronomie, sur données simulées.
- **« Denoising the Deep Sky »** ([arXiv 2601.23276](https://arxiv.org/abs/2601.23276), 2026) : un modèle physique
  du bruit des capteurs CCD, appliqué à des empilements profonds, pour fabriquer des paires d'entraînement. C'est une
  autre façon d'obtenir **des niveaux de bruit variés** (ta proposition de cette nuit, que nous faisons avec de vraies
  données : chaque jeu empilé avec toutes ses poses, puis avec un quart).
- **Les outils d'astrophotographie** :
  - NoiseXTerminator ([RC Astro](https://rc-astro.com/software/nxt)) : réseau propriétaire, entraîné uniquement sur des
    images du ciel profond, de façon « non générative » ; rien n'est publié de son architecture ni de ses données ;
  - GraXpert ([via Siril](https://siril.readthedocs.io/fr/latest/processing/graxpert.html)) : un réglage unique
    (l'intensité), un modèle ONNX d'environ 480 Mo d'après un [miroir](https://huggingface.co/jromanghf/graxpert-models)
    (lent sur processeur, de l'aveu de la documentation) ;
  - Cosmic Clarity (Seti Astro) : rien trouvé sur son entraînement.

### Pré-entraîner, puis adapter à l'image

- **SS-TTA** ([CVPR Workshops 2023](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Fahim_SS-TTA_Test-Time_Adaption_for_Self-Supervised_Denoising_Methods_CVPRW_2023_paper.pdf))
  : une adaptation au test qui s'applique à des débruiteurs auto-supervisés, avec du Noise2Noise pendant l'adaptation.
- **GainTuning** ([NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/c7558e9d1f956b016d1fdba7ea132378-Paper.pdf))
  : pour adapter un débruiteur pré-entraîné à une image, on n'ajuste qu'un **gain par canal de chaque couche** (très
  peu de paramètres) : rapide et sans sur-apprentissage. **Le candidat le plus intéressant pour l'affinage court, et
  surtout pour le processeur.**
- Autres : apprentissage « méta » pour une adaptation rapide ([arXiv 2001.02899](https://arxiv.org/pdf/2001.02899)),
  adaptation par méta-transfert ([arXiv 2207.02066](https://arxiv.org/pdf/2207.02066)), TTT-MIM
  ([ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_20)), adaptation robuste par a priori
  gaussien ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_Robust_Test-Time_Adaptation_for_Single_Image_Denoising_Using_Deep_Gaussian_ICCV_2025_paper.pdf)) ;
  en imagerie médicale, Noise2Noise affiné sur chaque examen ([PMC11216369](https://pmc.ncbi.nlm.nih.gov/articles/PMC11216369/)).

### Le bruit corrélé, un point propre à notre chaîne

Le drizzle (et tout rééchantillonnage) **corrèle le bruit entre pixels voisins**. Les méthodes à « angle mort »
(Noise2Void, Noise2Self) supposent un bruit indépendant d'un pixel à l'autre et sont trompées par cette corrélation
([AP-BSN](https://arxiv.org/pdf/2203.11799), [AT-BSN](https://cvpr.thecvf.com/virtual/2024/poster/30827),
[Noise2Sim](https://arxiv.org/pdf/2011.03384)) ; même Noise2Noise se dégrade si le bruit est corrélé **entre les deux
images d'une paire** ([eSURE](https://arxiv.org/pdf/1902.02452)). **Nos deux demi-empilements ont des bruits
indépendants l'un de l'autre** (des poses différentes) : la corrélation spatiale à l'intérieur de chacun ne gêne pas
Noise2Noise. En revanche, **Zero-Shot Noise2Noise** ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.pdf)),
un réseau d'environ 20 000 paramètres entraîné sur l'image même, très rapide sur processeur, fabrique ses paires en
sous-échantillonnant une seule image : il suppose un bruit indépendant entre pixels voisins, **ce que le drizzle
contredit**. Il n'est donc pas un bon candidat pour nous.

### La vitesse sur processeur

- **FFDNet** ([arXiv 1710.04026](https://arxiv.org/pdf/1710.04026)) : la carte du bruit en entrée (comme notre
  réseau), et un travail sur une image réduite par regroupement de pixels, d'où une vitesse sur processeur environ trois
  fois supérieure à DnCNN et BM3D.
- **NAFNet**, réputé efficace sur GPU, est lent sur processeur à cause de ses normalisations
  ([MFDNet, arXiv 2211.04687](https://arxiv.org/pdf/2211.04687)).
- Je n'ai pas trouvé de mesure de DRUNet sur processeur : à mesurer nous-mêmes.

### Ce que ça dit pour nous

1. **Noise2Noise sur deux groupes de poses est la bonne méthode** (ASTERIS, AstroSURE) : on la garde.
2. **Pré-entraîner sur beaucoup d'images, puis adapter à l'image** est une voie publiée ; pour l'adaptation, **ajuster
   seulement des gains (GainTuning)** est la piste la plus prometteuse, rapide et sûre, y compris sur processeur.
3. **Le jeu d'entraînement doit être divers** (capteurs, optiques, champs, niveaux de bruit) **et passer par notre
   chaîne**, pour que le réseau voie notre bruit corrélé : d'où l'empilement de tout `F:\dataset_astro`, chaque jeu
   entier et au quart. Nos 6 cas du banc restent hors entraînement, pour l'évaluation. (L'idée, écrite d'abord, que nos
   seules données seraient les meilleures était fausse : peu de capteurs et de télescopes, un modèle spécialisé.)
4. **Pour le processeur** : mesurer le temps de notre U-Net ; si c'est trop lent, s'inspirer de FFDNet (travail sur
   une image réduite).

## 3. Le jeu de données `F:\dataset_astro\data`

49 jeux de brutes et de calibrations, 182 Go (README, catalogue, contrôle qualité). Les jeux « mono » sont bien
monochromes (pas de motif de Bayer) ; **notre chaîne refuse le monochrome et le X-Trans** : ils ne sont pas utilisés
cette nuit. Sont aussi écartés les jeux de moins de 14 poses, le paysage à premier plan fixe, la comète et la mosaïque.

**Retenus : 28 jeux couleur** (Canon 6D, 40Da, 60Da, 5D II, Ra ; Nikon D810A, D800 ; Pentax 645Z ; Fujifilm GFX en
Bayer ; ASI2600, ASI294, ASI533 ; Seestar S50 et S30 ; Stellina ; Celestron Origin ; DWARF), plus tes 8 projets qui ont
leurs demi-empilements (lecture seule).

## 4. La nuit du 3 au 4 novembre

Outil : `tools/bench/denoise_dataset.py night --hours 10` ; journal : `F:\astro_essais\denoise_night.log`.
1. Chaque jeu empilé par notre chaîne jusqu'à l'empilement (`F:\astro_essais\dataset_projets\<id>`), puis les jeux de
   80 poses ou plus une seconde fois avec un quart de leurs poses (réparties sur la nuit).
2. Les paires A, B et la carte du bruit, normalisées comme le fait notre débruitage, en float16
   (`F:\astro_essais\denoise_dataset`).
3. Le pré-entraînement de notre U-Net actuel en Noise2Noise, sur des zones tirées au hasard dans toutes les paires,
   10 heures ; une sauvegarde et une mesure sur les 6 cas du banc tous les 2 500 pas
   (`pretrain_history.json`).

Ce qui s'est réellement passé : chaque jeu a été empilé à quatre niveaux (toutes les poses, la moitié, le quart, le
huitième, un niveau seulement s'il garde 10 poses ou plus) ; 84 paires d'entraînement. L'entraînement s'est arrêté
après **2 h 20** au lieu de 10 h : le nombre de pas avait été fixé d'après la vitesse des 200 premiers pas, quatre fois
plus lents que la suite. Corrigé pour la v2 (la vitesse d'apprentissage suit l'horloge).

## 5. La comparaison du 4 novembre

Outil : `tools/bench/denoise_eval.py` ; résultats : `F:\astro_essais\denoise_eval\` (`results*.json`, une image de
détail par cas). Protocole : les méthodes qui apprennent sur l'image n'en voient que les 80 % de gauche ; toutes sont
mesurées sur les 20 % de droite (une marge de 192 px pour le contexte). La mesure : l'erreur Noise2Noise sur cette zone
(A débruité comparé à B, et l'inverse). Plus bas = mieux ; le plancher (rien d'autre que le bruit de B) vaut à peu près
la moitié de la valeur « sans débruitage ».

### Une fuite dans la v1

Cinq des six cas du banc viennent des mêmes poses brutes que des paires d'entraînement : M8 du banc = le jeu 02 (le
même empilement), M51 = le jeu 03, Heart, Iris et Rosette = tes projets du même nom. Seul Seestar (M31) est
indépendant. Les chiffres de la v1 sur ces cinq cas sont donc trop flatteurs. Pour la v2, ces sources sont écartées de
l'entraînement (`HELD_OUT` dans `denoise_dataset.py`, 76 paires).

### Résultats de la v1 (modèle de la nuit)

| Cas | Sans débruitage | v1 seul | v1 + 500 pas | v1 + 2 000 pas | v1 + GainTuning 500 | Référence (de zéro, 16 000 pas) |
|---|---|---|---|---|---|---|
| Heart | 3,159 | 2,310 | 2,020 | 1,995 | 2,117 | 2,102 |
| Iris | 2,730 | 1,450 | 1,438 | 1,433 | 1,444 | 1,449 |
| M51 | 1,189 | 0,625 | 0,629 | 0,628 | 0,629 | 0,668 |
| M8 | 2,051 | 1,009 | 1,027 | 1,002 | 1,020 | **0,885** |
| Rosette | 1,943 | 1,059 | 1,089 | 1,112 | 1,060 | 1,124 |
| Seestar (indépendant) | 1,574 | **0,800** | 0,800 | 0,801 | 0,800 | 0,813 |

Temps sur la carte graphique (RTX 4060 Ti) : appliquer le modèle ≈ 1 s ; 500 pas d'adaptation ≈ 40 s ; la référence
≈ 21 min. Sur le processeur (8 fils) : appliquer le modèle à une image entière de Heart ≈ 40 s ; un pas d'adaptation
3,6 s (500 pas ≈ 30 min, la référence ≈ 16 h).

Ce qu'on en tire :
- Sur Seestar, le seul cas propre, le modèle pré-entraîné seul fait mieux que la référence.
- L'adaptation à l'image n'aide que Heart ; elle dégrade Rosette, et d'autant plus qu'elle dure.
- **M8 est le point faible** : son bruit n'a presque aucune corrélation d'un pixel au voisin (0,03 contre 0,15 à 0,35
  ailleurs), et aucune autre paire d'entraînement ne lui ressemble. La référence, qui apprend ce bruit sur l'image,
  fait nettement mieux.
- Sur Heart, l'erreur en trop est à 88 % dans le cœur des étoiles brillantes (le 1 % le plus lumineux) ; les étoiles
  saturées n'en font que 18 %. Le fond du ciel est bon.
- M8 vient d'une séance sans décalage entre les poses (dérive totale 4,4 px, contre 80 à 115 px ailleurs) : sans ce
  décalage, l'empilement ne mélange pas les pixels voisins et le bruit reste indépendant d'un pixel à l'autre.

### La v2 (sans les sources du banc) face à la référence

Modèle `pretrained_unet_v2_mid.pt` : le meilleur point de validation de la v2 copié à 11 h 10, le pas 52 500 (256 px par zone, 76 paires). C'est le modèle livré (`astro_gpu/resources/denoise/pretrained_unet.pt`, en demi-précision). La v2 a tourné 5 h (171 000 pas) : la fin de l'entraînement dégrade Heart (2,194 → 2,453) et M8 (1,488 → 1,595) et ne gagne que moins de 1 % sur M51, Rosette et Seestar ; entraîner plus longtemps n'apporte rien.

| Cas | Sans | v2 seul (1 s) | v2 + 500 pas, lr 2e-4 (40 s) | v2 + 4 000 pas, lr 3e-4 (5 min) | v2 + 8 000 pas (10 min) | Référence (21 min) |
|---|---|---|---|---|---|---|
| Heart | 3,159 | 2,194 | **1,975** | 1,982 | | 2,102 |
| Iris | 2,730 | 1,453 | 1,437 | | | 1,449 |
| M51 | 1,189 | 0,631 | 0,632 | | | 0,668 |
| M8 | 2,051 | 1,488 | 1,131 | 0,843 | **0,722** | 0,885 |
| Rosette | 1,943 | 1,063 | **1,000** | 1,075 | | 1,124 |
| Seestar | 1,574 | 0,801 | 0,800 | | | 0,813 |

Partir du modèle pré-entraîné bat la référence partout ; sur M8, il faut une adaptation longue (la référence, partie
de zéro, n'avait pas fini d'apprendre).

## 6. Face à GraXpert et Topaz

Outil : `tools/bench/denoise_compare.py` ; tout dans `F:\astro_essais\denoise_compare\` (résultats `results.json`,
images de détail `crops\<cas>.jpg`). Chaque outil débruite A et B séparément (l'image entière) ; même mesure que plus
haut, sur la même zone, sur l'image entière et sans les étoiles (les étoiles et leurs abords, 1 à 41 % de la zone).
- **GraXpert 3.0.2**, le dernier modèle de débruitage publié (les préversions 3.1 n'ajoutent que la déconvolution).
  Sa « force » est un mélange de l'entrée et du débruitage complet (vérifié) : il tourne une fois à 1, et la mesure
  donne sa force par défaut (0,5) et la meilleure pour chaque cas (toujours 1).
- **Topaz DeNoise AI**, passé par l'utilisateur sur A et B étirés (TIFF 16 bits, un étirement connu, défait pour la
  mesure).

Image entière :

| Cas | Sans | Nous, seul | Nous + 40 s | Nous + 5 min | GraXpert 0,5 | GraXpert 1 | Topaz |
|---|---|---|---|---|---|---|---|
| Heart | 3,159 | 2,196 | **1,976** | 1,984 | 2,818 | 2,680 | 105 |
| Iris | 2,730 | 1,454 | 1,437 | **1,426** | 1,872 | 1,532 | 34 |
| M51 | 1,189 | **0,631** | 0,632 | 0,642 | 0,825 | 0,676 | 2,10 |
| M8 | 2,051 | 1,487 | 1,130 | **0,843** | 1,925 | 1,871 | 124 |
| Rosette | 1,943 | 1,063 | **0,997** | 1,073 | 1,380 | 1,154 | 48 |
| Seestar | 1,574 | 0,801 | **0,800** | 0,802 | 1,027 | 0,813 | 5,5 |

Sans les étoiles (le plancher ≈ la moitié de « sans ») :

| Cas | Plancher ≈ | Nous, seul | Nous + 40 s | GraXpert 1 | Topaz |
|---|---|---|---|---|---|
| Heart | 0,488 | 0,521 | **0,518** | 0,520 | 0,645 |
| Iris | 1,214 | 1,230 | **1,228** | 1,238 | 1,275 |
| M51 | 0,521 | **0,525** | **0,525** | 0,533 | 0,551 |
| M8 | 0,162 | 0,175 | 0,173 | **0,172** | 0,246 |
| Rosette | 0,790 | **0,802** | **0,802** | 0,808 | 0,848 |
| Seestar | 0,766 | **0,770** | **0,770** | 0,776 | 0,796 |

Temps par image de la taille de Heart (12 Mpx) : nous ≈ 3 s sur la carte graphique, ≈ 40 s sur le processeur ;
GraXpert ≈ 105 s sur la carte graphique.

À l'œil (`crops\`) : chez nous, le fond est propre, toutes les étoiles faibles restent, les étoiles allongées et les
aigrettes sont respectées. GraXpert est fidèle aux étoiles mais laisse un grain sur le fond. Topaz efface beaucoup
d'étoiles faibles, rend les étoiles allongées en points à cœur noir, supprime les aigrettes et marbre le fond ; il
accentue les étoiles (le haut des étoiles brillantes de Heart passe de 46 500 à 51 700 sur 65 535), d'où ses scores
sur l'image entière.

## 7. Dans le logiciel (4 novembre)

L'étape Débruitage (`astro_gpu/stages/denoise.py`, version `2-<empreinte du modèle>` : changer le modèle livré rend
les résultats périmés) a deux méthodes, choisies sur les cartes du bloc :
- **Réseau pré-entraîné** (par défaut) : le modèle livré (`astro_gpu/resources/denoise/pretrained_unet.pt`) appliqué
  tel quel ; quelques secondes sur la carte graphique, environ une minute sur le processeur pour 12 Mpx.
- **Pré-entraîné + adapté à l'image** : 500 pas d'adaptation (courte, environ 40 s sur la carte graphique) ou 4000
  (longue, environ 5 min). L'étape propose d'elle-même la longue quand le bruit est presque indépendant d'un pixel à
  l'autre (corrélation sous 0,08 : une séance sans dithering) ; avec le réseau seul dans ce cas, elle le signale dans
  le journal.

Mesures affichées : bruit avant, après, réduction, corrélation du bruit entre voisins, pas d'adaptation. L'ancien
apprentissage complet sur chaque image n'existe plus ; ses réglages enregistrés dans un projet sont ignorés. Le banc a
deux workflows de plus par projet, « Classic stack · denoise » et « Classic stack · denoise adapted »
(`tools/bench/denoise_projects.py`).
