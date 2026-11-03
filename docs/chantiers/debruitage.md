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

## 5. Après la nuit

- Comparer, sur les cas du banc : le modèle pré-entraîné seul ; pré-entraîné + quelques centaines de pas d'affinage ;
  pré-entraîné + GainTuning ; notre Noise2Noise complet actuel (la référence). Mesures : la perte Noise2Noise sur les
  demi-empilements, le résidu sans structure (« rien n'est inventé »), le temps sur GPU et sur processeur.
