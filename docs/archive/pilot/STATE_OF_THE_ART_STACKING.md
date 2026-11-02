# État de l'art : l'empilement pour capteurs couleur

*1er novembre 2026. Pour le moteur astro_gpu : faire mieux que Siril 1.4 avec des méthodes publiées (règle 4b de
ENGINE_KNOWLEDGE). Capteurs couleur à matrice de Bayer : APN, hybrides, caméras couleur, télescopes intelligents.
Les sources sont numérotées entre crochets ([S5], [P12]…) et listées à la fin, avec leur lien. « Non vérifié » = lu
dans un résumé ou cité de mémoire, pas sur la page elle-même.*

## En bref

- **La couleur finale vient du drizzle Bayer.** PixInsight le dit en toutes lettres : c'est « la procédure
  recommandée » pour un capteur couleur, parce qu'elle n'interpole rien ; seule exception, un nombre marginal de
  poses [P12]. Siril 1.4, APP et DSS le proposent aussi. **Mais tous le règlent à l'échelle 1 et une goutte de 1**
  pour un capteur couleur [S5][P15][A8]. Notre réglage (échelle 1,5, goutte 0,7) est plus agressif que tous : une pose
  ne remplit que 12 % des pixels de sortie en rouge et en bleu, contre 25 % à 1 et 1. C'est la cause probable de nos
  points d'une seule couleur (à mesurer).
- **Rejet.** La référence pour un drizzle est celle de Hubble (AstroDrizzle) : chaque pose est comparée à une image
  médiane, avec le bruit du capteur et une tolérance pour les étoiles [L3][L4]. Notre rejet est de cette famille,
  avec des seuils maison (5 σ). Il ne rejette plus que **0,001 à 0,05 %** des échantillons sur le banc, quand Siril
  conseille de viser 0,1 à 0,5 % [S12]. À remplacer par les seuils publiés (3,5 puis 3,0 σ).
- **Pondération.** La meilleure pour une nébuleuse est l'inverse de la variance (transparence² / bruit²) [L6][L2].
  C'est presque la nôtre. Mesuré ici sur les 6 cas du banc : notre terme de finesse coûte moins de 1 % de bruit, et
  l'absence de pondération (le défaut de Siril) coûterait jusqu'à 8 % (Iris). La pondération n'explique donc pas
  notre retard sur Iris.
- **Sélection.** Pondérer plutôt qu'écarter. Zackay et Ofek (2017) : écarter des poses bien pondérées « nuit plus
  qu'il n'aide » [L6]. Siril et PixInsight gardent tout par défaut. Calcul de ce passage : sur la Rosette, garder les
  poses que nous écartons aurait donné environ 0,1 mag de profondeur.
- **Normalisation locale.** PixInsight l'active par défaut (WBPP) [P15] ; Siril ne l'a pas. La nôtre (un fond du
  2e degré par couleur, une échelle photométrique tirée des étoiles) est de cette famille. Défaut à corriger : une
  seule échelle pour les trois couleurs, alors que l'absorption du ciel dépend de la couleur.
- **Recalage des couleurs.** Aucun outil ne le fait pendant l'empilement par défaut : WBPP l'offre en option en
  alignant chaque couleur à part [P17], Siril seulement après l'empilement (et dans sa future 1.5) [S10]. Nos mesures :
  0,1 à 0,3 pixel au centre, jusqu'à 1,5 pixel dans les coins (M8, Heart), presque rien au Newton (M51 : 0,07). C'est
  un vrai atout. Pour l'améliorer : suivre la dispersion pose par pose, car elle change avec la hauteur [L13].
- **Personne n'a publié de mesure rigoureuse** des rejets selon le nombre de poses, du drizzle face au dématriçage
  selon l'échantillonnage, ni du gain du recalage des couleurs. Notre banc peut les faire : c'est là que nous pouvons
  dépasser l'état de l'art, pas seulement Siril.

---

## 1. Dématriçage ou drizzle Bayer

**Ce que font les outils**
- **PixInsight** : « the recommended procedure to generate integrated color images from CFA raw data is drizzle
  integration », car le drizzle « does not apply any interpolation to pixel data » ; exception : « when only a
  marginal number of raw frames are available » [P12]. Le dématriçage (VNG) ne sert qu'à aligner ; le drizzle relit
  les images brutes CFA [P18]. WBPP : drizzle CFA à l'échelle 1, goutte 1,0, noyau carré [P15] ; utile à partir de
  15 à 20 poses décalées [P17].
- **Siril 1.4** : le vrai drizzle de Hubble, avec Bayer drizzle, depuis la 1.4.0 [S8]. Sa documentation :
  - « it is recommended to apply CFA drizzle with scale = 1.0 and pixfrac = 1.0 » ;
  - « The real reason to drizzle CFA data is that the drizzled result has much cleaner noise » ;
  - le gain de finesse sur un capteur couleur est de « quelques % » ; la goutte doit valoir à peu près l'inverse de
    l'échelle ; noyaux carré ou gaussien quand les poses sont peu nombreuses [S5].
  - Sans drizzle : dématriçage **RCD**, choisi parce qu'il traite bien les bords ronds, donc les étoiles [S11][L10].
- **APP** : Bayer drizzle avec une goutte doublée (« 2,0 vaut 1,0 » en mono), noyau tophat, échelle 1 pour commencer
  [A8]. Dématriçage par défaut : AAD, dont l'algorithme n'est pas publié.
- **DSS** : Bayer drizzle « excellent avec beaucoup de poses » ; le drizzle « n'a pas de sens » avec peu de poses ;
  dématriçage bilinéaire par défaut [D1][D2].
- **Hubble (DrizzlePac)** : goutte de 0,7 à 1,0. Une règle chiffrée : la carte des poids ne doit pas varier de plus de
  20 % (écart-type sur médiane) [L2]. Le drizzle ne rend de la finesse que si l'image est sous-échantillonnée [L1][S5].

**Ce que disent les mesures**
- Une seule mesure publiée : Siril, 37 poses H-alpha, FWHM 3,59 avec agrandissement par interpolation contre 3,25
  avec drizzle 2× et goutte 0,5 [S5]. Pour des données couleur ordinaires, « quelques % » de finesse [S5].
- Aucune étude ne sépare les cas sous- et sur-échantillonnés. Hors astronomie, fusionner directement les pixels bruts
  d'une rafale bat le dématriçage suivi de la super-résolution (Farsiu 2006 [L12], Wronski 2019 [L11]) ; aucun
  logiciel d'astro ne l'emploie.
- **Piège des comparaisons de bruit** : l'interpolation, et une grande goutte, lissent le bruit d'un pixel à l'autre
  sans ajouter d'information. Le bruit pixel à pixel sous-estime alors le vrai bruit, d'un facteur qui atteint 1,5 et
  plus (facteur R de Fruchter et Hook) [L1][L5]. Il faut comparer à grande échelle, comme le fait notre banc.
- **Chez nous** (BENCH) : étoiles 5 à 12 % plus fines que Siril (Heart, M8), couleurs d'étoiles 1,2 à 3 fois plus
  justes. Mais 14 à 191 points d'une seule couleur par degré², contre 0 à 6 chez Siril.
  - À l'échelle 1,5 avec la goutte 0,7, une goutte fait 1,05 pixel de sortie, et un pixel rouge revient tous les
    3 pixels de sortie : une pose remplit (1,05/3)² ≈ **12 %** des pixels en rouge et en bleu. À l'échelle 1 avec la
    goutte 1 : (1/2)² = **25 %**. Le rouge et le bleu de chaque pixel reposent donc sur deux fois moins d'échantillons.
  - Notre FWHM d'alignement, mesurée sur des super-pixels 2×2, ne voit pas le sous-échantillonnage : 3,8 pixels
    mesurés sur Heart, pour des étoiles de 1,8 pixel dans la pile (11,65″ à 6,37″/pixel).

## 2. Alignement et interpolation

**Ce que font les outils**
- **Siril 1.4** : alignement global sur les étoiles, homographie par défaut, rééchantillonnage **Lanczos-4 avec
  écrêtage** par défaut. Selon sa doc, l'écrêtage donne « souvent une FWHM 8 % plus petite » que l'ancienne méthode
  par aire de pixel [S3]. Correction de distorsion tirée de l'astrométrie (polynômes SIP, `-disto`), nouvelle en 1.4
  [S4]. En drizzle CFA, les étoiles sont détectées sur les pixels verts et rien n'est interpolé [S4].
- **PixInsight** (StarAlignment) : modèle projectif par défaut ; surfaces splines puis plaques minces pour la
  distorsion (nettement améliorées en 1.9.0) [P7][P13]. Interpolation « Auto » : Lanczos-4 sans changement
  d'échelle, écrêtage 0,3 [P13]. La 1.9.5 corrige un biais de position du Lanczos [P8]. WBPP laisse la correction de
  distorsion éteinte par défaut [P15].
- **APP** : Lanczos-3 (selon ses utilisateurs), correction dynamique de distorsion [A10].
- **DSS** : polynôme du 1er ou du 2e degré selon le nombre d'étoiles [D1].

**Ce que disent les mesures** : seulement les 8 % de Siril.

**Chez nous** : un polynôme du 3e degré (précédé d'un modèle affine), ajusté sur les étoiles des super-pixels.
Le drizzle dépose chaque échantillon brut à sa position exacte : aucune interpolation. C'est la même idée que les
polynômes SIP de Siril ou le polynôme de DSS.

## 3. Normalisation

**Ce que font les outils**
- **Siril** : additive avec mise à l'échelle (estimateur IKSS), par couleur, globale ; pas de normalisation locale,
  ni en 1.4 ni dans la 1.5 en cours [S1][S8].
- **PixInsight** :
  - **LocalNormalization** (refaite en 1.8.9) : un modèle multi-échelle, pixel par pixel, du fond et de l'échelle de
    chaque pose ; l'échelle vient de la photométrie des étoiles (« PSF flux ») ; taille 1024 pixels par défaut [P4][P11].
  - **WBPP l'active par défaut** depuis la 2.4. Sa référence est l'intégration des **20 meilleures poses** ; sa taille
    vaut le quart du plus grand côté de l'image [P15][P16].
  - Normalisation adaptative (1.8.8-6) : la même idée, sans fichier auxiliaire [P3].
- **APP** : LNC, de degré 1 à 4 avec des itérations. Le développeur conseille le degré 1 avec 1 ou 2 itérations, et
  pas de rejet « linear fit » avec la LNC [A2][A5].

**Ce que disent les mesures** : rien de chiffré. La doc de PixInsight montre un exemple (M42), sans mesure [P11].

**Chez nous** : un fond additif du 2e degré par couleur, ajusté sur des médianes de blocs face à la pose de
référence, plus **une** échelle photométrique (photométrie d'ouverture des étoiles) commune aux trois couleurs.
C'est la LNC de degré 2 d'APP avec l'échelle photométrique de PixInsight. Deux écarts :
- une seule échelle pour les trois couleurs, alors que l'absorption par le ciel (nuages, hauteur) dépend de la
  couleur ; PixInsight et Siril normalisent chaque couleur ;
- la référence est une seule pose ; WBPP prend l'intégration des meilleures.

## 4. Rejet

**Ce que conseillent les outils selon le nombre de poses**

| Poses | PixInsight (doc) [P1][P2] | WBPP, choix auto [P15] | Siril (doc) [S1] | Hubble, AstroDrizzle [L3] |
|---|---|---|---|---|
| 3 à 5 | percentile (« 3 à 6 images ») | percentile (< 6) | percentile (« jusqu'à 6 ») | prédiction *minmed* : « très recommandée pour 3 images », bonne de 4 à 6 |
| 6 à 15 | sigma (≥ 8-10), sigma moyenné (8-10), winsorisé (≥ 15) | winsorisé | winsorisé 3/3 (le défaut) | médiane : « aussi bonne que minmed dès 4 environ » ; *minmed* à éviter dès 10 |
| 16 à 50 | winsorisé ; linear fit (≥ 15-25) ; ESD (≥ 25) | **linear fit** (depuis WBPP 2.8.0 ; l'ESD avant) | linear fit (grandes piles avec gradients) | médiane |
| plus de 50 | idem | linear fit | GESDT (« excellent au-delà de 50 ») | médiane |
| drizzle CFA | rejet calculé sur les poses dématricées et alignées, puis appliqué dans le drizzle [P18][P19] | idem | MAD : « semble le plus efficace » pour le rouge et le bleu drizzlés | comparaison de chaque pose à la médiane [L4] |

**La méthode de Hubble pour un drizzle** (AstroDrizzle, `driz_cr`) [L4] :
- une image médiane est construite avec toutes les poses, puis ramenée sur la grille de chaque pose ;
- un pixel est rejeté si |pose − médiane| > **3,5** × bruit + **1,2** × pente locale de la médiane ;
- ses 8 voisins sont rejetés s'ils dépassent **3,0** × bruit + **0,7** × pente ; extension de 1 pixel ;
- le bruit est celui du capteur : √(gain × (signal + fond) + lecture²) / gain ;
- le terme de pente protège les cœurs d'étoiles des petites erreurs d'alignement.

**Approches récentes**
- PixInsight 1.9.6 (2026) : une combinaison « biweight » qui écarte les valeurs aberrantes sans rejet explicite,
  « avec pratiquement l'efficacité statistique de la moyenne » ; selon ses développeurs, l'ESD atteint 96 à 97 %
  « dans nos tests ». Elle protège les cœurs d'étoiles. Une seule source, celle des auteurs [P9].
- Robust Chauvenet Rejection (Maples et al. 2018), dans PixInsight depuis la 1.8.9 [L9][P4].
- APP : rejet après correction par une médiane locale (rayon 32 pixels), et une « protection de la diffraction » qui
  plafonne le rejet par pixel [A4].

**Ce que disent les mesures** : aucune comparaison rigoureuse selon le nombre de poses. Il y a seulement les limites
de validité du test ESD (fiable dès 15, très fiable dès 25 valeurs) [L8] et les règles d'usage ci-dessus. Siril
conseille de viser un taux de rejet de 0,1 à 0,5 % [S12].

**Chez nous**
- La prédiction est la moyenne pondérée, nettoyée des valeurs très aberrantes. Le test emploie un modèle de bruit du
  capteur, recalé sur le ciel de chaque pose, avec deux tolérances : une pour les cœurs d'étoiles (0,5 × le signal
  au-dessus du fond), une pour l'échelle photométrique (5 %). Le seuil est κ = 5 ; les voisins sont testés à 1,5 σ.
- C'est la famille de `driz_cr` et du « CCD noise model » de PixInsight [P1], **mais les constantes sont maison**.
- Taux rejeté mesuré sur le banc : 0,001 à 0,05 % avec le moteur actuel, contre 0,05 à 5 % avec le précédent. Dans
  le même temps, les points d'une seule couleur ont augmenté (Rosette : de 23 à 77 par degré²). Le tutoriel de Siril
  dit de « vérifier que le taux de rejet est entre 0,1 et 0,5 % », 1 % pouvant aussi convenir [S12].

## 5. Pondération

**Ce que font les outils**
- **Siril** : pas de pondération par défaut. Les options : bruit (1 / (échelle² × bruit²)), FWHM pondérée, nombre
  d'étoiles, nombre d'images [S2][S6]. Avec « FWHM » et « nombre d'étoiles », la plus mauvaise pose reçoit un poids
  nul [S6].
- **PixInsight** :
  - **PSF Signal Weight**, le défaut de WBPP, mêle le flux des étoiles, leur concentration (donc la FWHM), le bruit et
    le fond. **PSF SNR** ne mesure que le signal sur bruit. **PSF Scale SNR** s'appuie sur la normalisation locale
    [P10][P4][P5].
  - Les auteurs écrivent que PSF Signal Weight « n'est pas prouvé comme la meilleure mesure » [P10].
  - En 1.9.6 : une protection contre les poids aberrants (au-delà de 10 fois le quartile haut) et un nouvel outil de
    mesure des poses, ImageAnalysis [P9].
- **APP** : poids égaux par défaut [A7] ; « qualité » (nombre d'étoiles × taille × rondeur, plus le bruit) conseillée
  au-delà de 40 poses [A6].
- **DSS** : sa « moyenne pondérée adaptative » pondère chaque pixel selon son écart à la moyenne (Stetson), pas
  chaque pose selon sa qualité ; sa moyenne pondérée par l'entropie (German et al. 2005) vise le HDR [D1][D3].

**Ce que dit la littérature**
- Zackay et Ofek (2017) [L6] :
  - pour un objet étendu limité par le fond du ciel (une nébuleuse), le poids optimal d'une pose normalisée est
    transparence² / bruit² : c'est l'inverse de la variance ;
  - pour les étoiles, la taille de la tache compte : poids d'Annis et al. (2014), transparence / (taille² × variance du
    fond) sur les poses brutes ;
  - leur coaddition « propre », qui filtre chaque pose par sa propre tache, gagne environ 5 % de S/B en simulation sur
    les étoiles, et 8 à 18 % face aux poids classiques (17 à 47 % face à pas de poids) sur les données réelles du PTF
    [L6][L7].
- Hubble : poids par l'inverse de la variance (IVM) quand le fond domine le bruit [L2].

**Chez nous** : (transparence / bruit)² × (FWHM de référence / FWHM)², sur des poses normalisées. C'est exactement
le poids d'Annis et al. 2014. Calcul de ce passage, sur les poses gardées par le banc : le terme de finesse coûte de
0,3 à 0,9 % de bruit sur la nébuleuse (+0,003 à +0,009 mag), contre l'inverse de la variance. Des poids égaux (le
défaut de Siril) coûteraient 8,4 % sur Iris en Express, 4,5 à 5 % sur la Rosette, 2,9 % sur Heart, rien sur M51, M8
et le Seestar.

## 6. Sélection des poses

**Ce que font les outils**
- **Siril** garde tout par défaut. Ses filtres (FWHM, FWHM pondérée, rondeur, fond, nombre d'étoiles) s'expriment en
  valeur, en pourcentage à garder, ou en k × σ [S2]. Son script Seestar écarte les poses aux étoiles les moins rondes.
- **PixInsight** : WBPP garde tout, avec un poids minimal de 0,05 [P15]. L'outil de sélection n'a pas d'expression
  par défaut [P21]. ImageAnalysis (1.9.6) détecte les traînées, les nuages et les étoiles allongées [P9].
- **APP et DSS** : garder un pourcentage des meilleures poses, selon un score [A6][D4].

**Ce que disent les mesures**
- Zackay et Ofek : « the rejection of images becomes a damaging operation » quand les poids sont justes [L6].
- Le « lucky imaging » demande des poses de quelques dizaines de millisecondes [L14] : il ne s'applique pas aux poses
  du ciel profond.
- Notre banc : en Express sur la Rosette, Siril va 0,25 mag plus profond que nous, qui avons écarté 14 poses sur 30.
- Calcul de ce passage : avec le bruit et la transparence mesurés à l'alignement, et des poids par l'inverse de la
  variance, garder toutes les poses alignées de la Rosette baisserait le bruit d'un facteur 0,914, soit environ
  **0,10 mag** de profondeur, en Express comme en Base. Ailleurs, rien ne change. Ce calcul suppose que les poses
  voilées n'ajoutent pas de défauts (gradients) : c'est le travail de la normalisation locale et du rejet.

**Chez nous** : nous écartons les poses sous 0,6 × la transparence des meilleures, et celles dont la FWHM dépasse
2 fois la médiane. Ces seuils sont maison.

## 7. Recalage des couleurs entre elles

**Ce que font les outils**
- **PixInsight** : ChannelMatch déplace une couleur d'une valeur fixe, réglée à la main [P22]. Le mode « Separated
  RGB » de WBPP aligne chaque couleur de chaque pose sur la meilleure pose verte, avec correction de distorsion,
  « pour réduire l'aberration chromatique » ; il est éteint par défaut [P17][P15].
- **Siril 1.4** : « RGB align » après l'empilement seulement. La commande `rgbalign` de la 1.5, en développement,
  vise la réfraction atmosphérique différentielle [S10].
- **APP** : « Align channels », qui ne marche, selon ses utilisateurs, qu'en enregistrant puis rechargeant les poses ;
  l'intégration est sur la liste du développeur [A9].
- **Aucun outil** ne le fait à l'intérieur du drizzle.

**La physique**
- La dispersion atmosphérique écarte le bleu du rouge selon la verticale. Elle croît vite à basse hauteur et change
  donc pendant la nuit (Filippenko 1982) [L13].
- L'aberration chromatique latérale d'une lunette est fixe et croît vers les bords ; un Newton n'en a pas.

**Ce que disent les mesures** : aucune mesure publiée du gain. Les nôtres (décalages mesurés par le moteur sur le
banc, en pixels de sortie) :

| Cas | Rouge au centre (x / y) | Rouge au coin (max) | Bleu au coin (max) |
|---|---|---|---|
| Heart (lunette, Base, échelle 1,5) | 0,17 / 0,08 | 1,33 | 0,17 |
| M8 (ASI2600MC, échelle 1) | 0,33 / 0,28 | 1,49 | 0,24 |
| Iris (Newton, Base) | 0,15 / 0,19 | 0,29 | 0,29 |
| Rosette (Newton, Base) | 0,16 / 0,02 | 0,20 | 0,25 |
| Seestar (Base) | 0,00 / 0,16 | 0,18 | 0,24 |
| M51 (Newton 200/1000, Base) | 0,02 / 0,03 | 0,05 | 0,05 |

Les coins de la lunette et de M8 montrent l'aberration latérale (près d'un pixel du capteur). Le centre montre la
dispersion (0,1 à 0,3 pixel). Le Newton de M51 n'a presque rien, comme attendu.

**Chez nous** : une surface du 2e degré par couleur, mesurée sur les étoiles de la première moyenne, puis appliquée
au moment de déposer chaque échantillon. Il n'y a donc pas d'interpolation de plus. C'est l'équivalent du mode
« Separated RGB » de WBPP, en une passe. Limite : une seule correction pour toute la nuit, alors que la dispersion
suit la hauteur de l'objet.

---

## 8. Les réglages de chaque outil

### Siril 1.4 (version actuelle 1.4.5)

| Étape | Réglage (commande) | Choix | Défaut | Source |
|---|---|---|---|---|
| Calibration | correction cosmétique `-cc=dark [bas haut]` | seuils en σ ; 0 = coupé | pixels chauds seuls, 3 σ | [S9] |
| Calibration | `-cfa`, `-equalize_cfa`, `-debayer` | oui / non | non (le script couleur officiel met les trois) | [S9][S7] |
| Dématriçage | algorithme | bilinéaire, VNG4, AHD, AMaZE, DCB, HPHD, IGV, LMMSE, **RCD** ; X-Trans : Markesteijn | RCD ; Markesteijn 1 passe | [S6][S11] |
| Alignement | méthode | globale (étoiles), 2 passes, 1-2-3 étoiles, motif, KOMBAT, comète, manuelle, astrométrique | globale | [S3] |
| Alignement | transformation `-transf=` | translation, similitude, affine, homographie | homographie | [S4] |
| Alignement | interpolation `-interp=` | aucune, plus proche, bilinéaire, bicubique, aire de pixel, Lanczos-4 | Lanczos-4 | [S3][S6] |
| Alignement | écrêtage (`-noclamp` pour l'enlever) | oui / non | oui | [S4] |
| Alignement | `-minpairs=` / `-maxstars=` | ≥ 4 / 100-2000 | 10 / 2000 | [S4] |
| Alignement | couche `-layer=` | 0, 1, 2 | vert | [S4] |
| Alignement | distorsion `-disto=` | image, fichier, maître | aucune | [S4] |
| Alignement | cadrage `-framing=` | courant, min, max, centre de gravité | courant | [S3] |
| Drizzle | `-drizzle`, `-scale=` | réel (0,1-3) | échelle 1 | [S4] |
| Drizzle | `-pixfrac=` | réel | 1,0 | [S4] |
| Drizzle | `-kernel=` | point, turbo, carré, gaussien, lanczos2, lanczos3 | carré | [S4] |
| Drizzle | `-flat=` | maître de flat | aucun | [S4] |
| Empilement | méthode | somme, moyenne avec rejet, médiane, max, min | somme (sans argument) ; « rej » dans le script | [S2][S7] |
| Empilement | rejet | aucun, percentile, sigma, MAD, sigma médian, **winsorisé**, linear fit, GESDT | winsorisé | [S2] |
| Empilement | seuils bas / haut | réels | 3 / 3 ; linear fit 5 / 5 ; GESDT 0,3 / 0,05 ; percentile 0,2 / 0,1 (le code stocke 3 / 3 : ambigu) | [S6] |
| Empilement | normalisation `-norm=` / `-nonorm` | additive, multiplicative, additive + échelle, multiplicative + échelle | additive + échelle | [S2] |
| Empilement | `-fastnorm` | IKSS ou médiane / MAD | IKSS | [S1] |
| Empilement | pondération `-weight=` | bruit, FWHM pondérée, nombre d'étoiles, nombre d'images | aucune | [S2][S6] |
| Empilement | filtres `-filter-fwhm/-wfwhm/-round/-bkg/-nbstars/-quality/-incl` | valeur, %, k × σ | aucun | [S2] |
| Empilement | `-rgb_equal`, `-output_norm`, `-32b` | oui / non | non (le script met les trois) | [S2][S7] |
| Empilement | `-maximize`, `-overlap_norm`, `-feather=`, `-upscale` | mosaïques, agrandissement ×2 | non | [S2] |
| Empilement | cartes de rejet `-rejmap` / `-rejmaps` | oui / non | non | [S2] |

Le script couleur officiel (`OSC_Preprocessing.ssf`) : `register pp_light` (homographie, Lanczos-4 écrêté, vert) puis
`stack r_pp_light rej 3 3 -norm=addscale -output_norm -rgb_equal -32b`, soit winsorisé 3/3, additive + échelle, **sans
pondération** [S7]. Sa variante Bayer drizzle : `register pp_light -drizzle -scale=1.0 -pixfrac=1.0 -kernel=square`
[S7]. La 1.4.5 corrige un défaut du GESDT qui rejetait toujours les deux valeurs extrêmes [S8].

### PixInsight 1.9.6

**ImageIntegration** (module 1.8.0). Les valeurs par défaut viennent de la doc de 2013 et du code de 2016 ; WBPP
impose les siennes, données à côté.

| Réglage | Choix | Défaut | Source |
|---|---|---|---|
| Combinaison | moyenne, médiane, min, max, **biweight** (1.9.6) | moyenne | [P9][P20] |
| Normalisation | aucune, additive, multiplicative, additive + échelle, multiplicative + échelle, **locale**, **adaptative** | additive + échelle ; WBPP : locale | [P1][P3][P4] |
| Pondération | indifférente, temps de pose, bruit, signal, médiane, moyenne, mot-clé, **PSF Signal Weight**, PSF SNR, PSF Scale SNR, propriétés standard (1.9.6) | bruit (doc 2013) ; PSF Signal Weight dans WBPP (défaut du processus : non vérifié) | [P1][P4][P5][P9][P15] |
| Estimateur d'échelle | AvgDev, MAD, BWMV (Sn, Qn, IKSS retirés en 1.8.8-6) | WBPP : BWMV | [P3][P15] |
| Poids minimal | 0 à 1 | WBPP : 0,05 | [P17] |
| Protection des poids aberrants | > 10 × le quartile haut → poids médian | coupée ; active dans WBPP | [P9] |
| Rejet | aucun, min/max, percentile, sigma, winsorisé, sigma moyenné, linear fit, modèle de bruit CCD, ESD généralisé, Robust Chauvenet | aucun (doc 2013) ; WBPP : auto | [P1][P2][P4] |
| Sigma bas / haut | | 4 / 2 (code 2016) ; WBPP 4 / 3 | [P20][P15] |
| Winsorisation | coupure | 5 σ (non vérifié) | [P1] |
| Linear fit bas / haut | | 5 / 2,5 (2016) ; WBPP 5 / 3,5 | [P20][P15] |
| Percentile bas / haut | | 0,2 / 0,1 | [P20] |
| Min / max | nombre bas / haut | 1 / 1 | [P20] |
| ESD | part de valeurs aberrantes / signification / relâche du bas | 0,3 / 0,05 / 1,5 (la relâche : rapport d'utilisateur) | [P2] |
| Robust Chauvenet | limite | 0,5 ; WBPP 0,1 | [P4][P15] |
| Modèle de bruit CCD | gain / lecture / bruit d'échelle | 1 e-/ADU / 10 e- / 0 | [P20] |
| Normalisation du rejet | échelle + décalage, égaliser les flux, adaptative | échelle + décalage | [P1][P3] |
| Rejet à grande échelle | couches, croissance, bas / haut | coupé ; 2, 2 | [P15] |
| Écrêtage de plage | bas / haut | bas oui, 0 ; haut non, 0,98 | [P20] |

**WBPP 3.x** (le script de prétraitement) : rejet auto (< 6 poses : percentile ; 6 à 15 : winsorisé ; 16 et plus :
linear fit) ; pondération PSF Signal Weight ; normalisation locale active, référence = intégration des 20 meilleures
poses, taille = quart du plus grand côté ; alignement en Auto, écrêtage 0,3, distorsion coupée ; dématriçage VNG ;
mode « Separated RGB » coupé ; drizzle à l'échelle 1, goutte 1,0 (couleur) ou 0,9 (mono), noyau carré, grille 16,
drizzle rapide actif ; passage à FastIntegration au-delà de 150 poses (200 en 3.2) [P15][P16][P17][P9].

| Processus | Réglage | Choix | Défaut | Source |
|---|---|---|---|---|
| LocalNormalization | échelle | 32 à 8192 pixels | 1024 | [P11] |
| LocalNormalization | évaluation de l'échelle | flux PSF, analyse multi-échelle | flux PSF | [P4][P11] |
| LocalNormalization | rejet des valeurs aberrantes, pixels chauds, réduction du bruit | | oui, 2 pixels, non | [P11] |
| LocalNormalization | seuils référence / cible ; écrêtage haut | | 3 / 3,2 ; 0,85 | [P11][P5] |
| LocalNormalization | S/B minimal de détection ; limite de rejet des échelles | | 40 ; 0,3 | [P5] |
| DrizzleIntegration | échelle / goutte | | 2 / 0,9 (code 2016) ; WBPP 1 / 1,0 en couleur | [P19][P15] |
| DrizzleIntegration | noyau | carré, circulaire, gaussien, forme variable | carré | [P7][P15] |
| DrizzleIntegration | rejet, pondération, surfaces splines, normalisation locale ou adaptative | oui / non | oui, oui, oui ; locale si disponible | [P19][P3] |
| DrizzleIntegration | drizzle CFA, motif | auto | activé par WBPP pour la couleur | [P15][P18] |
| DrizzleIntegration | drizzle rapide | oui / non | oui (1.9.0) | [P7] |
| StarAlignment | modèle | projectif, surfaces splines, plaques minces | projectif | [P13][P7] |
| StarAlignment | interpolation | Auto, Lanczos-3/4/5, spline bicubique, bilinéaire, plus proche, Mitchell-Netravali, Catmull-Rom, B-spline | Auto (Lanczos-4 sans changement d'échelle) | [P13] |
| StarAlignment | écrêtage | 0 à 1 | 0,3 | [P13] |
| StarAlignment | données drizzle | oui / non | non (non vérifié) ; WBPP : oui avec drizzle | [P15] |
| Debayer | méthode | SuperPixel, bilinéaire, VNG | VNG dans WBPP | [P12][P15] |
| FastIntegration | rejet | winsorisé, sigma (par lots) | winsorisé | [P14] |
| FastIntegration | pondération | coupée ; PSF SNR ou résolution | coupée | [P14] |
| FastIntegration | modèle, normalisation | homographie seule ; échelle + décalage globaux | | [P14] |
| ImageAnalysis (1.9.6) | mesures | FWHM, excentricité, signal PSF, S/B, bruit, fond, entropie, résidu PSF ; traînées, nuages, allongement | écrit un poids par couleur | [P9] |
| SubframeSelector | mesures | FWHM, excentricité, SNRWeight, médiane, bruit, étoiles, PSF Signal Weight, PSF SNR… | expressions vides ; « déconseillé » depuis 1.9.6 | [P21][P9] |
| ChannelMatch | décalage X / Y, facteur, par couleur | | 0 / 0, 1 | [P22] |

### APP et DSS

| Outil | Réglage | Choix | Défaut | Source |
|---|---|---|---|---|
| APP | mode | moyenne, médiane, auto | moyenne conseillée (défaut non vérifié) | [A1] |
| APP | rejet | 8 filtres : sigma, winsorisé, variantes MAD, variantes avec médiane locale (le linear fit retiré en 1.057) | conseil : κ 3, 1 itération | [A4][A5] |
| APP | protection de la diffraction | nombre maximal de rejets par pixel | conseil : 5 à 10 % des poses | [A5] |
| APP | pondération | égale, qualité, S/B, forme des étoiles | égale | [A7] |
| APP | LNC | degré 1 à 4, itérations | coupée (non vérifié) ; conseil degré 1, 1 à 2 itérations | [A2] |
| APP | MBB (fondu des bords) | % de chaque pose | conseil 5 à 15 % | [A3] |
| APP | dématriçage | AAD et autres | AAD (selon les utilisateurs) | [A8] |
| APP | drizzle | noyaux point, carré, tophat, gauss ; goutte ; échelle | Bayer : échelle 1, goutte 2,0, tophat (conseil) | [A8] |
| APP | interpolation | Lanczos-3 (avec protection), Mitchell-Netravali | Lanczos-3 (selon les utilisateurs) | [A10] |
| APP | recalage des couleurs | « Align channels » | coupé | [A9] |
| DSS | méthode | moyenne, médiane, max, kappa-sigma, kappa-sigma médian, moyenne pondérée adaptative, moyenne pondérée par l'entropie | moyenne | [D2] |
| DSS | kappa / itérations | | 2,0 / 5 | [D2] |
| DSS | dématriçage | bilinéaire, AHD, super-pixel, Bayer drizzle | bilinéaire | [D1][D2] |
| DSS | drizzle | ×1, ×2, ×3 | ×1 (coupé) | [D2] |
| DSS | sélection | X % des meilleures, ou seuil de score | | [D4] |

---

## 9. Ce que nous ferons

### 9.1 La méthode par défaut : drizzle Bayer, échelle 1, goutte 1

- **Drizzle Bayer par défaut** (ta décision), réglé comme Siril, WBPP et APP le règlent pour la couleur : **échelle 1,
  goutte 1, noyau carré** [S5][P15][A8]. Il garde ses atouts (pas de dématriçage, couleurs justes, aucun
  rééchantillonnage) sans affamer le rouge et le bleu.
- **Échelle 1,5 ou 2 proposée** seulement si deux conditions publiées sont vraies :
  - l'image est sous-échantillonnée, seul cas où le drizzle rend de la finesse [L1][S5]. Repère usuel : FWHM sous
    2 pixels environ (l'échantillonnage de Nyquist), mesurée par ajustement d'une tache (Moffat) sur la première
    pile, et non sur des super-pixels ;
  - la couverture suffit : la carte des poids du rouge et du bleu varie de moins de 20 % (règle de DrizzlePac [L2]).
    Elle se calcule avant l'empilement, avec les vraies positions de chaque pose.
- **Méthode « Classique » en second choix**, pour quand la couverture manque à l'échelle 1 (peu de poses, pas de
  décalage entre les poses) : dématriçage RCD, homographie, Lanczos-4 écrêté, comme Siril [S3][S11]. C'est aussi
  l'étalon du banc face à Siril.

### 9.2 Le rejet selon le nombre de poses

**Drizzle** : la méthode de Hubble, `driz_cr`, avec ses seuils publiés (3,5 puis 3,0 σ ; 1,2 puis 0,7 × la pente ;
extension 1 pixel) [L4]. Ce qui change avec le nombre de poses, c'est la prédiction [L3]. Le compte se fait par
couleur et par pixel : à l'échelle 1, un pixel rouge n'est couvert que par un quart des poses.

| Poses couvrant un pixel (par couleur) | Prédiction | Source |
|---|---|---|
| 1 ou 2 | pas de rejet possible, et on le dit | – |
| 3 à 5 | *minmed* : la médiane, ou le minimum là où la médiane s'en écarte trop | [L3] |
| 6 et plus | médiane | [L3] |

**Classique** (la règle de WBPP, et les seuils de Siril pour le winsorisé) :

| Poses | Rejet | Seuils | Source |
|---|---|---|---|
| 3 à 5 | percentile | 0,2 / 0,1 | [P1][P15][S1] |
| 6 à 15 | sigma winsorisé | 3 / 3 | [S2][P15] |
| 16 et plus | linear fit | 5 / 3,5 | [P15] |
| option au-delà de 50 | GESDT | 0,3 / 0,05 | [S1] |

Dans les deux méthodes, le taux de rejet est affiché à côté du repère de Siril (0,1 à 0,5 %) [S12], et la carte de
rejet est dans l'Inspection.

### 9.3 La pondération

- **Par défaut : l'inverse de la variance, par couleur** (transparence² / bruit², sur les poses normalisées). C'est
  l'optimum pour la nébuleuse [L6], le poids IVM de Hubble [L2] et le poids « bruit » de Siril [S2]. Il faut une
  transparence par couleur (voir 9.4).
- **Option « Étoiles fines »** : le poids d'Annis et al. 2014, notre formule actuelle [L6]. Son coût est affiché
  (moins de 1 % de bruit sur le banc).
- **Option « Aucune »**, comme le défaut de Siril.
- **Garde-fou** : un poids supérieur à 10 fois le quartile haut est ramené à la médiane (PixInsight 1.9.6) [P9].
- **À suivre** : la coaddition « propre » de Zackay et Ofek [L7] (+5 à 18 % de S/B sur les étoiles), qui demande un
  filtrage par la tache de chaque pose et ne sait pas encore rejeter. C'est un chantier pour plus tard, pas un défaut.

### 9.4 La normalisation locale : oui

- **Active par défaut**, comme WBPP [P15]. On garde notre forme : fond du 2e degré par couleur (la LNC degré 2
  d'APP [A2]), à deux corrections près :
  - **l'échelle photométrique par couleur**, mesurée sur les étoiles de chaque couleur, comme PixInsight et Siril
    [P11][S1] ;
  - **la référence = l'intégration des meilleures poses** (WBPP : 20) [P15], plutôt qu'une seule pose.
- **Option « Globale (comme Siril) »** : additive avec mise à l'échelle, IKSS [S1].

### 9.5 Les réglages à montrer

**La question : « Siril propose ces réglages en détail, avec beaucoup de paramètres : que feras-tu ? »**

Siril montre une vingtaine de réglages, avec des valeurs fixes et sans explication. Nous en montrons **6**. Chacun
est pré-rempli par une valeur automatique et sa raison. Tout ce que Siril offre pour un empilement simple reste
accessible en « Avancé ». Les options de mosaïque (`-maximize`, `-feather`, `-overlap_norm`) attendront les
mosaïques.

| Visible | Choix | Défaut |
|---|---|---|
| Méthode | Drizzle Bayer / Classique (comme Siril) | Drizzle Bayer |
| Échelle | 1× / 1,5× / 2× | 1×, sauf règle 9.1 |
| Rejet | Auto (selon les poses) / liste des méthodes | Auto |
| Pondération | Bruit / Étoiles fines / Aucune | Bruit |
| Recalage des couleurs | oui / non | oui |
| Poses | toutes les poses alignées / filtres | toutes, avec la liste des poses écartées et leur raison |

| Avancé | Choix | Défaut | Source |
|---|---|---|---|
| Taille de goutte | 0,5 à 1,0 | 1,0 à l'échelle 1 ; sinon la plus petite valeur ≥ 1 / échelle qui respecte les 20 % | [S5][L2] |
| Noyau | carré / gaussien | carré | [S5][P15] |
| Prédiction du rejet (drizzle) | auto / médiane / minmed | auto (9.2) | [L3] |
| Seuils du rejet (drizzle) | σ, pente, extension | 3,5 et 3,0 ; 1,2 et 0,7 ; 1 | [L4] |
| Seuils du rejet (classique) | bas / haut | selon la méthode (9.2) | [S2][P15] |
| Modèle de bruit | gain, bruit de lecture | mesurés (darks et poses) | [L4] |
| Normalisation | locale / globale / aucune ; degré 1 ou 2 | locale, degré 2 | [P15][A2] |
| Référence de normalisation | meilleures poses / une pose | 20 meilleures poses | [P15] |
| Filtres de poses | FWHM, rondeur, transparence ; en valeur, % ou k × σ | aucun | [S2] |
| Poids minimal | 0 à 0,2 | 0,05 | [P17] |
| Dématriçage (classique) | RCD, VNG, AMaZE, bilinéaire | RCD | [S11] |
| Interpolation (classique) | Lanczos-4, bicubique, bilinéaire ; écrêtage | Lanczos-4, écrêtage oui | [S3][P13] |
| Recalage des couleurs : suivi | toute la nuit / par bloc de poses | par bloc (à valider) | [P17][L13] |
| Motif fixe du capteur | oui / non | non | (méthode maison, voir 10) |

### 9.6 Les suggestions automatiques et leurs raisons

| Réglage | Règle | Raison affichée (exemple) | Source |
|---|---|---|---|
| Méthode | Drizzle si, à l'échelle 1, la carte des poids rouge et bleu varie de moins de 20 % ; sinon Classique | « 12 poses peu décalées : la couverture du rouge varie de 34 % ; Classique conseillé » | [L2][P12] |
| Échelle | 1,5 ou 2 si FWHM < 2 px (tache ajustée) **et** couverture < 20 % à cette échelle | « étoiles de 1,8 px : sous-échantillonné ; la couverture reste à 14 % : 1,5× » | [L1][L2][S5] |
| Goutte | 1 à l'échelle 1 ; sinon la plus petite valeur ≥ 1 / échelle qui garde la couverture sous 20 % | « 0,8 : la plus fine qui couvre le rouge et le bleu » | [S5][L2] |
| Rejet | selon les poses par pixel et par couleur (9.2) | « 9 poses par pixel rouge : prédiction médiane » | [L3][P15] |
| Pondération | Bruit ; le coût de l'option « Étoiles fines » est mesuré et affiché | « Étoiles fines : +0,6 % de bruit pour des étoiles 3 % plus fines » | [L6] |
| Normalisation | locale ; la raison donne l'écart de fond mesuré entre les poses | « les fonds diffèrent jusqu'à 4 σ d'un bord à l'autre (lune) » | [P15] |
| Recalage des couleurs | oui dès 50 étoiles isolées ; la raison donne les décalages | « rouge 0,3 px au centre, 1,5 px aux coins » | [P17] |
| Poses | toutes les poses alignées ; propose d'écarter les poses aberrantes en rondeur (k × σ, comme Siril) | « 3 poses aux étoiles allongées (vent) : à écarter ? » | [L6][S2] |
| Gain, lecture | des darks et du ciel des poses (déjà fait) | « bruit d'une dark seule » | [L4] |

---

## 10. Notre empilement aujourd'hui

### Ce qui est déjà bon

| Élément | Pourquoi c'est bon | Source |
|---|---|---|
| Drizzle Bayer, sans dématriçage | la méthode que PixInsight recommande pour la couleur ; étoiles plus fines et couleurs plus justes sur le banc | [P12], BENCH |
| Chaque échantillon brut déposé à sa position exacte, par le polynôme d'alignement | aucune interpolation (PixInsight relit aussi les poses brutes pour son drizzle CFA) | [P18] |
| Rejet contre une prédiction avec le bruit du capteur | la famille de la méthode de Hubble et du « modèle de bruit CCD » de PixInsight | [L4][P1] |
| Poids (transparence / bruit)² × finesse | le poids publié d'Annis et al. 2014 ; coût de la finesse < 1 % | [L6] |
| Échelle photométrique par photométrie d'ouverture des étoiles | le principe de la normalisation de PixInsight (« PSF flux ») | [P4][P11] |
| Fond additif du 2e degré par couleur | une normalisation locale, comme la LNC d'APP et le défaut de WBPP | [A2][P15] |
| Recalage des couleurs dans le drizzle | l'équivalent de « Separated RGB » de WBPP, sans interpolation ; Siril ne le fait pas pendant l'empilement | [P17][S10] |
| Pixels chauds et défauts exclus de la pile (poids nul) plutôt que remplacés par leurs voisins | la manière de Hubble (masque des pixels défectueux) | [L2] |
| Cartes de rejet, deux demi-piles pour mesurer le bruit | Siril et PixInsight produisent aussi des cartes de rejet | [S2][P1] |

### Ce qui est fait maison, à remplacer

| Aujourd'hui (fait maison) | Remplacement publié | Source |
|---|---|---|
| Échelle auto : 1,5 dès 40 poses et FWHM < 5,5 px, 2 dès 150 poses et FWHM < 3 px ; goutte 0,7 | échelle 1 et goutte 1 ; plus seulement si FWHM < 2 px et couverture < 20 % | [S5][P15][L2] |
| FWHM mesurée sur des super-pixels 2×2 (3,8 px mesurés pour 1,8 px réels sur Heart) | ajustement d'une tache (Moffat) sur la première pile, comme la PSF de Siril et de PixInsight | [S3][P11] |
| Seuils de rejet : κ 5, voisins à 1,5 σ, nettoyage à 2 κ, tolérance des cœurs 0,5 × le signal, tolérance d'échelle 5 %, couverture minimale 3 | `driz_cr` : 3,5 / 3,0 σ, 1,2 / 0,7 × la pente, extension 1 ; prédiction médiane ou *minmed* | [L3][L4] |
| Prédiction = moyenne pondérée nettoyée | médiane (*minmed* pour 3 à 5 poses par pixel) | [L3] |
| Poses écartées sous 0,6 × la transparence des meilleures, ou à FWHM > 2 × la médiane | garder toutes les poses alignées, pondérées ; filtres en option, à la manière de Siril (%, k × σ) ; poids minimal 0,05 | [L6][S2][P17] |
| Une échelle photométrique pour les trois couleurs | une échelle par couleur | [P11][S1] |
| Référence de normalisation = une pose | intégration des meilleures poses | [P15] |
| Recalage des couleurs : une correction pour toute la nuit | par bloc de poses, comme WBPP qui aligne chaque couleur de chaque pose ; la dispersion suit la hauteur | [P17][L13] |
| Suppression du motif fixe (coupée) | la laisser coupée : aucun outil de référence ne le fait à l'empilement | – |

Hors du bloc d'empilement, mais lié aux points d'une seule couleur : le seuil des pixels chauds de la calibration
(0,6 × le bruit du ciel, plus les pixels chauds vus dans les poses) est lui aussi maison. Siril emploie 3 σ sur la
dark par défaut [S9].

## 11. À mesurer sur notre banc avant d'adopter

Aucun de ces changements ne doit être annoncé comme meilleur avant d'être mesuré (règle 17). Le bruit se compare à
grande échelle (facteur R [L1]). Une passe Express puis Base, 6 cas, en changeant une chose à la fois :

1. **Drizzle 1 / 1, contre 1,5 / 0,7 et Classique** : points d'une seule couleur, bruit aux deux échelles,
   profondeur, FWHM. C'est la mesure qui manque à toute la littérature (sous-échantillonné : Heart, Seestar ;
   bien échantillonné : Iris, Rosette, M51).
2. **Rejet `driz_cr`, contre le nôtre** : taux rejeté (repère 0,1 à 0,5 %), points d'une seule couleur, traînées.
3. **Garder toutes les poses** sur la Rosette (Express et Base) : profondeur, fond.
4. **Échelle photométrique par couleur** : erreur de couleur des étoiles face à Gaia.
5. **Recalage des couleurs**, coupé, pour toute la nuit, ou par bloc : erreur de couleur, FWHM en rouge et en bleu,
   franges colorées aux coins.

## Décisions à prendre

1. Drizzle par défaut à l'échelle 1 et la goutte 1, et 1,5 ou 2 seulement selon la règle 9.1 ?
2. Remplacer nos seuils de rejet par ceux de Hubble (`driz_cr`) ?
3. Garder toutes les poses alignées par défaut, et ne proposer l'exclusion qu'avec sa raison ?
4. Recalage des couleurs suivi au cours de la nuit (par bloc de poses) ?

---

## Sources

**Siril**
- [S1] Empilement : https://siril.readthedocs.io/en/latest/preprocessing/stacking.html
- [S2] Commande `stack` (1.4) : https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/commands/stack.rst
- [S3] Alignement (1.4) : https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/preprocessing/registration.rst
- [S4] Commande `register` (1.4) : https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/commands/register.rst
- [S5] Drizzle : https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
- [S6] Valeurs par défaut et formules (code 1.4) : https://gitlab.com/free-astro/siril/-/blob/1.4/src/core/settings.c,
  https://gitlab.com/free-astro/siril/-/blob/1.4/src/core/command.c,
  https://gitlab.com/free-astro/siril/-/blob/1.4/src/stacking/median_and_mean.c
- [S7] Scripts officiels : https://gitlab.com/free-astro/siril/-/blob/1.4/scripts/OSC_Preprocessing.ssf (et
  `OSC_Preprocessing_BayerDrizzle.ssf` dans le même dossier)
- [S8] ChangeLog 1.4 : https://gitlab.com/free-astro/siril/-/blob/1.4/ChangeLog ; annonce du drizzle :
  https://siril.org/download/2025-04-26-siril-1-4-0-beta1/
- [S9] Commande `calibrate` : https://gitlab.com/free-astro/siril-doc/-/raw/1.4/doc/commands/calibrate.rst
- [S10] `rgbalign` (1.5 en développement) : https://gitlab.com/free-astro/siril-doc/-/raw/main/doc/commands/rgbalign.rst
- [S11] Le choix de RCD : https://siril.org/2020/09/whats-new-in-siril-0.99.4/
- [S12] Tutoriel (taux de rejet de 0,1 à 0,5 %) : https://siril.org/tutorials/tuto-manual/

**PixInsight**
- [P1] ImageIntegration, doc 2013 : https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/obsolete/ImageIntegration/ImageIntegration.pidoc
- [P2] 1.8.8 (ESD) : https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-8-released.13632/
- [P3] 1.8.8-6 (normalisation adaptative) : https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-8-6-released.15028/
- [P4] 1.8.9 (normalisation locale, PSF Signal Weight, PSF SNR, Robust Chauvenet) : https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-9-released.18148/
- [P5] 1.8.9-1 (PSF Scale SNR) : https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-9-1-released.18527/
- [P6] 1.8.9-2 (FastIntegration) : https://pixinsight.com/forum/index.php?threads/pixinsight-1-8-9-2-released.21228/
- [P7] 1.9.0 : https://pixinsight.com/forum/index.php?threads/pixinsight-1-9-0-lockhart-released.24605/
- [P8] 1.9.5 : https://pixinsight.net/dev/index.php?articles/pixinsight-1-9-5-lockhart-released.21/
- [P9] 1.9.6 (biweight, ImageAnalysis, garde-fou des poids) : https://pixinsight.net/dev/index.php?articles/pixinsight-1-9-6-lockhart-released.28/
- [P10] Pondération (PSF Signal Weight, 2022) : https://pixinsight.com/doc/docs/ImageWeighting/ImageWeighting.html
- [P11] LocalNormalization : https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/LocalNormalization/LocalNormalization.pidoc
- [P12] Debayer : https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/Debayer/Debayer.pidoc
- [P13] StarAlignment : https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/obsolete/StarAlignment/StarAlignment.pidoc
- [P14] FastIntegration : https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/FastIntegration/FastIntegration.pidoc
- [P15] WBPP, code source (BPP-Global.js, BPP-FrameGroup.js, change-log) : https://gitlab.com/pixinsight/PJSR/-/tree/master/src/scripts/BatchPreprocessing
- [P16] WBPP 2.4.0 : https://pixinsight.com/forum/index.php?threads/wbpp-2-4-0-released.18182/
- [P17] WBPP 2.5.0 (« Separated RGB », poids minimal, drizzle couleur) : https://pixinsight.com/forum/index.php?threads/wbpp-2-5-0-released.19112/
- [P18] Données drizzle et drizzle CFA : https://www.pixinsight.com/developer/pcl/doc/html/classpcl_1_1DrizzleData.html
- [P19] DrizzleIntegration, paramètres (code 2016) : https://raw.githubusercontent.com/aleixpuig/PCL/master/src/modules/processes/ImageIntegration/DrizzleIntegrationParameters.cpp
- [P20] ImageIntegration, paramètres (code 2016) : https://raw.githubusercontent.com/aleixpuig/PCL/master/src/modules/processes/ImageIntegration/ImageIntegrationParameters.cpp
- [P21] SubframeSelector : https://gitlab.com/pixinsight/Reference-Documentation/-/raw/master/tools/obsolete/SubframeSelector/SubframeSelector.pidoc
- [P22] ChannelMatch (code) : https://gitlab.com/pixinsight/PCL/-/raw/master/src/modules/processes/Geometry/ChannelMatchInstance.cpp

**Astro Pixel Processor** (forum du développeur, https://www.astropixelprocessor.com/community/)
- [A1] Grandes piles : …/tutorials-workflows/light-integration-settings-with-large-number-of-subs/
- [A2] LNC : …/main-forum/lnc-order-vs-iterations-tradeoff/
- [A3] MBB : …/faq/mbb-multi-band-blending-what-does-it-and-when-do-you-change-the-settings-for-it/
- [A4] Version 1.057 (rejets à médiane locale) : …/release-information/astro-pixel-processor-1-057-ready-for-download/
- [A5] Rejet : …/main-forum/outlier-rejection-question/
- [A6] Score de qualité : …/faq/quality-score-caculation/
- [A7] Pondération par défaut : …/main-forum/integration-weighting/
- [A8] Bayer drizzle : …/tutorials-workflows/bayer-drizzle/
- [A9] « Align channels » : …/main-forum/colored-centering-of-the-stars/
- [A10] Correction dynamique de distorsion : …/faq/dynamic-distortion-correction/

**DeepSkyStacker** (https://raw.githubusercontent.com/deepskystacker/DSS/master/)
- [D1] Aide technique : …/Help/english/technical.htm
- [D2] Valeurs par défaut : …/DeepSkyStackerKernel/Workspace.cpp
- [D3] Moyenne pondérée adaptative : …/DeepSkyStackerKernel/DSSTools.h
- [D4] FAQ : …/Help/english/faq.htm

**Publications et Hubble**
- [L1] Fruchter et Hook 2002, *PASP* 114, 144 (drizzle) : https://arxiv.org/abs/astro-ph/9808087
- [L2] DrizzlePac Handbook v3 (goutte, couverture < 20 %, poids IVM, masques) : https://hst-docs.stsci.edu/drizzpac
- [L3] AstroDrizzle, image médiane (*minmed*, médiane) : https://drizzlepac.readthedocs.io/en/deployment/median.html
- [L4] AstroDrizzle, `driz_cr` : https://drizzlepac.readthedocs.io/en/latest/api/drizzlepac.drizCR.drizCR.html (code :
  https://drizzlepac-jhunk.readthedocs.io/en/latest/_modules/drizzlepac/drizCR.html)
- [L5] Casertano et al. 2000, *AJ* 120, 2747 (bruit corrélé du drizzle) : https://ui.adsabs.harvard.edu/abs/2000AJ....120.2747C/abstract
- [L6] Zackay et Ofek 2017, *ApJ* 836, 187 (pondération optimale, poids d'Annis et al. 2014, rejet des poses) : https://arxiv.org/abs/1512.06872
- [L7] Zackay et Ofek 2017, *ApJ* 836, 188 (coaddition « propre ») : https://arxiv.org/abs/1512.06879
- [L8] Test ESD généralisé (Rosner 1983), manuel NIST : https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
- [L9] Maples et al. 2018, *ApJS* 238, 2 (Robust Chauvenet Rejection) : https://arxiv.org/abs/1807.05276
- [L10] RCD (Luis Sanz Rodríguez) : https://github.com/LuisSR/RCD-Demosaicing
- [L11] Wronski et al. 2019 (fusion de rafales brutes) : https://arxiv.org/abs/1905.03277 ; Lafenetre et al. 2023, *IPOL* 13 : https://doi.org/10.5201/ipol.2023.460
- [L12] Farsiu, Elad et Milanfar 2006, *IEEE TIP* 15, 141 : https://people.duke.edu/~sf59/TIP_Demosaicing_Final.pdf
- [L13] Filippenko 1982, *PASP* 94, 715 (réfraction atmosphérique différentielle) : https://ui.adsabs.harvard.edu/abs/1982PASP...94..715F/abstract
- [L14] Smith et al. 2009 (lucky imaging) : https://arxiv.org/abs/0906.3041

**Nos mesures** : docs/pilot/BENCH.md ; les décalages de couleur, les taux de rejet, le coût des poids et le gain des
poses écartées viennent des fichiers du banc (`F:/astro_essais/bench/<cas>/ours_<niveau>/store/`), lus sans rien
modifier pendant ce passage.
