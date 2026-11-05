# Chantier : le retrait des étoiles — le plan

Ouvert le 4 novembre 2026. État de l'art (sources) : `etoiles_etat_de_lart.md`. Échantillons et mesures :
`tools/bench/stars_samples.py`, `tools/bench/stars_compare.py`, `F:\astro_essais\stars_compare\`.

## Les objectifs

1. **Deux sujets séparés.** La séparation en deux couches (sans étoiles, étoiles seules) et leur recombinaison dans
   le Post-traitement ; le retrait des étoiles lui-même.
2. **Une qualité égale ou supérieure à StarNet, sans en dépendre.** StarNet reste une option (s'il est installé), jamais
   une dépendance.
3. **La vitesse.** Aujourd'hui, environ 80 min de carte graphique pour Heart express (12 Mpx) ; StarNet : 20 s.
   Cible : **le bloc Étoiles en moins d'une minute sur la carte graphique, quelques minutes sur le processeur**, pour
   une image de cette taille.
4. **Hors ligne.** Plus de téléchargement des étoiles faibles de Gaia pour le cas courant (la panne de M8, voir
   `docs/TODO.md`) : l'extrait Gaia local et la détection sur l'image suffisent.

## Où passe le temps aujourd'hui (mesuré le 4 novembre)

| Étape | Heart express (12 Mpx) | Seestar (2 Mpx) | Ce qu'elle fait |
|---|---|---|---|
| Optique | 1 min 55 | 18 s | profil des étoiles dans le champ, couleurs décalées corrigées |
| Étoiles | 29 min | 45 s | ajustement du profil de chaque étoile Gaia, puis modèle des faibles |
| Sans étoiles | 7 min 45 | 5 min 35 | soustraction, réparations, filtre de Wiener ; dont un réseau de netteté appris sur l'image (4 000 pas, environ 5 min) |
| Réseau de retrait | 33 min | 29 min | un réseau appris de zéro sur l'image (11 000 pas) |
| Sans étoiles, version réseau | 7 min 45 | 5 min 45 | la même chaîne avec la couche du réseau |

Comme pour l'ancien débruitage, presque tout le temps va à des apprentissages refaits sur chaque image, et à
l'ajustement étoile par étoile.

## Le plan

**Étape 0 — Confirmer le diagnostic (ce soir, environ 18 h 30).** Les cinq épreuves et les planches sur Heart,
Rosette, Iris et M51 (les nébuleuses, là où StarNet est réputé effacer la nébulosité faible). Si notre méthode actuelle
y reste derrière StarNet comme sur Seestar, on passe à l'étape 2 ; sinon, on en reparle.

**Étape 1 — Les deux couches et leur recombinaison (indépendante du retrait).**
- Le moteur rend deux couches linéaires en nombres à virgule : sans étoiles S, étoiles = original − S, sans couper les
  négatifs (la somme redonne l'original exactement).
- Le Post-traitement les traite comme deux calques ordinaires : chacun son étirement, puis **Écran** par défaut,
  **Addition** en option ; l'intensité, l'étirement propre et la saturation se règlent sur le calque d'étoiles. Les
  curseurs « magiques » actuels partent.
- La source de S est un choix : notre méthode, ou StarNet s'il est installé (sa ligne de commande, `--linear`).
- Vitesse : rien à gagner ici, c'est une soustraction.

**Étape 2 — Notre retrait : masquer, puis combler avec un réseau pré-entraîné (la piste A).** Méthode publiée :
maskfill (2023), Madarász et al. (2025).
- *Le masque, sans ajustement étoile par étoile* : la position et la magnitude Gaia (extrait local), le rayon donné par
  le profil mesuré par l'étape Optique et la magnitude, plus les étoiles absentes de Gaia détectées sur l'image.
  C'est quelques secondes, au lieu des 29 min de l'étape Étoiles.
- *Le remplissage* : un réseau qui comble les zones masquées à partir de leur entourage, pré-entraîné une fois sur nos
  empilements (le jeu de données du débruitage : 76 paires, quatre niveaux de bruit), en masquant au hasard des zones
  sans étoiles. Aucune vérité ni StarNet n'est nécessaire. Appliqué en quelques secondes. Une courte adaptation à
  l'image en option, comme pour le débruitage, si les mesures la justifient.
- *Hors des masques, l'image n'est pas touchée* : rien n'est inventé ailleurs que là où il y avait une étoile.
- *Mesures* : l'épreuve des faux masques (des zones sans étoiles masquées, comblées, comparées à ce qu'il y avait :
  une vraie erreur, que personne ne publie pour StarNet), plus les cinq épreuves et les planches face à StarNet.
- *Le pré-entraînement* : une nuit, sur les deux cartes graphiques.
- *Si la piste A bute* (étoiles faibles ou champs très denses) : la piste B, un réseau qui retire directement des
  étoiles de synthèse posées sur nos empilements, au profil réel de chaque image et aux couleurs de Gaia.

**Étape 3 — La vitesse du reste du bloc.**
- Le réseau de retrait appris sur chaque image (33 min) sort du chemin par défaut : il est remplacé par l'étape 2.
- Le réseau de netteté (5 min par image) : pré-entraîné une fois, comme le débruiteur, ou retiré s'il n'apporte
  rien de mesurable.
- L'ajustement étoile par étoile (étape Étoiles, 29 min) n'est plus nécessaire au retrait ; il reste comme recours pour
  **resynthétiser les étoiles** quand la mise au point ou le suivi sont ratés (option).
- Le téléchargement des étoiles faibles de Gaia n'est fait que si cette option est choisie.
- Cible mesurée à la fin : le bloc Étoiles sur Heart express, carte graphique et processeur.

**Étape 4 — Dans l'interface, après chaque étape.** Les méthodes en cartes comme pour le débruitage, les deux calques
dans le Post-traitement, les temps affichés justes, les captures vérifiées ; les tests du bloc reconstruits à ce
moment-là (les anciens tests des étoiles restent écartés jusque-là).

## Étape 0 : le diagnostic (4 novembre, 5 cas ; M8 attend Gaia)

Tous partent du même original : les deux demi-empilements de notre étape (Optique), débruités, moyennés, balance des
blancs appliquée. Couche d'étoiles = original − sans étoiles. Résultats : `F:\astro_essais\stars_compare\results.json`,
planches `<cas>_sheet.jpg` (sans étoiles) et `<cas>_stars.jpg` (couches d'étoiles).

| Cas | Hors étoiles, pente (1 = intact) : StarNet / notre modèle / notre réseau | Écart de fond hors étoiles (% du ciel) | Texture à la place des étoiles G < 14 (médiane ; 9 sur 10 sous) | Lumière restante, G 16–18,5 sur le ciel |
|---|---|---|---|---|
| Heart | 0,95 / 1,40 / 1,43 | 0,00 / −3,3 / −2,9 | 1,61 ; 2,2 / 1,35 ; 2,4 / 2,65 ; 28 | 0,5 % / 0,1 % / 0,4 % |
| Rosette | 1,00 / 0,48 / 0,49 | 0,00 / −6,5 / −6,3 | 1,09 ; 1,4 / 1,30 ; 2,3 / 6,6 ; 66 | 4,1 % / 2,0 % / 0,3 % |
| Iris | 0,99 / 0,94 / 0,95 | 0,00 / −0,18 / −0,09 | 0,71 ; 0,9 / 1,32 ; 2,7 / 4,6 ; 49 | 3,9 % / 2,6 % / 0,4 % |
| M51 | 1,00 / 0,99 / 0,99 | 0,00 / −0,12 / −0,14 | 0,77 ; 0,9 / 1,85 ; 6,8 / 9,4 ; 99 | 6,3 % / 3,5 % / 0,5 % |
| Seestar | 1,00 / 1,00 / 1,00 | 0,00 / −0,09 / −0,08 | 0,88 ; 1,1 / 1,17 ; 2,0 / 1,36 ; 11 | 11,6 % / 13,2 % / 6,3 % |

Les petites galaxies de M51 (NGC 5198, IC 4263, NGC 5169, NGC 5173) : gardées de la même façon par tous (la plus
petite, NGC 5173, perd 25 à 35 % chez tous). Les deux couches de StarNet redonnent l'original à 0,1–0,5 σ près sauf
Seestar (8,6 σ : sa couche d'étoiles est plafonnée) ; les nôtres exactement, par construction.

Ce qu'on en tire :
- **Notre modèle retire plus que les étoiles** : sur Heart et Rosette, le modèle des étoiles trop faibles pour être
  mesurées et les halos lointains emportent une large lumière diffuse, et avec elle une grande partie de la nébuleuse
  faible (Rosette : presque noire ; la couche d'étoiles porte un voile gris). C'est l'échec connu de la soustraction
  de profils sur fond de nébuleuse (Ginsburg 2023).
- **Notre modèle laisse de la structure à la place des étoiles brillantes** (texture 1,2 à 1,9, jusqu'à 6,8) ; notre
  réseau appris sur l'image bien davantage (jusqu'à 99 : des artefacts francs autour de certaines étoiles), même s'il
  laisse le moins de lumière d'étoiles faibles.
- **StarNet** ne touche pas au fond hors des étoiles, laisse un peu plus d'étoiles faibles (jusqu'à 12 % de leur
  lumière), et lisse un peu les zones comblées (texture 0,7 à 0,9), sauf Heart (1,6).
- Le diagnostic de Seestar tient : notre méthode actuelle est derrière StarNet. Les étapes 1 à 3 du plan s'appliquent.

## Gaia : plus de blocage (4 novembre au soir)

L'étape Catalogues téléchargeait toujours les étoiles jusqu'à G 18,5 et 20,5, même sans bloc Étoiles ; une panne des
services Gaia (VizieR, GAVO, archive de l'ESA, plusieurs jours de suite) bloquait alors la Couleur. Désormais
(`astro_gpu/stages/catalogs.py`, réglage `deep_stars` imposé par le workflow) : sans bloc Étoiles, les étoiles
brillantes (G < 16) et leurs spectres XP viennent de l'extrait Gaia local, sans aucun accès réseau (M8 express : 3,4 s,
3 827 spectres, hors ligne strict) ; le catalogue profond n'est demandé qu'avec le bloc Étoiles. Une base Gaia complète
en local (environ 66 Go, Gaia DR3 en Parquet, `data.lsdb.io` ou `stpubdata` sur AWS) reste possible si le recours
« resynthèse des étoiles » en a besoin un jour ; pas maintenant.

## La piste B lancée (4 novembre, nuit)

Choix fait après la revue : un réseau qui retire directement les étoiles, entraîné une fois sur nos empilements
(`tools/bench/stars_train.py`), plutôt que masquer puis combler (la piste A bute sur les champs denses : le masque
couvre 78 % de Heart).

- **Les paires** : 23 champs du jeu de données (sans le banc, sans le set 53 non relu), 56 images en tout (les niveaux
  de profondeur d'un même champ), chacune avec son image sans étoiles : celle de StarNet, corrigée à la main sous
  Photoshop par Nicolas (15 champs retouchés, de 0,01 % à 2,7 % de l'image). Les pixels retouchés reviennent dans les
  données en défaisant l'étirement connu des TIFF ; les autres gardent la précision de StarNet.
- **Zones ignorées** (aucun apprentissage) : le halo d'Alnitak (impossible à effacer proprement), l'étoile de NGC 2023
  restée dans l'image sans étoiles, un nœud de M33 effacé (région H II ?) ; dans les autres niveaux d'un champ, les
  zones retouchées sur le niveau relu.
- **Étoiles de synthèse** (60 % des exemples) : posées sur l'image sans étoiles (vérité exacte) ou sur l'empilement
  lui-même ; profil de Moffat (Moffat 1969) à la largeur mesurée sur les étoiles du champ ou tirée plus large, auréole
  en r^-3 (Racine 1996), aigrettes une fois sur cinq, couleurs autour du blanc de la balance, saturation au niveau du
  champ, un peu de bruit de photons.
- **Le réseau** : NAFNet (Chen et al. 2022), celui de Dark Star et de SyQon, 29 M paramètres, perte L1 sur un
  étirement asinh fixe et connu (défait exactement à la sortie).
- **Deux entraînements cette nuit**, pour savoir ce qu'apportent les étoiles de synthèse : « mixed » (vraies paires et
  synthèse, carte 0, lots de 8, 9 h) et « real » (vraies paires seules, carte 1, lots de 4, 8 h 36). Validation pendant
  l'entraînement : des étoiles de synthèse posées sur les images sans étoiles de StarNet du banc (une vérité exacte,
  sur des champs jamais vus). À la fin, chacun produit les images sans étoiles du banc
  (`stars_compare/<cas>/net_<nom>_starless.fits`) et les cinq épreuves sont refaites (`stars_compare.py`).

## Le résultat et l'intégration (5 novembre)

- **« mixed » est retenu** (rapport : `docs/POINT_MATINEE_2026-11-05.md`) : moins d'étoiles faibles laissées que
  StarNet sur les cinq champs du banc, aigrettes et halos des étoiles brillantes enlevés, fond inchangé ; « real »
  laissait un trou sombre au cœur des étoiles saturées. Le réseau est livré dans `astro_gpu/resources/stars/`
  (`starless_nafnet.pt`, demi-précision, 58 Mo).
- **Le bloc Étoiles** (`stages/star_removal.py`) remplace toute l'ancienne chaîne (modèle d'étoiles Gaia, PSF, réseau
  appris projet par projet, image sans étoiles par soustraction, fond de ciel, préparation du rendu, étoiles redessinées
  dans le Post) : il lit l'image de la Sortie (débruitée, sans gradient, équilibrée en couleur, ciel retiré : ce que le
  réseau a vu à l'entraînement) et donne deux images, le fond sans étoiles et les étoiles seules (leur somme est
  l'image, au pixel près). 14 s sur Heart (dont 7 s de réseau) sur la 4060 Ti. L'ancien code est retiré, outils et tests
  compris ; les options enregistrées « psf » et « psf_neural » se lisent comme la nouvelle.
- **Dans le graphe** : le bloc Étoiles a deux sorties (le fond en haut, les étoiles en bas) ; une troisième section,
  **Composition**, reçoit les blocs **Étoiles et fond** qui les recombinent. Une sortie tirée sur le Post en fait un
  calque tel quel. Un bloc Étoiles et fond est un calque du Post (partie « combined ») : ses réglages sont les mêmes dans
  le panneau du graphe et dans le Post, synchronisés dans les deux sens.
- **Le rendu des deux images** (`render/stars.py`), selon l'état de l'art (outil de recombinaison de Siril, notes de
  RC Astro sur StarXTerminator, calque en Écran sous Photoshop) : chaque image étirée à part, les étoiles remises en
  Écran (ou Densité linéaire, ou Éclaircir). Par défaut, les deux suivent la courbe de l'image entière : « Étoiles et
  fond » au réglage suggéré ressemble à la Sortie. Curseurs : ciel, contraste, saturation pour le fond ; Étoiles
  (l'étirement des étoiles, la réduction qu'apporte la séparation), Taille (filtres Minimum / Maximum de Photoshop),
  Couleur.

## Le calendrier proposé

| Quand | Quoi |
|---|---|
| Ce soir | Étape 0 (les mesures des nébuleuses) ; si le diagnostic tient, préparer les données de remplissage et lancer le pré-entraînement pour la nuit |
| Demain matin | Mesurer le réseau de remplissage (faux masques, cinq épreuves, planches) face à StarNet et à notre méthode actuelle |
| Demain | Étape 1 (les deux couches dans le moteur et le Post-traitement), puis l'étape 2 dans le moteur et l'interface |
| Ensuite | Étape 3 (vitesse du reste), étape 4 (finitions, tests) |
