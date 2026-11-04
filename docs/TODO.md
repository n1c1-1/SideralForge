# À faire

Ouvert le 4 novembre 2026. Remplace les listes de tâches précédentes (rapports du matin, notes des nuits), devenues
obsolètes.

## Chantiers, dans l'ordre

Le débruitage est fait (4 novembre, `docs/chantiers/debruitage.md`) : un réseau pré-entraîné, adapté à l'image en
option ; meilleur que notre ancien apprentissage sur chaque image, que GraXpert et que Topaz sur le banc.

1. **Le retrait des étoiles** (état de l'art : `docs/chantiers/etoiles_etat_de_lart.md` ; échantillons StarNet et
   notre méthode : `tools/bench/stars_samples.py`, `F:\astro_essais\stars_compare`). Deux sujets : la séparation en
   deux couches (sans étoiles, étoiles seules) et leur recombinaison dans le Post-traitement par les modes de fusion et
   des filtres ordinaires, plus de curseurs « magiques » ; le retrait des étoiles lui-même. StarNet (installé par
   l'utilisateur) sert à comparer et pourra être une option, jamais une dépendance ; la resynthèse des étoiles devient
   le recours en cas de mise au point ou de suivi ratés.

## Plus tard

- **L'empilement face au jeu de données varié** (28 jeux, nuit du 3 au 4 novembre) : relever ce que la chaîne a mal
  fait, à partir du rapport `docs/chantiers/empilement_jeu_varie.md` (régénéré par
  `python tools/bench/denoise_dataset.py report`). Déjà vus : le monochrome et le X-Trans refusés ; l'ASI533 en poses
  de 0,5 s (14 poses sur 15 non alignées) ; le Seestar S50 sur M51 (trop peu d'étoiles, même au seuil 5) ; 9 % de
  pixels défectueux sur la DWARF mini ; un alignement moins précis sur le Celestron Origin (0,94 px).

- **La couleur** (fiche : `docs/chantiers/couleur.md`) : c'est désormais une balance des blancs photométrique, comme
  Siril (la matrice qui produisait le magenta est sortie du code, la sortie simple prend le niveau du ciel du bloc
  Gradient). Mesurée face à Siril le 4 novembre sur 5 cas (fiche, §5) : d'accord à 0,5 % avec SPCC, PCC s'écarte de 3
  à 15 %. Reste M8 (SPCC), quand les services Gaia répondent.
- **Les tests de l'interface** sur Heart express : 42 rouges (aucun défaut du code). Écarter ceux du débruitage, des
  étoiles et du workflow Complet (« à réintroduire avec leur chantier »), recaler les chiffres de l'ancien petit projet
  de test, mettre à jour ceux de l'ancien Gradient.
- **M8 express** : relancer `python tools/bench/gradient_projects.py --cases m8` quand les services Gaia répondent.
- **Les étoiles faibles de Gaia bloquent la couleur** (vu le 4 novembre sur M8 : VizieR injoignable, GAVO trop lent,
  archive de l'ESA en erreur 500) : l'étape Catalogues télécharge toujours en ligne les étoiles jusqu'à G 18,5 (plus
  profond que l'extrait Gaia local), alors qu'elles ne servent qu'au bloc Étoiles (`optics`, `stars`, `starless`) ; si
  ce téléchargement échoue, toute l'étape échoue et la Couleur, qui n'a besoin que des spectres XP présents sur le
  disque, ne se calcule pas. Siril n'a pas ce problème : sa calibration lit l'extrait local seul. À faire avec le
  chantier Étoiles : ne télécharger les étoiles faibles que si le bloc Étoiles est là, et ne pas bloquer la Couleur
  sans elles.
- **L'édition des échantillons et zones du Gradient** (Inspect), restes de la revue d'ergonomie du 4 novembre :
  l'aperçu aussi dans l'onglet « Modèle du ciel » ; compacter les touches de pinceau enregistrées (chaque coup de
  pinceau entre dans la définition du projet) ; la question « Remplacer / Essayer en variante » des autres réglages
  n'est pas posée (chaque coup de pinceau la poserait) : à revoir si le besoin d'une variante vient ; le réglage
  « Protection en bruit du ciel », sans effet au banc, à retirer (en attente de l'accord).
- **Les filtres « netteté » et « débruitage » du Post-traitement** ne changent rien à l'image d'après les captures
  (`ml_sharpen`, `ml_denoise`) : à regarder.
