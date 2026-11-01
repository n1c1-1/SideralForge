# Spec : l'empilement, deux méthodes standard

**Décisions de l'utilisateur (1er novembre, 19 h), qui remplacent ce qui suit là où ça diffère :**
- le **drizzle est la méthode par défaut** ;
- le **recalage des couleurs (ChannelMatch) est activé par défaut** : c'est un atout majeur ;
- le but n'est pas de faire comme Siril, mais **mieux que Siril**. Siril est le plancher, pas le modèle ;
- avant d'aller plus loin : **une passe d'état de l'art sur l'empilement** (PixInsight, Siril, APP, DSS,
  publications), puis cette spec sera réécrite, avec les paramètres à exposer. Fait :
  docs/pilot/STATE_OF_THE_ART_STACKING.md ;
- **sélection des poses** : on **garde notre seuil de sélection**, comme Siril qui propose aussi de ne garder que les
  meilleures poses. Le défaut est peut-être trop sévère et sera mesuré. La vue « courbes + images » est appréciée :
  on y voit les passages nuageux.

**Idée pour plus tard (l'utilisateur) : ne plus juger une pose en tout ou rien, mais par tuiles.** Un nuage ne
couvre pas toute l'image. Pondérer, ou écarter, par zones de l'image garderait les parties claires d'une pose
partiellement voilée. Avant d'en faire quoi que ce soit, il faut une passe d'état de l'art sur la pondération
locale.

Statut : **à valider par l'utilisateur avant tout code.** Règle du projet (ENGINE_KNOWLEDGE 4b) : chaque méthode
repose sur un algorithme publié et utilisé, nommé ici. Rien n'est réglé sur nos images de test.

## Le principe

Le bloc Empilement propose **deux cartes de méthode** :

| Méthode | Référence | Pour qui |
|---|---|---|
| **Standard (comme Siril)**, par défaut | le traitement de Siril 1.4 pour capteur couleur (`OSC_Preprocessing`) | tous les cas ; c'est l'étalon |
| **Drizzle (Bayer)**, en option | Bayer drizzle (Fruchter & Hook 2002, *PASP* 114 ; l'option drizzle de Siril 1.3+, de DSS et de PixInsight) | données sous-échantillonnées, beaucoup d'images décalées (dithering) |

Ce qui reste commun aux deux : la calibration (darks, flats, offsets, correction des pixels chauds) et la mesure des
images (étoiles, transparence, sélection).

## Méthode Standard : exactement la chaîne de Siril

1. **Dématriçage** de chaque image calibrée : **RCD** (méthode par défaut de Siril depuis la 1.2).
2. **Alignement** : correspondance des étoiles (globale), puis rééchantillonnage **Lanczos-4 avec écrêtage**
   (« clamping », contre les anneaux autour des étoiles) sur l'image de référence. Ce sont les réglages par défaut
   de Siril.
3. **Normalisation** : **additive avec mise à l'échelle**, par canal (`-norm=addscale` de Siril).
4. **Rejet** : **Winsorized sigma clipping**, 3 σ en bas et en haut (`rej 3 3` de Siril). Autres rejets possibles en
   option ensuite : sigma clipping, linear fit clipping, GESD, tous dans Siril.
5. **Moyenne**, sans pondération par défaut. Option : pondération par le bruit (`-weight=noise` de Siril).
6. **Sortie** à la résolution du capteur, en 32 bits.

Comme c'est le même algorithme, ce chemin doit donner sur le banc **les mêmes mesures que Siril**, à la variation
d'un calcul à l'autre près. C'est son critère d'acceptation.

## Méthode Drizzle (Bayer)

- C'est l'empilement actuel du moteur : chaque échantillon brut, d'une seule couleur, est déposé directement sur une
  grille plus fine. Facteur d'échelle 1,5 ou 2, taille de goutte (pixfrac) 0,7, noyau carré.
- Rejet et pondération : ceux d'aujourd'hui (rejet contre une première prédiction, poids selon transparence, bruit
  et finesse), avec les corrections du 1er novembre.
- **Quand le proposer** (règle de STScI DrizzlePac et de la documentation de Siril) : seulement si l'image est
  sous-échantillonnée (étoiles de moins de 2 pixels environ de FWHM) **et** qu'il y a assez d'images décalées.
  Sinon, la carte Standard reste suggérée, avec la raison affichée.

## Option commune : recaler les couleurs entre elles

Le décalage du rouge et du bleu par rapport au vert (dispersion atmosphérique, chromatisme latéral) se corrige
en alignant chaque couleur sur le vert, comme le fait *ChannelMatch* dans PixInsight. C'est une option des deux
méthodes. **Elle est désactivée par défaut dans Standard**, pour rester identique à Siril. C'est à toi de décider.

## Ce qui sort du moteur actuel

- La correction n° 3 (« détail fin ») est déjà retirée (1er novembre).
- Le choix automatique « drizzle 1,5 dès 40 images » est remplacé par la règle ci-dessus.

## Dans le logiciel

- Bloc Empilement, deux cartes de méthode avec leur durée sur carte graphique et sur processeur, comme les autres
  blocs. Les réglages de chaque méthode sont dans le panneau : rejet et seuils, normalisation et pondération pour
  Standard ; échelle et taille de goutte pour Drizzle.
- Inspection › Empilement : carte de rejet, poids et normalisation, pour les deux méthodes.

## Validation

1. Banc Express puis Base, sur les 6 cas, en un seul passage.
2. **Standard** : mesures égales à celles de Siril sur chaque cas.
3. **Drizzle** : mesuré à côté, pour dire sur quels cas il apporte quelque chose (finesse des étoiles) et à quel
   prix (bruit).
