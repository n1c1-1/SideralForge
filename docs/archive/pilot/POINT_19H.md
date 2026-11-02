# Point du soir, lundi 2 novembre, 19 h

Tes 9 projets n'ont pas été touchés, et l'ancienne version de l'empilement est toujours là. Tout est enregistré.

## En bref

- **Le GUI complet est prêt à tester** : les deux méthodes d'empilement, l'Inspection, le Post avec les Courbes et les
  nouveaux filtres, l'export. Vérifié écran par écran, en français et en anglais.
- **L'écart de notre drizzle face à celui de Siril est compris** : il n'y a pas de défaut dans notre drizzle. Une
  partie de l'écart venait de la comparaison, une autre de la version de Siril du banc, et une autre de deux erreurs
  dans la façon dont le banc mesurait. Ces deux erreurs sont corrigées, et tout le banc a été remesuré.
- **Les deux anomalies de l'empilement classique sont expliquées**, et celle de la couleur est corrigée.
- **Sept décisions t'attendent** (en bas de page). Pour chacune, je donne mon conseil.

## 1. Ce que tu peux tester

Lance **client.bat**. La page [CE_QUI_A_CHANGE.md](CE_QUI_A_CHANGE.md) dit où regarder, écran par écran.

- Accueil › section **« Banc d'essai face à Siril »** : ouvre **Iris** (le plus complet), **Rosette** (où le conseil
  est l'empilement classique) ou **Seestar**.
- Chaque projet a 24 parcours. Deux seulement comptent ce soir :
  - **PI-own** = notre **drizzle**, avec les réglages du jour ;
  - **CL** = notre **empilement classique**, avec les réglages du jour.

  Les autres viennent des essais d'hier ou sont des variantes pour le banc.
- La carte graphique est libre : tu peux lancer un calcul sur ces projets du banc si tu veux. Pas sur tes 9 projets,
  tant que l'on n'a pas décidé.

**Où voir les nouveautés du jour :**

| Où | Ce que tu vois | Image |
|---|---|---|
| Graphe › bloc Empilement | Deux cartes, **Drizzle** et **Empilement classique**. Un petit cadre « Conseillé » sur l'une, avec un « i » qui dit pourquoi | [drizzle conseillé](shots/2026-11-02/point19h/01_drizzle_conseille.jpg), [classique conseillé](shots/2026-11-02/point19h/02_classique_conseille.jpg) |
| Graphe › bloc Sélection | La règle en une phrase : toutes les poses sont gardées, sauf celles qui pèsent moins de 5 % de la meilleure | [sélection](shots/2026-11-02/point19h/03_selection.jpg) |
| Inspection › Empilement › Poids | Le poids de chaque pose ; une pose écartée dit pourquoi (Rosette : 26 gardées sur 30) | [poids](shots/2026-11-02/point19h/04_poids_poses_ecartees.jpg) |
| Inspection › Empilement | La carte de rejet, et la couverture du rouge et du bleu pour le drizzle | [rejet](shots/2026-11-02/point19h/05_carte_de_rejet.jpg), [couverture](shots/2026-11-02/point19h/06_couverture.jpg) |
| Inspection › Couleur | Avant / après, et la balance des blancs mesurée sur les étoiles | [couleur](shots/2026-11-02/point19h/07_couleur_balance.jpg) |
| Inspection › Astrométrie | Nouveau cet après-midi : le menu **« Image de fond »**. La grille du ciel et les objets nommés s'affichent sur l'empilement, l'image calibrée en couleur, l'image au gradient retiré ou l'**image finale** | [image finale](shots/2026-11-02/point19h/14_astrometrie_image_finale.jpg), [menu](shots/2026-11-02/point19h/15_astrometrie_menu_image_de_fond.jpg) |
| Inspection › Couleur, Gradient | Comparaison côte à côte sur le même ciel | [côte à côte](shots/2026-11-02/point19h/08_cote_a_cote.jpg) |
| Post › + Filtre | Le menu en 7 groupes, les **Courbes** (points, canaux, préréglages) et les nouveaux filtres : GHS, arcsinh, transformation d'histogramme, SCNR, CLAHE, accentuation et réduction du bruit multi-échelles | [menu](shots/2026-11-02/point19h/09_menu_filtres.jpg), [courbes](shots/2026-11-02/point19h/10_courbes.jpg), [GHS](shots/2026-11-02/point19h/11_ghs.jpg), [accentuation](shots/2026-11-02/point19h/12_accentuation.jpg) |
| Post › Exporter… | FITS (avec la solution astrométrique), TIFF, PNG, JPEG | [export](shots/2026-11-02/point19h/13_export.jpg) |

**Défauts connus, sans gravité :**
- l'éditeur de Courbes ne tient pas en entier sur un écran de 1 080 pixels de haut : il faut faire défiler le panneau ;
- un master dark s'affiche en rose. C'est seulement l'affichage d'une image presque uniforme, les données sont justes ;
- dans l'Astrométrie, changer d'image de fond remet le zoom à « image entière ».

## 2. Le banc : comment on mesure, et ce qui a été corrigé

Le banc compare, sur les mêmes 30 poses et le même ciel, six cas : Heart, Iris, M51, M8, Rosette, Seestar. Il met face
à face Siril (classique et drizzle) et nos deux méthodes. Toutes les mesures sont faites sur l'image linéaire finale,
après la calibration des couleurs et le retrait du gradient.

Deux erreurs dans la façon dont le banc mesurait faussaient certaines conclusions de ce matin. Elles sont corrigées
avec les méthodes des logiciels de photométrie (SExtractor, DAOPHOT), puis tout a été remesuré, sans refaire aucun
empilement :

- **La profondeur** (l'étoile la plus faible qu'on distingue) était mesurée dans de petits disques faits de pixels
  entiers. Quand les étoiles passaient d'une largeur de 2,17 à 2,24 pixels, le disque sautait de 13 à 21 pixels et
  ramassait 56 % de bruit en plus. La profondeur perdait jusqu'à 0,45 mag sans aucun changement réel de l'image.
  Maintenant les disques sont exacts au pixel près, et la lumière qui sort du disque est corrigée.
- **Le signal sur bruit de la nébuleuse ou de la galaxie** était mesuré sur des zones choisies sur l'image de Siril.
  Ces zones tombaient là où le bruit de Siril était, par hasard, le plus favorable. Maintenant les zones sont choisies
  sur une image indépendante, et toutes les images sont mesurées aux mêmes endroits.

Le détail est dans [BENCH.md](BENCH.md), section « Banc du 2 novembre », et les planches dans
[shots/2026-11-02/bench/](shots/2026-11-02/bench/).

## 3. Les résultats, avec les mesures corrigées

**Notre drizzle face au drizzle de Siril : nous gagnons presque partout.**
- Plus profond dans les 6 cas : de 0,06 à 0,30 mag.
- Grain fin plus bas dans 5 cas sur 6, étoiles un peu plus fines dans 4 cas, couleurs égales ou meilleures (Heart :
  nettement), sauf la Rosette.
- Signal sur bruit de la nébuleuse ou de la galaxie : en moyenne 3 % en dessous. Nous sommes devant sur le Seestar
  (+5 %) et la Rosette (+6 %), derrière sur Heart (−14 %) et M51 (−10 %). Sur ces deux cas, la mesure est fragile : la
  nébuleuse y est plus faible que les dégradés du ciel, et le résultat bouge de 10 à 15 % selon la façon de mesurer.
- Pourquoi l'écart de ce matin était plus grand ([DRIZZLE_GAP.md](DRIZZLE_GAP.md)) :
  - le matin, « notre drizzle » était encore l'ancienne version, avec une goutte plus petite et une sélection qui
    écartait trop de poses ;
  - le Siril du banc (1.4.0-beta3) empilait le drizzle sans le poids de chaque goutte. Son image était donc un peu
    lissée : le grain paraissait plus doux, mais les étoiles étaient 7 % plus larges. Siril l'a corrigé lui-même dans
    la version suivante. Refait avec ces poids, le drizzle de Siril a le même grain, la même finesse et la même
    profondeur que le nôtre ;
  - sur Iris, c'était la mesure du banc.

**Nos deux méthodes entre elles.**
- **Le drizzle est meilleur sur Heart et M8.** Sur Heart, il fait des étoiles plus fines, des couleurs plus justes et un
  fond plus plat.
- **Le classique est meilleur sur Iris, M51, le Seestar et la Rosette.** Il a moins de grain fin et un meilleur signal
  sur bruit. Il a aussi moins de points d'une seule couleur : 2 à 14 par degré carré, contre 10 à 24 pour le drizzle.
- C'est ce qu'on attend en général : le drizzle donne de la finesse et des couleurs justes, et le classique un grain
  plus doux. Une partie de ce grain plus doux vient du dématriçage, qui prête au rouge et au bleu le détail du vert.
- **Face à Siril classique, notre drizzle est un peu derrière en profondeur** (de 0,06 à 0,15 mag sur Iris, la Rosette
  et le Seestar ; égal sur les trois autres cas). **Notre classique, lui, est au niveau de Siril classique, ou devant.**

**Les anomalies de l'empilement classique** ([CLASSIC_ANOMALIES.md](CLASSIC_ANOMALIES.md)) :
- **L'équilibrage des couleurs avant le dématriçage « coûtait » 0,57 mag sur Heart.** C'était surtout l'erreur de mesure
  du banc (0,45 mag). Avec la mesure corrigée, il reste 0,17 mag. Ce reste n'est pas une vraie perte : sans
  équilibrage, le dématriçage recopie de la lumière verte dans le bleu des étoiles (+36 % de flux en bleu sur une pose
  de Heart). Les étoiles paraissent alors plus brillantes qu'elles ne sont, et la profondeur mesurée monte.
- **Sans équilibrage, Heart sortait orangée et granuleuse, alors que Siril, avec le même dématriçage, ne l'était pas.**
  La cause était notre étape couleur. Elle avait utilisé des courbes de capteur génériques, fausses de 18 % sur ce
  champ, et en avait tiré un mélange des couleurs qui amplifiait le grain. **C'est corrigé** : quand aucune courbe ne
  décrit le capteur à 5 % près, l'étape prend un facteur par couleur, comme Siril et PixInsight. Heart sans
  équilibrage a maintenant le même fond que Siril.

**Le rejet de PixInsight dans le drizzle** (essayé ce matin, en option) :
- il répare nettement la Rosette : des couleurs d'étoiles plus justes et un fond plus plat ;
- il ne change presque rien ailleurs, et il est un peu moins bon sur M8 et le Seestar ;
- il ne retire pas les points d'une seule couleur, parce que ces points sont présents dans toutes les poses ;
- il rend l'empilement 1,6 à 4 fois plus lent.

**Le conseil de méthode** tombe sur la meilleure de nos deux méthodes dans 3 cas sur 6 seulement. La règle lit la
variation de couverture du rouge et du bleu, avec un seuil à 20 %. Sur les six cas, cette variation va de 18 à 22 %,
juste autour du seuil : cette mesure seule ne sépare pas les cas.

**Par rapport à l'ancienne version**, les réglages du jour sont à égalité ou un peu meilleurs partout, en profondeur,
en grain fin et en signal sur bruit. La Rosette gagne 15 %, parce qu'elle garde 26 poses au lieu de 16. En
revanche, les étoiles sont 1 à 4 % plus larges (Heart : 12,55″ contre 12,07″).

## 4. Tes décisions

1. **Le rejet de PixInsight dans le drizzle : par défaut ou en option ?**
   *Mon conseil : en option.* Il ne répare qu'un cas sur six, il est un peu moins bon sur deux autres, et il est
   jusqu'à 4 fois plus lent. Pour les points d'une seule couleur, qui sont présents dans toutes les poses, il faudra
   chercher ailleurs. Je propose d'en faire une étude séparée.

2. **L'équilibrage des couleurs avant le dématriçage (empilement classique) : allumé ou éteint par défaut ?**
   *Mon conseil : allumé.* Il donne un meilleur signal sur bruit dans les six cas (+2 à +8 %) et des couleurs d'étoiles
   plus justes dans quatre. La profondeur qu'il « perd » est surtout de la lumière verte que le dématriçage recopiait
   dans le bleu. Éteint, on a exactement le dématriçage de Siril : le réglage reste disponible.

3. **La correction de l'étape couleur (un facteur par couleur quand les courbes du capteur ne collent pas) : on la
   garde ?**
   *Mon conseil : oui.* Quand on ne connaît pas le capteur, on fait comme Siril, sans grain ni teinte inventés.
   Elle ne touche que trois images du banc.

4. **Ajouter une option « couleurs comme Siril », c'est-à-dire un facteur par couleur, même quand les courbes du capteur
   sont bonnes ?**
   Notre étape couleur va plus loin que Siril : elle corrige les couleurs avec un mélange des trois couleurs du
   capteur. Les couleurs sont plus justes, mais ce mélange multiplie le grain de couleur, jusqu'à 4,5 fois selon le cas
   (Heart en drizzle : ×1,4 ; le Seestar et la Rosette : jusqu'à ×4).
   *Mon conseil : oui, en option.* On garde le réglage actuel par défaut tant qu'un banc ne dit pas mieux.

5. **Le conseil de méthode (le cadre « Conseillé ») : il ne tombe juste que dans 3 cas sur 6.**
   *Mon conseil : le garder visible, puisqu'il ne change jamais ton choix, et revoir sa règle sur tes 9 projets avant
   de lui faire confiance.* Ce sera plus de cas, dont le monochrome quand il arrivera. Autre choix possible : le masquer
   d'ici là.

6. **Mettre à jour le Siril du banc (1.4.0-beta3 → 1.4.5, la version actuelle) ?**
   Le drizzle de notre version de Siril n'utilise pas le poids des gouttes : on compare donc notre drizzle à un Siril
   plus ancien que celui de tes utilisateurs.
   *Mon conseil : oui.* C'est une installation sur ta machine : je ne la fais qu'avec ton accord. Ensuite, je relance
   seulement la partie Siril du banc ; je te donne sa durée avant de la lancer.

7. **Ensuite seulement, dans cet ordre :**
   - retirer l'ancienne version de l'empilement ;
   - recalculer les copies de test (20 tests en dépendent ; les 799 autres passent) ;
   - recalculer tes 9 projets, par lots, avec une vérification par toi entre deux lots.

**Petit point ouvert, non bloquant** : le vert d'Iris. Avec la méthode neutre de ce midi, il était 6 à 8 % sous Siril,
et je n'en ai pas trouvé la cause. Je ne l'ai pas recalculé avec la mesure corrigée de l'après-midi. Je propose de le
regarder seulement s'il revient avec le Siril à jour.
