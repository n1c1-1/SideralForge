# Point d'étape : ce que la nuit a produit

Préparé le 1er novembre 2026 vers 8 h 30, pour que tu relises à la main. Rien n'a été modifié dans tes projets : les
images ont été recalculées à part, à 1/4 de la résolution, sur le processeur (les deux cartes graphiques continuent
de travailler pour toi).

Tout ce qui est montré ici est dans `F:\astro_projets\_review\`. Si une image est trop petite dans l'aperçu,
ouvre-la seule (les liens sont donnés), ou regarde les versions plus grandes dans le dossier du projet
(`F:\astro_projets\_review\images\<projet>\`).

---

## 1. Tes images

Les quatre projets entièrement terminés (les quatre workflows calculés) : **Horsehead, Iris, Rosette, Flaming Star**.
Aucun autre projet n'est terminé en entier pour l'instant (voir le tableau de la partie 4).

Les quatre workflows de ces projets, en clair :

| Nom dans le logiciel | Ce que c'est |
|---|---|
| **Classic stack** | L'empilement, la couleur, un fond de ciel simple. Pas de réduction de bruit. Le plus rapide. |
| **Classic + denoise** | Pareil, avec la réduction de bruit. |
| **Full** | Le traitement complet : réduction de bruit, étoiles modélisées, fond de ciel guidé par l'H-alpha. |
| **Full + earlier neural layer** | Comme Full, mais en reprenant la couche d'étoiles apprise de tes anciens projets. |

Chaque feuille montre les quatre workflows avec leurs **réglages par défaut**, plus tes deux anciennes images
(« faithful » et « artistic », celles de ton ancien projet). Même cadrage partout. Ces feuilles ne sont pas à
l'aveugle : tout est étiqueté.

**Important :** les trois corrections d'image (partie 3) ne sont **pas encore appliquées**. Les images Classic
ci-dessous sont donc « avant correction ».

### Horsehead

![Horsehead](point_etape/feuille_Horsehead.png)

Regarde d'abord la bande rouge (IC 434) et la nébuleuse de la Flamme : Full et « Full + earlier neural layer » sont
presque identiques à ton ancienne « faithful », « artistic » est plus saturée. Dans les deux Classic, la nébuleuse est
violette, le halo bleu d'Alnitak est très fort et les coins du ciel tirent vers le jaune-olive.

Fichier seul : [feuille_Horsehead.png](point_etape/feuille_Horsehead.png)

### Iris

![Iris](point_etape/feuille_Iris.png)

Regarde le cœur bleu de la nébuleuse et les poussières brunes autour : « Full + earlier neural layer » doit ressembler
à ton ancienne « faithful ». Full est le plus sombre des quatre ; dans les deux Full presque toutes les étoiles
faibles sont effacées, comme dans ton ancienne image.

Fichier seul : [feuille_Iris.png](point_etape/feuille_Iris.png)

### Rosette

![Rosette](point_etape/feuille_Rosette.png)

Regarde les filaments au bord de la nébuleuse et la couleur : les deux Classic la rendent magenta sur un ciel
verdâtre, les deux Full la rendent rouge-rose comme ton ancienne image. Les étoiles sont presque absentes de Full
(un peu plus dans l'autre) : c'est aussi le cas dans ton ancienne image, mais dis-moi si c'est ce que tu veux.

Fichier seul : [feuille_Rosette.png](point_etape/feuille_Rosette.png)

### Flaming Star

![Flaming Star](point_etape/feuille_FlamingStar.png)

Regarde les filaments rouges au centre et le fond brun-rouge autour : Full et « Full + earlier neural layer »
ressemblent beaucoup à ton ancienne image. Dans les Classic, le ciel a des plaques vertes et jaunes très visibles.

Fichier seul : [feuille_FlamingStar.png](point_etape/feuille_FlamingStar.png)

### Le Classic stack de tous les projets calculés

![Classic stack de tous les projets](point_etape/feuille_classic_stack_tous.png)

Huit projets sur neuf ont leur Classic stack (M33 n'en a pas : son calcul a planté, voir partie 4). Regarde la
couleur des nébuleuses (magenta ou violet partout) et l'uniformité du ciel : plaques vertes ou jaunes sur Rosette,
Flaming Star et Horsehead, bords violets sur M31.

Fichier seul : [feuille_classic_stack_tous.png](point_etape/feuille_classic_stack_tous.png)

### Ce qui a l'air anormal

- **Aucune image n'est cassée** : ni noire, ni verte d'un bout à l'autre, ni écrasée. Je n'ai rien caché.
- **Les Classic ont un ciel inégal** (plaques vertes, jaunes, bords violets). Ce ne sont pas des erreurs de calcul :
  c'est ce que donne le fond de ciel simple de Classic. Mais c'est à juger.
- **Les Full ont presque perdu leurs étoiles faibles** sur Iris, Rosette et Flaming Star (un peu aussi sur
  Horsehead). Tes anciennes images sont pareilles, donc ce n'est pas une nouveauté. À toi de dire si c'est voulu.
- Ces vignettes sont à 1/4 de la résolution : elles ne montrent ni la finesse du détail ni la netteté des étoiles. Les
  versions plus grandes (environ 1500 px de large) sont dans `F:\astro_projets\_review\images\`.

---

## 2. Le logiciel

**Mise à jour de 9 h : l'application est assemblée.** Tous les écrans ci-dessous sont maintenant réunis dans une
seule application, avec les finitions notées cette nuit : pluriels, durées en minutes, pourcentages arrondis, image
d'empilement sans dominante verte, nom complet des workflows en infobulle. Une visite complète scriptée de 87 écrans,
en anglais puis en français, passe sans erreur.

Pour l'essayer tout de suite, sur un projet d'essai et pas sur tes vrais projets pendant qu'ils calculent :
`C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe -m astro_gpu.client`, lancé depuis
`C:\Users\Nicolas\sandbox\astro`. Ouvre ensuite `F:\astro_essais\p13\heart_a7s` (créé depuis les brutes) ou
`F:\astro_essais\p13\Heart` (importé d'un ancien projet). Tes vrais projets, dans `F:\astro_projets\`, pourront
s'ouvrir quand leurs calculs seront finis.

Les captures ci-dessous ont été prises avant l'assemblage, écran par écran.

Ces captures viennent de projets d'essai (un petit jeu de données du Cœur, un projet factice « HeartSoul »), pas de tes
vrais projets. Les marques ◇ ou la mention « pas encore dans le moteur » indiquent un réglage affiché mais pas encore
branché. Tout est en français.

**1. Accueil** : la liste de tes projets récents, avec leur aperçu, et les boutons pour créer, importer ou ouvrir.
![Accueil](point_etape/ecrans/01_accueil.png)

**2. Graphe** : tout le chemin de traitement d'un coup, de « Images » à « Post-traitement », avec les quatre workflows
en branches ; l'étape choisie s'ouvre dans le panneau de droite.
![Graphe](point_etape/ecrans/02_graphe.png)

**3. Panneau d'une étape** : les réglages de l'étape choisie (ici « Débruitage »), avec la méthode, sa durée estimée et le
choix de s'appliquer à tous les workflows ou à un seul.
![Panneau](point_etape/ecrans/03_panneau_etape.png)

**4. Inspection** : on regarde le résultat de n'importe quelle étape (ici l'empilement) avec les mesures, l'histogramme
et les valeurs sous le curseur.
![Inspection](point_etape/ecrans/04_inspection.png)

**5. Images / sessions** : tes sessions (nuits, poses, nombre de brutes gardées) et les vérifications automatiques.
![Images, sessions](point_etape/ecrans/05a_images_sessions.png)

Le même écran en mode « Parcourir » : chaque brute en vignette, avec la taille des étoiles et la transparence mesurées.
![Images, parcourir](point_etape/ecrans/05b_images_parcourir.png)

**6. Lancer, le plan** : avant de calculer, la liste des étapes, les workflows concernés, la durée et la place disque.
![Lancer, plan](point_etape/ecrans/06_lancer_plan.png)

**7. Lancer, en cours** : le graphe pendant le calcul, avec la barre de progression et le bouton Arrêter.
![Lancer, en cours](point_etape/ecrans/07_lancer_en_cours.png)

**8. Lancer, le bilan** : ce qui a été calculé, en combien de temps, et quels calques sont prêts pour le post-traitement.
![Lancer, bilan](point_etape/ecrans/08_lancer_bilan.png)

**9. Post-traitement, les calques** : tes workflows deviennent des calques empilés avec des filtres (netteté,
saturation), des modes de fusion et une opacité.
![Post-traitement, calques](point_etape/ecrans/09a_post_calques.png)

**9 bis. Post-traitement, le masque** : un masque construit à partir de morceaux (étoiles du catalogue, plage de
luminosité, pinceau) ; ce qui est en rouge est la partie où le calque se voit. (C'est un masque d'essai, qui combine
les trois morceaux pour les tester, pas un masque pour une vraie image.)
![Post-traitement, masque](point_etape/ecrans/09b_post_masque.png)

**10. Export** : le format (TIFF 16 bits, PNG, JPEG, FITS), la taille, le nom du fichier, et ce qu'on joint (solution
astrométrique, historique, un fichier par calque, masques).
![Export](point_etape/ecrans/10_export.png)

**11. Paramètres** : le matériel de calcul, la limite de mémoire et le test de performance de la machine. L'écran a six
autres pages (Général, Dossiers et disque, Catalogues, Formats de fichier, Bibliothèque de masters, Affichage), dans
`F:\astro_projets\_review\ecrans\parametres\`.
![Paramètres](point_etape/ecrans/11_parametres.png)

**En plus** : une visite complète, de la création d'un projet jusqu'à l'export et au retour (plus de 80 captures
prises par un test automatique), existe **en anglais seulement** dans `F:\astro_projets\_review\ecrans\visite_complete_en\`. Elle
peut encore grossir si le test tourne toujours.

---

## 3. Tes décisions

### (a) La couleur de la nébuleuse dans les workflows Classic

**Les faits** (sur le Cœur, seul projet où c'était visible) :
- Avant correction, les Classic rendaient l'H-alpha en **violet** : c'était une erreur de calibrage de la couleur.
  La correction est prête (elle n'est pas encore appliquée) : la nébuleuse passe au **rose-magenta**.
- Ce rose-magenta est la couleur **mesurée** sur l'image, et c'est aussi la couleur de Full avant ses deux conventions.
- Full, lui, la rend **rouge** par une convention voulue : un renfort de l'H-alpha, puis un décalage de la teinte
  vers le rouge. Le renfort ajoute à lui seul un peu de rouge ; le décalage en ajoute davantage.
- Reporter ces deux conventions sur Classic a été essayé une fois. **Le décalage vers le rouge seul est sans risque**
  (il ne touche que la teinte de la nébuleuse). **Le renfort n'est pas sûr** : sur Classic stack, où il reste du
  bruit et des étoiles, il fait virer tout le ciel au brun.

**Le choix :**
1. **Garder la couleur mesurée en Classic** (rose-magenta), et réserver le rouge à Full.
2. **Appliquer le décalage vers le rouge** en Classic, sans le renfort : la nébuleuse devient plus rouge qu'en
   option 1 (teinte autour de rose-saumon), mais moins rouge que Full, et le ciel reste intact.

Les feuilles ci-dessous montrent toujours les mêmes quatre images. Les étiquettes sont en anglais dans l'image :
en haut à gauche **Full** (ne change pas), en haut à droite **Classic stack**, en bas à gauche **Classic + denoise**,
en bas à droite **Raw stack** (la sortie « Empilement brut », pas un des quatre workflows).

**Avant** la correction, la nébuleuse des deux Classic est violette :
![Avant](point_etape/couleur_nebuleuse/sheet_before.png)

**Après** la correction de couleur (c'est l'option 1) et celle du ciel de la sortie brute : nébuleuse rose-magenta,
et la sortie brute n'a plus de coin verdâtre :
![Après](point_etape/couleur_nebuleuse/sheet_after.png)

**Essai avec les deux conventions de Full** (étiquette « TEST »), donc le rouge de Full : Classic + denoise devient
rouge comme Full, mais le coin en bas à droite rougit aussi, et Classic stack prend un ciel brun :
![Essai](point_etape/couleur_nebuleuse/sheet_after2.png)

Zoom sur le ciel brun de Classic stack (à gauche la version de base, à droite avec les deux conventions de Full ;
l'image n'est pas légendée) :
![Zoom](point_etape/couleur_nebuleuse/test_conventions_classic_stack_crop.png)

**Ma lecture :** l'option 2 est le compromis sûr si tu veux un Classic plus proche de Full. L'option 1 garde un Classic
« honnête », c'est-à-dire la couleur mesurée sans mise en scène. À toi de trancher ; la correction elle-même (violet vers
rose-magenta) est faite dans les deux cas.

### (b) Les quatre workflows donnés à chaque projet

| Projet | Classic stack | Classic + denoise | Full | Quatrième workflow |
|---|---|---|---|---|
| Horsehead | oui | oui | oui | Full + earlier neural layer |
| Iris | oui | oui | oui | Full + earlier neural layer |
| Rosette | oui | oui | oui | Full + earlier neural layer |
| Flaming Star | oui | oui | oui | Full + earlier neural layer |
| Heart (IC 1805) | oui | oui | oui | Full + neural stars |
| NGC 7822 | oui | oui | oui | Full + neural stars |
| Wizard | oui | oui | oui | Full + neural stars |
| M31 | oui | oui | oui | Full + neural stars |
| M33 | oui | oui | oui | Full + neural stars |

- « Full + earlier neural layer » est donné aux projets qui avaient déjà une couche d'étoiles apprise (reprise telle quelle).
- « Full + neural stars » refait cette couche d'étoiles par réseau de neurones : c'est la partie la plus longue
  (46 à 54 minutes restantes par projet, d'après le planificateur).
- La sortie « Empilement brut » (l'empilement sans rien d'autre) n'est **pas** donnée aux projets.

**Question : est-ce le bon jeu de quatre ?** Faut-il ajouter ou retirer quelque chose, par exemple ajouter
l'empilement brut partout, ou ne garder que trois workflows pour les projets sans couche apprise ?

### (c) Ce qui reste, dans l'ordre

1. **Finir les calculs** : Wizard et M31 d'ici environ 1 heure, Heart d'ici environ 2 heures, NGC 7822 vers 11 h 30.
   **M33** a planté sur la petite carte graphique (« unspecified launch failure » dès l'empilement) : on le refait sur la
   grande, pas encore planifié.
2. **Les trois corrections d'image** (prêtes, testées sur copie, pas encore appliquées) :
   - la couleur H-alpha violette des Classic (la partie (a) ci-dessus, selon ta décision) ;
   - le curseur « ciel » du post-traitement, qui ne fait rien sur une partie de sa course (par exemple sur le Cœur) ;
   - le ciel verdâtre de la sortie « Empilement brut ».
   Seules quelques étapes de rendu se recalculent ensuite (10 à 30 secondes chacune). Les calculs longs ne sont pas refaits.
3. **Le contrôle de qualité à l'aveugle** contre tes anciennes images (cette fois sans étiquettes).
4. **L'évaluation indépendante** (des juges qui ne savent rien de ce qui a changé).
5. **Le rapport.**

**Question : cet ordre te convient-il ?** Je mets les corrections avant le contrôle à l'aveugle pour ne pas juger des
images qu'on va ensuite changer. Mais cela t'oblige à décider la couleur (a) avant.

---

## 4. L'état du recalcul

État à 8 h 26 (relevé de l'outil de suivi). Les durées sont les estimations du planificateur, à prendre comme un ordre
de grandeur. Il y a deux files de calcul : une par carte graphique.

| Projet | Classic stack | Classic + denoise | Full | Quatrième workflow | État | Fin prévue |
|---|---|---|---|---|---|---|
| Horsehead | fini | fini | fini | fini | **terminé** | |
| Iris | fini | fini | fini | fini | **terminé** | |
| Rosette | fini | fini | fini | fini | **terminé** | |
| Flaming Star | fini | fini | fini | fini | **terminé** | |
| Wizard | fini | fini | presque (1 étape en cours) | à faire (~46 min) | **en cours** (carte 1) | vers 9 h 15 |
| M31 | fini | fini | presque (1 étape en cours) | à faire (~46 min) | **en cours** (carte 2) | vers 9 h 15 |
| Heart (IC 1805) | fini | fini | à faire (~17 min) | à faire (~52 min) | attend son tour (carte 1) | vers 10 h 15 |
| NGC 7822 | fini | fini | à faire (~20 min) | à faire (~54 min) | attend son tour (carte 1) | vers 11 h 25 |
| M33 | **planté** | **planté** | **planté** | **planté** | à refaire sur la grande carte (~2 h) | pas encore planifié |

- Carte 1 : environ 2,9 h restantes (Wizard, puis Heart, puis NGC 7822). Carte 2 : environ 2,7 h restantes, dont
  1,9 h comptées pour M33, qui est planté : une fois M31 fini (vers 9 h 15), cette carte n'a donc plus de travail
  utile tant que M33 n'est pas refait.
- Les Classic stack et Classic + denoise des huit projets sont déjà terminés : seule la partie « Full » reste à calculer.
