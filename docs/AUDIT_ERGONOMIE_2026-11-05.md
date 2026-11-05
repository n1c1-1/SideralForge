# Audit d'ergonomie du 5 novembre 2026 (français et anglais)

Quatre relectures sur captures d'écran : Accueil, Graphe et Empilement ; Inspect et Images ; Étoiles et Débruitage ;
Post-traitement et Export. Les écrans ont été relus dans les deux langues. Les captures sont dans
`F:/astro_essais/shots/audit_2026-11-05/` (a, a_en, b, b_after, c).

## Corrigé (sans décision à prendre)

- **Étoiles → Post-traitement**
  - Un calque envoyé depuis le graphe est sélectionné dans le Post-traitement. Avant, il masquait l'image réglée sans
    qu'on le voie.
  - Un deuxième envoi de la même image n'ajoute pas de copie : un message dit qu'elle est déjà un calque.
- **Réglages des étoiles**
  - Les groupes portent le nom de l'image sur laquelle ils agissent : « Étoiles seules », « Image sans étoiles ».
  - Le menu s'appelle « Mode de fusion des étoiles ».
  - Les curseurs affichent leur valeur comme Lightroom : « 0 », « −1,0 », avec un vrai signe moins. Le mot
    « suggéré » disparaît, et avec lui sa faute d'accord.
- **Panneau du bloc Étoiles**
  - Les sorties sont en tête, une destination par ligne.
  - Il n'y a plus de choix de portée, puisque le bloc n'a pas de réglage.
  - « Supprimer le bloc » remplace « Retirer », qui prêtait à confusion sur un bloc qui retire des étoiles.
- **Graphe**
  - Les fils passent derrière les blocs, et les ports sont plus gros.
  - Un nom de workflow trop long est coupé au milieu : « Classic s…e adapted ».
- **Mots**
  - Un seul état pour ce qui n'est pas calculé : « Pas encore calculé ».
  - « Gardée » au lieu de « Incluse ».
  - « Trajectoire de dérive » au lieu de « Lissajous », et « Tons moyens » au lieu de « Tons moyens (MTF) ».
  - « + Masque », « Noms des objets », et « Adapter » au lieu de « Ajuster ».
  - « dithering » est dit en français.
  - Les phrases de la calibration et du gradient sont réécrites.
- **Débruitage** : les cartes de méthode n'ont plus de superlatif ni de durées en double. « Étapes d'adaptation »
  remplace « Pas d'adaptation ».
- **Rejet** : la méthode est dite une seule fois, avec ses seuils et le nombre d'images.
- **Divers**
  - « Réglages avancés » devient un triangle qui déplie, comme dans Lightroom.
  - La couverture est montrée dans l'ordre R, V, B.
  - Le double symbole « ≈ ≤ » est corrigé.
  - À l'export, une option que le format ne prend pas apparaît décochée.
  - « Processeur seul (pas de carte graphique) » est écrit en clair.

## Décisions du 5 novembre

- 1 : on garde « workflow », en français aussi.
- 3 : fait. La barre d'Inspect a un seul bouton « Exporter ▾ » (Fichier…, Copier l'image, Envoyer au
  Post-traitement), et elle tient en entier en français.
- 8 : règle majeure, aucun nom de logiciel externe dans l'interface, sauf un logiciel réellement utilisé. Les
  Courbes, les Niveaux, les préréglages « Courbes de tons classiques », les fichiers .acv et l'aide de la taille des
  étoiles ne nomment plus aucun logiciel. Les crédits des catalogues Gaia de Siril restent, car ces données sont
  réellement utilisées.

## À décider

1. **« workflow » ou « chaîne » en français.** Trois relecteurs le signalent : « les 6 workflows », « + Nouveau
   workflow ». Je propose « chaîne de traitement ». C'est un renommage global.
2. **La portée par défaut d'un changement.** Aujourd'hui, un changement s'applique par défaut à « tous les N
   workflows » qui partagent le bloc. Je propose par défaut « ce workflow seulement », avec l'autre choix à un clic.
3. **La barre d'outils d'Inspect est coupée en français.** On ne lit que « Co… ». Je propose de regrouper Exporter,
   Copier l'image et Envoyer au Post-traitement dans un seul menu « Exporter ▾ », comme Photoshop.
4. **La case « Ciel neutre (affichage seulement) » apparaît en double** au-dessus de l'image, en plus de celle du
   panneau d'étirement. Je l'ai gardée, parce qu'elle rend visible une option utile sur les images encore en couleurs
   de l'appareil. Faut-il l'enlever ?
5. **Orthographe anglaise.** « Color » côtoie « colour ». Je propose l'américain partout, puisque « Color » est déjà
   le nom du bloc. C'est un gros renommage.
6. **Nombres et dates en français.** Il manque l'espace des milliers (« 59 972 »), le vrai signe moins dans les
   tableaux et les axes, et une forme unique pour les dates. C'est un passage d'ensemble.
7. **Post-traitement**
   - Le panneau des propriétés est trop petit. Je propose un séparateur réglable entre Calques et Propriétés.
   - Un étirement ajouté sur un calque déjà étiré donne un double étirement sans prévenir. Il faut soit avertir,
     soit remplacer l'étirement automatique.
   - Le JPEG n'a pas de réglage de qualité.
   - La visibilité d'un calque est une case à cocher. Photoshop utilise un œil.
8. **Courbes.** La description cite Photoshop, et des préréglages s'appellent « Photoshop ». Ta règle vise Siril et
   PixInsight. Faut-il les garder ?
9. **Graphe au zoom par défaut.** Le texte des blocs est petit.

## Tests

La règle est désormais : aucun test en échec. Les tests qui reposaient sur l'ancien état du projet de test ont été
adaptés pour lire ses vrais chiffres (images, taille de l'empilement) et partir d'une définition fixe
(`full_only`). Ceux qui testaient des fonctions supprimées ont été retirés : réponse spectrale, relevé H-alpha dans
Inspect, choix de la correction de couleur. Le code mort trouvé au passage a été supprimé : la réponse spectrale et
ses graphiques, et 127 traductions que plus aucun texte n'utilise.
