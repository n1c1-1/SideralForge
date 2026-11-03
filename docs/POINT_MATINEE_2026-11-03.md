# Point de la matinée, mardi 3 novembre

Les cinq points de ton plan. Mesures faites dans le vrai client, sur la copie de M31, sans carte graphique (les deux
cartes calculaient le banc pendant ce temps).

## 1 et 2. Les vues lentes et l'interface qui se fige

**Ce qui ralentissait :**
- **Le cache des images était limité à 3 Go**, alors que ta machine a 69 Go. Un empilement préparé pèse plusieurs
  centaines de Mo : en revenant sur un onglet, l'image avait souvent été chassée du cache et était relue sur le disque
  (0,2 à 1,4 s à chaque retour).
- **Chaque onglet n'était préparé qu'au moment du clic**, en lisant de gros fichiers sur F:.
- **Les grands tableaux** (images de l'Alignement, poids, étoiles de couleur) créaient une cellule Qt par valeur : 0,4 s
  pendant lesquelles la fenêtre ne répondait plus.
- **Le bouton « Exécuter jusqu'ici »** relisait sur le disque les manifestes de toutes les étapes à chaque clic.
- **Le premier rendu chargeait le compilateur de torch** (1 s et plus), parce qu'un réglage de reproductibilité de
  torch le déclenche.

**Ce qui est fait :**
- Le cache des images suit la machine : un cinquième de la mémoire, entre 3 et 24 Go, soit environ 14 Go chez toi.
- Après l'image affichée, les images des autres onglets du bloc et du bloc suivant se préparent en arrière-plan.
- Les tableaux ne dessinent plus que les lignes visibles.
- L'état du bouton est gardé jusqu'au prochain changement.
- Torch est chargé au démarrage du client, en arrière-plan.
- Le rapport sur les darks et les flats se prépare dès qu'on ouvre la Calibration.

**Résultat, entre le clic et l'image affichée :**

| Cas | Avant | Après |
|---|---|---|
| Retour sur un onglet déjà vu | 0,2 à 1,4 s | 0,1 à 0,2 s |
| Tableau de l'Alignement | 0,43 s, fenêtre figée | 0,11 s |
| Couleur, première fois | 0,67 s | 0,11 s |
| Gradient › Résultat, première fois | 2,6 s | 0,12 s |
| Débruitage, première fois | 0,53 s | 0,11 s |

**Ce qui reste lent la première fois :**
- Calibration › Darks (5 s) : la lecture des métadonnées de chaque fichier brut. Elle se prépare maintenant dès
  l'ouverture de la Calibration.
- Couleur › Ajustement (1,8 s) : la mesure des étoiles du catalogue sur l'empilement.
- Post › Sortie : 12 s sans carte graphique ; c'est beaucoup moins avec ta carte.

Pense à relancer **client.bat** pour avoir tout cela.

## 3. Gaia

- **Les extraits Gaia hors ligne sont installés** (positions et spectres, ceux du projet Siril) : 2 min pour les
  positions, 29 min pour les spectres.
- **Une précision sur ce que je t'avais dit** : ces extraits servent à l'astrométrie et aux spectres, mais ils ne
  gardent que les étoiles les plus brillantes de chaque zone. Les étoiles faibles, dont nos catalogues ont besoin,
  viennent toujours d'Internet la première fois pour un champ, puis restent dans le cache.
- **Le champ de Heart** téléchargé cette nuit pour le projet de test était dans un cache à part. Je l'ai recopié dans ton
  cache principal.
- **L'outil du projet de test** lit maintenant tes extraits hors ligne.

## 4. L'indicateur (i)

- Quand l'alignement d'un workflow n'est pas à jour, le moteur ne peut pas conseiller de méthode. Le panneau le dit
  maintenant : « La méthode conseillée s'affiche une fois l'alignement calculé ».
- C'est le cas de tes projets depuis le 2 novembre (la mesure des étoiles a changé).
- Quand l'alignement est à jour, le cadre « Conseillé » et son (i) s'affichent (vérifié sur le banc Iris).

## 5. Le banc face à Siril

- Les projets « base » (100 images) étaient arrêtés à l'empilement depuis le 2 novembre au matin, quand les réglages
  par défaut de l'empilement ont changé. Les deux workflows de référence des projets « express » (le drizzle et le
  classique) étaient, eux, à jour.
- **Tous recalculés jusqu'au bout** (sur les deux cartes, de 6 h 58 à 8 h 01) : Heart, Iris, M51, Rosette et Seestar,
  les deux projets « base » de chacun. Plus aucun calcul en attente dans ces projets.
- Heart a d'abord été bloqué : son calcul avait démarré avant l'installation des extraits Gaia et téléchargeait ses
  étoiles faibles une case à la fois. Je l'ai arrêté et relancé une fois les cases de son champ recopiées. Il a ensuite
  pris 4 minutes.
- Les nombreux workflows d'essai des projets « express » (OLD, S1…) ne sont pas recalculés : ce sont les variantes des
  essais d'empilement du 1er novembre.

## Tests

- Série rapide : 839 réussis, 0 échec.
- Série lourde : 903 réussis, 1 échec qui passe quand on le relance seul. C'était pendant que les deux cartes
  calculaient le banc : ce test dépend de la mémoire graphique libre.
