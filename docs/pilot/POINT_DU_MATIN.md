# Point du matin, lundi 2 novembre

Rien n'a touché à tes 9 projets, et l'ancienne version de l'empilement est toujours là. Tout ce qui suit attend tes
décisions.

## Fait cette nuit

**1. L'empilement v2, derrière des interrupteurs** (spec : [STACKING.md](../spec/STACKING.md))
- 6 modules, chacun tiré d'une méthode publiée :
  - FWHM par ajustement de Moffat, et carte de couverture du drizzle (règle de Hubble) ;
  - rejet de Hubble ;
  - pondération par le bruit, et seuils de sélection aux valeurs de Siril ;
  - normalisation locale par couleur ;
  - recalage des couleurs au fil de la nuit ;
  - la méthode Classique : RCD et Lanczos-4, identique à Siril sur Heart.
- **Tout coupé = l'ancienne version au pixel près**, sans rien rendre caduc dans tes calculs.

**2. Le banc croisé** : 19 réglages × 6 cas en Express, puis 3 en Base. Tous les tableaux sont dans
[BENCH.md](BENCH.md).

| Changement | Verdict |
|---|---|
| **Moffat + règle de couverture** | **Recommandé.** Bruit fin plus bas, signal sur bruit +2 à +6 %, étoiles 1 à 4 % plus larges ; bat Siril dans 17 cases. Mais **en Base, il recule sur Rosette** : ciel magenta, couleur moins juste. |
| Rejet de Hubble | Aucun gain, plus de petits points sur M51 et Rosette. |
| Pondération par le bruit | Neutre seule ; −0,14 mag sur Iris combinée à Moffat. |
| Seuils de sélection | Sans Moffat, trop de poses écartées (jusqu'à −0,22 mag). |
| Normalisation locale, recalage des couleurs par blocs | Neutres sur nos mesures. |
| Méthode Classique | Même bruit que Siril, +0,11 à +0,17 mag sur 4 cas, mais **couleurs abîmées** (Heart orange). |

**3. Le Post** ([POST_TOOLS.md](POST_TOOLS.md)) :
- **un vrai outil Courbes** : 8 canaux, Akima, préréglages de Photoshop et préréglages astro, tes propres préréglages ;
- **de nouveaux filtres** :
  - GHS, arcsinh, transformation d'histogramme et SCNR, identiques à Siril ;
  - CLAHE ;
  - netteté et réduction de bruit multi-échelles ;
- un menu en 7 groupes.

**4. Inspection :**
- le tracé x/y de la séance sous le graphe de dérive ;
- les comparaisons recalées sur le même ciel ;
- la Sortie de M31 expliquée : ce sont les mêmes données que « Gradient retiré », étirées par les réglages du Post.

## Défauts du moteur trouvés, pas corrigés

1. La méthode Classique peut faire planter la carte graphique avec peu de poses (rejet winsorisé, M8).
2. La règle de couverture juge les poses avant la sélection. Pour Rosette, elle aurait dû conseiller Classique.
3. La méthode Classique suivie de notre étape couleur donne une dominante orange à Heart.

Corrigé : un « σ » dans un message faisait planter une étape sous Windows.

## Tes décisions

1. **La configuration de l'empilement.** Le banc recommande Moffat + couverture. Que fait-on du recul sur Rosette, et
   des changements neutres (les garder pour leur principe, ou non) ?
2. **La méthode Classique.** On corrige d'abord sa couleur (défaut 3) et sa stabilité (défaut 1) ?
3. **Ensuite seulement :** choisir les valeurs par défaut, supprimer l'ancienne version, recalculer les copies de test
   (22 tests en dépendent ; les 770 autres passent), puis tes 9 projets.
