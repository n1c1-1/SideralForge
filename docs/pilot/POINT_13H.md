# Point de la mi-journée, lundi 2 novembre

Tes 9 projets n'ont pas été touchés. Tout ce qui suit est enregistré.

## Fait ce matin

**1. Le drizzle de Siril, mesuré sur les mêmes images (30 poses, 6 cas)**
- **Le bruit en plus vient du drizzle en général, pas d'un défaut du nôtre.** Face à son propre empilement classique,
  le drizzle de Siril perd plus que le nôtre : notre drizzle va plus loin dans les étoiles faibles que celui de Siril.
- **Un point où nous sommes derrière :** le signal sur bruit de la nébuleuse ou de la galaxie, en moyenne 12 % sous
  le drizzle de Siril. À regarder ensemble à 19 h.
- **Les pixels chauds isolés :** le drizzle de Siril en laisse aussi, plus que nous sur Heart et M8. Les empilements
  classiques n'en laissent presque pas.
- Sur M51, l'extraction du fond de Siril a échoué (dominante rose et verte) ; c'est noté, sans retouche.
- Le détail est dans [BENCH.md](BENCH.md), section « Le drizzle de Siril ».

**2. Les deux méthodes dans le GUI**
- Dans les réglages de l'empilement, deux cartes : « Drizzle » (par défaut) et « Empilement classique ». Chacune
  n'affiche que ses réglages.
- Sur la méthode conseillée, un petit cadre bleu « Conseillé » et une icône « i » : la raison en quelques mots. Le
  conseil ne change jamais ton choix.
- Le conseil suit ton tableau (couleur ou monochrome, nombre de poses, couverture du rouge et du bleu, taille des
  étoiles) et ne regarde que les poses gardées. Sur la Rosette et M51, il conseille l'empilement classique ; sur Iris
  et le Seestar, le drizzle.
- Inspection › Empilement : les poses gardées et écartées avec leur raison, les pixels rejetés avec leur taux à côté
  du repère de Siril, et la couverture du rouge et du bleu pour le drizzle.

**3. Le moteur**
- Les quatre améliorations sans effet sont retirées.
- La nouvelle mesure de la taille des étoiles est le réglage par défaut.
- **Le seuil de sélection :** toutes les poses alignées sont gardées, sauf celles qui pèsent moins de 5 % de la
  meilleure (la valeur de PixInsight). Sur la Rosette, 26 poses sur 30 au lieu de 16. Le bloc Sélection le dit en une
  phrase ; les anciens seuils restent disponibles, grisés tant qu'on ne les choisit pas.

**4. L'empilement classique**
- **La dominante orange est corrigée.** Le dématriçage reconstruisait mal la couleur des petites étoiles, parce que
  l'image n'était pas équilibrée en couleur avant. On l'équilibre maintenant avant de dématricer, comme RawTherapee
  et darktable. Le blanc de Heart revient au niveau de Siril et de notre drizzle.
- **L'arrêt en erreur** venait du pilote de la carte graphique, saturée ce soir-là, pas de notre code. Le calcul en
  cause est réécrit, et si la carte cesse de répondre, le calcul reprend maintenant dans un processus neuf au lieu de
  tout faire échouer.

## Pour regarder le GUI

- Lance **client.bat**, section « Banc d’essai face à Siril », projet **Seestar** ou **Iris** (express).
- Évite la Rosette et Heart jusqu'à ce soir : un essai tourne dessus.
- Les empilements sont marqués « Obsolète » : les réglages par défaut ont changé ce matin. L'inspection montre donc
  le résultat précédent, marqué comme tel. Ne lance pas de calcul : la carte graphique est occupée par les essais.

## Cet après-midi

- **Les pixels chauds du drizzle,** avec la méthode publiée de PixInsight : repérer les pixels à écarter sur les
  poses dématricées, puis les écarter dans le drizzle. Fin vers 15 h 30. Ce sera un réglage : le banc compare
  l'ancienne et la nouvelle façon.
- **17 h à 18 h 30 : le banc sur 30 poses,** notre drizzle et notre empilement classique face à Siril, en classique
  et en drizzle.
- **18 h 30 : le point en français,** avec les images.

## Ce que tu auras à décider à 19 h

1. Les pixels chauds du drizzle : la nouvelle façon par défaut, ou non (selon le banc).
2. **Le reste de couleur de l'empilement classique.** Le dématriçage écrase encore un peu les couleurs des petites
   étoiles (Siril aussi). Deux façons connues de faire mieux :
   - garder pour l'empilement classique la correction de couleur mesurée sur le drizzle du même appareil ;
   - ou dématricer autrement (bilinéaire : couleurs exactes, mais étoiles un peu moins bien rendues).
3. Le signal sur bruit de la cible, 12 % sous le drizzle de Siril : on cherche pourquoi, ou non.
4. Ensuite seulement : retirer l'ancienne version, recalculer les copies de test, puis tes 9 projets.
