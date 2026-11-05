# Point du matin, mardi 3 novembre

## En bref

- **Tes corrections sur l'empilement sont faites** : l'ancienne mesure des étoiles et l'ancienne règle du drizzle sont
  retirées ; la sélection des images propose **Automatique**, **κσ** et **Seuils**.
- **Le code est nettoyé** : environ 60 000 lignes de moins dans le dépôt (maquettes, essais, ancien ComfyUI, ancienne
  interface), dont environ 7 000 dans le logiciel lui-même (moteur, client, tests, outils). Les commentaires disent ce que fait le code ;
  une carte d'une page explique l'ensemble : [ARCHITECTURE.md](ARCHITECTURE.md).
- **Rien n'a changé dans tes résultats** : les clés de calcul de tes 9 projets sont identiques à celles d'hier soir.
- **Les tests passent** (chiffres plus bas), sauf un test lent instable, expliqué au point 5 de la partie 6. Ils
  tournent maintenant sur un petit projet de test entretenu, au lieu des copies périmées.
- **630 Go de restes d'essais** sont rangés dans les dossiers `_trash` de chaque disque. **Rien n'est supprimé.**
- **À faire en premier** : fermer le client ouvert depuis hier soir et relancer **client.bat** (l'ancien tourne encore
  avec le code d'hier).

## 1. L'empilement

| Réglage | Avant | Maintenant |
|---|---|---|
| Mesure de la taille des étoiles | Moffat ou « Previous version » | Moffat seulement (le réglage disparaît) |
| Règle d'échelle du drizzle | couverture ou « Previous version » | couverture seulement (le réglage disparaît) |
| Sélection des images | Poids minimal, κσ, « Previous version » | **Automatique**, **κσ**, **Seuils** |

- **Seuils**, c'est l'ancienne règle, renommée : tu règles la transparence minimale et la taille d'étoiles maximale à la
  souris sur le graphique de la Sélection.
- L'exclusion d'une image à la main marche dans les trois cas.
- Un projet qui avait enregistré l'ancienne valeur est lu avec la nouvelle, sans rien recalculer d'autre.
- Où le voir : Graphe › bloc Sélection, et Inspection › Sélection.

## 2. Les textes du GUI

- Plus aucun nom de logiciel dans l'interface : « type GraXpert », « type BlurXTerminator », « type l'outil de référence»,
  « (DrizzlePac) », « comme dans Photoshop » sont remplacés par ce que fait la méthode.
- L'import des projets de l'ancien programme dit « une ancienne version » au lieu de « la version précédente ».
- En français : « FWHM (px) » était traduit par « Étoiles (px) » ; « retrait appris » devient « retrait neuronal »
  partout ; « spicules » devient « aigrettes » ; « dégradé du ciel » devient « gradient du ciel ».
- Un défaut corrigé : en français, la légende du graphique Qualité de l'Alignement débordait une fois zoomé. Elle passe
  maintenant sur une deuxième ligne quand elle ne tient pas.
- Un plantage rare corrigé : le client pouvait se fermer d'un coup quand Python faisait le ménage de sa mémoire depuis
  un fil de calcul secondaire. C'est arrivé une fois cette nuit dans les tests. Le ménage se fait maintenant depuis
  l'interface seulement, selon la méthode habituelle des programmes Qt.

## 3. Le code

- **Sortis du dépôt** : les maquettes, les essais, l'ancien ComfyUI et l'ancienne interface (remplacée par le client).
  Tout reste retrouvable dans git sous l'étiquette `archive/avant-nettoyage-2026-11-03`.
- **Moteur** : le code mort retiré, les fonctions en double regroupées, les commentaires réécrits sans dates ni récit ;
  les sources des méthodes publiées sont gardées.
- **Client** : le code mort et les restes des maquettes retirés, ainsi que 252 traductions qui ne servaient plus. Les
  captures d'écran d'avant et d'après le montrent : rien ne change à l'écran.
- **Notes du pilote** : rangées dans [archive/](archive/). Elles racontent l'histoire, elles ne servent plus de
  consignes.
- **Mes mémoires** : réduites à quelques règles courtes, comme avant le pilote.

## 4. Les vérifications

| Vérification | Résultat |
|---|---|
| Clés de calcul de tes 9 projets et des copies (137 workflows) | identiques ; seul change le workflow de banc qui utilisait les anciennes options |
| Tests rapides | 839 réussis, 0 échec (3 demandent une carte graphique visible) |
| Tests lourds | 904 réussis, 0 échec |
| Tests lents (vraies données) | 21 réussis, 1 instable (voir le point 5 de la partie 6) |
| Captures du client (175 écrans, en anglais et en français, sur la copie de M31) | identiques avant et après la nuit, hormis les changements voulus (les textes corrigés) et les noms de dossiers temporaires ; les mêmes alertes du script de captures avant et après |

## 5. Les disques

| Où | Taille | Ce que c'est |
|---|---|---|
| `F:\_trash\2026-11-03_copies_de_test_heart` | 43 Go | les 6 copies de test de Heart |
| `F:\_trash\2026-11-03_essais` | environ 160 Go | les vérifications du pilote (p7, p13, fixes…) et les copies de banc pour le GUI |
| `C:\_trash\2026-11-03_essais` | 129 Go | les copies de non-régression du 28 octobre, l'ancien cache de blocs ComfyUI |
| `D:\_trash\2026-11-03_essais` | 298 Go | les essais de diagnostic d'octobre (`astro_gpu_diag`), l'ancien dossier `D:\astro_gpu` |
| `C:\_trash\2026-11-03_depot_astro` | 0,4 Go | les restes non suivis par git des maquettes |

Pour remettre un dossier en place, il suffit de le redéplacer. La liste complète des 112 déplacements est dans le
fichier `moves.log` de cette nuit.

**Ce que je n'ai pas touché** : tes projets et leurs dossiers « scratch », les couches d'étoiles que tes projets
utilisent (`D:\astro_gpu_diag\learned` et `own_starnet`), tes images brutes, les jeux d'images téléchargés et les
rendus livrés de M31 (`D:\astro_gpu_m31`).

## 6. Ce qui attend ta décision

1. **Garder les X % meilleures images, comme Siril ?** Siril propose aussi ce mode dans sa sélection. Je ne l'ai pas
   ajouté cette nuit. *Mon conseil : oui, comme quatrième choix de la sélection.*
2. **Le banc face à Siril (`F:\astro_essais\bench`, 639 Go)** : je l'ai gardé, parce qu'il servira pour le chantier du
   gradient. Veux-tu le garder ou le mettre au `_trash` ?
3. **Des dossiers douteux, laissés en place** :
   - `F:\astro_essais\v2\M31_copy`, la copie de M31 que tu regardes dans le client ;
   - le projet `F:\astro_projets\test`, créé le 1er novembre à 18 h, peut-être par toi ;
   - `experiments\star_cleaner` dans le dossier du code : les images de référence posées le 1er au soir. Un programme
     le bloquait cette nuit, il est à sortir du dépôt ;
   - `C:\Users\Nicolas\astro_gpu_scratch\tools`, où l'outil de référence et NoX sont installés pour les comparaisons.
4. **Vider les `_trash`** quand tu auras vérifié : environ 630 Go.
5. **Une pause de l'interface pendant un calcul.** Pendant un vrai calcul, l'interface se fige parfois environ 0,2 s, à
   la limite de ce que le test tolère (une fois sur deux au-dessus). J'ai vérifié que ce n'est pas notre code Python
   mais le dessin de Qt lui-même. Pour aller plus loin, il faut un outil de mesure du code de Qt. *Mon conseil : le
   regarder quand on reprendra le GUI ; la pause est brève.*
6. **L'accentuation et la réduction du bruit multi-échelles** n'ont aucun effet visible sur les échelles fines (1 et
   2 pixels) quand l'image est vue en entier : ces détails sont plus petits qu'un pixel de l'écran, il faut zoomer à
   100 %. C'est logique, mais déroutant. *Mon conseil : un petit message « visible à 100 % » à côté de ces curseurs.*

## 7. Ce qui reste du nettoyage

- **Trois gros fichiers** gagneraient à être découpés : la visionneuse d'images (3 900 lignes) et l'Inspection
  (2 300 lignes) dans le client, l'interface entre le client et le moteur (2 600 lignes). Il y en a pour une demi-journée environ, sans changement
  visible.
- **L'ancien format de projet** sert encore à la ligne de commande et à l'import de tes anciens projets. Le retirer est
  une décision à prendre ensemble, pas un nettoyage.

## Les accrocs de la nuit

- **La construction du petit projet de test a pris 2 h de carte graphique au lieu de 30 min.** Les catalogues Gaia
  venaient d'un serveur lent, parce que VizieR était en panne.
- **Un agent a lancé par erreur une commande git qui a remis le dossier du code à l'état enregistré pendant
  10 secondes.** Tout est revenu ; les clés et les tests le confirment.
- **Un de mes enregistrements a embarqué trois fichiers d'un autre agent.** Je l'ai corrigé tout de suite.
