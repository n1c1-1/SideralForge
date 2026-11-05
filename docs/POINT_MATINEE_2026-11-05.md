# Point de la matinée, jeudi 5 novembre

## 1. Le retrait des étoiles : nos deux réseaux face à StarNet

Deux entraînements ont tourné cette nuit, de 22 h 48 à 7 h 50, chacun sur une carte. Le réseau est le même (NAFNet,
l'architecture publiée qu'utilisent Dark Star et SyQon). Il apprend sur tes 23 champs retouchés à la main (56
images, le set 53 laissé de côté) :

- **« real »** : tes paires seules.
- **« mixed »** : tes paires, plus des étoiles de synthèse ajoutées à la volée (profils mesurés champ par champ,
  halos, aigrettes, cœurs saturés).

Les cinq champs du banc ne sont jamais vus par l'entraînement. Les deux réseaux y ont produit leurs images sans
étoiles, mesurées et mises en planches face à StarNet, comme la dernière fois.

### Les mesures

**La lumière d'étoile qui reste** (étoiles faibles du catalogue, magnitude 16 à 18,5, sur fond de ciel ; plus c'est
bas, mieux c'est) :

| Champ | StarNet | mixed | real |
|---|---|---|---|
| Seestar (M31) | 11,6 % | 9,2 % | 8,6 % |
| Rosette | 4,1 % | 3,1 % | 2,3 % |
| Iris | 3,9 % | 3,4 % | 2,2 % |
| M51 | 6,3 % | 5,1 % | 3,2 % |
| Heart | 0,5 % | 0,4 % | 0,3 % |

Pour les étoiles plus brillantes, les trois sont à moins de 1 %. « real » descend même légèrement sous zéro
(−0,1 à −0,9 %) : il enlève un peu trop. « mixed » ne le fait pas.

**Le fond hors des étoiles** : identique pour les trois (même pente, aucun décalage). Aucun des réseaux n'éteint ni
n'éclaircit la nébuleuse.

**Le grain dans l'empreinte des étoiles brillantes** (1 = le même grain qu'autour) :

| Champ | StarNet | mixed | real |
|---|---|---|---|
| Seestar | 0,88 | 1,05 | 1,18 |
| Rosette | 1,09 | 1,37 | 1,42 |
| Iris | 0,71 | 1,23 | 1,28 |
| M51 | 0,77 | 1,29 | 1,22 |
| Heart | 1,61 | 1,53 | 1,61 |

StarNet est sous 1 : il lisse là où il a enlevé une étoile. Nos réseaux sont au-dessus : il reste un peu de
structure dans l'empreinte. La raison est visible sur les planches : ils enlèvent les grands halos et les aigrettes,
que StarNet laisse en partie.

### Ce que montrent les planches

Les planches sont dans `F:\astro_essais\stars_compare\` (`<champ>_sheet.jpg` pour les images sans étoiles,
`<champ>_stars.jpg` pour les couches d'étoiles).

- **Les aigrettes et les halos des étoiles brillantes** : c'est là que l'écart est le plus net. Sur M51, StarNet
  laisse quatre taches floues aux aigrettes de la grosse étoile ; « mixed » ne laisse qu'une ombre légère. Sur
  Rosette et Iris, nos réseaux enlèvent tout le halo bleu des étoiles brillantes, StarNet en laisse la moitié.
  C'est ton travail de retouche qui paie.
- **Le défaut de « real »** : au cœur des étoiles saturées, il laisse un petit trou sombre bordé de vert (la grosse
  étoile de M51, celle d'Iris). **« mixed » n'a pas ce défaut** : les cœurs saturés des étoiles de synthèse le lui
  ont appris.
- **La nébuleuse** (Rosette, Iris, Seestar) : les trois rendent le même fond.

### Le verdict

- **« mixed » est le meilleur des deux** : presque aussi bon que « real » sur les étoiles faibles, et sans le trou au
  cœur des étoiles saturées ni l'excès de soustraction.
- **Face à StarNet, il gagne** sur les étoiles faibles (sur les cinq champs), sur les aigrettes et les halos des
  étoiles brillantes ; il fait jeu égal sur le fond et la nébuleuse.
- **Il reste derrière StarNet** sur un point : un peu de structure dans l'empreinte des étoiles brillantes, là où
  StarNet lisse. À regarder à l'œil sur tes images : c'est parfois du halo bien enlevé, parfois un reste.
- **Un point à surveiller** : pendant l'entraînement, le contrôle de « mixed » sur Seestar sautait d'un contrôle à
  l'autre, alors que les quatre autres champs étaient stables. La version finale est bonne sur Seestar, mais c'est
  le champ où « mixed » est le moins sûr.

### Les chiffres de l'entraînement

Écart moyen sur les pixels d'étoiles des champs du banc, au dernier contrôle :

| | Heart | Iris | M51 | Rosette | Seestar |
|---|---|---|---|---|---|
| mixed (56 700 pas, 9 h) | 0,0010 | 0,0006 | 0,0008 | 0,0006 | 0,0070 |
| real (86 600 pas, 8 h 36) | 0,0016 | 0,0013 | 0,0013 | 0,0021 | 0,0019 |

M8 n'a pas servi au contrôle pendant l'entraînement. Il manquait d'abord aussi aux mesures, à cause d'un défaut de
mon outil, corrigé à 8 h 15. **Sur M8, StarNet fait un peu mieux** : 0,7 % de lumière restante sur les étoiles
faibles contre 1,1 % (« mixed ») et 1,0 % (« real »), et un peu moins de trous. L'original de M8 n'est pas tiré de
la même étape que les autres champs : il a déjà été débruité par l'ancienne méthode, donc il est moins bruité que ce
que le réseau a vu à l'entraînement.

## 2. Ta relecture des images sans étoiles

- **Quinze champs retouchés, tous localement** : aucun changement global, aucune zone lissée. Les deux taches bleues
  du set 33 sont corrigées ; il reste un disque très faible, sans conséquence.
- **Alnitak, l'étoile de NGC 2023 et le nœud de M33** sont des zones où le réseau n'apprend rien (ni bon ni mauvais
  exemple). Tu n'as donc pas à t'acharner sur Alnitak.
- **Le set 53** (NGC 6934) : les taches de couleur sont déjà dans l'empilement. Ce sont des pixels chauds que la
  dérive de la monture étire en traînées et que notre empilement ne rejette pas (Siril les enlève avant, par sa
  correction cosmétique). Le défaut vient donc de notre chaîne, à l'empilement. Les sets 14 et 15 ont le même. Le
  set 53 n'est pas utilisé pour l'entraînement.

## 3. L'interface

- **Aperçus en couleurs neutres** : le Gradient (la vue et l'éditeur d'échantillons) et le Débruitage s'ouvrent avec
  « Ciel neutre » coché. C'est une correction d'affichage seulement ; la case la retire. L'Empilement garde les
  vraies couleurs de l'appareil.
- **Le Débruitage se compare à l'image dont le gradient est retiré** (ce que le bloc reçoit), et non plus à
  l'empilement brut. L'effet du débruitage se voit seul.
- **SideralForge** : titre de la fenêtre, « À propos », fichiers FITS exportés, README sur GitHub. Une commande
  `sideralforge` lance l'interface. Le nom interne du code et le dossier de tes réglages ne changent pas.

## 4. Un défaut corrigé

Si tu choisissais toi-même l'adaptation « courte » du débruitage, le choix automatique « longue » pouvait l'emporter
sur une séance sans dithering. C'est corrigé.

## Tests

- Aucun nouvel échec. Les 54 tests qui échouent échouaient déjà sur `main` avant ce travail : surtout ceux de
  l'interface, laissés de côté jusqu'ici.
- `main` est poussé sur GitHub.

## La suite

Pour les étoiles (le plan de `docs/chantiers/etoiles.md`) :

1. **Brancher « mixed » dans la chaîne** avec les deux couches (sans étoiles, et étoiles seules) et leur
   recombinaison en Écran ou en Addition.
2. **La vitesse** : 8 s pour une image de 12 Mpx sur la 4060 Ti (16 s la première, le temps de charger le réseau).
3. **L'interface** du bloc Étoiles.

Ailleurs :

- Les pixels chauds de l'empilement (sets 14, 15 et 53).
- Les tests d'interface en échec.
- Compléter le projet M8 du banc pour qu'il revienne dans le contrôle.
