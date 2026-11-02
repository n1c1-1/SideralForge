# Catalogues of named deep-sky objects, shipped with the package

Read by `astro_gpu/core/skyobjects.py` (the names on the images, the target search). No download, no other program
needed. Each folder has the source's origin, version, licence or terms and citation in its README.md; all are made
again with `tools/catalogues/build_objects.py`.

| folder | source | version | terms |
|---|---|---|---|
| `openngc/` | OpenNGC, M. Verga | release v20260501 | CC BY-SA 4.0 |
| `steinicke/` | Revised NGC/IC, W. Steinicke | 8 February 2026 | free for non-commercial use; inform the author |
| `sharpless/` | Sharpless 1959, VizieR VII/20 | as served 2 November 2026 | CDS conditions of use |
| `lynds/` | Lynds 1962, VizieR VII/7A | as served 2 November 2026 | CDS conditions of use |

## Messier

There is no separate Messier list: the Messier numbers are OpenNGC's (its `M` column, and its addendum rows M 40 and
Mel 22 = M 45), which give all 110 objects. One exception: M 102, which OpenNGC marks as a duplicate of M 101
("Identification is controversial, here we take NED assumption"); the package follows Steinicke's identification,
M 102 = NGC 5866, as do most observing lists. Steinicke's cross-identifications give 108 Messier numbers (all but M 40
and M 45) and agree with OpenNGC on every other one. `tests/test_skyobjects.py` checks the 110.
