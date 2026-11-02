# Revised NGC/IC (W. Steinicke)

- File: `ngcic.csv.gz`, the sheet of "Revised NGC/IC Data" (`NI2026.xls` in `rev2000/NI2026.zip`) written as CSV
  (UTF-8, comma), its 38 columns unchanged (N, NI, A, C, D, S, P, CON, RH, RM, RS, V, DG, DM, DS, BMAG, VMAG, B-V,
  SB, X, Y, PA, TYPE, z, D(z), Dist, PGC, ID1..ID11; explained on the author's page `rev2000/Explan.htm`).
- Origin: Dr. Wolfgang Steinicke, http://www.klima-luft.de/steinicke/ngcic/ngcic_e.htm (plain http).
- Version: 8 February 2026 (the page's "Version" line; the xls is dated 2026-02-08). Downloaded 2 November 2026.
- Made with: `tools/catalogues/build_objects.py`.
- Used for: the cross-identifications of the NGC / IC objects (Messier numbers, NGC = IC duplicates, Sharpless
  numbers) and common names ("Crab nebula") in the catalogue of named objects (`astro_gpu/core/skyobjects.py`).

## Terms of use (the author's page, verbatim)

"Any non-commercial use of my data is free! If a commercial use is planned, please contact me! Please inform the
author Dr. Wolfgang Steinicke (also in case of non-commercial use, e.g. in a free software). In any case, a proper
acknowledgment of the source, including an answer-back to the author is necessary. These data are part of a scientific
project and subject to copyright!"

Contact: steinicke-zehnle@t-online.de

A commercial use of the product needs the author's permission first. The author is to be informed of this use.

## Citation

Steinicke, W., Revised NGC and IC Catalogue, version 8 February 2026,
http://www.klima-luft.de/steinicke/ngcic/ngcic_e.htm. See also Steinicke, W. 2010, Observing and Cataloguing Nebulae
and Star Clusters: From Herschel to Dreyer's New General Catalogue, Cambridge University Press.
