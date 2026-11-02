"""The external data the product reads, credited in Help › About ("Data sources").

Each entry was checked on the source itself on 2 November 2026 (the provider's page, the data's own header or README,
the repository): who made it, its licence, the acknowledgement it asks for (copied as written, in English, and never
translated), the references to cite. The names, licences, references and acknowledgements stay as they are; the
labels and what each source is used for are translated.

Gaia, NSNS and Steinicke's NGC/IC data are free for non-commercial use only: a commercial use of the product needs
their permission first.
"""
from __future__ import annotations

import html

from .i18n import _

#: {"name", "by", "use" (English, translated), "licence", "link", "cite" (references), "ack" (as written), "note"
#: (English, translated)}
DATA_SOURCES: list[dict] = [
    dict(name="Gaia DR3", by="ESA/Gaia/DPAC",
         use="Star positions, magnitudes and spectra: plate solving, colour calibration, star models.",
         licence="CC BY-NC 3.0 IGO", link="https://www.cosmos.esa.int/gaia",
         cite="Gaia Collaboration, Prusti et al. 2016, A&A 595, A1; Gaia Collaboration, Vallenari et al. 2023, "
              "A&A 674, A1; De Angeli et al. 2023, A&A 674, A2; Montegriffo et al. 2023, A&A 674, A3.",
         ack="This work has made use of data from the European Space Agency (ESA) mission Gaia "
             "(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, "
             "https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national "
             "institutions, in particular the institutions participating in the Gaia Multilateral Agreement."),
    dict(name="Siril's Gaia DR3 extracts", by="I. Cass, A. Knagg-Baugh, C. Melis, C. Richard (2025), Zenodo",
         use="Gaia stars and spectra read offline, when Siril's catalogues are installed.",
         licence="CC BY 4.0", link="https://doi.org/10.5281/zenodo.14692304",
         cite="Siril Astrometry Catalogue extracted from Gaia DR3, v1.0.0, doi:10.5281/zenodo.14692304; Siril "
              "Spectrophotometric Catalog extracted from Gaia DR3, v1.0.1, doi:10.5281/zenodo.14738271."),
    dict(name="Hipparcos and Tycho-2", by="ESA",
         use="The bright stars of the plate solving without a known position.",
         licence="ESA", link="https://www.cosmos.esa.int/web/hipparcos",
         cite="ESA 1997, The Hipparcos and Tycho Catalogues (VizieR I/239); Høg et al. 2000, A&A 355, L27 "
              "(VizieR I/259)."),
    dict(name="VizieR", by="CDS, Strasbourg",
         use="Access to the Gaia, Hipparcos, Tycho-2 and Sharpless catalogues.",
         licence="CDS", link="https://vizier.cds.unistra.fr",
         cite="Ochsenbein et al. 2000, A&AS 143, 23; doi:10.26093/cds/vizier.",
         ack="This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France "
             "(DOI : 10.26093/cds/vizier). The original description of the VizieR service was published in 2000, "
             "A&AS 143, 23"),
    dict(name="GAVO Data Center", by="Zentrum für Astronomie, Heidelberg",
         use="Gaia stars and spectra when VizieR does not answer.",
         licence="ESA/Gaia/DPAC", link="https://dc.zah.uni-heidelberg.de",
         cite="Demleitner and Andrae 2022, Monte Carlo sampled DR3 XP spectra, doi:10.21938/W:hWbPah3wBabuqewQULTA."),
    dict(name="ESA Gaia archive", by="ESA",
         use="Gaia, Hipparcos and Tycho-2 stars when the other services do not answer.",
         licence="CC BY-NC 3.0 IGO", link="https://gea.esac.esa.int/archive/"),
    dict(name="OpenNGC", by="Mattia Verga",
         use="Names, types and sizes of the NGC, IC and Messier objects.",
         licence="CC BY-SA 4.0", link="https://github.com/mattiaverga/OpenNGC"),
    dict(name="Sharpless catalogue of HII regions", by="S. Sharpless 1959, ApJS 4, 257 (VizieR VII/20)",
         use="Names and sizes of the Sharpless regions.", licence="CDS",
         link="https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII/20"),
    dict(name="Siril's catalogue lists", by="The Siril team (installed with Siril)",
         use="Object names when OpenNGC or the Sharpless catalogue cannot be had, and the links between their "
             "designations; dark nebula names for the target search.",
         licence="GPL-3.0 (Siril)", link="https://siril.org",
         cite="NGC and IC: W. Steinicke, Revised NGC/IC (http://www.klima-luft.de/steinicke), free for "
              "non-commercial use; Sharpless 1959 (VizieR VII/20); Lynds 1962, ApJS 7, 1 (VizieR VII/7A).",
         note="The origin of the Messier list could not be established."),
    dict(name="NSNS DR0.2 H-alpha", by="Stefan Ziegenbalg",
         use="The H-alpha reference of the background in the northern sky.",
         licence="CC BY-NC-SA 4.0", link="http://www.simg.de/nebulae3/dr0_2",
         cite="Ziegenbalg 2025, RNAAS 9, 227, doi:10.3847/2515-5172/adfec7.",
         ack="Northern Sky Narrowband Survey by Stefan Ziegenbalg. It uses WHAM data for background correction and "
             "intensity calibration (the Wisconsin H-alpha Mapper and its H-alpha Sky Survey have been funded "
             "primarily by the National Science Foundation)."),
    dict(name="Finkbeiner H-alpha composite", by="D. P. Finkbeiner (WHAM, VTSS and SHASSA surveys); HiPS by CDS",
         use="The H-alpha reference of the background outside the northern sky.",
         licence="ODbL-1.0 (CDS HiPS)", link="https://doi.org/10.26093/cds/aladin/3a4t-avb",
         cite="Finkbeiner 2003, ApJS 146, 407."),
    dict(name="hips2fits", by="CDS, Strasbourg",
         use="Cut-outs of the H-alpha surveys on the field.", licence="CDS",
         link="https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
         ack="This research made use of hips2fits, a service provided by CDS."),
    dict(name="SIMBAD", by="CDS, Strasbourg",
         use="The position of a target from its name.", licence="ODbL", link="https://simbad.cds.unistra.fr",
         cite="Wenger et al. 2000, A&AS 143, 9.",
         ack="This research has made use of the SIMBAD database, operated at CDS, Strasbourg, France"),
    dict(name="NED", by="NASA/IPAC",
         use="The position of a target from its name, when SIMBAD does not know it.",
         licence="NASA/IPAC", link="https://ned.ipac.caltech.edu",
         ack="This research has made use of the NASA/IPAC Extragalactic Database, which is funded by the National "
             "Aeronautics and Space Administration and operated by the California Institute of Technology."),
]


def data_sources_html() -> str:
    """The "Data sources" section of About, as rich text (the labels and uses translated)."""
    e = html.escape
    out = [f"<h3>{e(_('Data sources'))}</h3>",
           f"<p>{e(_('The product reads these data. Thanks to their authors.'))}</p>"]
    for s in DATA_SOURCES:
        bits = [f"<b>{e(s['name'])}</b> · {e(s['by'])}", e(_(s["use"])),
                e(_("Licence: {licence}").format(licence=s["licence"]))
                + f" · <a href='{e(s['link'])}'>{e(s['link'])}</a>"]
        if s.get("cite"):
            bits.append(e(_("To cite: {refs}").format(refs=s["cite"])))
        if s.get("ack"):
            bits.append(e(_("Acknowledgement asked:")) + f" <i>“{e(s['ack'])}”</i>")
        if s.get("note"):
            bits.append(e(_(s["note"])))
        out.append("<p>" + "<br>".join(bits) + "</p>")
    out.append(f"<p>{e(_('Gaia, NSNS and the NGC/IC data of W. Steinicke are free for non-commercial use only.'))}</p>")
    return "".join(out)
