"""The methods of Post's filters, without widgets and without the composition (postengine imports them).

Each module follows a published or documented method, named in its docstring (docs/pilot/POST_TOOLS.md has the
inventory against PixInsight and Siril):

* ``curves``: the Curves tool (PixInsight CurvesTransformation, Siril 1.4 Curves, Photoshop Curves): Akima, natural
  cubic or linear curves per channel (RGB/K, R, G, B, CIE L*, saturation, CIE c*, CIE hue), the presets of the pro tools,
  the user's presets, Photoshop's .acv files;
* ``tone``: the stretches (Siril's and PixInsight's generalised hyperbolic stretch, Siril's asinh transformation, the
  histogram transformation's midtones transfer function);
* ``detail``: the starlet (B3 spline a trous) multiscale sharpening and noise reduction, CLAHE;
* ``colour``: CIE L*a*b* conversions shared by the others, SCNR.

Every filter is the identity at its defaults (the composition keeps the image bit for bit).
"""
