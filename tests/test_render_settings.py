from astro_gpu.render.settings import PRESETS, RenderSettings


def test_defaults_are_plain_values():
    # a trailing comma after a field definition makes its default a tuple; the engine then fails on comparisons
    bad = [k for k, v in RenderSettings.model_fields.items() if not isinstance(v.default, (bool, int, float, str))]
    assert not bad, bad


def test_presets_are_valid():
    for name, values in PRESETS.items():
        RenderSettings(**values)
