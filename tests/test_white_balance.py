"""Issue #6: the camera's daylight white balance in Stacking. Off, the stack's key is the one it had before the
setting existed (the computed results keep their keys); on, the key changes; the balance is read from the raw files
(red and blue against green), none for a FITS frame; a new workflow takes the project's setting (on in a new
project), so that the workflows of a project share their stack."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from conftest import TEST_DATA

from astro_gpu import api
from astro_gpu.core import project2
from astro_gpu.core.params import params_hash
from astro_gpu.core.project import Target
from astro_gpu.io.frames import daylight_balance
from astro_gpu.stages.stack import StackParams, balanced


def test_off_the_stack_key_is_the_one_before_the_setting():
    p = StackParams()
    d = p.model_dump(mode="json")
    d.pop("white_balance")
    before = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert params_hash(p) == before
    assert params_hash(StackParams(white_balance=True)) != before


def test_the_balance_is_read_from_the_raw_files_and_applied_per_colour():
    raws = sorted((TEST_DATA / "lights").iterdir()) if TEST_DATA.exists() else []
    if not raws:
        pytest.skip("no test raw files")
    wb = daylight_balance(raws[0])
    assert wb is not None and wb[1] == 1.0 and 1.2 < wb[0] < 4.0 and 1.0 < wb[2] < 4.0    # red and blue raised
    a = np.ones((3, 2, 2), np.float32)
    assert np.allclose(balanced(a, wb)[:, 0, 0], wb) and balanced(a, None) is a


def test_a_new_workflow_takes_the_projects_setting(tmp_path):
    """The client's rule (model.add_path): on in a new project, else as the project's workflows (their stack shared);
    the engine's add_workflow stores only what it is given."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    p = project2.create(tmp_path / "P", "P", Target(name="M 1"))
    assert "stack" not in api.add_workflow(p, "Raw", "Classic stack").params           # the engine: as given
    p.workflows.clear()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    try:
        assert w.wait(120)
        full = m.add_path("Full", {"debruitage": True, "etoiles": "garder", "fond": "simple"})
        assert p.workflow(full.pid).params["stack"]["white_balance"] is True              # a new project: on
        p.workflow(full.pid).params["stack"]["white_balance"] = False                     # computed before: off
        q = m.add_path("Classic", {"debruitage": False, "etoiles": "garder", "fond": "simple"}, {"gradient": {}})
        assert not p.workflow(q.pid).params.get("stack", {}).get("white_balance", False)  # the stack shared
    finally:
        w.stop()
