"""Run given stages of a project with the astro_gpu found first on the Python path, without the dependency check.

    python tools/qc/run_stage.py PROJECT [--param stage.key=value ...] STAGE [STAGE ...]

The --param values change the parameters in memory only (project.json is not modified). Meant for test copies:
on a real project use the interface or `astro-gpu run`, which check what is out of date.
"""
import json
import sys

import astro_gpu
from astro_gpu.core.project import Project
from astro_gpu.core.runner import Pipeline
from astro_gpu.stages import all_stages


def main():
    print("code:", astro_gpu.__file__)
    args = sys.argv[2:]
    proj = Project(sys.argv[1])
    stages = []
    while args:
        a = args.pop(0)
        if a == "--param":
            k, v = args.pop(0).split("=", 1)
            st, key = k.split(".", 1)
            try:
                v = json.loads(v)
            except ValueError:
                pass
            proj.config.params.setdefault(st, {})[key] = v
            print("param", st, key, "=", v)
        else:
            stages.append(a)
    pipe = Pipeline(proj, all_stages())
    for name in stages:
        r = pipe._run_stage(pipe.by_name[name])
        print(name, r)
        if r != "done":
            break


if __name__ == "__main__":
    main()
