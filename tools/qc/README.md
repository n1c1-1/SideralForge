# Quality control tools

Scripts used to judge a change before it reaches a project. They run with the Python environment where astro_gpu is
installed (on the development PC: `C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe`).

| Script | What it does |
|---|---|
| `nonreg.py` | Copies a project, renders its deliverables with their saved settings, recomputes the out-of-date stages, renders again, and measures the difference (`nonreg.json`, a map of the changes). Never writes to the source project. |
| `blind.py` | Blind comparison sheets: candidate renders under letters, the key saved apart; the user's reference first when the project has one; bright and medium stars and chosen zones at 100 %. |
| `changed_zones.py` | The zones where two renders differ most (as fractions, for `blind.py --zones`). |
| `judge.py` | Sheets in the framing of the user's reference image (overview and fixed zones per project); used by `blind.py`. |
| `variants_sheet.py` | Stars and zones of several renders side by side; used by `blind.py`. |
| `qc_prompt.md` | The instruction given to the independent judges. |
| `render_saved.py` | Renders a project's deliverables again with their saved settings. |
| `run_stage.py` | Runs given stages of a test copy, with parameters changed in memory only. |
| `jeux.py` | Test sets: chosen frames of other cameras' sessions (and of the user's) copied to `F:\astro_essais`, processed as projects kept apart from the user's (`ASTRO_GPU_HOME`), rendered with 'Natural'. `build`, `run`, `list`. |
| `bulletin.py` | Report card of a processed set: the stages' measurements, the export's checks, sky flatness and colour cast, star size and elongation measured on the render; `--sheet` adds a sheet (whole image and 100 % crops). |

## The v2 set (P17)

`qc2.py` is the v2 port of `judge.py`, `blind.py`, `bulletin.py` and `nonreg.py` (modules `qc_addr`, `qc_pairs`,
`qc_frame`, `qc_sheets`, `qc_report`, `qc_render`, `qc_out`; tests `tests/test_qc_v2.py`). The old scripts above still
work on v1 projects and are not edited.

- A render is `PROJECT::WORKFLOW` (v2) or `PROJECT::DELIVERABLE` (v1), with `=IMAGE` to name the file to judge.
- Everything is written into the folder given by `--out` (P14: `F:/astro_projets/_qc/`): `sheets/` for the judges,
  `_keys/` (letter -> render, seed) kept apart, `bulletin/`, `nonreg/`, `pairs/`, `renders/`. An `--out` inside a project
  or in the folder of an image read is refused, and no write under a project is possible during a run.
- Each old image is paired with the v2 workflow of the same preparation (`pairs`): `render_prep_learned` goes with
  "Full + neural stars" or "Full + earlier neural layer", never with "Full".
- Each render is mapped with the alignment (or astrometry) of its own project; letters are drawn at random.

```
qc2.py pairs OLD [OLD ...] [--new NEW] [--out DIR]
qc2.py pair-sheets --out DIR --old OLD --new NEW [--candidates DIR | --new-image NAME=PATH | --render DEVICE]
qc2.py sheets --out DIR --short heart_01 [--seed N] CAND CAND ...
qc2.py bulletin --out DIR [--sheet] CAND ...        qc2.py nonreg --out DIR --case NAME BEFORE AFTER
```

## How a change is judged

1. Run it on copies (`nonreg.py`) of the reference projects: the Heart (refractor), Iris, Rosette and Horsehead
   (Newtonian). The real projects are not touched.
2. If the renders differ, make blind sheets (`blind.py`) of the before and after renders, with the user's
   reference.
3. Give the sheets to judges that know nothing of the change (`qc_prompt.md`); they score and give a verdict.
4. A change is kept only if it is not worse anywhere. Not worse does not mean identical.

Earlier diagnostic scripts (2026-10-13 to 2026-10-17) remain in `D:\astro_gpu_diag\scripts` as an archive; they
call a Python environment that no longer exists.
