# Quality control tools

Scripts used to judge a change before it reaches a project. They run with the Python environment where astro_gpu is
installed (on the development PC: `C:\Users\Nicolas\sandbox\ComfyUI\.venv\Scripts\python.exe`).

| Script | What it does |
|---|---|
| `nonreg.py` | Copies a project, renders its deliverables with their saved settings, recomputes the out-of-date stages, renders again, and measures the difference (`nonreg.json`, a map of the changes). Never writes to the source project. |
| `blind.py` | Blind comparison sheets: candidate renders under letters, the key saved apart; the user's reference first when the project has one; bright and medium stars and chosen zones at 100 %. |
| `judge.py` | Sheets in the framing of the user's reference image (overview and fixed zones per project); used by `blind.py`. |
| `variants_sheet.py` | Stars and zones of several renders side by side; used by `blind.py`. |
| `qc_prompt.md` | The instruction given to the independent judges. |
| `render_saved.py` | Renders a project's deliverables again with their saved settings. |
| `run_stage.py` | Runs given stages of a test copy, with parameters changed in memory only. |

## How a change is judged

1. Run it on copies (`nonreg.py`) of the reference projects: the Heart (refractor), Iris, Rosette and Horsehead
   (Newtonian). The real projects are not touched.
2. If the renders differ, make blind sheets (`blind.py`) of the before and after renders, with the user's
   reference.
3. Give the sheets to judges that know nothing of the change (`qc_prompt.md`); they score and give a verdict.
4. A change is kept only if it is not worse anywhere. Not worse does not mean identical.

Earlier diagnostic scripts (2026-10-13 to 2026-10-17) remain in `D:\astro_gpu_diag\scripts` as an archive; they
call a Python environment that no longer exists.
