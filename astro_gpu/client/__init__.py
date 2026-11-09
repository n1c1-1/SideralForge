"""The heavy client of astro_gpu, over v2 projects.

    python -m astro_gpu.client [project folder]

The shell: app.py (main window, menus, modes), home.py, settingsdlg.py, logview.py, dialogs.py, the shared style and
widgets (widgets.py, viewer.py), the language (i18n.py, locale/), the preferences (prefs.py) and the ApiWorker
(services/worker.py). The views of the blocks, the runs and Post are mounted in app.py (see its docstring).
"""

#: the product's name, as the user sees it (window titles, About, the files it writes); the package keeps its name
APP_NAME = "FLOAT"
#: what the name stands for (never translated)
APP_MOTTO = "FLOw for AsTrophotography"
