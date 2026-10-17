"""Check the live render node in ComfyUI's interface (headless Chromium): load a graph with an Astro: Render (live),
run it once, move a slider, and take screenshots before and after. python live_check.py project_folder out_prefix"""
import sys
import time

from playwright.sync_api import sync_playwright

project, out = sys.argv[1], sys.argv[2]
wf = {"1": {"class_type": "AstroProject", "inputs": {"project_folder": project}},
      "2": {"class_type": "AstroRenderLive", "inputs": {"astro": ["1", 0], "source": "delivered: faithful", "view": "",
                                                         "settings_json": "", "scale": 0.25}}}
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1900, "height": 1100})
    logs = []
    page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    page.goto("http://127.0.0.1:8188")
    page.wait_for_function("window.app && window.app.graph", timeout=60000)
    time.sleep(2)
    page.evaluate("wf => window.app.loadApiJson(wf, 'live_api.json')", wf)
    time.sleep(2)
    page.evaluate("() => window.app.canvas.ds.offset = [40, 40]")
    page.evaluate("() => window.app.queuePrompt(0, 1)")
    page.wait_for_function("""() => { const n = window.app.graph._nodes.find(n => n.type === 'AstroRenderLive');
                                       return n && n._astroLive && n._astroLive.state.project &&
                                              document.querySelector('.astro-live .al-img').src }""", timeout=900000)
    time.sleep(4)
    print("state:", page.evaluate("() => document.querySelector('.astro-live .al-state').textContent"))
    page.screenshot(path=f"{out}_1.png")
    # move the first slider of the section 'Nebula' (index 2)
    page.evaluate("""() => { const s = document.querySelector('.astro-live .al-sec'); s.value = '2';
                             s.dispatchEvent(new Event('change')); }""")
    time.sleep(1)
    page.evaluate("""() => { const r = document.querySelector('.astro-live input[type=range]');
                             r.value = String(Number(r.max) * 0.8); r.dispatchEvent(new Event('input')); }""")
    time.sleep(5)
    print("state:", page.evaluate("() => document.querySelector('.astro-live .al-state').textContent"))
    print("json:", page.evaluate("""() => window.app.graph._nodes.find(n => n.type === 'AstroRenderLive')
                                       .widgets.find(w => w.name === 'settings_json').value"""))
    page.screenshot(path=f"{out}_2.png")
    for line in logs:
        if "error" in line.lower() and "404" not in line:
            print(line[:300])
    b.close()
