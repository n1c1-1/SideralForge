"""Open a workflow in ComfyUI's interface (headless Chromium) and take a screenshot: checks that a graph loads with
its nodes and links. python ui_check.py workflow.json shot.png [--server http://127.0.0.1:8188] [--run]"""
import argparse
import json
import time

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("shot")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--run", action="store_true", help="queue the graph and wait for it")
    ap.add_argument("--wait", type=float, default=0, help="extra seconds before the screenshot")
    ap.add_argument("--width", type=int, default=1900)
    ap.add_argument("--height", type=int, default=1100)
    a = ap.parse_args()
    wf = json.load(open(a.workflow, encoding="utf-8"))
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": a.width, "height": a.height})
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.goto(a.server)
        page.wait_for_function("window.app && window.app.graph", timeout=60000)
        time.sleep(2)
        if "nodes" in wf:
            page.evaluate("wf => window.app.loadGraphData(wf)", wf)
        else:
            page.evaluate("wf => window.app.loadApiJson(wf, 'workflow_api.json')", wf)
        time.sleep(2)
        info = page.evaluate("""() => ({nodes: window.app.graph._nodes.map(n => [n.type, n.title,
                                 (n.widgets || []).length, (n.inputs || []).filter(i => i.link != null).length]),
                                 links: Object.keys(window.app.graph.links || {}).length})""")
        print(json.dumps(info))
        if a.run:
            page.evaluate("() => window.app.queuePrompt(0, 1)")
            time.sleep(3)
            page.wait_for_function("() => !document.querySelector('.p-progressbar') ", timeout=600000)
        time.sleep(a.wait)
        page.screenshot(path=a.shot)
        for line in logs:
            if "error" in line.lower():
                print(line[:300])
        b.close()


if __name__ == "__main__":
    main()
