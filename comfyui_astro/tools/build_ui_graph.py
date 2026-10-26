"""Build a ComfyUI graph (UI format, laid out) from a graph in API format and a layout, with a Note.

python build_ui_graph.py spec.json

spec.json: {"api": "graph_api.json", "out": "graph.json", "layout": {"<node id>": [x, y, w, h], ...},
            "groups": [{"title": ..., "bounding": [x, y, w, h], "color": "#..."}],
            "note": {"file": "note.txt", "pos": [x, y], "size": [w, h]}, "server": "http://127.0.0.1:8188"}
The widget values are read from the running ComfyUI (object_info) in its input order, advanced ones included; a widget
missing from the API graph takes its default. Nodes without a layout entry are placed in a column on the right.
"""
import json
import sys
import urllib.request
from pathlib import Path

WIDGETS = {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}


def main():
    spec_file = Path(sys.argv[1])
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    base = spec_file.parent
    api = json.loads((base / spec["api"]).read_text(encoding="utf-8"))
    server = spec.get("server", "http://127.0.0.1:8188")
    info: dict = {}

    def obj(cls):
        if cls not in info:
            with urllib.request.urlopen(f"{server}/object_info/{cls}") as r:
                info[cls] = json.loads(r.read())[cls]
        return info[cls]

    layout = spec.get("layout", {})
    nodes, links = {}, []
    spare_y = 0
    max_x = max([v[0] + v[2] for v in layout.values()] or [0])
    for sid in sorted(api, key=int):          # create every node first (outputs), then wire
        a = api[sid]
        o = obj(a["class_type"])
        if sid in layout:
            x, y, w, h = layout[sid]
        else:
            x, y, w, h = max_x + 80, spare_y, 400, 300
            spare_y += 340
        nodes[sid] = {"id": int(sid), "type": a["class_type"], "pos": [x, y], "size": [w, h], "flags": {},
                      "order": int(sid), "mode": 0, "inputs": [], "properties": {"Node name for S&R": a["class_type"]},
                      "widgets_values": [],
                      "outputs": [{"name": nm, "type": t, "links": [], "slot_index": i}
                                  for i, (t, nm) in enumerate(zip(o["output"], o["output_name"]))]}
    for sid in sorted(api, key=int):
        a, n = api[sid], nodes[sid]
        o = obj(a["class_type"])
        for sec in ("required", "optional"):
            group = o["input"].get(sec, {})
            for name in o.get("input_order", {}).get(sec, list(group)):
                spec_in = group[name]
                typ = spec_in[0] if isinstance(spec_in[0], str) else "COMBO"
                val = a["inputs"].get(name)
                if typ in WIDGETS:
                    n["widgets_values"].append(val if val is not None else spec_in[1].get("default"))
                    continue
                link = None
                if isinstance(val, list):
                    link = len(links) + 1
                    src, slot = str(val[0]), val[1]
                    links.append([link, int(src), slot, int(sid), len(n["inputs"]), typ])
                    nodes[src]["outputs"][slot]["links"].append(link)
                n["inputs"].append({"name": name, "type": typ, "link": link})
    nid = max(int(k) for k in nodes) + 1
    if spec.get("note"):
        nt = spec["note"]
        nodes[str(nid)] = {"id": nid, "type": "Note", "pos": nt["pos"], "size": nt["size"], "flags": {},
                           "order": nid, "mode": 0, "inputs": [], "outputs": [], "properties": {},
                           "widgets_values": [(base / nt["file"]).read_text(encoding="utf-8").strip()],
                           "color": "#432", "bgcolor": "#653"}
    groups = [{"title": g["title"], "bounding": g["bounding"], "color": g.get("color", "#3f789e"),
               "font_size": 24, "flags": {}} for g in spec.get("groups", [])]
    ui = {"last_node_id": max(int(k) for k in nodes), "last_link_id": len(links),
          "nodes": [nodes[k] for k in sorted(nodes, key=int)], "links": links, "groups": groups, "config": {},
          "extra": {}, "version": 0.4}
    (base / spec["out"]).write_text(json.dumps(ui, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{spec['out']}: {len(nodes)} nodes, {len(links)} links")


if __name__ == "__main__":
    main()
