"""Convert a workflow in ComfyUI's API format (what the nodes compute) into the interface's format (nodes placed and
linked on the canvas, to open by drag and drop), using a running ComfyUI's node descriptions.
python api_to_ui.py api.json ui.json [--server http://127.0.0.1:8188]"""
import argparse
import json
import urllib.request

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def node_inputs(info: dict) -> list[tuple[str, str, bool]]:
    """(name, type, is_widget) of a node's inputs in the interface's order (required, then optional)."""
    out = []
    order = info.get("input_order", {})
    for group in ("required", "optional"):
        spec = info.get("input", {}).get(group, {})
        for name in order.get(group, list(spec)):
            t = spec[name][0]
            typ = "COMBO" if isinstance(t, list) else t
            out.append((name, typ, typ in WIDGET_TYPES))
    return out


def convert(api: dict, info: dict, per_row: int = 5) -> dict:
    ids = {k: i + 1 for i, k in enumerate(api)}
    depth: dict[str, int] = {}

    def d(k):
        if k not in depth:
            ups = [v[0] for v in api[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in api]
            depth[k] = 1 + max((d(u) for u in ups), default=-1)
        return depth[k]
    for k in api:
        d(k)
    rows: dict[int, int] = {}
    rows_per_col = max(sum(1 for k in api if depth[k] == c) for c in set(depth.values()))
    links, nodes = [], []
    outputs_links: dict[tuple[str, int], list[int]] = {}
    for k, n in api.items():
        for name, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and v[0] in api:
                lid = len(links) + 1
                links.append([lid, ids[v[0]], v[1], ids[k], None, None, name])
                outputs_links.setdefault((v[0], v[1]), []).append(lid)
    for k, n in api.items():
        ni = info[n["class_type"]]
        col = depth[k]
        row = rows.get(col, 0)
        rows[col] = row + 1
        band = max(rows_per_col, 1)
        col, row = col % per_row, row + (col // per_row) * (band + 1)  # long chains wrap into rows of per_row
        inputs, widgets = [], []
        for slot_name, typ, is_widget in node_inputs(ni):
            v = n["inputs"].get(slot_name)
            linked = isinstance(v, list) and len(v) == 2 and v[0] in api
            if is_widget:
                default = ni["input"].get("required", {}).get(slot_name) or ni["input"].get("optional", {}).get(slot_name)
                opts = default[1] if len(default) > 1 else {}
                first = default[0][0] if isinstance(default[0], list) else None
                val = v if (v is not None and not linked) else opts.get("default", first)
                widgets.append(val)
                if linked:
                    inputs.append({"name": slot_name, "type": typ, "widget": {"name": slot_name}, "link": None})
            else:
                inputs.append({"name": slot_name, "type": typ, "link": None})
        for li in links:
            if li[3] == ids[k]:
                slot = next(i for i, s in enumerate(inputs) if s["name"] == li[6])
                inputs[slot]["link"] = li[0]
                li[4] = slot
                li[5] = inputs[slot]["type"]
        outs = []
        for s, (typ, oname) in enumerate(zip(ni.get("output", []), ni.get("output_name", []))):
            outs.append({"name": oname, "type": typ, "links": outputs_links.get((k, s), []), "slot_index": s})
        nodes.append({"id": ids[k], "type": n["class_type"], "pos": [40 + 420 * col, 40 + 330 * row],
                      "size": [380, 280], "flags": {}, "order": depth[k], "mode": 0,
                      "inputs": inputs, "outputs": outs,
                      "properties": {"Node name for S&R": n["class_type"]}, "widgets_values": widgets})
    return {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes,
            "links": [li[:6] for li in links], "groups": [], "config": {}, "extra": {}, "version": 0.4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("api")
    ap.add_argument("ui")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--per-row", type=int, default=5, help="columns before a long graph wraps")
    a = ap.parse_args()
    api = json.load(open(a.api, encoding="utf-8"))
    with urllib.request.urlopen(f"{a.server}/object_info") as r:
        info = json.loads(r.read())
    json.dump(convert(api, info, a.per_row), open(a.ui, "w", encoding="utf-8"), indent=1)
    print("written", a.ui)


if __name__ == "__main__":
    main()
