"""Send a workflow (ComfyUI API format) to a running ComfyUI and wait for it: python run_prompt.py workflow.json
[--server http://127.0.0.1:8188] [--set node.input=value ...]. Prints each node's outcome and the outputs."""
import argparse
import json
import sys
import time
import urllib.request
import uuid


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()
    wf = json.load(open(a.workflow, encoding="utf-8"))
    for item in a.set:
        k, v = item.split("=", 1)
        node, inp = k.split(".", 1)
        try:
            d = json.loads(v)
            v = v if isinstance(d, dict) else d      # JSON objects stay text (settings changes are JSON text)
        except ValueError:
            pass
        wf[node]["inputs"][inp] = v
    pid = post(f"{a.server}/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        h = get(f"{a.server}/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            print("status:", st.get("status_str"), "completed:", st.get("completed"), f"({time.time() - t0:.0f} s)")
            for m in st.get("messages", []):
                if m[0] in ("execution_error", "execution_interrupted"):
                    print(m[0], json.dumps(m[1], default=str)[:2000])
            for node, out in h[pid].get("outputs", {}).items():
                print("output", node, json.dumps(out)[:400])
            sys.exit(0 if st.get("status_str") == "success" else 1)
        time.sleep(2)
    print("timeout")
    sys.exit(2)


if __name__ == "__main__":
    main()
