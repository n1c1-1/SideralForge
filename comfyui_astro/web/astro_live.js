// Live render node (Astro: Render (live)): the rendering settings as sliders, by section, and a preview that
// follows them. The preview is rendered by the server (/astro/preview) from the processing's layers kept on the GPU,
// without running the graph. The sliders' changes are kept in the node ('settings_json'), on top of the source.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let schemaPromise = null;
function getSchema() {
  if (!schemaPromise) schemaPromise = api.fetchApi("/astro/render_schema").then((r) => r.json());
  return schemaPromise;
}

const CSS = `
.astro-live { display: flex; flex-direction: column; gap: 6px; width: 100%; height: 100%; box-sizing: border-box;
  font: 12px sans-serif; color: var(--input-text, #ddd); }
.astro-live .al-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.astro-live select, .astro-live button { background: var(--comfy-input-bg, #222); color: inherit;
  border: 1px solid var(--border-color, #444); border-radius: 4px; padding: 2px 6px; }
.astro-live .al-state { margin-left: auto; opacity: 0.7; }
.astro-live .al-from { max-width: 260px; }
.astro-live .al-cmp.al-held { background: var(--p-primary-color, #e8b04a); color: #111; }
.astro-live .al-body { display: flex; gap: 8px; flex: 1; min-height: 0; }
.astro-live .al-sliders { width: 380px; flex: none; overflow-y: auto; padding-right: 4px; }
.astro-live .al-row { display: grid; grid-template-columns: 170px 1fr 46px; gap: 6px; align-items: center;
  margin: 3px 0; }
.astro-live .al-row.al-changed .al-lab { color: var(--p-primary-color, #e8b04a); }
.astro-live .al-lab { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.astro-live .al-num { text-align: right; font-variant-numeric: tabular-nums; }
.astro-live input[type=range] { width: 100%; }
.astro-live .al-view { flex: 1; min-width: 0; display: flex; align-items: center; justify-content: center;
  background: #111; border-radius: 4px; overflow: hidden; }
.astro-live .al-img { max-width: 100%; max-height: 100%; object-fit: contain; }
`;

function fmt(v, step) {
  const d = step >= 1 ? 0 : Math.min(4, Math.max(0, Math.ceil(-Math.log10(step))));
  return Number(v).toFixed(d);
}

function setup(node) {
  const jsonW = node.widgets?.find((w) => w.name === "settings_json");
  if (jsonW) {
    jsonW.hidden = true;
    if (jsonW.element) jsonW.element.style.display = "none";
    jsonW.computeSize = () => [0, -4];
  }
  const st = { project: null, values: {}, source: {}, schema: null, section: 0, advanced: false,
               timer: null, busy: false, again: false, url: null, srcUrl: null, holding: false };
  const widget = (name) => node.widgets?.find((w) => w.name === name);
  function loadSaved() {                     // the sliders' changes kept in the node (restored after creation)
    try { st.values = jsonW && jsonW.value ? JSON.parse(jsonW.value) : {}; } catch { st.values = {}; }
  }
  loadSaved();

  const root = document.createElement("div");
  root.className = "astro-live";
  root.innerHTML = `<style>${CSS}</style>
    <div class="al-bar">
      <select class="al-sec" title="Section of the settings"></select>
      <label title="Show every setting of the section"><input type="checkbox" class="al-adv"> advanced</label>
      <button class="al-reset" title="All sliders back to the source's settings">Reset</button>
      <span class="al-state">Run the graph once: the sliders and the preview start then</span>
    </div>
    <div class="al-bar">
      <label title="The rendering the sliders start from: a delivered rendering, a saved view, a preset">start from
        <select class="al-from"></select></label>
      <button class="al-save" title="Save the sliders as a view of the project (shown in the heavy client too)">Save as view...</button>
      <button class="al-cmp" title="Hold to see the starting rendering, release to see the sliders' one">Compare (hold)</button>
      <label title="Blue: pure black   Red: pure white   Amber: nebula above 92 % (detail loss)"><input type="checkbox" class="al-clip"> clipping</label>
    </div>
    <div class="al-body"><div class="al-sliders"></div><div class="al-view"><img class="al-img"/></div></div>`;
  const sec = root.querySelector(".al-sec");
  const adv = root.querySelector(".al-adv");
  const reset = root.querySelector(".al-reset");
  const state = root.querySelector(".al-state");
  const sliders = root.querySelector(".al-sliders");
  const view = root.querySelector(".al-view");
  const img = root.querySelector(".al-img");
  const from = root.querySelector(".al-from");
  const cmp = root.querySelector(".al-cmp");
  const clip = root.querySelector(".al-clip");
  // the canvas must not take the mouse inside the sliders
  for (const ev of ["pointerdown", "mousedown", "wheel"]) root.addEventListener(ev, (e) => e.stopPropagation());

  node.addDOMWidget("astro_live", "astro_live", root, { serialize: false, getMinHeight: () => 560 });
  node.setSize([Math.max(node.size[0], 960), Math.max(node.size[1], 760)]);

  function current(f) {
    if (f.name in st.values) return st.values[f.name];
    if (f.name in st.source) return st.source[f.name];
    return f.default;
  }

  function build() {
    if (!st.schema) return;
    sliders.innerHTML = "";
    const s = st.schema[st.section];
    if (!s) return;
    for (const f of s.fields) {
      if (f.level === "advanced" && !st.advanced) continue;
      const row = document.createElement("div");
      row.className = "al-row" + (f.name in st.values ? " al-changed" : "");
      row.title = [f.help, f.tip, f.unit ? "Unit: " + f.unit : ""].filter(Boolean).join("\n");
      const lab = document.createElement("span");
      lab.className = "al-lab";
      lab.textContent = f.label;
      row.appendChild(lab);
      if (f.type === "bool") {
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = !!current(f);
        cb.onchange = () => { set(f, cb.checked); row.classList.add("al-changed"); };
        row.appendChild(cb);
        row.appendChild(document.createElement("span"));
      } else {
        const r = document.createElement("input");
        r.type = "range";
        r.min = f.ge ?? 0;
        r.max = f.le ?? 1;
        r.step = f.step ?? 0.01;
        r.value = current(f);
        const num = document.createElement("span");
        num.className = "al-num";
        num.textContent = fmt(current(f), Number(r.step) < 1 ? Number(r.step) : 1);   // the exact value until moved
        r.oninput = () => {
          num.textContent = fmt(r.value, Number(r.step));
          set(f, f.type === "int" ? parseInt(r.value) : parseFloat(r.value));
          row.classList.add("al-changed");
        };
        r.ondblclick = () => {          // double click: back to the source's value
          delete st.values[f.name];
          save();
          build();
          schedule();
        };
        row.appendChild(r);
        row.appendChild(num);
      }
      sliders.appendChild(row);
    }
  }

  function set(f, v) {
    st.values[f.name] = v;
    save();
    schedule();
  }

  function save() {
    if (jsonW) jsonW.value = Object.keys(st.values).length ? JSON.stringify(st.values) : "";
    node.setDirtyCanvas?.(true, true);
  }

  function schedule() {
    clearTimeout(st.timer);
    st.timer = setTimeout(preview, 120);
  }

  async function render(settings) {          // a preview as an object URL, or an error text
    const width = Math.round(Math.max(400, view.clientWidth) * (window.devicePixelRatio || 1));
    const r = await api.fetchApi("/astro/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root: st.project.root, variant: st.project.variant, settings, width,
                             clip: clip.checked }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.error || String(r.status));
    }
    return URL.createObjectURL(await r.blob());
  }

  async function preview() {
    if (!st.project) return;
    if (st.busy) { st.again = true; return; }
    st.busy = true;
    state.textContent = "rendering...";
    const t0 = performance.now();
    try {
      const u = await render(Object.assign({}, st.source, st.values));
      if (st.url) URL.revokeObjectURL(st.url);
      st.url = u;
      if (!st.holding) img.src = st.url;
      state.textContent = `${st.project.variant || "main"} - preview ${Math.round(performance.now() - t0)} ms`;
    } catch (e) {
      state.textContent = "error: " + e;
    } finally {
      st.busy = false;
      if (st.again) { st.again = false; preview(); }
    }
  }

  // the starting rendering: the node's 'source' and 'view' widgets, chosen here from the project's list
  async function fillFrom() {
    if (!st.project) return;
    const q = `root=${encodeURIComponent(st.project.root)}&variant=${encodeURIComponent(st.project.variant)}`;
    const r = await api.fetchApi(`/astro/render_sources?${q}`);
    if (!r.ok) return;
    const s = await r.json();
    const srcW = widget("source"), viewW = widget("view");
    const options = srcW?.options?.values || [];
    from.innerHTML = "";
    const group = (label, items) => {
      if (!items.length) return;
      const g = document.createElement("optgroup");
      g.label = label;
      for (const [text, value] of items) g.appendChild(new Option(text, value));
      from.appendChild(g);
    };
    group("Delivered", s.delivered.filter((d) => options.includes(`delivered: ${d}`))
                                  .map((d) => [d, `source|delivered: ${d}`]));
    group("Views", s.views.map((v) => [v, `view|${v}`]));
    group("Presets", s.presets.map((p) => [p, `source|preset: ${p}`]));
    from.value = viewW?.value ? `view|${viewW.value}` : `source|${srcW?.value}`;
  }

  async function startFrom(choice) {
    const [kind, name] = [choice.split("|")[0], choice.slice(choice.indexOf("|") + 1)];
    const srcW = widget("source"), viewW = widget("view");
    let source;
    if (kind === "view") {
      if (viewW) viewW.value = name;
      source = "view:" + name;
    } else {
      if (srcW) srcW.value = name;
      if (viewW) viewW.value = "";
      source = name.startsWith("delivered: ") ? "results/" + name.slice(11) : name.slice(8);
    }
    const q = `root=${encodeURIComponent(st.project.root)}&variant=${encodeURIComponent(st.project.variant)}` +
              `&source=${encodeURIComponent(source)}`;
    const r = await api.fetchApi(`/astro/source_settings?${q}`);
    const vals = await r.json();
    if (!r.ok) { state.textContent = "error: " + (vals.error || r.status); return; }
    st.source = vals;
    st.values = {};
    st.srcUrl = null;
    save();
    build();
    schedule();
  }
  from.onchange = () => { if (st.project) startFrom(from.value); };
  clip.onchange = () => { st.srcUrl = null; schedule(); };

  root.querySelector(".al-save").onclick = async () => {
    if (!st.project) return;
    const viewW = widget("view");
    const name = (window.prompt("Name of the view:", viewW?.value || "") || "").trim();
    if (!name) return;
    const r = await api.fetchApi("/astro/save_view", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root: st.project.root, variant: st.project.variant, name,
                             settings: Object.assign({}, st.source, st.values) }),
    });
    const e = await r.json().catch(() => ({}));
    state.textContent = r.ok ? `view '${name}' saved` : "error: " + (e.error || r.status);
    if (r.ok) await fillFrom();
  };

  async function hold(on) {                    // Compare: the starting rendering while the button is held
    if (!st.project) return;
    st.holding = on;
    cmp.classList.toggle("al-held", on);
    if (!on) { if (st.url) img.src = st.url; return; }
    try {
      if (!st.srcUrl) st.srcUrl = await render(st.source);
      if (st.holding) img.src = st.srcUrl;
    } catch (e) {
      state.textContent = "error: " + e.message;
    }
  }
  cmp.addEventListener("pointerdown", () => hold(true));
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) cmp.addEventListener(ev, () => hold(false));

  sec.onchange = () => { st.section = Number(sec.value); build(); };
  adv.onchange = () => { st.advanced = adv.checked; build(); };
  reset.onclick = () => { st.values = {}; save(); build(); schedule(); };

  getSchema().then((sc) => {
    st.schema = sc;
    sec.innerHTML = "";
    sc.forEach((s, i) => sec.add(new Option(s.section, String(i))));
    build();
  });

  node._astroLive = {
    loadSaved() { loadSaved(); build(); },
    executed(info) {
      st.project = { root: info.root, variant: info.variant || "" };
      st.source = info.source_settings || {};
      st.srcUrl = null;
      build();
      preview();
      fillFrom();
    },
    state: st,
  };
}

app.registerExtension({
  name: "astro.renderLive",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AstroRenderLive") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = created?.apply(this, arguments);
      setup(this);
      return ret;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {   // a saved graph may hold a smaller size: keep room for the preview
      const ret = configured?.apply(this, arguments);
      this.setSize([Math.max(this.size[0], 960), Math.max(this.size[1], 760)]);
      this._astroLive?.loadSaved();                     // the widget values are restored only now
      return ret;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      const info = message?.astro_live?.[0];
      if (info && this._astroLive) this._astroLive.executed(info);
    };
  },
});
