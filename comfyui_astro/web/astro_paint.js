// Painted mask node (Astro: Painted mask): a brush over the image in the node. The mask is the alpha channel of a
// canvas at the image's size (at most 1600 pixels wide), shown in red over the image, and kept in the node as a PNG
// ('painted' widget): the Python side resizes it to the image it is used with.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CSS = `
.astro-paint { display: flex; flex-direction: column; gap: 6px; width: 100%; height: 100%; box-sizing: border-box;
  font: 12px sans-serif; color: var(--input-text, #ddd); }
.astro-paint .ap-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.astro-paint button { background: var(--comfy-input-bg, #222); color: inherit; border: 1px solid var(--border-color, #444);
  border-radius: 4px; padding: 2px 8px; }
.astro-paint button.ap-on { background: var(--p-primary-color, #e8b04a); color: #111; }
.astro-paint label { display: flex; gap: 4px; align-items: center; }
.astro-paint input[type=range] { width: 90px; }
.astro-paint .ap-state { margin-left: auto; opacity: 0.7; }
.astro-paint .ap-stage { flex: 1; min-height: 0; position: relative; background: #111; border-radius: 4px;
  overflow: hidden; }
.astro-paint .ap-wrap { position: absolute; cursor: none; touch-action: none; }
.astro-paint .ap-wrap img, .astro-paint .ap-wrap canvas { position: absolute; left: 0; top: 0; width: 100%;
  height: 100%; }
.astro-paint .ap-cursor { position: absolute; border: 1px solid #fff; border-radius: 50%; pointer-events: none;
  box-shadow: 0 0 0 1px #000; display: none; }
`;

function setup(node) {
  const dataW = node.widgets?.find((w) => w.name === "painted");
  if (dataW) {
    dataW.hidden = true;
    if (dataW.element) dataW.element.style.display = "none";
    dataW.computeSize = () => [0, -4];
  }
  const root = document.createElement("div");
  root.className = "astro-paint";
  root.innerHTML = `<style>${CSS}</style>
    <div class="ap-bar">
      <button class="ap-paint ap-on" title="Paint where the other view is wanted (left button)">Paint</button>
      <button class="ap-erase" title="Erase (or the right button)">Erase</button>
      <label title="Brush size, in % of the image width (or the mouse wheel over the image)">size
        <input type="range" class="ap-size" min="0.2" max="25" step="0.1" value="4"></label>
      <label title="Brush edge: 0 = very soft, 1 = hard">hardness
        <input type="range" class="ap-hard" min="0" max="1" step="0.05" value="0.4"></label>
      <label title="How strongly the mask shows over the image">overlay
        <input type="range" class="ap-opac" min="0" max="1" step="0.05" value="0.45"></label>
      <button class="ap-undo" title="Undo the last stroke">Undo</button>
      <button class="ap-clear" title="Empty mask">Clear</button>
      <button class="ap-fill" title="Full mask">Fill</button>
      <span class="ap-state">Run the graph once: the image to paint on appears</span>
    </div>
    <div class="ap-stage"><div class="ap-wrap"><img/><canvas class="ap-over"></canvas>
      <div class="ap-cursor"></div></div></div>`;
  const $ = (s) => root.querySelector(s);
  const stage = $(".ap-stage"), wrap = $(".ap-wrap"), img = $(".ap-wrap img"), over = $(".ap-over");
  const cursor = $(".ap-cursor"), state = $(".ap-state"), size = $(".ap-size"), hard = $(".ap-hard");
  const opac = $(".ap-opac"), bPaint = $(".ap-paint"), bErase = $(".ap-erase");
  wrap.style.display = "none";
  for (const ev of ["pointerdown", "mousedown", "wheel", "contextmenu"]) {
    root.addEventListener(ev, (e) => e.stopPropagation());
  }
  node.addDOMWidget("astro_paint", "astro_paint", root, { serialize: false, getMinHeight: () => 480 });
  node.setSize([Math.max(node.size[0], 900), Math.max(node.size[1], 700)]);

  const mask = document.createElement("canvas");          // the mask: its alpha channel
  const mctx = mask.getContext("2d", { willReadFrequently: true });
  const octx = over.getContext("2d");
  const undo = [];
  let erase = false, drawing = null, last = null;

  function redraw() {
    if (!mask.width) return;
    if (over.width !== mask.width || over.height !== mask.height) {
      over.width = mask.width;
      over.height = mask.height;
    }
    octx.globalCompositeOperation = "copy";
    octx.drawImage(mask, 0, 0);
    octx.globalCompositeOperation = "source-in";
    octx.fillStyle = "rgb(255,70,40)";
    octx.fillRect(0, 0, over.width, over.height);
    octx.globalCompositeOperation = "source-over";
    over.style.opacity = opac.value;
  }

  function layout() {                                     // the image fitted into the stage
    if (!img.naturalWidth) return;
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const k = Math.min(sw / img.naturalWidth, sh / img.naturalHeight);
    const w = Math.round(img.naturalWidth * k), h = Math.round(img.naturalHeight * k);
    Object.assign(wrap.style, { width: w + "px", height: h + "px", left: Math.round((sw - w) / 2) + "px",
                                top: Math.round((sh - h) / 2) + "px", display: "block" });
  }
  new ResizeObserver(layout).observe(stage);

  function resizeMask(w, h) {                             // keep what is painted, stretched to the new size
    if (mask.width === w && mask.height === h) return;
    const old = mask.width ? mctx.getImageData(0, 0, mask.width, mask.height) : null;
    const tmp = document.createElement("canvas");
    if (old) {
      tmp.width = mask.width;
      tmp.height = mask.height;
      tmp.getContext("2d").putImageData(old, 0, 0);
    }
    mask.width = w;
    mask.height = h;
    if (old) mctx.drawImage(tmp, 0, 0, w, h);
  }

  function save() {
    if (!dataW) return;
    const px = mctx.getImageData(0, 0, mask.width, mask.height).data;
    let any = false;
    for (let i = 3; i < px.length; i += 64) if (px[i]) { any = true; break; }
    dataW.value = any ? mask.toDataURL("image/png") : "";
    node.setDirtyCanvas?.(true, true);
  }

  function pushUndo() {
    undo.push(mctx.getImageData(0, 0, mask.width, mask.height));
    if (undo.length > 8) undo.shift();
  }

  function radius() { return (Number(size.value) / 100) * mask.width / 2; }

  function stamp(x, y) {
    const r = Math.max(radius(), 0.5);
    const g = mctx.createRadialGradient(x, y, r * Number(hard.value) * 0.98, x, y, r);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    mctx.globalCompositeOperation = erase || drawing === "erase" ? "destination-out" : "source-over";
    mctx.fillStyle = g;
    mctx.beginPath();
    mctx.arc(x, y, r, 0, Math.PI * 2);
    mctx.fill();
    mctx.globalCompositeOperation = "source-over";
  }

  function toMask(e) {                                    // pointer -> mask pixels (the node may be zoomed)
    const r = wrap.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width * mask.width, (e.clientY - r.top) / r.height * mask.height, r];
  }

  function showCursor(e) {
    const [x, y, r] = toMask(e);
    const k = wrap.clientWidth / mask.width, d = radius() * 2 * k;
    Object.assign(cursor.style, { display: "block", width: d + "px", height: d + "px",
                                  left: (x * k - d / 2) + "px", top: (y * k - d / 2) + "px" });
    return [x, y, r];
  }

  wrap.addEventListener("pointerdown", (e) => {
    if (!mask.width) return;
    e.preventDefault();
    wrap.setPointerCapture(e.pointerId);
    pushUndo();
    drawing = e.button === 2 ? "erase" : "paint";
    const [x, y] = toMask(e);
    stamp(x, y);
    last = [x, y];
    redraw();
  });
  wrap.addEventListener("pointermove", (e) => {
    const [x, y] = showCursor(e);
    if (!drawing) return;
    const step = Math.max(radius() * 0.25, 0.5);
    const dx = x - last[0], dy = y - last[1], n = Math.ceil(Math.hypot(dx, dy) / step);
    for (let i = 1; i <= n; i++) stamp(last[0] + dx * i / n, last[1] + dy * i / n);
    last = [x, y];
    redraw();
  });
  const end = () => { if (drawing) { drawing = null; save(); } };
  wrap.addEventListener("pointerup", end);
  wrap.addEventListener("pointercancel", end);
  wrap.addEventListener("pointerleave", () => { cursor.style.display = "none"; });
  wrap.addEventListener("contextmenu", (e) => e.preventDefault());
  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const v = Number(size.value) * (e.deltaY > 0 ? 0.9 : 1.1);
    size.value = String(Math.min(25, Math.max(0.2, v)));
    showCursor(e);
  }, { passive: false });

  function setMode(er) {
    erase = er;
    bPaint.classList.toggle("ap-on", !er);
    bErase.classList.toggle("ap-on", er);
  }
  bPaint.onclick = () => setMode(false);
  bErase.onclick = () => setMode(true);
  opac.oninput = redraw;
  $(".ap-undo").onclick = () => {
    const d = undo.pop();
    if (d && d.width === mask.width && d.height === mask.height) { mctx.putImageData(d, 0, 0); redraw(); save(); }
  };
  $(".ap-clear").onclick = () => { if (!mask.width) return; pushUndo(); mctx.clearRect(0, 0, mask.width, mask.height);
                                   redraw(); save(); };
  $(".ap-fill").onclick = () => { if (!mask.width) return; pushUndo(); mctx.fillStyle = "#fff";
                                  mctx.fillRect(0, 0, mask.width, mask.height); redraw(); save(); };

  function loadSaved() {                                  // a mask saved with the graph
    if (!dataW || !dataW.value) return;
    const m = new Image();
    m.onload = () => {
      if (!mask.width) { mask.width = m.naturalWidth; mask.height = m.naturalHeight; }
      mctx.clearRect(0, 0, mask.width, mask.height);
      mctx.drawImage(m, 0, 0, mask.width, mask.height);
      redraw();
    };
    m.src = dataW.value;
  }
  loadSaved();

  node._astroPaint = {
    loadSaved,
    executed(info) {
      img.onload = () => {
        resizeMask(img.naturalWidth, img.naturalHeight);
        layout();
        redraw();
        state.textContent = `${info.width} x ${info.height} - left: paint, right: erase, wheel: size`;
      };
      img.src = api.apiURL(`/view?filename=${encodeURIComponent(info.filename)}&type=${info.type}` +
                           `&subfolder=${encodeURIComponent(info.subfolder || "")}&t=${Date.now()}`);
    },
    mask,
    img,
  };
}

app.registerExtension({
  name: "astro.paintedMask",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AstroPaintedMask") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = created?.apply(this, arguments);
      setup(this);
      return ret;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const ret = configured?.apply(this, arguments);
      this.setSize([Math.max(this.size[0], 900), Math.max(this.size[1], 700)]);
      this._astroPaint?.loadSaved();                     // the widget values are restored only now
      return ret;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      const info = message?.astro_paint?.[0];
      if (info && this._astroPaint) this._astroPaint.executed(info);
    };
  },
});
