'use strict';
/* Maquette alt1 : un moteur, plusieurs vues.
   L'image finale = somme des calques pondérés par zone (masques précalculés) puis la peinture de chaque calque. */

const D = window.DATA, A = window.ASSETS;
const W = D.size[0], H = D.size[1], N = W * H;
const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const ZK = D.zones.map((z) => z.key);
const ZONE = Object.fromEntries(D.zones.map((z) => [z.key, z]));
const ZCOL = { etoiles: '#ffffff', halos: '#ffc400', brillante: '#ff4d4d', faible: '#b46cff', fond: '#4f7dd1' };
const SUGGEST = { etoiles: ['discretes', 'sans', 'modele'], halos: ['halo', 'modele', 'discretes'], brillante: ['artistique'],
  faible: ['artistique'], fond: ['artistique'] };
const LAYERDEF = Object.fromEntries(D.layers.map((l) => [l.key, l]));
const STEPS = D.steps.concat([{ key: 'rendu', title: 'Rendu', stages: [], images: [] }]);
const SETTING_FR = {
  chroma: 'Saturation de la nébuleuse', lc_fine: 'Contraste fin', lc_medium: 'Contraste moyen', lc_large: 'Contraste large',
  neb_sharp: 'Netteté', sharpen: 'Netteté (forme d’étoile mesurée)', sharpen_lo_pct: 'Début de la netteté', fine_k: 'Détail fin',
  fine_lo_pct: 'Début du détail fin', sky_level: 'Luminosité du ciel', dark_level: 'Niveau des nuages sombres',
  white_mult: 'Point blanc', resid_boost: 'Tons bleus / reflets', colour_detail: 'Couleur fine', d10: 'Taille d’une étoile G=10',
  d7: 'Taille d’une étoile G=7', star_sat: 'Couleur des étoiles', sat_min: 'Couleur des étoiles faibles',
  learned_stars: 'Retrait appris des étoiles', stars: 'Montrer les étoiles', glare_nebula: 'Lueur avec la nébuleuse',
  star_glow: 'Lueur des étoiles brillantes',
};

/* ------------------------------------------------------------------ état */
const S = {
  layers: [],                    // calques de la pile, du bas vers le haut
  zone: Object.fromEntries(ZK.map((z) => [z, 'fidele'])),
  sel: { type: 'final' },        // final | layer | zone | step
  view: { kind: 'final' },       // final | layer | step
  zonesOn: true, compare: false, curtain: 0.5, brush: false,
  hoverZone: null, lastPoint: null, mouse: null,
  pending: new Set(),            // étapes communes marquées « à recalculer »
  changed: {},                   // réglages d'étape modifiés : {stage.name: value}
  run: null, undo: [], redo: [], showAdvanced: false, graph: false,
};
let wz = {}, zoneId, comp, compCtx, compImg, zoneOverlay = {}, stepImgs = {}, galleryImgs = {};
const fieldT = { z: 1, ox: 0, oy: 0 }, chartT = { z: 1, ox: 0, oy: 0 };

/* ------------------------------------------------------------------ outils */
function loadImage(uri) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = uri; });
}
function pixelsOf(im) {
  const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
  const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(im, 0, 0);
  return x.getImageData(0, 0, c.width, c.height).data;
}
function fmtDur(s) {
  if (s < 1) return `${s.toFixed(1).replace('.', ',')} s`;
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  const h = Math.floor(s / 3600), m = Math.round((s - h * 3600) / 60);
  return `${h} h ${String(m).padStart(2, '0')}`;
}
const fmtNum = (v, d = 1) => Number(v).toLocaleString('fr-FR', { maximumFractionDigits: d });
const layer = (k) => S.layers.find((l) => l.key === k);
const stepDef = (k) => STEPS.find((s) => s.key === k);
const stageOf = (n) => D.stages[n] || { seconds: 0, gb: 0, metrics: {}, params: [] };
const stepSeconds = (st) => st.key === 'rendu' ? 0.3 : st.stages.reduce((a, n) => a + stageOf(n).seconds, 0);
const stepGb = (st) => st.stages.reduce((a, n) => a + stageOf(n).gb, 0);
const usesLearned = (l) => l.settings.learned_stars !== false;
function zonesOf(k) { return ZK.filter((z) => S.zone[z] === k); }
function isUsed(l) { return zonesOf(l.key).length > 0 || l.paintAny; }

/* ------------------------------------------------------------------ calques */
async function makeLayer(key, opts = {}) {
  const def = LAYERDEF[key] || {};
  const uri = opts.uri || A.layers[key];
  const base = opts.base || pixelsOf(await loadImage(uri));
  return { key: opts.newKey || key, name: opts.name || def.name, note: opts.note || def.note, settings: { ...(opts.settings || def.settings) },
    origin: opts.origin || 'projet', base, pix: base, uri, adj: { sat: 0, contrast: 0, sky: 0 }, paint: null, paintAny: false,
    canvas: null, branch: opts.branch || null, renderS: def.render_s };
}
function applyAdj(l) {
  const { sat, contrast, sky } = l.adj;
  l.canvas = null;
  if (!sat && !contrast && !sky) { l.pix = l.base; return; }
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) {
    let x = v / 255;
    x = x + sky * 0.18 * Math.pow(1 - x, 3);                        // ciel : relève ou abaisse les bas niveaux
    const sc = x * x * (3 - 2 * x); x = x + contrast * 1.2 * (sc - x); // contraste : courbe en S
    lut[v] = Math.round(Math.min(1, Math.max(0, x)) * 255);
  }
  const b = l.base, p = new Uint8ClampedArray(b.length), k = 1 + sat;
  for (let j = 0; j < b.length; j += 4) {
    const r = lut[b[j]], g = lut[b[j + 1]], bl = lut[b[j + 2]];
    const L = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
    p[j] = L + (r - L) * k; p[j + 1] = L + (g - L) * k; p[j + 2] = L + (bl - L) * k; p[j + 3] = 255;
  }
  l.pix = p;
}
function layerCanvas(l) {
  if (!l.canvas) {
    l.canvas = document.createElement('canvas'); l.canvas.width = W; l.canvas.height = H;
    l.canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(l.pix), W, H), 0, 0);
  }
  return l.canvas;
}

/* ------------------------------------------------------------------ composition */
function compose(x0 = 0, y0 = 0, x1 = W, y1 = H, override = null) {
  const zc = override ? { ...S.zone, ...override } : S.zone;
  const P = ZK.map((z) => layer(zc[z]).pix), Wt = ZK.map((z) => wz[z]);
  const [p0, p1, p2, p3, p4] = P, [w0, w1, w2, w3, w4] = Wt;
  const paints = S.layers.filter((l) => l.paintAny);
  const out = compImg.data;
  for (let y = y0; y < y1; y++) {
    for (let x = x0, i = y * W + x0; x < x1; x++, i++) {
      const j = i * 4, a0 = w0[i], a1 = w1[i], a2 = w2[i], a3 = w3[i], a4 = w4[i];
      let r = a0 * p0[j] + a1 * p1[j] + a2 * p2[j] + a3 * p3[j] + a4 * p4[j];
      let g = a0 * p0[j + 1] + a1 * p1[j + 1] + a2 * p2[j + 1] + a3 * p3[j + 1] + a4 * p4[j + 1];
      let b = a0 * p0[j + 2] + a1 * p1[j + 2] + a2 * p2[j + 2] + a3 * p3[j + 2] + a4 * p4[j + 2];
      r /= 255; g /= 255; b /= 255;
      for (const l of paints) {
        const f = l.paint[i] / 255;
        if (f) { const q = l.pix; r += f * (q[j] - r); g += f * (q[j + 1] - g); b += f * (q[j + 2] - b); }
      }
      out[j] = r; out[j + 1] = g; out[j + 2] = b; out[j + 3] = 255;
    }
  }
  compCtx.putImageData(compImg, 0, 0, x0, y0, x1 - x0, y1 - y0);
}
function recompose() { compose(); draw(); renderLayers(); renderGraph(); renderLegend(); }
function renderLegend() {
  const box = $('#legend'); box.innerHTML = '';
  box.classList.toggle('hidden', !S.zonesOn || S.view.kind === 'step');
  for (const z of ZK) {
    const r = el('div', 'lg' + (S.sel.type === 'zone' && S.sel.key === z ? ' on' : ''),
      `<i style="background:${ZCOL[z]}"></i>${ZONE[z].name}<span>${layer(S.zone[z]).name}</span>`);
    r.onclick = () => { S.lastPoint = ZONE[z].rep; select({ type: 'zone', key: z }); };
    r.onmouseenter = () => { S.hoverZone = z; draw(); };
    r.onmouseleave = () => { S.hoverZone = null; draw(); };
    box.appendChild(r);
  }
}

/* ------------------------------------------------------------------ annuler */
function pushUndo(paintKey = null) {
  const l = paintKey && layer(paintKey);
  S.undo.push({ zone: { ...S.zone }, paintKey, paint: l ? (l.paint ? l.paint.slice() : null) : undefined });
  if (S.undo.length > 20) S.undo.shift();
  S.redo = [];
}
function restore(from, to) {
  const s = from.pop(); if (!s) return;
  const l = s.paintKey && layer(s.paintKey);
  to.push({ zone: { ...S.zone }, paintKey: s.paintKey, paint: l ? (l.paint ? l.paint.slice() : null) : undefined });
  S.zone = s.zone;
  if (l) { l.paint = s.paint; l.paintAny = !!(s.paint && s.paint.some((v) => v)); }
  recompose(); renderPanel();
}

/* ------------------------------------------------------------------ zones */
function buildZoneOverlay(z) {
  if (zoneOverlay[z]) return zoneOverlay[z];
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const x = c.getContext('2d'), im = x.createImageData(W, H), d = im.data, id = ZK.indexOf(z);
  const col = ZCOL[z], cr = parseInt(col.slice(1, 3), 16), cg = parseInt(col.slice(3, 5), 16), cb = parseInt(col.slice(5, 7), 16);
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    if (zoneId[i] === id) { d[j] = cr; d[j + 1] = cg; d[j + 2] = cb; d[j + 3] = 34; }
    else { d[j + 3] = 135; }
  }
  x.putImageData(im, 0, 0);
  return (zoneOverlay[z] = c);
}

/* ------------------------------------------------------------------ affichage */
const cv = $('#view'), ctx = cv.getContext('2d');
function resize() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  draw();
}
function source() {
  const v = S.view;
  if (v.kind === 'layer') return { img: layerCanvas(layer(v.key)), field: true, label: `Calque : ${layer(v.key).name}` };
  if (v.kind === 'step') {
    const st = stepDef(v.key), im = st.images[v.img || 0];
    return { img: stepImgs[im.key], field: im.kind === 'field', label: `Étape : ${st.title} <span class="r">· ${im.caption}</span>`, step: st, im };
  }
  const used = new Set([...ZK.map((z) => S.zone[z]), ...S.layers.filter((l) => l.paintAny).map((l) => l.key)]);
  return { img: comp, field: true, label: `Image finale <span class="r">· mélange de ${used.size} calque${used.size > 1 ? 's' : ''}</span>` };
}
function compareSource() {
  if (S.sel.type === 'layer') { const l = layer(S.sel.key); return { img: layerCanvas(l), label: `Calque : ${l.name}` }; }
  if (S.view.kind === 'step') { const s = source(); if (s.field) return { img: s.img, label: s.label.replace(/<[^>]+>/g, '') }; }
  if (S.view.kind === 'layer') { const l = layer(S.view.key); return { img: layerCanvas(l), label: `Calque : ${l.name}` }; }
  return { img: layerCanvas(layer('fidele')), label: 'Calque : Fidèle' };
}
function fit(t, w, h) {
  const r = cv.getBoundingClientRect();
  t.z = Math.min(r.width / w, r.height / h) * 0.98; t.ox = (r.width - w * t.z) / 2; t.oy = (r.height - h * t.z) / 2;
}
function curT() { return source().field ? fieldT : chartT; }
function draw() {
  const dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#08090b'; ctx.fillRect(0, 0, r.width, r.height);
  const src = source(), t = src.field ? fieldT : chartT;
  const w = src.field ? W : src.img.naturalWidth, h = src.field ? H : src.img.naturalHeight;
  ctx.imageSmoothingEnabled = t.z < 1.5; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src.img, t.ox, t.oy, w * t.z, h * t.z);
  $('#view-title').innerHTML = src.label;
  // rideau
  if (S.compare && src.field) {
    const cs = compareSource(), xd = r.width * S.curtain;
    ctx.save(); ctx.beginPath(); ctx.rect(xd, 0, r.width - xd, r.height); ctx.clip();
    ctx.drawImage(cs.img, t.ox, t.oy, W * t.z, H * t.z); ctx.restore();
    ctx.fillStyle = '#fff'; ctx.fillRect(xd - 1, 0, 2, r.height);
    ctx.beginPath(); ctx.arc(xd, r.height / 2, 9, 0, 7); ctx.fill();
    ctx.font = '12px Segoe UI'; ctx.fillStyle = 'rgba(0,0,0,.65)';
    const lt = src.label.replace(/<[^>]+>/g, ''), w2 = ctx.measureText(cs.label).width;
    ctx.fillRect(xd - ctx.measureText(lt).width - 22, r.height - 34, ctx.measureText(lt).width + 14, 22);
    ctx.fillRect(xd + 8, r.height - 34, w2 + 14, 22);
    ctx.fillStyle = '#fff'; ctx.fillText(lt, xd - ctx.measureText(lt).width - 15, r.height - 19); ctx.fillText(cs.label, xd + 15, r.height - 19);
  }
  // zone survolée (sur l'image ou dans la légende)
  if (S.zonesOn && !S.brush && src.field && S.view.kind !== 'step' && S.hoverZone && !S.compare) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(buildZoneOverlay(S.hoverZone), t.ox, t.oy, W * t.z, H * t.z);
  }
  // noms des objets (astrométrie)
  if (S.view.kind === 'step' && S.view.key === 'astrometrie' && (S.view.img || 0) === 0) {
    ctx.font = '600 14px Segoe UI'; ctx.lineWidth = 1.5;
    for (const lb of D.labels) {
      const x = t.ox + lb.x * t.z, y = t.oy + lb.y * t.z;
      ctx.strokeStyle = '#9fd0ff'; ctx.beginPath(); ctx.arc(x, y, 26, 0, 7); ctx.stroke();
      ctx.fillStyle = 'rgba(0,0,0,.6)'; const tw = ctx.measureText(lb.name).width;
      ctx.fillRect(x + 30, y - 12, tw + 12, 22); ctx.fillStyle = '#d8ecff'; ctx.fillText(lb.name, x + 36, y + 4);
    }
  }
  // pinceau
  if (S.brush && S.mouse && src.field) {
    ctx.strokeStyle = $('#brush-erase').checked ? '#ff8080' : '#fff'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(S.mouse.sx, S.mouse.sy, +$('#brush-size').value * t.z, 0, 7); ctx.stroke();
  }
  $('#zoom-label').textContent = `${Math.round(t.z * 100)} %`;
  $('#b-back').classList.toggle('hidden', S.view.kind === 'final');
}
function toImage(e) {
  const r = cv.getBoundingClientRect(), t = curT();
  const sx = e.clientX - r.left, sy = e.clientY - r.top;
  return { sx, sy, x: (sx - t.ox) / t.z, y: (sy - t.oy) / t.z };
}
function zoomAt(sx, sy, f) {
  const t = curT(), nz = Math.min(16, Math.max(0.05, t.z * f));
  t.ox = sx - (sx - t.ox) * nz / t.z; t.oy = sy - (sy - t.oy) * nz / t.z; t.z = nz; draw();
}
function setView(v) {
  const wasField = source().field;
  S.view = v;
  const src = source();
  if (!src.field && (wasField || v.kind === 'step')) fit(chartT, src.img.naturalWidth, src.img.naturalHeight);
  renderStepTabs(); renderLegend(); draw();
}
function renderStepTabs() {
  const box = $('#step-tabs'); box.innerHTML = '';
  if (S.view.kind !== 'step') return;
  const st = stepDef(S.view.key);
  if (st.images.length < 2) return;
  st.images.forEach((im, i) => {
    const b = el('button', 'toggle' + ((S.view.img || 0) === i ? ' on' : ''), im.caption);
    b.onclick = () => setView({ kind: 'step', key: st.key, img: i });
    box.appendChild(b);
  });
}

/* ------------------------------------------------------------------ souris */
let drag = null;
cv.addEventListener('contextmenu', (e) => e.preventDefault());
cv.addEventListener('wheel', (e) => { e.preventDefault(); const p = toImage(e); zoomAt(p.sx, p.sy, e.deltaY < 0 ? 1.15 : 1 / 1.15); }, { passive: false });
cv.addEventListener('dblclick', () => { const s = source(); fit(curT(), s.field ? W : s.img.naturalWidth, s.field ? H : s.img.naturalHeight); draw(); });
cv.addEventListener('mousedown', (e) => {
  const p = toImage(e);
  if (S.compare && Math.abs(p.sx - cv.getBoundingClientRect().width * S.curtain) < 14 && e.button === 0) { drag = { kind: 'curtain' }; return; }
  if (S.brush && e.button === 0 && source().field) {
    const l = brushLayer(); if (!l) return;
    pushUndo(l.key); drag = { kind: 'paint', last: p }; paintAt(l, p, p); return;
  }
  drag = { kind: 'pan', sx: p.sx, sy: p.sy, moved: false, t0: { ...curT() }, button: e.button };
  cv.classList.add('pan');
});
window.addEventListener('mousemove', (e) => {
  const p = toImage(e);
  if (e.target === cv || drag) S.mouse = p; else S.mouse = null;
  if (!drag) { if (e.target === cv) hover(p, e); return; }
  if (drag.kind === 'curtain') { S.curtain = Math.min(0.98, Math.max(0.02, p.sx / cv.getBoundingClientRect().width)); draw(); return; }
  if (drag.kind === 'paint') { paintAt(brushLayer(), drag.last, p); drag.last = p; return; }
  const dx = p.sx - drag.sx, dy = p.sy - drag.sy;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  if (drag.moved) { const t = curT(); t.ox = drag.t0.ox + dx; t.oy = drag.t0.oy + dy; draw(); }
});
window.addEventListener('mouseup', (e) => {
  if (!drag) return;
  const d = drag; drag = null; cv.classList.remove('pan');
  if (d.kind === 'paint') { renderLayers(); renderGraph(); renderPanel(); return; }
  if (d.kind === 'pan' && !d.moved && d.button === 0 && e.target === cv) clickImage(toImage(e));
});
cv.addEventListener('mouseleave', () => { S.hoverZone = null; S.mouse = null; $('#hover-tip').classList.add('hidden'); draw(); });

function zoneAt(p) {
  const x = Math.floor(p.x), y = Math.floor(p.y);
  if (x < 0 || y < 0 || x >= W || y >= H) return null;
  return ZK[zoneId[y * W + x]];
}
function hover(p, e) {
  const tip = $('#hover-tip'), src = source();
  if (!S.zonesOn || S.brush || !src.field || S.view.kind === 'step' || S.compare) {
    tip.classList.add('hidden');
    if (S.hoverZone || S.brush) { S.hoverZone = null; draw(); }
    return;
  }
  const z = zoneAt(p);
  if (z !== S.hoverZone) { S.hoverZone = z; draw(); }
  if (!z) { tip.classList.add('hidden'); return; }
  const r = cv.getBoundingClientRect();
  tip.innerHTML = `<i style="background:${ZCOL[z]}"></i>${ZONE[z].name} · vue : <b>${layer(S.zone[z]).name}</b>`;
  tip.style.left = `${e.clientX - r.left + 16}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
  tip.classList.remove('hidden');
}
function clickImage(p) {
  if (!source().field || S.view.kind === 'step') return;
  const z = zoneAt(p); if (!z) return;
  S.lastPoint = { x: Math.floor(p.x), y: Math.floor(p.y) };
  select({ type: 'zone', key: z });
}

/* ------------------------------------------------------------------ pinceau */
function brushLayer() {
  if (S.sel.type === 'layer') return layer(S.sel.key);
  return null;
}
function paintAt(l, a, b) {
  if (!l) return;
  if (!l.paint) l.paint = new Uint8Array(N);
  const R = +$('#brush-size').value, erase = $('#brush-erase').checked;
  const dist = Math.hypot(b.x - a.x, b.y - a.y), n = Math.max(1, Math.ceil(dist / (R / 4)));
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (let s = 1; s <= n; s++) {
    const cx = a.x + (b.x - a.x) * s / n, cy = a.y + (b.y - a.y) * s / n;
    const xa = Math.max(0, Math.floor(cx - R)), xb = Math.min(W, Math.ceil(cx + R));
    const ya = Math.max(0, Math.floor(cy - R)), yb = Math.min(H, Math.ceil(cy + R));
    for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) {
      const f = Math.min(1, Math.max(0, (R - Math.hypot(x - cx, y - cy)) / (0.5 * R)));
      if (!f) continue;
      const i = y * W + x;
      l.paint[i] = erase ? l.paint[i] * (1 - f * 0.6) : Math.min(255, l.paint[i] + f * 90);
    }
    x0 = Math.min(x0, xa); y0 = Math.min(y0, ya); x1 = Math.max(x1, xb); y1 = Math.max(y1, yb);
  }
  l.paintAny = true;
  if (x1 > x0 && y1 > y0) compose(x0, y0, x1, y1);
  draw();
}

/* ------------------------------------------------------------------ sélection */
function select(sel) {
  S.sel = sel;
  // la recette affichée en bas : celle du dernier calque choisi tant qu'on parcourt ses étapes
  if (sel.type === 'layer') S.recipeLayer = sel.key;
  else if (sel.type !== 'step') S.recipeLayer = null;
  if (sel.type === 'step') { if (stepDef(sel.key).images.length) setView({ kind: 'step', key: sel.key, img: 0 }); }
  else if (sel.type === 'layer' && S.view.kind === 'step') setView({ kind: 'final' });
  else if (sel.type === 'final') setView({ kind: 'final' });
  else if (sel.type === 'zone' && S.view.kind === 'step') setView({ kind: 'final' });
  updateBrushBar();
  renderLayers(); renderPanel(); renderRecipe(); renderGraph(); renderLegend(); draw();
}
function updateBrushBar() {
  const l = brushLayer();
  $('#brush-bar').classList.toggle('hidden', !S.brush);
  $('#brush-target').innerHTML = l ? `Pinceau : <b>${l.name}</b>` : 'Sélectionne d’abord un calque à peindre';
  cv.classList.toggle('brush', S.brush && !!l);
}

/* ------------------------------------------------------------------ calques (liste) */
function thumb(canvas, l) {
  const tw = canvas.width = 92, th = canvas.height = 62, x = canvas.getContext('2d'), im = x.createImageData(tw, th), d = im.data;
  const zs = l ? zonesOf(l.key).map((z) => wz[z]) : null;
  for (let v = 0; v < th; v++) for (let u = 0; u < tw; u++) {
    const i = Math.floor(v * H / th) * W + Math.floor(u * W / tw), j = i * 4, k = (v * tw + u) * 4;
    if (!l) { d[k] = compImg.data[j]; d[k + 1] = compImg.data[j + 1]; d[k + 2] = compImg.data[j + 2]; d[k + 3] = 255; continue; }
    let w = 0; for (const a of zs) w += a[i];
    w = Math.min(255, w + (l.paint ? l.paint[i] : 0)) / 255;
    const f = 0.3 + 0.7 * w;
    d[k] = l.pix[j] * f; d[k + 1] = l.pix[j + 1] * f; d[k + 2] = l.pix[j + 2] * f; d[k + 3] = 255;
  }
  x.putImageData(im, 0, 0);
}
function renderLayers() {
  const ul = $('#layers'); ul.innerHTML = '';
  const fi = el('li', 'final' + (S.sel.type === 'final' ? ' sel' : ''));
  fi.appendChild(el('span', 'eye on', '◉'));
  const fc = el('canvas'); thumb(fc, null); fi.appendChild(fc);
  fi.appendChild(el('div', '', `<div class="nm">Image finale</div><div class="sub">mélange par zones${S.layers.some((l) => l.paintAny) ? ' + pinceau' : ''}</div>`));
  fi.onclick = () => select({ type: 'final' });
  ul.appendChild(fi);
  for (const l of [...S.layers].reverse()) {
    const li = el('li', S.sel.type === 'layer' && S.sel.key === l.key ? 'sel' : '');
    const eye = el('span', 'eye' + (S.view.kind === 'layer' && S.view.key === l.key ? ' on' : ''), '👁');
    eye.title = 'Voir ce calque seul';
    eye.onclick = (e) => { e.stopPropagation(); setView(S.view.kind === 'layer' && S.view.key === l.key ? { kind: 'final' } : { kind: 'layer', key: l.key }); renderLayers(); };
    li.appendChild(eye);
    const c = el('canvas'); thumb(c, l); li.appendChild(c);
    const chips = zonesOf(l.key).map((z) => `<span class="chip" style="box-shadow:inset 3px 0 ${ZCOL[z]}">${ZONE[z].name}</span>`).join('')
      + (l.paintAny ? '<span class="chip">pinceau</span>' : '');
    const tag = l.branch ? `<span class="tag branch">${l.branch.status === 'done' ? 'branche' : 'branche · à calculer'}</span>`
      : `<span class="tag">${l.origin}</span>`;
    li.appendChild(el('div', '', `<div class="nm">${tag}${l.name}</div><div class="sub">${chips || 'pas utilisé dans l’image finale'}</div>`));
    li.onclick = () => select({ type: 'layer', key: l.key });
    ul.appendChild(li);
  }
}

/* ------------------------------------------------------------------ panneau de droite */
let panelFor = '';
function renderPanel() {
  const p = $('#panel'); p.innerHTML = '';
  const id = `${S.sel.type}:${S.sel.key || ''}`;
  if (id !== panelFor) { p.scrollTop = 0; panelFor = id; }
  ({ final: panelFinal, layer: panelLayer, zone: panelZone, step: panelStep })[S.sel.type](p);
}
function panelFinal(p) {
  p.appendChild(el('h2', '', 'Image finale'));
  p.appendChild(el('div', 'note', 'Chaque zone de l’image prend la vue d’un calque ; le pinceau corrige à la main par-dessus. Clique une zone sur l’image pour changer sa vue.'));
  const kv = el('div', 'kv');
  for (const z of ZK) kv.innerHTML += `<span><i class="zone-dot" style="background:${ZCOL[z]}"></i> ${ZONE[z].name} <small>(${fmtNum(ZONE[z].pct)} %)</small></span><span>${layer(S.zone[z]).name}</span>`;
  p.appendChild(kv);
  const painted = S.layers.filter((l) => l.paintAny);
  if (painted.length) p.appendChild(el('div', 'note', `Peint à la main : ${painted.map((l) => l.name).join(', ')}.`));
  p.appendChild(el('h3', '', 'Exporter'));
  p.appendChild(el('div', 'note', 'Dans l’application, l’export refait chaque calque utilisé à pleine résolution (environ 2 s chacun) puis applique le même mélange : TIFF 16 bits, PNG, JPEG, calques séparés, et la pile linéaire en FITS 32 bits.'));
  const b = el('div', 'btns'); const e = el('button', 'primary', 'Exporter (simulé)');
  e.onclick = () => simulateRun('Export de l’image finale', 2 * new Set(ZK.map((z) => S.zone[z])).size, () => {});
  b.appendChild(e); p.appendChild(b);
}
function panelLayer(p) {
  const l = layer(S.sel.key);
  p.appendChild(el('h2', '', l.name));
  p.appendChild(el('div', 'note', l.note || ''));
  if (l.branch) {
    const st = stepDef(l.branch.step), todo = STEPS.slice(STEPS.indexOf(st));
    const t = todo.reduce((a, s) => a + stepSeconds(s), 0);
    const box = el('div', 'impact');
    if (l.branch.status === 'done') box.innerHTML = '<b>Maquette :</b> le calcul n’a pas été fait ; ce calque montre l’image de son calque d’origine.';
    else {
      box.innerHTML = `<b>Branche à calculer</b> depuis « ${st.title} » : ${todo.length} étapes, ${fmtDur(t)} mesurés sur ton PC, environ ${fmtNum(todo.reduce((a, s) => a + stepGb(s), 0))} Go.<br>Les étapes d’avant sont communes : elles ne sont pas refaites.`;
      const b = el('button', 'warn', 'Calculer (simulé)');
      b.style.marginTop = '8px';
      b.onclick = () => { l.branch.status = 'running'; simulateRun(`Branche « ${l.name} »`, t, () => { l.branch.status = 'done'; renderLayers(); renderPanel(); renderRecipe(); renderGraph(); }); };
      box.appendChild(el('br')); box.appendChild(b);
    }
    p.appendChild(box);
  }
  // réglages du rendu
  p.appendChild(el('h3', '', 'Réglages du rendu de ce calque'));
  const diff = Object.entries(l.settings).filter(([k]) => SETTING_FR[k]);
  if (diff.length) {
    const kv = el('div', 'kv');
    for (const [k, v] of diff) kv.innerHTML += `<span>${SETTING_FR[k]}</span><span>${typeof v === 'boolean' ? (v ? 'oui' : 'non') : fmtNum(v, 2)}</span>`;
    p.appendChild(el('div', 'note', 'Différences avec le rendu fidèle (vrais réglages du moteur) :'));
    p.appendChild(kv);
  } else p.appendChild(el('div', 'note', 'Les réglages par défaut du moteur (rendu fidèle).'));
  for (const [k, lab] of [['sat', 'Saturation'], ['contrast', 'Contraste'], ['sky', 'Luminosité du ciel']]) {
    const row = el('div', 'row');
    row.innerHTML = `<span>${lab}</span><input type="range" min="-1" max="1" step="0.05" value="${l.adj[k]}"><span class="v">${fmtNum(l.adj[k], 2)}</span>`;
    const inp = row.querySelector('input');
    let tm = null;
    inp.oninput = () => {
      l.adj[k] = +inp.value; row.querySelector('.v').textContent = fmtNum(l.adj[k], 2);
      clearTimeout(tm); tm = setTimeout(() => { applyAdj(l); recompose(); }, 60);
    };
    p.appendChild(row);
  }
  p.appendChild(el('div', 'approx', 'Curseurs approchés par la maquette ; dans l’application, le vrai moteur refait le rendu de ce calque en 0,3 s.'));
  const b = el('div', 'btns');
  const g = el('button', 'primary', 'Proposer 9 variantes'); g.onclick = () => openGallery(l); b.appendChild(g);
  const v = el('button', '', S.view.kind === 'layer' && S.view.key === l.key ? 'Revoir l’image finale' : 'Voir ce calque seul');
  v.onclick = () => { setView(S.view.kind === 'layer' && S.view.key === l.key ? { kind: 'final' } : { kind: 'layer', key: l.key }); renderPanel(); renderLayers(); };
  b.appendChild(v);
  p.appendChild(b);
  // masque
  p.appendChild(el('h3', '', 'Où ce calque est utilisé'));
  const zc = el('div', 'zonechecks');
  for (const z of ZK) {
    const lab = el('label', '', `<input type="checkbox" ${S.zone[z] === l.key ? 'checked' : ''}><i style="background:${ZCOL[z]}"></i>${ZONE[z].name} <small style="color:var(--dim)">${S.zone[z] === l.key ? '' : '· ' + layer(S.zone[z]).name}</small>`);
    lab.querySelector('input').onchange = (e) => {
      pushUndo();
      S.zone[z] = e.target.checked ? l.key : (l.key === 'fidele' ? 'artistique' : 'fidele');
      recompose(); renderPanel();
    };
    zc.appendChild(lab);
  }
  p.appendChild(zc);
  const pb = el('div', 'btns');
  const br = el('button', S.brush ? 'toggle on' : 'toggle', 'Peindre ce calque'); br.onclick = () => toggleBrush(true);
  pb.appendChild(br);
  if (l.paintAny) {
    const cl = el('button', '', 'Effacer sa peinture');
    cl.onclick = () => { pushUndo(l.key); l.paint = null; l.paintAny = false; recompose(); renderPanel(); };
    pb.appendChild(cl);
  }
  if (!isUsed(l) && S.layers.length > 1 && l.key !== 'fidele') {
    const del = el('button', '', 'Supprimer le calque');
    del.onclick = () => { S.layers = S.layers.filter((x) => x !== l); select({ type: 'final' }); };
    pb.appendChild(del);
  }
  p.appendChild(pb);
}
function cropThumb(canvas, pix, cx, cy) {
  const cw = 150, ch = 100, sc = 2;
  canvas.width = cw * sc; canvas.height = ch * sc;
  const x0 = Math.min(W - cw, Math.max(0, cx - cw / 2)), y0 = Math.min(H - ch, Math.max(0, cy - ch / 2));
  const x = canvas.getContext('2d'), im = x.createImageData(cw, ch);
  for (let v = 0; v < ch; v++) for (let u = 0; u < cw; u++) {
    const j = ((y0 + v) * W + x0 + u) * 4, k = (v * cw + u) * 4;
    im.data[k] = pix[j]; im.data[k + 1] = pix[j + 1]; im.data[k + 2] = pix[j + 2]; im.data[k + 3] = 255;
  }
  const tmp = document.createElement('canvas'); tmp.width = cw; tmp.height = ch; tmp.getContext('2d').putImageData(im, 0, 0);
  x.imageSmoothingEnabled = false; x.drawImage(tmp, 0, 0, cw * sc, ch * sc);
}
function panelZone(p) {
  const z = S.sel.key, Z = ZONE[z], pt = S.lastPoint || { x: W / 2, y: H / 2 };
  p.appendChild(el('h2', '', `<i class="zone-dot" style="background:${ZCOL[z]}"></i> Zone : ${Z.name}`));
  p.appendChild(el('div', 'note', `${Z.note} ${fmtNum(Z.pct)} % de l’image.`));
  p.appendChild(el('div', '', `Vue choisie : <b>${layer(S.zone[z]).name}</b>`));
  p.appendChild(el('div', 'note', 'Détail au point cliqué, agrandi 2 fois. Survole une vue pour l’essayer sur toute l’image, clique pour la garder.'));
  const box = el('div', 'cands');
  const cands = [...S.layers.map((l) => l.key)];
  for (const k of SUGGEST[z] || []) if (!cands.includes(k)) cands.push(k);
  for (const k of cands) {
    const l = layer(k), def = LAYERDEF[k];
    const c = el('div', 'cand' + (S.zone[z] === k ? ' on' : ''));
    const canv = el('canvas'); c.appendChild(canv);
    c.appendChild(el('div', 'cl', `${l ? l.name : def.name}${l ? '' : '<small>vue proposée · deviendra un calque</small>'}`));
    const draw1 = async () => cropThumb(canv, (l || (await ensureDecoded(k))).pix, pt.x, pt.y);
    draw1();
    c.onmouseenter = async () => {
      const ll = l || await ensureDecoded(k);
      if (!layer(k)) { S.layers.push(ll); compose(0, 0, W, H, { [z]: k }); S.layers.pop(); } else compose(0, 0, W, H, { [z]: k });
      if (S.view.kind !== 'final') S.view = { kind: 'final' };
      draw();
    };
    c.onmouseleave = () => { compose(); draw(); };
    c.onclick = async () => {
      pushUndo();
      if (!layer(k)) S.layers.push(decoded[k] || await ensureDecoded(k));
      S.zone[z] = k; recompose(); renderPanel(); renderRecipe();
    };
    box.appendChild(c);
  }
  p.appendChild(box);
  if (z === 'brillante' || z === 'faible' || z === 'fond') {
    const b = el('div', 'btns'), g = el('button', '', 'Plus de choix : 9 variantes du calque Artistique');
    g.onclick = () => openGallery(layer('artistique')); b.appendChild(g); p.appendChild(b);
  }
}
const decoded = {};
async function ensureDecoded(k) {
  if (!decoded[k]) decoded[k] = await makeLayer(k, { origin: 'proposé' });
  return decoded[k];
}
function stepMetrics(key) {
  const m = (n) => stageOf(n).metrics || {};
  const r = [];
  try {
    if (key === 'poses') { const a = m('inventory'); r.push(`${a.n_lights} poses de ${a.exposure_s} s, soit ${fmtNum(a.integration_h)} h ; ${a.camera}, ISO ${a.iso_gain}`, `${a.n_darks} darks, ${a.n_flats} flats, ${a.n_bias} offsets`); }
    if (key === 'calibration') { const a = m('calibrate'); r.push(`Vignetage : le bord reçoit ${Math.round(a.vignetting_min * 100)} % de la lumière du centre`, `${fmtNum(a.bad_pixels_pct)} % de pixels défectueux ; bruit du ciel ${fmtNum(a.sky_noise_adu)} ADU`); }
    if (key === 'alignement') { const a = m('register'); r.push(`${a.aligned} poses alignées, écart moyen ${fmtNum(a.rms_px, 2)} px`, `Taille des étoiles ${fmtNum(a.fwhm_px)} px ; dérive pendant la nuit ${Math.round(a.drift_px)} px`); }
    if (key === 'pile') { const a = m('stack'); r.push(`${a.frames_used} poses gardées (${fmtNum(a.integration_used_h)} h)`, `Grille drizzle ${a.grid[0]} × ${a.grid[1]} ; ${fmtNum(a.rejected_pct, 2)} % de pixels rejetés (satellites, avions…)`); }
    if (key === 'astrometrie') { const a = m('astrometry'), c = m('catalogs'); r.push(`Centre ${fmtNum(a.ra, 3)}°, ${fmtNum(a.dec, 3)}° ; champ ${fmtNum(a.field_deg[0])} × ${fmtNum(a.field_deg[1])}°`, `${fmtNum(a.scale_arcsec, 2)}″ par pixel ; focale trouvée ${Math.round(a.focal_estimate_mm)} mm`, `${fmtNum(c.stars_deep, 0)} étoiles Gaia et ${fmtNum(c.xp_spectra, 0)} spectres`); }
    if (key === 'couleur') { const a = m('color'); r.push(`Réponse du capteur reconstruite sur ${a.stars} étoiles`, `Accord avec les couleurs Gaia : ${fmtNum(a.fit_agreement_pct)} %`); }
    if (key === 'debruitage') { const a = m('denoise'); r.push(`Bruit divisé par ${fmtNum(a.noise_reduction, 2)} (réseau entraîné sur les deux demi-piles)`); }
    if (key === 'etoiles') { const a = m('stars'), b = m('star_net'); r.push(`${fmtNum(a.modelled_stars, 0)} étoiles mesurées et ${fmtNum(a.faint_stars, 0)} étoiles faibles modélisées`, `Retrait appris : ${b.training_steps} pas en ${b.training_minutes} min`); }
    if (key === 'sans_etoiles') { const a = m('starless'); r.push(`${fmtNum(a.filled_pct)} % des pixels comblés sous les étoiles`); }
    if (key === 'fond') { const a = m('background'); r.push(`Fond ajusté sur ${fmtNum(a.sky_fit_blocks_pct)} % du champ, ancré sur le relevé H-alpha`); }
    if (key === 'preparation') r.push('Couches prêtes pour le réglage en temps réel : une par façon de retirer les étoiles (modèle, réseau appris)');
    if (key === 'rendu') r.push('Réglage final en temps réel sur la carte graphique : 0,15 à 0,35 s par aperçu, 2 s en pleine résolution');
  } catch (e) { /* mesure absente */ }
  return r;
}
function panelStep(p) {
  const st = stepDef(S.sel.key), i = STEPS.indexOf(st), down = STEPS.slice(i);
  const t = down.reduce((a, s) => a + stepSeconds(s), 0), gb = down.reduce((a, s) => a + stepGb(s), 0);
  p.appendChild(el('h2', '', `Étape : ${st.title}`));
  const shared = st.key === 'rendu' ? 1 : st.key === 'preparation' ? S.layers.length : S.layers.filter((l) => !l.branch || STEPS.indexOf(stepDef(l.branch.step)) > i).length;
  p.appendChild(el('div', 'note', st.key === 'rendu' ? 'Propre à chaque calque.' : `Commune à ${shared} calque${shared > 1 ? 's' : ''}. Calculée en ${fmtDur(stepSeconds(st))} sur ton PC, ${fmtNum(stepGb(st))} Go sur le disque.`));
  const ul = el('ul', 'metrics'); for (const m of stepMetrics(st.key)) ul.appendChild(el('li', '', m)); p.appendChild(ul);
  if (st.key !== 'rendu') {
    const b = el('div', 'btns'), br = el('button', '', 'Créer une branche ici');
    br.title = 'Un nouveau calque qui refait cette étape et les suivantes avec d’autres réglages';
    br.onclick = () => createBranch(st, null);
    b.appendChild(br);
    if (S.pending.size && [...S.pending].includes(st.key)) {
      const run = el('button', 'warn', `Lancer le recalcul (${fmtDur(t)}, simulé)`);
      run.onclick = () => simulateRun(`Recalcul depuis « ${st.title} »`, t, () => { S.pending.clear(); S.changed = {}; renderRecipe(); renderGraph(); renderPanel(); });
      b.appendChild(run);
    }
    p.appendChild(b);
  }
  // réglages
  const params = [];
  for (const n of st.stages) for (const q of stageOf(n).params) params.push({ ...q, stage: n });
  if (!params.length) return;
  const head = el('h3', '', 'Réglages ');
  const adv = el('button', 'toggle' + (S.showAdvanced ? ' on' : ''), 'avancés'); adv.style.fontSize = '11px';
  adv.onclick = () => { S.showAdvanced = !S.showAdvanced; renderPanel(); };
  head.appendChild(adv); p.appendChild(head);
  p.appendChild(el('div', 'note', 'Les vrais réglages de ton projet, tels que les donne le code actuel (en anglais). « auto » : valeur choisie d’après les données, avec la raison.'));
  const changedHere = Object.keys(S.changed).some((k) => st.stages.includes(k.split('.')[0]));
  if (changedHere) p.appendChild(impactBox(st, t, gb, shared));
  for (const q of params) {
    if (!S.showAdvanced && q.level !== 'basic' && q.origin !== 'auto') continue;
    const id = `${q.stage}.${q.name}`, val = id in S.changed ? S.changed[id] : q.value;
    const d = el('div', 'param' + (id in S.changed ? ' changed' : ''));
    let input;
    if (typeof q.value === 'boolean') input = `<input type="checkbox" ${val ? 'checked' : ''}>`;
    else if (typeof q.value === 'number') input = `<input type="number" value="${val}" step="any">`;
    else input = `<span style="color:var(--dim)">${val ?? '—'}</span>`;
    d.innerHTML = `<div class="pl"><b title="${(q.help || '').replace(/"/g, '&quot;')}">${q.label}${q.unit ? ` <small style="color:var(--dim)">(${q.unit})</small>` : ''}</b>${input}</div>`
      + (q.reason ? `<div class="why auto">${q.reason}</div>` : q.help ? `<div class="why">${q.help.length > 150 ? q.help.slice(0, 150) + '…' : q.help}</div>` : '');
    const inp = d.querySelector('input');
    if (inp) inp.onchange = () => {
      const v = inp.type === 'checkbox' ? inp.checked : +inp.value;
      if (v === q.value) delete S.changed[id]; else S.changed[id] = v;
      renderPanel();
    };
    p.appendChild(d);
  }
}
function impactBox(st, t, gb, shared) {
  const box = el('div', 'impact');
  const ch = Object.entries(S.changed).filter(([k]) => st.stages.includes(k.split('.')[0]));
  const down = STEPS.slice(STEPS.indexOf(st));
  box.innerHTML = `<b>Ce changement relance ${down.length} étapes</b> (${down.map((s) => s.title).join(' → ')}) : `
    + `${fmtDur(t)} d’après les durées mesurées sur ton PC. Cette étape est commune à ${shared} calque${shared > 1 ? 's' : ''}.`;
  const b = el('div', 'btns');
  const all = el('button', 'warn', 'Modifier pour tous les calques');
  all.onclick = () => { for (const s of down) if (s.key !== 'rendu') S.pending.add(s.key); renderRecipe(); renderGraph(); renderPanel(); };
  const br = el('button', 'primary', `Créer une branche (+${fmtNum(gb)} Go)`);
  br.onclick = () => createBranch(st, ch);
  b.appendChild(all); b.appendChild(br); box.appendChild(b);
  return box;
}
let branchN = 0;
async function createBranch(st, changes) {
  const parent = (S.recipeLayer && layer(S.recipeLayer)) || layer('fidele');
  const what = changes && changes.length ? changes.map(([k, v]) => `${k.split('.')[1]} = ${v}`).join(', ') : 'autres réglages';
  branchN += 1;
  const l = await makeLayer(parent.key, { base: parent.base, uri: parent.uri, newKey: `branche${branchN}`, origin: 'branche',
    name: `Branche ${branchN} · ${st.title}`, note: `Refait « ${st.title} » et la suite avec : ${what}. Copie de « ${parent.name} » jusqu’au calcul.`,
    settings: parent.settings, branch: { step: st.key, status: 'todo', parent: parent.key } });
  for (const [k] of changes || []) delete S.changed[k];
  S.layers.push(l);
  select({ type: 'layer', key: l.key });
}

/* ------------------------------------------------------------------ recette */
function stepThumbUri(st, l) {
  if (st.key === 'rendu') return l ? l.uri : A.layers.fidele;
  if (st.key === 'preparation' && l && !usesLearned(l)) return A.steps[st.images[0].key];
  const im = st.images[st.key === 'preparation' ? (st.images.length - 1) : 0];
  return im ? A.steps[im.key] : '';
}
function renderRecipe() {
  const box = $('#recipe'); box.innerHTML = '';
  const l = S.recipeLayer ? layer(S.recipeLayer) : null;
  $('#recipe-for').textContent = l ? `de « ${l.name} »` : 'commune à tous les calques, puis le mélange par zones';
  const bi = l && l.branch ? STEPS.indexOf(stepDef(l.branch.step)) : 99;
  STEPS.forEach((st, i) => {
    if (i) box.appendChild(el('span', 'arrow', '→'));
    const own = i >= bi, todo = (own && l.branch.status !== 'done') || S.pending.has(st.key);
    const sharedStep = st.key !== 'rendu' && !own;
    const c = el('div', `step${sharedStep ? ' shared' : ''}${own ? ' own' : ''}${todo ? ' todo' : ''}${S.sel.type === 'step' && S.sel.key === st.key ? ' sel' : ''}`);
    let sub = st.key === 'rendu' ? (l ? l.name : 'un par calque') : fmtDur(stepSeconds(st));
    if (st.key === 'preparation' && l) sub += usesLearned(l) ? ' · appris' : ' · modèle';
    c.innerHTML = `<div class="th" style="background-image:url('${stepThumbUri(st, l)}')"></div><div class="badge ${todo ? 'todo' : 'ok'}">${todo ? '⟳' : '✓'}</div>`
      + `<div class="st"><b>${st.title}</b><span>${sub}</span></div>`;
    c.onclick = () => (st.key === 'rendu' ? (l ? select({ type: 'layer', key: l.key }) : select({ type: 'final' })) : select({ type: 'step', key: st.key }));
    box.appendChild(c);
  });
  if (!l) {
    box.appendChild(el('span', 'arrow', '→'));
    const m = el('div', 'step' + (S.sel.type === 'final' ? ' sel' : ''));
    const tc = document.createElement('canvas'); tc.width = 236; tc.height = 160;
    tc.getContext('2d').drawImage(comp, 0, 0, 236, 160);
    m.innerHTML = `<div class="th" style="background-image:url('${tc.toDataURL('image/jpeg', 0.8)}')"></div><div class="badge ok">✓</div><div class="st"><b>Mélange par zones</b><span>image finale</span></div>`;
    m.onclick = () => select({ type: 'final' });
    box.appendChild(m);
  }
}

/* ------------------------------------------------------------------ graphe complet */
const SVGNS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function renderGraph() {
  if (!S.graph) return;
  const g = $('#graph-svg'); g.innerHTML = '';
  const NW = 112, NH = 42, DX = 128, trunk = STEPS.filter((s) => s.key !== 'preparation' && s.key !== 'rendu');
  const pos = {};
  const node = (id, x, y, title, sub, cls, onclick) => {
    pos[id] = { x, y };
    const gr = svg('g', { class: `node ${cls || ''}`, transform: `translate(${x},${y})` }, g);
    svg('rect', { width: NW, height: NH, rx: 7 }, gr);
    const t1 = svg('text', { x: 8, y: 17 }, gr); t1.textContent = title.length > 17 ? title.slice(0, 16) + '…' : title;
    const t2 = svg('text', { x: 8, y: 33, class: 's' }, gr); t2.textContent = sub;
    gr.onclick = onclick;
    const tt = svg('title', {}, gr); tt.textContent = title;
  };
  const edge = (a, b, cls, label) => {
    const p = pos[a], q = pos[b], x1 = p.x + NW, y1 = p.y + NH / 2, x2 = q.x, y2 = q.y + NH / 2, mx = (x1 + x2) / 2;
    svg('path', { class: `e ${cls || ''}`, d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}` }, g);
    if (label) { const t = svg('text', { class: 'el', x: x1 + 6, y: y1 - 5 }, g); t.textContent = label; }
  };
  const selStep = (k) => () => select({ type: 'step', key: k });
  const isSel = (k) => S.sel.type === 'step' && S.sel.key === k ? 'sel' : '';
  trunk.forEach((st, i) => node(st.key, 10 + i * DX, 20, st.title, fmtDur(stepSeconds(st)), `${isSel(st.key)} ${S.pending.has(st.key) ? 'todo' : ''}`, selStep(st.key)));
  for (let i = 1; i < trunk.length; i++) edge(trunk[i - 1].key, trunk[i].key);
  const xp = 10 + trunk.length * DX;
  node('prepA', xp, 20, 'Préparation · appris', fmtDur(stageOf('render_prep_learned').seconds), `${isSel('preparation')} ${S.pending.has('preparation') ? 'todo' : ''}`, selStep('preparation'));
  node('prepM', xp, 84, 'Préparation · modèle', fmtDur(stageOf('render_prep').seconds), `${isSel('preparation')} ${S.pending.has('preparation') ? 'todo' : ''}`, selStep('preparation'));
  edge(trunk[trunk.length - 1].key, 'prepA'); edge(trunk[trunk.length - 1].key, 'prepM');
  const xr = xp + DX + 40, base = S.layers.filter((l) => !l.branch), branches = S.layers.filter((l) => l.branch);
  base.forEach((l, j) => node(`L_${l.key}`, xr, 20 + j * 56, l.name, 'rendu 0,3 s', S.sel.type === 'layer' && S.sel.key === l.key ? 'sel' : '', () => select({ type: 'layer', key: l.key })));
  base.forEach((l) => edge(usesLearned(l) ? 'prepA' : 'prepM', `L_${l.key}`));
  // branches : une rangée par branche sous le tronc, leur rendu à la suite des autres calques
  branches.forEach((l, j) => {
    const st = stepDef(l.branch.step), i = STEPS.indexOf(st), x = 10 + Math.min(i, trunk.length) * DX;
    const y = 90 + j * 56, yl = 20 + (base.length + j) * 56, todo = l.branch.status !== 'done' ? 'todo' : '';
    node(`B_${l.key}`, x, y, `${st.title} → … (branche)`, l.branch.status === 'done' ? 'calculée' : `à calculer · ${fmtDur(STEPS.slice(i).reduce((a, s) => a + stepSeconds(s), 0))}`, todo, () => select({ type: 'layer', key: l.key }));
    const prev = i === 0 ? null : (i >= trunk.length ? (usesLearned(l) ? 'prepA' : 'prepM') : trunk[i - 1].key);
    if (prev) edge(prev, `B_${l.key}`);
    node(`L_${l.key}`, xr, yl, l.name, 'rendu 0,3 s', `${S.sel.type === 'layer' && S.sel.key === l.key ? 'sel' : ''} ${todo}`, () => select({ type: 'layer', key: l.key }));
    edge(`B_${l.key}`, `L_${l.key}`);
  });
  const nl = base.length + branches.length, ym = 20 + ((Math.max(nl, 1) - 1) * 56) / 2;
  const xm = xr + DX + 150;
  node('mix', xm, ym, 'Mélange par zones', 'masques + pinceau', S.sel.type === 'final' ? 'sel' : '', () => select({ type: 'final' }));
  for (const l of S.layers) {
    const zs = zonesOf(l.key).map((z) => ZONE[z].name.split(' ')[0].toLowerCase());
    if (l.paintAny) zs.push('pinceau');
    if (zs.length) edge(`L_${l.key}`, 'mix', 'mix', zs.join(', '));
  }
  node('final', xm + DX + 10, ym, 'Image finale', 'export', S.sel.type === 'final' ? 'sel' : '', () => select({ type: 'final' }));
  edge('mix', 'final');
  // à la largeur du panneau (réduit si besoin, jamais agrandi)
  const gw = xm + DX + 10 + NW + 20, gh = Math.max(90 + branches.length * 56, 20 + nl * 56) + 20;
  const avail = $('#graph').clientWidth - 24, k = Math.min(1, avail / gw);
  g.setAttribute('viewBox', `0 0 ${gw} ${gh}`);
  g.setAttribute('width', gw * k); g.setAttribute('height', gh * k);
}

/* ------------------------------------------------------------------ galerie */
let gSel = null, gFor = null;
async function openGallery(l) {
  gFor = l; gSel = null;
  $('#g-layer').textContent = `« ${l.name} »`;
  $('#g-info').innerHTML = l.key === 'artistique' ? 'Clique la variante que tu préfères. Chaque image est un vrai rendu du moteur (0,3 s).'
    : 'Dans la maquette, les variantes sont pré-calculées autour du calque « Artistique » ; dans l’application, autour du calque choisi (0,3 s par image).';
  $('#g-replace').disabled = $('#g-add').disabled = true;
  $('#g-replace').textContent = `Remplacer « ${l.name} »`;
  $('#gallery').classList.remove('hidden');
  for (const k of Object.keys(A.gallery)) if (!galleryImgs[k]) galleryImgs[k] = await loadImage(A.gallery[k]);
  drawGallery();
}
function drawGallery() {
  const grid = $('#g-grid'); grid.innerHTML = '';
  const detail = $('#g-detail').classList.contains('on'), pt = S.lastPoint || { x: W * 0.37, y: H * 0.35 };
  const keys = Object.keys(D.gallery).sort((a, b) => D.gallery[a].row - D.gallery[b].row || D.gallery[a].col - D.gallery[b].col);
  // trois rangées qui tiennent dans la fenêtre
  const cellH = (window.innerHeight * 0.92 - 200) / 3 - 8;
  grid.style.width = `${Math.min(window.innerWidth * 0.94 - 40, 3 * cellH * (detail ? 1.5 : W / H) + 16)}px`;
  for (const k of keys) {
    const gi = D.gallery[k], c = el('div', `gc${gSel === k ? ' on' : ''}${gi.row === 1 && gi.col === 1 ? ' cur' : ''}`);
    const cw = 480, ch = Math.round(480 * (detail ? 2 / 3 : H / W)), canv = el('canvas'); canv.width = cw; canv.height = ch;
    const x = canv.getContext('2d'), im = galleryImgs[k];
    x.imageSmoothingQuality = 'high';
    if (detail) { const sw = 360, sh = 240, sx = Math.min(W - sw, Math.max(0, pt.x - sw / 2)), sy = Math.min(H - sh, Math.max(0, pt.y - sh / 2)); x.drawImage(im, sx, sy, sw, sh, 0, 0, cw, ch); }
    else x.drawImage(im, 0, 0, cw, ch);
    c.appendChild(canv);
    c.appendChild(el('span', 'gt', `contraste ${gi.contraste} · saturation ${gi.saturation}`));
    c.title = `contraste ${gi.contraste}, saturation ${gi.saturation} · rendu en ${fmtNum(gi.render_s, 2)} s`;
    c.onclick = () => { gSel = k; $('#g-replace').disabled = $('#g-add').disabled = false; $('#g-info').innerHTML = `Choisi : contraste <b>${gi.contraste}</b>, saturation <b>${gi.saturation}</b>. Dans l’application, un 2<sup>e</sup> tour proposerait 9 variantes plus proches de ce choix.`; drawGallery(); };
    grid.appendChild(c);
  }
}
$('#g-whole').onclick = () => { $('#g-whole').classList.add('on'); $('#g-detail').classList.remove('on'); drawGallery(); };
$('#g-detail').onclick = () => { $('#g-detail').classList.add('on'); $('#g-whole').classList.remove('on'); drawGallery(); };
$('#g-replace').onclick = async () => {
  const gi = D.gallery[gSel], l = gFor;
  l.base = pixelsOf(galleryImgs[gSel]); l.uri = A.gallery[gSel]; l.settings = gi.settings;
  l.note = `Variante choisie dans la galerie : contraste ${gi.contraste}, saturation ${gi.saturation}.`;
  applyAdj(l); $('#gallery').classList.add('hidden'); recompose(); renderPanel(); renderRecipe();
};
let galN = 0;
$('#g-add').onclick = async () => {
  const gi = D.gallery[gSel]; galN += 1;
  const l = await makeLayer('artistique', { base: pixelsOf(galleryImgs[gSel]), uri: A.gallery[gSel], newKey: `galerie${galN}`, origin: 'galerie',
    name: `Galerie · contraste ${gi.contraste}, saturation ${gi.saturation}`, note: 'Choisi dans la galerie. Assigne-le à une zone (clic sur l’image) ou peins-le où tu le veux.', settings: gi.settings });
  S.layers.push(l); $('#gallery').classList.add('hidden'); select({ type: 'layer', key: l.key });
};

/* ------------------------------------------------------------------ calcul simulé */
function simulateRun(label, seconds, done) {
  if (S.run) return;
  const t0 = performance.now(), dur = 3500;
  S.run = true;
  const tick = () => {
    const f = Math.min(1, (performance.now() - t0) / dur);
    $('#run-status').innerHTML = `${label} <span class="bar"><i style="width:${f * 100}%"></i></span> simulé · sur ton PC : ${fmtDur(seconds)} au total, reste ${fmtDur(seconds * (1 - f))}`;
    if (f < 1) requestAnimationFrame(tick);
    else { S.run = null; $('#run-status').textContent = `${label} : terminé (simulé)`; setTimeout(() => { if (!S.run) $('#run-status').textContent = ''; }, 4000); done(); }
  };
  tick();
}

/* ------------------------------------------------------------------ boutons, clavier */
function toggleBrush(on) {
  S.brush = on === undefined ? !S.brush : on;
  $('#b-brush').classList.toggle('on', S.brush);
  if (S.brush && S.view.kind !== 'final') setView({ kind: 'final' });
  updateBrushBar(); renderPanel(); draw();
}
$('#b-zones').onclick = () => { S.zonesOn = !S.zonesOn; $('#b-zones').classList.toggle('on', S.zonesOn); S.hoverZone = null; draw(); };
$('#b-brush').onclick = () => toggleBrush();
$('#b-compare').onclick = () => { S.compare = !S.compare; $('#b-compare').classList.toggle('on', S.compare); draw(); };
$('#b-fit').onclick = () => { const s = source(); fit(curT(), s.field ? W : s.img.naturalWidth, s.field ? H : s.img.naturalHeight); draw(); };
$('#b-100').onclick = () => { const r = cv.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1 / curT().z); };
$('#b-back').onclick = () => select({ type: 'final' });
$('#b-undo').onclick = () => restore(S.undo, S.redo);
$('#b-redo').onclick = () => restore(S.redo, S.undo);
$('#b-help').onclick = () => $('#help').classList.remove('hidden');
$('#brush-clear').onclick = () => { const l = brushLayer(); if (l) { pushUndo(l.key); l.paint = null; l.paintAny = false; recompose(); renderPanel(); } };
$('#brush-size').oninput = () => draw();
document.querySelectorAll('[data-close]').forEach((b) => { b.onclick = () => $('#' + b.dataset.close).classList.add('hidden'); });
document.querySelectorAll('.modal').forEach((m) => m.addEventListener('mousedown', (e) => { if (e.target === m) m.classList.add('hidden'); }));
$('#t-recipe').onclick = () => { S.graph = false; $('#t-recipe').classList.add('on'); $('#t-graph').classList.remove('on'); $('#recipe').classList.remove('hidden'); $('#graph').classList.add('hidden'); resize(); };
$('#t-graph').onclick = () => { S.graph = true; $('#t-graph').classList.add('on'); $('#t-recipe').classList.remove('on'); $('#graph').classList.remove('hidden'); $('#recipe').classList.add('hidden'); renderGraph(); resize(); };
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); restore(S.undo, S.redo); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); restore(S.redo, S.undo); return; }
  const k = e.key.toLowerCase();
  if (k === 'escape') { document.querySelectorAll('.modal').forEach((m) => m.classList.add('hidden')); if (S.brush) toggleBrush(false); else select({ type: 'final' }); }
  else if (k === 'z') $('#b-zones').click();
  else if (k === 'b') $('#b-brush').click();
  else if (k === 'c') $('#b-compare').click();
  else if (k === 'f') $('#b-fit').click();
  else if (k === '1') $('#b-100').click();
});
window.addEventListener('resize', resize);

/* ------------------------------------------------------------------ démarrage */
(async function init() {
  $('#project-name').textContent = `${D.project} · Heart & Soul · 721 poses de 15 s`;
  const [ma, mb] = await Promise.all([loadImage(A.masks.a), loadImage(A.masks.b)]);
  const pa = pixelsOf(ma), pb = pixelsOf(mb);
  for (const z of ZK) wz[z] = new Uint8Array(N);
  zoneId = new Uint8Array(N);
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    let s = pa[j], h = pa[j + 1], b = pa[j + 2], f = pb[j];
    const t = s + h + b + f;
    if (t > 255) { const k = 255 / t; s = Math.round(s * k); h = Math.round(h * k); b = Math.round(b * k); f = 255 - s - h - b; }
    const k0 = 255 - s - h - b - f;
    wz.etoiles[i] = s; wz.halos[i] = h; wz.brillante[i] = b; wz.faible[i] = f; wz.fond[i] = k0;
    let m = s, id = 0; if (h > m) { m = h; id = 1; } if (b > m) { m = b; id = 2; } if (f > m) { m = f; id = 3; } if (k0 > m) id = 4;
    zoneId[i] = id;
  }
  // un point représentatif par zone (pour la légende) : le plus fort de la zone, le plus près du centre
  for (const [id, z] of ZK.entries()) {
    let best = null, bw = -1, bd = Infinity;
    for (let y = 40; y < H - 40; y += 6) for (let x = 60; x < W - 60; x += 6) {
      const i = y * W + x; if (zoneId[i] !== id) continue;
      const w = wz[z][i], d = Math.hypot(x - W / 2, y - H / 2);
      if (w > bw || (w === bw && d < bd)) { bw = w; bd = d; best = { x, y }; }
    }
    ZONE[z].rep = best || { x: W / 2, y: H / 2 };
  }
  for (const k of ['fidele', 'artistique', 'modele']) S.layers.push(await makeLayer(k));
  for (const st of D.steps) for (const im of st.images) stepImgs[im.key] = await loadImage(A.steps[im.key]);
  comp = document.createElement('canvas'); comp.width = W; comp.height = H;
  compCtx = comp.getContext('2d'); compImg = compCtx.createImageData(W, H);
  compose();
  $('#loading').remove();
  resize(); fit(fieldT, W, H);
  select({ type: 'final' });
  let seen = false;
  try { seen = !!localStorage.getItem('alt1-help'); localStorage.setItem('alt1-help', '1'); } catch (e) { /* fichier local */ }
  if (!seen) $('#help').classList.remove('hidden');
})();
