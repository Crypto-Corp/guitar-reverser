import "./style.css";
import { chordSVG } from "./chords";
import { Fretboard3D } from "./fretboard3d";
import { MediaEngine } from "./media";
import { TabView } from "./tabview";
import type { Chord, Mode, Song, TabEvent, Voicing } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const store = {
  get: (k: string, d: string) => { try { return localStorage.getItem("gr:" + k) ?? d; } catch { return d; } },
  set: (k: string, v: string) => { try { localStorage.setItem("gr:" + k, v); } catch { /* ignore */ } },
};

const engine = new MediaEngine();
const board = new Fretboard3D($("board"));
const tab = new TabView($("tab"));
const video = $("video") as HTMLVideoElement;
const handsCanvas = $("hands") as HTMLCanvasElement;

let song: Song | null = null;
let base = "";
let mode: Mode = (store.get("mode", "chords") as Mode);
let events: TabEvent[] = [];
let sectionTimes: { label: string; start: number; end: number }[] = [];

tab.onSeek = (t) => engine.seek(t);

async function init() {
  let idx: { songs: { slug: string; title: string }[] } = { songs: [] };
  try {
    idx = await (await fetch("songs/index.json", { cache: "no-store" })).json();
  } catch { /* none */ }
  if (!idx.songs?.length) {
    $("empty").hidden = false;
    return;
  }
  const sel = $("song-select") as HTMLSelectElement;
  sel.innerHTML = idx.songs.map((s) => `<option value="${s.slug}">${escapeHtml(s.title)}</option>`).join("");
  const want = new URLSearchParams(location.search).get("song") || store.get("song", idx.songs[0].slug);
  sel.value = idx.songs.some((s) => s.slug === want) ? want : idx.songs[0].slug;
  sel.onchange = () => loadSong(sel.value);
  await loadSong(sel.value);
}

async function loadSong(slug: string) {
  store.set("song", slug);
  base = `songs/${slug}`;
  song = await (await fetch(`${base}/song.json`, { cache: "no-store" })).json();
  const s = song!;
  await engine.load(base, s.media, s.media.video ? video : null);
  $("video-wrap").style.display = s.media.video ? "" : "none";
  engine.beats = s.beats;
  engine.downbeatOffset = s.downbeatOffset;
  engine.meter = s.meta.meter;
  board.setCapo(s.meta.capo);
  ($("dl-tab") as HTMLAnchorElement).href = `${base}/tab.txt`;
  ($("dl-mid") as HTMLAnchorElement).href = `${base}/transcription.mid`;

  const beatTime = (b: number) => s.beats[Math.max(0, Math.min(s.beats.length - 1, b))] ?? 0;
  sectionTimes = s.sections.map((sec) => ({
    label: sec.label,
    start: beatTime(s.downbeatOffset + sec.startBar * s.meta.meter),
    end: beatTime(s.downbeatOffset + sec.endBar * s.meta.meter),
  }));
  renderMeta(s);
  renderChordStrip(s);
  renderTimeline(s);
  renderInfo(s);
  setMode(mode);
  if (engine.time < s.meta.musicStart - 2) engine.seek(Math.max(0, s.meta.musicStart - 1));
}

function setMode(m: Mode) {
  mode = m;
  store.set("mode", m);
  document.querySelectorAll<HTMLButtonElement>("#mode-seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  if (!song) return;
  events = m === "full" ? song.events : song.easyEvents;
  engine.notes = (m === "chords" ? chordSheetNotes(song) : events.flatMap((e) => e.notes.map((n) => ({ t: e.t, end: n.end, p: n.p, amp: n.amp }))));
  renderPattern(song);
}

/** For the synth in chord mode: sound the voicing on every strum of the sheet. */
function chordSheetNotes(s: Song) {
  const out: { t: number; end: number; p: number; amp: number }[] = [];
  s.chordSheet.forEach((st, i) => {
    const next = s.chordSheet[i + 1]?.t ?? st.t + 0.5;
    const order = st.dir === "U" ? [5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5];
    let k = 0;
    for (const string of order) {
      const f = st.frets[string];
      if (f < 0) continue;
      const p = s.meta.tuning[string] + s.meta.capo + f;
      out.push({ t: st.t + k++ * 0.012, end: Math.min(next, st.t + 1.2), p, amp: st.dir === "U" ? 0.35 : 0.5 });
    }
  });
  return out;
}

function renderMeta(s: Song) {
  const m = s.meta;
  const pills = [
    `<span class="pill">Key <b>${m.key}</b></span>`,
    `<span class="pill"><b>${m.tempo}</b> BPM · ${m.meter}/4</span>`,
    `<span class="pill ${m.capo ? "hot" : ""}">Capo <b>${m.capo || "none"}</b></span>`,
    `<span class="pill">${m.style === "strum" ? "Strumming" : "Fingerstyle"}</span>`,
    `<span class="pill">Tuning <b>${m.tuningOffsetCents > 0 ? "+" : ""}${m.tuningOffsetCents}¢</b></span>`,
  ];
  if (m.easyCapo !== m.capo) pills.push(`<span class="pill hot" title="Play open shapes with a capo">Easier: capo ${m.easyCapo}</span>`);
  $("meta").innerHTML = pills.join("");
}

function uniqueChords(s: Song) {
  const seen = new Map<string, Chord>();
  for (const c of s.chords) if (c.name !== "N" && c.voicing && !seen.has(c.name)) seen.set(c.name, c);
  return [...seen.values()];
}

function renderChordStrip(s: Song) {
  const strip = $("chord-strip");
  strip.innerHTML = uniqueChords(s).map((c) => `<div data-name="${c.name}" title="Jump to ${c.name}">${chordSVG(c.shape || c.name, c.voicing, { size: 72 })}</div>`).join("");
  strip.querySelectorAll<HTMLDivElement>("div[data-name]").forEach((d) => {
    d.onclick = () => {
      const name = d.dataset.name!;
      const t = engine.time;
      const next = s.chords.find((c) => c.name === name && c.start > t + 0.1) || s.chords.find((c) => c.name === name);
      if (next) engine.seek(Math.max(0, next.start - 0.5));
    };
  });
}

function renderTimeline(s: Song) {
  const tl = $("timeline");
  const dur = s.meta.duration || 1;
  const pct = (t: number) => `${(100 * t) / dur}%`;
  const palette = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171", "#2dd4bf", "#fb923c"];
  const names = [...new Set(s.chords.filter((c) => c.name !== "N").map((c) => c.name))];
  let html = "";
  for (const c of s.chords) {
    if (c.name === "N") continue;
    html += `<div class="blk" style="left:${pct(c.start)};width:${pct(c.end - c.start)};background:${palette[names.indexOf(c.name) % palette.length]}">${c.name}</div>`;
  }
  for (const sec of sectionTimes) html += `<div class="sec" style="left:${pct(sec.start)}">${sec.label}</div>`;
  for (const t of s.meta.vision?.guitarTimeline ?? []) html += `<div class="vis" style="left:${pct(t)};width:${pct(1)}" title="guitar visible in video"></div>`;
  html += `<div class="loopz" id="loopz" hidden></div><div class="head" id="head"></div>`;
  tl.innerHTML = html;
  tl.onclick = (e) => {
    const r = tl.getBoundingClientRect();
    engine.seek(((e.clientX - r.left) / r.width) * dur);
  };
}

function renderInfo(s: Song) {
  const m = s.meta;
  const v = m.vision || {};
  const ns = m.noteStats || {};
  $("info").innerHTML = `
    <div class="card-title">How this was reversed</div>
    <b>${s.events.length}</b> note events from <b>${ns.raw ?? "?"}</b> raw detections
    (${(ns.raw ?? 0) - (ns.final ?? 0)} removed as ghosts, vocal bleed or noise).<br/>
    ${v.guitarVisibleSeconds ? `Guitar visible <b>${v.guitarVisibleSeconds}s</b> · hands tracked in <b>${v.handFrames}</b> frames ·
    fret position read in <b>${v.positionFrames}</b> frames (green marks on the timeline).<br/>` : "Audio only.<br/>"}
    Pipeline: ${m.pipeline.filter(Boolean).join(" → ")}.<br/>
    <span title="Shortcuts">⌨ space play · ←/→ seek · L loop · 1/2/3 modes · −/+ speed</span>`;
}

function renderPattern(s: Song) {
  if (mode === "chords" && s.pattern.style !== "strum") {
    // Fingerpicked original: chord mode teaches an easy strum over the same chords.
    const cells = ["D", "-", "D", "U", "-", "U", "D", "U"].slice(0, s.meta.meter * 2);
    $("pattern-title").textContent = "Beginner strum (simplified)";
    $("pattern").style.gridTemplateColumns = `repeat(${cells.length}, 1fr)`;
    $("pattern").innerHTML = cells.map((c, i) => `<div class="cell ${c === "-" ? "" : c}" data-i="${i}">${c === "D" ? "↓" : c === "U" ? "↑" : "·"}<small>${i % 2 ? "&" : i / 2 + 1}</small></div>`).join("");
    $("pattern-hint").innerHTML = `The original is <b>fingerpicked</b>. To start, strum these chords with this classic pattern. When you're ready, switch to <b>Easy tab</b> to learn the real picking.`;
    return;
  }
  const p = s.pattern;
  const fingerstyle = p.style !== "strum";
  $("pattern-title").textContent = fingerstyle ? "Picking pattern (most common bar)" : "Strum pattern (most common bar)";
  const cells: string[] = p.cells.length ? p.cells : Array<string>(s.meta.meter * 2).fill("-");
  const counts = Array.from({ length: cells.length }, (_, i) => (i % 2 === 0 ? String(i / 2 + 1) : "&"));
  $("pattern").style.gridTemplateColumns = `repeat(${cells.length}, 1fr)`;
  const PK: Record<string, string> = { p: "#a78bfa", i: "#60a5fa", m: "#34d399", a: "#fbbf24" };
  $("pattern").innerHTML = cells.map((c, i) => {
    if (c === "D" || c === "U") return `<div class="cell ${c}" data-i="${i}">${c === "D" ? "↓" : "↑"}<small>${counts[i]}</small></div>`;
    if (c === "-") return `<div class="cell" data-i="${i}">·<small>${counts[i]}</small></div>`;
    // fingerstyle token like "p5" or "p4.m2": string number big, finger letter coloured
    const parts = c.split(".").map((t) => `<span style="color:${PK[t[0]] ?? "inherit"}">${t.slice(1)}<sup>${t[0]}</sup></span>`);
    return `<div class="cell" data-i="${i}" title="string ${c.split(".").map((t) => t.slice(1)).join(" + ")}">${parts.join("")}<small>${counts[i]}</small></div>`;
  }).join("");
  $("pattern-hint").innerHTML = fingerstyle
    ? `Numbers are strings (1 = high e … 6 = low E). Letters are picking fingers: <b style="color:#a78bfa">p</b> thumb (bass strings), <b style="color:#60a5fa">i</b> index, <b style="color:#34d399">m</b> middle, <b style="color:#fbbf24">a</b> ring. Close to this in ${p.count}/${p.of ?? "?"} bars.`
    : `<b style="color:#22d3ee">↓</b> down-strum, <b style="color:#fb923c">↑</b> up-strum. Keep your hand moving down on the beats and up on the "&" even when you skip a strum. Found in ${p.count}/${p.of ?? "?"} bars.`;
}

function chordAt(t: number): { cur: Chord | null; next: Chord | null } {
  if (!song) return { cur: null, next: null };
  const cs = song.chords;
  let lo = 0, hi = cs.length - 1, idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cs[mid].start <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1;
  }
  const cur = idx >= 0 && cs[idx].end > t && cs[idx].name !== "N" ? cs[idx] : null;
  let next: Chord | null = null;
  for (let k = Math.max(0, idx + 1); k < cs.length; k++) if (cs[k].name !== "N" && cs[k].name !== cur?.name) { next = cs[k]; break; }
  return { cur, next };
}

let lastCur = "", lastNext = "";
function updatePanels(now: number) {
  if (!song) return;
  const { cur, next } = chordAt(now);
  const curKey = cur?.name ?? "-", nextKey = next?.name ?? "-";
  if (curKey !== lastCur) {
    $("chord-now").innerHTML = cur ? chordSVG(cur.shape || cur.name, cur.voicing, { size: 120, colored: true }) : `<div style="width:120px;text-align:center;color:var(--muted)">—</div>`;
    $("now-chord").innerHTML = cur ? `${cur.shape || cur.name}${cur.shape && cur.shape !== cur.name ? `<small>sounds as ${cur.name} (capo ${song.meta.capo})</small>` : ""}` : "";
    document.querySelectorAll<HTMLDivElement>("#chord-strip > div").forEach((d) => d.classList.toggle("on", d.dataset.name === cur?.name));
    lastCur = curKey;
  }
  if (nextKey !== lastNext) {
    $("chord-next").innerHTML = next ? chordSVG(next.shape || next.name, next.voicing, { size: 90, colored: true }) : "";
    lastNext = nextKey;
  }
  if (next) {
    const dt = next.start - now;
    const span = Math.max(0.5, next.start - (cur?.start ?? now));
    $("next-in").innerHTML = `change in ${dt.toFixed(1)}s<div class="next-bar"><div style="width:${Math.max(0, 100 - (100 * dt) / span)}%"></div></div>`;
  } else $("next-in").textContent = "";
  // pattern cursor
  const b = beatIndex(now);
  const cellIdx = Math.floor(((b - song.downbeatOffset) % song.meta.meter + song.meta.meter) % song.meta.meter * 2);
  document.querySelectorAll<HTMLDivElement>("#pattern .cell").forEach((c, i) => c.classList.toggle("on", engine.playing && i === cellIdx));
  // timeline head + clock
  const dur = song.meta.duration || 1;
  const head = document.getElementById("head");
  if (head) head.style.left = `${(100 * now) / dur}%`;
  const lz = document.getElementById("loopz");
  if (lz) {
    lz.hidden = !engine.loop;
    if (engine.loop) { lz.style.left = `${(100 * engine.loop[0]) / dur}%`; lz.style.width = `${(100 * (engine.loop[1] - engine.loop[0])) / dur}%`; }
  }
  $("clock").textContent = `${fmt(now)} / ${fmt(engine.duration || dur)}`;
}

function beatIndex(t: number) {
  const bs = song!.beats;
  if (!bs.length) return 0;
  if (t <= bs[0]) return (t - bs[0]) / ((bs[1] ?? bs[0] + 0.8) - bs[0]);
  for (let i = 0; i < bs.length - 1; i++) if (t < bs[i + 1]) return i + (t - bs[i]) / (bs[i + 1] - bs[i]);
  return bs.length - 1;
}

function drawHands(now: number) {
  const s = song;
  const ctx = handsCanvas.getContext("2d")!;
  const W = (handsCanvas.width = handsCanvas.clientWidth * 2), H = (handsCanvas.height = handsCanvas.clientHeight * 2);
  ctx.clearRect(0, 0, W, H);
  if (!s || !($("show-hands") as HTMLInputElement).checked || !s.hands.length) return;
  let lo = 0, hi = s.hands.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (s.hands[mid].t < now) lo = mid + 1; else hi = mid; }
  const fr = s.hands[lo];
  if (!fr || Math.abs(fr.t - now) > 0.15) return;
  const bones = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
  if (fr.ax) {
    ctx.strokeStyle = "rgba(255,255,255,.55)";
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(fr.ax[0] * W, fr.ax[1] * H);
    ctx.lineTo(fr.ax[2] * W, fr.ax[3] * H);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  fr.h.forEach((pts, i) => {
    const col = i === fr.f ? "#facc15" : i === fr.p ? "#e879f9" : "rgba(148,163,184,.6)";
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    ctx.lineWidth = 3;
    for (const [a, b] of bones) {
      ctx.beginPath();
      ctx.moveTo(pts[a * 2] * W, pts[a * 2 + 1] * H);
      ctx.lineTo(pts[b * 2] * W, pts[b * 2 + 1] * H);
      ctx.stroke();
    }
    for (let k = 0; k < 21; k++) {
      ctx.beginPath();
      ctx.arc(pts[k * 2] * W, pts[k * 2 + 1] * H, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (i === fr.f && fr.pos !== null) {
      ctx.font = "700 26px Inter, sans-serif";
      ctx.fillStyle = "#facc15";
      ctx.fillText(`fret ≈ ${fr.pos}`, pts[16] * W - 40, pts[17] * H - 30);
    }
  });
}

function currentSection(now: number) {
  return sectionTimes.find((s) => s.start <= now && now < s.end) || null;
}

function toggleLoop() {
  if (engine.loop) engine.loop = null;
  else {
    const sec = currentSection(engine.time);
    if (sec) engine.loop = [sec.start, sec.end];
    else {
      const bs = song?.beats ?? [];
      const b = Math.floor(beatIndex(engine.time));
      const m = song?.meta.meter ?? 4;
      const start = bs[Math.max(0, b - ((b - (song?.downbeatOffset ?? 0)) % m))] ?? engine.time;
      const end = bs[Math.min(bs.length - 1, b + 2 * m)] ?? engine.time + 8;
      engine.loop = [start, end];
    }
  }
  $("loop").classList.toggle("on", !!engine.loop);
}

async function countIn() {
  if (!song) return;
  const beatLen = 60 / song.meta.tempo / engine.rate;
  const overlay = document.createElement("div");
  overlay.className = "countdown";
  $("board").appendChild(overlay);
  const wasClick = engine.clickOn;
  for (let i = song.meta.meter; i >= 1; i--) {
    overlay.textContent = String(i);
    await new Promise((r) => setTimeout(r, beatLen * 1000));
  }
  overlay.remove();
  engine.clickOn = wasClick;
  engine.play();
}

// ---------------------------------------------------------------- controls
document.querySelectorAll<HTMLButtonElement>("#mode-seg button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode as Mode)));
document.querySelectorAll<HTMLButtonElement>("#view-seg button").forEach((b) => {
  b.onclick = () => {
    board.setView(b.dataset.view as any);
    store.set("view", b.dataset.view!);
    document.querySelectorAll("#view-seg button").forEach((x) => x.classList.toggle("on", x === b));
  };
});
const savedView = store.get("view", "player");
board.setView(savedView as any);
document.querySelectorAll<HTMLButtonElement>("#view-seg button").forEach((x) => x.classList.toggle("on", x.dataset.view === savedView));

$("play").onclick = () => engine.toggle();
$("back").onclick = () => engine.seek(engine.time - 5);
$("loop").onclick = toggleLoop;
$("countin").onclick = countIn;
const speed = $("speed") as HTMLInputElement;
speed.oninput = () => {
  engine.setRate(parseFloat(speed.value));
  $("speed-val").textContent = `${Math.round(parseFloat(speed.value) * 100)}%`;
};
const vg = $("vol-guitar") as HTMLInputElement, vb = $("vol-backing") as HTMLInputElement;
const applyVol = () => engine.setVolumes(parseFloat(vg.value), parseFloat(vb.value));
vg.oninput = vb.oninput = applyVol;
applyVol();
($("synth") as HTMLInputElement).onchange = (e) => engine.setSynth((e.target as HTMLInputElement).checked);
($("click") as HTMLInputElement).onchange = (e) => engine.setClick((e.target as HTMLInputElement).checked);
engine.on(() => ($("play").textContent = engine.playing ? "❚❚" : "▶"));

window.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); engine.toggle(); }
  else if (e.code === "ArrowLeft") engine.seek(engine.time - 5);
  else if (e.code === "ArrowRight") engine.seek(engine.time + 5);
  else if (e.key === "l" || e.key === "L") toggleLoop();
  else if (e.key === "1") setMode("chords");
  else if (e.key === "2") setMode("easy");
  else if (e.key === "3") setMode("full");
  else if (e.key === "-" || e.key === "+" || e.key === "=") {
    const v = Math.max(0.4, Math.min(1, parseFloat(speed.value) + (e.key === "-" ? -0.05 : 0.05)));
    speed.value = String(v);
    speed.oninput!(new Event("input"));
  }
});

function frame() {
  engine.tick();
  const now = engine.time;
  if (song) {
    const { cur, next } = chordAt(now);
    const v: Voicing | null = cur?.voicing ?? null;
    board.update(now, mode, events, song.chordSheet, v, next?.voicing ?? null, next?.start ?? null);
    tab.draw(now, mode, events, song.chords, song.chordSheet, song.beats, song.downbeatOffset, song.meta.meter, engine.loop);
    updatePanels(now);
    drawHands(now);
  }
  requestAnimationFrame(frame);
}

function escapeHtml(s: string) {
  return s.replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c] as string);
}

init().then(() => requestAnimationFrame(frame));
