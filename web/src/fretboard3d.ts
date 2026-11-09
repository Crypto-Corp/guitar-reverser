import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Mode, SheetStrum, TabEvent, Voicing } from "./types";
import { FINGER_COLORS } from "./types";

const SCALE = 26; // scale length in scene units
const N_FRETS = 15;
const HORIZON = 3.2; // seconds of look-ahead
const FALL = 2.4; // units per second
const MAX_GEMS = 160;

export const fretX = (n: number) => SCALE * (1 - Math.pow(2, -n / 12));
const fretCenter = (n: number) => (n <= 0 ? -0.55 : (fretX(n - 1) + fretX(n)) / 2 + 0.12 * (fretX(n) - fretX(n - 1)));
// Real necks are ~4.3 cm wide for a 65 cm scale; widened ~1.6x so strings are easy to tell apart.
const neckHalfWidth = (x: number) => 0.98 + (x / fretX(N_FRETS)) * 0.3;
const stringZ = (s: number, x = 0) => ((s - 2.5) / 2.5) * neckHalfWidth(x) * 0.86;

type View = "player" | "teacher" | "top";

export class Fretboard3D {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  private strings: THREE.Mesh[] = [];
  private stringMats: THREE.MeshStandardMaterial[] = [];
  private gems: THREE.InstancedMesh;
  private tails: THREE.InstancedMesh;
  private hits: THREE.InstancedMesh;
  private ghost: THREE.InstancedMesh;
  private labelPool: THREE.Sprite[] = [];
  private labelTex = new Map<string, THREE.Texture>();
  private arrows: THREE.Mesh[] = [];
  private capoMesh: THREE.Mesh;
  private tmp = new THREE.Object3D();
  private color = new THREE.Color();
  capo = 0;
  view: View = "player";

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    this.scene.add(new THREE.HemisphereLight(0xfff4e0, 0x202030, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(6, 14, 10);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.6);
    rim.position.set(-8, 6, -10);
    this.scene.add(rim);

    this.buildNeck();
    const gemGeo = new THREE.SphereGeometry(0.17, 20, 14);
    gemGeo.scale(1.25, 0.8, 1);
    this.gems = new THREE.InstancedMesh(gemGeo, new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.1, emissiveIntensity: 0.35 }), MAX_GEMS);
    this.gems.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.gems);
    this.tails = new THREE.InstancedMesh(new THREE.BoxGeometry(0.07, 1, 0.07), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.45 }), MAX_GEMS);
    this.scene.add(this.tails);
    const hitGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.06, 28);
    this.hits = new THREE.InstancedMesh(hitGeo, new THREE.MeshStandardMaterial({ emissive: 0xffffff, emissiveIntensity: 0.6, roughness: 0.4 }), 12);
    this.scene.add(this.hits);
    this.ghost = new THREE.InstancedMesh(hitGeo, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.28, depthWrite: false }), 12);
    this.scene.add(this.ghost);
    for (let i = 0; i < 48; i++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, transparent: true }));
      sp.scale.set(0.42, 0.42, 1);
      sp.renderOrder = 10;
      sp.visible = false;
      this.scene.add(sp);
      this.labelPool.push(sp);
    }
    const arrowShape = new THREE.Shape();
    arrowShape.moveTo(-0.12, -0.9);
    arrowShape.lineTo(0.12, -0.9);
    arrowShape.lineTo(0.12, 0.45);
    arrowShape.lineTo(0.3, 0.45);
    arrowShape.lineTo(0, 0.95);
    arrowShape.lineTo(-0.3, 0.45);
    arrowShape.lineTo(-0.12, 0.45);
    for (let i = 0; i < 6; i++) {
      const m = new THREE.Mesh(new THREE.ShapeGeometry(arrowShape), new THREE.MeshBasicMaterial({ transparent: true, side: THREE.DoubleSide, depthWrite: false }));
      m.rotation.x = -Math.PI / 2;
      m.visible = false;
      this.scene.add(m);
      this.arrows.push(m);
    }
    this.capoMesh = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.22, neckHalfWidth(4) * 2.4),new THREE.MeshStandardMaterial({ color: 0x1b1b1f, roughness: 0.5, metalness: 0.4 }));
    this.capoMesh.visible = false;
    this.scene.add(this.capoMesh);

    this.setView("player");
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
  }

  private buildNeck() {
    const xEnd = fretX(N_FRETS) + 0.8;
    // fretboard with a subtle wood texture
    const tex = woodTexture();
    const shape = new THREE.Shape();
    shape.moveTo(-0.05, -neckHalfWidth(0));
    shape.lineTo(xEnd, -neckHalfWidth(xEnd));
    shape.lineTo(xEnd, neckHalfWidth(xEnd));
    shape.lineTo(-0.05, neckHalfWidth(0));
    const boardGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.18, bevelEnabled: false });
    boardGeo.rotateX(Math.PI / 2);
    const board = new THREE.Mesh(boardGeo, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.75, color: 0x8a5a3a }));
    board.position.y = 0;
    this.scene.add(board);
    // neck back (rounded look)
    const neckBack = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, xEnd, 32, 1, false, 0, Math.PI), new THREE.MeshStandardMaterial({ color: 0xb07a45, roughness: 0.6 }));
    neckBack.rotation.z = Math.PI / 2;
    neckBack.rotation.y = 0;
    neckBack.scale.set(0.35, 1, 0.72);
    neckBack.position.set(xEnd / 2, -0.18, 0);
    neckBack.rotateX(Math.PI);
    this.scene.add(neckBack);
    // headstock
    const head = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.2, 2.4), new THREE.MeshStandardMaterial({ color: 0x2a1a10, roughness: 0.5 }));
    head.position.set(-1.75, -0.08, 0);
    this.scene.add(head);
    for (let s = 0; s < 6; s++) {
      const peg = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.35, 12), new THREE.MeshStandardMaterial({ color: 0xd8d8d8, metalness: 0.9, roughness: 0.25 }));
      peg.position.set(-0.7 - (s % 3) * 0.9, 0.15, s < 3 ? -0.85 : 0.85);
      this.scene.add(peg);
    }
    // nut
    const nut = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.12, neckHalfWidth(0) * 2.05), new THREE.MeshStandardMaterial({ color: 0xf2ead8, roughness: 0.4 }));
    nut.position.set(-0.05, 0.06, 0);
    this.scene.add(nut);
    // frets
    const fretMat = new THREE.MeshStandardMaterial({ color: 0xd9d9d9, metalness: 0.95, roughness: 0.2 });
    for (let n = 1; n <= N_FRETS; n++) {
      const x = fretX(n);
      const w = neckHalfWidth(x) * 2;
      const fret = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, w, 8), fretMat);
      fret.rotation.x = Math.PI / 2;
      fret.position.set(x, 0.02, 0);
      this.scene.add(fret);
      const lbl = textSprite(String(n), "#cbd5e1", 64, "rgba(0,0,0,0)");
      lbl.position.set(fretCenter(n), -0.05, neckHalfWidth(x) + 0.42);
      lbl.scale.set(0.38, 0.38, 1);
      this.scene.add(lbl);
    }
    // inlays
    const inlayMat = new THREE.MeshStandardMaterial({ color: 0xf5f0e6, roughness: 0.3, metalness: 0.2 });
    for (const n of [3, 5, 7, 9, 12, 15]) {
      const zs = n === 12 ? [-0.32, 0.32] : [0];
      for (const z of zs) {
        const dot = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.01, 20), inlayMat);
        dot.position.set(fretCenter(n) - 0.12 * (fretX(n) - fretX(n - 1)), 0.005, z);
        this.scene.add(dot);
      }
    }
    // strings: 4 wound bronze + 2 plain steel
    for (let s = 0; s < 6; s++) {
      const r = [0.032, 0.028, 0.024, 0.02, 0.014, 0.012][s];
      const mat = new THREE.MeshStandardMaterial({
        color: s < 4 ? 0xc9a36a : 0xe5e7eb, metalness: 0.85, roughness: 0.3, emissive: 0x000000,
      });
      const x0 = -0.05, x1 = xEnd;
      const z0 = stringZ(s, 0), z1 = stringZ(s, x1);
      const len = Math.hypot(x1 - x0, z1 - z0);
      const geo = new THREE.CylinderGeometry(r, r, len, 8);
      const m = new THREE.Mesh(geo, mat);
      m.position.set((x0 + x1) / 2, 0.12, (z0 + z1) / 2);
      m.rotation.z = Math.PI / 2;
      m.rotation.y = -Math.atan2(z1 - z0, x1 - x0);
      this.scene.add(m);
      this.strings.push(m);
      this.stringMats.push(mat);
      const name = textSprite(["E", "A", "D", "G", "B", "e"][s], "#e2e8f0", 64, "rgba(0,0,0,0)");
      name.position.set(-3.6, 0.1, stringZ(s, 0));
      name.scale.set(0.34, 0.34, 1);
      this.scene.add(name);
    }
    // faint fret lanes for reading the height of falling notes
    const laneMat = new THREE.MeshBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false });
    for (let n = 0; n <= N_FRETS; n++) {
      const x = n === 0 ? -0.05 : fretX(n);
      const lane = new THREE.Mesh(new THREE.PlaneGeometry(neckHalfWidth(x) * 2, HORIZON * FALL), laneMat);
      lane.rotation.y = Math.PI / 2;
      lane.position.set(x, (HORIZON * FALL) / 2, 0);
      this.scene.add(lane);
    }
  }

  private fitZoom = 1;

  setView(v: View) {
    this.view = v;
    // Narrow (portrait) screens: centre on the first frets and pull back so the nut stays visible.
    const a = this.camera.aspect || 1.8;
    const k = (this.fitZoom = Math.min(1.9, Math.max(1, Math.pow(1.6 / a, 0.7))));
    const cx = a < 1.2 ? fretX(2.6) : fretX(3.6);
    this.controls.target.set(cx, 2.0, 0);
    // ~48° elevation: strings are easy to separate and falling notes stay readable
    const off = v === "player" ? [0, 11.0, 10.2] : v === "teacher" ? [0, 11.0, -10.2] : [0, 16, 0.01];
    this.camera.position.set(cx + off[0] * k, 2.0 + off[1] * k, off[2] * k);
    this.controls.update();
  }

  setCapo(c: number) {
    this.capo = c;
    this.capoMesh.visible = c > 0;
    if (c > 0) this.capoMesh.position.set(fretX(c) - 0.22, 0.16, 0);
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const k = Math.min(1.9, Math.max(1, Math.pow(1.6 / this.camera.aspect, 0.7)));
    if (Math.abs(k - this.fitZoom) > 0.04) this.setView(this.view);
  }

  private label(i: number, text: string, color: string, pos: THREE.Vector3, size = 0.42) {
    const sp = this.labelPool[i];
    if (!sp) return;
    const key = `${text}|${color}`;
    let tex = this.labelTex.get(key);
    if (!tex) {
      tex = makeTextTexture(text, "#0b1020", 96, color);
      this.labelTex.set(key, tex);
    }
    (sp.material as THREE.SpriteMaterial).map = tex;
    (sp.material as THREE.SpriteMaterial).needsUpdate = true;
    sp.position.copy(pos);
    sp.scale.set(size, size, 1);
    sp.visible = true;
  }

  /** Draw the state of the song at time `now`. */
  update(now: number, mode: Mode, events: TabEvent[], sheet: SheetStrum[], current: Voicing | null, next: Voicing | null, nextAt: number | null) {
    const capo = this.capo;
    let gi = 0, li = 0, hi = 0, gh = 0;
    const ringing = new Array(6).fill(0);
    const place = (s: number, f: number, y: number) => {
      const fx = f === 0 ? (capo ? fretX(capo) - 0.45 : -0.55) : fretCenter(f + capo);
      return new THREE.Vector3(fx, y, stringZ(s, fx));
    };

    if (mode !== "chords") {
      // binary search start
      let lo = 0, hiI = events.length;
      while (lo < hiI) {
        const mid = (lo + hiI) >> 1;
        if (events[mid].end < now - 0.05) lo = mid + 1; else hiI = mid;
      }
      for (let k = lo; k < events.length; k++) {
        const ev = events[k];
        if (ev.t > now + HORIZON) break;
        for (const n of ev.notes) {
          const dt = ev.t - now;
          if (n.end < now) continue;
          const color = FINGER_COLORS[n.fg] ?? "#fff";
          if (dt > 0) {
            if (gi < MAX_GEMS) {
              const p = place(n.s, n.f, 0.2 + dt * FALL);
              this.tmp.position.copy(p);
              this.tmp.scale.setScalar(1);
              this.tmp.updateMatrix();
              this.gems.setMatrixAt(gi, this.tmp.matrix);
              this.gems.setColorAt(gi, this.color.set(color));
              const tl = Math.min(HORIZON, n.end - ev.t) * FALL;
              this.tmp.position.set(p.x, p.y + tl / 2, p.z);
              this.tmp.scale.set(1, Math.max(0.01, tl), 1);
              this.tmp.updateMatrix();
              this.tails.setMatrixAt(gi, this.tmp.matrix);
              this.tails.setColorAt(gi, this.color.set(color));
              gi++;
              if (dt < 1.6 && li < this.labelPool.length) {
                const txt = n.f === 0 ? "0" : String(n.fg || n.f);
                this.label(li++, txt, color, p.clone().add(new THREE.Vector3(0, 0.02, 0)), 0.36);
              }
            }
          } else {
            // sounding now
            ringing[n.s] = Math.max(ringing[n.s], Math.max(0, 1 - (now - ev.t) / Math.max(0.3, n.end - ev.t)));
            if (hi < 12) {
              const p = place(n.s, n.f, 0.14);
              this.tmp.position.copy(p);
              const pulse = 1 + 0.35 * Math.max(0, 1 - (now - ev.t) * 4);
              this.tmp.scale.set(pulse, 1, pulse);
              this.tmp.updateMatrix();
              this.hits.setMatrixAt(hi, this.tmp.matrix);
              this.hits.setColorAt(hi, this.color.set(color));
              hi++;
              if (li < this.labelPool.length) {
                this.label(li++, n.f === 0 ? "0" : String(n.fg || n.f), color, p.clone().add(new THREE.Vector3(0, 0.45, 0)), 0.44);
              }
            }
          }
        }
      }
    }

    // chord shapes: current (solid in chord mode, ghost otherwise) + next falling in
    const drawShape = (v: Voicing | null, solid: boolean, y: number) => {
      if (!v) return;
      v.frets.forEach((f, s) => {
        if (f < 0) return;
        const p = place(s, f, y);
        if (solid && hi < 12) {
          this.tmp.position.copy(p);
          this.tmp.scale.set(1, 1, 1);
          this.tmp.updateMatrix();
          this.hits.setMatrixAt(hi, this.tmp.matrix);
          this.hits.setColorAt(hi, this.color.set(FINGER_COLORS[v.fingers[s]] ?? "#fff"));
          hi++;
          if (li < this.labelPool.length) this.label(li++, f === 0 ? "0" : String(v.fingers[s] || f), FINGER_COLORS[v.fingers[s]] ?? "#fff", p.clone().add(new THREE.Vector3(0, 0.45, 0)), 0.42);
        } else if (!solid && gh < 12) {
          this.tmp.position.copy(p);
          this.tmp.scale.set(1.1, 1, 1.1);
          this.tmp.updateMatrix();
          this.ghost.setMatrixAt(gh, this.tmp.matrix);
          this.ghost.setColorAt(gh, this.color.set(FINGER_COLORS[v.fingers[s]] ?? "#fff"));
          gh++;
        }
      });
    };
    if (mode === "chords") {
      drawShape(current, true, 0.14);
      if (next && nextAt !== null && nextAt - now < HORIZON) {
        const y = 0.2 + (nextAt - now) * FALL;
        next.frets.forEach((f, s) => {
          if (f < 0 || gi >= MAX_GEMS) return;
          const p = place(s, f, y);
          this.tmp.position.copy(p);
          this.tmp.scale.setScalar(1);
          this.tmp.updateMatrix();
          this.gems.setMatrixAt(gi, this.tmp.matrix);
          this.gems.setColorAt(gi, this.color.set(FINGER_COLORS[next.fingers[s]] ?? "#fff"));
          this.tmp.scale.set(1, 0.001, 1);
          this.tmp.updateMatrix();
          this.tails.setMatrixAt(gi, this.tmp.matrix);
          gi++;
        });
      }
      // strum arrows
      let ai = 0;
      for (const st of sheet) {
        const dt = now - st.t;
        if (dt < -0.02) break;
        if (dt > 0.35 || ai >= this.arrows.length) continue;
        const a = this.arrows[ai++];
        const prog = dt / 0.35;
        const mat = a.material as THREE.MeshBasicMaterial;
        mat.color.set(st.dir === "U" ? "#fb923c" : st.dir === "D" ? "#22d3ee" : "#a78bfa");
        mat.opacity = 0.85 * (1 - prog);
        const x = fretX(N_FRETS) + 1.6;
        const zFrom = st.dir === "U" ? 0.9 : -0.9;
        a.position.set(x, 0.3, zFrom + (st.dir === "U" ? -1 : 1) * prog * 1.8);
        a.rotation.set(-Math.PI / 2, 0, st.dir === "U" ? 0 : Math.PI);
        a.visible = true;
        for (let s = 0; s < 6; s++) if (st.frets[s] >= 0) ringing[s] = Math.max(ringing[s], 1 - prog);
      }
      for (; ai < this.arrows.length; ai++) this.arrows[ai].visible = false;
    } else {
      drawShape(current, false, 0.13);
      this.arrows.forEach((a) => (a.visible = false));
      // strums in tab modes
      let ai = 0;
      for (const ev of events) {
        if (ev.t > now + 0.02) break;
        if (ev.kind !== "strum" || now - ev.t > 0.35 || ai >= this.arrows.length) continue;
        const a = this.arrows[ai++];
        const prog = (now - ev.t) / 0.35;
        const mat = a.material as THREE.MeshBasicMaterial;
        mat.color.set(ev.dir === "U" ? "#fb923c" : "#22d3ee");
        mat.opacity = 0.85 * (1 - prog);
        a.position.set(fretX(N_FRETS) + 1.6, 0.3, (ev.dir === "U" ? 0.9 : -0.9) + (ev.dir === "U" ? -1 : 1) * prog * 1.8);
        a.rotation.set(-Math.PI / 2, 0, ev.dir === "U" ? 0 : Math.PI);
        a.visible = true;
      }
    }

    this.gems.count = gi;
    this.tails.count = gi;
    this.hits.count = hi;
    this.ghost.count = gh;
    this.gems.instanceMatrix.needsUpdate = true;
    this.tails.instanceMatrix.needsUpdate = true;
    this.hits.instanceMatrix.needsUpdate = true;
    this.ghost.instanceMatrix.needsUpdate = true;
    for (const m of [this.gems, this.tails, this.hits, this.ghost]) if (m.instanceColor) m.instanceColor.needsUpdate = true;
    for (let i = li; i < this.labelPool.length; i++) this.labelPool[i].visible = false;

    // vibrating strings
    const time = performance.now() / 1000;
    for (let s = 0; s < 6; s++) {
      const r = ringing[s];
      const m = this.strings[s];
      const wob = r > 0.01 ? 1 + r * (1.6 + 0.6 * Math.sin(time * 90 + s)) : 1;
      m.scale.set(wob, 1, wob);
      this.stringMats[s].emissive.setRGB(r * 0.9, r * 0.75, r * 0.3);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function makeTextTexture(text: string, fg: string, size: number, bg: string): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  if (bg !== "rgba(0,0,0,0)") {
    g.fillStyle = bg;
    g.beginPath();
    g.arc(size / 2, size / 2, size * 0.46, 0, Math.PI * 2);
    g.fill();
    g.lineWidth = size * 0.05;
    g.strokeStyle = "rgba(0,0,0,0.55)";
    g.stroke();
  }
  g.fillStyle = fg;
  g.font = `700 ${Math.round(size * (text.length > 1 ? 0.5 : 0.62))}px Inter, system-ui, sans-serif`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, size / 2, size / 2 + size * 0.03);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function textSprite(text: string, fg: string, size: number, bg: string) {
  return new THREE.Sprite(new THREE.SpriteMaterial({ map: makeTextTexture(text, fg, size, bg), transparent: true, depthWrite: false }));
}

function woodTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = 1024;
  c.height = 128;
  const g = c.getContext("2d")!;
  g.fillStyle = "#3a2417";
  g.fillRect(0, 0, c.width, c.height);
  for (let i = 0; i < 140; i++) {
    const y = Math.random() * c.height;
    g.strokeStyle = `rgba(${20 + Math.random() * 30},${10 + Math.random() * 15},${5},${0.15 + Math.random() * 0.25})`;
    g.lineWidth = 0.5 + Math.random() * 2;
    g.beginPath();
    g.moveTo(0, y);
    for (let x = 0; x <= c.width; x += 32) g.lineTo(x, y + Math.sin(x / 90 + i) * 2.5);
    g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}
