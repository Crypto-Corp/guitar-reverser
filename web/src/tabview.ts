import type { Chord, Mode, SheetStrum, TabEvent } from "./types";
import { FINGER_COLORS, PLUCK_COLORS, STRING_NAMES } from "./types";

/** Horizontally scrolling 6-line tab with a fixed playhead. Click to seek. */
export class TabView {
  private canvas: HTMLCanvasElement;
  private g: CanvasRenderingContext2D;
  pxPerSec = 150;
  onSeek: (t: number) => void = () => {};
  private lastNow = 0;

  constructor(private host: HTMLElement) {
    this.canvas = document.createElement("canvas");
    host.appendChild(this.canvas);
    this.g = this.canvas.getContext("2d")!;
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    let dragX: number | null = null, dragT = 0;
    this.canvas.addEventListener("pointerdown", (e) => {
      dragX = e.clientX;
      dragT = this.lastNow;
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (dragX === null) return;
      const dx = e.clientX - dragX;
      if (Math.abs(dx) > 4) this.onSeek(Math.max(0, dragT - dx / this.pxPerSec));
    });
    this.canvas.addEventListener("pointerup", (e) => {
      if (dragX !== null && Math.abs(e.clientX - dragX) <= 4) {
        const r = this.canvas.getBoundingClientRect();
        const x = e.clientX - r.left;
        this.onSeek(Math.max(0, this.lastNow + (x - this.playheadX()) / this.pxPerSec));
      }
      dragX = null;
    });
    this.canvas.addEventListener("wheel", (e) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        this.pxPerSec = Math.max(50, Math.min(500, this.pxPerSec * (e.deltaY > 0 ? 0.9 : 1.1)));
      }
    }, { passive: false });
  }

  private resize() {
    const dpr = Math.min(2, window.devicePixelRatio);
    this.canvas.width = this.host.clientWidth * dpr;
    this.canvas.height = this.host.clientHeight * dpr;
    this.canvas.style.width = this.host.clientWidth + "px";
    this.canvas.style.height = this.host.clientHeight + "px";
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private playheadX() {
    return this.host.clientWidth * 0.22;
  }

  draw(now: number, mode: Mode, events: TabEvent[], chords: Chord[], sheet: SheetStrum[], beats: number[], downbeatOffset: number, meter: number, loop: [number, number] | null) {
    this.lastNow = now;
    const g = this.g;
    const W = this.host.clientWidth, H = this.host.clientHeight;
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim() || "#e5e7eb";
    const muted = css.getPropertyValue("--muted").trim() || "#64748b";
    const line = css.getPropertyValue("--line").trim() || "#334155";
    g.clearRect(0, 0, W, H);
    const top = 34, bottom = H - 26;
    const gap = (bottom - top) / 5;
    const ph = this.playheadX();
    const x = (t: number) => ph + (t - now) * this.pxPerSec;
    const t0 = now - ph / this.pxPerSec, t1 = now + (W - ph) / this.pxPerSec;
    const yOf = (s: number) => top + (5 - s) * gap; // high e on top

    if (loop) {
      g.fillStyle = "rgba(250, 204, 21, 0.08)";
      g.fillRect(x(loop[0]), 0, x(loop[1]) - x(loop[0]), H);
    }
    // strings
    g.strokeStyle = line;
    g.lineWidth = 1;
    for (let s = 0; s < 6; s++) {
      g.beginPath();
      g.moveTo(0, yOf(s));
      g.lineTo(W, yOf(s));
      g.stroke();
    }
    g.fillStyle = muted;
    g.font = "600 11px Inter, system-ui, sans-serif";
    g.textAlign = "left";
    g.textBaseline = "middle";
    // beats & bars
    beats.forEach((b, i) => {
      if (b < t0 - 1 || b > t1 + 1) return;
      const isBar = (i - downbeatOffset) % meter === 0;
      g.strokeStyle = isBar ? muted : line;
      g.globalAlpha = isBar ? 0.9 : 0.35;
      g.lineWidth = isBar ? 1.5 : 1;
      g.beginPath();
      g.moveTo(x(b), top);
      g.lineTo(x(b), bottom);
      g.stroke();
      g.globalAlpha = 1;
      if (isBar) {
        g.fillStyle = muted;
        g.font = "500 10px Inter, system-ui, sans-serif";
        g.fillText(String(Math.floor((i - downbeatOffset) / meter) + 1), x(b) + 3, top - 8);
      }
    });
    // chord names
    g.font = "700 14px Inter, system-ui, sans-serif";
    for (const c of chords) {
      if (c.end < t0 || c.start > t1 || c.name === "N") continue;
      const cx = Math.max(x(c.start), 30);
      if (x(c.end) < 40) continue;
      g.fillStyle = c.start <= now && now < c.end ? "#facc15" : ink;
      g.fillText(c.name, cx + 4, 14);
    }
    // notes
    g.textAlign = "center";
    if (mode === "chords") {
      for (const st of sheet) {
        if (st.t < t0 - 0.5 || st.t > t1 + 0.5) continue;
        const xx = x(st.t);
        const active = Math.abs(st.t - now) < 0.12;
        g.fillStyle = st.dir === "U" ? "#fb923c" : st.dir === "D" ? "#22d3ee" : "#a78bfa";
        g.globalAlpha = active ? 1 : 0.8;
        drawArrow(g, xx, top, bottom, st.dir === "U");
        g.globalAlpha = 1;
        g.font = "700 12px Inter, system-ui, sans-serif";
        g.fillText(st.dir === "P" ? "pick" : st.dir === "D" ? "↓" : "↑", xx, bottom + 14);
      }
    } else {
      for (const ev of events) {
        if (ev.end < t0 - 0.5) continue;
        if (ev.t > t1 + 0.5) break;
        const xx = x(ev.t);
        const past = ev.t < now - 0.02;
        if (ev.kind === "strum") {
          g.fillStyle = ev.dir === "U" ? "rgba(251,146,60,0.35)" : "rgba(34,211,238,0.35)";
          drawArrow(g, xx - 13, top, bottom, ev.dir === "U", 4);
        }
        for (const n of ev.notes) {
          const y = yOf(n.s);
          const txt = String(n.f);
          g.font = "700 13px JetBrains Mono, ui-monospace, monospace";
          const w = g.measureText(txt).width + 8;
          // sustain bar
          g.fillStyle = FINGER_COLORS[n.fg] + "33";
          g.fillRect(xx, y - 3, Math.max(0, (n.end - ev.t) * this.pxPerSec), 6);
          g.fillStyle = past ? "#1e293b" : "#0f172a";
          const active = ev.t <= now && now < n.end;
          g.strokeStyle = FINGER_COLORS[n.fg];
          g.lineWidth = active ? 2.5 : 1.5;
          roundRect(g, xx - w / 2, y - 9, w, 18, 5);
          g.fill();
          g.stroke();
          g.fillStyle = active ? "#facc15" : ink;
          g.fillText(txt, xx, y + 1);
        }
        // picking fingers under the staff
        if (ev.kind === "pluck") {
          const pk = ev.notes.map((n) => n.pk).filter(Boolean).join("");
          if (pk) {
            g.font = "700 11px Inter, system-ui, sans-serif";
            g.fillStyle = PLUCK_COLORS[pk[0]] ?? muted;
            g.fillText(pk, xx, bottom + 14);
          }
        } else if (ev.dir) {
          g.font = "800 11px Inter, system-ui, sans-serif";
          g.fillStyle = ev.dir === "U" ? "#fb923c" : "#22d3ee";
          g.fillText(ev.dir === "U" ? "↑" : "↓", xx, bottom + 14);
        }
      }
    }
    // string names
    g.textAlign = "left";
    g.font = "700 11px JetBrains Mono, ui-monospace, monospace";
    for (let s = 0; s < 6; s++) {
      g.fillStyle = "rgba(15,23,42,0.85)";
      g.fillRect(0, yOf(s) - 8, 18, 16);
      g.fillStyle = muted;
      g.fillText(STRING_NAMES[s], 5, yOf(s) + 1);
    }
    // playhead
    g.strokeStyle = "#facc15";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(ph, top - 14);
    g.lineTo(ph, bottom + 4);
    g.stroke();
  }
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

function drawArrow(g: CanvasRenderingContext2D, x: number, top: number, bottom: number, isUpStrum: boolean, w = 6) {
  // Beginners read "down" as physically down, so ↓ = down-strum regardless of staff orientation.
  const y0 = isUpStrum ? bottom - 4 : top + 4, y1 = isUpStrum ? top + 4 : bottom - 4;
  const dir = Math.sign(y1 - y0);
  g.beginPath();
  g.moveTo(x - w / 2, y0);
  g.lineTo(x + w / 2, y0);
  g.lineTo(x + w / 2, y1 - dir * 10);
  g.lineTo(x + w * 1.4, y1 - dir * 10);
  g.lineTo(x, y1);
  g.lineTo(x - w * 1.4, y1 - dir * 10);
  g.lineTo(x - w / 2, y1 - dir * 10);
  g.closePath();
  g.fill();
}
