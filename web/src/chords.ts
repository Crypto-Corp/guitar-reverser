import type { Voicing } from "./types";
import { FINGER_COLORS } from "./types";

/** SVG chord box in the classic chart style: name badge, x/o marks, numbered dots. */
export function chordSVG(name: string, v: Voicing | null | undefined, opts: { size?: number; colored?: boolean } = {}): string {
  const size = opts.size ?? 150;
  const W = 120, H = 150;
  const left = 18, top = 44, gridW = 84, frets = 4, gridH = 88;
  const sx = gridW / 5, fy = gridH / frets;
  const parts: string[] = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" width="${size}" height="${(size * H) / W}" xmlns="http://www.w3.org/2000/svg" class="chord-svg">`);
  const label = name.replace("maj7", "maj7").replace("add9", "add9");
  const bw = Math.max(36, 9 * label.length + 14);
  parts.push(`<rect x="${(W - bw) / 2}" y="2" width="${bw}" height="24" rx="6" class="cbadge"/>`);
  parts.push(`<text x="${W / 2}" y="20" text-anchor="middle" class="cname">${escapeXml(label)}</text>`);
  if (!v) {
    parts.push(`<text x="${W / 2}" y="90" text-anchor="middle" class="cnone">?</text></svg>`);
    return parts.join("");
  }
  const fretted = v.frets.filter((f) => f > 0);
  const maxF = fretted.length ? Math.max(...fretted) : 0;
  const base = maxF > 4 ? Math.min(...fretted) : 1;
  // nut or base fret label
  if (base === 1) parts.push(`<rect x="${left - 1}" y="${top - 4}" width="${gridW + 2}" height="5" class="cnut"/>`);
  else parts.push(`<text x="${left - 5}" y="${top + fy / 2 + 4}" text-anchor="end" class="cbase">${base}fr</text>`);
  for (let i = 0; i <= frets; i++) parts.push(`<line x1="${left}" y1="${top + i * fy}" x2="${left + gridW}" y2="${top + i * fy}" class="cline"/>`);
  for (let s = 0; s < 6; s++) parts.push(`<line x1="${left + s * sx}" y1="${top}" x2="${left + s * sx}" y2="${top + gridH}" class="cline"/>`);
  // barre
  if (v.barre) {
    const strings = v.frets.map((f, s) => (f === v.barre ? s : -1)).filter((s) => s >= 0);
    if (strings.length >= 2) {
      const y = top + (v.barre - base + 0.5) * fy;
      const x0 = left + Math.min(...strings) * sx, x1 = left + Math.max(...strings) * sx;
      parts.push(`<rect x="${x0 - 6}" y="${y - 6}" width="${x1 - x0 + 12}" height="12" rx="6" class="cdot" ${opts.colored ? `style="fill:${FINGER_COLORS[1]}"` : ""}/>`);
    }
  }
  v.frets.forEach((f, s) => {
    const x = left + s * sx;
    if (f < 0) parts.push(`<text x="${x}" y="${top - 9}" text-anchor="middle" class="cmark">×</text>`);
    else if (f === 0) parts.push(`<circle cx="${x}" cy="${top - 13}" r="4" class="copen"/>`);
    else {
      const y = top + (f - base + 0.5) * fy;
      const fg = v.fingers[s] || 0;
      const fill = opts.colored && fg ? `style="fill:${FINGER_COLORS[fg]}"` : "";
      parts.push(`<circle cx="${x}" cy="${y}" r="7" class="cdot" ${fill}/>`);
      if (fg) parts.push(`<text x="${x}" y="${y + 3.5}" text-anchor="middle" class="cfinger">${fg}</text>`);
    }
  });
  parts.push("</svg>");
  return parts.join("");
}

function escapeXml(s: string) {
  return s.replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c] as string);
}
