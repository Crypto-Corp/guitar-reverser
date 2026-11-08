export interface Voicing {
  name: string;
  frets: number[]; // low E -> high E, -1 muted
  fingers: number[];
  barre: number | null;
  baseFret: number;
  open: boolean;
}

export interface Chord {
  start: number;
  end: number;
  name: string;
  root: number | null;
  quality: string;
  confidence: number;
  voicing?: Voicing | null;
  shape?: string;
  simple?: string;
}

export interface TabNote {
  s: number; // string 0 = low E
  f: number; // fret (relative to capo)
  p: number; // midi pitch
  fg: number; // fretting finger 0..4
  pk: string; // picking finger p/i/m/a or ""
  end: number;
  amp: number;
}

export interface TabEvent {
  t: number;
  end: number;
  kind: "pluck" | "strum";
  dir: "D" | "U" | null;
  beat: number;
  bar: number;
  pos: number;
  chord: string;
  notes: TabNote[];
}

export interface SheetStrum {
  t: number;
  dir: "D" | "U" | "P";
  chord: string;
  frets: number[];
}

export interface HandFrame {
  t: number;
  h: number[][]; // each 42 numbers (21 x,y normalised)
  f: number | null;
  p: number | null;
  pos: number | null;
  ax: number[] | null;
}

export interface Section {
  label: string;
  startBar: number;
  endBar: number;
  repeats: number;
  chords: string[][];
}

export interface Song {
  version: number;
  meta: {
    title: string;
    slug: string;
    source: string;
    duration: number;
    tuning: number[];
    tuningName: string;
    tuningOffsetCents: number;
    capo: number;
    easyCapo: number;
    key: string;
    keyConfidence: number;
    tempo: number;
    meter: number;
    musicStart: number;
    musicEnd: number;
    style: string;
    hasVideo: boolean;
    noteStats: Record<string, number>;
    vision: { guitarVisibleSeconds?: number; handFrames?: number; positionFrames?: number; guitarTimeline?: number[] };
    pipeline: (string | null)[];
  };
  media: { video: string | null; mix: string; guitar: string; backing: string };
  beats: number[];
  downbeatOffset: number;
  chords: Chord[];
  barChords: string[][];
  sections: Section[];
  pattern: { style: string; cells: string[]; count: number; of?: number };
  barPatterns: { bar: number; style: string; cells: string[]; picks: string[] }[];
  events: TabEvent[];
  easyEvents: TabEvent[];
  chordSheet: SheetStrum[];
  hands: HandFrame[];
}

export type Mode = "chords" | "easy" | "full";

export const FINGER_COLORS = ["#f4f1ea", "#3b82f6", "#22c55e", "#f59e0b", "#ec4899"]; // open,1,2,3,4
export const FINGER_NAMES = ["open", "index", "middle", "ring", "pinky"];
export const PLUCK_COLORS: Record<string, string> = { p: "#a78bfa", i: "#60a5fa", m: "#34d399", a: "#fbbf24" };
export const STRING_NAMES = ["E", "A", "D", "G", "B", "e"];
