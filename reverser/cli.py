"""Command line: `reverser analyze <video-or-audio> [--title ...]`."""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

from .pipeline import ROOT, run


def main(argv=None):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("root", "coremltools", "transformers", "absl"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    ap = argparse.ArgumentParser(prog="reverser", description="Reverse a guitar video/song into a playable tab.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse a video or audio file")
    a.add_argument("source", type=Path)
    a.add_argument("--title")
    a.add_argument("--slug")
    a.add_argument("--no-video", action="store_true", help="ignore the picture, audio only")
    a.add_argument("--capo", type=int, default=None, help="force a capo position (default: auto)")
    a.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "songs")
    a.add_argument("--work", type=Path, default=ROOT / "work")
    a.add_argument("--force", action="store_true", help="recompute everything (ignore caches)")
    b = sub.add_parser("bench", help="measure accuracy on GuitarSet (see README: Benchmark)")
    b.add_argument("--root", type=Path, default=ROOT / "work" / "guitarset")
    b.add_argument("--limit", type=int, default=None, help="evaluate a spread-out subset of N excerpts")
    b.add_argument("--players", help="comma-separated player ids, e.g. 00,05")
    b.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        run(args.source, args.out, args.work, title=args.title, slug=args.slug, use_video=not args.no_video,
            capo=args.capo, force=args.force)
    elif args.cmd == "bench":
        from . import bench

        s = bench.run(args.root, limit=args.limit, jobs=args.jobs, players=args.players)
        print(bench.format_report(s))
        print(f"\n({s['seconds']}s, per-track rows in {args.root / 'results.json'})")


if __name__ == "__main__":
    main()
