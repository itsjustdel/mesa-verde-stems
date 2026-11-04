"""Stems only: no transcription, no arrangement.

Usage: python stems.py <audio> [<audio> ...]
Each song goes to out/<id>/stems/ (id: track number + title, e.g. 05-no-more-bad-future): Demucs htdemucs_6s,
the bass from htdemucs_ft, and the guitar split into guitar_L / guitar_R by pan position when the two sides differ.
"""
import re
import shutil
import sys
from pathlib import Path

from analyze import better_bass, separate, split_guitar

ROOT = Path(__file__).resolve().parent


def song_id(path):
    title = path.stem.split(" - ")[-1]  # "Mesa Verde - Discography; 2004-2009 - 05 No More Bad Future"
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main():
    for arg in sys.argv[1:]:
        src = Path(arg).resolve()
        sid = song_id(src)
        copy = ROOT / "in" / f"{sid}{src.suffix}"  # short path: Demucs' output paths get too long otherwise
        copy.parent.mkdir(exist_ok=True)
        if not copy.exists():
            shutil.copy2(src, copy)
        out = ROOT / "out" / sid
        out.mkdir(parents=True, exist_ok=True)
        stems = out / "stems"
        if not (stems / "bass.wav").exists():
            stems = separate(copy, out)
        better_bass(copy, out, stems)
        split_guitar(stems)
        print(f"{sid}: {', '.join(sorted(p.stem for p in stems.glob('*.wav')))}", flush=True)


if __name__ == "__main__":
    main()
