"""Cello and double bass from the bass guitar, at every smoothing level.

Usage: python bass_lines.py <song_id>
Needs the output of analyze.py, export_midi.py and arrange_strings.py. Writes
out/<song_id>/strings/{cello,contrabass}_s{0-4}.wav and registers them in strings.json as "bass_smooth".

  0  every bass note, as transcribed
  1  repeated notes merged, pick noise dropped
  2  longer merges: the line only moves when the bass really changes note
  3  the chord's root (bass + guitars, as in violin_section.py), re-bowed on the bass guitar's strongest hits
  4  the chord's root, held until the chord changes
The double bass plays the line as written, the cello an octave above it.
"""
import json
import sys

import numpy as np
import pretty_midi as pm

import arrange_strings as a
import violin_section as vs

DEFAULT_LEVEL = 1
MERGE = {1: (0.15, 0.10), 2: (0.30, 0.25)}  # level -> (merge repeats within this gap, drop notes shorter than this)
CAP = {1: 2.0, 2: 3.0, 3: 8.0, 4: 8.0}      # longest a note may ring on into the next one


def roots_line(events, bass, lo, hi):
    """One note per chord on its root, in the octave the bass guitar was actually playing there."""
    out, prev = [], None
    for t0, t1, (_, root) in events:
        played = [p for s, e, p in bass if s < t1 and e > t0 and p % 12 == root]
        if played:
            p = int(np.median(played))
        else:
            cands = [c for c in range(lo, hi + 1) if c % 12 == root]
            p = min(cands, key=lambda c: abs(c - (prev if prev is not None else (lo + hi) / 2)))
        prev = a.into_range(p, lo, hi)
        out.append([t0, t1, prev])
    return out


def line_for(level, bass, events, times, strength, lo, hi):
    written = [[s, e, a.into_range(p, lo, hi)] for s, e, p in bass]
    if level == 0:
        return written
    if level in MERGE:
        gap, min_len = MERGE[level]
        return [n for n in a.sustain(written, gap) if n[1] - n[0] >= min_len]
    # the bass guitar hits far more often than the guitars, so only its very strongest hits re-bow at level 3
    share = 0.05 if level == 3 else 0.0
    return vs.rhythm(roots_line(events, bass, lo, hi), times, strength, share, tie=True)


def main():
    sid = sys.argv[1]
    base = a.ROOT / "out" / sid
    out = base / "strings"
    data = json.load(open(base / "analysis.json"))
    pcs = a.key_pcs(data["key"])
    full = pm.PrettyMIDI(str(base / f"{sid}.mid"))
    bass = sorted((n.start, n.end, n.pitch) for i in full.instruments if i.name == "Bass" for n in i.notes)
    guitar = [n for side in ("guitar_L", "guitar_R") for n in data["notes"].get(side, [])] or data["notes"]["guitar"]
    events = vs.chord_events(guitar, bass, pcs, max(e for _, e, _ in guitar))
    stem = base / "stems" / "bass.wav"
    times, strength = vs.accents(stem)
    env = a.envelope(stem)

    _, lo, hi, _ = a.PARTS["contrabass"]
    for level in a.LEVELS:
        written = line_for(level, bass, events, times, strength, lo, hi)
        for name in ("cello", "contrabass"):
            program, clo, chi, pan = a.PARTS[name]
            notes = written if name == "contrabass" else [[s, e, a.into_range(p + 12, clo, chi)] for s, e, p in written]
            notes = a.shape(notes, env)
            if level > 0:
                notes = a.legato(notes, cap=CAP[level])
            a.render(name, notes, program, pan, out, env, stem=f"{name}_s{level}")
        lens = [n[1] - n[0] for n in a.shape(written, env)]
        print(f"bass smooth {level}: {len(lens):4d} notes  median {np.median(lens):.2f}s", flush=True)

    for old in ("cello.wav", "contrabass.wav"):  # the single render from before this slider existed
        (out / old).unlink(missing_ok=True)
    info = json.load(open(out / "strings.json"))
    info["bass_smooth"] = {"parts": ["cello", "contrabass"], "levels": len(a.LEVELS), "default": DEFAULT_LEVEL}
    json.dump(info, open(out / "strings.json", "w"))
    print("registered bass_smooth")


if __name__ == "__main__":
    main()
