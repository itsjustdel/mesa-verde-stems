"""Text tabs for the bass and each guitar of a split song.

Usage: python tabs.py <song_id>
Needs the output of analyze.py and export_midi.py. Writes out/<song_id>/tabs/{bass,guitar_L,guitar_R}.txt
(or guitar.txt when the guitars weren't split).

  Notes   bass: the pYIN line from the song's MIDI (one note at a time), with octave slips (a note far above the
          line around it) moved down an octave. Guitars: Basic Pitch notes, cleaned up:
          notes starting together are one hit; overtones (repeats of a note an octave or more up) and blips are
          dropped, a hit keeps at most four notes, and stray high hits (over an octave above what the guitar plays
          around them) are dropped.
  Frets   standard tuning (bass E A D G, guitar E A D G B E). Each hit gets the playable string/fret positions
          that stay closest to where the hand already is, preferring low frets and open strings. Notes below the
          lowest string are taken as octave mistakes and moved up an octave.
  Bars    from the beat tracker: 4 beats a bar, 16 slots a bar (sixteenth notes), each line starting with its
          time in the song so it can be found in the player. The tempo varies, so timing is approximate.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi as pm

ROOT = Path(__file__).resolve().parent
TUNINGS = {"bass": [28, 33, 38, 43], "guitar": [40, 45, 50, 55, 59, 64]}  # low string first
STRING_NAMES = {"bass": "EADG", "guitar": "EADGBe"}
MAX_FRET, SPAN = 20, 4  # highest fret used; widest stretch between fretted notes of one hit
SLOTS, BARS_PER_LINE = 16, 2
ONSET_GROUP = 0.04  # notes starting this close together are one hit
BASS_TOP = 60  # C4: bass "notes" above this are harmonics or bleed
REST = 1.0  # after a gap this long the hand may start again from the low end of the neck


def hits_from(notes, kind):
    """[(time, [pitches])]: one entry per hit, cleaned up as described above."""
    lo = TUNINGS[kind][0]
    hits = []
    for s, e, p in sorted(notes):
        if e - s < (0.06 if kind == "bass" else 0.08):
            continue
        if kind == "bass" and p > BASS_TOP:  # harmonics or bleed, not bass notes
            continue
        while p < lo:
            p += 12
        if hits and s - hits[-1][0] < ONSET_GROUP and kind == "guitar":
            hits[-1][1].append(p)
        else:
            hits.append((s, [p]))
    out = []
    for t, ps in hits:
        ps = sorted(set(ps))
        if kind == "guitar":
            base = ps[0]
            # overtones: the same note name again above the lowest note, or anything far above it
            ps = [p for p in ps if p == base or ((p - base) % 12 != 0 and p - base <= 19)][:4]
        out.append((t, ps[:1] if kind == "bass" else ps))
    if kind == "guitar":
        out = drop_strays(out)
    else:
        out = fix_bass_octaves(out, lo)
    return out


def fix_bass_octaves(hits, lowest, window=1.5, above=9):
    """pYIN sometimes locks onto the bass's harmonic and reports a note an octave high. Move a note more than
    `above` semitones over what the bass plays around it (median within `window` s) down an octave."""
    times = np.array([t for t, _ in hits])
    pitches = np.array([ps[0] for _, ps in hits])
    out = []
    for i, (t, ps) in enumerate(hits):
        p = ps[0]
        if p > np.median(pitches[np.abs(times - t) <= window]) + above and p - 12 >= lowest:
            p -= 12
        out.append((t, [p]))
    return out


def drop_strays(hits, window=1.5, above=12):
    """Basic Pitch hears some overtones and bleed as separate high notes in the middle of a low riff. Drop a hit
    whose lowest note is more than an octave above what the guitar plays around it (median within `window` s)."""
    times = np.array([t for t, _ in hits])
    lows = np.array([ps[0] for _, ps in hits])
    keep = []
    for i, (t, ps) in enumerate(hits):
        near = lows[np.abs(times - t) <= window]
        if lows[i] <= np.median(near) + above:
            keep.append((t, ps))
    return keep


def positions(pitch, tuning):
    return [(i, pitch - open_) for i, open_ in enumerate(tuning) if 0 <= pitch - open_ <= MAX_FRET]


def fingering(pitches, tuning, hand):
    """Best (string, fret) for each pitch: one note per string, within a hand span, near the current position."""
    best, best_cost = None, None
    options = [positions(p, tuning) for p in pitches]
    if any(not o for o in options):  # unplayable in this tuning (too high): drop those notes
        options = [o for o in options if o]
        if not options:
            return None
    for combo in itertools.product(*options):
        if len({s for s, _ in combo}) < len(combo):
            continue
        fretted = [f for _, f in combo if f > 0]
        if fretted and max(fretted) - min(fretted) > SPAN:
            continue
        where = min(fretted) if fretted else hand
        # staying near the hand matters, but not so much that one high note keeps the hand up the neck
        cost = abs(where - hand) * 0.6 + where * 0.35 + sum(f for _, f in combo) * 0.1 - 0.5 * sum(f == 0 for _, f in combo)
        if best_cost is None or cost < best_cost:
            best, best_cost = combo, cost
    return best


def bar_starts(beats, end):
    beats = list(beats)
    step = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    while beats[0] > 0:  # pad back to the song's start so nothing before the first beat is lost
        beats.insert(0, beats[0] - step)
    while beats[-1] < end + 4 * step:  # and past the end, where the beat tracker often gives up
        beats.append(beats[-1] + step)
    return beats


def layout(hits, kind, beats, title):
    tuning, names = TUNINGS[kind], STRING_NAMES[kind]
    end = hits[-1][0] if hits else 0
    beats = bar_starts(beats, end)
    bars = [beats[i:i + 5] for i in range(0, len(beats) - 4, 4)]  # each bar: its 4 beats plus the next downbeat
    starts = [b[0] for b in bars]
    grid = [[[None] * len(tuning) for _ in range(SLOTS)] for _ in bars]
    hand, placed, moved, dropped, last = 0, 0, 0, 0, -1e9
    for t, ps in hits:
        if t - last > REST:  # a rest: the hand is free to move, so start from the low end again
            hand = 0
        last = t
        fing = fingering(ps, tuning, hand)
        if not fing:
            dropped += 1
            continue
        fretted = [f for _, f in fing if f > 0]
        hand = min(fretted) if fretted else hand
        b = max(0, int(np.searchsorted(starts, t, side="right")) - 1)
        if b >= len(bars):
            continue
        beat = min(3, max(0, int(np.searchsorted(bars[b], t, side="right")) - 1))
        frac = (t - bars[b][beat]) / max(1e-6, bars[b][beat + 1] - bars[b][beat])
        slot = min(SLOTS - 1, beat * 4 + int(round(frac * 4)))
        while slot < SLOTS and any(grid[b][slot]):  # a fast note lands on a taken slot: push it right
            slot += 1
            moved += 1
        if slot == SLOTS:
            dropped += 1
            continue
        for s, f in fing:
            grid[b][slot][s] = f
        placed += 1

    lines = [f"{title}", f"{kind} tab, standard tuning ({' '.join(names.upper() if kind == 'bass' else 'E A D G B E'.split())})",
             f"{placed} hits placed, {moved} nudged right to fit, {dropped} dropped. 4 beats a bar, 16 slots a bar.", ""]
    last_bar = max((i for i, g in enumerate(grid) if any(any(c) for c in g)), default=-1)
    for b0 in range(0, last_bar + 1, BARS_PER_LINE):
        t0 = starts[b0]
        lines.append(f"{int(max(0, t0) // 60)}:{max(0, t0) % 60:04.1f}  (bar {b0 + 1})")
        for s in reversed(range(len(tuning))):  # high string on top, as tab is read
            row = f"{names[s]}|"
            for b in range(b0, min(b0 + BARS_PER_LINE, last_bar + 1)):
                row += "".join(f"{grid[b][k][s]:-<3}" if grid[b][k][s] is not None else "---" for k in range(SLOTS)) + "|"
            lines.append(row)
        lines.append("")
    return "\n".join(lines), placed, moved, dropped


def main():
    sid = sys.argv[1]
    base = ROOT / "out" / sid
    data = json.load(open(base / "analysis.json"))
    title = Path(next((base.parent.parent / "in").glob(f"{sid}.*"), Path(sid))).stem if (ROOT / "in").exists() else sid
    out = base / "tabs"
    out.mkdir(exist_ok=True)
    full = pm.PrettyMIDI(str(base / f"{sid}.mid"))
    parts = {"bass": ("bass", [(n.start, n.end, n.pitch) for i in full.instruments if i.name == "Bass" for n in i.notes])}
    for side in ("guitar_L", "guitar_R", "guitar") if "guitar_L" in data["notes"] else ("guitar",):
        if side in data["notes"] and not (side == "guitar" and "guitar_L" in data["notes"]):
            parts[side] = ("guitar", data["notes"][side])
    for name, (kind, notes) in parts.items():
        text, placed, moved, dropped = layout(hits_from(notes, kind), kind, data["beats"], f"{title}: {name}")
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"{name}: {placed} hits, {moved} nudged, {dropped} dropped -> {out / (name + '.txt')}")


if __name__ == "__main__":
    main()
