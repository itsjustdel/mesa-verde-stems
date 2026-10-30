"""Split one guitar into several violins by rank: at every moment the lowest sounding note goes to violin 1,
the next one up to violin 2, and so on up to RANKS (a power chord becomes root / fifth / octave).

Usage: python rank_violins.py <song_id> [guitar_L|guitar_R] [--force]      (default guitar_L)
Needs the output of analyze.py, export_midi.py and arrange_strings.py. Adds violinL1..L3 (or violinR1..R3)
to out/<song_id>/strings/ at every smoothing level in arrange_strings.LEVELS and registers them in strings.json.

Basic Pitch can't say which string played a note, so this is a split by pitch rank, not by string. Notes above
the last rank are left out. All ranks move up the same two octaves so the chord's shape is kept.
"""
import json
import sys

import numpy as np

import arrange_strings as a

RANKS = 6  # a guitar has six strings, so six notes at once is the most it can sound
SHIFT = 24  # guitar register -> violin register, the same for every rank so intervals survive
PANS = {"L": (12, 24, 36, 48, 60, 72), "R": (115, 103, 91, 79, 67, 55)}


def rank_lines(notes, ranks=RANKS, lo=40, hi=88, min_len=0.05):
    """Sweep through time; between note boundaries, rank k gets the k-th lowest distinct sounding pitch."""
    ns = sorted(n for n in notes if lo <= n[2] <= hi and n[1] - n[0] >= min_len)
    cuts = sorted({t for s, e, _ in ns for t in (s, e)})
    lines = [[] for _ in range(ranks)]
    active, i = [], 0
    for t0, t1 in zip(cuts, cuts[1:]):
        while i < len(ns) and ns[i][0] <= t0 + 1e-9:
            active.append(ns[i])
            i += 1
        active = [n for n in active if n[1] > t0 + 1e-9]
        pitches = sorted({n[2] for n in active})
        for r in range(min(ranks, len(pitches))):
            line = lines[r]
            if line and line[-1][2] == pitches[r] and abs(line[-1][1] - t0) < 1e-6:
                line[-1][1] = t1
            else:
                line.append([t0, t1, pitches[r]])
    return lines


def to_violin(p):
    p += SHIFT
    while p > 96:
        p -= 12
    while p < 55:
        p += 12
    return p


def level_line(line, level, pcs):
    out = [list(n) for n in line]
    cap = 1.2
    if a.LEVELS[level]:
        c = a.LEVELS[level]
        out = [n for n in a.sustain(out, c["gap"]) if n[1] - n[0] >= c["min_len"]]
        for n in out:
            n[2] = a.snap(n[2], pcs)
        cap = c["cap"]
    for n in out:
        n[2] = to_violin(n[2])
    return out, cap


def main():
    sid = sys.argv[1]
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    side = args[1] if len(args) > 1 else "guitar_L"
    force = "--force" in sys.argv  # by default ranks that are already rendered are kept (rank k doesn't depend on RANKS)
    tag = side[-1]  # L or R
    base = a.ROOT / "out" / sid
    out = base / "strings"
    data = json.load(open(base / "analysis.json"))
    pcs = a.key_pcs(data["key"])
    env = a.envelope(base / "stems" / f"{side}.wav")
    lines = rank_lines(data["notes"][side])
    print(f"{side}: {len(data['notes'][side])} notes -> ranks of {[len(l) for l in lines]} segments", flush=True)

    names = []
    for r, line in enumerate(lines):
        name = f"violin{tag}{r + 1}"
        names.append(name)
        for level in a.LEVELS:
            if not force and (out / f"{name}_s{level}.wav").exists():
                continue
            notes, cap = level_line(line, level, pcs)
            notes = a.legato(a.shape(notes, env), cap=cap)
            a.render(name, notes, 40, PANS[tag][r], out, env, stem=f"{name}_s{level}")
            med = np.median([n[1] - n[0] for n in notes]) if notes else 0
            print(f"{name} smooth {level}: {len(notes):5d} notes  median {med:.2f}s", flush=True)

    info = json.load(open(out / "strings.json"))
    for n in names:
        if n not in info["parts"]:
            info["parts"].append(n)
        if n not in info["smooth"]["parts"]:
            info["smooth"]["parts"].append(n)
    json.dump(info, open(out / "strings.json", "w"))
    print("registered:", ", ".join(names))


if __name__ == "__main__":
    main()
