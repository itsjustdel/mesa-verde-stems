"""Tuned percussion (marimba / steel drums / kalimba) played from the drum part, on the song's chords.

Usage: python percussion_layer.py <song_id>
Needs the output of analyze.py, export_midi.py (the bass line) and arrange_strings.py. Writes
out/<song_id>/strings/perc_<instrument>_s{0-4}.wav and registers "perc" in strings.json ("perc_smooth":
busyness levels 0-4, variants marimba / steel / kalimba).

Every drum hit becomes a note from the chord sounding at that moment (chords as in violin_section.py):
  kick     the chord's root, low
  snare    the chord's other two notes together, in the middle
  hats     a bubbling pattern up and down the chord's notes, high
  offbeats halfway between consecutive kicks, the same bubbling pattern: the tropical / highlife bounce
           (the drum stem has few hats, so this is where the lilt comes from)
Busyness: 0 everything, 1 kick + snare + offbeats (default), 2 kick + offbeats, 3 kick + snare, 4 kick only.
"""
import bisect
import json
import sys

import numpy as np
import pretty_midi as pm

import arrange_strings as a
import piano_layer as pl
import violin_section as vs

INSTRUMENTS = {"marimba": (12, 0), "steel": (114, 12), "kalimba": (108, 12)}  # GM program, octave shift
DEFAULT_INSTRUMENT = "marimba"
DEFAULT_LEVEL = 1
LEVELS = {0: {"kick", "snare", "hats", "off"}, 1: {"kick", "snare", "off"}, 2: {"kick", "off"},
          3: {"kick", "snare"}, 4: {"kick"}}
LOW, MID, HIGH = (48, 59), (60, 72), (67, 84)  # kick root / snare pair / bubbling notes
VELOCITY = {"kick": 105, "snare": 95, "hats": 75, "off": 82}
LENGTH = {"kick": 0.3, "snare": 0.2, "hats": 0.14, "off": 0.16}
PAN = 70


def away(times, others, gap):
    """Times not within `gap` seconds of any of the other hits."""
    others = np.sort(np.concatenate([o for o in others if len(o)])) if any(len(o) for o in others) else np.array([])
    if not len(others):
        return times
    i = np.clip(np.searchsorted(others, times), 1, len(others) - 1)
    near = np.minimum(np.abs(others[i - 1] - times), np.abs(others[i] - times))
    return times[near > gap]


def drum_hits(path):
    """Kick (real kicks only, attack-accurate), snare and hats from the drum stem, each a sorted array of times."""
    kt, _ = vs.accents(path, fmax=pl.KICK_CUTOFF, n_mels=8)
    kicks, peaks = pl.refine_kicks(path, kt)
    kicks = kicks[peaks >= 0.3 * np.percentile(peaks, 90)]
    st, ss = vs.accents(path, fmin=200, fmax=4000, n_mels=24)
    snare = away(st[ss >= 0.3 * np.percentile(ss, 90)], [kicks], 0.04)
    ht, hs = vs.accents(path, fmin=6000, n_mels=16)
    hats = away(ht[hs >= 0.2 * np.percentile(hs, 90)], [kicks, snare], 0.03)
    # offbeats: halfway between two kicks of a steady pulse, unless the drummer already plays something there
    gaps = np.diff(kicks)
    mids = (kicks[:-1] + gaps / 2)[(gaps > 0.25) & (gaps < 0.7)]
    off = away(mids, [snare, hats], 0.06)
    return {"kick": np.sort(kicks), "snare": snare, "hats": hats, "off": off}


def in_range(pcs, lo, hi):
    return sorted(p for p in range(lo, hi + 1) if p % 12 in pcs)


def notes_for(hits, events, kinds, shift):
    """Each hit of the chosen kinds -> notes from the chord sounding at that moment."""
    starts = [e[0] for e in events]
    timeline = sorted((t, k) for k in kinds for t in hits[k])
    notes, step, last_chord = [], 0, None
    for t, kind in timeline:
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t >= events[i][1]:
            continue  # no chord here (the guitars are silent)
        tones, root = events[i][2]
        if i != last_chord:
            step, last_chord = 0, i
        if kind == "kick":
            pitches = [in_range({root}, *LOW)[0]]
        elif kind == "snare":
            others = [p for p in tones if p != root] or [root]
            pitches = sorted(in_range(set(others), *MID), key=lambda p: abs(p - 64))[:2]
        else:  # hats and offbeats bubble up and down the chord's notes
            ladder = in_range(set(tones), *HIGH)
            cycle = ladder + ladder[-2:0:-1]  # up then back down, without repeating the ends
            pitches = [cycle[step % len(cycle)]]
            step += 1
        notes += [[t, t + LENGTH[kind], p + shift, VELOCITY[kind]] for p in pitches]
    return notes


def main():
    sid = sys.argv[1]
    base = a.ROOT / "out" / sid
    out = base / "strings"
    data = json.load(open(base / "analysis.json"))
    pcs = a.key_pcs(data["key"])
    full = pm.PrettyMIDI(str(base / f"{sid}.mid"))
    bass = [(n.start, n.end, n.pitch) for i in full.instruments if i.name == "Bass" for n in i.notes]
    guitar = [n for side in ("guitar_L", "guitar_R") for n in data["notes"].get(side, [])] or data["notes"]["guitar"]
    events = vs.chord_events(guitar, bass, pcs, max(e for _, e, _ in guitar))

    stem = base / "stems" / "drums.wav"
    hits = drum_hits(stem)
    print(", ".join(f"{k}: {len(v)}" for k, v in hits.items()), flush=True)
    env = a.envelope(stem)

    for inst, (program, shift) in INSTRUMENTS.items():
        for level, kinds in LEVELS.items():
            raw = notes_for(hits, events, kinds, shift)
            # loudness follows the drums, gently, on top of each kind's own accent level
            notes = [[s, e, p, int(min(127, v * (0.55 + 0.45 * a.loudness(env[min(int(s / a.HOP), len(env) - 1)]))))]
                     for s, e, p, v in raw]
            a.render("perc", notes, program, PAN, out, env, stem=f"perc_{inst}_s{level}")
            print(f"perc {inst:8s} level {level}: {len(notes)} notes", flush=True)

    info = json.load(open(out / "strings.json"))
    if "perc" not in info["parts"]:
        info["parts"].append("perc")
    info["perc_smooth"] = {"parts": ["perc"], "levels": len(LEVELS), "default": DEFAULT_LEVEL,
                           "variants": list(INSTRUMENTS), "variant": DEFAULT_INSTRUMENT}
    json.dump(info, open(out / "strings.json", "w"))
    print("registered perc")


if __name__ == "__main__":
    main()
