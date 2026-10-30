"""Turn a transcribed song into a simple string arrangement and render each part to audio.

Usage: python arrange_strings.py <song_id>
Needs out/<song_id>/analysis.json and <song_id>.mid (from analyze.py and export_midi.py),
plus tools/fluidsynth and tools/soundfonts/MuseScore_General.sf3.

Writes out/<song_id>/strings/{violin,viola,cello,contrabass}.wav, strings.json and arrangement.mid.

  Guitar L -> violin   guitar_R -> second violin   (a calm string line out of each guitar's riff; LEVELS sets how calm)
  Bass     -> contrabass (as written) and cello (an octave above it): one bass line in octaves
Dynamics: every part follows the loudness of the stem it came from (note velocity plus an expression curve),
and notes where that stem is near silent are dropped.
The vocal is left out on purpose.
"""
import bisect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi as pm
import soundfile as sf

ROOT = Path(__file__).resolve().parent
HOP = 0.05  # seconds per loudness frame
FLUID = next((ROOT / "tools" / "fluidsynth").glob("*/bin/fluidsynth.exe"))
SOUNDFONT = ROOT / "tools" / "soundfonts" / "MuseScore_General.sf3"

# name -> (GM program, lowest and highest note of the instrument, pan 0-127)
PARTS = {
    "violin": (40, 55, 96, 36),
    "violin2": (40, 55, 96, 92),
    "cello": (42, 36, 76, 64),
    "contrabass": (43, 28, 55, 64),
}


def hits(notes, window=0.06, lo=40, hi=76):
    """Collapse the many overtone notes of one guitar hit into a single root note."""
    out, group = [], []
    for s, e, p in sorted(notes):
        if group and s - group[0][0] > window:
            out.append(group)
            group = []
        group.append((s, e, p))
    if group:
        out.append(group)
    line = []
    for g in out:
        usable = [n for n in g if lo <= n[2] <= hi]
        if not usable:
            continue
        s, e, p = min(usable, key=lambda n: n[2])
        if max(n[1] for n in g) - s >= 0.08:  # ignore blips
            line.append([s, max(n[1] for n in g), p])
    return line


def into_range(p, lo, hi):
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def legato(line, cap=1.2, gap=0.02):
    """Strings sustain: each note rings until the next one starts."""
    for i, n in enumerate(line):
        nxt = line[i + 1][0] if i + 1 < len(line) else n[1]
        n[1] = max(n[0] + 0.1, min(nxt - gap, n[0] + cap))
    return line


NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_STEPS, MINOR_STEPS = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]

# Smoothing level -> how aggressively the guitar riff is turned into a calm string line.
#   gap: merge repeated notes on one voice if they restart within this many seconds (kills the re-picked, staccato feel)
#   min_len: drop notes shorter than this (pick noise); cap: longest a note may ring until the next one
LEVELS = {
    0: None,  # the plain line: lowest note of every guitar hit
    1: {"gap": 0.12, "min_len": 0.10, "follow": True, "cap": 1.2},
    2: {"gap": 0.20, "min_len": 0.20, "follow": True, "cap": 2.0},
    3: {"gap": 0.30, "min_len": 0.40, "follow": True, "cap": 3.0},
    4: {"gap": 0.45, "min_len": 0.80, "follow": True, "cap": 4.0},
}
DEFAULT_LEVEL = 2


def key_pcs(key):
    name, mode = key.split()
    root = NOTE_NAMES.index(name)
    return {(root + i) % 12 for i in (MINOR_STEPS if mode == "minor" else MAJOR_STEPS)}


def snap(p, pcs):
    """Nudge a note that is outside the song's key onto the nearest scale note."""
    for d in (0, -1, 1, -2, 2):
        if (p + d) % 12 in pcs:
            return p + d
    return p


def voices(notes, lo=40, hi=88, min_len=0.05, max_voices=8):
    """Split a polyphonic transcription into voices. Basic Pitch doesn't say which string played a note, so each
    note joins the free voice whose last pitch is closest, roughly like following one string at a time."""
    out = []
    for s, e, p in sorted(notes):
        if not lo <= p <= hi or e - s < min_len:
            continue
        best, dist = None, 10
        for v in out:
            if v[-1][1] <= s + 0.03 and abs(v[-1][2] - p) < dist:
                best, dist = v, abs(v[-1][2] - p)
        if best is None:
            if len(out) < max_voices:
                out.append([[s, e, p]])
                continue
            free = [v for v in out if v[-1][1] <= s + 0.03]
            best = min(free or out, key=lambda v: abs(v[-1][2] - p))
        best.append([s, e, p])
    return out


def sustain(voice, gap):
    """A string that re-sounds the same pitch within `gap` seconds is one held note, not a new hit."""
    out = []
    for s, e, p in voice:
        if out and out[-1][2] == p and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, p])
    return out


def lead_line(notes, min_len):
    """One note at a time, longest first, so ringing notes win over pick noise and overtones."""
    starts, placed = [], []
    for n in sorted((n for n in notes if n[1] - n[0] >= min_len), key=lambda n: -(n[1] - n[0])):
        i = bisect.bisect(starts, n[0])
        clash = any(min(n[1], x[1]) - max(n[0], x[0]) > 0.35 * min(n[1] - n[0], x[1] - x[0])
                    for x in placed[max(0, i - 3): i + 3])
        if not clash:
            starts.insert(i, n[0])
            placed.insert(i, n)
    return placed


def follow(line, lo, hi):
    """Keep each note in the octave closest to the previous one, so the line doesn't leap around."""
    prev = None
    for n in line:
        cands = [c for c in range(lo, hi + 1) if c % 12 == n[2] % 12] or [into_range(n[2], lo, hi)]
        n[2] = min(cands, key=lambda c: abs(c - (prev if prev is not None else (lo + hi) / 2)))
        prev = n[2]
    return line


def guitar_line(raw, level, lo, hi, pcs):
    if level == 0:
        return [[s, e, into_range(p, lo, hi)] for s, e, p in hits(raw)], 1.2
    c = LEVELS[level]
    merged = [n for v in voices(raw) for n in sustain(v, c["gap"])]
    line = lead_line(merged, c["min_len"])
    for n in line:
        n[2] = into_range(snap(n[2], pcs), lo, hi)
    return (follow(line, lo, hi) if c["follow"] else line), c["cap"]


def envelope(path):
    """Loudness of a stem over time in HOP-second frames, smoothed over ~0.25s; 1.0 = a typical loud passage."""
    y, sr = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1)
    n = int(sr * HOP)
    rms = np.sqrt((y[: len(y) // n * n].reshape(-1, n) ** 2).mean(axis=1))
    rms = np.convolve(rms, np.ones(5) / 5, mode="same")
    return rms / (np.percentile(rms, 95) or 1.0)


def loudness(lv):
    """Gentle response to the source's loudness (square root of it), so quiet passages such as a picked intro
    come out softer but stay clearly audible instead of nearly vanishing."""
    return min(1.0, max(0.0, float(lv))) ** 0.5


def shape(line, env, floor=0.1):
    """Give each note a velocity that follows its source stem's loudness; drop notes where the stem is near silent
    (there the 'note' is bleed from other instruments)."""
    out = []
    for s, e, p in line:
        a = int(s / HOP)
        if a >= len(env):
            continue
        lv = float(env[a: max(int(e / HOP), a + 1)].mean())
        if lv >= floor:
            out.append([s, e, p, int(40 + 78 * loudness(lv))])
    return out


def render(name, notes, program, pan, out, env, stem=None):
    stem = stem or name
    mid = pm.PrettyMIDI(initial_tempo=120)
    inst = pm.Instrument(program=program, name=name)
    inst.control_changes.append(pm.ControlChange(10, pan, 0))
    last = -99
    for i in range(0, len(env), 2):  # expression (CC11) curve: the part swells and fades with its source
        v = int(45 + 82 * loudness(env[i]))
        if abs(v - last) >= 4:
            inst.control_changes.append(pm.ControlChange(11, v, i * HOP))
            last = v
    for s, e, p, v in notes:
        inst.notes.append(pm.Note(velocity=v, pitch=p, start=s, end=e))
    mid.instruments.append(inst)
    mid_path, wav = out / f"{stem}.mid", out / f"{stem}.wav"
    mid.write(str(mid_path))
    subprocess.run([str(FLUID), "-ni", "-g", "1.0", "-T", "wav", "-F", str(wav), "-r", "44100", str(SOUNDFONT), str(mid_path)],
                   check=True, capture_output=True)
    mid_path.unlink()
    return inst


def main():
    sid = sys.argv[1]
    base = ROOT / "out" / sid
    out = base / "strings"
    out.mkdir(exist_ok=True)
    data = json.load(open(base / "analysis.json"))
    full = pm.PrettyMIDI(str(base / f"{sid}.mid"))
    bass = sorted((n.start, n.end, n.pitch) for i in full.instruments if i.name == "Bass" for n in i.notes)

    source = {"violin": "guitar_L", "violin2": "guitar_R", "cello": "bass", "contrabass": "bass"}
    envs = {}

    def env_for(name):
        stem = base / "stems" / f"{source[name]}.wav"
        if not stem.exists():  # song without a L/R guitar split
            stem = base / "stems" / "guitar.wav"
        return envs.setdefault(stem, envelope(stem))

    pcs = key_pcs(data["key"])
    for old in ("viola.wav", "violin.wav"):  # from before the second violin and the slider existed
        (out / old).unlink(missing_ok=True)
    arr = pm.PrettyMIDI(initial_tempo=120)
    # violin and viola are rendered at every smoothing level so the player's slider can switch between them
    for name in ("violin", "violin2"):
        raw = data["notes"].get(source[name]) or data["notes"].get("guitar", [])
        program, lo, hi, pan = PARTS[name]
        for level in LEVELS:
            line, cap = guitar_line(raw, level, lo, hi, pcs)
            line = legato(shape(line, env_for(name)), cap=cap)
            inst = render(name, line, program, pan, out, env_for(name), stem=f"{name}_s{level}")
            if level == DEFAULT_LEVEL:
                arr.instruments.append(inst)
            leaps = np.abs(np.diff([n[2] for n in line])) if len(line) > 1 else np.zeros(1)
            print(f"{name} smooth {level}: {len(line):5d} notes  median {np.median([n[1] - n[0] for n in line]):.2f}s  "
                  f"avg leap {leaps.mean():.1f} st  leaps over a fifth {100 * (leaps > 7).mean():.0f}%", flush=True)
        (out / f"{name}.wav").unlink(missing_ok=True)  # from before the slider existed
    # one bass line in octaves, the usual orchestral way: double bass as written, cello an octave above it
    contrabass = [[s, e, into_range(p, PARTS["contrabass"][1], PARTS["contrabass"][2])] for s, e, p in bass]
    for name in ("cello", "contrabass"):
        program, lo, hi, pan = PARTS[name]
        notes = contrabass if name == "contrabass" else [[s, e, into_range(p + 12, lo, hi)] for s, e, p in contrabass]
        line = shape(notes, env_for(name))
        arr.instruments.append(render(name, line, program, pan, out, env_for(name)))
        print(f"{name}: {len(bass)} bass notes -> {len(line)}")
    arr.write(str(base / "arrangement.mid"))
    info = {"parts": ["violin", "violin2", "cello", "contrabass"],
            "smooth": {"parts": ["violin", "violin2"], "levels": len(LEVELS), "default": DEFAULT_LEVEL}}
    if (out / "strings.json").exists():  # keep parts other scripts added, such as rank_violins.py's violinL1..
        old = json.load(open(out / "strings.json"))
        info["parts"] += [n for n in old["parts"] if n not in info["parts"]]
        info["smooth"]["parts"] += [n for n in old["smooth"]["parts"] if n not in info["smooth"]["parts"]]
        for k, v in old.items():  # e.g. bass_lines.py's "bass_smooth"
            info.setdefault(k, v)
        for k, v in old["smooth"].items():  # e.g. violin_section.py's texture variants
            info["smooth"].setdefault(k, v)
    json.dump(info, open(out / "strings.json", "w"))
    print("rendered")


if __name__ == "__main__":
    main()
