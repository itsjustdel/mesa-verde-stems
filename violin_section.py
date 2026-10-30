"""A four-part violin section written for the violin rather than copied from the guitar.

Usage: python violin_section.py <song_id> [both|guitar_L|guitar_R] [--force]
  both (default)  one section from both guitars together: secE / secA / secD / secG
  guitar_L / _R   a section from that guitar alone, panned slightly to its side: secLE.. / secRE..
Needs the output of analyze.py, export_midi.py and arrange_strings.py. Renders every smoothing level in
arrange_strings.LEVELS into out/<song_id>/strings/ and registers the parts in strings.json.
Existing renders are kept unless --force.

  1. Harmony: the guitar notes say which pitch classes sound; the bass picks the root. Chords that last less
     than half a second are absorbed into their neighbour. The key's third is added above the root, since a
     string section voices full chords where the guitar plays power chords.
  2. Voicing: each part lives around one violin string (E, A, D, G) and moves to the nearest note of the next
     chord, so the lines move in small steps instead of following the guitar's pick pattern.
  3. Rhythm: the parts re-bow on the guitar's strongest hits. The smoothing level sets how many hits count,
     from every hit (0) to none, holding until the chord changes (4).
  Two textures are rendered: "full" (all four parts on every chord) and "follow" (only as many violins as the
  guitar plays notes at once, on the guitar's own notes; resting where it is silent). Files: <part>_<texture>_s<level>.
  4. Picked stretches (the guitar picking one note at a time, like an intro pattern) are played as arpeggios:
     the picked notes in order, each on the E/A/D/G part whose range it sits in. Levels 0-3 let the notes ring
     longer as they go up; level 4 keeps chords everywhere.
"""
import itertools
import json
import sys
from collections import Counter

import librosa
import numpy as np
import pretty_midi as pm

import arrange_strings as a

FRAME = 0.25  # seconds per harmony frame
MIN_CHORD_FRAMES = 2  # chords shorter than this are absorbed into their neighbour
# each part's home is one violin string, from the E string down: (lowest note, highest note)
STRINGS = {"E": (76, 88), "A": (69, 81), "D": (62, 74), "G": (55, 67)}
# pan per source: the combined section spreads across the stage, each guitar's section leans gently to its side
PANS = {
    "both": {"E": 34, "A": 50, "D": 78, "G": 94},
    "guitar_L": {"E": 40, "A": 44, "D": 48, "G": 52},
    "guitar_R": {"E": 88, "A": 84, "D": 80, "G": 76},
}
ACCENT_SHARE = [1.0, 0.6, 0.35, 0.15, 0.0]  # per smoothing level: share of the strongest guitar hits that re-bow
MIN_NOTE = 0.12
# A stretch counts as picked (a quiet passage played one note at a time, like an intro pattern) at no more than
# PICKED_RATE hits a second, at least PICKED_SINGLE of them a single note name, and the guitar at most PICKED_LOUD
# of its loud level, judged over PICK_WINDOW seconds; stretches under PICK_MIN are ignored. Loudness matters:
# a separated guitar plays single notes in its loud riffs too, and those should stay chords.
PICKED_RATE, PICKED_SINGLE, PICKED_LOUD, PICK_WINDOW, PICK_MIN = 7.0, 0.75, 0.5, 4.0, 5.0
ARP_RING = {0: 0.6, 1: 0.9, 2: 1.2, 3: 2.0}  # smoothing level -> longest an arpeggio note rings; level 4: chords only


def frames(notes, n, weight=1.0, lo=28, hi=88):
    m = np.zeros((n, 12))
    for s, e, p in notes:
        if not lo <= p <= hi:
            continue
        for f in range(int(s / FRAME), min(n, int(e / FRAME) + 1)):
            ov = min(e, (f + 1) * FRAME) - max(s, f * FRAME)
            if ov > 0:
                m[f, p % 12] += ov * weight
    return m


def chord_of(h, b, pcs):
    if h.sum() + b.sum() < 1e-6:
        return None
    root = int(b.argmax()) if b.sum() > 0 else int(h.argmax())
    tones = [root]
    v = h / (h.max() or 1.0)
    for pc in np.argsort(-v):
        if len(tones) >= 3:
            break
        if v[pc] >= 0.35 and int(pc) not in tones:
            tones.append(int(pc))
    third = next((t for t in ((root + 3) % 12, (root + 4) % 12) if t in pcs), None)
    if third is not None and third not in tones:
        tones.append(third)
    return tuple(sorted(tones)), root


def chord_events(guitar, bass, pcs, dur):
    n = int(dur / FRAME) + 1
    h, b = frames(guitar, n, lo=40), frames(bass, n, lo=28, hi=60)
    per_frame = [chord_of(h[f], b[f], pcs) for f in range(n)]
    runs = []
    for f, c in enumerate(per_frame):
        if runs and runs[-1][2] == c:
            runs[-1][1] = f + 1
        else:
            runs.append([f, f + 1, c])
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] < MIN_CHORD_FRAMES or merged[-1][2] == r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    # Chords are found on a FRAME grid, so their changes would land up to a quarter second away from the hit
    # that made them. Move every change onto the nearest real guitar note start within a frame, so strikes and
    # bow changes happen on the hit. Shared boundaries snap to the same time, so neighbours stay touching.
    onsets = np.array(sorted({s for s, _, p in guitar if 40 <= p <= 88}))

    def snap(t):
        i = np.searchsorted(onsets, t)
        near = [onsets[j] for j in (i - 1, i) if 0 <= j < len(onsets)]
        best = min(near, key=lambda o: abs(o - t), default=t)
        return float(best) if abs(best - t) <= FRAME else t

    return [(snap(f0 * FRAME), snap(f1 * FRAME), c) for f0, f1, c in merged
            if c and snap(f1 * FRAME) - snap(f0 * FRAME) > 0.05]


def voice(events):
    """Pick one chord note per part: no crossing, cover the chord, root in the lowest part, smallest total movement.
    Returns the parts' lines (one note per voiced chord, in step) and the (start, end) of each voiced chord."""
    ranges = list(STRINGS.values())
    prev = [(lo + hi) / 2 for lo, hi in ranges]
    lines = [[] for _ in ranges]
    voiced = []
    for t0, t1, (tones, root) in events:
        cands = [[p for p in range(lo, hi + 1) if p % 12 in tones] for lo, hi in ranges]
        best = None
        for combo in itertools.product(*cands):
            if any(combo[i] <= combo[i + 1] for i in range(len(combo) - 1)):
                continue
            cost = sum(abs(c - p) for c, p in zip(combo, prev))
            cost += 4 * (len(tones) - len({c % 12 for c in combo}))
            cost += 0 if combo[-1] % 12 == root else 3
            if best is None or cost < best[0]:
                best = (cost, combo)
        if best is None:
            continue
        for i, c in enumerate(best[1]):
            lines[i].append([t0, t1, c])
        prev = list(best[1])
        voiced.append((t0, t1))
    return lines, voiced


def accents(path, **band):
    """Hits in a stem: (times, strengths). The time is backtracked from the onset peak to where the attack starts,
    so parts land on the hit rather than a few tens of ms after it. `band` limits the frequencies looked at
    (e.g. fmax=150 for just a bass drum)."""
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, **band)
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    starts = librosa.onset.onset_backtrack(peaks, env)
    return librosa.frames_to_time(starts, sr=sr), env[peaks]


def rhythm(line, times, strength, share, tie):
    """Re-bow held notes on the strongest guitar hits; at high smoothing, tie repeated notes across chord changes."""
    cuts = np.sort(times[strength >= np.quantile(strength, 1 - share)]) if share > 0 else np.array([])
    out = []
    for s, e, p in line:
        inner = [t for t in cuts[(cuts > s + MIN_NOTE) & (cuts < e - MIN_NOTE)]]
        for a0, a1 in zip([s] + inner, inner + [e]):
            # tie only across a chord change (a0 == s), never across a re-bow the accents just made
            if tie and a0 == s and out and out[-1][2] == p and a0 - out[-1][1] < 0.05:
                out[-1][1] = a1
            else:
                out.append([a0, a1, p])
    return out


def guitar_hits(notes, lo=40, hi=76):
    """Notes starting within 40 ms of each other are one hit: [time, lowest pitch, note names]. A picked note and
    its octave overtone share one name; a power chord has two."""
    out = []
    for s, e, p in sorted(n for n in notes if lo <= n[2] <= hi and n[1] - n[0] >= 0.05):
        if out and s - out[-1][0] < 0.04:
            out[-1][1] = min(out[-1][1], p)
            out[-1][2].add(p % 12)
        else:
            out.append([s, p, {p % 12}])
    return out


def picked_spans(hits, dur, env):
    """Stretches where the guitar quietly picks one note at a time: few hits a second, nearly all single notes,
    well below its loud level (env from arrange_strings.envelope)."""
    t = np.array([h[0] for h in hits])
    single = np.array([len(h[2]) == 1 for h in hits])
    half = int(PICK_WINDOW / 2 / a.HOP)
    flags = []
    for f in range(int(dur / FRAME) + 1):
        m = np.abs(t - f * FRAME) < PICK_WINDOW / 2
        i = int(f * FRAME / a.HOP)
        quiet = env[max(0, i - half): i + half].mean() <= PICKED_LOUD if i < len(env) else False
        flags.append(bool(m.sum() >= 4 and m.sum() / PICK_WINDOW <= PICKED_RATE
                          and single[m].mean() >= PICKED_SINGLE and quiet))
    runs = []
    for f, v in enumerate(flags):
        if runs and runs[-1][2] == v:
            runs[-1][1] = f + 1
        else:
            runs.append([f, f + 1, v])
    merged = []
    for r in runs:
        if merged and ((r[1] - r[0]) * FRAME < PICK_MIN or merged[-1][2] == r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(f0 * FRAME, f1 * FRAME) for f0, f1, v in merged if v]


def cut(notes, spans):
    """Remove the parts of notes that fall inside the spans."""
    out = []
    for n in notes:
        pieces = [(n[0], n[1])]
        for a0, a1 in spans:
            pieces = [q for s, e in pieces for q in
                      ([(s, e)] if e <= a0 or s >= a1 else [x for x in ((s, a0), (a1, e)) if x[1] > x[0]])]
        out += [[s, e, *n[2:]] for s, e in pieces if e - s >= MIN_NOTE]
    return out


def arpeggio(hits, spans, ring):
    """Inside picked stretches the violins play the picked notes themselves, in order. Each note goes to the
    E/A/D/G part whose range it sits in, like a violinist crossing strings, and rings until that part's next note."""
    ranges = list(STRINGS.values())
    centers = [(lo + hi) / 2 for lo, hi in ranges]
    parts = [[] for _ in ranges]
    for a0, a1 in spans:
        inside = [h for h in hits if a0 <= h[0] < a1]
        if not inside:
            continue
        shift = 12 * round((69 - np.median([h[1] for h in inside])) / 12)  # the passage, by octaves, around A4
        for t, p, _ in inside:
            q = p + shift
            while q > 88:
                q -= 12
            while q < 55:
                q += 12
            k = min(range(len(ranges)), key=lambda i: abs(q - centers[i]))
            parts[k].append([t, min(t + ring, a1), q])
    for line in parts:
        line.sort()
        for j in range(len(line) - 1):
            line[j][1] = min(line[j][1], line[j + 1][0] - 0.02)
    return [[n for n in line if n[1] - n[0] >= 0.05] for line in parts]


def follow_parts(lines, voiced, hits):
    """The 'follow guitar' texture: per chord, only as many parts play as the guitar sounds note names at once
    (one note -> one violin, a power chord -> two), on the guitar's own notes, most-played first. Where the guitar
    is silent the violins rest. Returns the set of playing part indices for each voiced chord."""
    starts = np.array([h[0] for h in hits])
    keep = []
    for j, (t0, t1) in enumerate(voiced):
        inside = [hits[i] for i in np.nonzero((starts >= t0) & (starts < t1))[0]]
        if not inside:
            keep.append(set())
            continue
        n = max(1, int(round(np.median([len(h[2]) for h in inside]))))
        chosen = []
        for pc, _ in Counter(pc for h in inside for pc in h[2]).most_common():
            part = next((i for i, line in enumerate(lines) if line[j][2] % 12 == pc and i not in chosen), None)
            if part is not None:
                chosen.append(part)
            if len(chosen) >= n:
                break
        keep.append(set(chosen) or {len(lines) - 1})  # the guitar's note isn't in this voicing: the G part plays
    return keep


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    sid = args[0]
    source = args[1] if len(args) > 1 else "both"
    force = "--force" in sys.argv
    base = a.ROOT / "out" / sid
    out = base / "strings"
    data = json.load(open(base / "analysis.json"))
    pcs = a.key_pcs(data["key"])

    if source == "both":
        guitar = [n for side in ("guitar_L", "guitar_R") for n in data["notes"].get(side, [])] or data["notes"]["guitar"]
        stem, tag = base / "stems" / "guitar.wav", ""
    else:
        guitar, stem, tag = data["notes"][source], base / "stems" / f"{source}.wav", source[-1]
    full = pm.PrettyMIDI(str(base / f"{sid}.mid"))
    bass = [(n.start, n.end, n.pitch) for i in full.instruments if i.name == "Bass" for n in i.notes]
    dur = max(e for _, e, _ in guitar)

    events = chord_events(guitar, bass, pcs, dur)
    lens = [t1 - t0 for t0, t1, _ in events]
    print(f"{source}: {len(events)} chords, median {np.median(lens):.2f}s; first ones:", flush=True)
    for t0, t1, (tones, root) in events[:8]:
        print(f"  {t0:6.2f}s  {a.NOTE_NAMES[root]:2s}  {'-'.join(a.NOTE_NAMES[t] for t in tones)}")

    lines, voiced = voice(events)
    moves = [abs(x[2] - y[2]) for line in lines for x, y in zip(line, line[1:])]
    print(f"voice leading: average move {np.mean(moves):.1f} semitones, {100 * np.mean(np.array(moves) > 4):.0f}% over a third")

    env = a.envelope(stem)
    hits = guitar_hits(guitar)
    spans = picked_spans(hits, dur, env)
    print("picked stretches (played as arpeggios):",
          ", ".join(f"{int(s // 60)}:{s % 60:04.1f}-{int(e // 60)}:{e % 60:04.1f}" for s, e in spans) or "none")
    arps = {level: arpeggio(hits, spans, ring) for level, ring in ARP_RING.items()}

    keep = follow_parts(lines, voiced, hits)
    sizes = Counter(len(k) for k in keep)
    print("follow guitar: violins playing per chord:", ", ".join(f"{n}: {100 * c / len(keep):.0f}%" for n, c in sorted(sizes.items())))
    textures = {"full": lines, "follow": [[n for n, kp in zip(line, keep) if k in kp] for k, line in enumerate(lines)]}

    times, strength = accents(stem)
    names = []
    for k, string in enumerate(STRINGS):
        name = f"sec{tag}{string}"
        names.append(name)
        for texture, parts in textures.items():
            for level in a.LEVELS:
                target = out / f"{name}_{texture}_s{level}.wav"
                legacy = out / f"{name}_s{level}.wav"  # a full-section render from before textures existed
                if texture == "full" and legacy.exists() and not force:
                    legacy.replace(target)
                if not force and target.exists():
                    continue
                notes = rhythm(parts[k], times, strength, ACCENT_SHARE[level], tie=level >= 3)
                notes = a.legato(a.shape(notes, env), cap=8.0)
                if level in arps:  # picked stretches: the chord part steps aside for the arpeggio
                    notes = sorted(cut(notes, spans) + a.shape(arps[level][k], env))
                a.render(name, notes, 40, PANS[source][string], out, env, stem=target.stem)
                print(f"{name} {texture} smooth {level}: {len(notes):4d} notes", flush=True)

    info = json.load(open(out / "strings.json"))
    for n in names:
        if n not in info["parts"]:
            info["parts"].append(n)
        if n not in info["smooth"]["parts"]:
            info["smooth"]["parts"].append(n)
    info["smooth"]["variants"] = list(textures)
    info["smooth"].setdefault("variant", "full")
    json.dump(info, open(out / "strings.json", "w"))
    print("registered:", ", ".join(names))


if __name__ == "__main__":
    main()
