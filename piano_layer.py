"""A piano ("keys") layer written from the song's chords.

Usage: python piano_layer.py <song_id>
Needs the output of analyze.py, export_midi.py (the bass line) and arrange_strings.py. Writes
out/<song_id>/strings/keys_<rhythm>_s{0-4}.wav and registers "keys" in strings.json with its own smoothing and
rhythm choice ("keys_smooth": levels 0-4, variants guitar / drums / bass).

  Left hand:  the chord's root (from the bass), low, struck at every chord change.
  Right hand: the chord in close position around middle C, moving to the nearest notes between chords
              (chords as in violin_section.py: both guitars for the notes, the bass for the root).
  Rhythm:     a piano dies away, so it re-strikes. "kick" (default): both hands only on the bass drum, playing the
              chord sounding at each kick. "guitar" / "drums" / "bass": the right hand on that stem's strongest
              hits. The smoothing level sets how many hits count, from all (0) to chord changes only (4); in kick
              mode every level stays on real kicks (see kick_choice), so the piano is never off the kick.
  Chords only, everywhere: the guitars' picked melodies are left to the violins.
"""
import bisect
import itertools
import json
import sys

import numpy as np
import pretty_midi as pm
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import arrange_strings as a
import violin_section as vs

RH = (55, 79)  # right hand: G3-G5
LH = (36, 47)  # left hand root: C2-B2
PAN = 64
DEFAULT_LEVEL = 2
# what the strikes follow: (stem, frequency band). "kick" listens to the drum stem's lowest band only, so the
# piano hits with the bass drum; "drums" is every drum hit (kick, snare, hats).
RHYTHM = {"kick": ("drums.wav", {"fmax": 150, "n_mels": 8}), "guitar": ("guitar.wav", {}),
          "drums": ("drums.wav", {}), "bass": ("bass.wav", {})}
DEFAULT_RHYTHM = "kick"
MAX_RING = 2.5  # longest a struck chord is held before the next strike
DRUMLESS = 1.0  # in kick mode a chord gets its own strike only if no kick fell in this many seconds before it
KICK_CUTOFF = 150  # Hz: the bass drum's range in the drum stem


def rh_voicing(tones, prev):
    """The chord in close position, one note per chord tone, as near the previous voicing as possible."""
    cands = [[p for p in range(RH[0], RH[1] + 1) if p % 12 == t] for t in tones]
    ref = prev or [67]
    best = None
    for combo in itertools.product(*cands):
        notes = sorted(combo)
        if notes[-1] - notes[0] > 14:  # within one hand's reach
            continue
        cost = sum(min(abs(n - r) for r in ref) for n in notes) + 0.5 * abs(np.mean(notes) - np.mean(ref))
        if best is None or cost < best[0]:
            best = (cost, notes)
    return best[1] if best else None


def hand_strikes(voiced, cuts):
    """Right hand: at each chord change and on the hits in `cuts` inside it. Left hand: the root per chord."""
    rh_notes, lh_notes = [], []
    for t0, t1, rh, lh in voiced:
        if not rh:
            continue
        strikes = [t0] + [float(t) for t in cuts[(cuts > t0 + vs.MIN_NOTE) & (cuts < t1 - vs.MIN_NOTE)]]
        for s, e in zip(strikes, strikes[1:] + [t1]):
            rh_notes += [[s, e, p] for p in rh]
        lh_notes.append([t0, t1, lh])
    return rh_notes, lh_notes


def refine_kicks(path, times):
    """Move each detected kick onto its real attack, to the millisecond. The onset detector works in 23 ms frames
    and lands 0-40 ms early; here we look at the low-passed drum audio around each kick for the point where the
    level rises a quarter of the way from the quiet before the hit to the hit's peak."""
    y, sr = sf.read(str(path), always_2d=True)
    y = sosfiltfilt(butter(4, KICK_CUTOFF, "low", fs=sr, output="sos"), y.mean(axis=1))
    w = int(sr * 0.002)
    env = np.convolve(np.abs(y), np.ones(w) / w, "same")
    out, peaks = [], []
    for t in times:
        a0 = int(max(0.0, t - 0.03) * sr)
        seg = env[a0:int((t + 0.08) * sr)]
        if len(seg) < 2:
            out.append(float(t))
            peaks.append(0.0)
            continue
        pk = int(np.argmax(seg))
        lo = int(np.argmin(seg[:pk + 1]))
        rise = np.flatnonzero(seg[lo:pk + 1] >= seg[lo] + 0.25 * (seg[pk] - seg[lo]))
        out.append((a0 + lo + (int(rise[0]) if len(rise) else 0)) / sr)
        peaks.append(float(seg[pk]))
    return np.array(out), np.array(peaks)  # attack times, and each hit's low-end peak (how kick-like it is)


def kick_choice(level, voiced, kicks, peaks):
    """Which kicks the keys strike on at each smoothing level. Every choice is a real kick, so the piano always
    lands on one; the levels thin them musically rather than by loudness:
      0 every kick found, ghost notes included    1 real kicks (weak ones, under 30% of a typical kick, dropped)
      2 real kicks, a quick double kick counted once    3 the first real kick of each chord
      4 like 3, but only when the chord's notes change"""
    k = kicks
    if level >= 1:
        k = k[peaks >= 0.3 * np.percentile(peaks, 90)]
    if level >= 2:
        kept = []
        for t in k:
            if not kept or t - kept[-1] >= 0.2:
                kept.append(float(t))
        k = np.array(kept)
    if level >= 3:
        firsts, prev = [], None
        for t0, t1, rh, _ in voiced:
            inside = k[(k >= t0) & (k < t1)]
            if len(inside) and rh and (level == 3 or rh != prev):
                firsts.append(float(inside[0]))
                prev = rh
        k = np.array(firsts)
    return k


def kick_strikes(voiced, kicks, all_kicks):
    """Both hands only on the bass drum: at each of `kicks`, the chord sounding then; a chord change between
    kicks waits for the next one. Where the drums aren't playing (none of `all_kicks` in the last DRUMLESS
    seconds), a chord is struck once where it starts. Each strike rings until the next one (at most MAX_RING)."""
    starts = [v[0] for v in voiced]
    times = sorted(set(map(float, kicks)) | {t0 for t0, t1, rh, _ in voiced
                                              if rh and not ((all_kicks >= t0 - DRUMLESS) & (all_kicks < t1)).any()})
    rh_notes, lh_notes = [], []
    for k, s in enumerate(times):
        i = bisect.bisect_right(starts, s) - 1
        if i < 0 or s >= voiced[i][1] or not voiced[i][2]:
            continue  # no chord here (the guitars are silent)
        e = min(times[k + 1] if k + 1 < len(times) else s + MAX_RING, s + MAX_RING)
        rh_notes += [[s, e, p] for p in voiced[i][2]]
        lh_notes.append([s, e, voiced[i][3]])
    return rh_notes, lh_notes


def main():
    sid = sys.argv[1]
    base = a.ROOT / "out" / sid
    out = base / "strings"
    data = json.load(open(base / "analysis.json"))
    pcs = a.key_pcs(data["key"])
    full = pm.PrettyMIDI(str(base / f"{sid}.mid"))
    bass = [(n.start, n.end, n.pitch) for i in full.instruments if i.name == "Bass" for n in i.notes]
    guitar = [n for side in ("guitar_L", "guitar_R") for n in data["notes"].get(side, [])] or data["notes"]["guitar"]
    dur = max(e for _, e, _ in guitar)
    events = vs.chord_events(guitar, bass, pcs, dur)

    env = a.envelope(base / "stems" / "guitar.wav")  # dynamics always follow the guitars
    accents = {src: vs.accents(base / "stems" / f, **band) for src, (f, band) in RHYTHM.items()
               if (base / "stems" / f).exists()}
    if "kick" in accents:
        accents["kick"] = refine_kicks(base / "stems" / RHYTHM["kick"][0], accents["kick"][0])
    print(f"{len(events)} chords")

    voiced, prev_rh, prev_lh = [], None, 41
    for t0, t1, (tones, root) in events:
        rh = rh_voicing(tones, prev_rh) or prev_rh
        lh = min((p for p in range(LH[0], LH[1] + 1) if p % 12 == root), key=lambda p: abs(p - prev_lh))
        voiced.append((t0, t1, rh, lh))
        prev_rh, prev_lh = rh, lh

    for src, (times, strength) in accents.items():
        for level in a.LEVELS:
            if src == "kick":  # every level strikes on real kicks; see kick_choice
                rh_notes, lh_notes = kick_strikes(voiced, kick_choice(level, voiced, times, strength), times)
            else:
                share = vs.ACCENT_SHARE[level]
                cuts = np.sort(times[strength >= np.quantile(strength, 1 - share)]) if share > 0 else np.array([])
                rh_notes, lh_notes = hand_strikes(voiced, cuts)
            notes = sorted(a.shape(rh_notes, env) + a.shape(lh_notes, env))
            a.render("keys", notes, 0, PAN, out, env, stem=f"keys_{src}_s{level}")
            print(f"keys rhythm {src:6s} smooth {level}: {len({round(n[0], 3) for n in notes}):4d} strikes, "
                  f"{len(notes)} notes", flush=True)

    for old in out.glob("keys_s?.wav"):  # from before the rhythm choice existed
        old.unlink()
    info = json.load(open(out / "strings.json"))
    if "keys" not in info["parts"]:
        info["parts"].append("keys")
    info["keys_smooth"] = {"parts": ["keys"], "levels": len(a.LEVELS), "default": DEFAULT_LEVEL,
                           "variants": list(accents), "variant": DEFAULT_RHYTHM}
    json.dump(info, open(out / "strings.json", "w"))
    print("registered keys")


if __name__ == "__main__":
    main()
