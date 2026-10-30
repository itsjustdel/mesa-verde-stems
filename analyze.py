"""Split a song into stems, transcribe each to MIDI, and summarise it.

Usage: .venv/Scripts/python analyze.py <song.(wav|mp3|flac)> [out_dir]

Writes to out_dir (default: out/<song name>/):
  stems/*.wav        Demucs htdemucs_6s stems (bass from htdemucs_ft instead), plus guitar_L/guitar_R (split by pan
                     position) when the guitars sit apart
  midi/*.mid         Basic Pitch transcription of each pitched stem
  summary.md         tempo, key, chords per bar, note ranges, bass/lead lines
  analysis.json      the same data for player.html (chords, beats, notes)
"""
import json
import sys
from pathlib import Path

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PITCHED = ["bass", "guitar", "guitar_L", "guitar_R", "piano", "other", "vocals"]
HARMONY = ["bass", "guitar", "piano", "other"]

# Krumhansl-Kessler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CHORD_TEMPLATES = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11],
                   "m7": [0, 3, 7, 10], "sus4": [0, 5, 7], "sus2": [0, 2, 7], "dim": [0, 3, 6], "5": [0, 7]}


def name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def separate(audio, out):
    import demucs.separate
    demucs.separate.main(["-n", "htdemucs_6s", "-o", str(out / "_demucs"), str(audio)])
    src = out / "_demucs" / "htdemucs_6s" / audio.stem
    stems = out / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.wav"):
        f.replace(stems / f.name)
    return stems


def better_bass(audio, out, stems):
    """htdemucs_6s's bass ducks on many kick hits (the kick takes the low end). The fine-tuned 4-stem model
    (htdemucs_ft, a specialist model per stem) separates bass more cleanly (on 04: 27% fewer dropouts), so its
    bass replaces the 6-stem one; the 6-stem model still provides the guitars and piano. Runs once per song;
    stems/.bass_ft marks it done."""
    marker = stems / ".bass_ft"
    if marker.exists():
        return
    ft = out / "_ft" / "htdemucs_ft" / audio.stem / "bass.wav"
    if not ft.exists():
        import demucs.separate
        demucs.separate.main(["-n", "htdemucs_ft", "-o", str(out / "_ft"), str(audio)])
    import shutil
    shutil.copy(ft, stems / "bass.wav")
    (stems / "bass_ft.wav").unlink(missing_ok=True)  # the comparison lane is now the bass itself
    marker.touch()
    print("bass: replaced with htdemucs_ft's bass")


def split_guitar(stems, threshold=0.8, sharpness=8.0):
    """Demucs puts every guitar in one stem but keeps the stereo picture. When the guitars sit apart in it
    (low L/R correlation), split them by where each sound sits: every time-frequency bin goes to the left or
    right guitar by its pan position, with a soft crossover in the middle. A plain channel split keeps most of
    the other guitar (a guitar panned half left is still loud in the right channel); this removes most of it.
    Where both guitars play the same note in the same spot, nothing can tell them apart. Each output keeps its
    original stereo position."""
    import librosa
    import soundfile as sf
    wav = stems / "guitar.wav"
    if not wav.exists():
        return None
    y, sr = sf.read(str(wav), always_2d=True)
    corr = float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]) if y.shape[1] == 2 and y.std() > 0 else 1.0
    for side in ("guitar_L", "guitar_R"):
        (stems / f"{side}.wav").unlink(missing_ok=True)
    if corr < threshold:
        hop = 512
        L = librosa.stft(y[:, 0].astype(np.float32), n_fft=2048, hop_length=hop)
        R = librosa.stft(y[:, 1].astype(np.float32), n_fft=2048, hop_length=hop)
        aL, aR = np.abs(L), np.abs(R)
        pan = (aR - aL) / (aR + aL + 1e-9)  # -1 only left, 0 centre, +1 only right
        left = 1.0 / (1.0 + np.exp(sharpness * pan))  # soft mask: ~1 left of centre, 0.5 at centre, ~0 right
        del aL, aR, pan
        for name, m in (("guitar_L", left), ("guitar_R", 1.0 - left)):
            out = np.stack([librosa.istft(m * L, hop_length=hop, length=len(y)),
                            librosa.istft(m * R, hop_length=hop, length=len(y))], 1)
            sf.write(str(stems / f"{name}.wav"), out, sr, subtype="PCM_16")
        print(f"guitar L/R correlation {corr:.2f}: split by pan position into guitar_L / guitar_R")
    else:
        print(f"guitar L/R correlation {corr:.2f}: one guitar part, not split")
    return corr


def drop_cross_bleed(stems, left, right, margin_db=1.0):
    """The L/R guitar split leaves some of each guitar in the other's stem, and the transcription picks those notes
    up. For every note, compare how loud its pitch is in its own stem and in the other guitar's stem over its first
    quarter second; a note more than `margin_db` louder in the other stem is bleed and is dropped. Notes about as
    loud in both (both guitars playing them) are kept."""
    import librosa
    sr, hop, lo, n_bins = 22050, 256, 40, 72  # CQT bins line up with MIDI notes from E2 (40) up
    cqt = {}
    for side in ("guitar_L", "guitar_R"):
        y, _ = librosa.load(str(stems / f"{side}.wav"), sr=sr, mono=True)
        cqt[side] = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(lo), n_bins=n_bins,
                                       bins_per_octave=12))

    def level(side, s, e, p):
        a = int(s * sr / hop)
        return float(cqt[side][p - lo, a:max(int(min(e, s + 0.25) * sr / hop), a + 1)].mean()) + 1e-9

    kept = {}
    for own, other, notes in (("guitar_L", "guitar_R", left), ("guitar_R", "guitar_L", right)):
        kept[own] = [n for n in notes if not lo <= n[2] < lo + n_bins
                     or 20 * np.log10(level(own, n[0], n[1], n[2]) / level(other, n[0], n[1], n[2])) >= -margin_db]
        print(f"{own}: dropped {len(notes) - len(kept[own])} of {len(notes)} notes as bleed from the other guitar")
    return kept["guitar_L"], kept["guitar_R"]


def transcribe(stems, out):
    from basic_pitch.inference import predict
    midi_dir = out / "midi"
    midi_dir.mkdir(exist_ok=True)
    notes = {}
    for stem in PITCHED:
        wav = stems / f"{stem}.wav"
        if not wav.exists():
            continue
        print(f"transcribing {stem}...")
        _, midi, events = predict(str(wav))
        midi.write(str(midi_dir / f"{stem}.mid"))
        # events: (start, end, pitch, amplitude, bends)
        notes[stem] = [(s, e, int(p), float(a)) for s, e, p, a, *_ in events if a > 0.3]
    return notes


def stem_energy(stems):
    import soundfile as sf
    rms = {}
    for f in stems.glob("*.wav"):
        y, _ = sf.read(str(f), always_2d=True)
        rms[f.stem] = float(np.sqrt(np.mean(y ** 2)))
    return rms


def beats(audio):
    import librosa
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0]), librosa.frames_to_time(frames, sr=sr)


def pc_hist(notes, t0=None, t1=None):
    h = np.zeros(12)
    for s, e, p, a in notes:
        if t0 is not None:
            s, e = max(s, t0), min(e, t1)
        if e > s:
            h[p % 12] += (e - s) * a
    return h


def estimate_key(h):
    best = None
    for root in range(12):
        for prof, mode in ((MAJOR, "major"), (MINOR, "minor")):
            r = np.corrcoef(h, np.roll(prof, root))[0, 1]
            if best is None or r > best[0]:
                best = (r, f"{NOTE_NAMES[root]} {mode}")
    return best


def guess_chord(h, bass_h):
    if h.sum() < 1e-6:
        return "-"
    h = h / h.max()
    best, best_score = "-", 0.0
    for root in range(12):
        for q, ivs in CHORD_TEMPLATES.items():
            tmpl = np.zeros(12)
            tmpl[[(root + i) % 12 for i in ivs]] = 1
            score = (h @ tmpl) / len(ivs) - 0.3 * (h @ (1 - tmpl)) / (12 - len(ivs))
            if bass_h.sum() > 0 and int(bass_h.argmax()) == root:
                score += 0.1
            if score > best_score:
                best, best_score = NOTE_NAMES[root] + q, score
    if bass_h.sum() > 0:
        b = int(bass_h.argmax())
        root = best.rstrip("m7aj54sudi")
        if NOTE_NAMES[b] != root and best != "-":
            best += f"/{NOTE_NAMES[b]}"
    return best


def line(notes, limit=48):
    """Melodic line as note names, collapsing simultaneous notes to the highest/lowest."""
    seq = sorted(notes)
    return " ".join(name(p) for _, _, p, _ in seq[:limit])


def main():
    audio = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out") / audio.stem
    out.mkdir(parents=True, exist_ok=True)

    stems = out / "stems"
    if not (stems / "bass.wav").exists():
        stems = separate(audio, out)
    better_bass(audio, out, stems)
    guitar_corr = split_guitar(stems)
    notes = transcribe(stems, out)
    if "guitar_L" in notes and "guitar_R" in notes:
        notes["guitar_L"], notes["guitar_R"] = drop_cross_bleed(stems, notes["guitar_L"], notes["guitar_R"])
    rms = stem_energy(stems)
    tempo, beat_times = beats(audio)

    harmony = [n for s in HARMONY for n in notes.get(s, [])]
    r, key = estimate_key(pc_hist(harmony + notes.get("vocals", [])))

    md = [f"# {audio.stem}", "", f"- Tempo: ~{tempo:.0f} BPM (beat tracker; may be half/double)",
          f"- Key: {key} (confidence r={r:.2f})"]
    if guitar_corr is not None:
        md.append(f"- Guitar L/R correlation: {guitar_corr:.2f}"
                  + (" (split into guitar_L / guitar_R)" if (stems / "guitar_L.wav").exists() else ""))
    md += ["", "## Stem levels (RMS)", ""]
    loud = max(rms.values()) or 1
    for s, v in sorted(rms.items(), key=lambda kv: -kv[1]):
        md.append(f"- {s}: {v / loud:.2f}" + ("  (near-silent, probably absent)" if v / loud < 0.05 else ""))

    md += ["", "## Chords per bar (4 beats/bar, auto-guessed)", ""]
    bars, timeline = [], []
    for i in range(0, len(beat_times) - 4, 4):
        t0, t1 = beat_times[i], beat_times[i + 4]
        half = (t0 + t1) / 2
        c1 = guess_chord(pc_hist(harmony, t0, half), pc_hist(notes.get("bass", []), t0, half))
        c2 = guess_chord(pc_hist(harmony, half, t1), pc_hist(notes.get("bass", []), half, t1))
        bars.append(c1 if c1 == c2 else f"{c1} {c2}")
        timeline += [{"t0": float(t0), "t1": float(half), "chord": c1},
                     {"t0": float(half), "t1": float(t1), "chord": c2}]

    json.dump({"title": audio.stem, "source": str(audio), "tempo": tempo, "key": key,
               "beats": [float(b) for b in beat_times], "chords": timeline,
               "stems": {s: rms[s] / loud for s in rms},
               "notes": {s: [[round(a, 3), round(b, 3), p] for a, b, p, _ in ns] for s, ns in notes.items()}},
              open(out / "analysis.json", "w"))
    for i in range(0, len(bars), 4):
        t = beat_times[i * 4]
        md.append(f"- {int(t // 60)}:{t % 60:04.1f}  | " + " | ".join(bars[i:i + 4]) + " |")

    md += ["", "## Per stem", ""]
    for s, ns in notes.items():
        if not ns:
            continue
        ps = [p for _, _, p, _ in ns]
        top = np.argsort(-pc_hist(ns))[:5]
        md += [f"### {s}", f"- {len(ns)} notes, range {name(min(ps))}-{name(max(ps))}",
               f"- Most used: {', '.join(NOTE_NAMES[i] for i in top)}",
               f"- First notes: {line(ns)}", ""]

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done -> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
