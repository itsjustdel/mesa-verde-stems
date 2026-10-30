"""Build one multitrack MIDI file for an already-split song.

Usage: python export_midi.py <song_id>      (reads out/<song_id>/, writes out/<song_id>/<song_id>.mid)

Notes sit at their real times in the audio (no quantizing, no tempo map), so the file lines up with the
stems even when the tempo drifts. The file's nominal tempo is a fixed 120 BPM and means nothing musically.

  bass, vocals   pYIN, one note at a time (cleaner than Basic Pitch for single lines)
  drums          onset detector, hits sorted into kick / snare / hat by where their energy sits
  guitar_L/R     Basic Pitch notes from analysis.json (distorted guitar stays rough)
  piano, other   skipped: near-silent stems are mostly bleed from other instruments
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi as pm

SR = 16000
HOP = 256


def mono(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def pyin_notes(path, lo, hi, min_len=0.08):
    y = mono(path)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz(lo), fmax=librosa.note_to_hz(hi),
                                 sr=SR, frame_length=2048, hop_length=HOP)
    midi = np.where(voiced & ~np.isnan(f0), librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0][:len(midi)]
    loud = rms / (rms.max() or 1)
    dt = HOP / SR
    notes, i = [], 0
    while i < len(midi):
        if np.isnan(midi[i]):
            i += 1
            continue
        p = int(round(midi[i]))
        j = i
        # extend while the pitch stays within a semitone of this note (tolerates vibrato and short dropouts)
        gap = 0
        while j + 1 < len(midi) and gap < 3:
            j += 1
            if np.isnan(midi[j]):
                gap += 1
            elif abs(midi[j] - p) <= 0.9:
                gap = 0
            else:
                j -= 1
                break
        end = j + 1
        if (end - i) * dt >= min_len:
            seg = midi[i:end]
            notes.append((i * dt, end * dt, int(round(np.nanmedian(seg))), float(loud[i:end].mean())))
        i = max(end, i + 1)
    return notes


def drum_hits(path):
    y = mono(path)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, backtrack=False, delta=0.2)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    bands = {"low": (30, 150), "mid": (150, 2500), "high": (5000, 8000)}
    hits = []
    top = env.max() or 1
    for f in frames:
        w = S[:, f:f + 4].sum(axis=1)
        e = {k: w[(freqs >= a) & (freqs < b)].sum() for k, (a, b) in bands.items()}
        tot = sum(e.values()) or 1
        low, mid, high = e["low"] / tot, e["mid"] / tot, e["high"] / tot
        pitch = 36 if low > 0.5 else 42 if high > 0.3 and low < 0.25 else 38  # kick / closed hat / snare
        hits.append((f * HOP / SR, pitch, float(env[f] / top)))
    return hits


def main():
    sid = sys.argv[1]
    out = Path("out") / sid
    data = json.load(open(out / "analysis.json"))
    mid = pm.PrettyMIDI(initial_tempo=120)
    counts = {}

    def add(name, program, notes, is_drum=False):
        inst = pm.Instrument(program=program, is_drum=is_drum, name=name)
        for s, e, p, v in notes:
            inst.notes.append(pm.Note(velocity=int(40 + 80 * min(1.0, v)), pitch=int(p), start=float(s), end=float(max(e, s + 0.05))))
        if inst.notes:
            mid.instruments.append(inst)
        counts[name] = len(inst.notes)

    print("bass...", flush=True)
    add("Bass", 33, pyin_notes(out / "stems" / "bass.wav", "E1", "G4"))
    print("vocals...", flush=True)
    add("Vocals", 54, pyin_notes(out / "stems" / "vocals.wav", "C3", "C6", min_len=0.1))
    print("drums...", flush=True)
    add("Drums", 0, [(t, t + 0.1, p, v) for t, p, v in drum_hits(out / "stems" / "drums.wav")], is_drum=True)
    for side in ("guitar_L", "guitar_R"):
        if side in data["notes"]:
            add(side.replace("_", " "), 30, [(s, e, p, 0.7) for s, e, p in data["notes"][side]])
    if "guitar_L" not in data["notes"] and "guitar" in data["notes"]:
        add("Guitar", 30, [(s, e, p, 0.7) for s, e, p in data["notes"]["guitar"]])

    path = out / f"{sid}.mid"
    mid.write(str(path))
    print(f"wrote {path}")
    for k, v in counts.items():
        print(f"  {k}: {v} notes")


if __name__ == "__main__":
    main()
