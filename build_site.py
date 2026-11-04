"""Put split songs on the band site (docs/, served by GitHub Pages).

Usage: python build_site.py <song_id> [<song_id> ...]     (ids as in out/, e.g. 05-no-more-bad-future)
       python build_site.py --all                         every song in out/ with stems

For each song: its stems as 192k MP3s in docs/songs/<id>/, and an entry in docs/songs.json (title, track number,
stems). The Demucs "piano" stem is mostly bleed on these songs, so it is folded into "other"; when the guitars
were split by pan position, guitar_L / guitar_R replace the combined guitar. So all the stems still add up to
the whole song. Songs already on the site are re-encoded; other entries in songs.json are kept.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "docs"
BITRATE = "192k"
NAMES = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "guitar": "Guitar", "guitar_L": "Guitar (left)",
         "guitar_R": "Guitar (right)", "other": "Other"}
ORDER = list(NAMES)


def source_of(sid):
    """The song file this id came from, for its title and track number."""
    for p in (ROOT / "music").rglob("*.flac"):
        if p.stem.split(" - ")[-1] and re.sub(r"[^a-z0-9]+", "-", p.stem.split(" - ")[-1].lower()).strip("-") == sid:
            return p
    return None


def tags(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags=title,track,album",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    fmt = json.loads(out)["format"]
    t = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    return t, float(fmt["duration"])


def encode(inputs, dest):
    """One MP3 from one WAV, or from several summed (no scaling, so the stems still add up to the song)."""
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for i in inputs:
        args += ["-i", str(i)]
    if len(inputs) > 1:
        args += ["-filter_complex", f"amix=inputs={len(inputs)}:normalize=0:duration=longest"]
    subprocess.run(args + ["-c:a", "libmp3lame", "-b:a", BITRATE, str(dest)], check=True)


def build(sid):
    stems = ROOT / "out" / sid / "stems"
    if not (stems / "bass.wav").exists():
        raise SystemExit(f"{sid}: not split (no {stems / 'bass.wav'})")
    have = {p.stem for p in stems.glob("*.wav")}
    parts = {s: [stems / f"{s}.wav"] for s in ("vocals", "drums", "bass") if s in have}
    if {"guitar_L", "guitar_R"} <= have:
        parts["guitar_L"], parts["guitar_R"] = [stems / "guitar_L.wav"], [stems / "guitar_R.wav"]
    elif "guitar" in have:
        parts["guitar"] = [stems / "guitar.wav"]
    parts["other"] = [stems / f"{s}.wav" for s in ("other", "piano") if s in have]
    dest = SITE / "songs" / sid
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.mp3"):
        old.unlink()
    for s, inputs in parts.items():
        encode(inputs, dest / f"{s}.mp3")
    src = source_of(sid)
    t, dur = tags(src) if src else ({}, 0.0)
    track = int(re.match(r"\d+", t.get("track", "0")).group()) if re.match(r"\d+", t.get("track", "")) else 0
    if not track and re.match(r"\d+", sid):
        track = int(re.match(r"\d+", sid).group())
    title = t.get("title") or sid
    entry = {"id": sid, "title": title, "track": track, "duration": round(dur, 2),
             "stems": [{"id": s, "name": NAMES[s], "file": f"songs/{sid}/{s}.mp3"} for s in ORDER if s in parts]}
    size = sum(f.stat().st_size for f in dest.glob("*.mp3")) / 1e6
    print(f"{sid}: {title}, {len(parts)} stems, {size:.0f} MB")
    return entry


def main():
    ids = sys.argv[1:]
    if ids == ["--all"]:
        ids = sorted(p.parent.parent.name for p in (ROOT / "out").glob("*/stems/bass.wav"))
    if not ids:
        raise SystemExit(__doc__)
    index = SITE / "songs.json"
    songs = {s["id"]: s for s in json.loads(index.read_text())} if index.exists() else {}
    for sid in ids:
        songs[sid] = build(sid)
    index.write_text(json.dumps(sorted(songs.values(), key=lambda s: (s["track"], s["title"])), indent=1))
    cover = next((ROOT / "music").rglob("cover.jpg"), None)
    if cover and not (SITE / "cover.jpg").exists():
        (SITE / "cover.jpg").write_bytes(cover.read_bytes())
    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1e6
    print(f"site: {len(songs)} songs, {total:.0f} MB in {SITE}")


if __name__ == "__main__":
    main()
