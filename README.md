# Song splitter

Split a song into stems, transcribe them, and build an orchestral arrangement on top (two E/A/D/G violin
sections, cello, double bass, piano) that plays in a browser stem player next to the original.

## Setup (Windows)

Python 3.12 environment at a short path (torch's CUDA DLLs break on long paths):

```bash
python -m venv C:/Users/itsju/.venvs/stems
P=C:/Users/itsju/.venvs/stems/Scripts/python.exe
$P -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$P -m pip install demucs soundfile librosa pretty_midi mir_eval resampy scipy onnxruntime
$P -m pip install --no-deps basic-pitch   # its TensorFlow pin doesn't install on 3.12; it uses the ONNX model
```

Rendering needs, in `tools/` (not committed):
- FluidSynth 2.6.1 win10-x64 from github.com/FluidSynth/fluidsynth/releases, unzipped into `tools/fluidsynth/`
- `MuseScore_General.sf3` (MIT) from ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/, in `tools/soundfonts/`

ffmpeg on PATH is used for MP3 export.

## Player

```bash
C:/Users/itsju/.venvs/stems/Scripts/python.exe -u server.py
```

Open http://127.0.0.1:8765/player.html, pick a song from the Music folder and press **Split this song**
(Demucs stems, guitar split by pan position, Basic Pitch transcription).

## Building the arrangement for a split song

This happens automatically: after a split the server runs `build_arrangement.py`, which builds the layers in
its `DEFAULT_LAYERS` (currently just the tuned percussion), and the player loads them when they're ready. Opening
a song that was split earlier starts its build too. For more layers:

```bash
C:/Users/itsju/.venvs/stems/Scripts/python.exe build_arrangement.py <id> violins bass keys perc   # or: all
```

The steps it runs, with the song id from the player's URL (`?song=<id>`):

| Script | Makes |
|---|---|
| `export_midi.py <id>` | multitrack MIDI (bass and vocals via pYIN, drums, guitars) |
| `arrange_strings.py <id>` | `strings.json`, single-line violins, cello/bass (first version) |
| `violin_section.py <id> guitar_L` / `guitar_R` | an E/A/D/G violin section per guitar: textures full / follow, smoothing 0-4, arpeggios in quiet picked passages |
| `bass_lines.py <id>` | cello + double bass at smoothing 0-4 |
| `piano_layer.py <id>` | keys: chords on the kick (or guitar / drums / bass), smoothing 0-4 |
| `percussion_layer.py <id>` | tuned percussion (marimba / steel drums / kalimba) from the drum hits on the chords, busyness 0-4 |
| `rank_violins.py <id> guitar_L` | optional: one violin per note rank (not used by default) |

Everything is pre-rendered; the player's sliders and buttons swap between the rendered files live.
