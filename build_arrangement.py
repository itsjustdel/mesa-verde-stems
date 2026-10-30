"""Build the arrangement layers for a split song.

Usage: python build_arrangement.py <song_id> [layer ...]
  layers: perc (tuned percussion), violins (an E/A/D/G section per guitar), bass (cello + double bass), keys,
  or all. Default: DEFAULT_LAYERS. The MIDI step (bass line for the chord roots) always runs first.
Runs after analyze.py (the server starts it automatically after every split). Songs whose guitars weren't split
into L/R get one violin section from the combined guitar. Stops at the first step that fails.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAYERS = ("violins", "bass", "keys", "perc")
DEFAULT_LAYERS = ("perc",)  # what a split builds on its own; ask for more by name


def main():
    sid = sys.argv[1]
    wanted = set(sys.argv[2:]) or set(DEFAULT_LAYERS)
    if "all" in wanted:
        wanted = set(LAYERS)
    base = ROOT / "out" / sid
    strings = base / "strings"
    strings.mkdir(exist_ok=True)
    if not (strings / "strings.json").exists():  # the parts list the player reads; each step adds its parts
        json.dump({"parts": [], "smooth": {"parts": [], "levels": 5, "default": 2}}, open(strings / "strings.json", "w"))

    sides = ("guitar_L", "guitar_R") if (base / "stems" / "guitar_L.wav").exists() else ("both",)
    steps = [["export_midi.py", sid]]
    if "violins" in wanted:
        steps += [["violin_section.py", sid, side] for side in sides]
    if "bass" in wanted:
        steps.append(["bass_lines.py", sid])
    if "keys" in wanted:
        steps.append(["piano_layer.py", sid])
    if "perc" in wanted:
        steps.append(["percussion_layer.py", sid])
    for step in steps:
        print(f"== {step[0]} {' '.join(step[2:])}", flush=True)
        if subprocess.run([sys.executable, *step], cwd=ROOT).returncode:
            sys.exit(f"{step[0]} failed")
    print("arrangement done", flush=True)


if __name__ == "__main__":
    main()
