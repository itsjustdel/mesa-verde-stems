"""Local server for player.html: lists songs, runs analyze.py on request, serves files.

Usage: <venv python> server.py [music_dir] [port]
  music_dir defaults to ~/Music, port to 8765. Binds to 127.0.0.1 only.

API
  GET  /api/songs           every audio file under music_dir, with split status
  GET  /api/original?id=    the untouched source file
  POST /api/split?id=<id>   copy the song to in/<id><ext>, run analyze.py on it, then build_arrangement.py
                            (one job at a time; the stems are playable while the arrangement builds)
  POST /api/arrange?id=     build the arrangement for a song split before that happened automatically
  POST /api/export          {id, parts: [{path, gain}], label} or {id, original: true}: an MP3 of the mix,
                            kept in out/<id>/exports
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parent
MUSIC = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Music"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}

# id -> {"state": "queued"|"running"|"arranging"|"done"|"error", "log": str}
# running = splitting (analyze.py); arranging = building the arrangement layers (build_arrangement.py)
jobs = {}
queue_lock = threading.Lock()


def song_id(path):
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    if len(slug) > 40:  # keep the end (track number + title), dropping the cut-off first word
        slug = slug[-40:].split("-", 1)[-1]
    return f"{slug}-{hashlib.sha1(str(path).encode()).hexdigest()[:6]}"


def scan():
    songs = {}
    for p in sorted(MUSIC.rglob("*")):
        if p.suffix.lower() in EXTS and p.is_file():
            sid = song_id(p)
            songs[sid] = {"id": sid, "title": p.stem, "folder": str(p.parent.relative_to(MUSIC)) or ".",
                          "path": str(p), "ext": p.suffix.lower()}
    return songs


def status(s):
    """The split's status: the stems are playable once analysis.json exists, even while the arrangement builds."""
    state = jobs.get(s["id"], {}).get("state", "new")
    if (ROOT / "out" / s["id"] / "analysis.json").exists() and state not in ("queued", "running"):
        return "done"
    return state


def run_step(sid, args):
    """Run one pipeline script, keeping the tail of its output as the job's log. True if it succeeded."""
    proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    for line in proc.stdout:
        if "WARNING:root" not in line and "Warning" not in line:
            jobs[sid]["log"] = (jobs[sid]["log"] + line)[-2000:]
    return proc.wait() == 0


def run_split(s):
    """Split the song, then build its arrangement layers (build_arrangement.py's default ones)."""
    sid = s["id"]
    src = ROOT / "in" / f"{sid}{s['ext']}"
    with queue_lock:  # one job at a time; the GPU and RAM can't take more
        jobs[sid] = {"state": "running", "log": ""}
        try:
            src.parent.mkdir(exist_ok=True)
            if not src.exists():
                shutil.copy2(s["path"], src)
            if not run_step(sid, ["analyze.py", str(src), str(ROOT / "out" / sid)]):
                jobs[sid]["state"] = "error"
                return
            jobs[sid]["state"] = "arranging"
            jobs[sid]["state"] = "done" if run_step(sid, ["build_arrangement.py", sid]) else "arrange_error"
        except Exception as e:  # surface it in the player rather than killing the thread silently
            jobs[sid] = {"state": "error", "log": repr(e)}


def run_arrange(s):
    """Build the arrangement for a song that was split before this happened automatically."""
    sid = s["id"]
    with queue_lock:
        jobs[sid] = {"state": "arranging", "log": ""}
        try:
            jobs[sid]["state"] = "done" if run_step(sid, ["build_arrangement.py", sid]) else "arrange_error"
        except Exception as e:
            jobs[sid] = {"state": "arrange_error", "log": repr(e)}


def export_mp3(req):
    """Mix the parts the player says are audible (each a WAV inside this song's out folder, at its volume) into
    one MP3 in out/<id>/exports, or transcode the untouched original."""
    s = scan().get(str(req.get("id", "")))
    if not s:
        raise ValueError("unknown song")
    song_dir = (ROOT / "out" / s["id"]).resolve()
    exports = song_dir / "exports"
    exports.mkdir(exist_ok=True)
    label = re.sub(r"[^\w ,+-]", "", str(req.get("label", "mix")))[:60].strip() or "mix"
    title = re.sub(r"[^\w ,()-]", "", s["title"])[:80].strip()
    out = exports / f"{title} - {label} - {time.strftime('%H%M%S')}.mp3"
    if req.get("original"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", s["path"], "-map", "0:a", "-b:a", "256k", str(out)]
    else:
        parts = req.get("parts") or []
        if not 0 < len(parts) <= 64:
            raise ValueError("nothing to export")
        inputs, chains = [], []
        for i, p in enumerate(parts):
            path = (ROOT / str(p.get("path", ""))).resolve()
            if song_dir not in path.parents or path.suffix != ".wav" or not path.is_file():
                raise ValueError(f"not a part of this song: {p.get('path')}")
            inputs += ["-i", str(path)]
            chains.append(f"[{i}:a]volume={min(2.0, max(0.0, float(p.get('gain', 1)))):.3f}[a{i}]")
        # sum the parts as the player does (no auto-scaling), with a limiter at -1 dB so the sum can't clip,
        # leaving room for the MP3 encoder's small overshoots
        graph = (";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(parts)))
                 + f"amix=inputs={len(parts)}:normalize=0:duration=longest,alimiter=limit=0.89:level=0[out]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
               "-b:a", "256k", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return {"url": "/" + quote(out.relative_to(ROOT).as_posix()), "name": out.name, "path": str(out)}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # stems get regenerated under the same names; never let the browser reuse an old copy
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/songs":
            songs = scan()
            return self.send_json([{**s, "status": status(s), "log": jobs.get(s["id"], {}).get("log", ""),
                                    "arranging": jobs.get(s["id"], {}).get("state") in ("arranging", "arrange_queued"),
                                    "arrange_failed": jobs.get(s["id"], {}).get("state") == "arrange_error"}
                                   for s in songs.values()])
        if url.path == "/api/original":
            # the untouched file straight from the music folder, so you hear exactly what you picked
            s = scan().get(parse_qs(url.query).get("id", [""])[0])
            if not s:
                return self.send_error(404)
            path = Path(s["path"])
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
            return
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/split":
            sid = parse_qs(url.query).get("id", [""])[0]
            s = scan().get(sid)
            if not s:
                return self.send_json({"error": "unknown song"}, 404)
            if status(s) not in ("new", "error"):
                return self.send_json({"status": status(s)})
            jobs[sid] = {"state": "queued", "log": ""}
            threading.Thread(target=run_split, args=(s,), daemon=True).start()
            return self.send_json({"status": "queued"})
        if url.path == "/api/arrange":
            s = scan().get(parse_qs(url.query).get("id", [""])[0])
            if not s or status(s) != "done":
                return self.send_json({"error": "song not split"}, 404)
            if jobs.get(s["id"], {}).get("state") in ("queued", "running", "arranging", "arrange_queued"):
                return self.send_json({"status": "arranging"})
            jobs[s["id"]] = {"state": "arrange_queued", "log": ""}
            threading.Thread(target=run_arrange, args=(s,), daemon=True).start()
            return self.send_json({"status": "arranging"})
        if url.path == "/api/export":
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self.send_json(export_mp3(req))
            except (ValueError, TypeError) as e:
                return self.send_json({"error": str(e)}, 400)
            except subprocess.CalledProcessError as e:
                return self.send_json({"error": "ffmpeg failed: " + (e.stderr or "")[-300:]}, 500)
        self.send_error(404)

    def log_message(self, fmt, *args):
        if "/api/songs" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"Music: {MUSIC}\nPlayer: http://127.0.0.1:{PORT}/player.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
