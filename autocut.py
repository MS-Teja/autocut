#!/usr/bin/env python3
"""
AutoCut analyzer for talking-head footage.

Finds the speech in a clip, throws away silences and stray noise blips
(the single-frame clicks Resolve's own silence cut leaves behind), and
groups repeated / restarted sentences into "take groups" so you can pick
the best one.

It writes <video>.autocut.fcpxml next to the video: one timeline with the silences
cut and repeats raised onto V2/V3. Import it into Resolve with File > Import >
Timeline. <video>.autocut.json holds the result and the cached transcript.

Usage:
  autocut clip.mp4                 # analyze one clip
  autocut ~/Footage/episode12/     # every video in a folder
  autocut clip.mp4 --pace tight    # tighter jump-cut pacing (re-uses transcript)
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path

import numpy as np

VERSION = 1
TRANSCRIPT_METHOD = "per-region-2"  # transcripts cached by other methods are missing repeated lines
SR = 16000
FPS_E = 100  # energy frames per second (10 ms hop)
HOP = SR // FPS_E

# pace -> (min silence that gets cut, padding before speech, padding after speech)
PACES = {
    "tight": (0.25, 0.0, 0.0),  # cut right at the speech; frame rounding still never clips it
    "natural": (0.35, 0.08, 0.12),
    "loose": (0.60, 0.12, 0.20),
}

# Measured on hand-labeled real footage; see "Tuning" in CLAUDE.md before changing these.
TEXT_MATCH = 0.75       # openings spelled this alike are the same line
MIXED_MATCH = 0.6       # ...or this alike, if they also sound alike
SOUND_MATCH = 2.0       # voice distance that backs up a mixed match
SOUND_ONLY = 1.75       # neighbours this close match on sound alone (for garbled transcripts)
SANDWICH_MAX = 5.0      # shorter clips between two takes of a line count as attempts at it
SELF_REPEAT = 1.93      # voice distance at which a clip may be repeating its own opening
SELF_REPEAT_TEXT = 0.7  # ...confirmed if its two halves, heard apart, open this alike
CLEAR_TEXT = 0.7        # Whisper's mean word confidence below which a clip's text is treated as garbled
FEATURE_HOP = 4         # 40 ms per voice feature frame
FEATURE_FPS = SR / (160 * FEATURE_HOP)
VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".m4v", ".avi", ".mts", ".m2ts", ".wav", ".mp3", ".m4a", ".aac", ".flac"}


# --------------------------------------------------------------------------- terminal

TTY = sys.stdout.isatty()
COLOR = TTY and not os.environ.get("NO_COLOR")


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def log(msg: str = "") -> None:
    if TTY:
        sys.stdout.write("\r\033[K")  # clear any status or bar line first
    print(msg, flush=True)


def section(title: str) -> None:
    log(f"{paint('::', '1;34')} {paint(title, '1')}")


def step(label: str, detail: str, ok: bool = True) -> None:
    log(f"   {paint('✓', '32') if ok else paint('✗', '31')} {label:<11} {detail}")


def status(msg: str) -> None:
    """A passing note ("loading Whisper…"), shown only in a terminal and replaced by the next line."""
    if TTY:
        sys.stdout.write(f"\r\033[K   {paint(msg, '2')}")
        sys.stdout.flush()


def clock(sec: float) -> str:
    m, s = divmod(int(round(max(0.0, sec))), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def took(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.1f} {unit}" if unit in ("MB", "GB") else f"{n:.0f} {unit}"
        n /= 1000


def candy(fraction: float, width: int = 30, chomp: int = 0) -> str:
    """A progress bar the way pacman draws it with ILoveCandy: Pac-Man eats a row of pellets."""
    fraction = min(max(fraction, 0.0), 1.0)
    at = round(fraction * (width - 1))
    cells = []
    for i in range(width):
        if i < at:
            cells.append(paint("-", "2"))
        elif i == at:
            cells.append(paint("c" if chomp % 2 or fraction >= 1 else "C", "1;33"))
        else:
            cells.append("o" if (i - at) % 3 == 0 else " ")
    return "[" + "".join(cells) + "]"


class Progress:
    """One pacman bar, redrawn in place up to 10 times a second. Terminal only."""

    def __init__(self, label: str, total: float, unit: str = "s"):
        self.label, self.total, self.unit = label, total, unit
        self.start, self.drawn, self.chomp, self.done = time.time(), 0.0, 0, 0.0

    def update(self, done: float, total: float | None = None, force: bool = False) -> None:
        self.done, self.total = done, total if total else self.total
        now = time.time()
        if not TTY or (now - self.drawn < 0.1 and not force):
            return
        self.drawn, self.chomp = now, self.chomp + 1
        fraction = done / self.total if self.total else 0.0
        if self.unit == "B":
            rate = done / max(now - self.start, 1e-6)
            extra = f"{size(done)}/{size(self.total)}  {size(rate)}/s"
        else:
            rate = done / max(now - self.start, 1e-6)
            extra = f"{100 * fraction:3.0f}%"
        if 0 < fraction < 1 and rate > 0:
            extra += f"  {clock((self.total - done) / rate)} left"
        sys.stdout.write(f"\r\033[K   {self.label:<12} {candy(fraction, 30, self.chomp)}  {extra}")
        sys.stdout.flush()

    def finish(self) -> None:
        if TTY and self.drawn:
            self.update(self.total, force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()


# --------------------------------------------------------------------------- audio

def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for cand in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("ffmpeg not found. Run install.sh again (it installs a bundled ffmpeg).")


def load_audio(path: str, track: int = 0) -> np.ndarray:
    cmd = [find_ffmpeg(), "-nostdin", "-v", "error", "-i", path, "-map", f"0:a:{track}",
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    audio = np.frombuffer(res.stdout, np.float32).copy()
    if audio.size < SR // 10:
        raise RuntimeError("no usable audio track found")
    return audio


def normalize(audio: np.ndarray) -> np.ndarray:
    """Gain-normalize for detection only (output media is untouched)."""
    ref = float(np.percentile(np.abs(audio), 99.9))
    if ref < 1e-6:
        return audio
    return np.clip(audio * min(0.5 / ref, 30.0), -1.0, 1.0).astype(np.float32)


def frame_db(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // HOP
    fr = audio[: n * HOP].reshape(n, HOP)
    return 20.0 * np.log10(np.sqrt(np.mean(fr * fr, axis=1) + 1e-12))


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


# --------------------------------------------------------------------------- speech detection

def detect_speech(audio: np.ndarray, min_silence: float, threshold: float) -> list[list[float]]:
    """Neural VAD (Silero, bundled with faster-whisper): ignores clicks, bumps and room noise."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # min_speech is applied by the caller so dropped blips can be counted
    opts = VadOptions(threshold=threshold, min_speech_duration_ms=0,
                      min_silence_duration_ms=int(min_silence * 1000), speech_pad_ms=0)
    return [[t["start"] / SR, t["end"] / SR] for t in get_speech_timestamps(audio, opts)]


def tighten(regions, db, floor_db, margin_db=10.0, min_island=0.12, island_gap=0.10):
    """Snap region edges to where the audio is actually loud, and strip short
    noise islands stuck to an edge (e.g. a click 0.2 s after the last word)."""
    loud = db > (floor_db + margin_db)
    isl, gap = int(min_island * FPS_E), int(island_gap * FPS_E)
    out = []
    for s, e in regions:
        a, b = max(0, int(s * FPS_E)), min(len(loud), int(math.ceil(e * FPS_E)))
        runs = true_runs(loud[a:b])
        if not runs:
            out.append([s, e])
            continue
        while len(runs) > 1 and runs[0][1] - runs[0][0] < isl and runs[1][0] - runs[0][1] >= gap:
            runs.pop(0)
        while len(runs) > 1 and runs[-1][1] - runs[-1][0] < isl and runs[-1][0] - runs[-2][1] >= gap:
            runs.pop()
        out.append([max(s, (a + runs[0][0]) / FPS_E), min(e, (a + runs[-1][1]) / FPS_E)])
    return out


def voice_onset(db, start, earliest, rise_db=6.0, look=0.15):
    """Where the voice really starts. Silero marks speech up to ~140 ms late on soft first sounds
    ("Sh", "S", "H") and after long pauses, which clipped the first sound of a fifth of the clips
    once tight pace had no padding. Walk back while the level stays well above the quietest moment
    just before; a breath followed by a dip stays out."""
    k = min(int(round(start * FPS_E)), len(db))
    window = max(0, k - int(look * FPS_E))
    a = max(int(math.ceil(earliest * FPS_E)), window)
    if k <= a:
        return start
    base = db[window:k].min()  # the quiet comes from the whole window, even where the cut can't go
    j = k
    while j > a and db[j - 1] > base + rise_db:
        j -= 1
    return min(start, j / FPS_E)


# --------------------------------------------------------------------------- comparing voices

def voice_features(audio: np.ndarray) -> np.ndarray:
    """One 13-number description of the voice per frame.
    The same speaker reading the same words looks alike here even when Whisper garbles them."""
    from faster_whisper.feature_extractor import FeatureExtractor

    mel = FeatureExtractor()(audio, padding=0)[:, ::FEATURE_HOP]
    n = mel.shape[0]
    dct = np.cos(np.pi / n * (np.arange(n) + 0.5)[None, :] * np.arange(1, 14)[:, None])
    f = (dct @ mel).T
    energy = mel.mean(0)  # normalize over speech only, so distances don't depend on how much silence a file has
    low, high = np.percentile(energy, [10, 90])
    speech = f[energy > (low + high) / 2]
    return (f - speech.mean(0)) / (speech.std(0) + 1e-6)


def voice_distance(a: np.ndarray, b: np.ndarray) -> float:
    """How far all of a is from the opening of b, per frame, allowing the speaking rate
    to differ by up to 2:1. Infinite when there's too little audio to compare."""
    n, m = len(a), len(b)
    if n < 3 or m < 3:
        return math.inf
    dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    before, last = np.full(m, np.inf), np.full(m, np.inf)
    last[0] = dist[0, 0]
    for i in range(1, n):
        cur = np.full(m, np.inf)
        cur[1:] = last[:-1]
        cur[2:] = np.minimum(cur[2:], last[:-2])
        cur[1:] = np.minimum(cur[1:], before[:-1])
        cur += dist[i]
        before, last = last, cur
    return float(last.min() / n)


def self_repeat(features: np.ndarray, start: float, end: float, opening=0.8, min_lag=0.6) -> float | None:
    """Seconds into start..end where the speech sounds like it repeats its own opening, if anywhere."""
    a, b = int(start * FEATURE_FPS), int(end * FEATURE_FPS)
    k, lag = int(opening * FEATURE_FPS), int(min_lag * FEATURE_FPS)
    if b - a < k + lag + k // 2:
        return None
    head = features[a:a + k]
    costs = {at: voice_distance(head, features[at:at + 2 * k]) for at in range(a + lag, b - k // 2, 2)}
    at = min(costs, key=costs.get)
    return (at - a) / FEATURE_FPS if costs[at] <= SELF_REPEAT else None


# --------------------------------------------------------------------------- transcription

_loaded_models: dict = {}


def candy_tqdm():
    """huggingface_hub reports download progress through a tqdm class: draw its byte count as
    the pacman bar, and skip its other bars (files fetched, reconstruction)."""
    from tqdm.std import tqdm

    class CandyTqdm(tqdm):
        drawing: list = []  # huggingface_hub leaves its bar open after the download; closed by the caller

        def __init__(self, *args, **kwargs):
            self._bar = None
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            desc = (self.desc or "").lower()
            if self.unit == "B" and "reconstruct" not in desc:
                self._bar = Progress("downloading", self.total or 0, unit="B")
                CandyTqdm.drawing.append(self)

        def display(self, msg=None, pos=None):
            if self._bar:
                self._bar.update(self.n, self.total)

        def close(self):
            if self._bar:
                self._bar.finish()
                self._bar = None
                CandyTqdm.drawing.remove(self)
            self.disable = True  # tqdm's own close would print its bar
            super().close()

    return CandyTqdm


def load_whisper(name: str, compute_type: str):
    """The Whisper model, loaded once per run. faster-whisper downloads silently (a first download
    can take most of an hour), so a missing model is downloaded here with a visible bar."""
    if (name, compute_type) in _loaded_models:
        return _loaded_models[(name, compute_type)]
    from faster_whisper import WhisperModel

    path = name
    try:
        from faster_whisper.utils import _MODELS, download_model
    except ImportError:
        _MODELS = {}
    if name in _MODELS:
        try:
            path = download_model(name, local_files_only=True)
        except Exception:
            from huggingface_hub import snapshot_download

            section(f"Downloading Whisper {name} (one time)")
            bars = candy_tqdm()
            path = snapshot_download(_MODELS[name], tqdm_class=bars, allow_patterns=[
                "config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"])
            for bar in list(bars.drawing):
                bar.close()
            step("Whisper", f"{name} ready")
    status(f"loading Whisper {name}…")
    _loaded_models[(name, compute_type)] = WhisperModel(path, device="auto", compute_type=compute_type)
    return _loaded_models[(name, compute_type)]


def transcribe(audio, model_name, language, compute_type, beam_size, duration):
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    model = load_whisper(model_name, compute_type)
    # One call per pause-separated region. Handed several at once (vad_filter),
    # Whisper reads them as one passage and silently drops a re-read line, which
    # is exactly the repeated take we want to see.
    pad = 200
    regions = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=pad))
    if regions and not language:
        # From all the speech: one short region can guess wrong (a few seconds of English
        # came back as Danish). The generator isn't consumed, so this only detects.
        status("detecting the language…")
        speech = np.concatenate([audio[r["start"]:r["end"]] for r in regions])
        language = model.transcribe(speech)[1].language
    features = voice_features(audio)
    restart_times = []

    def hear(a: int, b: int) -> list[dict]:
        segments, _ = model.transcribe(
            audio[a:b],
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,  # same reason: it also skips what it just heard
            beam_size=beam_size,
        )
        heard = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    heard.append({"start": round(a / SR + float(w.start), 3), "end": round(a / SR + float(w.end), 3),
                                  "text": text, "p": round(float(w.probability), 3)})
        return heard

    def hear_attempts(a: int, b: int) -> list[dict]:
        # A line restarted without a pause comes back from Whisper as one read. If the audio
        # sounds like it repeats its opening, hear the two halves apart and compare.
        margin = int(pad / 1000 * SR)
        lag = self_repeat(features, (a + margin) / SR, (b - margin) / SR)
        if lag is None:
            return hear(a, b)
        at = a + margin + int(lag * SR)
        near = max(a, at - SR // 7)
        at = near + int(np.argmin(frame_db(audio[near:at + SR // 7]))) * HOP  # cut in the quietest spot
        first, second = hear(a, at), hear(at, b)
        if opens_alike(norm_words(" ".join(w["text"] for w in first)), norm_words(" ".join(w["text"] for w in second))):
            restart_times.append(at / SR)
            return first + hear_attempts(at, b)
        return hear(a, b)

    words, bar = [], Progress("transcribing", duration)
    bar.update(0, force=True)
    for r in regions:
        words += hear_attempts(r["start"], r["end"])
        bar.update(min(duration, r["end"] / SR))
    bar.finish()
    return words, language, restart_times


# --------------------------------------------------------------------------- text helpers

def norm_words(text: str) -> list[str]:
    """Lowercase, strip punctuation; keeps combining marks so Indic scripts work."""
    out = []
    for tok in text.lower().split():
        tok = "".join(ch for ch in tok if unicodedata.category(ch)[0] not in "PS")
        if tok:
            out.append(tok)
    return out


def opening_similarity(a: list[str], b: list[str]) -> float:
    """How alike do the openings of two lines look, by rough sound? The shorter against the same
    length of the longer, so a false start matches its full take. Vowels are nearly noise in
    Whisper's guesses, so they're merged: "on the comer" and "on the corner" stay close."""
    x, y = (re.sub(r"[aeiouy]+", "a", "".join(w)) for w in (a, b))
    if not x or not y:
        return 0.0
    if len(x) > len(y):
        x, y = y, x
    return SequenceMatcher(None, x, y[:len(x)], autojunk=False).ratio()


def same_line(a: list[str], b: list[str], distance: float, adjacent: bool) -> bool:
    """Are these two clips attempts at the same line? Every clip stays on the timeline, so a false
    match only raises a clip, while a miss hides a repeat. This leans toward catching repeats."""
    short = min(len(a), len(b))
    if short < 2 and not adjacent:  # a word or less is only evidence right next door
        return False
    similarity = opening_similarity(a, b)
    if similarity >= (TEXT_MATCH if short >= 2 else 0.99):
        return True
    if similarity >= MIXED_MATCH and distance <= SOUND_MATCH:
        return True
    return adjacent and distance <= SOUND_ONLY


def opens_alike(first: list[str], second: list[str]) -> bool:
    """Does the second attempt open with the same words as the first? Only the first 3 words count:
    over 6, parallel sentences passed on their shared middle ("the red team playing on the right" /
    "...the team playing on the left")."""
    n = min(len(first), len(second), 3)
    return bool(n) and opening_similarity(first[:n], second[:n]) >= SELF_REPEAT_TEXT


def group_lines(full: list[bool], match, window: int) -> dict[int, int]:
    """Which clips are attempts at the same line, as {clip: first clip of its line} for grouped clips.
    Lines are built from full takes only. A fragment, or a clip Whisper wasn't sure of, then joins
    one line (the next it matches, else the previous), so it can't chain two lines together: the
    false start "and after that" matched both "after that break we saw..." and "and after the
    game...". match(i, j, adjacent) says whether clips i < j look like the same line."""
    n = len(full)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    fulls = [k for k in range(n) if full[k]]
    for a, i in enumerate(fulls):
        for j in fulls[a + 1:a + 1 + window]:
            if match(i, j, all(not full[k] for k in range(i + 1, j))):  # neighbours if only fragments between
                parent[find(j)] = find(i)
    for k in range(n):
        if full[k]:
            continue
        later = [j for j in range(k + 1, min(n, k + 1 + window)) if match(k, j, j == k + 1)]
        earlier = [i for i in range(max(0, k - window), k) if match(i, k, k == i + 1)]
        for t in later[:1] or earlier[-1:]:  # a false start belongs to the line after it
            parent[find(k)] = find(t)

    members: dict[int, list[int]] = {}
    for k in range(n):
        members.setdefault(find(k), []).append(k)
    return {k: m[0] for m in members.values() if len(m) > 1 for k in m}


def restarts_at(before: list[str], after: list[str]) -> bool:
    """Do the words after a point repeat the opening or the tail of the words before it?"""
    for n in (3, 2):
        if len(before) >= n and len(after) >= n and after[:n] in (before[:n], before[-n:]):
            return True
    return bool(before and after) and after[0] == before[-1] and len(after[0]) >= 3  # "Tomorrow, tomorrow we"


def ts(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60)
    return f"{int(m):02d}:{s:05.2f}"


def split_restarts(clips: list[dict], words: list[dict], db: np.ndarray) -> list[dict]:
    """Split a clip where the speaker restarts mid-clip without a real pause
    ("ask what you— ask what you can do…"), cutting at the quietest 10 ms
    between the two words so each attempt becomes its own timeline clip."""
    wn = ["".join(norm_words(w["text"])) for w in words]
    out = []
    for c in clips:
        ws = c["w"]
        start = 0  # index into ws where the current piece begins
        cur = dict(c, w=[])
        for k in range(1, len(ws)):
            before = [wn[i] for i in ws[start:k] if wn[i]]
            after = [wn[i] for i in ws[k:k + 3] if wn[i]]
            if not restarts_at(before, after):
                continue
            lo, hi = words[ws[k - 1]]["end"] - 0.1, words[ws[k]]["start"] + 0.1
            a, b = max(0, int(lo * FPS_E)), min(len(db), int(math.ceil(hi * FPS_E)))
            cut_t = (a + int(np.argmin(db[a:b]))) / FPS_E if b > a else (lo + hi) / 2
            cut_t = min(max(cut_t, cur["s0"] + 0.05), c["e0"] - 0.05)
            cur["w"], cur["e0"] = ws[start:k], cut_t
            out.append(cur)
            cur = dict(c, s0=cut_t, w=[])
            start = k
        cur["w"] = ws[start:]
        out.append(cur)
    return out


# --------------------------------------------------------------------------- core

def analyze(audio: np.ndarray, words: list[dict] | None, args, restart_times=()) -> dict:
    duration = len(audio) / SR
    min_silence, pad_before, pad_after = PACES[args.pace]
    if args.min_silence is not None:
        min_silence = args.min_silence

    det = normalize(audio)
    db = frame_db(det)
    floor_db = float(np.percentile(db, 10))

    raw = detect_speech(det, min_silence, args.vad_threshold)
    # Edge tightening needs a real noise floor; on noisy/compressed audio skip it.
    spread = float(np.percentile(db, 90)) - floor_db
    tightened = tighten(raw, db, floor_db) if spread >= 20 else raw
    regions = [r for r in tightened if r[1] - r[0] >= args.min_speech]
    removed = [{"start": round(s, 3), "end": round(e, 3), "reason": "blip"}
               for s, e in raw if not any(rs < e and s < re_ for rs, re_ in regions)]

    # ---- cut where transcription found a line restarted without a pause
    cut_at = set()
    for t in restart_times:
        for k, (s, e) in enumerate(regions):
            if s + 0.1 < t < e - 0.1:
                regions[k:k + 1] = [[s, t], [t, e]]
                cut_at.add(t)
                break

    # ---- attach words to regions
    clips = [{"s0": s, "e0": e, "w": []} for s, e in regions]
    orphans = 0
    if words:
        starts = [c["s0"] for c in clips]
        for wi, w in enumerate(words):
            mid = (w["start"] + w["end"]) / 2
            k = bisect.bisect_right(starts, mid) - 1
            best, dist = None, 1e9
            for j in (k, k + 1):
                if 0 <= j < len(clips):
                    c = clips[j]
                    d = 0.0 if c["s0"] <= mid <= c["e0"] else min(abs(mid - c["s0"]), abs(mid - c["e0"]))
                    if d < dist:
                        best, dist = j, d
            if best is not None and dist <= 0.35:
                clips[best]["w"].append(wi)
            else:
                orphans += 1

        # Undo a restart cut where the two pieces don't open with the same words. Transcription
        # checks this too, but transcripts cached before the check got stricter would need redoing.
        said = lambda c: norm_words(" ".join(words[k]["text"] for k in c["w"]))
        joined = []
        for c in clips:
            prev = joined[-1] if joined else None
            if c["s0"] in cut_at and prev and prev["e0"] == c["s0"] and not opens_alike(said(prev), said(c)):
                prev["e0"], prev["w"] = c["e0"], prev["w"] + c["w"]
            else:
                joined.append(c)

        kept = []
        for c in joined:
            if not c["w"] and c["e0"] - c["s0"] < args.noise_max:
                removed.append({"start": round(c["s0"], 3), "end": round(c["e0"], 3), "reason": "noise (no words)"})
            else:
                kept.append(c)
        clips = split_restarts(kept, words, db)

    # ---- pad, never overlapping neighbours
    for i, c in enumerate(clips):
        lo = clips[i - 1]["e0"] if i else 0.0
        hi = clips[i + 1]["s0"] if i + 1 < len(clips) else duration
        earliest = (lo + c["s0"]) / 2 if i else 0.0
        c["s"] = max(voice_onset(db, c["s0"], earliest) - pad_before, earliest)
        c["e"] = min(c["e0"] + pad_after, (c["e0"] + hi) / 2 if i + 1 < len(clips) else duration)
        c["text"] = " ".join(words[k]["text"] for k in c["w"]) if words else ""
        c["nw"] = norm_words(c["text"])

    # ---- group attempts at the same line
    group_of: dict[int, int] = {}
    if words:
        features = voice_features(det)
        voice = [features[int(c["s0"] * FEATURE_FPS):max(int(c["e0"] * FEATURE_FPS), int(c["s0"] * FEATURE_FPS) + 1)]
                 for c in clips]
        confidence = [float(np.mean([words[k].get("p", 1.0) for k in c["w"]])) if c["w"] else 0.0 for c in clips]
        full = [len(c["nw"]) >= 4 and confidence[i] >= CLEAR_TEXT for i, c in enumerate(clips)]

        def match(i: int, j: int, adjacent: bool) -> bool:
            x, y = sorted((voice[i], voice[j]), key=len)
            distance = voice_distance(x[:150], y[:2 * min(len(x), 150) + 10])  # the first 6 s is plenty
            return same_line(clips[i]["nw"], clips[j]["nw"], distance, adjacent)

        group_of = group_lines(full, match, args.window)

    # a short clip between two takes of a line is a failed attempt, whatever Whisper made of it
    raised = sorted(group_of)
    for i, c in enumerate(clips):
        if i not in group_of and c["e0"] - c["s0"] <= SANDWICH_MAX:
            k = bisect.bisect_left(raised, i)
            if 0 < k < len(raised) and group_of[raised[k - 1]] == group_of[raised[k]]:
                group_of[i] = group_of[raised[k]]

    grouped: dict[int, list[int]] = {}
    for i in sorted(group_of):
        grouped.setdefault(group_of[i], []).append(i)
    groups = []
    for ids in sorted(grouped.values(), key=lambda m: m[0]):
        longest = max(len(clips[i]["nw"]) for i in ids)
        # pick = last attempt that is (nearly) complete; people re-read until it's right
        pick = next(i for i in reversed(ids) if len(clips[i]["nw"]) >= 0.8 * longest)
        groups.append({"id": len(groups), "clips": ids, "pick_take": ids.index(pick) + 1, "start": round(clips[ids[0]]["s0"], 3),
                       "texts": [" ".join(clips[i]["nw"]) for i in ids]})

    # ---- per-clip result
    placed = {i: (g, n) for g in groups for n, i in enumerate(g["clips"], 1)}
    out_clips = []
    for ci, c in enumerate(clips):
        role = "single" if c["w"] or not words else "no_words"
        item = {"id": ci, "start": round(c["s"], 3), "end": round(c["e"], 3),
                "speech_start": round(c["s0"], 3), "speech_end": round(c["e0"], 3),
                "text": c["text"], "role": role}
        label = c["text"] if role == "single" else "(no words detected - check)"
        if ci in placed:
            g, n = placed[ci]
            role = "pick" if n == g["pick_take"] else "take"
            item.update(role=role, group=g["id"], take=n, takes=len(g["clips"]))
            label = f"{'PICK' if role == 'pick' else 'take'} {n}/{len(g['clips'])} · {c['text']}"
        item["name"] = label[:80]
        out_clips.append(item)

    kept_time = sum(c["e"] - c["s"] for c in clips)
    sel_time = sum(clips[i]["e"] - clips[i]["s"] for i in range(len(clips)) if out_clips[i]["role"] != "take")
    return {
        "clips": out_clips,
        "groups": groups,
        "removed": sorted(removed, key=lambda r: r["start"]),
        "stats": {"duration": round(duration, 2), "kept": round(kept_time, 2), "selects": round(sel_time, 2),
                  "clips": len(clips), "groups": len(groups),
                  "blips_removed": sum(r["reason"] == "blip" for r in removed),
                  "noise_removed": sum(r["reason"] != "blip" for r in removed),
                  "unassigned_words": orphans},
        "settings": {"pace": args.pace, "min_silence": min_silence, "pad_before": pad_before,
                     "pad_after": pad_after, "min_speech": args.min_speech, "noise_max": args.noise_max,
                     "vad_threshold": args.vad_threshold},
    }


# --------------------------------------------------------------------------- output

def print_takes(r: dict) -> None:
    """Every repeat group and its takes (--verbose); → marks the last complete take."""
    for g in r["groups"]:
        log(f"\n   [V{raised_track(g['id'])}] {ts(g['start'])} · {len(g['texts'])} takes")
        for n, t in enumerate(g["texts"], 1):
            mark = paint("→", "33") if n == g["pick_take"] else " "
            log(f"     {mark} {n}. {t[:90]}")
    if r["groups"]:
        log()


def parse_media(ffmpeg_output: str) -> dict | None:
    """Video facts from `ffmpeg -i` output, None for audio-only files. Uses tbr, not
    fps: phones write an average into fps (29.88) for footage that is really 30."""
    video = re.search(r"Stream #.*?Video: .*?(\d{2,5})x(\d{2,5}).*?([\d.]+)(k?) tbr", ffmpeg_output)
    duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", ffmpeg_output)
    if not video or not duration:
        return None
    audio = re.search(r"Stream #.*?Audio: .*?(\d+) Hz, (mono|stereo|[\d.]+)", ffmpeg_output)
    tc = re.search(r"timecode\s*: (\d+):(\d+):(\d+)[:;.](\d+)", ffmpeg_output)
    h, m, s = duration.groups()
    channels = {"mono": 1, "stereo": 2}.get(audio.group(2), 2) if audio else 2
    return {"width": int(video.group(1)), "height": int(video.group(2)),
            "fps": float(video.group(3)) * (1000 if video.group(4) else 1),
            "duration": int(h) * 3600 + int(m) * 60 + float(s),
            "audio_rate": int(audio.group(1)) if audio else 48000, "audio_channels": channels,
            "timecode": tuple(int(x) for x in tc.groups()) if tc else None}


def probe_media(path: str) -> dict | None:
    res = subprocess.run([find_ffmpeg(), "-hide_banner", "-nostdin", "-i", path], capture_output=True)
    return parse_media(res.stderr.decode(errors="replace"))


def frame_duration(fps: float) -> Fraction:
    for nominal in (24, 30, 48, 60, 120):  # NTSC rates are exact fractions, not 29.97
        if abs(fps - nominal / 1.001) < 0.01:
            return Fraction(1001, nominal * 1000)
    if abs(fps - round(fps)) < 0.01:
        return Fraction(1, round(fps))
    return Fraction(1 / fps).limit_denominator(100000)


def fcp_time(t: Fraction) -> str:
    return f"{t.numerator}s" if t.denominator == 1 else f"{t.numerator}/{t.denominator}s"


def raised_track(group: int) -> int:
    """Which video track a repeat group sits on: V2 or V3, so neighbouring groups differ."""
    return 2 + group % 2


def build_fcpxml(result: dict, media: dict, source: str, title: str) -> bytes:
    """The whole cut as one timeline, in recording order with nothing overlapping. Single
    takes sit on V1 and repeats are raised onto V2/V3. Resolve drops clip names and markers
    on import, so the track is the only label that survives. Cuts land on source frames:
    floor(start) to ceil(end), clamped so back-to-back pieces never repeat a frame."""
    frame = frame_duration(media["fps"])
    nominal = round(1 / frame)
    total = int(media["duration"] / frame)
    src_tc = 0
    if media["timecode"]:
        h, m, s, f = media["timecode"]
        src_tc = ((h * 3600 + m * 60 + s) * nominal) + f

    pieces, prev_end = [], 0
    for c in sorted(result["clips"], key=lambda c: c["start"]):
        first = max(math.floor(c["start"] / frame), prev_end)
        end = min(math.ceil(c["end"] / frame), total)
        if end > first:
            pieces.append((c, first, end))
            prev_end = end

    root = ET.Element("fcpxml", version="1.10")
    res = ET.SubElement(root, "resources")
    ET.SubElement(res, "format", id="r1", frameDuration=fcp_time(frame),
                  width=str(media["width"]), height=str(media["height"]))
    asset = ET.SubElement(res, "asset", id="r2", name=Path(source).stem, start=fcp_time(src_tc * frame),
                          duration=fcp_time(total * frame), hasVideo="1", hasAudio="1", format="r1",
                          audioSources="1", audioChannels=str(media["audio_channels"]),
                          audioRate=str(media["audio_rate"]))
    ET.SubElement(asset, "media-rep", kind="original-media", src=Path(source).absolute().as_uri())

    event = ET.SubElement(ET.SubElement(root, "library"), "event", name="AutoCut")
    tc_start = 3600 * nominal  # Resolve's usual 01:00:00:00
    length = sum(end - first for _, first, end in pieces)
    sequence = ET.SubElement(ET.SubElement(event, "project", name=title), "sequence", format="r1",
                             duration=fcp_time(length * frame), tcStart=fcp_time(tc_start * frame), tcFormat="NDF",
                             audioLayout="mono" if media["audio_channels"] == 1 else "stereo",
                             audioRate="48k" if media["audio_rate"] == 48000 else "44.1k")
    spine = ET.SubElement(sequence, "spine")
    position = tc_start
    for c, first, end in pieces:
        at, duration = fcp_time(position * frame), fcp_time((end - first) * frame)
        clip = {"ref": "r2", "name": c["name"], "format": "r1", "tcFormat": "NDF", "offset": at,
                "start": fcp_time((src_tc + first) * frame), "duration": duration}
        if "group" in c:  # a gap on V1 holds the place; the take is connected above it
            gap = ET.SubElement(spine, "gap", name="Gap", offset=at, start=at, duration=duration)
            ET.SubElement(gap, "asset-clip", lane=str(raised_track(c["group"]) - 1), **clip)
        else:
            ET.SubElement(spine, "asset-clip", **clip)
        position += end - first
    return b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + ET.tostring(root, encoding="utf-8")


def write_timeline(result: dict, media: dict, source: str, base: str) -> str:
    out = f"{base}.autocut.fcpxml"
    with open(out, "wb") as f:
        f.write(build_fcpxml(result, media, source, f"{Path(source).stem} · AutoCut"))
    return out


def process(path: str, args, index: int = 1, count: int = 1) -> None:
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, os.path.basename(path))
    json_path = base + ".autocut.json"

    t0 = time.time()
    status(f"reading {os.path.basename(path)}…")
    audio = load_audio(path, args.audio_track)
    duration = len(audio) / SR
    log()
    section(f"[{index}/{count}] {os.path.basename(path)} · {clock(duration)}")

    cached = {}
    if os.path.exists(json_path) and not args.retranscribe:
        try:
            with open(json_path, encoding="utf-8") as f:
                cached = json.load(f).get("transcript") or {}
        except Exception:
            cached = {}
    words, language, restart_times, transcript = None, None, [], cached or None
    if not args.no_transcript:
        if (cached.get("model") == args.model and cached.get("audio_track") == args.audio_track
                and cached.get("method") == TRANSCRIPT_METHOD
                and "words" in cached and (args.language in (None, cached.get("language")))):
            words, language, restart_times = cached["words"], cached.get("language"), cached.get("restarts", [])
            step("transcript", f"reused · {len(words)} words")
        else:
            words, language, restart_times = transcribe(normalize(audio), args.model, args.language,
                                                        args.compute_type, args.beam_size, duration)
            step("transcript", f"{len(words)} words ({language})")
            transcript = {"model": args.model, "audio_track": args.audio_track, "method": TRANSCRIPT_METHOD,
                          "language": language, "words": words, "restarts": restart_times}

    result = analyze(audio, words, args, restart_times)
    result.update(version=VERSION, source=os.path.abspath(path), language=language,
                  transcribed=words is not None)

    full = dict(result)
    if transcript:  # keep it even on --no-transcript runs so the next full run is instant
        full["transcript"] = transcript
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=1)

    st = result["stats"]
    step("cut", f"{clock(st['duration'])} → {clock(st['kept'])} · {st['clips']} clips · {st['groups']} repeat groups")
    if args.verbose:
        print_takes(result)
    media = probe_media(path)
    if media and result["clips"]:
        out = write_timeline(result, media, result["source"], base)
        log(f"   {paint('→', '36')} {os.path.basename(out)}  ({took(time.time() - t0)})")
    else:
        step("timeline", "none: this file has no video, or no speech to keep", ok=False)


def expand(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p)
                          if os.path.splitext(f)[1].lower() in VIDEO_EXTS and not f.startswith("."))
        elif os.path.exists(p):
            out.append(p)
        else:
            log(f"   skip (not found): {p}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="autocut", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="video/audio files or folders")
    ap.add_argument("--pace", choices=PACES, default="natural", help="how much pause to leave at cuts (default natural)")
    ap.add_argument("--model", default="small", help="Whisper model: base, small, medium, large-v3 (default small)")
    ap.add_argument("--language", default=None, help="e.g. en, hi, te. Default: auto-detect")
    ap.add_argument("--audio-track", type=int, default=0, help="which audio stream to analyze (0 = first)")
    ap.add_argument("--no-transcript", action="store_true", help="silence/noise cut only, skip Whisper and repeat detection")
    ap.add_argument("--retranscribe", action="store_true", help="ignore the cached transcript")
    ap.add_argument("--out-dir", default=None, help="write results here instead of next to the video")
    ap.add_argument("--verbose", action="store_true", help="list every repeat group and its takes")
    adv = ap.add_argument_group("tuning")
    adv.add_argument("--min-silence", type=float, default=None, help="override the pace's minimum silence (s)")
    adv.add_argument("--min-speech", type=float, default=0.15, help="speech shorter than this is a blip (s)")
    adv.add_argument("--noise-max", type=float, default=0.8, help="wordless sounds shorter than this are cut (s)")
    adv.add_argument("--vad-threshold", type=float, default=0.5, help="speech detector sensitivity 0-1 (lower = keeps more)")
    adv.add_argument("--window", type=int, default=3, help="how many following clips to compare for repeats")
    adv.add_argument("--compute-type", default="int8")
    adv.add_argument("--beam-size", type=int, default=5)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    files = expand(args.paths)
    if not files:
        sys.exit("nothing to do")
    t0 = time.time()
    what = "silence cut only" if args.no_transcript else f"model {args.model}"
    log(f"{paint('autocut', '1')} · {len(files)} video{'s' if len(files) > 1 else ''} · {what} · pace {args.pace}")
    failed = []
    for n, f in enumerate(files, 1):
        try:
            process(f, args, n, len(files))
        except KeyboardInterrupt:
            log()
            sys.exit("cancelled")
        except Exception as e:  # keep going on batch runs
            failed.append((f, e))
            step("failed", str(e), ok=False)
    log()
    section(f"Done: {len(files) - len(failed)}/{len(files)} videos in {took(time.time() - t0)}")
    for f, e in failed:
        log(f"   {paint('✗', '31')} {os.path.basename(f)}: {e}")
    log("   Next: in Resolve, File → Import → Timeline, and pick each .fcpxml")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
