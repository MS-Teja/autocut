"""Shared test fixtures: a synthetic 'messy script read' built from real speech.

Downloads the 11 s JFK sample from the openai/whisper repo once (tests/data/),
then splices it into a recording with false starts, a 3-take sentence, a junk
"um", desk clicks and a cough, plus a ground-truth word list that stands in for
Whisper (so tests don't need a model download).
"""
import json
import os
import sys
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import autocut as A  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data")
JFK_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"


def speech_sample():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "jfk.flac")
    if not os.path.exists(path):
        urllib.request.urlretrieve(JFK_URL, path)
    return path


def make_rough(out_dir):
    """Returns (wav_path, words, events). Silero finds these JFK phrases at
    0.32-2.24 'And so my fellow Americans', 3.33-7.62 'ask not | what | your
    country can do for you', 8.22-10.50 'ask what you can do for your country'."""
    SR = A.SR
    src = A.load_audio(speech_sample())
    rng = np.random.default_rng(0)
    parts, words, events, t = [], [], {}, [0.0]

    def sil(d):
        parts.append(np.zeros(int(d * SR), np.float32))
        t[0] += d

    def noise(d, amp, shape=None):
        n = rng.standard_normal(int(d * SR)).astype(np.float32) * amp
        if shape == "hann":
            n *= np.hanning(len(n)).astype(np.float32)
        parts.append(n)
        t[0] += len(n) / SR

    def speech(s, e, spans):
        seg = src[int(s * SR):int(e * SR)]
        parts.append(seg)
        for a, b, txt in spans:
            ws = txt.split()
            step = (b - a) / len(ws)
            for i, w in enumerate(ws):
                words.append({"start": round(t[0] + (a - s) + i * step + 0.02, 3),
                              "end": round(t[0] + (a - s) + (i + 1) * step - 0.02, 3), "text": w, "p": 0.9})
        t[0] += len(seg) / SR

    sil(1.0)
    events["A_abort"] = t[0]; speech(0.32, 1.20, [(0.32, 1.20, "And so my")])
    sil(0.6); events["click1"] = t[0]; noise(0.033, 0.35); sil(0.6)
    events["A_full"] = t[0]; speech(0.32, 2.24, [(0.32, 2.24, "And so, my fellow Americans,")])
    sil(0.9); events["cough"] = t[0]; noise(0.3, 0.12, "hann"); sil(1.0)
    events["B"] = t[0]; speech(3.33, 7.62, [(3.33, 3.74, "ask not"), (4.03, 4.35, "what"),
                                            (5.44, 7.62, "your country can do for you,")])
    sil(1.5)
    events["C_abort"] = t[0]; speech(8.22, 9.10, [(8.22, 9.10, "ask what you")])
    sil(0.4)
    events["C_2"] = t[0]; speech(8.22, 10.50, [(8.22, 10.50, "ask what you can do for your country.")])
    sil(1.2)
    events["junk"] = t[0]; speech(0.32, 0.62, [(0.32, 0.62, "um")])
    sil(1.2)
    events["C_3"] = t[0]; speech(8.22, 10.50, [(8.22, 10.50, "ask what you can do for your country.")])
    sil(0.15); events["click2"] = t[0]; noise(0.033, 0.35)
    sil(1.5)

    audio = np.concatenate(parts)
    audio += (rng.standard_normal(len(audio)) * 10 ** (-55 / 20)).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "rough.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return wav, words, events


def write_transcript_cache(wav, words, model="small"):
    """Pretend autocut already transcribed this file, so the CLI reuses it."""
    with open(wav + ".autocut.json", "w") as f:
        json.dump({"transcript": {"model": model, "audio_track": 0, "method": A.TRANSCRIPT_METHOD,
                                  "language": "en", "words": words}}, f)
