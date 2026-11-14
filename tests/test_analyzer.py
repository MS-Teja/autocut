"""Analyzer tests. Run: ~/.autocut/venv/bin/python tests/run_all.py"""
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from fractions import Fraction

import numpy as np

from fixtures import A, ROOT, make_rough, speech_sample, write_transcript_cache

TMP = tempfile.mkdtemp(prefix="autocut-test-")
WAV, WORDS, EVENTS = make_rough(TMP)


def args(*extra):
    return A.build_parser().parse_args(["x", *extra])


def test_groups_and_picks():
    r = A.analyze(A.load_audio(WAV), WORDS, args())
    roles = [c["role"] for c in r["clips"]]
    # The short "um" between takes of line C counts as an attempt at it. "ask not what your country
    # can do for you" reuses most of line C's words, so it gets raised with it. That's accepted:
    # the clip stays on the timeline, while requiring a closer match loses real repeats on real footage.
    assert roles == ["take", "pick", "take", "take", "take", "take", "take", "pick"], roles
    assert [g["clips"] for g in r["groups"]] == [[0, 1], [2, 3, 4, 5, 6, 7]]
    assert [g["pick_take"] for g in r["groups"]] == [2, 6]


def test_clicks_and_cough_not_kept():
    r = A.analyze(A.load_audio(WAV), WORDS, args())
    for name in ("click1", "cough"):
        t = EVENTS[name]
        assert not any(c["start"] <= t <= c["end"] for c in r["clips"]), name


def test_restart_split_with_sensitive_vad():
    # at a very low threshold the VAD merges "ask what you" + retake; the
    # restart splitter must cut them apart again
    r = A.analyze(A.load_audio(WAV), WORDS, args("--vad-threshold", "0.1"))
    names = [c["name"] for c in r["clips"]]
    assert any(n.startswith("take 1/4 · ask what you") for n in names), names


def test_wordless_short_region_is_noise():
    words = [w for w in WORDS if w["text"] != "um"]
    r = A.analyze(A.load_audio(WAV), words, args())
    assert any(x["reason"].startswith("noise") for x in r["removed"])


def test_tighten_strips_edge_click():
    db = np.full(300, -60.0)
    db[50:150] = -15
    db[175:178] = -10   # click 0.25 s after speech
    db[20:23] = -10     # click before speech
    assert A.tighten([[0.1, 1.9]], db, -60.0) == [[0.5, 1.5]]


def test_no_transcript_mode():
    r = A.analyze(A.load_audio(WAV), None, args("--no-transcript"))
    assert {c["role"] for c in r["clips"]} == {"single"}
    assert not r["groups"]


def test_same_line_rules():
    N = A.norm_words
    far = 9.0  # voices that don't match
    near = 2.0  # voices that do
    cases = [
        # false starts and fragments of the line that follows
        ("so today we are going to", "so today we are going to talk about money", far, True, True),
        ("so", "so the first thing is", far, True, True),
        ("so", "so the first thing is", far, False, False),  # one word is only evidence right next door
        ("let's go.", "let's go.", far, True, True),
        ("so today we are gonna", "so today we are going to talk", far, True, True),
        # garbled false starts, the way Whisper hears them on real footage
        ("Onthe cowner", "On the comer, PRTL8.", near, True, True),
        ("On the comer, PRTL8.", "On the corner he gives a lot and he can even run", near, True, True),
        # garbled beyond reading, but it sounds like the next take
        ("Thank you so much", "The new plan is something we can test", 1.7, True, True),
        # different lines
        ("The market opens at nine", "They usually close by five", far, True, False),
        ("you need a budget before anything else", "a budget is the first thing you need", far, True, False),
        ("because it creates space behind them", "however it leaves the midfield open", far, False, False),
    ]
    bad = [(a, b) for a, b, d, adj, e in cases if A.same_line(N(a), N(b), d, adj) != e]
    assert not bad, bad


def test_restart_inside_a_clip():
    N = A.norm_words
    assert A.restarts_at(N("I also"), N("I also think"))
    assert A.restarts_at(N("so if an attacker drops"), N("so if a striker"))
    assert A.restarts_at(N("Tomorrow"), N("Tomorrow morning we"))
    assert not A.restarts_at(N("then they find the ball and"), N("then play it"))
    assert not A.restarts_at(N("as"), N("as well"))  # too short to mean anything


def test_restart_times_cut_the_clip_only_where_the_opening_repeats():
    audio = A.load_audio(WAV)
    without = A.analyze(audio, WORDS, args())
    # "your country can | do for you": the transcript says restart, but the words don't repeat
    clip = [w for w in WORDS if EVENTS["B"] + 2.0 < w["start"] < EVENTS["B"] + 4.4]
    t = (clip[2]["end"] + clip[3]["start"]) / 2
    assert len(A.analyze(audio, WORDS, args(), [t])["clips"]) == len(without["clips"])
    # "your country can | your country ...": it does
    redone = [dict(w, text={id(clip[3]): "your", id(clip[4]): "country"}.get(id(w), w["text"])) for w in WORDS]
    assert len(A.analyze(audio, redone, args(), [t])["clips"]) > len(without["clips"])


def test_restart_needs_the_same_opening():
    N = A.norm_words
    assert not A.opens_alike(N("The red team playing on the right."), N("to the team playing on the left."))
    assert A.opens_alike(N("And after that"), N("And after lunch, the team met"))
    assert not A.opens_alike([], N("anything"))


def test_a_false_start_cannot_join_two_lines():
    N = A.norm_words
    texts = ["After the break we saw chances down the left side",
             "After that break we saw chances down the left side and the right",
             "And after that...",
             "And after the game the coach said",
             "And after the game the coach said something similar."]
    nw = [N(t) for t in texts]
    match = lambda i, j, adjacent: A.same_line(nw[i], nw[j], float("inf"), adjacent)
    assert match(1, 2, True) and match(2, 3, True)  # the fragment looks like the opening of both lines
    groups = A.group_lines([len(w) >= 4 for w in nw], match, 3)
    assert groups == {0: 0, 1: 0, 2: 2, 3: 2, 4: 2}, groups


def test_voice_onset_walks_back_to_the_rise():
    db = np.full(100, 5.0)
    db[52:] = 60          # Silero's start, at the vowel
    db[40:52] = 25        # the "sh" before it
    assert A.voice_onset(db, 0.52, 0.0) == 0.40
    assert A.voice_onset(db, 0.52, 0.45) == 0.45  # never before the previous clip's half of the gap
    db[30:38] = 20        # a breath, then a dip, then the "sh"
    assert A.voice_onset(db, 0.52, 0.0) == 0.40
    flat = np.full(100, 5.0)
    flat[52:] = 60
    assert A.voice_onset(flat, 0.52, 0.0) == 0.52


def test_voice_distance_links_a_repeat_and_not_another_line():
    audio = A.normalize(A.load_audio(speech_sample()))
    f = A.voice_features(audio)
    span = lambda s, e: f[int(s * A.FEATURE_FPS):int(e * A.FEATURE_FPS)]
    assert A.voice_distance(span(8.22, 9.10), span(8.22, 10.50)) < 0.01  # identical audio
    assert A.voice_distance(span(8.22, 9.10), span(0.32, 2.24)) > A.SOUND_MATCH  # a different line
    assert A.voice_distance(span(8.22, 8.25), span(8.22, 10.50)) == float("inf")  # too short to compare


def test_self_repeat_finds_a_line_restarted_without_a_pause():
    src = A.load_audio(speech_sample())
    line = src[int(8.22 * A.SR):int(10.50 * A.SR)]
    other = src[int(5.44 * A.SR):int(7.62 * A.SR)]
    tone = np.zeros(A.SR, np.float32)
    audio = A.normalize(np.concatenate([tone, line[:A.SR], line, tone, other, tone]))
    f = A.voice_features(audio)
    lag = A.self_repeat(f, 1.0, 1.0 + 1.0 + len(line) / A.SR)
    assert lag is not None and abs(lag - 1.0) < 0.2, lag
    assert A.self_repeat(f, 1.0 + 1.0 + len(line) / A.SR + 1.0, len(audio) / A.SR - 1.0) is None  # one read, no repeat


def test_candy_bar_is_pacman_eating_pellets():
    assert A.candy(0.0, 10, 0) == "[C  o  o  o]"
    assert A.candy(0.5, 10, 1) == "[----c  o  ]"
    assert A.candy(1.0, 10, 0) == "[---------c]"


def test_indic_tokenizer_keeps_vowel_signs():
    assert A.norm_words("नमस्ते दोस्तों, आज हम") == ["नमस्ते", "दोस्तों", "आज", "हम"]
    assert A.norm_words("అందరికీ నమస్కారం.") == ["అందరికీ", "నమస్కారం"]


def test_real_whisper_hears_every_read_of_a_line():
    # The other tests use ground-truth words. This one exists because real Whisper
    # dropped re-reads when given all the speech at once. Needs the model, so it
    # only runs where install.sh has downloaded it.
    if not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-small")):
        print("SKIP real Whisper test: model not downloaded")
        return
    src = A.load_audio(speech_sample())
    rng = np.random.default_rng(0)
    tone = (rng.standard_normal(int(1.6 * A.SR)) * 10 ** (-55 / 20)).astype(np.float32)
    part, line_a, line_c = (0.32, 1.20), (0.32, 2.24), (8.22, 10.50)

    def heard(reads):
        audio = [tone]
        for s, e in reads:
            audio += [src[int(s * A.SR):int(e * A.SR)], tone]
        audio = A.normalize(np.concatenate(audio))
        words = A.transcribe(audio, "small", "en", "int8", 5, len(audio) / A.SR)[0]
        return [t for w in words for t in A.norm_words(w["text"])]

    # With vad_filter these came back with one read missing: 5 words not 9, 21 not 26.
    assert heard([part, line_a]).count("so") == 2  # false start, then the full line
    words = heard([line_a, line_a, line_c, line_c])
    assert words.count("fellow") == 2 and words.count("country") == 2, words


MEDIA = {"width": 1920, "height": 1080, "fps": 29.97, "duration": 23.3, "audio_rate": 48000,
         "audio_channels": 2, "timecode": None}


def timeline(result, media=MEDIA):
    """(root, [(spine element, clip element)]) in timeline order."""
    root = ET.fromstring(A.build_fcpxml(result, media, "/footage/my clip.mov", "t"))
    return root, [(el, el if el.tag == "asset-clip" else el.find("asset-clip")) for el in root.find(".//spine")]


def secs(s):
    return Fraction(s[:-1])


def test_fcpxml_is_one_continuous_timeline_of_every_clip():
    r = A.analyze(A.load_audio(WAV), WORDS, args())
    frame = Fraction(1001, 30000)
    root, items = timeline(r)
    assert root.find(".//format").get("frameDuration") == "1001/30000s"
    assert root.find(".//media-rep").get("src") == "file:///footage/my%20clip.mov"
    assert len(root.findall(".//project")) == 1
    assert [clip.get("name") for _, clip in items] == [c["name"] for c in r["clips"]], "not every clip, in order"
    assert secs(items[0][0].get("offset")) == 108000 * frame  # 01:00:00:00
    for (a, a_clip), (b, b_clip) in zip(items, items[1:]):
        assert secs(a.get("offset")) + secs(a.get("duration")) == secs(b.get("offset")), "gap or overlap"
        assert secs(a_clip.get("start")) + secs(a_clip.get("duration")) <= secs(b_clip.get("start")), "frame repeated"
    total = sum(secs(el.get("duration")) for el, _ in items)
    assert secs(root.find(".//sequence").get("duration")) == total
    for (el, clip), c in zip(items, r["clips"]):
        start, dur = secs(clip.get("start")), secs(clip.get("duration"))
        assert (start / frame).denominator == 1 and (dur / frame).denominator == 1, "off the frame grid"
        assert start <= Fraction(c["speech_start"]) and start + dur >= Fraction(c["speech_end"]), "speech clipped"


def test_fcpxml_raises_repeats_above_single_takes():
    def clip(i, group=None):
        c = {"id": i, "start": i * 2.0, "end": i * 2.0 + 1, "name": f"clip {i}"}
        return c if group is None else dict(c, group=group)
    r = {"clips": [clip(0), clip(1, 0), clip(2, 0), clip(3), clip(4, 1), clip(5, 1), clip(6), clip(7, 2)]}
    root, items = timeline(r)
    assert [c.get("lane") for _, c in items] == [None, "1", "1", None, "2", "2", None, "1"]
    assert not root.findall(".//marker")
    for el, c in items:
        if c.get("lane"):  # held in place by a same-length gap on V1, at the gap's own time
            assert el.tag == "gap" and el.get("start") == el.get("offset") == c.get("offset")
            assert el.get("duration") == c.get("duration")
        else:
            assert el is c


def test_fcpxml_source_timecode_offsets_cuts():
    r = A.analyze(A.load_audio(WAV), WORDS, args())
    plain = timeline(r, dict(MEDIA, fps=30.0))[1]
    root, tc = timeline(r, dict(MEDIA, fps=30.0, timecode=(1, 0, 0, 0)))
    assert secs(root.find(".//asset").get("start")) == 3600
    assert all(secs(b.get("start")) - secs(a.get("start")) == 3600 for (_, a), (_, b) in zip(plain, tc))


def test_writes_a_single_timeline_file():
    r = A.analyze(A.load_audio(WAV), WORDS, args())
    out_dir = tempfile.mkdtemp(prefix="autocut-test-")
    out = A.write_timeline(r, MEDIA, "/footage/my clip.mov", os.path.join(out_dir, "my clip.mov"))
    assert os.listdir(out_dir) == ["my clip.mov.autocut.fcpxml"] and os.path.basename(out) == os.listdir(out_dir)[0]


def test_parse_media_uses_tbr_not_average_fps():
    out = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'Conclusion.mov':
  Metadata:
    major_brand     : qt
  Duration: 00:00:40.62, start: 0.000000, bitrate: 8635 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main 10) (hvc1 / 0x31637668), yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67), 1920x1080, 8432 kb/s, 29.88 fps, 30 tbr, 600 tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 102 kb/s (default)
"""
    m = A.parse_media(out)
    assert (m["fps"], m["width"], m["height"], m["duration"]) == (30.0, 1920, 1080, 40.62)
    assert m["audio_rate"] == 48000 and m["audio_channels"] == 2 and m["timecode"] is None
    assert A.parse_media(out + "      timecode        : 01:00:00:00\n")["timecode"] == (1, 0, 0, 0)
    assert A.parse_media("  Duration: 00:00:10.00\n  Stream #0:0: Audio: flac, 16000 Hz, mono") is None


def test_cli_reuses_cached_transcript_and_keeps_it():
    write_transcript_cache(WAV, WORDS)
    run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "autocut.py"), WAV, *a],
                                    capture_output=True, text=True)
    p = run()
    assert p.returncode == 0, p.stderr + p.stdout
    assert "reused" in p.stdout and "repeat groups" in p.stdout
    assert "\r" not in p.stdout and "\033[" not in p.stdout  # piped output: no bars, no colour
    assert "1. " not in p.stdout  # the take list is only for --verbose
    assert "1. " in run("--verbose").stdout
    p = run("--no-transcript", "--pace", "tight")
    assert p.returncode == 0, p.stderr
    with open(WAV + ".autocut.json") as f:
        data = json.load(f)
    assert data["transcript"]["words"], "transcript cache lost after --no-transcript run"
    assert data["settings"]["pace"] == "tight"
