# AutoCut — first-pass rough cut for talking-head footage in DaVinci Resolve

Built for one job: footage of someone reading a script to camera, full of pauses, false starts and re-reads.

**What it does**

You get one timeline for the video:

1. **Silences are cut, without leaving noise frames.** It uses a neural speech detector (Silero VAD), not a volume gate, so clicks, desk bumps and room noise aren't treated as speech in the first place. On top of that, anything under 0.15 s is dropped as a blip, and wordless sounds under 0.8 s (coughs, lip smacks, breaths before nothing) are dropped too.
2. **Every line stays, in recording order.** Lines said once sit on V1.
3. **Repeats are raised onto V2 and V3.** It transcribes the clip locally with Whisper, compares both the words and the sound of the voice, and finds false starts and re-reads of the same line. That includes fragments Whisper garbles, and restarts with no pause at all, which get cut into separate clips. You choose the take you want.

It works by importing a timeline file, so it doesn't depend on Resolve's scripting. Tested on free 21.0.0. Everything runs offline after the first model download.

---

## Install (macOS)

```bash
git clone https://github.com/MS-Teja/autocut.git
cd autocut
bash install.sh
```

The installer creates a private Python environment in `~/.autocut`, adds an `autocut` command, and downloads the Whisper model once.

## Use

**1. Analyze** (Terminal; you can keep editing while it runs):

```bash
autocut ~/Footage/ep12/A001.mp4      # one clip
autocut ~/Footage/ep12/              # every video in the folder
```

It writes `A001.mp4.autocut.fcpxml` next to the clip, and shows its progress like this:

```
autocut · 1 video · model small · pace natural

:: Downloading Whisper small (one time)
   downloading  [-----------------C  o  o  o  o ]  301.4 MB/486.0 MB  4.2 MB/s  0:44 left

:: [1/1] A001.mp4 · 10:46
   transcribing [-----------------------------c]  100%
   ✓ transcript  1099 words (en)
   ✓ cut         10:46 → 5:49 · 119 clips · 26 repeat groups
   → A001.mp4.autocut.fcpxml  (3m 20s)
```

The model downloads once, on first use. Pass `--verbose` to also list every repeat group and its takes.

**2. Import** (Resolve): **File → Import → Timeline** and pick that `.fcpxml`.

## Reading the timeline

```
V3                  [B1][B2]
V2       [A1][A2]
V1  [one]        [two]       [three]
```

The timeline plays straight through in recording order: one, A1, A2, two, B1, B2, three. Nothing overlaps.

| Where | Meaning |
|---|---|
| V1 | Said once. Nothing to decide. |
| V2 / V3 | Takes of a repeated line. Each group is on one track, and groups alternate between V2 and V3 so neighbours look different. |

To choose a take, ripple-delete the ones you don't want. The gap closes.

With `--verbose`, the terminal lists each group's takes and marks the last complete one with `→`. It's only a suggestion, and nothing is removed for you.

Clip names and markers don't survive the import into Resolve, so the track is the only label.

## Options

| Option | What it does |
|---|---|
| `--pace tight` / `natural` / `loose` | How much pause stays at each cut (default `natural`). `tight` cuts right at the speech; rounding to whole frames can leave at most 1 frame. Re-runs reuse the transcript, so trying another pace takes seconds. |
| `--language en` (`hi`, `te`, …) | Skip auto-detect. Set this for Hindi/Telugu or mixed-language videos. |
| `--model medium` | More accurate transcript, slower. `base` is faster, less accurate. Default `small`. |
| `--audio-track 1` | Analyze the second audio stream (e.g. if the lav mic is on track 2). |
| `--no-transcript` | Silence/noise cut only. Fast, no repeat detection. |
| `--vad-threshold 0.35` | Lower = keeps quieter speech (default 0.5). Raise if noise still sneaks in. |
| `--min-speech 0.15` | Speech shorter than this counts as a blip (seconds). |
| `--noise-max 0.8` | Wordless sounds shorter than this are cut (seconds). |
| `--retranscribe` | Ignore the cached transcript. |
| `--verbose` | List every repeat group and its takes, with the last complete take marked `→`. |

## What it can't do

- **It doesn't judge delivery.** Choosing the take is your call.
- **It isn't perfect.** On real footage it caught about 94% of repeated clips, as judged from transcripts. A repeat it misses stays on V1, so you still see it play. Now and then a real line gets raised, or neighbouring lines end up on one track, but nothing is ever removed. Accuracy is lower on heavily mixed Hindi/Telugu-English; use `--language` and `--model medium` there.
- **Takes are whole clips.** If he re-reads only the second half of a sentence, both parts get grouped, but splicing half of one take with half of another is on you.

## Troubleshooting

- **Clip shows as offline after import** → the `.fcpxml` points at the video's path when it was analyzed. Keep the file where it was, or run `autocut` again after moving it.
- **Synced external audio** → analyze the camera file as usual; the scratch audio has the same timing.

## License

MIT. See [LICENSE](LICENSE).

## Uninstall

```bash
rm -rf ~/.autocut ~/.local/bin/autocut
```
