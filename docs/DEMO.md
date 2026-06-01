# Recording the demo GIF

A short, punchy GIF in the README is the single biggest driver of stars for a CLI
tool. The catch: a **real** `papermind analyze` run makes several LLM calls and
takes 30–60s — far too slow for a "wow in the first 2 seconds" GIF.

So PaperMind ships a built-in, offline replay: **`papermind demo`**. It renders a
curated report + a grounded, layered Q&A with built-in pacing — no API key, no
network, fully deterministic, ~12 seconds. Record that.

```bash
papermind demo            # self-paced (~12s), clears screen between the two beats
papermind demo --speed 1.3   # a bit faster (~9.5s)
papermind demo --speed 0     # instant (for piping / screenshots)
```

The demo is two beats so each gets a clean, readable frame:
1. `analyze` → the structured four-module report (contributions / technical+图 / connections / reproduction).
2. `ask` → the **grounded, layered answer** (论文事实 / 推理+置信度 / 原文依据) — the differentiator.

## Recommended: VHS (deterministic, scriptable, no manual recording)

[VHS](https://github.com/charmbracelet/vhs) renders a GIF from a script — repeatable
and CI-friendly. A ready-made tape lives at [`docs/demo.tape`](demo.tape):

```bash
# install VHS:  brew install vhs   |   scoop install vhs   |   see repo for Linux
pip install -e .
vhs docs/demo.tape          # -> writes docs/demo.gif
```

Then uncomment the `<img src="docs/demo.gif">` line near the top of the
[README](../README.md). Tune `FontSize` / `Width` / `Height` in the tape, or pass a
different `--speed`, to taste. A taller `Height` (≈1000) keeps the whole report on
one frame.

## Alternatives

- **macOS / Linux:** [asciinema](https://asciinema.org) + [agg](https://github.com/asciinema/agg)
  ```bash
  asciinema rec demo.cast --cols 120 --rows 40 -c "papermind demo"
  agg --font-size 20 demo.cast docs/demo.gif
  ```
- **Windows:** [ScreenToGif](https://www.screentogif.com/) — record the Windows
  Terminal window running `papermind demo`, trim, export GIF.

## Tips

- Keep the GIF **under 20s**; the magic should be visible in the first 2.
- Use a clean theme and a font large enough to read in a thumbnail.
- The looping GIF lets viewers re-read — favor a comfortable pace over cramming.
- Record a second GIF of the live `tutor`/`debug` flow for the wiki (needs a key/Ollama).
