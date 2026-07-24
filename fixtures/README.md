# Fixtures

Synthetic audio helpers. `prepare` writes `data/tone16k.wav` and `data/tones/*.wav` for offline smoke tests.

Optional LibriSpeech clips:

```bash
make gpu-prepare          # or: make prepare  (tones only, CPU image)
# LibriSpeech: make gpu-prepare N=25
```
