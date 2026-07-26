# Fixtures

Synthetic audio helpers. `prepare` writes `data/tone16k.wav` and `data/tones/*.wav` for offline smoke tests.

Optional LibriSpeech clips via Hugging Face `datasets`:

```bash
make gpu-build            # image includes datasets
make gpu-prepare          # tones + LibriSpeech; override with N=25
make prepare              # tones only (CPU image)
```
