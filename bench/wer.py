from __future__ import annotations

from bench.normalize import normalize_text


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after shared normalization. Returns 0..inf (can exceed 1)."""
    from jiwer import wer as jiwer_wer

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer_wer(ref, hyp))


def pooled_wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus-level WER over (reference, hypothesis) pairs."""
    from jiwer import wer as jiwer_wer

    refs = [normalize_text(r) for r, _ in pairs]
    hyps = [normalize_text(h) for _, h in pairs]
    # Drop empty refs to avoid jiwer edge cases.
    filtered = [(r, h) for r, h in zip(refs, hyps) if r]
    if not filtered:
        return 0.0
    refs_f, hyps_f = zip(*filtered)
    return float(jiwer_wer(list(refs_f), list(hyps_f)))
