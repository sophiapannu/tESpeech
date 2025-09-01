#spec_with_words.py no comments
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft, resample_poly
from math import gcd
ORIG_WAV = "harvard.wav"                      
SYNTH_WAV = "harvard_preview_onset_coda.wav"  
TARGET_SR = 48000
YLIM = 4000

WORDS_TEXTGRID = "harvard.words.TextGrid"   
WORDS_CSV      = "harvard_words.csv"        
PHONES_TEXTGRID = None
LABEL_EVERY_WORD  = 2
LABEL_EVERY_PHONE = 8
def load_mono(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float32), sr
def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down).astype(np.float32)
def make_spec(x, sr, nperseg=1024, noverlap=768):
    f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap,
                   window="hann", boundary=None)
    S_db = 20 * np.log10(np.abs(Z) + 1e-10)
    return f, t, S_db
def load_spans_from_textgrid(path, tier_candidates):
    try:
        from textgrid import TextGrid
    except ImportError:
        raise RuntimeError("Missing dependency: install with `pip install textgrid`")
    tg = TextGrid.fromFile(path)
    tier = None
    for name in tier_candidates:
        try:
            tier = tg.getFirst(name)
            break
        except Exception:
            continue
    if tier is None:
        raise ValueError(f"No tier named any of {tier_candidates} found in {path}")
    spans = []
    for itv in tier.intervals:
        lab = (itv.mark or "").strip()
        s, e = float(itv.minTime), float(itv.maxTime)
        if lab and e > s:
            spans.append((lab, s, e))
    return spans 
def load_words_from_csv(path):
    import csv
    spans = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = row["word"].strip()
            s = float(row["start"])
            e = float(row["end"])
            if lab and e > s:
                spans.append((lab, s, e))
    return spans
def overlay_spans(ax, spans, y_top, label_every=2, alpha=0.12, fontsize=8, rotate=90):
    for k, (lab, s, e) in enumerate(spans):
        ax.axvspan(s, e, color="k", alpha=alpha, lw=0)
        if label_every and (k % label_every) == 0:
            xm = 0.5 * (s + e)
            ax.text(
                xm, 0.96 * y_top, lab,
                ha="center", va="top", fontsize=fontsize, rotation=rotate,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
            )
def main():
    x0, sr0 = load_mono(ORIG_WAV)
    x1, sr1 = load_mono(SYNTH_WAV)
    x0 = resample_to(x0, sr0, TARGET_SR)
    x1 = resample_to(x1, sr1, TARGET_SR)
    n = min(len(x0), len(x1))
    x0, x1 = x0[:n], x1[:n]
    f0, t0, S0 = make_spec(x0, TARGET_SR)
    f1, t1, S1 = make_spec(x1, TARGET_SR)
    vmax = max(np.max(S0), np.max(S1))
    vmin = vmax - 80
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    im0 = axs[0].pcolormesh(t0, f0, S0, shading="auto", vmin=vmin, vmax=vmax)
    axs[0].set_title("Original")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Freq (Hz)")
    axs[0].set_ylim(0, YLIM)
    im1 = axs[1].pcolormesh(t1, f1, S1, shading="auto", vmin=vmin, vmax=vmax)
    axs[1].set_title("Synthesized")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(0, YLIM)
    cbar = fig.colorbar(im1, ax=axs[1], location="right", fraction=0.046, pad=0.04)
    cbar.set_label("dB")
    words = None
    if WORDS_TEXTGRID:
        try:
            words = load_spans_from_textgrid(WORDS_TEXTGRID, ("words", "word", "Words", "Word"))
        except Exception:
            words = None
    if (words is None) and WORDS_CSV:
        try:
            words = load_words_from_csv(WORDS_CSV)
        except Exception:
            words = None
    if words:
        overlay_spans(axs[0], words, y_top=YLIM, label_every=LABEL_EVERY_WORD, alpha=0.10, fontsize=8)
        overlay_spans(axs[1], words, y_top=YLIM, label_every=LABEL_EVERY_WORD, alpha=0.10, fontsize=8)

    
    if PHONES_TEXTGRID:
        try:
            phones = load_spans_from_textgrid(PHONES_TEXTGRID, ("phones", "phonemes", "Phones", "Phonemes"))
            overlay_spans(axs[0], phones, y_top=YLIM, label_every=LABEL_EVERY_PHONE, alpha=0.06, fontsize=6)
            overlay_spans(axs[1], phones, y_top=YLIM, label_every=LABEL_EVERY_PHONE, alpha=0.06, fontsize=6)
        except Exception:
            pass
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
