#spec_with_words_debug.py
import os, sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft, resample_poly
from math import gcd

#config
ORIG_WAV = "harvard.wav"            
SYNTH_WAV = "harvard_preview_onset_coda.wav"  #generated
TARGET_SR = 48000
YLIM = 4000
WORDS_TEXTGRID = "harvard.words.TextGrid"   
WORDS_CSV      = "harvard_words.csv"        
#optional: overlay phonemes as well (tier "phones"); leave None to skip :)
PHONES_TEXTGRID = None
#label density (to avoid clutter)
LABEL_EVERY_WORD  = 2
LABEL_EVERY_PHONE = 8
#utilities
def here(p): return os.path.abspath(p)
def load_mono(path):
    print(f"[INFO] Loading {here(path)}")
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    print(f"[OK] {os.path.basename(path)}: shape={getattr(x,'shape',None)}, sr={sr}")
    return x.astype(np.float32), sr
def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    print(f"[INFO] Resampling {sr_in} -> {sr_out} via up={up}, down={down}")
    return resample_poly(x, up, down).astype(np.float32)
def make_spec(x, sr, nperseg=1024, noverlap=768):
    f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap,
                   window="hann", boundary=None)
    S_db = 20*np.log10(np.abs(Z) + 1e-10)
    return f, t, S_db

#alignment to load
def load_spans_from_textgrid(path, tier_candidates):
    print(f"[INFO] Trying TextGrid: {here(path)}")
    try:
        from textgrid import TextGrid
    except ImportError:
        print("[WARN] textgrid not installed. Install with: pip install textgrid")
        return None
    if not os.path.exists(path):
        print("[WARN] TextGrid file not found.")
        return None
    tg = TextGrid.fromFile(path)
    tier = None
    for name in tier_candidates:
        try:
            tier = tg.getFirst(name)
            print(f"[OK] Found tier: {name}")
            break
        except Exception:
            pass
    if tier is None:
        print(f"[WARN] No tier named any of {tier_candidates}")
        return None
    spans = []
    for itv in tier.intervals:
        lab = (itv.mark or "").strip()
        s, e = float(itv.minTime), float(itv.maxTime)
        if lab and e > s:

            spans.append((lab, s, e))
    print(f"[OK] Loaded {len(spans)} spans from TextGrid")
    return spans  #returns list[(label, start, end)

def load_words_from_csv(path):
    print(f"[INFO] Trying CSV: {here(path)}")
    if not os.path.exists(path):
        print("[WARN] CSV file not found.")
        return None
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
    print(f"[OK] Loaded {len(spans)} spans from CSV")
    return spans
#to overlay/debug if needed
def debug_bounds(name, spans, tmax):
    if not spans:
        print(f"[INFO] {name}: no spans")
        return []
    s_min = min(s for _, s, _ in spans)
    e_max = max(e for _, _, e in spans)
    print(f"[OK] {name}: {len(spans)} spans, time range [{s_min:.3f}, {e_max:.3f}]s, plot tmax={tmax:.3f}s")
    clipped = []
    for lab, s, e in spans:
        if e < 0 or s > tmax:
            continue
        s2 = max(0.0, s)
        e2 = min(tmax, e)
        if e2 > s2 + 1e-4:
            clipped.append((lab, s2, e2))
    print(f"[OK] {name}: {len(clipped)} spans after clipping to [0, {tmax:.3f}]s")
    return clipped
def overlay_spans_clean(ax, spans, y_top, label_every=2, alpha=0.18, fontsize=8, rotate=90):
    for k, (lab, s, e) in enumerate(spans):
        ax.axvspan(s, e, color="k", alpha=alpha, lw=0, zorder=10)
        ax.axvline(s, color="w", lw=0.6, alpha=0.7, zorder=11)
        ax.axvline(e, color="w", lw=0.6, alpha=0.7, zorder=11)
        if label_every and (k % label_every) == 0:
            xm = 0.5 * (s + e)
            ax.text(
                xm, 0.96 * y_top, lab,
                ha="center", va="top", fontsize=fontsize, rotation=rotate,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=12
            )

#main!
def main():
    #to load the audio
    try:
        x0, sr0 = load_mono(ORIG_WAV)
        x1, sr1 = load_mono(SYNTH_WAV)
    except Exception as e:
        print("[ERR] Audio load failed:", e)
        sys.exit(1)
    #resamples and trims
    x0 = resample_to(x0, sr0, TARGET_SR)
    x1 = resample_to(x1, sr1, TARGET_SR)
    n = min(len(x0), len(x1))
    x0, x1 = x0[:n], x1[:n]
    print(f"[OK] Common samples: {n}, duration ~{n/TARGET_SR:.2f}s")
    #the spectrograms
    try:
        f0, t0, S0 = make_spec(x0, TARGET_SR)
        f1, t1, S1 = make_spec(x1, TARGET_SR)
    except Exception as e:
        print("[ERR] Spectrogram failed:", e)
        sys.exit(1)
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

    #computes plot tmax from spec grid
    tmax0 = float(t0[-1]) if len(t0) else len(x0) / TARGET_SR
    tmax1 = float(t1[-1]) if len(t1) else len(x1) / TARGET_SR
    tmax = min(tmax0, tmax1)

    #for word spans
    words = None
    if WORDS_TEXTGRID:
        words = load_spans_from_textgrid(WORDS_TEXTGRID, ("words", "word", "Words", "Word"))
    if words is None and WORDS_CSV:
        words = load_words_from_csv(WORDS_CSV)

    #clips n overlay words
    words_clipped = debug_bounds("WORDS", words or [], tmax)
    if words_clipped:
        overlay_spans_clean(axs[0], words_clipped, y_top=YLIM, label_every=LABEL_EVERY_WORD, alpha=0.18, fontsize=8)
        overlay_spans_clean(axs[1], words_clipped, y_top=YLIM, label_every=LABEL_EVERY_WORD, alpha=0.18, fontsize=8)
    else:
        print("[INFO] No word overlay drawn.")

    #for optional phoneme spans
    if PHONES_TEXTGRID:
        phones = load_spans_from_textgrid(PHONES_TEXTGRID, ("phones", "phonemes", "Phones", "Phonemes"))
        phones_clipped = debug_bounds("PHONES", phones or [], tmax)
        if phones_clipped:
            overlay_spans_clean(axs[0], phones_clipped, y_top=YLIM, label_every=LABEL_EVERY_PHONE, alpha=0.08, fontsize=6)
            overlay_spans_clean(axs[1], phones_clipped, y_top=YLIM, label_every=LABEL_EVERY_PHONE, alpha=0.08, fontsize=6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
