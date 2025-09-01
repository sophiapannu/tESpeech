#synth_modulate.py
#read harvard.TextGrid (phonemes tier) to multiband vowel carriers n onset/coda consonant modulations to WAV

import re
from pathlib import Path
import numpy as np
import soundfile as sf
from textgrid import TextGrid

TEXTGRID_PATH = Path("harvard.TextGrid")  
TIER_NAME = "phones"
SR = 48000
GAIN = 0.9
OUT_WAV = "harvard_preview_onset_coda.wav"

#vowel carrier bands // non-overlapping
VOWEL_BANDS = {
    "AA": (250, 450),  "AE": (450, 650),  "AH": (650, 850),  "AO": (850, 1050),
    "AW": (1050,1250), "AY": (1250,1450), "EH": (1450,1650), "ER": (1650,1850),
    "EY": (1850,2050), "IH": (2050,2250), "IY": (2250,2450), "OW": (2450,2650),
    "OY": (2650,2850), "UH": (2850,3050), "UW": (3050,3250),
}
VOWELS = set(VOWEL_BANDS.keys())
#consonant groupings (ARPAbet??), onset/coda modulation
BLANK = {"", "SP", "SIL", "SPN"}       #group 1: none (silence)
PTK   = {"P","T","K"}                  #group 2: sharp gaussian impulse (10–20 ms)
F_TH_S_SH = {"F","TH","S","SH"}        #group 3: 30–50 Hz tremolo (50–100 ms)
BDG   = {"B","D","G"}                  #group 4: soft low-freq AM pulse (10ish Hz, 30ish ms)
VZ    = {"V","Z"}                      #group 5: mid-rate AM 15–25 Hz (100ish ms trailing)
MN    = {"M","N"}                      #group 6: smooth raised-cosine fade (100ish ms)

#envelope constants
FADE_MS_VOWEL = 6       #gen vowel attack/release
GAUSS_MS_PTK  = 15      #10–20 ms
GAUSS_AMP_PTK = 1.0
TREMO_HZ      = 40.0    #fricatives 30–50 Hz
TREMO_MS      = 60      #50–100 ms
TREMO_DEPTH   = 0.35
PULSE_HZ_BDG  = 10.0    #voiced stops 5–15ish Hz
PULSE_MS_BDG  = 30
PULSE_GAIN_BDG= 0.6
MID_HZ_VZ     = 20.0    #v/z 15–25ish Hz
MID_MS_VZ     = 100
MID_DEPTH_VZ  = 0.30
SMOOTH_MS_MN  = 100     #m/n raised cosine

#utils
def base_phone(label: str) -> str:
    return re.sub(r"\d$", "", str(label).strip().upper())
def is_vowel(label: str) -> bool:
    return base_phone(label) in VOWELS
def band_limited_noise(n, sr, lo, hi, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0/sr)
    mask = (freqs >= lo) & (freqs <= hi)
    Xf = np.zeros_like(X); Xf[mask] = X[mask]
    y = np.fft.irfft(Xf, n=n).astype(np.float32)
    y /= (np.max(np.abs(y)) + 1e-8)
    return y
def window_indices(s_samp, e_samp, win_samp, edge, N):
    if edge == "onset":
        i0 = s_samp; i1 = min(e_samp, s_samp + win_samp)
    else:  #coda
        i1 = e_samp; i0 = max(s_samp, e_samp - win_samp)
    i0 = max(0, min(N, i0)); i1 = max(0, min(N, i1))
    return i0, i1
def raised_cosine(n):
    if n <= 1: return np.ones(n, dtype=np.float32)
    x = np.linspace(0, np.pi, n, dtype=np.float32)
    return 0.5*(1 - np.cos(x))

#loading textgrid
tg = TextGrid.fromFile(str(TEXTGRID_PATH))
tier = next((t for t in tg.tiers if t.name == TIER_NAME), None)
if tier is None:
    raise SystemExit(f"Tier '{TIER_NAME}' not found. Available: {[t.name for t in tg.tiers]}")

intervals = [(float(it.minTime), float(it.maxTime), (it.mark or "").strip()) for it in tier.intervals]
T_sec = max((e for _, e, _ in intervals), default=0.0)
N = int(round(T_sec * SR))

labels = [lab for _, _, lab in intervals]
starts = [int(round(s*SR)) for s,_,_ in intervals]
ends   = [int(round(e*SR)) for _,e,_ in intervals]

def prev_nonblank(i):
    j = i-1
    while j >= 0 and base_phone(labels[j]) in BLANK:
        j -= 1
    return j

def next_nonblank(i):
    j = i+1
    while j < len(labels) and base_phone(labels[j]) in BLANK:
        j += 1
    return j

#build carriers n envelopes
#one noise carrier per vowel band
carriers = {v: band_limited_noise(N, SR, *VOWEL_BANDS[v], seed=hash(v) % (2**32)) for v in VOWELS}
g_band   = {v: np.zeros(N, dtype=np.float32) for v in VOWELS}
overlay  = np.zeros(N, dtype=np.float32)  #multiplicative overlay (>=0)
#vowels: steady on w soft edges
fade_v = int(FADE_MS_VOWEL/1000 * SR)
for idx, lab in enumerate(labels):
    b = base_phone(lab)
    if is_vowel(lab):
        s, e = starts[idx], ends[idx]
        if e > s and b in g_band:
            env = np.ones(e - s, dtype=np.float32)
            fi = min(fade_v, len(env)//2)
            if fi > 1:
                env[:fi] *= np.linspace(0, 1, fi, endpoint=False, dtype=np.float32)
                env[-fi:] *= np.linspace(1, 0, fi, endpoint=False, dtype=np.float32)
            g_band[b][s:e] = np.maximum(g_band[b][s:e], env)

#consonant mods at onset/coda windows
for i, lab in enumerate(labels):
    b = base_phone(lab)
    s, e = starts[i], ends[i]
    if e <= s or b in BLANK or is_vowel(lab):
        continue
    j_prev = prev_nonblank(i)
    j_next = next_nonblank(i)
    is_onset = (j_next != -1 and j_next < len(labels) and is_vowel(labels[j_next]))
    is_coda  = (j_prev != -1 and j_prev >= 0 and is_vowel(labels[j_prev]))
    #group 2: P T K to gaussian impulse (10–20 ms) at onset/coda
    if b in PTK:
        win = int(GAUSS_MS_PTK/1000 * SR)
        sigma = max(1, win // 4)
        def place_gauss(i0,i1):
            n = i1 - i0
            if n <= 0: return
            x = np.linspace(-1, 1, n, dtype=np.float32)
            gauss = GAUSS_AMP_PTK * np.exp(-0.5 * (x * (win/(2.0*sigma)))**2)
            overlay[i0:i1] = np.maximum(overlay[i0:i1], gauss)
        if is_onset:
            i0,i1 = window_indices(s,e,win,"onset",N); place_gauss(i0,i1)
        if is_coda:
            i0,i1 = window_indices(s,e,win,"coda",N);  place_gauss(i0,i1)
        continue
    #group 3: F TH S SH to high‑rate AM (30–50 Hz) 50–100ish ms at onset/coda
    if b in F_TH_S_SH:
        win = int(TREMO_MS/1000 * SR)
        def place_trem(i0,i1):
            n = i1 - i0
            if n <= 0: return
            tloc = np.arange(n, dtype=np.float32)/SR
            trem = TREMO_DEPTH * 0.5 * (1.0 + np.sin(2*np.pi*TREMO_HZ*tloc))
            #light 6 ms taper inside window
            fi = min(n//2, int(6/1000*SR))
            if fi > 1:
                trem[:fi] *= np.linspace(0,1,fi,endpoint=False,dtype=np.float32)
                trem[-fi:] *= np.linspace(1,0,fi,endpoint=False,dtype=np.float32)
            overlay[i0:i1] = np.maximum(overlay[i0:i1], trem)
        if is_onset:
            i0,i1 = window_indices(s,e,win,"onset",N); place_trem(i0,i1)
        if is_coda:
            i0,i1 = window_indices(s,e,win,"coda",N);  place_trem(i0,i1)
        continue
    #group 4: B D G to soft low‑freq AM pulse (10ish Hz, 30ishh ms) onset/coda
    if b in BDG:
        win = int(PULSE_MS_BDG/1000 * SR)
        def place_pulse(i0,i1):
            n = i1 - i0
            if n <= 0: return
            tloc = np.arange(n, dtype=np.float32)/SR
            pulse = PULSE_GAIN_BDG * 0.5 * (1.0 + np.sin(2*np.pi*PULSE_HZ_BDG*tloc))
            pulse *= (1.0 - np.arange(n, dtype=np.float32)/max(1,n))  #decay
            overlay[i0:i1] = np.maximum(overlay[i0:i1], pulse)
        if is_onset:
            i0,i1 = window_indices(s,e,win,"onset",N); place_pulse(i0,i1)
        if is_coda:
            i0,i1 = window_indices(s,e,win,"coda",N);  place_pulse(i0,i1)
        continue
    #group 5: V Z to mid‑rate AM (15–25 Hz) on trailing edge (100ish ms) 
    if b in VZ:
        win = int(MID_MS_VZ/1000 * SR)
        i0,i1 = window_indices(s,e,win,"coda",N)
        n = i1 - i0
        if n > 0:
            tloc = np.arange(n, dtype=np.float32)/SR
            mid = MID_DEPTH_VZ * 0.5 * (1.0 + np.sin(2*np.pi*MID_HZ_VZ*tloc))
            fi = min(n//2, int(6/1000*SR))
            if fi > 1:
                mid[:fi] *= np.linspace(0,1,fi,endpoint=False,dtype=np.float32)
                mid[-fi:] *= np.linspace(1,0,fi,endpoint=False,dtype=np.float32)
            overlay[i0:i1] = np.maximum(overlay[i0:i1], mid)
        continue
    #group 6: M N to smooth raised‑cosine fade (100ish ms) at onset and coda
    if b in MN:
        win = int(SMOOTH_MS_MN/1000 * SR)
        def place_rc(i0,i1):
            n = i1 - i0
            if n <= 0: return
            rc = raised_cosine(n) * 0.25
            overlay[i0:i1] = np.maximum(overlay[i0:i1], rc)
        if is_onset:
            i0,i1 = window_indices(s,e,win,"onset",N); place_rc(i0,i1)
        if is_coda:
            i0,i1 = window_indices(s,e,win,"coda",N);  place_rc(i0,i1)
        continue
    #else: leave unchanged

#render mix
y = np.zeros(N, dtype=np.float32)
for v, g in g_band.items():
    if np.any(g):
        y += carriers[v] * g

#multiplicative overlay (>=0): final =base * (1 + overlay)
y *= (1.0 + np.clip(overlay, 0.0, 1.5))
y /= (np.max(np.abs(y)) + 1e-8)
y = (GAIN * y).astype(np.float32)

sf.write(OUT_WAV, y, SR, subtype="PCM_16")
print(f"Saved {OUT_WAV} @ {SR} Hz")
