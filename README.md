# tESpeech
Computational framework mapping speech into tRNS-compatible signals for transcranial electrical stimulation

# About this Project  
This repository presents the first proof-of-concept system for transforming continuous speech into transcranial random noise stimulation signals (tRNS). The long-term vision is to enable a direct neurostimulation-based channel for true speech perception, a communication pathway for individuals with profound hearing loss who cannot benefit from cochlear implants.  

By combining speech processing, forced alignment, phoneme encoding, and signal synthesis, this framework demonstrates how speech can be mapped into neuro-compatible waveforms that leverage the brain’s adaptability (neuroplasticity) to learn novel stimulation patterns as meaningful language.  

# Overview  
The pipeline takes raw audio, generates transcripts (via Whisper), aligns phonemes (via Montreal Forced Aligner), and maps them into onset-nucleus-coda segmented units. These units are then converted into tRNS-compatible waveforms, which can be visualized and compared to the original speech via spectrogram analysis.  

# Repository Contents:
- `harvard.wav` - Sample input audio (6 sample Harvard Sentences, open source)  
- `harvard.txt` - Transcript for forced alignment (6 sample Harvard Sentence transcriptions, corresponding to `.wav`)  
- `harvard.TextGrid` - Forced alignment output (from MFA), viewable in **Praat** to show spectrogram/phoneme alignment  
- `harvard_preview_onset_coda.wav` - Example of synthesized output  
- `harvard_step1_table.csv` - Intermediate encoding table  
- `compare_spectrograms.py` - Spectrogram comparison tool  
- `compare_spectrogramsWords.py` - Word-level spectrogram analysis  
- `synth_modulate.py` - Signal synthesis and modulation  
- `table1.py` - Symbol mapping table generator  
- `LICENSE` - MIT License  
- `README.md` — Project documentation  

#
LICENSE # MIT License
README.md # Project documentation

# Author  
- **Sophia Pannu** — Lead researcher & developer  

# Acknowledgments  
- **Timothy Scargill (Duke University)** — Research mentor and advisor  
