# FFT-audio-visuals
 A real-time audio visualizer built in Python using Pygame. Supports MP3, WAV, and FLAC files.
# Spectrogram Audio Visualizer

A real-time audio visualizer built in Python using Pygame. Supports MP3, WAV, and FLAC files.

## Features
- **Scrolling spectrogram** — per-frequency normalized FFT visualization synced to playback
- **Waveform display** — RMS energy waveform with glow, scrolls with a center playhead
- **Frequency bars** — smoothed live frequency bars with lazy rise/fall animation

## Install 
```bash
pip install pygame numpy pydub librosa
```
Also install ffmpeg
## Usage
```bash
python Spectogram.py "path/to/your/song.flac"
python Spectogram.py "path/to/your/song.mp3"
python Spectogram.py "path/to/your/song.wav"
```
## How it works
Audio is decoded into raw PCM samples, chunked into 2048-sample windows with 50% overlap, and passed through an FFT to get frequency content at each moment in time. Each frequency bin is normalized independently so bass-heavy and treble-heavy songs both display with full contrast. The result is pre-rendered into a pygame surface and scrolled in sync with audio playback via `pygame.mixer.music.get_pos()`.

<img width="1204" height="817" alt="image" src="https://github.com/user-attachments/assets/d45c20f5-e183-4d88-a2ce-2401f40b4fdb" />

