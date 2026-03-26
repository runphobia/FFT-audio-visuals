import numpy as np 
from pydub import AudioSegment
import os 
import pygame
import librosa

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(44100)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples())
    samples = samples.astype(np.float32)
    samples /= float(2 ** 15)
    file_name = os.path.basename(file_path)
    print(f"Loaded audio file: {file_name}")
    print(f"Duration: {len(audio) / 1000:.2f} seconds")
    print(f"Sample rate: {audio.frame_rate} Hz")
    return samples, audio.frame_rate

#
def detect_bpm(samples, sample_rate):
    print("Detecting BPM...")
    tempo, beats = librosa.beat.beat_track(y=samples, sr=sample_rate)
    bpm = float(np.squeeze(tempo))
    beat_times = librosa.frames_to_time(beats, sr=sample_rate)
    print(f"BPM: {bpm:.1f}")
    return bpm, beat_times

#Calculations for the spectrogrram using hanning windows and FFT to convert to decibel scale for visualization
def spectogrammath(samples, sample_rate, chunk_size=2048):
    window = np.hanning(chunk_size)
    hop = chunk_size // 2
    spectrogram = []
    for i in range(0, len(samples) - chunk_size, hop):
        chunk = samples[i:i + chunk_size]
        windowed_chunk = chunk * window
        fft = np.fft.rfft(windowed_chunk)
        magnitude = np.abs(fft)
        decibels = 20 * np.log10(magnitude + 1e-10)
        spectrogram.append(decibels)
    spectrogram = np.array(spectrogram).T
    print(f"Shape: {spectrogram.shape}, dB range: {np.min(spectrogram):.1f} to {np.max(spectrogram):.1f}")
    return spectrogram

def get_energy_over_time(samples, hop_size):
    rms = librosa.feature.rms(y=samples, frame_length=hop_size*2, hop_length=hop_size)[0]
    rms_max = rms.max() if rms.max() > 0 else 1.0
    return rms / rms_max

def decibel_to_color(t):
    # back to original colorful palette
    colors = [
        (0,   0,   0),#black 
        (0,   0,   255),#blue
        (0,   255, 255),#cyan
        (0,   255, 0),#green
        (255, 255, 0),#yellow
        (255, 0,   0),#red
    ]
    scaled = t * (len(colors) - 1)
    index = int(scaled)
    fraction = scaled - index
    if index >= len(colors) - 1:
        return colors[-1]
    c1, c2 = colors[index], colors[index + 1]
    r = int(c1[0] + fraction * (c2[0] - c1[0]))
    g = int(c1[1] + fraction * (c2[1] - c1[1]))
    b = int(c1[2] + fraction * (c2[2] - c1[2]))
    return (r, g, b)

def prerenderspectogram(spectrogram, heightt):
    frequency_bins, time_bins = spectrogram.shape
    surface = pygame.Surface((time_bins, heightt))

    normalized = np.zeros_like(spectrogram)
    for f in range(frequency_bins):
        row = spectrogram[f, :]
        row_min = np.percentile(row, 5)
        row_max = np.percentile(row, 95)
        if row_max > row_min:
            normalized[f, :] = (row - row_min) / (row_max - row_min)

    for x in range(time_bins):
        column = normalized[:, x]
        for y in range(heightt):
            freq_index = int((1 - y / heightt) * (frequency_bins - 1))
            t = float(np.clip(column[freq_index], 0.0, 1.0))
            color = decibel_to_color(t)
            surface.set_at((x, y), color)
        if x % 1000 == 0:
            print(f"Prerendering spectrogram: {x}/{time_bins}")

    print("Spectrogram prerender complete!")
    return surface

#Pre rendering the waveform chart
def prerenderwaveform(samples, sample_rate, heightt, hop_size):
    rms = librosa.feature.rms(y=samples, frame_length=hop_size*2, hop_length=hop_size)[0]
    total_frames = len(rms)
    surface = pygame.Surface((total_frames, heightt))
    surface.fill((15, 15, 20))

    mid = heightt // 2
    pygame.draw.line(surface, (35, 35, 50), (0, mid), (total_frames, mid), 1)

    rms_max = rms.max() if rms.max() > 0 else 1.0
    rms_normalized = rms / rms_max

    for x, energy in enumerate(rms_normalized):
        t = float(energy)
        bar_height = max(1, min(int(t * mid), mid - 1))

        # blue → cyan → white
        if t < 0.5:
            t2 = t * 2
            r = int(20  + t2 * 95) 
            g = int(80  + t2 * 90)
            b = int(180 + t2 * 30)
        else:
            t2 = (t - 0.5) * 2
            r = 255
            g = int(90  + t2 * 165)   # 90  -> 255
            b = int(30  + t2 * 225)   # 30  -> 255

        # glow — just draw wider dimmer lines, no alpha needed
        glow_h1 = min(bar_height + 4, mid - 1)
        glow_h2 = min(bar_height + 8, mid - 1)
        pygame.draw.line(surface, (r//4, g//4, b//4),
                         (x, mid - glow_h2), (x, mid + glow_h2), 3)
        pygame.draw.line(surface, (r//2, g//2, b//2),
                         (x, mid - glow_h1), (x, mid + glow_h1), 2)
        pygame.draw.line(surface, (r, g, b),
                         (x, mid - bar_height), (x, mid + bar_height), 1)

    print("Waveform prerender complete!")
    return surface

bar_heights = [0.0] * 128  # consistent smoothed state for bar heights

#drawing the frequency bars 
def draw_freq_bars(screen, x, y, width, height, spectrogram, scroll_x, smooth_energy, bar_heights):
    num_bars = 48
    freq_bins = spectrogram.shape[0]
    bar_w = width // num_bars

    #get current column of spectrogram based on scroll position
    col_idx = min(scroll_x, spectrogram.shape[1] - 1)
    column = spectrogram[:, col_idx]
    col_min = column.min()
    col_max = column.max()
    if col_max > col_min:
        column = (column - col_min) / (col_max - col_min)#normalize current column for bar heights

    pygame.draw.rect(screen, (10, 10, 15), (x, y, width, height))

    for i in range(num_bars):
        freq_idx = int((i / num_bars) * (freq_bins - 1))
        target = float(np.clip(column[freq_idx], 0.0, 1.0))

        # slow rise, even slower fall — big lazy movements
        if target > bar_heights[i]:
            bar_heights[i] += (target - bar_heights[i]) * 0.5  # slow rise
        else:
            bar_heights[i] += (target - bar_heights[i]) * 0.1  # very slow fall

        t = bar_heights[i]
        bar_h = max(2, int(t * height * 0.75))
        bx = x + i * bar_w
        by = y + height - bar_h

        brightness = 0.6 + t * 0.4 # base brightness + extra from energy
        r = int(20  * brightness)
        g = int(180 * brightness)
        b = int(220 * brightness)

        pygame.draw.rect(screen, (r, g, b), (bx + 1, by, bar_w - 2, bar_h))
        pygame.draw.rect(screen, (min(r+40,255), min(g+40,255), min(b+40,255)),
                         (bx + 1, by, bar_w - 2, 2))

    pygame.draw.line(screen, (50, 50, 70), (x, y), (x + width, y), 1)

# details bar at the top with song name, time, and BPM
def draw_details_bar(screen, width, height, font, song_name, playback_ms, total_ms, bpm):
    pygame.draw.rect(screen, (18, 18, 24), (0, 0, width, height))
    pygame.draw.line(screen, (40, 40, 55), (0, height - 1), (width, height - 1), 1)

    name_surf = font.render(song_name, True, (200, 200, 220))
    screen.blit(name_surf, (16, height // 2 - name_surf.get_height() // 2))

    current = int(playback_ms / 1000)
    total   = int(total_ms / 1000)
    time_str = f"{current // 60}:{current % 60:02d}  /  {total // 60}:{total % 60:02d}"
    time_surf = font.render(time_str, True, (120, 120, 150))
    screen.blit(time_surf, (width // 2 - time_surf.get_width() // 2,
                            height // 2 - time_surf.get_height() // 2))

    bpm_surf = font.render(f"{bpm:.1f} BPM", True, (80, 180, 200))
    screen.blit(bpm_surf, (width - bpm_surf.get_width() - 16,
                           height // 2 - bpm_surf.get_height() // 2))

#Setting and drawing the frequency labels for reference 100Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz
def draw_freq_labels(screen, spec_top, spec_height, sample_rate, frequency_bins, font):
    freq_markers = [100, 500, 1000, 2000, 4000, 8000, 16000]
    for hz in freq_markers:
        bin_index = int(hz / (sample_rate / 2) * frequency_bins)
        y = spec_top + int((1 - bin_index / frequency_bins) * spec_height)
        if spec_top <= y <= spec_top + spec_height:
            pygame.draw.line(screen, (35, 35, 50), (0, y), (screen.get_width(), y), 1)
            label = font.render(f"{hz}Hz", True, (70, 70, 90))
            screen.blit(label, (4, y - 8))


#main loop handles all the drawing and updates based on current playback position and energy
def main_loop(screen, spec_surface, wave_surface, spectrogram, energy,
              width, height, sample_rate, hop_size, song_name, total_ms, bpm):

    details_h  = int(height * 0.08)
    wave_h     = int(height * 0.30)
    freq_bar_h = int(height * 0.22)
    spec_h     = int(height * 0.40)

    details_top  = 0
    wave_top     = details_h
    freq_bar_top = wave_top + wave_h
    spec_top     = freq_bar_top + freq_bar_h

    frequency_bins = 1025
    timer      = pygame.time.Clock()
    font_large = pygame.font.SysFont("monospace", 15)
    font_small = pygame.font.SysFont("monospace", 11)

    total_spec_frames = spec_surface.get_width()
    total_wave_frames = wave_surface.get_width()
    smooth_energy = 0.0
    bar_heights = [0.0] * 128  # consistent smoothed state for bar heights
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        if not pygame.mixer.music.get_busy():
            pygame.quit()
            return

        playback_ms  = pygame.mixer.music.get_pos()
        playback_sec = playback_ms / 1000.0
        scroll_x     = int(playback_sec * sample_rate / hop_size)
        scroll_x     = max(0, min(scroll_x, total_spec_frames - width))

        energy_idx    = min(scroll_x, len(energy) - 1)
        current_energy = float(energy[energy_idx])
        smooth_energy  = smooth_energy * 0.92 + current_energy * 0.08

        screen.fill((15, 15, 20))

        # details
        draw_details_bar(screen, width, details_h, font_large,
                         song_name, playback_ms, total_ms, bpm)

        # waveform
        wave_scroll = max(0, min(scroll_x - width // 2, total_wave_frames - width))
        screen.blit(wave_surface, (0, wave_top), (wave_scroll, 0, width, wave_h))
        pygame.draw.line(screen, (200, 200, 220),
                         (width // 2, wave_top), (width // 2, wave_top + wave_h), 1)
        pygame.draw.line(screen, (40, 40, 55), (0, wave_top), (width, wave_top), 1)

        # frequency bars
        draw_freq_bars(screen, 0, freq_bar_top, width, freq_bar_h,
               spectrogram, scroll_x, smooth_energy, bar_heights)

        # spectrogram
        spec_scroll = max(0, min(scroll_x, total_spec_frames - width))
        screen.blit(spec_surface, (0, spec_top), (spec_scroll, 0, width, spec_h))
        draw_freq_labels(screen, spec_top, spec_h, sample_rate, frequency_bins, font_small)
        pygame.draw.line(screen, (40, 40, 55), (0, spec_top), (width, spec_top), 1)

        pygame.display.flip()
        timer.tick(60)

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    print(f"Playing: {os.path.basename(file_path)}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

    chunk_size = 2048
    hop_size   = chunk_size // 2

    samples, rate = load_audio(path)
    total_ms  = len(samples) / rate * 1024
    song_name = os.path.splitext(os.path.basename(path))[0]
    bpm, beat_times = detect_bpm(samples, rate)

    spectrogram = spectogrammath(samples, rate, chunk_size)
    energy      = get_energy_over_time(samples, hop_size)

    pygame.init()
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Spectrogram")

    spec_h = int(HEIGHT * 0.40)
    wave_h = int(HEIGHT * 0.30)

    spec_surface = prerenderspectogram(spectrogram, spec_h)
    wave_surface = prerenderwaveform(samples, rate, wave_h, hop_size)

    play_audio(path)
    print("Running... Press ESC to quit.")
    main_loop(screen, spec_surface, wave_surface, spectrogram, energy,
              WIDTH, HEIGHT, rate, hop_size, song_name, total_ms, bpm)