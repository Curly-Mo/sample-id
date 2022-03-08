import librosa
import numpy as np


def cqtgram(y, sr, hop_length=512, octave_bins=24, n_octaves=8, fmin=40, perceptual_weighting=False):
    s_complex = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
    )
    specgram = np.abs(s_complex)
    if perceptual_weighting:
        freqs = librosa.cqt_frequencies(specgram.shape[0], fmin=fmin, bins_per_octave=octave_bins)
        specgram = librosa.perceptual_weighting(specgram**2, freqs, ref=np.max)
    else:
        specgram = librosa.amplitude_to_db(specgram, ref=np.max)
    return specgram


def chromagram(y, hop_length=512, n_fft=1024, n_chroma=12):
    specgram = librosa.feature.chroma_stft(y, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    return specgram
