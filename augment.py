# Google Colab usage
# 1. Store augment.py in your Google Drive
# 2. Mount your Google Drive
# 3. Copy it by !cp /content/drive/MyDrive/augment.py .
# 4. import augment

import librosa
import numpy as np


def time_shift(mel: np.ndarray, shift_range: float = 0.2) -> np.ndarray:
    """
    Shift the time axis of a spectrogram randomly within the given range.

    Parameters
    ----------
    mel : np.ndarray
        Input spectrogram of shape (n_channels, n_frames)
    shift_range : float
        Range of time shifting in seconds.
    """
    n_frames = mel.shape[1]
    shift = int(n_frames * np.random.uniform(-shift_range, shift_range))
    mel = np.roll(mel, shift, axis=1)
    if shift > 0:
        mel[:, :shift] = 0
    else:
        mel[:, shift:] = 0
    return mel


def time_stretch(
    mel: np.ndarray, rate_range=(0.8, 1.25), n_fft: int = 64
) -> np.ndarray:
    """
    Stretch the time axis of a spectrogram randomly within the given range.
    Parameters
    ----------
    mel : np.ndarray
        Input spectrogram of shape (n_channels, n_frames)
    rate_range : tuple
        Range of time stretching rate.
    """
    n_frames = mel.shape[1]
    rate = np.random.uniform(*rate_range)
    mel = librosa.effects.time_stretch(mel, rate=rate, n_fft=n_fft)
    if mel.shape[1] > n_frames:
        mel = mel[:, :n_frames]
    else:
        mel = np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])), mode="constant")
    return mel


def add_noise(mel: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    """
    Add noise to a spectrogram randomly within the given range.

    Parameters
    ----------
    mel : np.ndarray
        Input spectrogram.
    """
    noise = np.random.randn(*mel.shape) * noise_factor
    return mel + noise


def pitch_shift(
    mel: np.ndarray, sr: int = 16000, n_steps: int = 4, n_fft: int = 64
) -> np.ndarray:
    """
    Shift the pitch of a spectrogram randomly within the given range.

    Parameters
    ----------
    mel : np.ndarray
        Input spectrogram.
    sr : int
        Sampling rate.
    n_steps : int
        Range of pitch shifting in semitones.
    n_fft : int
        Number of FFT components.
    """
    return librosa.effects.pitch_shift(
        mel,
        sr=sr,
        n_steps=n_steps * np.random.uniform(-1, 1),
        n_fft=n_fft,
    )
