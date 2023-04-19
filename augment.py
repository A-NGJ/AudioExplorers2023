# Google Colab usage
# 1. Store augment.py in your Google Drive
# 2. Mount your Google Drive
# 3. Copy it by !cp /content/drive/MyDrive/augment.py .
# 4. import augment

from collections import Counter
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


def augment_with_ratio(train_data: np.ndarray, train_labels: np.ndarray, ratio: float):
    """
    Augment the training data with a given ratio.

    Parameters
    ----------
    train_data : np.ndarray
        Training data.
    train_labels : np.ndarray
        Training labels.
    ratio : float
        Ratio of augmented data to original data.
    """
    n_samples = len(train_data)

    # Augment class data
    train_data_time_shift = np.array(
        [time_shift(mel, shift_range=0.5) for mel in train_data]
    )
    train_data_time_stretch = np.array(
        [time_stretch(mel, rate_range=(0.5, 1.5)) for mel in train_data]
    )
    train_data_add_noise = np.array(
        [add_noise(mel, noise_factor=0.5) for mel in train_data]
    )
    train_data_pitch_shift = np.array([pitch_shift(mel) for mel in train_data])

    # Concatenate the augmented data
    train_data_aug = np.concatenate(
        (
            train_data_time_shift,
            train_data_time_stretch,
            train_data_add_noise,
            train_data_pitch_shift,
        )
    )

    # Concatenate the augmented labels
    train_labels_aug = np.concatenate((train_labels,) * 4)

    # Pick randomly n_samples * ratio samples from the augmented data
    n_augmented_samples = int(n_samples * (ratio - 1))
    if n_augmented_samples > len(train_data_aug):
        # Select all indieces
        indices = np.arange(len(train_data_aug))
    else:
        indices = np.random.choice(
            len(train_data_aug), n_augmented_samples, replace=False
        )

    return (
        np.concatenate((train_data, train_data_aug[indices])),
        np.concatenate((train_labels, train_labels_aug[indices])),
    )
