import numpy as np

import augment


def test_time_shift():
    mel = np.random.randn(32, 96)
    mel_shifted = augment.time_shift(mel)
    assert mel_shifted.shape == mel.shape
    assert mel_shifted.sum() != mel.sum()


def test_time_stretch():
    mel = np.random.randn(32, 96)
    mel_stretched = augment.time_stretch(mel)
    assert mel_stretched.shape == mel.shape
    assert mel_stretched.sum() != mel.sum()


def test_pitch_shift():
    mel = np.random.randn(32, 96)
    mel_shifted = augment.pitch_shift(mel)
    assert mel_shifted.shape == mel.shape
    assert mel_shifted.sum() != mel.sum()


def test_add_noise():
    mel = np.random.randn(32, 96)
    mel_noisy = augment.add_noise(mel)
    assert mel_noisy.shape == mel.shape
    assert mel_noisy.sum() != mel.sum()
