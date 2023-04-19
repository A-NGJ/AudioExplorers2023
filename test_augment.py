from collections import Counter
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


def test_augment_with_ratio():
    data = np.random.randn(10, 32, 96)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])

    class_counter = Counter(labels)
    most_common_class_n = class_counter.most_common(1)[0][1]

    class_ratios = {k: most_common_class_n / v for k, v in class_counter.items()}

    # Augment each class by the ratio of the most common class
    for class_, ratio in class_ratios.items():
        class_indices = np.where(labels == class_)[0]
        class_data = data[class_indices]
        class_labels = labels[class_indices]

        class_data, class_labels = augment.augment_with_ratio(
            class_data,
            class_labels,
            ratio,
        )

        assert class_data.shape[0] == class_labels.shape[0]
        # Check if the number of samples is equal to the expected number of samples
        # with a tolerance of 10%
        assert (
            most_common_class_n - 1 <= class_data.shape[0] <= most_common_class_n + 1
        ), f"Expected {most_common_class_n} samples, got {class_data.shape[0]} for class {class_}"
