from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf

keras = tf.keras
from keras.callbacks import EarlyStopping
from keras.metrics import (
    Precision,
    Recall,
)
from keras.optimizers import Adam
import augment
from models import ModelFactory
import os
from tensorflow.keras.optimizers.schedules import ExponentialDecay

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s")


def stratify_train_test_split(
    X: np.ndarray, y: np.ndarray, test_samples: int, n_classes: int, random_state: int
) -> tuple:
    """
    Split the data into training and testing sets
    Where in test set there is the same number of samples from each class

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Input labels.
    test_samples : int
        Number of samples in test set, equal for each class.
    n_classes : int
        Number of classes.
    random_state : int
        Random seed.

    Returns
    -------
    train_data : np.ndarray
        Training data.
    train_labels : np.ndarray
        Training labels.
    test_data : np.ndarray
        Testing data.
    test_labels : np.ndarray
        Testing labels.
    """

    # Split the data into training and testing sets
    # Where in test set there is the same number of samples from each class
    _, test_data, _, test_labels, _, test_indices = train_test_split(
        X,
        y,
        np.arange(y.shape[0]),
        test_size=(test_samples * n_classes / y.shape[0]),
        stratify=y,
        random_state=random_state,
    )

    # Remove test_indices from train_data and train_labels
    train_data = np.delete(X, test_indices, axis=0)
    train_labels = np.delete(y, test_indices, axis=0)

    return train_data, test_data, train_labels, test_labels


def main(args):
    # Create a directory to save the model
    save_model_dir = Path(args.save_dir)
    save_model_dir.mkdir(parents=True, exist_ok=True)

    save_results_dir = Path(args.results_dir)
    save_results_dir.mkdir(parents=True, exist_ok=True)

    data = loadmat(args.train_data)["data"]
    labels = loadmat(args.train_labels)["data"][0]
    n_classes = np.unique(labels).shape[0]
    class_counter = Counter(labels)

    # One-hot encode the labels
    labels = keras.utils.to_categorical(labels)

    train_data, test_data, train_labels, test_labels = stratify_train_test_split(
        data,
        labels,
        args.test_samples,
        n_classes,
        42,
    )

    train_data, val_data, train_labels, val_labels = stratify_train_test_split(
        train_data,
        train_labels,
        args.val_samples,
        n_classes,
        42,
    )

    if args.augment:
        # Decode one-hot encoded labels
        train_labels_decoded = np.argmax(train_labels, axis=1)

        most_common_class_n = class_counter.most_common(1)[0][1]

        # calculate the ratio of each class to the most common class and store in a dictionary
        class_ratios = {k: most_common_class_n / v for k, v in class_counter.items()}

        # Augment each class by the ratio of the most common class
        for class_, ratio in class_ratios.items():
            logging.info(f"Augmenting class {class_} by a factor of {ratio}")
            class_indices = np.where(train_labels_decoded == class_)[0]
            class_data = train_data[class_indices]
            class_labels = train_labels[class_indices]

            # Augment the data
            class_data, class_labels = augment.augment_with_ratio(
                class_data,
                class_labels,
                ratio,
            )

            # Concatenate the augmented data with the training data
            train_data = np.concatenate((train_data, class_data))

            # Concatenate the augmented labels with the class labels
            train_labels = np.concatenate((train_labels, class_labels))

    if train_labels.shape[0] != train_data.shape[0]:
        raise ValueError(
            "Number of training labels does not match number of training data"
        )

    # Create EarlyStopping callback on validation loss
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )

    model_factory = ModelFactory((*train_data.shape[1:], 1), train_labels.shape[1])
    model = model_factory.create_model(args.model_type)
    if args.lr_decay:
        lr_schedule = ExponentialDecay(initial_learning_rate=args.lr, decay_steps=10000, decay_rate=0.9)
    # optimizer = Adam(learning_rate=args.lr)
    else: lr_schedule = args.lr
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print(model.summary())

    # Train the model
    model.fit(
        train_data,
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(val_data, val_labels),
        callbacks=[early_stopping],
        shuffle=True,
    )

    # Save the model
    model.save(save_model_dir / args.save_name)

    results = {}

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    results["loss"] = test_loss
    results["accuracy"] = test_acc

    precision = Precision()
    precision.update_state(test_labels, model.predict(test_data))
    results["precision"] = float(precision.result().numpy())

    recall = Recall()
    recall.update_state(test_labels, model.predict(test_data))
    results["recall"] = float(recall.result().numpy())

    # Read results file if it exists
    try:
        results_file_path = os.path.join(args.results_dir, args.results_name)
        with open(save_results_dir / args.results_name , "r") as f:
            results_json = json.load(f)
    except FileNotFoundError:
        results_json = []

    # Add new results to the results file
    results_json.append(
        {
            "args": vars(args),
            "results": results,
        }
    )
    with open(save_results_dir / args.results_name , "w") as f:
        json.dump(results_json, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, help="Path to training data", required=True
    )
    parser.add_argument(
        "--train-size",
        type=float,
        help="Size of training data",
        default=0.6,
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        help="Number of samples in validation set, equal for each class",
        default=400,
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        help="Number of samples in test set, equal for each class",
        default=400,
    )
    parser.add_argument(
        "--train-labels", type=str, help="Path to training labels", required=True
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Type of model to train",
        required=True,
        choices=["CNN", "MiniResNet", "CNN_1D", "CNN_optimized", "Transformer"],
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lr-decay", action="store_true", 
            help="choose to have exponential decay learning rate",
            )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument("--augment", action="store_true", help="Augment data")
    parser.add_argument(
        "--results-dir", 
        type=str,
          help="Path to save results", 
        default="saved_results",
    )
    parser.add_argument(
        "--results-name", 
        type=str, 
        help="Name to save results <model_type>_<timestamp>.json", 
    
    )
    parser.add_argument(
        "--save-name",
        type=str,
        help="Name to save the model <model_type>_<timestamp>.h5",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save the model",
        default="saved_models",
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="Comment to add to results",
        default="",
    )

    args = parser.parse_args()

    default_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not args.save_name:
        args.save_name = (
            default_name + ".h5"
        )

    if not args.results_name:
        args.results_name = (
            default_name + ".json"
        )

    logging.info(f"Model name: {args.save_name}")
    main(args)


# Sample command:
# python run.py --train-data training.mat --train-labels training_labels.mat --model-type MiniResNet  --patience 10  --epochs 2
