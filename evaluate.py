import sys
from argparse import (
    ArgumentParser,
    Namespace,
)
import numpy as np
from scipy.io import loadmat

import tensorflow as tf

keras = tf.keras
from keras.models import load_model


def main(args: Namespace):
    # load tensorflow model
    model = load_model(args.model)
    if model is None:
        print("Error: Could not load model")
        sys.exit(1)
    print(model.summary())

    # load test data
    test_data = loadmat(args.test_data)["data"]

    # predict labels
    predictions = model.predict(test_data)
    predictions = np.argmax(predictions, axis=1)

    # save predictions to .txt file
    np.savetxt("predictions.txt", predictions, fmt="%d")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to model")
    parser.add_argument("--test-data", type=str, help="Path to test data")

    args = parser.parse_args()

    main(args)
