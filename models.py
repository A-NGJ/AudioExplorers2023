from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model


def create_cnn_model(input_shape: tuple, num_classes: int, name: str = "CNN") -> Model:
    """
    Creates a CNN model with following layers:
    Conv2D -> MaxPooling2D -> Dropout -> Conv2D -> MaxPooling2D -> Dropout ->
    Flatten -> Dense -> Dropout -> Dense

    Parameters
    ----------
    input_shape : tuple
        Shape of input data
    num_classes : int
        Number of classes to predict
    """

    inputs = Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x, name=name)


def _resnet_block(
    input_data, num_filters, kernel_size=3, strides=1, conv_shortcut=False
):
    shortcut = input_data
    if conv_shortcut:
        shortcut = layers.Conv2D(num_filters, 1, strides=strides, padding="same")(
            input_data
        )
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(num_filters, kernel_size, strides=strides, padding="same")(
        input_data
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, kernel_size, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)

    return x


def create_resnet(
    input_shape: tuple, num_classes: int, name: str = "MiniResNet"
) -> Model:
    """
    Create a ResNet model.
    With following layers:
    - Conv2D
    - BatchNormalization
    - Activation
    - MaxPooling2D
    - ResNet block (16, 32, 64, 96 filters)
    - GlobalAveragePooling2D
    - Dense
    - Softmax

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data.
    num_classes : int
        Number of classes to predict.
    """
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _resnet_block(x, 16, conv_shortcut=True)
    x = _resnet_block(x, 16)

    x = _resnet_block(x, 32, conv_shortcut=True)
    x = _resnet_block(x, 32)

    x = _resnet_block(x, 64, conv_shortcut=True)
    x = _resnet_block(x, 64)

    x = _resnet_block(x, 96, conv_shortcut=True)
    x = _resnet_block(x, 96)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x, name=name)
    return model
