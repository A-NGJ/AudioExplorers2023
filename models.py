import tensorflow as tf

keras = tf.keras
from keras import layers, Input
from keras.models import Model


class ModelFactory:
    """
    Factory class for creating models.
    """

    def __init__(self, input_shape: tuple, num_classes: int):
        """
        Parameters
        ----------
        input_shape : tuple
            Shape of input data
        num_classes : int
            Number of classes to predict
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_model(self, model_type: str) -> Model:
        """
        Create a model of type `model_type`.

        Parameters
        ----------
        model_type : str
            Type of model to create

        Returns
        -------
        Model
            The created model
        """
        if model_type == "CNN":
            return create_cnn_model(self.input_shape, self.num_classes)
        elif model_type == "MiniResNet":
            return create_resnet(self.input_shape, self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


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


def create_cnn_model_optimized(input_shape: tuple, num_classes: int, name: str = "CNN") -> Model:
    """
    Creates a CNN model 
    on augmented data 89.8% accuracy
    Parameters
    ----------
    input_shape : tuple
        Shape of input data
    num_classes : int
        Number of classes to predict
    """

    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu') (inputs),
    x = layers.BatchNormalization()(x),
    x = layers.MaxPooling2D((2, 2))(x),

    x = layers.Conv2D(64, (3, 3), activation='relu')(x),
    x = layers.BatchNormalization()(x),
    x = layers.MaxPooling2D((2, 2))(x),

    x = layers.Conv2D(128, (3, 3), activation='relu')(x),
    x = layers.BatchNormalization()(x),
    x = layers.MaxPooling2D((2, 2))(x),

    x = layers.Flatten()(x),
    x = layers.Dense(128, activation='relu', kernel_regularizer = regularizers.l2( l=0.01))(x),
    x = layers.Dropout(0.5)(x),
    x = layers.Dense(5, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x, name=name)

def create_1Dcnn_model(input_shape: tuple, num_classes: int, name: str = "CNN") -> Model:
    """
    Creates a 1D CNN model 
    on augmented data 84% accuracy
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input data
    num_classes : int
        Number of classes to predict
    """

    inputs = Input(shape=input_shape)
    x = layers.Reshape((32, 96), input_shape=inputs),

    x = layers.Conv1D(32, 3, activation='relu')(x),
    x = layers.BatchNormalization()(x),
    x = layers.MaxPooling1D(2)(x),

    x = layers.Conv1D(64, 3, activation='relu')(x), 
    x = layers.BatchNormalization()(x),
    x = layers.MaxPooling1D(2)(x),

    x = layers.Conv1D(128, 3, activation='relu')(x),
    x = layers.BatchNormalization()(x),
    x = layers.MaxPooling1D(2)(x),

    x = layers.Flatten()(x),
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x), 
    x = layers.Dropout(0.5)(x),
    x = layers.Dense(5, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x, name=name)
