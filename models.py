import tensorflow as tf

keras = tf.keras
from keras import layers, Input
from keras.models import Model
from tensorflow.keras.applications import MobileNet

import TransformerEncoder


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
        elif model_type == "CNN_1D":
            # input = tf.keras.layers.Reshape((32, 96), input_shape=(32, 96, 1)),
            return create_1Dcnn_model(self.input_shape, self.num_classes)
        elif model_type ==  "CNN_optimized":
            return create_cnn_model_optimized(self.input_shape, self.num_classes)
        elif model_type ==  "Transformer":
            return create_transformer_model(self.input_shape, self.num_classes)
        elif model_type == "CNN_1D_MultiHeaded":
            return create_1Dcnn_MultiHeaded_model(self.input_shape, self.num_classes)
        # elif model_type == "MobileNet":
        #     return create_MobileNet_model(self.input_shape, self.num_classes)
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


def create_cnn_model_optimized(input_shape: tuple, num_classes: int, name: str = "CNN_optimized") -> Model:
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

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer =tf.keras.regularizers.l2( l=0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


    return model

def create_1Dcnn_model(input_shape: tuple, num_classes: int, name: str = "CNN_1D") -> Model:
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

    model = tf.keras.Sequential([

        tf.keras.layers.Reshape((32, 96), input_shape=input_shape),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
        
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_transformer_model(input_shape: tuple, num_classes: int, name: str = "Transformer") -> Model:
    """
    Creates a Transformer model 
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input data
    num_classes : int
        Number of classes to predict
    """

    # Define model parameters
    max_length = 32 # input_shape.shape[1]
    input_dim = 96 # input_shape.shape[2]
    output_dim = num_classes
    d_model =  120
    num_heads = 3 # 3
    dff = 180 # 128
    num_layers = 2
    dropout_rate = 0.2

    # Define input layer
    inputs = Input(shape=(max_length, input_dim))

    # Add input transformation layer
    x = layers.Dense(d_model, activation='linear')(inputs)

    # Add Transformer encoder layers
    for i in range(num_layers):
        transformer_encoder = TransformerEncoder.TransformerEncoder(d_model, num_heads, dff, dropout_rate)
        x = transformer_encoder(x, training=True)

    # Flatten and add output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)

    # Define and compile model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model

def create_1Dcnn_MultiHeaded_model(input_shape: tuple, num_classes: int, name: str = "CNN_1D_MultiHeaded") -> Model:
    """
    Creates a 1D CNN Multi_headed model
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input data
    num_classes : int
        Number of classes to predict
    """
    input_layer = tf.keras.layers.Input(shape=(32, 96))

    head1 = tf.keras.layers.Conv1D(32, 3, activation='relu')(input_layer)
    head1 = tf.keras.layers.BatchNormalization()(head1)
    head1 = tf.keras.layers.MaxPooling1D(2)(head1)
    head1 = tf.keras.layers.Conv1D(64, 3, activation='relu')(head1)
    head1 = tf.keras.layers.BatchNormalization()(head1)
    head1 = tf.keras.layers.MaxPooling1D(2)(head1)
    head1 = tf.keras.layers.Conv1D(128, 3, activation='relu')(head1)
    head1 = tf.keras.layers.BatchNormalization()(head1)
    head1 = tf.keras.layers.MaxPooling1D(2)(head1)
    
    head1 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(head1)  # Add padding='same'
    head1 = tf.keras.layers.BatchNormalization()(head1)
    head1 = tf.keras.layers.MaxPooling1D(2)(head1)
    
    head1 = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(head1)  # Add padding='same'
    head1 = tf.keras.layers.BatchNormalization()(head1)
    # head1 = tf.keras.layers.MaxPooling1D(2)(head1)


    head1 = tf.keras.layers.Flatten()(head1)

    head2 = tf.keras.layers.Conv1D(32, 3, activation='relu')(input_layer)
    head2 = tf.keras.layers.BatchNormalization()(head2)
    head2 = tf.keras.layers.MaxPooling1D(2)(head2)
    head2 = tf.keras.layers.Conv1D(64, 3, activation='relu')(head2)
    head2 = tf.keras.layers.BatchNormalization()(head2)
    head2 = tf.keras.layers.MaxPooling1D(2)(head2)
    head2 = tf.keras.layers.Conv1D(128, 3, activation='relu')(head2)
    head2 = tf.keras.layers.BatchNormalization()(head2)
    head2 = tf.keras.layers.MaxPooling1D(2)(head2)
    head2 = tf.keras.layers.Flatten()(head2)

    head3 = tf.keras.layers.Conv1D(32, 3, activation='relu')(input_layer)
    head3 = tf.keras.layers.BatchNormalization()(head3)
    head3 = tf.keras.layers.MaxPooling1D(2)(head3)
    head3 = tf.keras.layers.Conv1D(64, 3, activation='relu')(head3)
    head3 = tf.keras.layers.BatchNormalization()(head3)
    head3 = tf.keras.layers.MaxPooling1D(2)(head3)
    head3 = tf.keras.layers.Conv1D(128, 3, activation='relu')(head3)
    head3 = tf.keras.layers.BatchNormalization()(head3)
    head3 = tf.keras.layers.MaxPooling1D(2)(head3)
    head3 = tf.keras.layers.Flatten()(head3)

    # Concatenate heads
    merged = tf.keras.layers.Concatenate()([head1])
    merged = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    output_layer = tf.keras.layers.Dense(5, activation='softmax')(merged)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    # optimizer = Adam(learning_rate=lr_schedule)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def create_MobileNet_model(input_shape: tuple, num_classes: int, name: str = "MobileNet") -> Model:
    """
    Creates a MobileNet model - Work in progress
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input data
    num_classes : int
        Number of classes to predict
    """
    # Load MobileNet without the top classification layer and set input shape
    print(input_shape)
    input_tensor = Input(shape=input_shape)
    base_model = MobileNet(include_top=False, input_shape=input_tensor, weights='imagenet')

    # Add a custom classification layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

        # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Fine-tuning: Unfreeze the top 30 layers of the base model
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)


    return model
