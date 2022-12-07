from types import SimpleNamespace

import numpy as np
import tensorflow as tf

###############################################################################################


def get_default_CNN_model():
    """
    Sets up your model architecture and compiles it using the appropriate optimizer, loss, and
    metrics.

    :returns compiled model
    """

    Conv2D = tf.keras.layers.Conv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dropout = tf.keras.layers.Dropout

    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=37, output_mode="one_hot"
    )
    
    augment_fn = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = CustomSequential(
        [
            Conv2D(16, 5, strides=(2,2), padding='same', activation='leaky_relu'),
            BatchNormalization(),
            #tf.keras.layers.LeakyReLU(),
            Conv2D(32, 5, strides=(2,2), padding='same'),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.MaxPool2D(),
            Dropout(0.2),
            Conv2D(32, 5, strides=(2,2), padding='same'),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            Conv2D(64, 5, strides=(2,2), activation='sigmoid', padding='same'),
            BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(300, activation='leaky_relu'),
            tf.keras.layers.Dense(37, activation='sigmoid'),
        ], output_prep_fn=output_prep_fn, augment_fn=augment_fn

    )

    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=15, batch_size=50)


###############################################################################################


class CustomSequential(tf.keras.Sequential):
    """
    Subclasses tf.keras.Sequential to allow us to specify preparation functions that
    will modify input and output data.

    :param output_prep_fn: Modifies input labels prior to running forward pass
    :param augment_fn: Augments input images prior to running forward pass
    """

    def __init__(
        self,
        *args,
        output_prep_fn=lambda x: x,
        augment_fn=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_prep_fn = output_prep_fn
        self.augment_fn = augment_fn

    def batch_step(self, data, training=False):

        x, y_raw = data

        y = self.output_prep_fn(y_raw)
        if training:
            x = self.augment_fn(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def predict_step(self, inputs):
        return self(inputs)
