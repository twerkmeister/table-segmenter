from typing import Text
from tensorflow import keras
from tensorflow.keras import layers
import table_segmenter.conf as conf
import table_segmenter.metrics


def load_model(model_path: Text) -> keras.Model:
    """Loads a model from disc."""
    metrics = {'combined_loss': table_segmenter.metrics.combined_loss,
               'regression_mean_absolute_error':
                   table_segmenter.metrics.regression_mean_absolute_error,
               'decision_accuracy': table_segmenter.metrics.decision_accuracy,
               'regression_mean_error': table_segmenter.metrics.regression_mean_error,
               'regression_error_stddev':
                   table_segmenter.metrics.regression_error_stddev}
    return keras.models.load_model(model_path,
                                   custom_objects=metrics)


def build():
    inputs = keras.Input(shape=(conf.image_max_height, conf.image_max_width, 1))
    y = layers.Conv2D(32, (3, 3), activation='relu', padding="same")(inputs)
    y = layers.MaxPooling2D((2, 4))(y)
    y = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(y)
    y = layers.MaxPooling2D((2, 4))(y)
    y = layers.Conv2D(128, (3, 3), activation='relu', padding="same")(y)
    y = layers.MaxPooling2D((2, 4))(y)
    y = layers.Conv2D(64, (3, 3), activation='relu', padding="same")(y)
    y = layers.MaxPooling2D((1, 4))(y)
    y = layers.Conv2D(32, (3, 3), activation='relu', padding="same")(y)
    y = layers.MaxPooling2D((1, 4))(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    regression_head = layers.Dense(128, activation='relu')(y)
    regression_head = layers.Dense(1, activation='linear')(regression_head)
    regression_head = \
        layers.ReLU(conf.image_max_height / conf.image_downscale_factor,
                    0.01)(regression_head)

    decision_head = layers.Dense(128, activation='relu')(y)
    decision_head = layers.Dense(1, activation='sigmoid')(decision_head)

    outputs = layers.concatenate([regression_head, decision_head])

    return keras.Model(inputs=inputs, outputs=outputs)
