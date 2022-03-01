import argparse
from typing import Text
import os

import table_segmenter.model
import table_segmenter.io
import table_segmenter.preprocessing
import table_segmenter.metrics
import tensorflow
from tensorflow import keras


def load_data_for_training(data_path: Text):
    """Convenience method."""
    image_names, images = table_segmenter.io.load_images(data_path)
    targets = table_segmenter.io.load_targets(data_path, image_names)
    original_image_shapes = [image.shape for image in images]
    x = table_segmenter.preprocessing.preprocess_images(images)
    x_augmented, augmented_targets = \
        table_segmenter.preprocessing.augment_multi(x, targets, original_image_shapes)
    y = table_segmenter.preprocessing.preprocess_targets(augmented_targets)
    return x_augmented, y


def train(train_data_path: Text, val_data_path: Text, experiment_dir: Text):
    tensorflow.compat.v1.disable_eager_execution()
    # tensorflow.config.run_functions_eagerly(True)
    os.makedirs(experiment_dir, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=experiment_dir)
    early_stopping_callback = keras.callbacks.EarlyStopping("val_loss", patience=7,
                                                            verbose=1,
                                                            restore_best_weights=True)

    print("Loading training data")
    x_train, y_train = load_data_for_training(train_data_path)
    print("Loading validation data")
    x_val, y_val = load_data_for_training(val_data_path)

    model = table_segmenter.model.build()
    model.compile(loss=table_segmenter.metrics.combined_loss,
                  optimizer='adam',
                  # run_eagerly=True,
                  metrics=[table_segmenter.metrics.regression_mean_absolute_error,
                           table_segmenter.metrics.decision_accuracy,
                           table_segmenter.metrics.regression_mean_error,
                           table_segmenter.metrics.regression_error_stddev])
    model.fit(x_train,
              y_train,
              validation_data=(x_val, y_val),
              epochs=80,
              batch_size=16,
              verbose=True,
              callbacks=[tensorboard_callback, early_stopping_callback])

    model.save(experiment_dir)

    # MAE evaluation
    score = model.evaluate(x_val,
                           y_val,
                           batch_size=16,
                           verbose=True)

    print("nTest MAE: %.1f%%" % (score[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train the table segmenter.')

    parser.add_argument("train_data_path",
                        help='Path to the training data folder.')

    parser.add_argument("val_data_path",
                        help='Path to the validation data folder.')

    parser.add_argument("experiment_folder",
                        help='Path to the output folder for the model and logs.')

    args = parser.parse_args()
    train(args.train_data_path, args.val_data_path, args.experiment_folder)
