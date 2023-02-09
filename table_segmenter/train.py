import argparse
import random
from typing import Text
import os

import table_segmenter.model
import table_segmenter.io
import table_segmenter.preprocessing
import table_segmenter.metrics
import tensorflow as tf
import numpy as np
import table_segmenter.conf as conf
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


def make_dataset(data_path: Text):
    image_names = table_segmenter.io.list_images(data_path)

    def generate_examples():
        while True:
            random.shuffle(image_names)
            for image_name in image_names:
                image_path = os.path.join(data_path, image_name)
                image = table_segmenter.io.read_image(image_path)
                targets = table_segmenter.io.read_targets_for_image(image_path)
                preprocessed_image = table_segmenter.preprocessing.preprocess_image(
                    image)
                preprocessed_targets = \
                    np.asarray([targets[0] / conf.image_downscale_factor, targets[1]])
                yield preprocessed_image, preprocessed_targets

    return \
        tf.data.Dataset.from_generator(
            generate_examples,
            output_signature=(
                tf.TensorSpec(shape=(conf.image_max_height, conf.image_max_width, 1),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.int32)
            )).batch(conf.batch_size).prefetch(100)


def train(train_data_path: Text, val_data_path: Text, experiment_dir: Text):
    tf.compat.v1.disable_eager_execution()
    # tf.config.run_functions_eagerly(True)
    os.makedirs(experiment_dir, exist_ok=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=experiment_dir)
    early_stopping_callback = keras.callbacks.EarlyStopping("val_loss", patience=7,
                                                            verbose=1,
                                                            restore_best_weights=True)
    num_train_examples = len(table_segmenter.io.list_images(train_data_path))
    num_validation_examples = len(table_segmenter.io.list_images(val_data_path))
    print("Loading training data")
    train_dataset = make_dataset(train_data_path)
    print("Loading validation data")
    val_dataset = make_dataset(val_data_path)

    model = table_segmenter.model.build()
    model.compile(loss=table_segmenter.metrics.combined_loss,
                  optimizer='adam',
                  # run_eagerly=True,
                  metrics=[table_segmenter.metrics.regression_mean_absolute_error,
                           table_segmenter.metrics.decision_accuracy,
                           table_segmenter.metrics.regression_mean_error,
                           table_segmenter.metrics.regression_error_stddev])
    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=80,
              verbose=True,
              steps_per_epoch=num_train_examples//conf.batch_size,
              validation_steps=num_validation_examples//conf.batch_size,
              callbacks=[tensorboard_callback, early_stopping_callback])

    model.save(experiment_dir)

    # MAE evaluation
    score = model.evaluate(val_dataset,
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
