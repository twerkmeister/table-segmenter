import argparse
from typing import Text

import table_segmenter.model
import table_segmenter.io
import table_segmenter.preprocessing
import table_segmenter.conf
import table_segmenter.metrics
import table_segmenter.model


def evaluate(model_path: Text, data_path: Text):
    model = table_segmenter.model.load_model(model_path)

    print("Loading data")
    names, images = table_segmenter.io.load_images(data_path)
    targets = table_segmenter.io.load_targets(data_path, names)
    x = table_segmenter.preprocessing.preprocess_images(images)
    y = table_segmenter.preprocessing.preprocess_targets(targets)

    score = model.evaluate(x,
                           y,
                           batch_size=16,
                           verbose=True)

    print(f"loss: {score[0]}"
          f"MAE: {score[1]}"
          f"Dec. Acc.: {score[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train the table segmenter.')

    parser.add_argument("model_path",
                        help='Path to the model.')
    parser.add_argument("data_path",
                        help='Path to the data folder with images to eval.')

    args = parser.parse_args()
    evaluate(args.model_path, args.data_path)
