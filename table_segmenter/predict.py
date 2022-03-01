import argparse
from typing import Text
import os

import table_segmenter.model
import table_segmenter.io
import table_segmenter.preprocessing
import table_segmenter.conf
import table_segmenter.metrics
import table_segmenter.model
import numpy as np
import cv2


def predict(model_path: Text, data_path: Text, output_path: Text):
    model = table_segmenter.model.load_model(model_path)

    print("Loading data")
    names, images = table_segmenter.io.load_images(data_path)
    x = table_segmenter.preprocessing.preprocess_images(images)

    os.makedirs(output_path, exist_ok=True)
    y_pred = model.predict(x)

    for i in range(len(names)):
        image = np.copy(images[i])
        px_position = int(y_pred[i][0] * table_segmenter.conf.image_downscale_factor)
        if y_pred[i][1] >= 0.5:
            cv2.line(image,
                     (0, px_position),
                     (image.shape[1], px_position),
                     (0, 255, 0),
                     thickness=2)
        pred_image_name = f"{os.path.splitext(names[i])[0]}_pred.jpg"
        table_segmenter.io.write_image(os.path.join(output_path, pred_image_name), image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train the table segmenter.')

    parser.add_argument("model_path",
                        help='Path to the model.')
    parser.add_argument("data_path",
                        help='Path to the data folder with images to predict.')
    parser.add_argument("output_path",
                        help='Path to the folder to write predictions to.')

    args = parser.parse_args()
    predict(args.model_path, args.data_path, args.output_path)
