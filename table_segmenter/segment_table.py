from typing import List
from tensorflow import keras
import numpy as np

import table_segmenter.preprocessing
import table_segmenter.conf


def segment_table(model: keras.Model, table_image: np.ndarray) -> List[int]:
    """Segments a table into rows (for now)."""
    should_continue = True
    rows = []
    while should_continue:
        offset = 0 if len(rows) == 0 else rows[-1]
        offset_table_image = table_image[offset:]
        prepared_table_image = \
            table_segmenter.preprocessing.preprocess_image(offset_table_image)
        x = np.expand_dims(prepared_table_image, 0)
        y_pred = model.predict(x)
        next_row = (offset +
                    int(y_pred[0][0] * table_segmenter.conf.image_downscale_factor))

        has_next_row = y_pred[0][1] >= 0.5
        is_valid_next_row = next_row + 10 < table_image.shape[0]
        should_continue = has_next_row and is_valid_next_row
        if should_continue:
            rows.append(next_row)

    return rows

