import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import backend
from tensorflow.keras import metrics
from tensorflow.python.ops import math_ops

from table_segmenter import conf


# @tf.autograph.experimental.do_not_convert
def combined_loss(y_true, y_pred):
    decision_target = y_true[:, 1]
    decision_pred = y_pred[:, 1]

    regression_loss = regression_mean_absolute_error(y_true, y_pred) \
                      / conf.image_downscale_factor

    decision_loss = losses.binary_crossentropy(decision_target,
                                               decision_pred)

    return regression_loss + decision_loss


def regression_errors(y_true, y_pred):
    regression_target = y_true[:, 0]
    regression_pred = y_pred[:, 0]

    decision_target = y_true[:, 1]
    return (regression_target - regression_pred) * decision_target \
           * conf.image_downscale_factor


def sum_and_divide_by_positive_targets(t, y_true):
    decision_target = y_true[:, 1]
    return (math_ops.reduce_sum(t) /
            math_ops.maximum(math_ops.reduce_sum(decision_target), backend.epsilon()))


# @tf.autograph.experimental.do_not_convert
def regression_mean_absolute_error(y_true, y_pred):
    return sum_and_divide_by_positive_targets(
        math_ops.abs(regression_errors(y_true, y_pred)),
        y_true
    )


def regression_mean_error(y_true, y_pred):
    return sum_and_divide_by_positive_targets(
        regression_errors(y_true, y_pred),
        y_true
    )


def regression_error_variance(y_true, y_pred):
    mean_error = regression_mean_error(y_true, y_pred)
    return sum_and_divide_by_positive_targets(
        math_ops.square(mean_error - regression_errors(y_true, y_pred)),
        y_true
    )


def regression_error_stddev(y_true, y_pred):
    return tf.math.sqrt(regression_error_variance(y_true, y_pred))


# @tf.autograph.experimental.do_not_convert
def regression_top_k_abs_error_mean(y_true, y_pred):
    k = 5
    abs_errors = math_ops.abs(regression_errors(y_true, y_pred))
    top_k_errors = tf.math.top_k(abs_errors, k).values
    return tf.math.reduce_sum(top_k_errors) / k


def decision_accuracy(y_true, y_pred):
    decision_target = y_true[:, 1]
    decision_pred = y_pred[:, 1]
    return metrics.binary_accuracy(decision_target, decision_pred)
