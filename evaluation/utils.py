import numpy as np
from sklearn.metrics import roc_curve

rng_seed = 100  # control reproducibility
rng = np.random.RandomState(rng_seed)


def bootstrap(y, preds):
    # bootstrap by sampling with replacement on the prediction indices
    indices = rng.randint(0, len(preds), len(preds))
    return y[indices], preds[indices]


def get_index(mat, row):
    result = []
    for i in range(len(mat)):
        line = mat[i]
        if (line == row).all():
            result.append(i)
    return result


def calc_youden_index(y_test, y_score):
    """
    :param y_test:
    :param y_score:
    :return: [Youden's J statistic, threshold]
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    youden_J = np.max(tpr - fpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return youden_J, optimal_threshold