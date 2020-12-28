# -*-coding:utf-8 -*-

"""
Calc the Polytomous Discrimination Index(PDI)
according to
Extending the c-statistic to nominal polytomous outcomes: the Polytomous Discrimination Index
"""

import numpy as np
import pandas as pd


def calc_pdi(y, preds, classes):
    """
    :param y: 1-dim label numpy array of samples

    :param preds: 2-dim probability numpy array of samples

    :param classes: array-like of shape [n_classes]
        Uniquely holds the label for each class.

    :return: [dpi, category_dpi_0, category_dpi_1, category_dpi_2]
    """
    probs_0 = preds[y == classes[0]]
    probs_1 = preds[y == classes[1]]
    probs_2 = preds[y == classes[2]]
    total = len(probs_0) * len(probs_1) * len(probs_2)
    dpis = []
    for i in range(3):
        cnt = 0
        for prob_0 in probs_0:
            for prob_1 in probs_1:
                for prob_2 in probs_2:
                    mat = np.vstack((prob_0, prob_1, prob_2))
                    if np.argmax(mat[:, i]) == i:
                        cnt += 1
        dpis.append(cnt / total)
    return [np.average(dpis), ] + dpis


if __name__ == '__main__':
    filename = 'data/pdi/input.csv'

    df = pd.read_csv(filename)

    y = np.array(df['label'])
    preds = np.array(df.iloc[:, 1:4])
    classes = df.columns[1:]

    result = calc_pdi(y, preds, classes)

    print("dpi: %.3f" % result[0])
    for i in range(3):
        print("category_dpi_%s: %.3f" % (classes[i], result[i + 1]))
