# -*- coding: utf-8 -*-

import pandas as pd
from glob import glob
from sklearn.metrics import confusion_matrix

from evaluation.utils import calc_youden_index


def get_confusion_matrix(filename):
    df = pd.read_csv(filename)
    y_test = df.values[:, 0]
    y_score = df.values[:, 1]
    _, cutoff = calc_youden_index(y_test, y_score)
    y_predict = (y_score > cutoff).astype(int)
    return confusion_matrix(y_test, y_predict)


if __name__ == '__main__':
    for filename in glob('data/roc_pr/category2/*'):
        if filename.endswith("csv"):
            print('# process ' + filename)
            print(get_confusion_matrix(filename))
