# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from glob import glob
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(filename):
    df = pd.read_csv(filename)
    classes = df.columns[1:4]

    y_test = df['label']
    y_score = np.array(df[classes])
    y_pred = classes[np.argmax(y_score, axis=1)]

    return confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    for filename in glob('data/roc_pr/category3/*'):
        if filename.endswith("csv"):
            print('# process ' + filename)
            print(get_confusion_matrix(filename))
