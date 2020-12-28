# -*-coding:utf-8 -*-

import pandas as pd
import numpy as np
from evaluation.pdi import calc_pdi
from pprint import pprint

from evaluation.utils import bootstrap

n_bootstraps = 30

# array([[0.95503435, 0.90334044, 0.98114191],
#        [0.95555556, 0.90623075, 0.98640133],
#        [0.94740583, 0.87145226, 0.9960199 ],
#        [0.96214167, 0.902914  , 0.9978678 ]]
if __name__ == '__main__':
    filename = 'data/pdi/input.csv'
    df = pd.read_csv(filename)
    y = np.array(df['label'])
    preds = np.array(df.iloc[:, 1:4])
    classes = df.columns[1:]
    pdi_all = calc_pdi(y, preds, classes)

    # bootstrap for ci
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        y_boot = np.empty((0), dtype=np.int32)
        preds_boot = np.empty([0, 3])

        # bootstrap by category, ensure that each category has enough samples
        for label in np.unique(y):
            y_i = y[y == label]
            preds_i = preds[y == label]

            y_boot_i, preds_boot_i = bootstrap(y_i, preds_i)

            y_boot = np.hstack((y_boot, y_boot_i))
            preds_boot = np.vstack((preds_boot, preds_boot_i))

        pdi = calc_pdi(y_boot, preds_boot, classes)
        print("bootstrap, i = %d, pdi=%s" % (i, pdi))
        bootstrapped_scores.append(pdi)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort(axis=0)
    lowers = sorted_scores[int(0.025 * len(sorted_scores))]
    uppers = sorted_scores[int(0.975 * len(sorted_scores))]

    pdi_with_ci = np.vstack((pdi_all, lowers, uppers)).T
    pprint(pdi_with_ci)
