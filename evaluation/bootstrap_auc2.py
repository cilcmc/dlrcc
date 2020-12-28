# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc
import os
from glob import glob
import pandas as pd

from evaluation.utils import bootstrap, calc_youden_index

MACRO_LABEL = 'macro-average'
MICRO_LABEL = 'micro-average'

n_classes = 3
n_bootstraps = 100

matplotlib.rcParams['font.family'] = "Arial"


def calc_auc_with_ci(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_all = auc(fpr, tpr)

    # bootstrap for ci
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        y_test_boot = []
        y_score_boot = []

        # bootstrap by category, ensure that each category has enough samples
        for label in np.unique(y_test, axis=0):
            y_test_i = y_test[y_test == label]
            y_score_i = y_score[y_test == label]

            y_test_boot_i, y_score_boot_i = bootstrap(y_test_i, y_score_i)
            y_test_boot = np.hstack((y_test_boot, y_test_boot_i))
            y_score_boot = np.hstack((y_score_boot, y_score_boot_i))

        fpr_boot, tpr_boot, _ = roc_curve(y_test_boot, y_score_boot)
        auc_boot = auc(fpr_boot, tpr_boot)

        # print("bootstrap, i = %d, auc=%s" % (i, auc_boot))
        bootstrapped_scores.append(auc_boot)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort(axis=0)

    lowers = sorted_scores[int(0.025 * len(sorted_scores))]
    uppers = sorted_scores[int(0.975 * len(sorted_scores))]

    auc_with_ci = np.hstack((auc_all, lowers, uppers))

    return auc_with_ci, fpr, tpr


def save_roc_plot(filename):
    # read data and calc auc
    df = pd.read_csv(filename)

    y_test = df.values[:, 0]
    y_score = df.values[:, 1]

    print("youden_index=", calc_youden_index(y_test, y_score))

    auc_with_ci, fpr, tpr = calc_auc_with_ci(y_test, y_score)

    # plot
    title_name = os.path.basename(filename)
    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(2.2, 2.2))
        ax = fig.add_subplot(111)
        ax.set_xlabel("x-label", color="black")
        ax.tick_params(axis='x', colors="black")
        ax.set_ylabel("y-label", color="black")
        ax.tick_params(axis='y', colors="black")
        ax.tick_params(bottom=False, left=False, pad=0)

        plt.plot((0, 1), (0, 1), c='#808080', lw=0.5, ls='--', alpha=0.7)

        # x: 1-Specificity   y: Sensitivity
        alpha = 1.0
        plt.plot(fpr, tpr, c='#FF7733', alpha=alpha, lw=1,
                 # label=u'AUC=%.3f, %.3f, %.3f' % (auc_with_ci[0], auc_with_ci[1],
                 #                                  auc_with_ci[2]))
                 label=u'AUC=%.3f, CI=%.3f–%.3f' % (auc_with_ci[0], auc_with_ci[1],
                                                  auc_with_ci[2]))
        plt.legend(loc=(7 / 100, 1 / 100), frameon=False, framealpha=0.8,
                   fontsize=7, labelspacing=0.5)

        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.00, 1.01))
        plt.xticks(np.arange(0, 1.01, 0.2), fontsize=7)
        plt.yticks(np.arange(0, 1.01, 0.2), fontsize=7)
        plt.xlabel('Specificity', fontsize=7)
        plt.ylabel('Sensitivity', fontsize=7)
        plt.grid(b=None)

        plt.savefig(os.path.dirname(filename) + '/' + os.path.splitext(title_name)[0] + '_roc' + '.png', dpi=300,
                    bbox_inches='tight')


if __name__ == '__main__':
    for filename in glob('data/roc_pr/category2/*'):
        if filename.endswith("csv"):
            print('# process ' + filename)
            save_roc_plot(filename)
            # break
