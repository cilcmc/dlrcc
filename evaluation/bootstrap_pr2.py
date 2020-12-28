# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd

from glob import glob

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from evaluation.utils import bootstrap

n_bootstraps = 100

matplotlib.rcParams['font.family'] = "Arial"


def calc_pr_with_ci(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_pr = average_precision_score(y_test, y_score)

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

        precision_boot, recall_boot, _ = precision_recall_curve(y_test_boot, y_score_boot)
        avg_pr_boot = average_precision_score(y_test_boot, y_score_boot)

        # print("bootstrap, i = %d, pr=%s" % (i, pr_boot))
        bootstrapped_scores.append(avg_pr_boot)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort(axis=0)

    lowers = sorted_scores[int(0.025 * len(sorted_scores))]
    uppers = sorted_scores[int(0.975 * len(sorted_scores))]

    pr_with_ci = np.hstack((avg_pr, lowers, uppers))

    return pr_with_ci, precision, recall


def save_roc_plot(filename):
    # read data and calc pr
    df = pd.read_csv(filename)

    y_test = df.values[:, 0]
    y_score = df.values[:, 1]

    pr_with_ci, precision, recall = calc_pr_with_ci(y_test, y_score)

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
        plt.plot(np.insert(recall, 0, 1), np.insert(precision, 0, 0), c='#FF7733', alpha=alpha, lw=1,
                 label=u'PR=%.3f, %.3f, %.3f' % (pr_with_ci[0], pr_with_ci[1],
                                                 pr_with_ci[2]))

        plt.legend(loc=(48 / 100, 1 / 100), frameon=False, framealpha=0.8,
                   fontsize=7, labelspacing=0.5)

        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.00, 1.01))
        plt.xticks(np.arange(0, 1.01, 0.2), fontsize=7)
        plt.yticks(np.arange(0, 1.01, 0.2), fontsize=7)
        plt.xlabel('recall', fontsize=7)
        plt.ylabel('precision', fontsize=7)

        plt.grid(b=None)

        plt.savefig(os.path.dirname(filename) + '/' + os.path.splitext(title_name)[0] + '_pr' + '.png', dpi=300,
                    bbox_inches='tight')


if __name__ == '__main__':
    for filename in glob('data/roc_pr/category2/*'):
        if filename.endswith("csv"):
            print('# process ' + filename)
            save_roc_plot(filename)
