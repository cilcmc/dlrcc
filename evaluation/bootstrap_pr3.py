# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import interp
import os
from glob import glob
import pandas as pd
from pprint import pprint
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from evaluation.utils import bootstrap, get_index

from sklearn.preprocessing import label_binarize

MACRO_LABEL = 'macro-average'
MICRO_LABEL = 'micro-average'

n_classes = 3
n_bootstraps = 100

matplotlib.rcParams['font.family'] = "Arial"


def calc_pr_with_ci(y_test, y_score, legends):
    auc_map, precision, recall = calc_pr(y_test, y_score, legends)
    auc_all = [auc_map[key] for key in legends]
    print("auc_all: ", auc_all)

    # bootstrap for ci
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        y_boot = np.empty([0, 3], dtype=np.int32)
        preds_boot = np.empty([0, 3])

        # bootstrap by category, ensure that each category has enough samples
        for label in np.unique(y_test, axis=0):
            index = get_index(y_test, label)
            y_i = y_test[index]
            preds_i = y_score[index]

            y_boot_i, preds_boot_i = bootstrap(y_i, preds_i)
            y_boot = np.vstack((y_boot, y_boot_i))
            preds_boot = np.vstack((preds_boot, preds_boot_i))

        auc_map_boot, _, _ = calc_pr(y_boot, preds_boot, legends)

        auc_boot = [auc_map_boot[key] for key in legends]
        # print("bootstrap, i = %d, auc=%s" % (i, auc_boot))
        bootstrapped_scores.append(auc_boot)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort(axis=0)

    lowers = sorted_scores[int(0.025 * len(sorted_scores))]
    uppers = sorted_scores[int(0.975 * len(sorted_scores))]

    auc_with_ci = np.vstack((auc_all, lowers, uppers)).T

    # build map for plot
    average_precision_with_ci = {}
    for i in range(len(legends)):
        average_precision_with_ci[legends[i]] = auc_with_ci[i]

    return average_precision_with_ci, precision, recall


def calc_pr(y_test, y_score, legends):
    # Compute ROC curve and ROC area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[legends[i]], recall[legends[i]], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[legends[i]] = average_precision_score(y_test[:, i], y_score[:, i])
    # Compute micro-average ROC curve and ROC area
    precision[MICRO_LABEL], recall[MICRO_LABEL], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision[MICRO_LABEL] = average_precision_score(y_test, y_score, average="micro")

    all_precision = np.unique(np.concatenate([precision[legends[i]] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in range(n_classes):
        mean_recall += interp(all_precision, precision[legends[i]], recall[legends[i]])
    # Finally average it and compute AUC
    mean_recall /= n_classes
    precision[MACRO_LABEL], recall[MACRO_LABEL] = all_precision, mean_recall
    average_precision[MACRO_LABEL] = average_precision_score(y_test, y_score)

    return average_precision, precision, recall


def save_roc_plot(filename):
    # read data and calc auc
    df = pd.read_csv(filename)
    classes = list(df.columns[1:4])
    legends = classes + [MACRO_LABEL, MICRO_LABEL]

    y_test = label_binarize(df['label'], classes=classes)
    y_score = np.array(df[classes])

    avg_precision, precision, recall = calc_pr_with_ci(y_test, y_score, legends)
    pprint(avg_precision)

    # plot
    title_name = os.path.basename(filename)
    line_colors = ["#D12837", "#147F3A", "#2369B2", "#236EC6", "#0E842C", "#DE2837", "#646464", "#808080",
                   "#808080"]
    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(2.2, 2.2))
        ax = fig.add_subplot(111)
        ax.set_xlabel("x-label", color="black")
        ax.tick_params(axis='x', colors="black")
        ax.set_ylabel("y-label", color="black")
        ax.tick_params(axis='y', colors="black")
        ax.tick_params(bottom=False, left=False, pad=0)

        plt.plot((0, 1), (0, 1), c='#808080', lw=0.5, ls='--', alpha=0.7)

        lines = []
        for i in range(3):
            legend = legends[i]

            line_color = line_colors[i]
            # x: 1-Specificity   y: Sensitivity
            alpha = 1.0
            if legend == MACRO_LABEL or legend == MICRO_LABEL:
                alpha = 0.0

            line, = plt.plot(np.insert(recall[legend], 0, 1), np.insert(precision[legend], 0, 0), c=line_color,
                             alpha=alpha, lw=1,
                             label=u'%s(AUC=%.3f,%.3f,%.3f)' % (legend, avg_precision[legend][0],
                                                                avg_precision[legend][1],
                                                                avg_precision[legend][2]))
            lines.append(line)
            # plt.title(os.path.splitext(title_name)[0], fontsize=6)

        l1 = plt.legend(handles=lines[:3], loc=(23.5 / 100, 17.5 / 100), frameon=False, framealpha=0.8,
                        fontsize=7, labelspacing=0.5)

        for i in range(3, 5):
            legend = legends[i]
            line_color = line_colors[i]
            # x: 1-Specificity   y: Sensitivity
            alpha = 1.0
            if legend == MACRO_LABEL or legend == MICRO_LABEL:
                alpha = 0.0

            line, = plt.plot(recall[legend], precision[legend], c=line_color, alpha=alpha, lw=1,
                             label=u'%s(AUC=%.3f,%.3f,%.3f)' % (legend, avg_precision[legend][0],
                                                                avg_precision[legend][1],
                                                                avg_precision[legend][2]))
            lines.append(line)

        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.00, 1.01))
        plt.xticks(np.arange(0, 1.01, 0.2), fontsize=7)
        plt.yticks(np.arange(0, 1.01, 0.2), fontsize=7)
        plt.xlabel('recall', fontsize=7)
        plt.ylabel('precision', fontsize=7)
        plt.grid(b=None)

        plt.gca().add_artist(l1)
        plt.legend(handles=lines[3:], loc=(7 / 100, 1 / 100), frameon=False, framealpha=0.8, fontsize=7,
                   labelspacing=0.5)
        plt.savefig(os.path.dirname(filename) + '/' + os.path.splitext(title_name)[0] + '_pr' + '.png', dpi=300,
                    bbox_inches='tight')


if __name__ == '__main__':
    for result_csv in glob('data/roc_pr/category3/*.csv'):
        if result_csv.endswith("csv"):
            print('# process ' + result_csv)
            save_roc_plot(result_csv)
