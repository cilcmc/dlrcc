## Basic Usage

The project is organized as follows

1. Diagnosis

2. Prognosis

3. Evaluation

**Diagnosis** and **Prognosis** work on [Jupyter Notebook](https://jupyter.org/) for more interactive and **Evaluation** works on normal python environment.

### Diagnosis

Patches were at first predicted as tumor or normal by the binary classifier with a threshold of 0.5. Subsequently, tumor patches were classified as KIRC, KIRP, KICH by the three-way subtype classifier with the most possibility. The prediction results were aggregated by calculating percentage of normal/tumor patches over all patches, or percentage of each subtype tumor patches over predicted tumor patches in a slide.

#### Train

There are two models to train:

- Normal vs. Tumor
- KICH vs. KIRC vs. KIRP

Start Jupyter notebook and open file `diagnosis/diagnosis_train.ipynb`. Parameters need to modify  for model training:

- **input_shape**: pixel size of patch. For InceptionV3, it should be `[299, 299]`
- **batch size**: batch size for model training.
- **train_dir**: the directory containing all the training patches
  - For normal vs. tumor, the `train_dir` should have the following structure:

    ```shell
    train_dir
    └── normal
        ├── normal_patch_0.jpeg
        ├── normal_patch_1.jpeg
        ├── ...
        └── normal_patch_n.jpeg
    └── tumor
        ├── tumor_patch_0.jpeg
        ├── tumor_patch_1.jpeg
        ...
        └── tumor_patch_n.jpeg
    ```

  - For KICH vs. KIRC vs. KIRP, the `train_dir` should have the following structure:

    ```shell
    train_dir
    └── kich
        ├── kich_patch_0.jpeg
        ├── kich_patch_1.jpeg
        ├── ...
        └── kich_patch_n.jpeg
    └── kirc
        ├── kirc_patch_0.jpeg
        ├── kirc_patch_1.jpeg
        ├── ...
        └── kirc_patch_n.jpeg
    └── kirp
        ├── kirp_patch_0.jpeg
        ├── kirp_patch_1.jpeg
        ...
        └── kirp_patch_n.jpeg
    ```

- **tune_dir**: the directory containing all the tuning patches, `tune_dir` should have the same structure with `train_dir`.

#### Test

The test is divided into two steps

1. Each patient was predicted for normal and tumor with `diagnosis/diagnosis_test_tumor`. It will predict the probability of tumor and normal for each patient and write the result to csv file, and save the filename of predicted normal patches into `normal_list.txt`. Besides, the heatmap will be generated.
2. Subtype of of typing for patients is predicted with `diagnosis/diagnosis_test-subtype.ipynb`. It will predict the probability of three subtypes for each patient using the tumor patches filtered by the `normal_list.txt`, and output the result and heatmap.

The parameter need to modify:

- **weight_file**: h5 weight file for loading
- **test_dirs** and **mag**: the directory containing all the tuning patches using the following format

    ```shell
    test_dir
    └── TCGA_slide_0_files
        └──mag
            ├── slide_0_patch_0.jpeg
            ├── slide_0_patch_1.jpeg
            ├── slide_0_patch_2.jpeg
            ...
            └── slide_0_patch_3.jpeg
    └── TCGA_slide_1_files
        └──mag
            ├── slide_1_patch_0.jpeg
            ├── slide_1_patch_1.jpeg
            ├── slide_1_patch_2.jpeg
            ...
            └── slide_1_patch_3.jpeg
        ...
    └── TCGA_slide_n_files
    ```

- **true_label**: For binary classifier, true_label was assigned as `tumor` or `normal`. For three-way classifier, true_label was assigned as `kich`, `kirc` or `kirp`.

### Prognosis

We developed our predictive algorithms by combining the 19-layer Visual Geometry Group (VGG19) architecture23 and Cox regression model. The VGG19 made use of various convolutions from original architecture to extract image features. The last layer was changed to a fully connected layer containing one node, which predicted a risk score for the input subpatch. The risk score in each subpatch were presented to a Cox proportional hazards layer allowing the use of censored data to calculate the Cox loss function.

#### How to use

Start Jupyter notebook and open file `prognosis/prognosis_train.ipynb` for train or `prognosis/prognosis_test.ipynb`. Modify the config in section `Load data` for training or loading model:

**model_h5_file**: model weight file

**train_dir**, **tune_dir** and **test_dir** are the patch dirs containing the training, tuning and testing patches (1196 * 1196 pixels), the structure of the dirs is as follows:

```shell
dir
├── patch_0.jpeg
├── patch_1.jpeg
├── patch_2.jpeg
├── patch_3.jpeg
...
└── patch_n.jpeg
```

**metadata_path**: csv file containing patient metadata, such as patient id, survival time, censored, and so on.

The section `Test Model` will predict the origin risk and level 2 risk for each patient.

### Evaluation

In evaluation module, the program will read the result which **diagnosis** and **prognosis** output and calculate the following evaluation indicators.

- [AUC and ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

- [Precision and Recall](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

- [Youden's J statistic](https://en.wikipedia.org/wiki/Youden%27s_J_statistic)

- PDI (the polytomous discrimination index) according to the reference of `Extending the c-statistic to nominal polytomous outcomes: the Polytomous Discrimination Index`

The directory `data` is an example of the result file, and it's worth noting that the binary classification is different from the three-way classification.
For binary classification, the format of result file is as follows:

```csv
y_true,y_pred
1,0.22
1,0.86
1,0.96
...
0,0.12
0,0.07
0,0.08
```

For three-way classification, the format of result file is as follows:

```csv
label,ccRCC,pRCC,chRCC
chRCC,0.02,0.03,0.95
chRCC,0.77,0.04,0.19
...
ccRCC,0.84,0.16,0
ccRCC,0.52,0.48,0
ccRCC,0.97,0.03,0
...
pRCC,0.05,0.90,0.05
pRCC,0.040.96,0
```

Usage:

`bootstrap_auc2.py`: plot ROC curves and calculate AUC with CI for binary classification. Youden's J statistic will also be calculated.

`bootstrap_auc3.py`: plot ROC curves and calculate AUC with CI for three-way classification. Youden's J statistic will also be calculated.

`bootstrap_pr2.py`: plot precision-recall curve for binary classification

`bootstrap_pr3.py`: plot precision-recall curve for three-way classification

`bootstrap_pdi.py`: calculate the PDI with CI

`get_confusion_matrix2.py`: calculate confusion matrix for binary classification

`get_confusion_matrix3.py`: calculate confusion matrix for three-way classification
