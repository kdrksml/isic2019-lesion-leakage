# Same-lesion leakage in ISIC 2019: a paired evaluation

This repository accompanies the paper *Quantifying Performance Inflation from Same-Lesion Data Leakage in Skin Lesion
Classification: A Paired Evaluation on ISIC 2019*. It contains everything needed to check the reported results:

* patient-extended groups for all 25,331 ISIC 2019 training images and the matched image-level and lesion-level
  partitions,
* the per-image predictions of all 45 trained networks on their internal test partition and on the official test set,
* the training code that produced these predictions,
* the statistical analysis and a script that recomputes every number reported in the paper.

The ISIC 2019 images are not redistributed. Every result of the paper except retraining can be recomputed on a laptop
CPU from the files here.

## Quick start

```bash
pip install -r requirements.txt
python analysis/check_splits.py      # integrity of the groups and partitions               (seconds)
python analysis/verify_results.py    # recomputes the reported numbers from the predictions   (about 5 min)
python analysis/paired_statistics.py        # all intervals, tests and subgroup analyses            (about 5 min)
python analysis/make_figures.py      # figures of the paper into figures/                    (about 2 min)
```

`verify_results.py` ends with a line such as `261 checks, 261 passed, 0 failed` and writes every check with the reported
and the recomputed value to `analysis/output/verification.csv`.

## Study design in brief

The ISIC 2019 training set is split twice in a 70/10/20 ratio with seed 42, once at random over images (P1) and once
over patient-extended groups (P2). Both splits have partitions of 17,731, 2533 and 5067 images, and their class counts
differ by at most one image per class and partition. Under P1, 2926 of the 5067 test images (57.7%) have a photograph of
the same group in training, and 2910 (57.4%) a photograph of the same lesion. Under P2, none has. The same ten model
configurations are trained on both splits and evaluated on the internal test partition of their split and on the same
5920 scored images of the official ISIC 2019 test set, in which no same-lesion overlap with training was found. Inflation compares the two internal estimates, the gap compares
each internal estimate with its official result and the difference-in-differences (DiD) compares the two gaps.

## Repository layout

```
resources/   groups and partitions, official test table, identifier and file-hash tables, review decisions
audit/       near-duplicate screen: candidate pairs and similarity summary
results/     evaluation table, learning-rate grid and one folder per trained network
analysis/    integrity checks, statistics, figures and verification of the reported numbers
training/    training pipeline, executed as numbered cells
tools/       construction of the groups and partitions, duplicate and near-duplicate screens
figures/     the figures of the paper, drawn from the released predictions
```

## Groups and partitions

`resources/splits_isic2019.csv` has one row per training image.

| Column | Content |
|---|---|
| `image`, `label`, `target`, `source` | image name, diagnosis, class index and source collection |
| `lesion_id`, `lesion_id_archive`, `patient_id_archive` | dataset lesion identifier and ISIC Archive lesion and patient identifiers |
| `md5_bytes` | MD5 hash of the image file |
| `group_id`, `group_size` | patient-extended group and its number of images |
| `p1_image_level`, `p2_lesion_level` | `train`, `val` or `test` under the image-level and the lesion-level split |
| `cv{k}_lesion`, `cv{k}_image` | five grouped and five stratified cross-validation folds |
| `age_approx`, `sex`, `anatom_site_general` | ISIC metadata |

The class order is MEL, NV, BCC, AK, BKL, DF, VASC, SCC.

A lesion group is a connected component of links between images that share a dataset lesion identifier, an ISIC
Archive lesion identifier or a byte-identical or thumbnail-identical file, or that were accepted as the same lesion in
the visual review of the near-duplicate screen (`resources/neardup_adjudication.csv`). These links give 13,913 lesion
groups. The ISIC Archive patient identifier, available for 417 images, joins 78 of these lesion groups into 30 groups.
Patient links act at patient level and can join different lesions of one patient, so the released `group_id` marks
13,865 patient-extended groups, called groups below. All splits, leakage counts and cluster bootstraps use these
groups, which makes the lesion-level split stricter for the joined images. `lesion_groups()` in
`analysis/paired_statistics.py` rebuilds the 13,913 lesion groups from the released tables.
`resources/split_audit.json` summarises the groups and the contribution of each link type.

To evaluate a classifier on the lesion-level partitions:

```python
import pandas as pd

splits = pd.read_csv("resources/splits_isic2019.csv")
train = splits[splits.p2_lesion_level == "train"]
val = splits[splits.p2_lesion_level == "val"]
test = splits[splits.p2_lesion_level == "test"]
assert not set(test.group_id) & set(train.group_id)      # no group shared with training
```

`resources/official_test.csv` lists the official test images with their labels, ISIC Archive lesion identifiers and
acquisition source. The analyses use the 5920 images with `scored` and `known_class` true.

## Trained networks and predictions

Each network is stored in `results/runs/<split>/<run>/<seed>/`. `<split>` is `image` (P1), `lesion` (P2), or
`image_cv{k}` and `lesion_cv{k}` for the cross-validation folds.

| File | Content |
|---|---|
| `pred_{val,test,official}.csv.gz` | logits of the final-epoch weights, the primary model of every run |
| `pred_{val,test,official}_best.csv.gz` | logits of the best-validation checkpoint, used in a sensitivity analysis |
| `history.csv` | training loss and validation macro-F1 and balanced accuracy per epoch |
| `done.json` | configuration, summary metrics and software environment |

A prediction file has the columns `image`, `y_true`, `group` (patient-extended group, or ISIC Archive lesion for
the official test) and one logit `z_<class>` per class. `results/evaluation_long.csv` holds the metrics of every run and checkpoint.

| Run | Model | Recipe | Seeds under P1 / P2 |
|---|---|---|---|
| `R1_ResNet50` | ResNet-50 | plain | 42 / 42 |
| `R2_EfficientNetV2S` | EfficientNetV2-S | plain | 42 / 42 |
| `R3_ViTS16` | ViT-S/16 | plain | 42 / 42 |
| `R4_SwinV2T` | SwinV2-T | plain | 42 / 42, 43, 44 |
| `R5_ConvNeXtV2T` | ConvNeXtV2-T | plain | 42 / 42, 43, 44 |
| `R6_MaxViTT` | MaxViT-T | plain | 42 / 42 |
| `B1_SwinV2T` | SwinV2-T | shared | 42 / 42, 43, 44 |
| `B1c_ConvNeXtV2T` | ConvNeXtV2-T | shared | 42 / 42, 43, 44 |
| `B2_SwinV2T_GLC` | SwinV2-T + GLC | shared | 42 / 42, 43, 44 |
| `B3_ConvNeXtV2T_GLC` | ConvNeXtV2-T + GLC | shared | 42 / 42, 43, 44 |
| `C2_ConvNeXtV2T_384_GLC2` | ConvNeXtV2-T + GLC-v2 at 384 px, exploratory | shared | 42 / 42, 43, 44 |
| `CV_R4_SwinV2T` | SwinV2-T, cross-validation | plain | 42 in every fold |

The first ten rows form the main analysis. Fold 0 of the grouped cross-validation is identical to the lesion-level
split, so `lesion_cv0` holds a copy of `lesion/R4_SwinV2T/seed42`. The checkpoints of the pretrained backbones are the
timm models `resnet50.a1_in1k`, `tf_efficientnetv2_s.in21k_ft_in1k`, `vit_small_patch16_224.augreg_in21k_ft_in1k`,
`swinv2_tiny_window16_256.ms_in1k`, `convnextv2_tiny.fcmae_ft_in22k_in1k`, `maxvit_rmlp_tiny_rw_256.sw_in1k` and
`convnextv2_tiny.fcmae_ft_in22k_in1k_384`.

## Where each result comes from

| Result in the paper | Reproduced by |
|---|---|
| Groups, lesion groups, partition sizes, leaked share of the image-level split | `analysis/check_splits.py`, `resources/split_audit.json` |
| Leaked share by class | `resources/leakage_by_class.csv` |
| Macro-F1, balanced accuracy, accuracy, AUROC and sensitivities of every run | `analysis/verify_results.py`, `results/evaluation_long.csv` |
| Inflation and DiD per model and on average, with 95% intervals | `analysis/verify_results.py`, `analysis/paired_statistics.py` |
| Sensitivity analyses: baselines only, each institution, best-validation checkpoints, tripled seed SD, 384-px network | `analysis/verify_results.py` |
| Official-test gain with its interval, seed break-even | `analysis/paired_statistics.py` |
| Placebo DiD and protocol comparison on the 1004 common test images, class-standardised accuracy | `analysis/paired_statistics.py` |
| Accuracy by number of same-group training images | `analysis/paired_statistics.py` |
| Melanoma and malignant sensitivity and specificity | `analysis/verify_results.py`, `analysis/paired_statistics.py` |
| Institutions of the official test, class-mix exchange | `analysis/paired_statistics.py` |
| Cross-validation folds and their results | `analysis/check_splits.py`, `analysis/verify_results.py` |
| Patient links, conflicting labels of identical files | `analysis/paired_statistics.py` |
| Rank agreement, validation dynamics, checkpoint comparison, lesion-aggregated predictions | `analysis/verify_results.py` |
| Near-duplicate review | `resources/neardup_adjudication.csv`, `audit/` |

`analysis/paired_statistics.py` writes all quantities to `analysis/statistics_results.json`. The nearest-neighbour analysis and
the near-duplicate sensitivity analysis also need the frozen ResNet-50 features of the training images, which
`tools/leakage_probe.py` computes from the ISIC 2019 images into `_local_cache/`. Without them these two analyses are
skipped.

## Statistical analysis

Confidence intervals use a cluster bootstrap over the groups with B = 2000 replicates. Each replicate draws one
multinomial weight vector over all groups and applies it to both internal test partitions, which share images. Official test
images are resampled independently by ISIC Archive lesion. The same replicate weights are used for every model, seed
and split, and the generator seeds are fixed, so the intervals are reproduced exactly. Training-seed variance, pooled
within models, is added to the bootstrap variance, and normal-theory 95% intervals are reported. The image-level models
have one seed each, so their seed variance is set equal to that of the lesion-level models.

## Retraining

The networks were trained on one NVIDIA A100 80 GB GPU with Python 3.13, PyTorch 2.11, timm 1.0.30 and
albumentations 2.0.8. Training all 45 networks takes about 19 GPU hours. Every run uses 30 epochs, batch size 32,
AdamW, bfloat16 and the final-epoch weights.

```bash
export ISICLEAK_PROJECT=/path/to/this/repository
export ISICLEAK_DATA=/path/to/isic2019      # contains ISIC_2019_Training_Input/ and ISIC_2019_Test_Input/
python training/run_all.py                  # executes training/01_setup.py ... 09_run.py in one namespace
```

The same files can be run as notebook cells in numerical order, and on Colab the images are downloaded from the ISIC
challenge bucket. A run whose `done.json` exists is skipped, so move `results/runs` away to retrain. The peak learning
rate and layer decay of the shared recipe are read from `results/recipe_lock.json`, which records the grid of four
learning rates with and without layer decay on the lesion-level validation partition. Without this file the grid runs
first. `ISICLEAK_SMOKE=1` runs the whole pipeline on a tiny subset with two epochs. GPU kernels are not bit-deterministic,
so a retrained network differs slightly from the released one, within the reported seed variance.

## Rebuilding the groups

```bash
python tools/resolve_archive_lesion_ids.py ISIC_2019_Training_GroundTruth.csv ISIC_2019_Test_GroundTruth.csv resources/isic2019_archive_ids.csv
python tools/duplicate_audit.py ISIC_2019_Training_Input.zip resources/isic2019_train_hashes.csv
python tools/isic_splits.py ISIC_2019_Training_GroundTruth.csv ISIC_2019_Training_Metadata.csv .
python tools/neardup_audit.py . ISIC_2019_Training_Input.zip ISIC_2019_Test_Input.zip
python tools/neardup_global.py . ISIC_2019_Training_Input.zip
```

The ground-truth and metadata files are those of the ISIC 2019 challenge. `isic_splits.py` reads the archive
identifiers, the file hashes and the review decisions in `resources/` and writes the splits, the official test table and
the audit summary.

## Figures

`analysis/make_figures.py` draws the figures into `figures/` from the released predictions and tables.

| File | Figure in the paper |
|---|---|
| `fig_dataset.pdf` | Figure 1, class distribution of the training set |
| `fig_examples.png` | Figure 2, example images; needs `resources/figure_tiles_256.npz`, built from the ISIC images, and is skipped without it |
| `fig1_design.pdf` | Figure 3, paired evaluation design |
| `fig_leakage.pdf` | Figure 4, same-group leakage by class and accuracy by number of same-group training images |
| `fig_common.pdf` | Figure 5, test images common to both splits and per-class F1 inflation |
| `fig3_estimate_vs_official.pdf` | Figure 6, internal and official macro-F1 of each model |
| `fig_confusion.pdf` | Figure 7, confusion matrices of the reference model, ConvNeXtV2-T with GLC |

## Data and licences

The ISIC 2019 images, labels and metadata are available from https://challenge.isic-archive.com/data/ and are not
redistributed here. The ISIC 2019 training data are released under CC BY-NC. As the challenge requires, the data are
attributed as follows:

* BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
* HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
* MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

When the data are used, please cite:

1. Tschandl, P.; Rosendahl, C.; Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic
   images of common pigmented skin lesions. *Sci. Data* **2018**, 5, 180161. https://doi.org/10.1038/sdata.2018.161
2. Codella, N.C.F.; Gutman, D.; Celebi, M.E.; Helba, B.; Marchetti, M.A.; Dusza, S.W.; Kalloo, A.; Liopyris, K.;
   Mishra, N.; Kittler, H.; Halpern, A. Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017
   International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC).
   arXiv:1710.05006, 2017.
3. Hernández-Pérez, C.; Combalia, M.; Podlipnik, S.; Codella, N.C.F.; Rotemberg, V.; Halpern, A.C.; Reiter, O.;
   Carrera, C.; Barreiro, A.; Helba, B.; Puig, S.; Vilaplana, V.; Malvehy, J. BCN20000: Dermoscopic lesions in the wild.
   *Sci. Data* **2024**, 11, 641. https://doi.org/10.1038/s41597-024-03387-w

The derived tables and predictions in `resources/`, `audit/` and `results/` are released under CC BY-NC 4.0, and the
code under the MIT licence.

## Citation

Citation details will be added after publication.
