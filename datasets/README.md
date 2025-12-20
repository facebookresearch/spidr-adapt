# Dataset Preperation

## Download Voxpopuli data

First download raw Voxpopuli data by following the instructions under https://github.com/facebookresearch/voxpopuli. This raw data can be deleted once data segmentation is complete.

All subsets except Danish should be downloaded. Download to `$DATA`

Following downloading, you should have the following structure.
```
> tree -L 2 $DATA
└── raw_audios
    ├── bg
    ├── cs
    ├── de
    ├── ...
    └── sv
```

## Prepare Train datasets

To segment the raw data into training utterences and create train manifest files, run the following:

```
python datasets/prepare_train_datasets.py --dataset_path $DATA --train_subset [TRAIN_SUBSET]
```

`TRAIN_SUBSET` specifies which of the training sets to prepare: `vp_19lang` (OoD train set), `vp_en` [ID train set], `vp_fr` [ID train set], `vp_de` [ID train set]

This script will download needed metadata and use the metadata to segment Voxpopuli raw data and create manifest files for training. Training manifests will be stored under `$DATA/spidr-adapt/train` and segmented_audios will be stored under `$DATA/spidr-adapt/segmented_audios`.

The raw Voxpopuli data can now be deleted.

## Prepare Adaptation datasets

To prepare adaptation datasets, run the following:

```
python datasets/prepare_adapt_datasets.py --dataset_path $DATA --adapt_subset [ADAPT_SUBSET]
```

`ADAPT_SUBSET` specifies which of the test adaptation sets to prepare: `vp_en`, `vp_fr`, `vp_de`.

This script will download the needed metadata and use it to create the adaptation manifest files. The utterences in these adaptation datasets are derived from the ID train sets and will point to the corresponding .wav files.


You should now have the following file structure where `$DATA/spidr-adapt/adapt` contains adaptation data split:
```
> tree -L 3 $DATA
└── spidr-adapt
    ├── segmented_audios
    │   ├── bg
    │   ├── cs
    │   ├── de
    │   ├── ...
    │   └── sv
    ├── adapt
    │   ├── vp_en
    │   ├── vp_fr
    │   ├── vp_de
    └── train
        ├── vp_19lang_manifest.csv
        ├── vp_en_manifest.csv
        ├── vp_fr_manifest.csv
        └── vp_de_manifest.csv
```
