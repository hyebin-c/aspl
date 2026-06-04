# Acoustic Prompting via Stage-wise Modulation for Few-Shot Learning in Audio Language Models (INTERSPEECH 2026)

> [**Acoustic Prompting via Stage-wise Modulation for Few-Shot Learning in Audio Language Models**]()
>
> [Hyebin Cho](https://hyebin-c.github.io/), [Jaehyuk Jang](https://sites.google.com/view/jaehyukjang), [Changick Kim](https://cilabs.kaist.ac.kr/members/professor), and [Joon Son Chung](https://mm.kaist.ac.kr/joon/)

<hr />

| ![main figure](/media/aspl.png) |
|:--|
| <p align="justify">This repository implements our method, ASPL (Audio-side Prompt Learning), for few-shot audio classification. ASPL is designed as a plug-in module that can be integrated into CoOp, CoCoOp, and PALM, and is built on top of the PENGI audio-language model.</p> |

<hr />

## Overview

This repository contains:

- ASPL (Audio-side Prompt Learning) for few-shot audio classification
- audio-side prompting with `ASPL`
- stage-wise audio prompting with `ASPL+`
- training scripts for PALM, CoOp, and CoCoOp variants on multiple audio classification datasets

## Table of Contents

- [Installation](#installation)
- [Model](#model)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

<a name="installation"/>

## Installation

1. Create a conda environment.

```bash
conda create --name aspl python=3.8
conda activate aspl
```

2. Install dependencies.

```bash
git clone <REPO_URL>
cd palm
pip install -r requirements.txt
```

<a name="model"/>

## Model

All experiments use [PENGI](https://github.com/microsoft/Pengi) as the underlying audio-language model.

Download the pre-trained PENGI checkpoint and place it in [`pengi/configs`](/pengi/configs).

| Model | Link | Size |
|:--|:--|:--:|
| PENGI | [Download](https://zenodo.org/records/8387083/files/base.pth) | 2.2 GB |

You can also download it with:

```bash
wget https://zenodo.org/records/8387083/files/base.pth
```

<a name="datasets"/>

## Datasets

We keep the dataset preparation pipeline from the original PALM setup. Instructions for downloading and processing datasets are provided in [DATASETS.md](DATASETS.md). A Jupyter notebook for downloading datasets is also provided at [media/DownloadAudioDatasets.ipynb](/media/DownloadAudioDatasets.ipynb).

| Dataset | Type | Classes | Size | Link |
|:-- |:-- |:--: |--: |:-- |
| [Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset) | Instrument Classification | 4 | 69 MB | [Instructions](DATASETS.md#beijing-opera) |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Emotion Recognition | 6 | 606 MB | [Instructions](DATASETS.md#crema-d) |
| [ESC50](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 50 | 881 MB | [Instructions](DATASETS.md#esc50) |
| [ESC50-Actions](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 10 | 881 MB | [Instructions](DATASETS.md#esc50-actions) |
| [GT-Music-Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Music Analysis | 10 | 1.3 GB | [Instructions](DATASETS.md#gt-music-genre) |
| [NS-Instruments](https://magenta.tensorflow.org/datasets/nsynth) | Instrument Classification | 10 | 18.5 GB | [Instructions](DATASETS.md#ns-instruments) |
| [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8) | Emotion Recognition | 8 | 1.1 GB | [Instructions](DATASETS.md#ravdess) |
| [SESA](https://zenodo.org/records/3519845) | Surveillance Sound Classification | 4 | 70 MB | [Instructions](DATASETS.md#sesa) |
| [TUT2017](https://zenodo.org/records/400515) | Acoustic Scene Classification | 15 | 12.3 GB | [Instructions](DATASETS.md#tut2017) |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Sound Event Classification | 10 | 6.8 GB | [Instructions](DATASETS.md#urbansound8k) |
| [VocalSound](https://github.com/YuanGongND/vocalsound) | Vocal Sound Classification | 6 | 8.2 GB | [Instructions](DATASETS.md#vocalsound) |

All datasets should be placed in a directory named `Audio-Datasets`, and the path should be configured through `DATASET_ROOT` in the shell scripts under [`scripts`](/scripts/).

Expected directory structure:

```text
Audio-Datasets/
    ├── Beijing-Opera/
    ├── CREMA-D/
    ├── ESC50/
    ├── ESC50-Actions/
    ├── GT-Music-Genre/
    ├── NS-Instruments/
    ├── RAVDESS/
    ├── SESA/
    ├── TUT2017/
    ├── UrbanSound8K/
    ├── VocalSound/
```

<a name="code-structure"/>

## Code Structure

There are three main folders in this repository.

- [`pengi`](/pengi): PENGI-based model components and audio encoder code
- [`palm`](/palm): PALM, CoOp, CoCoOp, and ASPL/ASPL+ model implementations
- [`utils`](/utils): dataset loading, training, evaluation, and logging utilities

<a name="run-experiments"/>

## Run Experiments

The current release focuses on the `ASPL` and `ASPL+` settings.

- `ASPL`: pass `1`
- `ASPL+`: pass `2`

### PALM

```bash
bash scripts/run_all_datasets_palm.sh 1
bash scripts/run_all_datasets_palm.sh 2
```

### CoOp

```bash
bash scripts/run_all_datasets_coop.sh 1
bash scripts/run_all_datasets_coop.sh 2
```

### CoCoOp

```bash
bash scripts/run_all_datasets_cocoop.sh 1
bash scripts/run_all_datasets_cocoop.sh 2
```

The launcher scripts currently fix the following settings:

- `LR=0.01`
- `EPOCH=100`
- `SHOT=16`

By default, the launcher scripts use `CUDA_VISIBLE_DEVICES=0`. If you want to run on a different GPU, override it at launch time:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_datasets_palm.sh 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_datasets_coop.sh 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_datasets_cocoop.sh 2
```

Logs are saved under [`logs`](/logs) using directories such as:

- `logs/palm_aspl1_16`
- `logs/palm_aspl2_16`
- `logs/coop_aspl1_16`
- `logs/cocoop_aspl2_16`

<a name="citation"/>

## Citation

Citation information will be updated here.

<a name="acknowledgement"/>

## Acknowledgement

We use [PENGI](https://github.com/microsoft/Pengi) for model instantiation. This repository builds on the original [PALM codebase](https://github.com/asif-hanif/palm/tree/main), which already includes CoOp and CoCoOp-based prompt learning implementations adapted from [CoOp](https://github.com/KaiyangZhou/CoOp) and [CoCoOp](https://github.com/KaiyangZhou/CoOp).
