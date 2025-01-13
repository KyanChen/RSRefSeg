<div align="center">
    <h2>
        RSRefSeg: Referring Remote Sensing Image Segmentation with Foundation Models
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/RSRefSeg.png" width="800"/>
</div>
<br>
<div align="center">
  <a href="https://github.com/KyanChen/RSRefSeg">
    <span style="font-size: 20px; ">Homepage</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/xxxx">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/RSRefSeg.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/RSRefSeg)](https://github.com/KyanChen/RSRefSeg)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2403.xx-b31b1b.svg)](https://arxiv.org/abs/2403.xxx)

<br>
<br>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>


## Introduction

This repository contains the code implementation for the paper [RSRefSeg: Referring Remote Sensing Image Segmentation with Foundation Models](https://arxiv.org/abs/2403.xxx), developed based on the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) project.

The current branch has been tested on Linux with PyTorch 2.x and CUDA 12.1, supports Python 3.10+, and can be compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️. Your support is our greatest motivation.

<details open>
<summary>Main Features</summary>

- Consistent API and usage with MMSegmentation
- Open-sourced different versions of the RSRefSeg model as mentioned in the paper
- Supported training and testing with multiple datasets

</details>

## Changelog

🌟 **2025.01.12** Released the RSRefSeg project with APIs consistent with MMSegmentation.


## Contents

- [Introduction](#introduction)
- [Changelog](#changelog)
- [Contents](#contents)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Image Prediction](#image-prediction)
- [FAQ](#faq)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### Dependencies

- Linux system, Windows is also supported
- Python 3.10+, recommended to use 3.11
- PyTorch 2.0 or higher, recommended to use 2.3
- CUDA 11.7 or higher, recommended to use 12.1
- MMCV 2.0 or higher, recommended to use 2.2

### Environment Setup

We recommend setting up the environment using Miniconda. The following commands will create a virtual environment named `rsrefseg` and install PyTorch and MMCV. The default installation steps assume CUDA version **12.1**. If your CUDA version is different, please adjust accordingly.

Note: If you are experienced with PyTorch and already have it installed, you can skip to the next section. Otherwise, follow the steps below for preparation.

<details open>

**Step 0**: Install [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).

**Step 1**: Create a virtual environment named `rsrefseg` and activate it.

```shell
conda create -n rsrefseg python=3.11 -y
conda activate rsrefseg
```

**Step 2**: Install [PyTorch2.3.x](https://pytorch.org/get-started/previous-versions/).

Linux/Windows:

```shell
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
or
```shell
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV2.1.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.2.0
# or
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
```

**Step 4**: Install other dependencies.

```shell
pip install modelindex ipdb ms-swift transformers peft modelscope accelerate qwen_vl_utils pycocotools -U
```


**Step 5**: [Optional] Install DeepSpeed.

If you want to use DeepSpeed to train models, you need to install DeepSpeed and uncomment the `DeepSpeed training config` in the Config file. The installation method for DeepSpeed can be found in the DeepSpeed official documentation.

```shell
pip install deepspeed
```

Note: DeepSpeed support on Windows is not yet complete, and we recommend using DeepSpeed on Linux systems. Windows systems can only use AMP training, and we recommend uncommenting the `AMP training config` in the Config file.

</details>


### Install RSRefSeg

Download or clone the RSRefSeg repository.

```shell
git clone git@github.com:KyanChen/RSRefSeg.git
cd RSRefSeg
```

## Dataset Preparation

<details open>

### Referring Remote Sensing Image Segmentation Datasets

We provide methods to prepare the Referring Remote Sensing Image Segmentation datasets used in the paper.

#### RRSIS-D Dataset

- Download link for images and annotations: [RRSIS-D Dataset](https://github.com/Lsan2401/RMSIN#Datasets).


#### Organization Format

You can download datasets from other sources as well but need to organize them in the following format:

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/data
├── rrsisd
│   ├── refs(unc).p
│   └── instances.json
├── images
    └── rrsisd
        ├── JPEGImages
        └── ann_split
```

#### Dataset Conversion

We provide a script to convert the dataset into the required format and generate JSONL files.

Note: We have already provided the converted JSONL files in the `datainfo` folder. You can directly use them. Additionally, we provide a [Python script](tools_RSRefSeg/convert_data_to_jsonl.py) for dataset conversion.


### Other Datasets

If you wish to use other datasets, refer to [this Python script](tools_RSRefSeg/convert_data_to_jsonl.py) for dataset preparation.

</details>

## Model Training

### RSRefSeg Model

#### Config Files and Key Parameter Explanations

We provide configuration files for RSRefSeg models of different parameter sizes as mentioned in the paper, which you can find in the [configs_RSRefSeg](configs_RSRefSeg) folder. Config files maintain consistency with the MMSegmentation API and usage. Below are some key parameter explanations. For more detailed descriptions of parameters, refer to the [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html).

<details>

**Parameter Explanations**:

- `work_dir`: Output path for model training, usually does not need modification.
- `data_root`: Dataset root directory, **modify to the absolute path of the dataset root directory**.
- `batch_size`: Batch size per GPU, **needs adjustment depending on VRAM size**.
- `max_epochs`: Maximum number of training epochs, usually does not need modification.
- `val_interval`: Interval for validation set, usually does not need modification.
- `vis_backends/WandbVisBackend`: Configuration for network-side visualization tools, **open the comment if needed, requires registering an account on `wandb` website to view visual results of the training process in a web browser**.
- `resume`: Whether to resume from checkpoint, usually does not need modification.
- `load_from`: Pre-trained checkpoint path for the model, usually does not need modification.
- `init_from`: Pre-trained checkpoint path for the model, usually keep as None unless resuming from a checkpoint, in which case modify it accordingly.
- `default_hooks/CheckpointHook`: Configuration for checkpoint saving during model training, usually does not need modification.
- `model/lora_cfg`: Configuration for efficient model tuning, usually does not need modification.
- `model/backbone`: Visual backbone of SAM model, **adjust according to actual needs**, base corresponds to `sam-vit-base`, large corresponds to `sam-vit-large`, huge corresponds to `sam-vit-huge`.
- `model/clip_vision_encoder`: Visual encoder of the CLIP model, usually does not need modification.
- `model/clip_text_encoder`: Text encoder of the CLIP model, usually does not need modification.
- `model/sam_prompt_encoder`: Prompt encoder of the SAM model, usually does not need modification.
- `model/sam_mask_decoder`: Decoder of the SAM model, usually does not need modification.
- `model/decode_head`: Pseudo decode head of the RSRefSeg model, usually does not need modification.
- `AMP training config`: Configuration for mixed precision training. If not using DeepSpeed training, uncomment this section. Usually does not need modification.
- `DeepSpeed training config`: Configuration for DeepSpeed training. If using DeepSpeed training, uncomment this section and comment out `AMP training config`. Note that DeepSpeed training is not supported on Windows.
- `dataset_type`: Dataset type, usually does not need modification.
- `data_preprocessor/mean/std`: Mean and standard deviation for data preprocessing, usually does not need modification.

</details>


#### Single-GPU Training

```shell
python tools/train.py configs_RSRefSeg/name_to_config.py  # Replace 'name_to_config.py' with the desired config file
```

#### Multi-GPU Training

```shell
sh tools/dist_train.sh configs_RSRefSeg/name_to_config.py ${GPU_NUM}  # Replace 'name_to_config.py' with the desired config file, GPU_NUM with the number of GPUs to use
```

## Model Testing

#### Single-GPU Testing

```shell
python tools/test.py configs_RSRefSeg/name_to_config.py ${CHECKPOINT_FILE}  # Replace 'name_to_config.py' with the desired config file, 'CHECKPOINT_FILE' with the desired checkpoint file
```

#### Multi-GPU Testing

```shell
sh tools/dist_test.sh configs_RSRefSeg/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # Replace 'name_to_config.py' with the desired config file, 'CHECKPOINT_FILE' with the desired checkpoint file, and GPU_NUM with the number of GPUs to use
```

## Image Prediction

#### Predict a Single Image

```shell
python demo/image_demo.py ${IMAGE_FILE}  configs_RSRefSeg/name_to_config.py --checkpoint ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}  # Replace 'IMAGE_FILE' with the image file to predict, 'name_to_config.py' with the desired config file, 'CHECKPOINT_FILE' with the desired checkpoint file, and 'OUTPUT_DIR' with the output directory for prediction results
```

#### Predict Multiple Images

```shell
python demo/image_demo.py ${IMAGE_DIR}  configs_RSRefSeg/name_to_config.py --checkpoint ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}  # Replace 'IMAGE_DIR' with the image directory to predict, 'name_to_config.py' with the desired config file, 'CHECKPOINT_FILE' with the desired checkpoint file, and 'OUTPUT_DIR' with the output directory for prediction results
```

## FAQ

<details open>

We have listed some common problems and their solutions for usage. If you find any issue missing here, feel free to open a PR to enrich this list. If you cannot find help here, please use [issue](https://github.com/KyanChen/RSRefSeg/issues) to seek help. Please fill out all required information in the template, which helps us locate the problem faster.

### 1. Do I need to install MMSegmentation?

We recommend not installing MMSegmentation as we have made some modifications to the code of MMSegmentation, and installing it might cause errors. If you encounter a 'Module not registered' error, please check:

- If the module is a package that needs to be installed, install it if necessary.
- Whether MMSegmentation is installed; if so, uninstall it.
- Whether `@MODELS.register_module()` is added before class names; add it if not.
- Whether `from .xxx import xxx` is added in `__init__.py`; add it if not.
- Whether `custom_imports = dict(imports=['rsris'], allow_failed_imports=False)` is added to the config file; add it if not.


### 2. Solution for 'Bad substitution' error in dist_train.sh

If you encounter a 'Bad substitution' error while running `dist_train.sh`, use `bash dist_train.sh` to execute the script.

</details>

## Acknowledgments

This project was developed based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), thanks to the developers of the MMSegmentation project.

## Citation

If you use the code or benchmarks from this project in your research, please cite RSRefSeg using the following bibtex.

```
@article{chen2025rsrefseg,
  title={RSRefSeg: Referring Remote Sensing Image Segmentation with Foundation Models},
  author={Chen, Keyan and Zhang, Jiafan and Liu, Chenyang and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2501.xxxx},
  year={2025}
}
```

## License

This project uses the [Apache 2.0 open source license](LICENSE).

## Contact

For further questions❓, feel free to contact us 👬