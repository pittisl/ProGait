# ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users

This is the official repository for the paper ["ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users"](https://arxiv.org/abs/2211.15692) (ICCV'23). The repo contains Human3.6M 3D WholeBody (H3WB) annotations proposed in this paper.

For the 3D whole-body benchmark and results please refer to [benchmark.md](benchmark.md).

## 🆕Updates
- **`2025/8/12`** We have published our dataset at Hugging Face.
  - [Link to download dataset](https://huggingface.co/datasets/ericyxy98/ProGait).

## Table of Content
- [About H3WB](#what-is-h3wb)
- [Dataset](#h3wb-dataset)
- [Pretrained models](#pretrained-models)
- [Tasks](#tasks)
- [Evaluation](#evaluation)
- [How to cite](#how-to-cite)

## What is ProGait

ProGait is a multi-purpose video dataset aimed to support multiple vision tasks on prosthesis users, including Video Object Segmentation, 
2D Human Pose Estimation, and Gait Analysis. ProGait provides 412 video clips from four above-knee amputees when testing multiple 
newly-fitted prosthetic legs through walking trials, and depicts the presence, contours, poses, and gait patterns of human subjects with 
transfemoral prosthetic legs.

Example annotations:

<img src="imgs/example.pdf" width="800" height="400">


## ProGait Dataset

### Download

- The raw videos and annotations can be downloaded from the [HERE]([http://vision.imar.ro/human3.6m/](https://huggingface.co/datasets/ericyxy98/ProGait)).

### Usage

- TBD

## Baseline models

ProGait comes with baseline models fine-tuned on this datasets. Please find chekpoints [HERE]():

## Tasks

ProGait provides annotations for 3 different tasks:

#### Video Object Segmentation (VOS)
 - Bounding boxes and segmentation masks of the prothesis user

#### 2D Human Pose Estimation (HPE)
 - 23 pose keypoints of the target (17 for body and 6 for feet, following the COCO-wholebody keypoints definition)

#### Gait Analysis
 - Text descriptions of four key components:
    - The general gait category
    - The specific gait deviation
    - Recommendations on how to adjust the prosthesis to correct the gait
    - The reasons of these recommendations

## Evaluation


## Terms of Use

1. This project is released under the [MIT License](https://github.com/pittisl/ProGait/LICENSE.md). 

## How to cite

If you find ProGait dataset useful for your project, please cite our paper as follows.

> Xiangyu Yin, Boyuan Yang, Weichen Liu, Qiyao Xue, Abrar Alamri, Goeran Fiedler, Wei Gao, "ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users", ICCV, 2025.

BibTeX entry:
```
TBD
```
