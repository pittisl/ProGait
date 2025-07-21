---
license: cc-by-nc-sa-4.0
task_categories:
- keypoint-detection
- mask-generation
- video-classification
tags:
- progait
configs:
- config_name: default
  data_files:
  - videos/inside/*.mp4
  - videos/outside/*.mp4
  - previews/inside/*.mp4
  - previews/outside/*.mp4
  - metadata.jsonl
  drop_labels: true
language:
- en
---

# ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users

This is the official repository for the paper ["ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users"](https://arxiv.org/abs/2507.10223) (ICCV'25).

## What is ProGait

ProGait is a multi-purpose video dataset aimed to support multiple vision tasks on prosthesis users, including Video Object Segmentation, 
2D Human Pose Estimation, and Gait Analysis. ProGait provides 412 video clips from four above-knee amputees when testing multiple 
newly-fitted prosthetic legs through walking trials, and depicts the presence, contours, poses, and gait patterns of human subjects with 
transfemoral prosthetic legs.

## Tasks

ProGait provides annotations for 3 different tasks:

#### Video Object Segmentation (VOS)
 - Bounding boxes and segmentation masks of the prothesis user

#### 2D Human Pose Estimation (HPE)
 - 23 pose keypoints of the target (17 for body and 6 for feet, following the [COCO-wholebody](https://github.com/jin-s13/COCO-WholeBody) definition)

#### Gait Analysis
 - Text descriptions of four key components:
    - The general gait category
    - The specific gait deviation
    - Recommendations on how to adjust the prosthesis to correct the gait
    - The reasons of these recommendations

## Disclaimer

We are aware that Orthocare Innovations PLLC also used "ProGait" as the name of their mobile app product. Our work and dataset are not affiliated with Orthocare Innovations PLLC, and are not associated with their ProGait app, Europa+ system, or any other product.

## How to cite

If you find ProGait dataset useful for your project, please cite our paper as follows.

> Xiangyu Yin, Boyuan Yang, Weichen Liu, Qiyao Xue, Abrar Alamri, Goeran Fiedler, Wei Gao, "ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users", ICCV, 2025.