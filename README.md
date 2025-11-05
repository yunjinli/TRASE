# TRASE: Tracking-free 4D Segmentation and Editing

## [Project page](https://yunjinli.github.io/project-sadg/) | [Paper](https://arxiv.org/abs/2411.19290)

## News

- 2025/05/11: We released the code for rendering and evaluation. We also updated the scripts for downloading / processing datasets. For more details, please check [documentation](./docs/).
- 2025/01/20: We released the [standalone GUI](./gui_standalone.py). See [GUI Tutorial](./docs/gui.md) for details.
- 2024/11/24: We released the [website](https://yunjinli.github.io/project-sadg/) for SADG
- 2024/11/23: We plan to release the rest of the source code and also the Mask-Benchmarks later. If you would like to compare your method against SADG now, please contact the first author via email (yunjin.li@tum.de).

## Introduction

We introduce TRASE, Tracking-free 4D Segmentation and Editing, a novel approach that combines dynamic Gaussian Splatting representation and semantic information without reliance on object IDs. We propose to learn semantically-aware features by leveraging masks generated from the Segment Anything Model (SAM) and utilizing our novel contrastive learning objective based on hard pixel mining. The learned Gaussian features can be effectively clustered without further post-processing. This enables fast computation for further object-level editing, such as object removal, composition, and style transfer by manipulating the Gaussians in the scene. Due to the lack of consistent evaluation protocol, we extend several dynamic novel-view datasets with segmentation benchmarks that allow testing of learned feature fields from unseen viewpoints. We evaluate SADG on proposed benchmarks and demonstrate the superior performance of our approach in segmenting objects within dynamic scenes along with its effectiveness for further downstream editing tasks.

![teaser](assets/teaser.jpg)

## Installation

```
## Setup the environment
git clone https://github.com/yunjinli/SADG-SegmentAnyDynamicGaussian.git
cd SADG
git submodule update --init --recursive
conda create -n SADG python=3.8 -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python plyfile tqdm scipy wandb opencv-python scikit-learn lpips imageio[ffmpeg] dearpygui kmeans_pytorch hdbscan scikit-image bitarray
python -m pip install submodules/diff-gaussian-rasterization
python -m pip install submodules/simple-knn

## Install SAM weights
cd dependency
bash install.bash

## For enabling Text Prompt
git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
```

Note: If you have an error from Grounding-DINO: `TypeError: annotate() got an unexpected keyword argument 'labels'`, install `Supervision` to the 0.21.0 version

```
pip install supervision==0.21.0
```

## Dataset Preparation

See [here](./docs/prepare_dataset.md)

## Train

~~See [here](./docs/train.md)~~ (TBA)

## GUI

See [here](./docs/gui.md)

## Render

See [here](./docs/render.md)

## Scene Editing Application

~~See [here](./docs/editing.md)~~ (TBA)

## Evaluation on our Mask-Benchmarks

See [here](./docs/evaluation.md)

## BibTex

```
@article{li2024sadg,
    title={SADG: Segment Any Dynamic Gaussian Without Object Trackers},
    author={Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Cremers, Daniel},
    journal={arXiv preprint arXiv:2411.19290},
    year={2024}
}
```

## Acknowledgement

We appreciate all the authors from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians), [SC-GS](https://github.com/yihua7/SC-GS), [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), [SAGA](https://github.com/Jumpat/SegAnyGAussians) for sharing their amazing works to promote further research in this area. Consider also citing their paper.

```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```

```
@article{huang2023sc,
    title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
    author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
    journal={arXiv preprint arXiv:2312.14937},
    year={2023}
}
```

```
@inproceedings{gaussian_grouping,
    title={Gaussian Grouping: Segment and Edit Anything in 3D Scenes},
    author={Ye, Mingqiao and Danelljan, Martin and Yu, Fisher and Ke, Lei},
    booktitle={ECCV},
    year={2024}
}
```

```
@article{cen2023saga,
      title={Segment Any 3D Gaussians},
      author={Jiazhong Cen and Jiemin Fang and Chen Yang and Lingxi Xie and Xiaopeng Zhang and Wei Shen and Qi Tian},
      year={2023},
      journal={arXiv preprint arXiv:2312.00860},
}
```
