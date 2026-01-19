# TRASE: Tracking-free 4D Segmentation and Editing

## [Project page](https://yunjinli.github.io/project-sadg/) | [Paper](https://arxiv.org/abs/2411.19290)

## News

- 2025/11/6: Accepted to 3DV 2026 👏. Release training code. The camera-ready version will be updated later.
- 2025/05/11: We released the code for rendering and evaluation. We also updated the scripts for downloading / processing datasets. For more details, please check [documentation](./docs/).
- 2025/01/20: We released the [standalone GUI](./gui_standalone.py). See [GUI Tutorial](./docs/gui.md) for details.
- 2024/11/24: We released the [website](https://yunjinli.github.io/project-sadg/).
- 2024/11/23: We plan to release the rest of the source code and also the Mask-Benchmarks later. If you would like to compare your method against us now, please contact the first author via email (yunjin.li@tum.de).

## Introduction

we introduce TRASE, a novel tracking-free 4D segmentation method for dynamic scene understanding. TRASE learns a 4D segmentation feature field in a weakly-supervised manner, leveraging a soft-mined contrastive learning objective guided by SAM masks. The resulting feature space is semantically coherent and well-separated, and final object-level segmentation is obtained via unsupervised clustering. This enables fast editing, such as object removal, composition, and style transfer, by directly manipulating the scene's Gaussians. We evaluate TRASE on five dynamic benchmarks, demonstrating state-of-the-art segmentation performance from unseen viewpoints and its effectiveness across various interactive editing tasks.

![teaser](assets/teaser.jpg)

## Installation

```
## Setup the environment
git clone https://github.com/yunjinli/TRASE.git
cd TRASE
git submodule update --init --recursive
conda create -n TRASE python=3.8 -y
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" ## We are using v0.7.6
pip install opencv-python plyfile tqdm scipy opencv-python scikit-learn lpips imageio[ffmpeg] dearpygui kmeans_pytorch hdbscan scikit-image bitarray
python -m pip install submodules/diff-gaussian-rasterization
python -m pip install submodules/simple-knn

## Install SAM weights
cd dependency
bash install.bash

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything ## Segment Anything is needed for running extract_masks.py
python -m pip install -e GroundingDINO ## For text prompt in the GUI
```

Note: If you have an error from Grounding-DINO: `TypeError: annotate() got an unexpected keyword argument 'labels'`, install `Supervision` to the 0.21.0 version

```
pip install supervision==0.21.0
```

## Dataset Preparation

See [here](./docs/prepare_dataset.md)

## Train

See [here](./docs/train.md)

## GUI

See [here](./docs/gui.md)

## Render

See [here](./docs/render.md)

## Scene Editing Application

See [here](./docs/editing.md)

## Evaluation on our Mask-Benchmarks

See [here](./docs/evaluation.md)

## BibTex

```
@article{li2024trase,
    title={TRASE: Tracking-free 4D Segmentation and Editing},
    author={Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Cremers, Daniel},
    journal={arXiv preprint arXiv:2411.19290},
    year={2024}
}
```

## Acknowledgement

We appreciate all the authors from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians), [SC-GS](https://github.com/yihua7/SC-GS), [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), [SAGA](https://github.com/Jumpat/SegAnyGAussians) for sharing their amazing works to promote further research in this area. Consider also citing their paper.

<details> <summary><b>Expand BibTeX</b></summary>
    
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
