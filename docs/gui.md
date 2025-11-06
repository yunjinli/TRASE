# Tutorial on GUI

## Running the Standalone GUI script

The standalone GUI script provides the same features as the normal GUI. We provide some examples for you to play with:

### Install example scenes

```
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
# If not, then
sudo apt-get install git-lfs
git clone https://huggingface.co/yunjinli/sadg_example_models
cd sadg_example_models
git lfs pull
```

### Run in Standalone GUI

```
python gui_standalone.py -m ./sadg_example_models/sear_steak --iteration 30000
## Or
python gui_standalone.py -m ./sadg_example_models/split-cookie --iteration 30000
```

As the standalone GUI doesn't have information regarding the length (in second) of the dynamic scene, this needs to be configured per scene in the GUI with the slider `Duration (sec)` (Default: 10 seconds). For detailed functionalities of the GUI, please see the following sections.

## Running the GUI script

```
python gui.py -m output/<DATASET>/<NAME> --load_mask_on_the_fly --load_image_on_the_fly --eval --load2gpu_on_the_fly --iteration 30000
```

## Changing Different Rendering Mode

- You can use the cursor to drag to different novel views in the GUI
- Scroll to adjust the FoV
- Select different rendering modes ( `Render, Rendered Features, Gaussian Features, Gaussian Clusters, Segmentation, Point Cloud, Depth` ) in the combo box
- Please make sure to run `Clustering` before rendering the in `Gaussian Clusters, Segmentation` mode

https://github.com/user-attachments/assets/c4781799-50ef-4897-9ff8-da13786271bf

## Click Prompt

- Make sure you run `Clustering` beforehand
- Drag the slider of `Freeze Time` to change to different time
- Hold the key `A` and `Left-Click` on the object of interest in the novel view
- Hold the key `D` and `Left-Click` to deselect the click
- Drag the slider of `Score Threshold` to adjust the threshold for filtering unwanted Gaussians

https://github.com/user-attachments/assets/54c468c5-476e-44de-a5e4-4d51c3e403c3

## Text Prompt

- Make sure you run `Clustering` beforehand
- Enter the text prompt and click `Enter`
- Click `Remove Object` to toggle the visibility of the selected objects (selected / removal)

https://github.com/user-attachments/assets/5c787cd5-8d78-44bc-87da-a6319105efc6

## Other Buttons

- You can change to different clustering methods in the combo box (DBSCAN / K-Means)
- `Render Mask` renders transparant segmentation mask on the rendering
- `Save Object` saves the current visible Gaussians (for later composition with other scenes)
- `Render Object` renders to the test camera views

## Acknowledgement

We appreciate the authors from [SC-GS](https://github.com/yihua7/SC-GS) for sharing their amazing work. Our GUI is built upon their code. Please consider also citing their paper.

```
@article{huang2023sc,
    title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
    author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
    journal={arXiv preprint arXiv:2312.14937},
    year={2023}
}
```
