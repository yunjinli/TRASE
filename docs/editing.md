# Further Scene Editing Application

## Style Transfer

Style Transfer of the selected object (segment_ids).

```
python train_style_transfer_nnfm.py -s data/<DATASET>/<NAME> -m output/<DATASET>/<NAME> --load2gpu_on_the_fly --load_mask_on_the_fly --load_image_on_the_fly --load_iteration 30000 --iterations 35000 --eval --monitor_mem --save_iterations 35000 --segment_ids <0> <1> <...> --reference_img_path ./styles/<picture>.png
```

The implementation refers to StyleSplat.

```
@article{jain2024stylesplat,
  title={StyleSplat: 3D Object Style Transfer with Gaussian Splatting},
  author={Jain, Sahil and Kuthiala, Avik and Sethi, Prabhdeep Singh and Saxena, Prakanshul},
  journal={arXiv preprint arXiv:2407.09473},
  year={2024}
}
```
