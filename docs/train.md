# Training

The script [train.py](../train.py) is all you need :)

<details>
<summary><span style="font-weight: bold;">Important Command Line Arguments for train.py</span></summary>
      
  #### --source_path / -s
  Path to the source directory containing the dataset.
  #### --model_path / -m 
  Path where the trained model should be stored.
  #### --iterations
  The total number of iterations for training (```30_000``` by default).
  #### --warm_up
  The iteration index until which MLP optimization is paused (```3000``` by default).
  #### --warm_up_3d_features
  The iteration index until which the Gaussian features start to optimized. (```10000``` by default) Need to use together with the ```iterative_opt_interval```.
  #### --iterative_opt_interval
  The default mode is optimizaing only the colors and geometry of the scene. After ```iterative_opt_interval``` iteration, the mode changes to optimizing only the Gaussian features.
  #### --monitor_mem
  Configure to monitor the RAM and CUDA used.
  #### --lambda_reg_deform
  Apply regularization to the deformation of the Gaussians. (```0.0``` by default)
  #### --num_sampled_pixels
  Number of sampled pixels per image for contrastive semantically-aware learning. (```5000``` by default)
  #### --num_sampled_masks
  Number of sampled masks per image for contrastive semantically-aware learning. (```50``` by default)
  #### --smooth_K
  Number of neighbors for computing smooth Gaussian features (```16``` by default)
  #### --load2gpu_on_the_fly 
  Configure to load images / masks to VRAM on the fly. (For training on small GPU)
  #### --load_image_on_the_fly 
  Configure to load images from storage to RAM on the fly (For training on limited RAM)
  #### --load_mask_on_the_fly
  Configure to load masks from storage to RAM on the fly (For training on limited RAM)
  #### --eval
  Add this flag to do training/test split for evaluation.
  #### --white_background / -w
  Add this flag to use white background instead of black (default)
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model.
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --contrastive_mode
  Select different mode for contrastive learning. Default: soft (available modes: soft, all, hard)
  #### --mask_black_bg
  Configure to mask out loss on black background pixels. Only used in Technicolor dataset.

</details>

### NeRF-DS

```
python train.py -s data/NeRF-DS/<NAME> -m output/NeRF-DS/<NAME> --warm_up 3000 --warm_up_3d_features 15000 --iterative_opt_interval 20000 --iterations 30000 --test_iterations 5000 10000 20000 30000 --save_iterations 20000 30000 --monitor_mem --densify_until_iter 15000 --lambda_reg_deform 0.0 --eval --num_sampled_pixels 5000 --num_sampled_masks 25 --smooth_K 16 --contrastive_mode soft ## You can configure --load2gpu_on_the_fly --load_image_on_the_fly --load_mask_on_the_fly for running on smaller GPU or local machine having less RAM.

```

### HyperNeRF

```
python train.py -s data/HyperNeRF/<interp/misc>/<NAME> -m output/HyperNeRF/<NAME> --warm_up 1500 --warm_up_3d_features 15000 --iterative_opt_interval 20000 --iterations 30000 --test_iterations 5000 10000 15000 20000 30000 --save_iterations 20000 30000 --monitor_mem --densify_until_iter 9000 --lambda_reg_deform 0.0 --eval --num_sampled_pixels 5000 --num_sampled_masks 25 --smooth_K 16 --contrastive_mode soft ## You can configure --load2gpu_on_the_fly --load_image_on_the_fly --load_mask_on_the_fly for running on smaller GPU or local machine having less RAM.
```

### Neu3D

```
python train.py    -s data/Neu3D/<NAME> -m output/Neu3D/<NAME> --warm_up 3000 --warm_up_3d_features 15000 --iterative_opt_interval 20000 --iterations 30000 --test_iterations 10000 15000 20000 30000 --save_iterations 10000 15000 20000 30000 --monitor_mem --densify_until_iter 8000 --lambda_reg_deform 0 --eval --load2gpu_on_the_fly --num_sampled_pixels 10000 --num_sampled_masks 50 --smooth_K 16 --contrastive_mode soft --load_mask_on_the_fly --load_image_on_the_fly ## For multiview dataset, it's suggested to load images and anything-masks on-the-fly to reduce RAM usage
```

### Immersive

```
python train.py    -s data/immersive/<NAME> -m output/immersive/<NAME> --warm_up 1000 --warm_up_3d_features 15000 --iterative_opt_interval 20000 --iterations 30000 --test_iterations 5000 10000 15000 20000 30000 --save_iterations 10000 15000 20000 30000 --monitor_mem --densify_until_iter 3000 --lambda_reg_deform 0 --eval --load2gpu_on_the_fly --num_sampled_pixels 10000 --num_sampled_masks 50 --contrastive_mode soft --load_mask_on_the_fly --load_image_on_the_fly --end_frame 50 ## For multiview dataset, it's supported to load images and anything-masks on-the-fly to reduce RAM usage
```

### Technicolor

```
python train.py    -s data/technicolor/Undistorted/<NAME> -m output/technicolor/<NAME> --warm_up 3000 --warm_up_3d_features 15000 --iterative_opt_interval 20000 --iterations 30000 --test_iterations 5000 10000 15000 20000 30000 --save_iterations 10000 15000 20000 30000 --monitor_mem --densify_until_iter 5000 --lambda_reg_deform 0 --eval --load2gpu_on_the_fly --num_sampled_pixels 10000 --num_sampled_masks 50 --contrastive_mode soft --load_mask_on_the_fly --load_image_on_the_fly --mask_black_bg ## For multiview dataset, it's supported to load images and anything-masks on-the-fly to reduce RAM usage
```
