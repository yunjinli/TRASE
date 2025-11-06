#
# Copyright (C) 2024, TRASE
# Technical University of Munich CVG
# All rights reserved.
#
# TRASE is heavily based on other research. Consider citing their works as well.
# 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
# Deformable-3D-Gaussians: https://github.com/ingra14m/Deformable-3D-Gaussians
# gaussian-grouping: https://github.com/lkeab/gaussian-grouping
# SAGA: https://github.com/Jumpat/SegAnyGAussians
# SC-GS: https://github.com/yihua7/SC-GS
# 4d-gaussian-splatting: https://github.com/fudan-zvg/4d-gaussian-splatting
# ------------------------------------------------------------------------
# Modified from codes in Deformable-3D-Gaussians: https://github.com/ingra14m/Deformable-3D-Gaussians


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformModelType
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, model_type='DeformNetwork'):
        self.deform = DeformModelType[model_type](is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.model_type = model_type

        self.spatial_lr_scale = 5
            
    def step(self, xyz, time_emb, f=None):
        return self.deform(xyz, time_emb, f) if self.model_type == 'DeformSemanticNetwork' else self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, name=None):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        if name:
            torch.save(self.deform.state_dict(), os.path.join(out_weights_path, f'{name}.pth'))
            
        else:
            torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1, name=None):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        if name:
            weights_path = os.path.join(model_path, "deform/iteration_{}/{}.pth".format(loaded_iter, name))
        else:
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def change_optimization_target(self, opt_state='GAUSSIAN'):
        if opt_state == 'GAUSSIAN':
            for param_group in self.optimizer.param_groups:
                param_group['params'][0].requires_grad = True
                param_group['params'][0].grad = torch.zeros_like(param_group['params'][0])
        else:
            for param_group in self.optimizer.param_groups:
                param_group['params'][0].requires_grad = False
                param_group['params'][0].grad = None