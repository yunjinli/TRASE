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
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#



import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree
import pytorch3d.ops as ops
from torch.nn.functional import mse_loss, cosine_similarity, sigmoid
from torchvision.utils import save_image

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]


    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

def loss_reg_3d_feature(gaussian_feats, gaussian_xyz, k, max_points=-1):
    if gaussian_feats.size(0) > max_points and max_points != -1:
        indices = torch.randperm(gaussian_feats.size(0))[:max_points]
        feats = gaussian_feats[indices]
        xyz = gaussian_xyz[indices]
    else:
        feats = gaussian_feats
        xyz = gaussian_xyz
    
    knn_obj = ops.knn_points(
            xyz.unsqueeze(0),
            xyz.unsqueeze(0),
            K=(k + 1),
        )
    ijs_1 = knn_obj.idx.squeeze(0)[:, 1:]
    N, K = ijs_1.shape
    aux_ids = torch.arange(N).unsqueeze(-1).expand(N, K).reshape(-1)
    eps = 1e-10
    kl = sigmoid(feats[aux_ids]) * (torch.log(sigmoid(feats[aux_ids]) + eps) - torch.log(sigmoid(feats[ijs_1.reshape(-1)]) + eps))
    loss = kl.mean()
    return loss
    
def loss_feature3d(gaussian_feats, gaussian_xyz, kp=16, kn=4, max_points=10000, lambda_p=1.0, lambda_n=1.0):
    if gaussian_feats.size(0) > max_points:
        indices = torch.randperm(gaussian_feats.size(0))[:max_points]
        feats = gaussian_feats[indices]
        xyz = gaussian_xyz[indices]
    else:
        feats = gaussian_feats
        xyz = gaussian_xyz
    N, _ = feats.shape
    
    dists = torch.cdist(xyz, xyz)  # Compute pairwise distances
    # print("dists:", dists.shape)
    _, nn_indices = dists.topk(kp, largest=False)
    _, fn_indices = dists.topk(kn, largest=True)

    aux_ids_n = torch.arange(N).unsqueeze(-1).expand(N, kp).reshape(-1)
    aux_ids_f = torch.arange(N).unsqueeze(-1).expand(N, kn).reshape(-1)

    near_loss = lambda_p * sigmoid(1 - cosine_similarity(feats[aux_ids_n, :], feats[nn_indices.reshape(-1), :], dim=1)).mean()
    far_loss = lambda_n * sigmoid(cosine_similarity(feats[aux_ids_f, :], feats[fn_indices.reshape(-1), :], dim=1)).mean()
    
    return near_loss + far_loss



def loss_rigid_body_motion_reg_loss(xyz1, xyz2, cluster_ids, num_neighbors=128, max_points=-1):
    loss = 0
    valid_cluster_ids = torch.unique(cluster_ids)
    for cls in valid_cluster_ids:
        p1 = xyz1[cluster_ids == cls]
        p2 = xyz2[cluster_ids == cls]
        
        if max_points != -1:
            if p1.size(0) > max_points:
                indices = torch.randperm(p1.size(0))[:max_points]
                p1 = p1[indices]
                p2 = p2[indices]
        ## At t1
        knn_obj = ops.knn_points(
            p1.unsqueeze(0),
            p1.unsqueeze(0),
            K=(num_neighbors + 1),
        )
        ijs_1 = knn_obj.idx.squeeze(0)[:, 1:]
        N, K = ijs_1.shape

        aux_ids = torch.arange(N).unsqueeze(-1).expand(N, K).reshape(-1)
        e_ijs_1 = (p1[aux_ids, :] - p1[ijs_1.reshape(-1)]).reshape(N, K, 3)

        ## At t2
        
        e_ijs_2 = (p2[aux_ids, :] - p2[ijs_1.reshape(-1)]).reshape(N, K, 3)
        
        
        S_i = torch.bmm(e_ijs_1.transpose(1, 2), e_ijs_2) ## uniform edge weighting

        U_i, sig, V_i = torch.svd(S_i)
    
        R_i = torch.bmm(V_i, U_i.transpose(1, 2))
        
        loss_i = (e_ijs_1 - torch.bmm(R_i, e_ijs_2.transpose(1, 2)).transpose(1, 2)).pow(2).sum(2).sum(1).mean()
        
        if torch.isnan(loss_i):
            print(f"RBM loss is NaN in cluster {cls}")
        else:
            loss += loss_i
        
    return loss / valid_cluster_ids.shape[0]

def loss_nnfm_style(feat1, feats2):
    feat1_hats = feat1 / torch.linalg.norm(feat1, dim=0)
    feat2_hats = feats2 / torch.linalg.norm(feats2, dim=0)
    min_dists = torch.amin(1.0 - torch.matmul(feat1_hats.T, feat2_hats), dim=1)
    loss = torch.mean(min_dists)
    return loss

def calc_mean_std(x, eps=1e-8):
        """
        calculating channel-wise instance mean and standard variance
        x: shape of (N,C,*)
        """
        mean = torch.mean(x.flatten(2), dim=-1, keepdim=True) # size of (N, C, 1)
        std = torch.std(x.flatten(2), dim=-1, keepdim=True) + eps # size of (N, C, 1)
        
        return mean, std

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    gram = None
    b, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram
  
def cal_adain_style_loss(x, y):
    """
    style loss in one layer

    Args:
        x, y: feature maps of size [N, C, H, W]
    """
    x_mean, x_std = calc_mean_std(x)
    y_mean, y_std = calc_mean_std(y)

    return torch.nn.functional.mse_loss(x_mean, y_mean) \
         + torch.nn.functional.mse_loss(x_std, y_std)

def cal_style_loss(target, style, weight):
    _, d, h, w = target.shape
    target_gram = gram_matrix(target)
    style_gram = gram_matrix(style)
    layer_style_loss = weight * torch.mean((target_gram - style_gram) ** 2)
    return layer_style_loss / (d * h * w)
    
def cal_mse_content_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)

def pixel_mask_correspondence_loss_positive(C, C_F, positive_th=0.75, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)

    positive_mask = torch.any(C == 1, dim = 0)
    positive_mask = torch.logical_and(positive_mask, ~diag_mask)
    positive_mask = torch.triu(positive_mask, diagonal=0) ## set the symmetric part to false
    number_of_all_pixel_pair = torch.nonzero(positive_mask).shape[0]
    positive_mask = torch.logical_and(positive_mask, C == 1)
    
    positive_mask = positive_mask.bool()
        
    if weights is not None:
        return (-weights[positive_mask]* C_F[positive_mask]).sum() / number_of_all_pixel_pair
    else:
        return (-C_F[positive_mask]).sum() / number_of_all_pixel_pair

def pixel_mask_correspondence_loss_negative(C, C_F, negative_th=0.5, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    negative_mask = torch.any(C == 0, dim = 0)
    negative_mask = torch.logical_and(negative_mask, ~diag_mask)
    negative_mask = torch.triu(negative_mask, diagonal=0) ## set the symmetric part to false
    number_of_all_pixel_pair = torch.nonzero(negative_mask).shape[0]
    negative_mask = torch.logical_and(negative_mask, C == 0)
    negative_mask = negative_mask.bool()
        
    if weights is not None:
        return (weights[negative_mask] * torch.relu(C_F[negative_mask])).sum() / number_of_all_pixel_pair
    else:
        return (torch.relu(C_F[negative_mask])).sum() / number_of_all_pixel_pair

def pixel_mask_correspondence_loss_soft_hard_positive(C, C_F, positive_th=0.75, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    soft_hard_positive_mask = torch.any(torch.logical_and(C_F < positive_th, C == 1), dim = 0)
    soft_hard_positive_mask = torch.logical_and(soft_hard_positive_mask, ~diag_mask)
    soft_hard_positive_mask = torch.triu(soft_hard_positive_mask, diagonal=0) ## set the symmetric part to false
    
    number_of_all_pixel_pair = torch.nonzero(soft_hard_positive_mask).shape[0]
    soft_hard_positive_mask = torch.logical_and(soft_hard_positive_mask, C == 1)
    
    soft_hard_positive_mask = soft_hard_positive_mask.bool()
    
    if soft_hard_positive_mask.sum() == 0: ## No positvie sample found
        print("[WARNING] no positive sample found")
        return 0.0
    else:
        if weights is not None:
            loss = (-weights[soft_hard_positive_mask] * C_F[soft_hard_positive_mask]).sum() / number_of_all_pixel_pair
            
        else:
            loss = (-C_F[soft_hard_positive_mask]).sum() / number_of_all_pixel_pair
            
        return loss

def pixel_mask_correspondence_loss_soft_negative(C, C_F, negative_th=0.5, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    soft_hard_negative_mask = torch.any(torch.logical_and(C_F > negative_th, C == 0), dim = 0)
    
    soft_hard_negative_mask = torch.logical_and(soft_hard_negative_mask, ~diag_mask)
    
    soft_hard_negative_mask = torch.triu(soft_hard_negative_mask, diagonal=0) ## set the symmetric part to false
    
    number_of_all_pixel_pair = torch.nonzero(soft_hard_negative_mask).shape[0]
    
    soft_hard_negative_mask = torch.logical_and(soft_hard_negative_mask, C == 0)
    soft_hard_negative_mask = soft_hard_negative_mask.bool()
        
    if soft_hard_negative_mask.sum() == 0:
        print("[WARNING] no negative sample found")
        return 0.0
    else:
        if weights is not None:
            loss = (weights[soft_hard_negative_mask] * torch.relu(C_F[soft_hard_negative_mask])).sum() / number_of_all_pixel_pair
        else:
            loss = (torch.relu(C_F[soft_hard_negative_mask])).sum() / number_of_all_pixel_pair

        return loss

def pixel_mask_correspondence_loss_hard_positive(C, C_F, positive_th=0.75, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C.shape[0], dtype=bool, device=C_F.device)

    # Find hard positive indices (i, j)
    hard_positive_mask = torch.triu((C_F < positive_th) & (C == 1) & (~diag_mask), diagonal=0)
    hard_positive_indices = torch.nonzero(hard_positive_mask, as_tuple=False)

    if hard_positive_indices.shape[0] == 0:
        print("[WARNING] no hard positive sample found")
        return torch.tensor(0.0, device=C_F.device)

    i, j = hard_positive_indices[:, 0], hard_positive_indices[:, 1]

    # Compute loss
    C_F_hard = C_F[i, j]  # Use indexed values instead of masked tensor

    if weights is not None:
        loss = (-weights[i, j] * C_F_hard).mean()
    else:
        loss = (-C_F_hard).mean()

    return loss

def pixel_mask_correspondence_loss_hard_negative(C, C_F, negative_th=0.5, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C.shape[0], dtype=bool, device=C_F.device)

    # Find hard negative indices (i, j)
    hard_negative_mask = torch.triu((C_F > negative_th) & (C == 0) & (~diag_mask), diagonal=0)
    hard_negative_indices = torch.nonzero(hard_negative_mask, as_tuple=False)
    if hard_negative_indices.shape[0] == 0:
        print("[WARNING] no hard negative sample found")
        return torch.tensor(0.0, device=C_F.device)

    i, j = hard_negative_indices[:, 0], hard_negative_indices[:, 1]

    # Compute loss
    C_F_hard = C_F[i, j]

    if weights is not None:
        loss = (weights[i, j] * torch.relu(C_F_hard)).mean()
    else:
        loss = (torch.relu(C_F_hard)).mean()

    return loss
    
positive_pixel_pair_loss = {
    'hard': pixel_mask_correspondence_loss_hard_positive,
    'all': pixel_mask_correspondence_loss_positive,
    'soft': pixel_mask_correspondence_loss_soft_hard_positive
}

negative_pixel_pair_loss = {
    'hard': pixel_mask_correspondence_loss_hard_negative,
    'all': pixel_mask_correspondence_loss_negative,
    'soft': pixel_mask_correspondence_loss_soft_negative
}