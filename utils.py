import time
from datetime import datetime

import numpy as np
from timm.data import ImageDataset
from torchvision.datasets import CIFAR100
import torch.nn.functional as F

class ImageNetInstanceSample(ImageDataset):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR100InstanceSample(CIFAR100, ImageNetInstanceSample):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR100.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 100
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class TimePredictor:
    def __init__(self, steps, most_recent=30, drop_first=True):
        self.init_time = time.time()
        self.steps = steps
        self.most_recent = most_recent
        self.drop_first = drop_first  # drop iter 0

        self.time_list = []
        self.temp_time = self.init_time

    def update(self):
        time_interval = time.time() - self.temp_time
        self.time_list.append(time_interval)

        if self.drop_first and len(self.time_list) > 1:
            self.time_list = self.time_list[1:]
            self.drop_first = False

        self.time_list = self.time_list[-self.most_recent:]
        self.temp_time = time.time()

    def get_pred_text(self):
        single_step_time = np.mean(self.time_list)
        end_timestamp = self.init_time + single_step_time * self.steps
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')


import torch
import math

def _get_classifier_weight(model):
    """尝试找到 model 里的分类层权重（返回 nn.Parameter 或 Tensor，形状 [num_classes, dim]）"""
    m = model.module if hasattr(model, 'module') else model
    # 常见名字按优先级搜
    if hasattr(m, 'pool_head'):
        return m.pool_head.fc.weight
    cand = ['head', 'classifier', 'fc', 'linear', 'pre_logits', 'head_dist','linear_head','pool_head']
    for name in cand:
        if hasattr(m, name):
            layer = getattr(m, name)
            # 如果是 nn.Linear 或有 weight 属性
            if hasattr(layer, 'weight'):
                return layer.weight
            # 如果是 Sequential，找最后一个有 weight 的子模块
            if isinstance(layer, torch.nn.Sequential):
                for sub in reversed(layer):
                    if hasattr(sub, 'weight'):
                        return sub.weight
    raise RuntimeError("找不到分类层权重，请自行指定或检查模型结构。")

def _vit_patches_from_featmap(feat_map):
    """输入 feat_map 可能是 [B, N+1, D]（含 CLS）或 [B, N, D]（仅 patches）。
       返回 patch_tokens [B, N_patches, D] 和 (H, W)"""
    b, n, d = feat_map.shape
    # 判断是否含 CLS token：如果 n-1 是完全平方数则可能有 CLS
    if int(math.isqrt(n - 1)) ** 2 == (n - 1):
        patch_tokens = feat_map[:, 1:, :]  # 去掉 CLS
        num_patches = n - 1
    else:
        # 否则假设没有 CLS
        patch_tokens = feat_map
        num_patches = n
    h = w = int(math.isqrt(num_patches))
    if h * w != num_patches:
        raise ValueError(f"无法把 {num_patches} 个 patch 恢复成方阵 (H*W)。")
    return patch_tokens, (h, w)

def get_CAM(model, feat_map, arch=None, dataset=None, target_idx=None):
    """
    通用 get_CAM，支持 CNN / ViT / DINOv2。
    - model: 包含分类层权重的模型实例。
    - feat_map:
        * CNN: [B, C, H, W]
        * ViT/DINOv2: [B, N(+1), D] 或 [B, N, D]
    - target_idx: None -> 所有类别；或 shape (B,) -> 每个样本一个类；或全局类列表
    返回：CAM 张量 [B, num_classes 或 1 或 K, 224, 224]
    """
    m = model.module if hasattr(model, 'module') else model
    weight = _get_classifier_weight(m)  # [num_classes, dim]

    if feat_map.dim() == 4:
        # CNN 分支
        B, C, H, W = feat_map.shape
        x = feat_map.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        feat = x @ weight.t()  # (B*H*W, num_classes)
        cam_all = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, num_classes, H, W]

        if target_idx is None:
            cams = cam_all
        else:
            target = torch.as_tensor(target_idx, device=cam_all.device)
            if target.dim() == 1 and target.numel() == B:
                cams = cam_all[torch.arange(B), target]  # [B, H, W]
            else:
                cams = cam_all[:, target, :, :]  # [B, K, H, W]

    elif feat_map.dim() == 3:
        # ViT / DINOv2 分支
        patch_tokens, (h, w) = _vit_patches_from_featmap(feat_map)  # [B, N_patches, D]
        B, N, D = patch_tokens.shape

        if arch and "dinov2" in arch.lower():
            # DINOv2 分类层输入是 [cls_token, mean_patch_tokens]
            # weight 形状: [num_classes, 2*D]           
            if weight.shape[1] != 2 * D:
                raise RuntimeError(f"DINOv2 分类层权重维度({weight.shape[1]})应为 2*{D}。")
            # 只取 patch 部分权重
            patch_weight = weight[:, D:]  # [num_classes, D]
            W_used = patch_weight
        else:
            # 普通 ViT
            if weight.shape[1] != D:
                raise RuntimeError(f"分类权重维度({weight.shape[1]})与 token dim({D})不匹配。")
            W_used = weight

        if target_idx is None:
            # 所有类
            cams = torch.einsum('bnd,cd->bnc', patch_tokens, W_used)  # [B, N, C]
            cams = cams.permute(0, 2, 1).reshape(B, -1, h, w)  # [B, C, H, W]
        else:
            target = torch.as_tensor(target_idx, device=patch_tokens.device)
            if target.dim() == 1 and target.numel() == B:
                cw = W_used[target]           # [B, D]
                cams = torch.einsum('bnd,bd->bn', patch_tokens, cw)  # [B, N]
                cams = cams.reshape(B, 1, h, w)
            else:
                cw = W_used[target]  # [K, D]
                cams = torch.einsum('bnd,kd->bnk', patch_tokens, cw)  # [B, N, K]
                cams = cams.permute(0, 2, 1).reshape(B, -1, h, w)

    else:
        raise ValueError("feat_map 维度不支持（期望 3D 或 4D）")

    # # 统一插值到 (224, 224)
    # cams = F.interpolate(cams, size=(224, 224), mode="bilinear", align_corners=False)
    return cams

def CAT_loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, rho=0.8):
    """
    计算基于随机掩码对齐的 CAM 损失。
    
    Args:
        rho (float): 掩码比例，默认为 0.8。表示有 80% 的元素会被用于计算损失。
    """
    # 1. 调整分辨率 (保持原有逻辑)
    CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))
    
    # 2. 归一化 (保持原有逻辑)
    # 注意：通常建议在 mask 之前做归一化，保证特征分布的完整性
    S_norm = F.normalize(CAM_Student)
    T_norm = F.normalize(CAM_Teacher)
    
    # 3. 应用随机掩码 (新增逻辑)
    if rho < 1.0 and rho > 0.0:
        # 生成与特征图形状相同的随机矩阵，值在 [0, 1) 之间
        # 如果随机值小于 rho，则 mask 为 True (保留)，否则为 False (丢弃)
        mask = torch.rand_like(S_norm) < rho
        
        # 使用掩码筛选元素。
        # 注意：S_norm[mask] 会将张量展平 (flatten) 为一维向量，只包含 mask 为 True 的元素
        loss = F.mse_loss(S_norm[mask], T_norm[mask])
    else:
        # 如果 rho >= 1.0，则进行全量对齐 (原有逻辑)
        loss = F.mse_loss(S_norm, T_norm)
        
    return loss


# def get_CAM(model, feat_map, arch=None, dataset=None, target_idx=None):
#     """
#     通用 get_CAM。
#     - model: 包含分类层权重的模型实例（可能是 teacher/student）。
#     - feat_map: 对于 CNN 是 [B, C, H, W]；对于 ViT 是最后一层 tokens [B, N(+1), D] 或 [B, N, D]。
#     - target_idx: None -> 返回所有类别 [B, num_classes, H, W]
#                   若为 shape (B,) 的 tensor/list -> 每个样本指定一个类别，返回 [B, 1, H, W]
#                   若为 list/1D tensor（长度 K）(公用给所有样本) -> 返回 [B, K, H, W]
#     返回：CAM 张量（float），layout 与原 CNN 实现一致。
#     """
#     m = model.module if hasattr(model, 'module') else model
#     weight = _get_classifier_weight(m)  # [num_classes, dim]

#     if feat_map.dim() == 4:
#         # CNN 分支: feat_map [B, C, H, W]
#         B, C, H, W = feat_map.shape
#         x = feat_map.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
#         feat = x @ weight.t()  # (B*H*W, num_classes)
#         cam_all = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, num_classes, H, W]

#         if target_idx is None:
#             cams = cam_all  # [B, num_classes, H, W]
#         else:
#             # 下面处理 target_idx 的几种情形
#             target = torch.as_tensor(target_idx, device=cam_all.device)
#             if target.dim() == 1 and target.numel() == B:
#                 # 每个样本一个类 -> 取出对应通道，返回 [B,1,H,W]
#                 cams = cam_all[torch.arange(B), target]  # [B, H, W]                
#             else:
#                 # target 为全局类索引列表，例如 [5, 10, 12] -> 返回 [B, K, H, W]
#                 cams = cam_all[:, target, :, :]  # [B, K, H, W]
            

#     elif feat_map.dim() == 3:
#         # ViT 分支: feat_map [B, N(+1), D] 或 [B, N, D]
#         patch_tokens, (h, w) = _vit_patches_from_featmap(feat_map)  # [B, N_patches, D]
#         B, N, D = patch_tokens.shape
#         # weight: [num_classes, D]  (D must match)
#         if weight.shape[1] != D:
#             raise RuntimeError(f"分类权重维度({weight.shape[1]})与 token dim({D})不匹配。")
#         if target_idx is None:
#             # 所有类：先得 [B, N, C] -> 再 reshape成 [B, C, H, W]
#             cams = torch.einsum('bnd,cd->bnc', patch_tokens, weight)  # [B, N, C]
#             cams = cams.permute(0, 2, 1).reshape(B, -1, h, w)  # [B, C, H, W]
#         else:
#             target = torch.as_tensor(target_idx, device=patch_tokens.device)
#             if target.dim() == 1 and target.numel() == B:
#                 # 每个样本一个类 -> 取每个样本对应的 class weight
#                 cw = weight[target]           # [B, D]
#                 cams = torch.einsum('bnd,bd->bn', patch_tokens, cw)  # [B, N]
#                 cams = cams.reshape(B, 1, h, w)  # [B, 1, H, W]
#             else:
#                 # 全局 K 类
#                 cw = weight[target]  # [K, D]
#                 cams = torch.einsum('bnd,kd->bnk', patch_tokens, cw)  # [B, N, K]
#                 cams = cams.permute(0, 2, 1).reshape(B, -1, h, w)  # [B, K, H, W]                
#     else:
#         raise ValueError("feat_map 维度不支持（期望 3D 或 4D）")
#     # 统一插值到 (224, 224)
#     cams = F.interpolate(cams, size=(224, 224), mode="bilinear", align_corners=False)
#     return cams  # [B, C或K或1, 224, 224]
