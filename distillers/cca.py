import torch
import numpy as np

from sklearn.cross_decomposition import CCA
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseDistiller
import math
from .registry import register_distiller

import math
import torch
import torch.nn.functional as F
from utils import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA as CCAH
from sklearn.decomposition import PCA
import os

# def cca_heatmap(feature_teacher, feature_student, save_path="/nfs4/wangyb/projects/OFAKD/cca_heatmap.png", n_components=64):
#     """
#     feature_teacher: torch.Tensor [B, D, H, W]
#     feature_student: torch.Tensor [B, D, H, W]
#     """
#     B, D, H, W = feature_teacher.shape
#     heatmap = np.zeros((H, W))

#     # 遍历空间位置
#     for i in range(H):
#         for j in range(W):
#             ft = feature_teacher[:, :, i, j].detach().cpu().numpy()  # [B, D]
#             fs = feature_student[:, :, i, j].detach().cpu().numpy()  # [B, D]

#             # PCA 降维，避免 rank 不足
#             n_comp = min(n_components, B // 2, D)
#             pca_t = PCA(n_components=n_comp)
#             pca_s = PCA(n_components=n_comp)
#             ft_pca = pca_t.fit_transform(ft)
#             fs_pca = pca_s.fit_transform(fs)

#             # CCA
#             cca = CCAH(n_components=1)
#             ft_c, fs_c = cca.fit_transform(ft_pca, fs_pca)

#             # 取第一主相关系数
#             corr = np.corrcoef(ft_c.T, fs_c.T)[0, 1]
#             heatmap[i, j] = corr

#     # 绘制热力图
#     plt.figure(figsize=(6, 5))
#     plt.imshow(heatmap, cmap="coolwarm", vmin=-1, vmax=1)
#     plt.colorbar(label="CCA correlation")
#     plt.title("Teacher vs Student CCA Heatmap")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"CCA heatmap saved to {save_path}")

#     return heatmap

# def cca_heatmap(feature_teacher: torch.Tensor,
#                  feature_student: torch.Tensor,
#                  save_dir: str = "./cca_results",
#                  n_components: int = 50,
#                  batch_idx: int = 0,
#                  save_path="/nfs4/wangyb/projects/OFAKD/",
#                  save_npy: bool = True,
#                  save_img: bool = True):
#     """
#     计算教师与学生特征的 CCA 差异化分布图 (H×W)

#     Args:
#         feature_teacher: [B, C, H, W] 教师特征
#         feature_student: [B, C, H, W] 学生特征
#         save_dir: 保存路径
#         n_components: CCA 分量数量，默认取通道最小值
#         batch_idx: 可视化和保存时选取的 batch 索引
#         save_npy: 是否保存为 .npy 文件
#         save_img: 是否保存为 .png 图片
#     Returns:
#         corr_map: [B, H, W] 的相关性分布
#     """


#     B, C, H, W = feature_teacher.shape

#     # 如果学生特征 H W 不一致，需要上采样
#     if feature_student.shape[2:] != (H, W):
#         feature_student = F.interpolate(feature_student, size=(H, W), mode='bilinear', align_corners=False)

#     # 展平 [B, C, H, W] -> [B*H*W, C]
#     teacher_flat = feature_teacher.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()
#     student_flat = feature_student.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()

#     # 标准化
#     teacher_flat = (teacher_flat - teacher_flat.mean(axis=0, keepdims=True)) / (teacher_flat.std(axis=0, keepdims=True) + 1e-6)
#     student_flat = (student_flat - student_flat.mean(axis=0, keepdims=True)) / (student_flat.std(axis=0, keepdims=True) + 1e-6)

#     # CCA
#     if n_components is None:
#         n_components = min(teacher_flat.shape[1], student_flat.shape[1])
#     cca = CCAH(n_components=n_components)
#     cca.fit(teacher_flat, student_flat)

#     teacher_c, student_c = cca.transform(teacher_flat, student_flat)  # [B*H*W, n_components]

#     # 每个位置计算相关系数
#     corr_map_flat = np.array([np.corrcoef(teacher_c[i], student_c[i])[0, 1] for i in range(teacher_c.shape[0])])

#     # reshape 回 [B, H, W]
#     corr_map = corr_map_flat.reshape(B, H, W)

#     # 保存 .npy 文件
#     if save_npy:
#         np.save(os.path.join(save_path,"cca_corr_map.npy"), corr_map)

#     # 保存可视化图片
#     if save_img:
#         plt.figure(figsize=(6,6))
#         plt.imshow(corr_map[batch_idx], cmap='coolwarm', vmin=-1, vmax=1)
#         plt.colorbar(label="CCA correlation")
#         plt.title(f"CCA similarity map (batch {batch_idx})")
#         plt.savefig(os.path.join(save_path, f"cca_map_batch{batch_idx}.png"), dpi=300, bbox_inches="tight")
#         plt.close()

#     return corr_map


# def cca_heatmap(feature_teacher: torch.Tensor,
#                 feature_student: torch.Tensor,
#                 save_dir: str = "./cca_results",
#                 n_components: int = 50,
#                 batch_idx: int = 0,
#                 save_path: str = "./",
#                 save_npy: bool = True,
#                 save_img: bool = True):
#     """
#     计算教师与学生特征的 CCA 差异化分布图 (H×W)，GPU 版
#     （仅修复变量名冲突与 reshape 问题，其他逻辑不变）
#     """

#     # <-- 确保 batch_size/H/W 都是 Python int，避免后面传入 reshape 时出错 -->
#     batch_size, C, H, W = map(int, feature_teacher.shape)
#     device = feature_teacher.device

#     # 如果学生特征 H W 不一致，需要上采样
#     if feature_student.shape[2:] != (H, W):
#         feature_student = F.interpolate(feature_student, size=(H, W), mode='bilinear', align_corners=False)

#     # 展平 [B, C, H, W] -> [N, C]
#     teacher_flat = feature_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
#     student_flat = feature_student.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]

#     # 标准化 (零均值 + 单位方差)
#     teacher_flat = (teacher_flat - teacher_flat.mean(dim=0, keepdim=True)) / (teacher_flat.std(dim=0, keepdim=True) + 1e-6)
#     student_flat = (student_flat - student_flat.mean(dim=0, keepdim=True)) / (student_flat.std(dim=0, keepdim=True) + 1e-6)

#     # 协方差矩阵
#     N = teacher_flat.shape[0]
#     C_tt = teacher_flat.T @ teacher_flat / (N - 1)   # [C, C]
#     C_ss = student_flat.T @ student_flat / (N - 1)   # [C, C]
#     C_ts = teacher_flat.T @ student_flat / (N - 1)   # [C, C]

#     # 稳定性: 加小的正则项
#     eps = 1e-6
#     C_tt += eps * torch.eye(C, device=device)
#     C_ss += eps * torch.eye(C, device=device)

#     # whiten
#     U_t, S_t, Vt_t = torch.linalg.svd(C_tt)
#     U_s, S_s, Vt_s = torch.linalg.svd(C_ss)
#     C_tt_inv_sqrt = (U_t @ torch.diag(1.0 / torch.sqrt(S_t)) @ Vt_t)
#     C_ss_inv_sqrt = (U_s @ torch.diag(1.0 / torch.sqrt(S_s)) @ Vt_s)

#     # 相关矩阵
#     T = C_tt_inv_sqrt @ C_ts @ C_ss_inv_sqrt
#     U, S, Vt = torch.linalg.svd(T)

#     # 限制 n_components
#     n_components = min(n_components, S.shape[0])

#     # 投影矩阵（改名，避免覆盖 batch_size）
#     A_proj = C_tt_inv_sqrt @ U[:, :n_components]
#     B_proj = C_ss_inv_sqrt @ Vt.T[:, :n_components]

#     # 投影到 CCA 空间
#     teacher_c = teacher_flat @ A_proj   # [N, n_components]
#     student_c = student_flat @ B_proj   # [N, n_components]

#     # 每个位置计算相似度 (余弦相似度)
#     sim = torch.nn.functional.cosine_similarity(teacher_c, student_c, dim=1)  # [N]

#     # 检查元素数目是否匹配 batch_size*H*W
#     expected_N = batch_size * H * W
#     if sim.numel() != expected_N:
#         raise RuntimeError(f"sim.numel() ({sim.numel()}) != batch_size*H*W ({expected_N}). "
#                            "请检查 teacher_flat/ student_flat reshape 是否正确。")

#     # reshape 回 [batch_size, H, W]（这里使用 reshape，输入都是 Python int）
#     corr_map = sim.reshape(batch_size, H, W)

#     # 保存 .npy 文件
#     if save_npy:
#         np.save(os.path.join(save_path, "cca_corr_map.npy"), corr_map.detach().cpu().numpy())

#     # 保存可视化图片
#     if save_img:
#         plt.figure(figsize=(6, 6))
#         plt.imshow(corr_map[batch_idx].detach().cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
#         plt.colorbar(label="CCA correlation")
#         plt.title(f"CCA similarity map (batch {batch_idx})")
#         plt.savefig(os.path.join(save_path, f"cca_map_batch{batch_idx}.png"), dpi=300, bbox_inches="tight")
#         plt.close()

#     return corr_map
def cca_heatmap(feature_teacher: torch.Tensor,
                feature_student: torch.Tensor,
                save_dir: str = "./cca_results",
                n_components: int = 50,
                batch_idx: int = 0,
                save_path: str = "./",
                save_npy: bool = True,
                save_img: bool = True):
    """
    计算教师与学生特征的 CCA 差异化分布图 (H×W)，基于 sklearn
    """

    B, C, H, W = feature_teacher.shape

    # 如果学生特征 H W 不一致，需要上采样
    if feature_student.shape[2:] != (H, W):
        feature_student = F.interpolate(feature_student, size=(H, W), mode='bilinear', align_corners=False)

    # 展平 [B, C, H, W] -> [N, C]
    teacher_flat = feature_teacher.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()
    student_flat = feature_student.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()

    # CCA
    if n_components is None:
        n_components = min(teacher_flat.shape[1], student_flat.shape[1])
    cca = CCAH(n_components=min(n_components, min(C, teacher_flat.shape[1], student_flat.shape[1])))
    cca.fit(teacher_flat, student_flat)

    teacher_c, student_c = cca.transform(teacher_flat, student_flat)  # [N, n_components]

    # 每个位置计算相关系数
    corr_map_flat = np.array([np.corrcoef(teacher_c[i], student_c[i])[0, 1] for i in range(teacher_c.shape[0])])

    # reshape 回 [B, H, W] —— 关键修改点
    corr_map = corr_map_flat.reshape((B, H, W))

    # 保存 .npy 文件
    if save_npy:
        np.save(os.path.join(save_path, "cca_corr_map.npy"), corr_map)

    # 保存可视化图片
    if save_img:
        plt.figure(figsize=(6, 6))
        plt.imshow(corr_map[batch_idx], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="CCA correlation")
        plt.title(f"CCA similarity map (batch {batch_idx})")
        plt.savefig(os.path.join(save_path, f"cca_map_batch{batch_idx}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return corr_map
# def cca_heatmap(feature_teacher: torch.Tensor,
#                  feature_student: torch.Tensor,
#                  save_path: str = "./cca_results",
#                  n_components: int = 50,
#                  batch_idx: int = 0,
#                  save_npy: bool = True,
#                  save_img: bool = True):
#     """
#     计算教师与学生特征的 CCA 差异化分布图 (H×W)
#     Args:
#         feature_teacher: [B, C, H, W] 教师特征
#         feature_student: [B, C, H, W] 学生特征
#         save_path: 保存路径
#         n_components: CCA 分量数量，默认取通道最小值
#         batch_idx: 可视化和保存时选取的 batch 索引
#         save_npy: 是否保存为 .npy 文件
#         save_img: 是否保存为 .png 图片
#     Returns:
#         corr_map: [B, H, W] 的相关性分布
#     """
#     os.makedirs(save_path, exist_ok=True)

#     B, C, H, W = feature_teacher.shape

#     # 上采样学生特征，保证 H W 一致
#     if feature_student.shape[2:] != (H, W):
#         feature_student = F.interpolate(feature_student, size=(H, W),
#                                         mode='bilinear', align_corners=False)

#     # 展平到 [B*H*W, C]，转到 CPU (sklearn 不支持 GPU)
#     teacher_flat = feature_teacher.permute(0, 2, 3, 1).reshape(-1, C).detach()
#     student_flat = feature_student.permute(0, 2, 3, 1).reshape(-1, C).detach()

#     # 归一化（在 GPU 上做）
#     teacher_flat = (teacher_flat - teacher_flat.mean(0, keepdim=True)) / (teacher_flat.std(0, keepdim=True) + 1e-6)
#     student_flat = (student_flat - student_flat.mean(0, keepdim=True)) / (student_flat.std(0, keepdim=True) + 1e-6)

#     # 转 numpy 到 CPU 交给 sklearn
#     teacher_np = teacher_flat.cpu().numpy()
#     student_np = student_flat.cpu().numpy()

#     # 自动确定 n_components
#     if n_components is None:
#         n_components = min(teacher_np.shape[1], student_np.shape[1], teacher_np.shape[0])

#     cca = CCAH(n_components=n_components)
#     cca.fit(teacher_np, student_np)

#     teacher_c, student_c = cca.transform(teacher_np, student_np)  # [B*H*W, n_components]

#     # 每个位置计算相关系数
#     corr_map_flat = np.array([
#         np.corrcoef(teacher_c[i], student_c[i])[0, 1]
#         for i in range(teacher_c.shape[0])
#     ])

#     # reshape 回 [B, H, W]
#     corr_map = torch.tensor(corr_map_flat, dtype=torch.float32).view(B, H, W)

#     # 保存 .npy 文件
#     if save_npy:
#         np.save(os.path.join(save_path, "cca_corr_map.npy"), corr_map.cpu().numpy())

#     # 保存可视化图片
#     if save_img:
#         plt.figure(figsize=(6, 6))
#         plt.imshow(corr_map[batch_idx].cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
#         plt.colorbar(label="CCA correlation")
#         plt.title(f"CCA similarity map (batch {batch_idx})")
#         plt.savefig(os.path.join(save_path, f"cca_map_batch{batch_idx}.png"), dpi=300, bbox_inches="tight")
#         plt.close()

#     return corr_map
eps = 1e-12

def cosine_map_sample0(feature_teacher: torch.Tensor, feature_student: torch.Tensor, sample_idx=0):
    """
    只计算 batch 的第 sample_idx 个样本的逐位置余弦相似度。
    输入：feature_teacher / feature_student: [B, C, H, W]
    返回：cos_map: [H, W] （cpu numpy）
    """
    B, C, H, W = feature_teacher.shape
    assert feature_student.shape == feature_teacher.shape
    assert 0 <= sample_idx < B

    ft = feature_teacher[sample_idx]       # [C, H, W]
    fs = feature_student[sample_idx]       # [C, H, W]
    # 转成 (H*W, C)
    ft_vecs = ft.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    fs_vecs = fs.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

    # 逐行计算余弦
    cos = F.cosine_similarity(ft_vecs, fs_vecs, dim=1)  # [H*W]
    cos_map = cos.view(H, W).detach().cpu().numpy()
    return cos_map

def linear_cka_matrix(X: torch.Tensor, Y: torch.Tensor):
    """
    稳定的 linear CKA（单个位置的实现）
    X, Y: [N, D]  (N = 样本数, D = 特征维度)
    返回 scalar (0~1)
    参考等价公式: numerator = || X^T Y ||_F^2, denom = || X^T X ||_F * || Y^T Y ||_F
    要求 N >= 2
    """
    # center by column (feature)
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)

    # compute cross-covariance (D x D')
    XtY = Xc.t().mm(Yc)       # [D, D]
    num = torch.norm(XtY, p='fro') ** 2

    XtX = Xc.t().mm(Xc)
    YtY = Yc.t().mm(Yc)
    denom = (torch.norm(XtX, p='fro') * torch.norm(YtY, p='fro')) + eps

    cka = num / denom
    # clamp to [0,1] 数值稳定
    return torch.clamp(cka, 0.0, 1.0)

def cka_map_per_position(feature_teacher: torch.Tensor, feature_student: torch.Tensor):
    """
    在每个空间位置 (h,w) 计算 CKA（在 batch 维度上把每个样本作为一个观测）。
    输入: [B, C, H, W]
    返回: cka_map: [H, W] (torch tensor, 在输入 device 上)
    注意: 需要 B >= 2，否则 CK A无法计算（会返回全 nan）
    """
    B, C, H, W = feature_teacher.shape
    assert feature_student.shape == feature_teacher.shape

    device = feature_teacher.device
    cka_map = torch.full((H, W), float('nan'), device=device, dtype=feature_teacher.dtype)

    if B < 2:
        # 无法在 batch 维度上计算 CKA
        return cka_map

    # 遍历每个位置（H*W 小，通常 14*14 = 196，可接受）
    for h in range(H):
        for w in range(W):
            X = feature_teacher[:, :, h, w]  # [B, C]
            Y = feature_student[:, :, h, w]  # [B, C]
            # 注意：如果某个位置上所有样本都完全相同（退化），分母可能非常小，eps 已处理
            cka_val = linear_cka_matrix(X, Y)
            cka_map[h, w] = cka_val
    return cka_map

def save_maps(feature_teacher, feature_student, save_path="/nfs4/wangyb/projects/OFAKD/"):
    os.makedirs(save_path, exist_ok=True)

    # 1) cosine map for batch[0]
    cos_map = cosine_map_sample0(feature_teacher, feature_student, sample_idx=0)
    plt.figure(figsize=(5,5))
    im = plt.imshow(cos_map, cmap="bwr", vmin=-1, vmax=1, origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Cosine similarity per position (sample 0)")
    plt.tight_layout()
    fname1 = os.path.join(save_path, "cosine_map_sample0.png")
    plt.savefig(fname1, dpi=200)
    plt.close()

    # 2) CKA map per position (across batch)
    B = feature_teacher.shape[0]
    if B < 2:
        print(f"⚠️ Batch size {B} < 2，无法按 batch 维度计算位置级 CKA。跳过 CKA 热力图保存。")
        return {"cosine_map": fname1, "cka_map": None}

    cka_map = cka_map_per_position(feature_teacher, feature_student).detach().cpu().numpy()
    plt.figure(figsize=(5,5))
    im2 = plt.imshow(cka_map, cmap="plasma", vmin=0, vmax=1, origin='lower')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.title("CKA per position (across batch)")
    plt.tight_layout()
    fname2 = os.path.join(save_path, "cka_map_per_position.png")
    plt.savefig(fname2, dpi=200)
    plt.close()

    print(f"✅ 已保存：\n - 注意力（余弦）热力图 (sample0): {fname1}\n - 位置级 CKA 热力图: {fname2}")
    return {"cosine_map": fname1, "cka_map": fname2}
# def cosine_similarity_per_position(f_teacher: torch.Tensor, f_student: torch.Tensor) -> torch.Tensor:
#     """
#     计算每个空间位置 (h,w) 的余弦相似度
#     输入: f_teacher, f_student: [B, C, H, W]
#     输出: [B, H, W] 每个样本对应 H×W 的相似度图
#     """
#     B, C, H, W = f_teacher.shape
#     f_t = f_teacher.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
#     f_s = f_student.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

#     sim = F.cosine_similarity(f_t, f_s, dim=-1)  # [B*H*W]
#     sim_map = sim.view(B, H, W)
#     return sim_map

# def linear_CKA_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#     """
#     Linear CKA for matrices
#     X, Y: [N, D]  (N = 样本数, D = 特征维度)
#     return: scalar
#     """
#     X = X - X.mean(0, keepdim=True)
#     Y = Y - Y.mean(0, keepdim=True)

#     K = X @ X.T
#     L = Y @ Y.T
#     HSIC = (K * L).sum()

#     norm_x = torch.sqrt((K * K).sum())
#     norm_y = torch.sqrt((L * L).sum())
#     return HSIC / (norm_x * norm_y + 1e-12)


# def cka_per_position(f_teacher: torch.Tensor, f_student: torch.Tensor) -> torch.Tensor:
#     """
#     在每个空间位置 (h, w) 计算 CKA
#     输入: f_teacher, f_student: [B, C, H, W]
#     输出: [H, W] 的 CKA 热力图
#     """
#     B, C, H, W = f_teacher.shape
#     cka_map = torch.zeros(H, W, device=f_teacher.device)

#     for h in range(H):
#         for w in range(W):
#             X = f_teacher[:, :, h, w]  # [B, C]
#             Y = f_student[:, :, h, w]  # [B, C]
#             cka_map[h, w] = linear_CKA_matrix(X, Y)
#     return cka_map


# def save_similarity_visuals(feature_teacher, feature_student, save_path):
#     os.makedirs(save_path, exist_ok=True)

#     # 1. 每个空间位置的余弦相似度 (batch=0 展示)
#     from torch.nn import functional as F
#     f_t = feature_teacher.permute(0, 2, 3, 1).reshape(-1, feature_teacher.shape[1])
#     f_s = feature_student.permute(0, 2, 3, 1).reshape(-1, feature_student.shape[1])
#     cos_sim = F.cosine_similarity(f_t, f_s, dim=-1).view(feature_teacher.shape[0], feature_teacher.shape[2], feature_teacher.shape[3])
#     sim_map = cos_sim[0].detach().cpu().numpy()

#     plt.figure(figsize=(5,5))
#     plt.imshow(sim_map, cmap="viridis", vmin=-1, vmax=1)
#     plt.colorbar()
#     plt.title("Cosine Similarity per Position (Sample 0)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "cosine_similarity_map.png"))
#     plt.close()

#     # 2. 每个位置的 CKA
#     cka_map = cka_per_position(feature_teacher, feature_student).detach().cpu().numpy()
#     plt.figure(figsize=(5,5))
#     plt.imshow(cka_map, cmap="plasma", vmin=0, vmax=1)
#     plt.colorbar()
#     plt.title("CKA per Position (across batch)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, "feature_CKA_map.png"))
#     plt.close()

#     print(f"✅ 已保存到 {save_path}: cosine_similarity_map.png, feature_CKA_map.png")

# CUDA_VISIBLE_DEVICES=2 python /nfs4/wangyb/projects/OFAKD/train.py /nfs4/wangyb/projects/OFAKD/data/tiny-imagenet-200 --config /nfs4/wangyb/projects/OFAKD/configs/tiny_imagenet/dot/vca.yaml --model resnet18_vcaV2 --teacher dinov2s_finetuehead --teacher-pretrained /nfs4/wangyb/projects/OFAKD/teacher_checkpoint/dinov2_tiny_imagenet_head_best.pth -d tiny-imagenet-200 --num-classes 200 --distiller vca
@register_distiller
class CCA(BaseDistiller):
    requires_feat = False
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(CCA, self).__init__(student, teacher, criterion, args)
        self.args = args
        self.teacher_type = args.teacher  # 'dinov2s' / 'dinov2b' / 'vit'...
        # loss weights
        self.gt_loss_weight = args.gt_loss_weight
        self.feat_loss_weight = args.feat_loss_weight
        self.logits_loss_weight = args.kd_loss_weight        
        
        self.temperature = args.kd_temperature
        

        self.ce_loss = nn.CrossEntropyLoss()
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

        # freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Try to infer student's feat3 channels via student.stage_info(3)
        student_module = self._get_module(self.student)


    def _get_module(self, model):
        return model.module if  hasattr(model, "module") else model

    def _reshape_teacher_tokens_to_map(self, teacher_tokens):
        """
        teacher_tokens: [B, N, C] => [B, C, s, s] where s = sqrt(N)
        """
        if teacher_tokens is None:
            return None
        B, N, C = teacher_tokens.shape
        s = int(math.sqrt(N))
        if s * s != N:
            raise RuntimeError(f"Teacher tokens N ({N}) is not a perfect square. Can't reshape to map.")
        teacher_map = teacher_tokens.permute(0, 2, 1).contiguous().view(B, C, s, s)
        return teacher_map


    def forward(self, image, target, epoch=1, **kwargs):
        """
        main training step:
        - 获取学生 feat3（layer3 输出）
        - 获取教师 patch tokens -> reshape -> resize 到 feat3 spatial
        - adapter(student_feat3) -> adapter_feat
        - compute feature loss between adapter_feat and teacher_map_resized (flatten->tokens)
        - fusion_input = cat(student_feat3, adapter_feat), fusion_conv -> fusion_input_reduced
        - call student.layer4(fusion_input_reduced) -> feat4
        - call student's head/fc to get logits_student
        - compute CE / logits KD if teacher logits exist
        """
        student_module = self._get_module(self.student)
        teacher_module = self._get_module(self.teacher)

        # ------------------ 获取 student feat3 ------------------
        # Try a few ways robustly
        student_feat3 = None
        feature_student = None
        logits_student_orig = None

        # ---------------- Student forward ----------------
        # 期待 student(image) 返回 (logits, features_dict)
        res = student_module(image)
        if not (isinstance(res, tuple) and len(res) == 2):
            raise RuntimeError("student(image) must return (logits, features_dict). features_dict should contain 'distill_feat'.")
        logits_student, feature_student = res
        if "adapter_feat" not in feature_student:
            raise RuntimeError("feature_student must contain 'adapter_feat' (student-side adapter feature).")

        student_distill_feat = feature_student["adapter_feat"]  # expected [B, C_s, H_s, W_s] e.g. [B,384,16,16]
        if student_distill_feat is None or not isinstance(student_distill_feat, torch.Tensor):
            raise RuntimeError("feature_student['adapter_feat'] must be a torch.Tensor with shape [B,C,H,W].")

        B, C_s, H_s, W_s = student_distill_feat.shape
        # student tokens: [B, N_s, C_s] where N_s = H_s * W_s
        student_tokens = student_distill_feat.flatten(2).transpose(1, 2)  # [B, N_s, C_s]


        # ------------------ 获取 teacher tokens & build teacher_map ------------------
        with torch.no_grad():
            # try teacher.forward_features (DINOv2 style) first
            teacher_tokens = None
            teacher_logits = None
            if self.teacher_type == 'dinov2s_finetuehead':
                teacher_logits,teacher_feat = self.teacher(image,return_feat = True) 
                # teacher_tokens = teacher_feat
                teacher_tokens = teacher_feat['x_norm_patchtokens']
                feature_teacher = teacher_tokens.reshape(teacher_tokens.size(0), int(teacher_tokens.size(1)**0.5), int(teacher_tokens.size(1)**0.5), teacher_tokens.size(2)).permute(0, 3, 1, 2)  # [B, D, H, W]
                feat_tea = feature_teacher            
            elif self.teacher_type == 'vit_small_patch16_224':
                teacher_feat = self.teacher.forward_features(image) 
                teacher_tokens = teacher_feat[:, 1:, :]
                feature_teacher = teacher_tokens.reshape(teacher_tokens.size(0), int(teacher_tokens.size(1)**0.5), int(teacher_tokens.size(1)**0.5), teacher_tokens.size(2)).permute(0, 3, 1, 2)  # [B, D, H, W]
                feat_tea = feature_teacher 
            elif self.teacher_type == 'swin_tiny_patch4_window7_224':
                teacher_feat = self.teacher.forward_features(image) 
                teacher_logits = self.teacher.forward_head(teacher_feat)
                teacher_tokens = teacher_feat                
                feature_teacher = teacher_tokens.reshape(teacher_tokens.size(0), int(teacher_tokens.size(1)**0.5), int(teacher_tokens.size(1)**0.5), teacher_tokens.size(2)).permute(0, 3, 1, 2)  # [B, D, H, W]
                feat_tea = feature_teacher 
            # elif self.teacher_type == 'dinov2s':
            #     teacher_feat = self.teacher.forward_features(image) 
            #     teacher_tokens = teacher_feat['x_norm_patchtokens']
            #     feature_teacher = teacher_tokens.reshape(teacher_tokens.size(0), int(teacher_tokens.size(1)**0.5), int(teacher_tokens.size(1)**0.5), teacher_tokens.size(2)).permute(0, 3, 1, 2)  # [B, D, H, W]
            #     feat_tea = feature_teacher
            else:
                t_out = self.teacher(image) # for dinov2
                if isinstance(t_out, tuple) and len(t_out) == 2:
                    teacher_out,teacher_logits = t_out  # teacher 就是embeddings
                    teacher_tokens = teacher_out['x_norm_patchtokens']
                    feature_teacher = teacher_tokens.reshape(teacher_tokens.size(0), int(teacher_tokens.size(1)**0.5), int(teacher_tokens.size(1)**0.5), teacher_tokens.size(2)).permute(0, 3, 1, 2)  # [B, D, H, W]
                    feat_tea = feature_teacher

                # If teacher_tokens has cls token for ViT-style, try to remove it
                if teacher_tokens is not None and teacher_tokens.dim() == 3 and teacher_tokens.shape[1] > 1:
                    # If teacher likely returns [B, N+1, C] with cls token, remove the cls token
                    # Heuristic: if N is not perfect square, try removing first token
                    N = teacher_tokens.shape[1]
                    s = int(math.sqrt(N))
                    if s * s != N:
                        # remove first token
                        teacher_tokens = teacher_tokens[:, 1:, :]
        from datetime import datetime
        save_path = "/nfs4/wangyb/projects/OFAKD/similarity_visuals/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(save_path,exist_ok=True)
        save_maps(feature_teacher, student_distill_feat, save_path)

        # heatmap = cca_heatmap(feature_teacher, student_distill_feat, save_path="/nfs4/wangyb/projects/OFAKD/")
        # print("Heatmap saved. Shape:", heatmap.shape)


        losses_dict = {}

        return logits_student, losses_dict


    def preprocess_teacher_tokens(self,teacher_tokens, target_size=14):
        """
        将 DINOv2 的 x_norm_patchtokens (B, N, D) 从 16×16 下采样到 target_size×target_size，
        返回 (B, target_size*target_size, D)，方便直接和学生 token 做蒸馏。
        
        teacher_tokens: [B, N, D]，N=256（16×16），可能包含 CLS token
        target_size: int，下采样的目标空间尺寸（学生 feature 的空间尺寸）
        """
        B, N, D = teacher_tokens.shape
        H = W = int(N ** 0.5)
        
        # 如果有 CLS token，去掉
        if H * W != N:
            teacher_tokens = teacher_tokens[:, 1:, :]  # 去掉 CLS
            B, N, D = teacher_tokens.shape
            H = W = int(N ** 0.5)

        # [B, N, D] -> [B, D, H, W]
        teacher_2d = teacher_tokens.transpose(1, 2).reshape(B, D, H, W)
        
        # 下采样到 target_size×target_size
        teacher_2d_down = F.adaptive_avg_pool2d(teacher_2d, (target_size, target_size))
        
        # [B, D, target, target] -> [B, target*target, D]
        teacher_tokens_down = teacher_2d_down.flatten(2).transpose(1, 2)
        
        return teacher_tokens_down

    
    def dyt_loss(self,dyt_student,dyt_teacher):    
        loss_dyt = F.mse_loss(dyt_student,dyt_teacher)
        return loss_dyt
    
    def logits_distillation_loss(self, student_logits, teacher_logits):
        """Logits蒸馏损失（KL散度）"""
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        return self.kldiv_loss(soft_student, soft_teacher) * (self.temperature ** 2)
    
    def cosine_distance_loss(self, student, teacher):
        """余弦距离损失（0~2范围）"""
        s_flat = student.flatten(1)  # [B, D]
        t_flat = teacher.flatten(1)
        cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # [B]
        return (1 - cosine_sim).mean()
    # def cosine_distance_loss(self, student, teacher):
    #     """
    #     student, teacher: [B, N, C] or [B, C, H, W] -> flatten then compute 1 - cos_sim mean
    #     """
    #     # flatten to [B, D]
    #     s_flat = student.flatten(1) if student.dim() > 2 else student
    #     t_flat = teacher.flatten(1) if teacher.dim() > 2 else teacher
    #     s_norm = F.normalize(s_flat, p=2, dim=1)
    #     t_norm = F.normalize(t_flat, p=2, dim=1)
    #     return F.mse_loss(s_norm, t_norm)

    @staticmethod
    def get_dynamic_beta(epoch, total_epochs=240, beta_max=500, beta_min=5, mode="cosine"):
        """动态调整特征蒸馏损失权重"""
        progress = epoch / total_epochs
        if mode == "linear":
            beta = beta_max - (beta_max - beta_min) * progress
        elif mode == "cosine":
            beta = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(math.pi * progress))
        else:
            beta = beta_max
        return beta
    
    @staticmethod
    def get_logits_weight(epoch, total_epochs, max_weight, min_weight, schedule="constant"):
        """动态调整Logits蒸馏权重"""
        if schedule == "constant":
            return max_weight
        
        progress = epoch / total_epochs
        
        if schedule == "linear":
            return max_weight - (max_weight - min_weight) * progress
        
        elif schedule == "inverse":
            # 后期增加权重
            return min_weight + (max_weight - min_weight) * progress
        
        elif schedule == "step":
            if epoch < 30:
                return min_weight  # 0.05
            
            # 平衡阶段 (30-60epoch)
            elif epoch < 60:
                # 线性增长：0.05 → 0.2
                progress = (epoch - 30) / (60 - 30)
                return min_weight + (max_weight * 0.67 - min_weight) * progress
            
            # 任务优化阶段 (60-90epoch)
            elif epoch < 90:
                # 继续增长到最大值：0.2 → 0.3
                progress = (epoch - 60) / (90 - 60)
                return max_weight * 0.67 + (max_weight - max_weight * 0.67) * progress
            
            # 最终微调阶段 (90-100epoch)
            else:
                # 保持最大值
                return max_weight





# ====== Example usage ======
if __name__ == "__main__":
    B, D, H, W = 8, 64, 7, 7
    feature_teacher = torch.randn(B, D, H, W)
    feature_student = torch.randn(B, D, H, W)

    cca_corr, corrs = cca_similarity(feature_teacher, feature_student, n_components=10)
    print("Average CCA correlation:", cca_corr)
    print("All canonical correlations:", corrs)
