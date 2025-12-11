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


@register_distiller
class HCA(BaseDistiller):
    requires_feat = False
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(HCA, self).__init__(student, teacher, criterion, args)
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
        # teacher_module = 
        
        # try:
        #     _, shape = student_module.stage_info(3)
        #     # shape could be tuple (C,H,W) or int
        #     if isinstance(shape, tuple) or isinstance(shape, list):
        #         self.student_feat_channels = int(shape[0])
        #     else:
        #         self.student_feat_channels = int(shape)
        # except Exception as e:
        #     raise RuntimeError(
        #         "AT requires your student to implement stage_info(3) to infer feat3 channels. "
        #         "Please implement stage_info or set adapter/fusion manually."
        #     )

        # determine teacher token/channel dim according to teacher type
        # You can expand mapping or set via cfg if needed
        if 'dinov2b' in self.teacher_type:
            self.teacher_token_dim = 768
        elif 'dinov2s' in self.teacher_type:
            self.teacher_token_dim = 384
        elif 'swin' in self.teacher_type:
            # swin tiny/base etc often 768 for base; adjust if needed
            self.teacher_token_dim = getattr(cfg.DISTILLER, "TEACHER_DIM", 768)
        else:
            # default fallback (user can override via cfg)
            self.teacher_token_dim = getattr(cfg.DISTILLER, "TEACHER_DIM", None)
            if self.teacher_token_dim is None:
                raise RuntimeError("Unknown teacher type and no TEACHER_DIM provided in cfg.")


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

            t_out = self.teacher(image) # for dinov2
            if isinstance(t_out, tuple) and len(t_out) == 2:
                teacher_out,teacher_logits = t_out  # teacher 就是embeddings
                teacher_tokens = teacher_out['x_norm_patchtokens']

            # If teacher_tokens has cls token for ViT-style, try to remove it
            if teacher_tokens is not None and teacher_tokens.dim() == 3 and teacher_tokens.shape[1] > 1:
                # If teacher likely returns [B, N+1, C] with cls token, remove the cls token
                # Heuristic: if N is not perfect square, try removing first token
                N = teacher_tokens.shape[1]
                s = int(math.sqrt(N))
                if s * s != N:
                    # remove first token
                    teacher_tokens = teacher_tokens[:, 1:, :]

            # teacher_map = None
            # if teacher_tokens is not None:
            #     teacher_map = self._reshape_teacher_tokens_to_map(teacher_tokens)  # [B, C_t, s, s]

        losses_dict = {}

        
        # ------------------ cat loss -------------------------------
        # def get_CAM(model, feat_map, arch=None, dataset=None, target_idx=None):
        if teacher_tokens is not None:
            cam_tea = get_CAM(model=self.teacher,feat_map=teacher_tokens,arch='dinov2')
            cam_stu = get_CAM(model=self.student,feat_map=feature_student['feat4'])
            loss_cat = CAT_loss(cam_stu,cam_tea,CAM_RESOLUTION=4)
            losses_dict["loss_cat"] = 2000 * loss_cat



        # ---------------- feature loss ---------------------------------------
        # if teacher_map exists, resize teacher_map to (H_s, W_s)
        if teacher_tokens is not None:
            
            # compute feature loss (cosine distance variant)
            dynamic_beta = self.get_dynamic_beta(
                epoch,
                total_epochs=self.args.epochs,
                beta_max=self.feat_loss_weight,
                beta_min=5,
                mode='cosine'
            )
            # teacher_tokens = self.preprocess_teacher_tokens(teacher_tokens, target_size=14) # 适合resnet_dinoV2.py resnet保留14*14.
            loss_feat = self.cosine_distance_loss(student_tokens, teacher_tokens)
            losses_dict["loss_feat"] = dynamic_beta * loss_feat

        # ------------------ classification loss ------------------
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, target)
        losses_dict["loss_gt"] = self.gt_loss_weight * loss_gt

        # ------------------ logits KD (if teacher logits available) ------------------
        if teacher_logits is not None and self.logits_loss_weight > 0:
            logits_weight = self.get_logits_weight(
                epoch,
                total_epochs=self.args.epochs,
                max_weight=self.logits_loss_weight,
                min_weight=0.05,
                schedule='step'
            )
            loss_logits = self.logits_distillation_loss(logits_student, teacher_logits)
            losses_dict["loss_kd"] = logits_weight * loss_logits

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

