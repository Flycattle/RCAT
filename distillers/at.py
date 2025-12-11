import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseDistiller
import math
from .registry import register_distiller

import math
import torch
import torch.nn.functional as F

@register_distiller
class AT(BaseDistiller):
    requires_feat = False
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(AT, self).__init__(student, teacher, criterion, args)
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
        if 'dinov2b' or 'dinov3b' in self.teacher_type:
            self.teacher_token_dim = 768
        elif 'dinov2s' or 'dinov2s' in self.teacher_type:
            self.teacher_token_dim = 384
        elif 'swin' in self.teacher_type:
            # swin tiny/base etc often 768 for base; adjust if needed
            self.teacher_token_dim = getattr(cfg.DISTILLER, "TEACHER_DIM", 768)
        else:
            # default fallback (user can override via cfg)
            self.teacher_token_dim = getattr(cfg.DISTILLER, "TEACHER_DIM", None)
            if self.teacher_token_dim is None:
                raise RuntimeError("Unknown teacher type and no TEACHER_DIM provided in cfg.")

        # # Build adapter: student_feat3_channels -> teacher_dim
        # self.adapter = nn.Sequential(
        #     nn.Conv2d(self.student_feat_channels, self.teacher_token_dim, kernel_size=1, bias=False),
        #     # BatchNorm2d 更稳健，LayerNorm([C,H,W]) 在某些实现/版本会出问题
        #     nn.BatchNorm2d(self.teacher_token_dim),
        #     nn.ReLU(inplace=True)
        # )

        # # fusion_conv: (student_ch + teacher_ch) -> student_ch (so layer4 can accept fusion_input)
        # self.fusion_conv = nn.Conv2d(self.student_feat_channels + self.teacher_token_dim,
        #                              self.student_feat_channels,
        #                              kernel_size=1, bias=False)
        # # upsample if needed (assume teacher_map usually 16x16 for 224 input)
        # self.upsample_to_teacher = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)

        # # optional small teacher projection if teacher channels != adapter out (should not be needed)
        # self.teacher_proj = None  # lazy create if needed in forward

        # # init weights (kaiming)
        # for m in self.adapter.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.fusion_conv.weight, a=0, mode='fan_in')

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
            try:
                teacher_out = teacher_module.forward_features(image)
                # try commonly used keys
                for key in ["x_norm_patchtokens", "x_patchtokens", "patch_tokens", "tokens"]:
                    if isinstance(teacher_out, dict) and key in teacher_out:
                        teacher_tokens = teacher_out[key]
                        break
                # fallback: maybe forward_features returns tokens directly
                if teacher_tokens is None:
                    if isinstance(teacher_out, tuple) and len(teacher_out) >= 1:
                        maybe = teacher_out[0]
                        if isinstance(maybe, torch.Tensor) and maybe.dim() == 3:
                            teacher_tokens = maybe
                # teacher logits if exists
                if isinstance(teacher_out, dict) and "logits" in teacher_out:
                    teacher_logits = teacher_out["logits"]
            except Exception:
                # fallback to teacher(image) if forward_features not available
                t_out = self.teacher(image)
                if isinstance(t_out, tuple) and len(t_out) == 2:
                    teacher_logits, teacher_tokens = t_out

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

# @register_distiller
# class AT(BaseDistiller):
#     requires_feat = False
#     """
#     增强版 AT（只进行对齐，不创建 adapter/fusion）
#     关键假设:
#       - student(image) -> (logits_student, feature_student)
#       - feature_student 包含 "distill_feat"，形状为 [B, C_s, H_s, W_s]
#       - student 的 distill_feat 已经实现（例如由 student 中 adapter 生成）并且空间为 16x16（或其它）
#       - teacher.forward_features(image) 能返回包含 patch tokens 的 dict（例如 key "x_norm_patchtokens"），或 teacher(image) 返回 (logits, tokens)
#     """
#     def __init__(self, student, teacher, cfg):
#         super(AT, self).__init__(student, teacher)
#         self.cfg = cfg
#         self.teacher_type = cfg.DISTILLER.TEACHER  # e.g. 'dinov2', 'dinov2b' ...
#         self.ce_loss_weight = cfg.AT.LOSS.CE_WEIGHT
#         self.feat_loss_weight = cfg.AT.LOSS.FEAT_WEIGHT
#         self.logits_loss_weight = cfg.AT.LOSS.LOGITS_WEIGHT
#         self.temperature = cfg.AT.TEMPERATURE

#         self.ce_loss = torch.nn.CrossEntropyLoss()
#         self.kldiv_loss = torch.nn.KLDivLoss(reduction='batchmean')

#         # freeze teacher
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#         self.teacher.eval()

#     def _get_module(self, model):
#         # 兼容 DataParallel / 单卡
#         return model.module if hasattr(model, "module") else model

#     def _reshape_tokens_to_map(self, tokens):
#         """
#         tokens: [B, N, C] -> [B, C, s, s]  (s = sqrt(N))
#         若 tokens 已经是 map (B, C, H, W), 直接返回
#         """
#         if tokens is None:
#             return None
#         if tokens.dim() == 4:
#             return tokens  # already map
#         if tokens.dim() == 3:
#             B, N, C = tokens.shape
#             s = int(math.sqrt(N))
#             if s * s != N:
#                 # Not square: caller might have CLS token included; don't reshape blindly
#                 raise RuntimeError(f"Can't reshape teacher tokens (N={N}) to square map. Remove cls token or adjust teacher output.")
#             # [B, N, C] -> [B, C, s, s]
#             return tokens.permute(0, 2, 1).contiguous().view(B, C, s, s)
#         raise RuntimeError("Unsupported teacher tokens dimension: {}".format(tokens.dim()))

#     def _extract_teacher_tokens(self, image):
#         """
#         Attempt to extract token tensor from teacher in robust way.
#         Returns: (teacher_logits_or_None, teacher_tokens_or_None)
#         teacher_tokens will be either [B,N,C] or [B,C,H,W] (map), not flattened to student shape yet.
#         """
#         teacher_module = self._get_module(self.teacher)
#         teacher_tokens = None
#         teacher_logits = None
#         # try forward_features (common for DINOv2)
#         try:
#             out = teacher_module.forward_features(image)
#             # out may be dict or tensor or tuple
#             if isinstance(out, dict):
#                 # try common keys
#                 for key in ("x_norm_patchtokens", "x_patchtokens", "patch_tokens", "tokens", "x_norm_clstoken"):
#                     if key in out:
#                         teacher_tokens = out[key]
#                         break
#                 # sometimes logits stored
#                 if "logits" in out:
#                     teacher_logits = out["logits"]
#             else:
#                 # forward_features might return a tensor (e.g., tokens) or tuple
#                 if isinstance(out, tuple) and len(out) >= 1:
#                     maybe = out[0]
#                     if isinstance(maybe, torch.Tensor):
#                         teacher_tokens = maybe
#         except Exception:
#             # fallback: call teacher(image) directly
#             try:
#                 tout = teacher_module(image)
#                 if isinstance(tout, tuple) and len(tout) >= 2:
#                     teacher_logits, teacher_tokens = tout[0], tout[1]
#             except Exception:
#                 # give up gracefully
#                 teacher_tokens = None
#                 teacher_logits = None

#         # If tokens include CLS token (N not perfect square), try to remove first token heuristically
#         if teacher_tokens is not None and teacher_tokens.dim() == 3:
#             N = teacher_tokens.shape[1]
#             s = int(math.sqrt(N))
#             if s * s != N and N > 1:
#                 # remove first token (likely cls token)
#                 teacher_tokens = teacher_tokens[:, 1:, :]

#         return teacher_logits, teacher_tokens

#     def forward(self, image, target, epoch=1, **kwargs):
#         """
#         image: input images
#         target: labels
#         epoch: current epoch (for dynamic beta)
#         returns (logits_student, losses_dict)
#         losses_dict keys: loss_ce, loss_feat (if computed), loss_kd (if logits kd)
#         """
#         # student_module = self._get_module(self.student)
#         # teacher_module = self._get_module(self.teacher)

#         # ---------------- Student forward ----------------
#         # 期待 student(image) 返回 (logits, features_dict)
#         res = self.student(image)
#         if not (isinstance(res, tuple) and len(res) == 2):
#             raise RuntimeError("student(image) must return (logits, features_dict). features_dict should contain 'distill_feat'.")
#         logits_student, feature_student = res
#         if "distill_feat" not in feature_student:
#             raise RuntimeError("feature_student must contain 'distill_feat' (student-side adapter feature).")

#         student_distill_feat = feature_student["adapter_feat"]  # expected [B, C_s, H_s, W_s] e.g. [B,384,16,16]
#         if student_distill_feat is None or not isinstance(student_distill_feat, torch.Tensor):
#             raise RuntimeError("feature_student['adapter_feat'] must be a torch.Tensor with shape [B,C,H,W].")

#         B, C_s, H_s, W_s = student_distill_feat.shape
#         # student tokens: [B, N_s, C_s] where N_s = H_s * W_s
#         student_tokens = student_distill_feat.flatten(2).transpose(1, 2)  # [B, N_s, C_s]

#         # ---------------- Teacher forward (no grad) ----------------
#         with torch.no_grad():
#             teacher_logits, teacher_tokens = self._extract_teacher_tokens(image)

#         losses_dict = {}
#         # CE loss
#         loss_ce = self.ce_loss(logits_student, target)
#         losses_dict["loss_ce"] = self.ce_loss_weight * loss_ce

#         # ---------------- Feature distillation ----------------
#         if teacher_tokens is not None:
#             # teacher_tokens may be [B,N_t,C_t] or [B,C_t,H_t,W_t]
#             # We need to obtain teacher_tokens_resized as [B, N_s, C_t_res] where C_t_res == C_s ideally
#             # Step1: If teacher tokens are map-like, convert to map
#             teacher_map = None
#             if teacher_tokens.dim() == 3:
#                 # shape [B, N_t, C_t]
#                 N_t = teacher_tokens.shape[1]
#                 s = int(math.sqrt(N_t))
#                 if s * s == N_t:
#                     teacher_map = teacher_tokens.permute(0, 2, 1).contiguous().view(B, teacher_tokens.shape[2], s, s)
#                 else:
#                     # If not perfect square here, we already tried to remove cls token earlier; if still not square, abort
#                     raise RuntimeError(f"Teacher tokens count N_t={N_t} cannot be reshaped to square map. Teacher tokens: {teacher_tokens.shape}")
#             elif teacher_tokens.dim() == 4:
#                 teacher_map = teacher_tokens  # [B, C_t, H_t, W_t]
#             else:
#                 raise RuntimeError(f"Unsupported teacher tokens dim: {teacher_tokens.dim()}")

#             # Step2: resize teacher_map to student spatial (H_s, W_s) if necessary
#             if (teacher_map.shape[-2], teacher_map.shape[-1]) != (H_s, W_s):
#                 teacher_map_resized = F.interpolate(teacher_map, size=(H_s, W_s), mode='bilinear', align_corners=False)
#             else:
#                 teacher_map_resized = teacher_map

#             # Step3: ensure channel dims match (C_t_res == C_s)
#             C_t_res = teacher_map_resized.shape[1]
#             if C_t_res != C_s:
#                 # we don't create extra learnable projection here (by your request).
#                 # So we require channels to match; otherwise tell user to make student adapter match teacher channels.
#                 raise RuntimeError(
#                     f"Channel mismatch between student distill feature ({C_s}) and teacher feature ({C_t_res}). "
#                     "Please make student's adapter output channels equal to teacher token channels."
#                 )

#             # Step4: flatten teacher_map_resized -> [B, N_s, C_s] to compare with student_tokens
#             teacher_tokens_resized = teacher_map_resized.flatten(2).transpose(1, 2)  # [B, N_s, C_s]

#             # compute dynamic beta
#             dynamic_beta = self.get_dynamic_beta(
#                 epoch,
#                 total_epochs=getattr(self.cfg.SOLVER, "EPOCHS", 240),
#                 beta_max=self.feat_loss_weight,
#                 beta_min=5,
#                 mode='cosine'
#             )

#             # compute feature loss using your existing method
#             # assumes cosine_distance_loss accepts (student_tokens, teacher_tokens_resized)
#             loss_feat = self.cosine_distance_loss(student_tokens, teacher_tokens_resized)
#             losses_dict["loss_feat"] = dynamic_beta * loss_feat

#         # ---------------- Logits distillation (optional) ----------------
#         if teacher_logits is not None and self.logits_loss_weight > 0:
#             # logits weight schedule support
#             if hasattr(self, "get_logits_weight"):
#                 logits_weight = self.get_logits_weight(
#                     epoch,
#                     total_epochs=getattr(self.cfg.SOLVER, "EPOCHS", 240),
#                     max_weight=self.logits_loss_weight,
#                     min_weight=0.05,
#                     schedule='step'
#                 )
#             else:
#                 logits_weight = self.logits_loss_weight

#             loss_kd = self.logits_distillation_loss(logits_student, teacher_logits)
#             losses_dict["loss_kd"] = logits_weight * loss_kd

#         return logits_student, losses_dict
    
#     def dyt_loss(self,dyt_student,dyt_teacher):    
#         loss_dyt = F.mse_loss(dyt_student,dyt_teacher)
#         return loss_dyt
    
#     def logits_distillation_loss(self, student_logits, teacher_logits):
#         """Logits蒸馏损失（KL散度）"""
#         soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
#         soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
#         return self.kldiv_loss(soft_student, soft_teacher) * (self.temperature ** 2)
    
#     def cosine_distance_loss(self, student, teacher):
#         """余弦距离损失（0~2范围）"""
#         s_flat = student.flatten(1)  # [B, D]
#         t_flat = teacher.flatten(1)
#         cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # [B]
#         return (1 - cosine_sim).mean()
#     # def cosine_distance_loss(self, student, teacher):
#     #     """
#     #     student, teacher: [B, N, C] or [B, C, H, W] -> flatten then compute 1 - cos_sim mean
#     #     """
#     #     # flatten to [B, D]
#     #     s_flat = student.flatten(1) if student.dim() > 2 else student
#     #     t_flat = teacher.flatten(1) if teacher.dim() > 2 else teacher
#     #     s_norm = F.normalize(s_flat, p=2, dim=1)
#     #     t_norm = F.normalize(t_flat, p=2, dim=1)
#     #     return F.mse_loss(s_norm, t_norm)

#     @staticmethod
#     def get_dynamic_beta(epoch, total_epochs=240, beta_max=500, beta_min=5, mode="cosine"):
#         """动态调整特征蒸馏损失权重"""
#         progress = epoch / total_epochs
#         if mode == "linear":
#             beta = beta_max - (beta_max - beta_min) * progress
#         elif mode == "cosine":
#             beta = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(math.pi * progress))
#         else:
#             beta = beta_max
#         return beta
    
#     @staticmethod
#     def get_logits_weight(epoch, total_epochs, max_weight, min_weight, schedule="constant"):
#         """动态调整Logits蒸馏权重"""
#         if schedule == "constant":
#             return max_weight
        
#         progress = epoch / total_epochs
        
#         if schedule == "linear":
#             return max_weight - (max_weight - min_weight) * progress
        
#         elif schedule == "inverse":
#             # 后期增加权重
#             return min_weight + (max_weight - min_weight) * progress
        
#         elif schedule == "step":
#             if epoch < 30:
#                 return min_weight  # 0.05
            
#             # 平衡阶段 (30-60epoch)
#             elif epoch < 60:
#                 # 线性增长：0.05 → 0.2
#                 progress = (epoch - 30) / (60 - 30)
#                 return min_weight + (max_weight * 0.67 - min_weight) * progress
            
#             # 任务优化阶段 (60-90epoch)
#             elif epoch < 90:
#                 # 继续增长到最大值：0.2 → 0.3
#                 progress = (epoch - 60) / (90 - 60)
#                 return max_weight * 0.67 + (max_weight - max_weight * 0.67) * progress
            
#             # 最终微调阶段 (90-100epoch)
#             else:
#                 # 保持最大值
#                 return max_weight




# class AT(BaseDistiller):
#     requires_feat = False
#     def __init__(self, student, teacher, criterion, args, **kwargs):
#         super(AT, self).__init__(student, teacher, criterion, args)
#         self.args = args
#         self.teacher_type = args.teacher  # 'dinov2s' / 'dinov2b' / 'vit'...
#         # loss weights
#         self.gt_loss_weight = args.gt_loss_weight
#         self.feat_loss_weight = args.feat_loss_weight
#         self.logits_loss_weight = args.kd_loss_weight
#         self.temperature = args.kd_temperature

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

#         # freeze teacher
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#         self.teacher.eval()

#         # Try to infer student's feat3 channels via student.stage_info(3)
#         student_module = self._get_module(self.student)
#         # teacher_module = 
        
#         try:
#             _, shape = student_module.stage_info(3)
#             # shape could be tuple (C,H,W) or int
#             if isinstance(shape, tuple) or isinstance(shape, list):
#                 self.student_feat_channels = int(shape[0])
#             else:
#                 self.student_feat_channels = int(shape)
#         except Exception as e:
#             raise RuntimeError(
#                 "AT requires your student to implement stage_info(3) to infer feat3 channels. "
#                 "Please implement stage_info or set adapter/fusion manually."
#             )

#         # determine teacher token/channel dim according to teacher type
#         # You can expand mapping or set via cfg if needed
#         if 'dinov2b' in self.teacher_type:
#             self.teacher_token_dim = 768
#         elif 'dinov2s' in self.teacher_type:
#             self.teacher_token_dim = 384
#         elif 'swin' in self.teacher_type:
#             # swin tiny/base etc often 768 for base; adjust if needed
#             self.teacher_token_dim = getattr(cfg.DISTILLER, "TEACHER_DIM", 768)
#         else:
#             # default fallback (user can override via cfg)
#             self.teacher_token_dim = getattr(cfg.DISTILLER, "TEACHER_DIM", None)
#             if self.teacher_token_dim is None:
#                 raise RuntimeError("Unknown teacher type and no TEACHER_DIM provided in cfg.")

#         # Build adapter: student_feat3_channels -> teacher_dim
#         self.adapter = nn.Sequential(
#             nn.Conv2d(self.student_feat_channels, self.teacher_token_dim, kernel_size=1, bias=False),
#             # BatchNorm2d 更稳健，LayerNorm([C,H,W]) 在某些实现/版本会出问题
#             nn.BatchNorm2d(self.teacher_token_dim),
#             nn.ReLU(inplace=True)
#         )

#         # fusion_conv: (student_ch + teacher_ch) -> student_ch (so layer4 can accept fusion_input)
#         self.fusion_conv = nn.Conv2d(self.student_feat_channels + self.teacher_token_dim,
#                                      self.student_feat_channels,
#                                      kernel_size=1, bias=False)
#         # upsample if needed (assume teacher_map usually 16x16 for 224 input)
#         self.upsample_to_teacher = nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)

#         # optional small teacher projection if teacher channels != adapter out (should not be needed)
#         self.teacher_proj = None  # lazy create if needed in forward

#         # init weights (kaiming)
#         for m in self.adapter.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#         nn.init.kaiming_normal_(self.fusion_conv.weight, a=0, mode='fan_in')

#     def _get_module(self, model):
#         return model.module if  hasattr(model, "module") else model

#     def _reshape_teacher_tokens_to_map(self, teacher_tokens):
#         """
#         teacher_tokens: [B, N, C] => [B, C, s, s] where s = sqrt(N)
#         """
#         if teacher_tokens is None:
#             return None
#         B, N, C = teacher_tokens.shape
#         s = int(math.sqrt(N))
#         if s * s != N:
#             raise RuntimeError(f"Teacher tokens N ({N}) is not a perfect square. Can't reshape to map.")
#         teacher_map = teacher_tokens.permute(0, 2, 1).contiguous().view(B, C, s, s)
#         return teacher_map


#     def forward(self, image, target, epoch=1, **kwargs):
#         """
#         main training step:
#         - 获取学生 feat3（layer3 输出）
#         - 获取教师 patch tokens -> reshape -> resize 到 feat3 spatial
#         - adapter(student_feat3) -> adapter_feat
#         - compute feature loss between adapter_feat and teacher_map_resized (flatten->tokens)
#         - fusion_input = cat(student_feat3, adapter_feat), fusion_conv -> fusion_input_reduced
#         - call student.layer4(fusion_input_reduced) -> feat4
#         - call student's head/fc to get logits_student
#         - compute CE / logits KD if teacher logits exist
#         """
#         student_module = self._get_module(self.student)
#         teacher_module = self._get_module(self.teacher)

#         # ------------------ 获取 student feat3 ------------------
#         # Try a few ways robustly
#         student_feat3 = None
#         feature_student = None
#         logits_student_orig = None

#         # Preferred: student.forward_features(..., requires_feat=True) returns (x_layer3, feat_list) or x_layer3
#         if hasattr(student_module, "forward_features"):
#             out = student_module.forward_stage1(image, requires_feat=True)  #  return (x, feat) if requires_feat else x
#             # can be (x_layer3, feat_list) or x_layer3
#             if isinstance(out, tuple):
#                 student_feat3, feature_student = out[0], out[1] 
#             else:
#                 student_feat3 = out
#         # fallback: student(image) -> (logits, features) where features["distill_feat"] is feat3
#         if student_feat3 is None:
#             # call student(image) as fallback; this may have already run layer4 inside student
#             # so later we can't rerun layer4 on fusion_input. We still compute feature loss but won't fuse into layer4.
#             res = self.student(image)
#             if isinstance(res, tuple) and len(res) == 2:
#                 logits_student_orig, feature_student = res
#                 student_feat3 = feature_student.get("distill_feat", None)
#             else:
#                 raise RuntimeError("Unable to extract student feat3. Make sure student.forward_features or student(image) returns feature dict with 'distill_feat'.")

#         # sanity check
#         if student_feat3 is None:
#             raise RuntimeError("student_feat3 is None. Student must expose layer3 output as 'distill_feat' or via forward_features().")

#         B, C_s, H_s, W_s = student_feat3.shape

#         # ------------------ 获取 teacher tokens & build teacher_map ------------------
#         with torch.no_grad():
#             # try teacher.forward_features (DINOv2 style) first
#             teacher_tokens = None
#             teacher_logits = None
#             try:
#                 teacher_out = teacher_module.forward_features(image)
#                 # try commonly used keys
#                 for key in ["x_norm_patchtokens", "x_patchtokens", "patch_tokens", "tokens"]:
#                     if isinstance(teacher_out, dict) and key in teacher_out:
#                         teacher_tokens = teacher_out[key]
#                         break
#                 # fallback: maybe forward_features returns tokens directly
#                 if teacher_tokens is None:
#                     if isinstance(teacher_out, tuple) and len(teacher_out) >= 1:
#                         maybe = teacher_out[0]
#                         if isinstance(maybe, torch.Tensor) and maybe.dim() == 3:
#                             teacher_tokens = maybe
#                 # teacher logits if exists
#                 if isinstance(teacher_out, dict) and "logits" in teacher_out:
#                     teacher_logits = teacher_out["logits"]
#             except Exception:
#                 # fallback to teacher(image) if forward_features not available
#                 t_out = self.teacher(image)
#                 if isinstance(t_out, tuple) and len(t_out) == 2:
#                     teacher_logits, teacher_tokens = t_out

#             # If teacher_tokens has cls token for ViT-style, try to remove it
#             if teacher_tokens is not None and teacher_tokens.dim() == 3 and teacher_tokens.shape[1] > 1:
#                 # If teacher likely returns [B, N+1, C] with cls token, remove the cls token
#                 # Heuristic: if N is not perfect square, try removing first token
#                 N = teacher_tokens.shape[1]
#                 s = int(math.sqrt(N))
#                 if s * s != N:
#                     # remove first token
#                     teacher_tokens = teacher_tokens[:, 1:, :]

#             teacher_map = None
#             if teacher_tokens is not None:
#                 teacher_map = self._reshape_teacher_tokens_to_map(teacher_tokens)  # [B, C_t, s, s]

#         losses_dict = {}
#         # ------------------ adapter: student_feat3 -> adapter_feat ------------------
#         adapter_feat = self.adapter(student_feat3)  # [B, C_t (teacher_dim), H_s, W_s] ideally
#         # if teacher_map exists, resize teacher_map to (H_s, W_s)
#         if teacher_map is not None:
#             target_h, target_w = adapter_feat.shape[-2], adapter_feat.shape[-1]
#             if (teacher_map.shape[-2], teacher_map.shape[-1]) != (target_h, target_w):
#                 teacher_map_resized = F.interpolate(teacher_map, size=(target_h, target_w),
#                                                     mode='bilinear', align_corners=False)
#             else:
#                 teacher_map_resized = teacher_map

#             # if channels mismatch (rare), create teacher_proj to map teacher_map channels -> adapter channels
#             if teacher_map_resized.shape[1] != adapter_feat.shape[1]:
#                 if self.teacher_proj is None:
#                     self.teacher_proj = nn.Conv2d(teacher_map_resized.shape[1],
#                                                   adapter_feat.shape[1],
#                                                   kernel_size=1, bias=False).to(adapter_feat.device)
#                     nn.init.kaiming_normal_(self.teacher_proj.weight, a=0, mode='fan_in')
#                 teacher_map_resized = self.teacher_proj(teacher_map_resized)

#             # student_tokens & teacher_tokens both [B, N_s, C_a]
#             student_tokens = adapter_feat.flatten(2).transpose(1, 2)
#             teacher_tokens_resized = teacher_map_resized.flatten(2).transpose(1, 2)

#             # compute feature loss (cosine distance variant)
#             dynamic_beta = self.get_dynamic_beta(
#                 epoch,
#                 total_epochs=self.args.epochs,
#                 beta_max=self.feat_loss_weight,
#                 beta_min=5,
#                 mode='cosine'
#             )
#             loss_feat = self.cosine_distance_loss(student_tokens, teacher_tokens_resized)
#             losses_dict["loss_feat"] = dynamic_beta * loss_feat

#         # ------------------ fusion: combine student_feat3 & adapter_feat -> fusion_input_reduced ------------------
#         fusion_input = torch.cat([student_feat3, adapter_feat], dim=1)  # [B, C_s + C_t, H_s, W_s]
#         fusion_input_reduced = self.fusion_conv(fusion_input)  # [B, C_s, H_s, W_s]

#         # ------------------ call student's layer4 on fusion_input_reduced to get new logits ------------------
#         # # Use student_module.layer4 and student_module.forward_head / student_module.fc if available
#         # if hasattr(student_module, "layer4"):
#         #     feat4 = student_module.layer4(fusion_input_reduced)
#         # else:
#         #     raise RuntimeError("student has no layer4 attribute; cannot continue fusion -> layer4 call.")

#         # # call forward_head if exists to get pre_logits
#         # if hasattr(student_module, "forward_head"):
#         #     pre_logits = student_module.forward_head(feat4, pre_logits=True)
#         #     logits_student = student_module.fc(pre_logits)
#         # else:
#         #     # fallback: do avg pool + flatten + fc
#         #     x = student_module.avgpool(F.relu(feat4))
#         #     x = x.view(x.size(0), -1)
#         #     logits_student = student_module.fc(x)
#         logits_student = student_module.forward_stage2(fusion_input_reduced)

#         # ------------------ classification loss ------------------
#         loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, target)
#         losses_dict["loss_gt"] = self.gt_loss_weight * loss_gt

#         # ------------------ logits KD (if teacher logits available) ------------------
#         if teacher_logits is not None and self.logits_loss_weight > 0:
#             logits_weight = self.get_logits_weight(
#                 epoch,
#                 total_epochs=self.args.epochs,
#                 max_weight=self.logits_loss_weight,
#                 min_weight=0.05,
#                 schedule='step'
#             )
#             loss_logits = self.logits_distillation_loss(logits_student, teacher_logits)
#             losses_dict["loss_kd"] = logits_weight * loss_logits

#         return logits_student, losses_dict

    
#     def dyt_loss(self,dyt_student,dyt_teacher):    
#         loss_dyt = F.mse_loss(dyt_student,dyt_teacher)
#         return loss_dyt
    
#     def logits_distillation_loss(self, student_logits, teacher_logits):
#         """Logits蒸馏损失（KL散度）"""
#         soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
#         soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
#         return self.kldiv_loss(soft_student, soft_teacher) * (self.temperature ** 2)
    
#     def cosine_distance_loss(self, student, teacher):
#         """余弦距离损失（0~2范围）"""
#         s_flat = student.flatten(1)  # [B, D]
#         t_flat = teacher.flatten(1)
#         cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=1)  # [B]
#         return (1 - cosine_sim).mean()
#     # def cosine_distance_loss(self, student, teacher):
#     #     """
#     #     student, teacher: [B, N, C] or [B, C, H, W] -> flatten then compute 1 - cos_sim mean
#     #     """
#     #     # flatten to [B, D]
#     #     s_flat = student.flatten(1) if student.dim() > 2 else student
#     #     t_flat = teacher.flatten(1) if teacher.dim() > 2 else teacher
#     #     s_norm = F.normalize(s_flat, p=2, dim=1)
#     #     t_norm = F.normalize(t_flat, p=2, dim=1)
#     #     return F.mse_loss(s_norm, t_norm)

#     @staticmethod
#     def get_dynamic_beta(epoch, total_epochs=240, beta_max=500, beta_min=5, mode="cosine"):
#         """动态调整特征蒸馏损失权重"""
#         progress = epoch / total_epochs
#         if mode == "linear":
#             beta = beta_max - (beta_max - beta_min) * progress
#         elif mode == "cosine":
#             beta = beta_min + 0.5 * (beta_max - beta_min) * (1 + math.cos(math.pi * progress))
#         else:
#             beta = beta_max
#         return beta
    
#     @staticmethod
#     def get_logits_weight(epoch, total_epochs, max_weight, min_weight, schedule="constant"):
#         """动态调整Logits蒸馏权重"""
#         if schedule == "constant":
#             return max_weight
        
#         progress = epoch / total_epochs
        
#         if schedule == "linear":
#             return max_weight - (max_weight - min_weight) * progress
        
#         elif schedule == "inverse":
#             # 后期增加权重
#             return min_weight + (max_weight - min_weight) * progress
        
#         elif schedule == "step":
#             if epoch < 30:
#                 return min_weight  # 0.05
            
#             # 平衡阶段 (30-60epoch)
#             elif epoch < 60:
#                 # 线性增长：0.05 → 0.2
#                 progress = (epoch - 30) / (60 - 30)
#                 return min_weight + (max_weight * 0.67 - min_weight) * progress
            
#             # 任务优化阶段 (60-90epoch)
#             elif epoch < 90:
#                 # 继续增长到最大值：0.2 → 0.3
#                 progress = (epoch - 60) / (90 - 60)
#                 return max_weight * 0.67 + (max_weight - max_weight * 0.67) * progress
            
#             # 最终微调阶段 (90-100epoch)
#             else:
#                 # 保持最大值
#                 return max_weight
