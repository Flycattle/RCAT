import torch
import torch.nn as nn
# 对于cifar100，需要单独加载分类头
class Dinov2Teacher(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    @torch.no_grad()
    def extract_feat(self, x: torch.Tensor, use_cls: bool = True) -> torch.Tensor:
        """
        返回单向特征（默认 CLS token 归一化后的向量）。
        你的 dinov2 模型通常有 forward_features，返回 dict：
        - x_norm_clstoken: [B, dim]
        - x_norm_patchtokens: [B, N, dim]
        """
        self.backbone.eval()
        out = self.backbone.forward_features(x)
        return out
        # if use_cls and "x_norm_clstoken" in out:
        #     return out["x_norm_clstoken"]
        # # 兜底：对 patch token 做 mean pool
        # if "x_norm_patchtokens" in out:
        #     return out["x_norm_patchtokens"].mean(dim=1)
        # # 再兜底：如果有 'pooled'
        # if "pooled" in out:
        #     return out["pooled"]
        raise RuntimeError("Unknown feature keys in dinov2.forward_features output.")

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        feat = self.extract_feat(x, use_cls=True)   # [B, dim]
        logits = self.head(feat['x_norm_clstoken'])
        if return_feat:
            return logits, feat
        return logits