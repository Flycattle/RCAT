import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.resnet import ResNet 
__all__ = ["MobileNetV2_VCA", "mobilenetV2_vcaV2",  ]



# ---------- Bridge 组件 ----------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class DilatedDWConv(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

def make_bridge(mode: str, channels: int):
    if mode == 'none':
        return nn.Identity()
    elif mode == 'se':
        return SEBlock(channels)
    elif mode == 'dilated':
        return DilatedDWConv(channels, dilation=2)
    else:
        raise ValueError(f"Unsupported bridge mode: {mode}")



# ---------- PoolHead（与你之前一致，便于取 .fc.weight 做 CAM） ----------
class PoolHead(nn.Module):
    def __init__(self, in_c, num_classes, mode='avg'):
        super().__init__()
        self.mode = mode
        if mode == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_c, num_classes)
        elif mode == 'gap_gmp':
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.gmp = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Linear(in_c * 2, num_classes)
        elif mode == 'attn':
            self.attn = nn.Sequential(
                nn.Conv2d(in_c, in_c // 8, 1), nn.ReLU(inplace=True),
                nn.Conv2d(in_c // 8, 1, 1), nn.Sigmoid()
            )
            self.fc = nn.Linear(in_c, num_classes)
        else:
            raise ValueError(f"Unsupported pool mode: {mode}")

    def forward(self, x):
        if self.mode == 'avg':
            x = self.pool(x).flatten(1)
        elif self.mode == 'gap_gmp':
            gap = self.gap(x).flatten(1)
            gmp = self.gmp(x).flatten(1)
            x = torch.cat([gap, gmp], dim=1)
        elif self.mode == 'attn':
            attn = self.attn(x)
            x = (x * attn).sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)
        return self.fc(x)


class AdapterDW(nn.Module):
    def __init__(self, in_c=1280, out_c=384, r=192, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, r, 1, bias=False)]
        if use_bn: layers += [nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False)]
        if use_bn: layers += [nn.BatchNorm2d(r), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(r, out_c, 1, bias=False)]
        if use_bn: layers += [nn.BatchNorm2d(out_c)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class AdapterBottleneck(nn.Module):
    """
    将学生特征映射到教师维度的轻量化 Adapter
    结构: 1x1 降维 -> 3x3 -> 1x1 升维
    支持 BatchNorm 和可选的通道 LayerNorm
    """
    def __init__(self, in_c=256, out_c=384, mid_c=192, use_bn=True, use_ln=True, H=16, W=16):
        super().__init__()
        self.use_ln = use_ln
        self.H = H
        self.W = W

        layers = []
        # 1x1 降维
        layers.append(nn.Conv2d(in_c, mid_c, 1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(mid_c))
        layers.append(nn.GELU())

        # 3x3 卷积
        layers.append(nn.Conv2d(mid_c, mid_c, 3, padding=1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(mid_c))
        layers.append(nn.GELU())

        # 1x1 升维
        layers.append(nn.Conv2d(mid_c, out_c, 1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))

        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

        # 可选的轻量 LayerNorm，只在通道维度
        if use_ln:
            # 对 (B, C, H, W) 做 LN: 先 permute -> (B, H, W, C)
            # 参数量 = 2 * out_c，非常小
            self.ln = nn.LayerNorm(out_c)

    def forward(self, x):
        x = self.net(x)
        if self.use_ln:
            # 转换成 (B, H, W, C)
            x = x.permute(0, 2, 3, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2)  # 回到 (B, C, H, W)
        return self.act(x)

class FusionCrossAttention(nn.Module):
    """
    轻量 Cross-Attention：Q=student, K/V=teacher (都在空间上 16x16)
    将学生主动“取信息”，再映射回学生通道数。
    """
    def __init__(self, stu_c, tea_c, heads=4, dim_qk=256, dim_v=256):
        super().__init__()
        self.heads = heads
        self.scale = (dim_qk // heads) ** -0.5

        self.q_proj = nn.Conv2d(stu_c, dim_qk, 1, bias=False)
        self.k_proj = nn.Conv2d(tea_c, dim_qk, 1, bias=False)
        self.v_proj = nn.Conv2d(tea_c, dim_v, 1, bias=False)
        self.out_proj = nn.Conv2d(dim_v, stu_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(stu_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, stu_x, tea_x):
        B, _, H, W = stu_x.shape
        q = self.q_proj(stu_x)  # [B, dim_qk, H, W]
        k = self.k_proj(tea_x)
        v = self.v_proj(tea_x)

        def reshape_heads(t):
            B, C, H, W = t.shape
            t = t.view(B, self.heads, C // self.heads, H * W)  # [B,h,d,HW]
            return t

        q = reshape_heads(q)     # [B,h,d,HW]
        k = reshape_heads(k)     # [B,h,d,HW]
        v = reshape_heads(v)     # [B,h,dv,HW]

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # [B,h,HW,HW]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))  # [B,h,HW,dv]
        out = out.transpose(-2, -1).contiguous().view(B, -1, H, W)  # [B, dim_v, H, W]

        out = self.out_proj(out)
        out = self.act(self.bn(out + stu_x))  # 残差
        return out

# ---------- MobileNetV2 学生（上采样放在 layer4 后） ----------
@register_model
class MobileNetV2_VCA(nn.Module):
    """
    流程：
      stem+stage1+stage2+stage3(14x14) -> stage4(stride=1, 输出仍为 14x14) -> Upsample到16x16
      -> Adapter(到 teacher_dim) -> Fusion(回到 C4) -> Bridge -> PoolHead(对 fused 分类)
    返回 feats：
      - feats["feats"]        = [ReLU(stem), ReLU(f1), ReLU(f2), ReLU(f3), ReLU(f4)]
      - feats["preact_feats"] = [stem, f1, f2, f3, f4]
      - feats["adapter_feat"] = adapter_feat (B, teacher_dim, 16, 16)
      - feats["hca_feat"]     = fused        (B, C4,         16, 16)
    """
    def __init__(
        self,
        num_classes=200,
        teacher_dim=384,
        up_size=16,
        adapter='simple',   # 'simple' or 'linear',默认simple
        fusion='concat',        # 'concat' or 'add',新添加xattn
        bridge='se',            # 'none' / 'se' / 'dilated'
        pool_mode='avg',
        pretrained=False
    ):
        super().__init__()
        mb = tv.mobilenet_v2(pretrained=pretrained)

        # --- 拆分 MobileNetV2 ---
        # 官方实现：features[0] 是 stem (stride=2, 112x112)
        # 1..3 -> 56x56；4..6 -> 28x28；7..13 -> 14x14；14..(last) -> 7x7(我们会改为 14x14)
        self.stem = mb.features[0]            # 112x112
        self.stage1 = nn.Sequential(*mb.features[1:4])   # 56x56
        self.stage2 = nn.Sequential(*mb.features[4:7])   # 28x28
        self.stage3 = nn.Sequential(*mb.features[7:14])  # 14x14

        # # 修改 stage4 的第一层 depthwise stride=1，保持 14x14
        # self.stage4 = self._make_stage4_stride1(mb.features[14:])
        self.stage4 = nn.Sequential(*mb.features[14:])    # 7x7 (保持原始stride=2)

        # 通道数（固定：MobileNetV2 最后是 1280）
        c3 = 96     # stage3 输出通道
        c4 = 1280   # stage4 输出通道（分类前的高维特征）

        # --- 上采样(放在 layer4 后) ---
        self.upsample = nn.Upsample(size=(up_size, up_size), mode='bilinear', align_corners=False)

        # --- Adapter: 把 c4 -> teacher_dim（用于对齐教师 token 维度） ---
        if adapter == 'simple':
            self.adapter = nn.Sequential(
                nn.Conv2d(c4, teacher_dim, 1, bias=False),
                nn.BatchNorm2d(teacher_dim),
                nn.ReLU(inplace=True),
            )
        elif adapter == 'bottleneck':
            self.adapter = AdapterBottleneck(c4, teacher_dim, 192, True, True, up_size, up_size)
        elif adapter == 'dw':
            self.adapter = AdapterDW(c4,teacher_dim,192,True)
        elif adapter == 'linear':
            self.adapter = nn.Conv2d(c4, teacher_dim, 1, bias=True)
        else:
            raise ValueError(f"Unsupported adapter: {adapter}")

        # --- Fusion: 把 teacher 引导后的信息融回学生分支，输出回到 c4 ---
        if fusion == 'concat':
            self.fusion_conv = nn.Conv2d(c4 + teacher_dim, c4, 1, bias=False)
            self.fusion = lambda s, t: self.fusion_conv(torch.cat([s, t], dim=1))
        elif fusion == 'xattn':
            self.fusion = FusionCrossAttention(stu_c=c4, tea_c=teacher_dim, heads=4, dim_qk=256, dim_v=256)
        elif fusion == 'add':
            # 先把 adapter_feat 映射回 c4 再相加
            self.t2s = nn.Conv2d(teacher_dim, c4, 1, bias=False)
            self.fusion = lambda s, t: s + self.t2s(t)
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

        # --- Bridge: 细调融合后特征 ---
        self.bridge = make_bridge(bridge, c4)

        # --- 分类头：直接对 fused 做池化+FC（便于 CAM 取 fc.weight） ---
        self.pool_head = PoolHead(c4, num_classes, mode=pool_mode)

    @torch.no_grad()
    def _make_stage4_stride1(self, stage4: nn.Sequential):
        """
        把 stage4 的第一个 InvertedResidual 的 depthwise conv stride 改成 1，
        避免从 14x14 下采样到 7x7。
        """
        import copy
        stage4 = copy.deepcopy(stage4)
        b0 = stage4[0]
        # torchvision 的 InvertedResidual: b0.conv = Sequential([...])
        # 其中 depthwise conv 在 b0.conv[3]
        if hasattr(b0, "conv") and isinstance(b0.conv, nn.Sequential):
            for m in b0.conv.modules():
                # 找到 depthwise conv：groups = in_channels = out_channels
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
                    m.stride = (1, 1)
                    break
        stage4[0] = b0
        return stage4

    def forward(self, x):
        # --- Stem / Stages ---
        stem = self.stem(x)              # 112x112
        f1 = self.stage1(stem)           # 56x56
        f2 = self.stage2(f1)             # 28x28
        f3 = self.stage3(f2)             # 14x14
        f4 = self.stage4(f3)             # 14x14 

        # --- Upsample after layer4 ---
        f4_up = self.upsample(f4)        # [B, c4, 16, 16]

        # --- Adapter (学生对齐教师 token 维度) ---
        adapter_feat = self.adapter(f4_up)      # [B, teacher_dim, 16, 16]

        # --- Fusion (teacher -> student) ---
        fused = self.fusion(f4_up, adapter_feat)  # [B, c4, 16, 16]

        # --- Bridge ---
        fused = self.bridge(fused)              # [B, c4, 16, 16]

        # --- 分类头：对 fused 直接分类（便于 CAM）---
        logits = self.pool_head(fused)

        # --- 打包 KD / 可视化所需特征 ---
        feats = {}
        feats["feats"] = [
            F.relu(stem),
            F.relu(f1),
            F.relu(f2),
            F.relu(f3),
            F.relu(f4),
        ]
        feats["preact_feats"] = [stem, f1, f2, f3, f4]
        feats["adapter_feat"] = adapter_feat   # 学生用于对齐教师的特征
        feats["hca_feat"] = fused             # 融合后的特征，可做 HCA/CAM loss

        return logits, feats
        
@register_model
def mobilenetv2_proj_dim768(**kwargs):
    model = MobileNetV2_VCA( 
        num_classes=100,
        up_size=14,
        adapter='simple', #dw;bottleneck； 或者linear；或者simple轻量卷积
        teacher_dim=768,
        fusion='xattn', #'concat' 残差融合；或 'xattn' 轻量 Cross-Attn;或add
        pool_mode='avg',
        bridge='se',
        pretrained=False
    )    
    return model    

@register_model
def mobilenetv2_proj_dim384(**kwargs):
    model = MobileNetV2_VCA( 
        num_classes=100,
        up_size=14,
        adapter='simple', #dw;bottleneck； 或者linear；或者simple轻量卷积
        teacher_dim=768,
        fusion='xattn', #'concat' 残差融合；或 'xattn' 轻量 Cross-Attn;或add
        pool_mode='avg',
        bridge='se',
        pretrained=False
    )    
    return model    