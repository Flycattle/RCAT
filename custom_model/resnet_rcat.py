# 这个是第一版 ResNet-VCA 代码，基于 torchvision 的 ResNet 实现
# 主要改动： 使用复杂的Adapter和Fusion模块 + 可选Bridge + 可选Pooling Head
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.resnet import ResNet 
from timm.models.resnet import Bottleneck, _create_resnet,BasicBlock
__all__ = ["ResNet", "resnet18_vca",  "resnet50_vca",]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

import torch
import torch.nn as nn
from timm.models.resnet import ResNet, Bottleneck

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 小组件
# ---------------------------

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class DilatedConvBlock(nn.Module):
    """ 3x3 膨胀卷积增强感受野，保持通道不变 """
    def __init__(self, c, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# class AdapterBottleneck(nn.Module):
#     """
#     更强的 Adapter：
#     1x1 (升/降维) -> 3x3 -> 1x1，默认把 in_c 映射到 out_c
#     """
#     def __init__(self, in_c, out_c, mid_mul=1.5, use_ln=True, H=16, W=16):
#         super().__init__()
#         mid_c = max(out_c, int(in_c * mid_mul))
#         self.net = nn.Sequential(
#             nn.Conv2d(in_c, mid_c, 1, bias=False),
#             nn.BatchNorm2d(mid_c),
#             nn.GELU(),
#             nn.Conv2d(mid_c, mid_c, 3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_c),
#             nn.GELU(),
#             nn.Conv2d(mid_c, out_c, 1, bias=False),
#             nn.BatchNorm2d(out_c),
#         )
#         self.use_ln = use_ln
#         if use_ln:
#             self.ln = nn.LayerNorm([out_c, H, W])
#         self.act = nn.ReLU(inplace=True)
#     def forward(self, x):
#         x = self.net(x)
#         if self.use_ln:
#             x = self.ln(x)
#         return self.act(x)

import torch
import torch.nn as nn

class AdapterBottleneck(nn.Module):
    """
    将学生特征映射到教师维度的轻量化 Adapter
    结构: 1x1 降维 -> 3x3 -> 1x1 升维
    支持 BatchNorm 和可选的通道 LayerNorm
    """
    def __init__(self, in_c=256, out_c=384, mid_c=192, use_relu=True,use_bn=True, use_ln=True, H=16, W=16):
        super().__init__()
        self.use_ln = use_ln
        self.use_relu = use_relu
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
        return self.act(x) if self.use_relu else x

 
class AdapterSimple(nn.Module):
    """ 轻量 Adapter：1x1 + LN + ReLU """
    def __init__(self, in_c, out_c, H=16, W=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.LayerNorm([out_c, H, W]),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class FusionConcatResidual(nn.Module):
    """ 拼接 -> 1x1 降维 + 残差到学生路径 """
    def __init__(self, stu_c, tea_c, out_c=None):
        super().__init__()
        out_c = out_c or stu_c
        self.reduce = nn.Conv2d(stu_c + tea_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, stu_x, tea_x):
        x = torch.cat([stu_x, tea_x], dim=1)
        x = self.act(self.bn(self.reduce(x)))
        # 残差回到学生空间（如果维度匹配）
        if x.shape == stu_x.shape:
            x = x + stu_x
        return x

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

class PoolHead(nn.Module):
    """
    可选池化头：
    - 'avg'：GAP（默认）
    - 'gap_gmp'：GAP+GMP concat，再线性映射
    - 'attn'：注意力池化（learnable）
    """
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
            raise ValueError(f"Unknown pool mode: {mode}")

    def forward(self, x):  # x: [B, C, H, W]
        if self.mode == 'avg':
            x = self.pool(x).flatten(1)
            return self.fc(x)
        elif self.mode == 'gap_gmp':
            a = self.gap(x)
            m = self.gmp(x)
            x = torch.cat([a, m], dim=1).flatten(1)
            return self.fc(x)
        elif self.mode == 'attn':
            w = self.attn(x)                 # [B,1,H,W]
            x = (x * w).sum(dim=[2, 3])      # 加权求和
            return self.fc(x)

# ---------------------------
# 主体封装
# ---------------------------
@register_model
class ResNet(nn.Module):
    """
    包装 torchvision.models.resnet50（或18/34/101，同接口）
    - 在 stage3 输出(14x14)上采样到 16x16
    - Adapter 将学生映射到 teacher_dim
    - Fusion 模块将教师信息融合回学生路径
    - layer4 的 stride 固定为 1，保持 16x16
    - 可选 Bridge（SE / Dilated）置于 layer4 之前
    - 可选 Pooling Head（avg / gap_gmp / attn）
    - forward 返回 (logits, dict)，dict['distill_feat'] 给蒸馏损失用
    """
    def __init__(
        self,
        backbone,                 # torchvision resnet50
        num_classes=1000,
        teacher_dim=384,          # DINOv2-S=384, DINOv2-B=768
        up_size=16,               # 16x16 对齐 teacher tokens
        adapter='bottleneck',     # 'bottleneck' | 'simple'
        fusion='concat',          # 'concat' | 'xattn'
        bridge='none',            # 'none' | 'dilated' | 'se'
        pool_mode='avg'           # 'avg' | 'gap_gmp' | 'attn'
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.up_size = up_size
        self.teacher_dim = teacher_dim

        # 取出一些结构引用
        self.conv1 = backbone.conv1
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu if hasattr(backbone, 'relu') else nn.ReLU(inplace=True)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # ---- 修改 layer4 的 stride=1，保持 16x16 ----
        self.layer4 = backbone.layer4

        # ---- 上采样到 16x16，用于对齐 teacher token map ----
        self.upsample = nn.Upsample(size=(up_size, up_size), mode='bilinear', align_corners=False)

        # ---- Adapter: student c4 -> teacher_dim ----
        c4 = self._infer_stage4_channels()
        if adapter == 'bottleneck':
            self.adapter = AdapterBottleneck(c4, teacher_dim, use_relu=False, use_ln=True, H=up_size, W=up_size)
        elif adapter == 'simple':
            self.adapter = AdapterSimple(c4, teacher_dim, H=up_size, W=up_size)
        else:
            raise ValueError(f"Unknown adapter: {adapter}")

        # ---- Fusion: 融合 teacher 到 student ----
        if fusion == 'concat':
            self.fusion = FusionConcatResidual(stu_c=c4, tea_c=teacher_dim, out_c=c4)
        elif fusion == 'xattn':
            self.fusion = FusionCrossAttention(stu_c=c4, tea_c=teacher_dim, heads=4, dim_qk=256, dim_v=256)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")
        self.fusion_type = fusion

        # ---- Bridge: layer4 之前增强 ----
        if bridge == 'dilated':
            self.bridge = DilatedConvBlock(c4, dilation=2)
        elif bridge == 'se':
            self.bridge = SEBlock(c4, r=16)
        else:
            self.bridge = nn.Identity()

        # ---- 池化 & 分类头 ----
        c4 = self._infer_stage4_channels()
        self.pool_head = PoolHead(in_c=c4, num_classes=num_classes, mode=pool_mode)

        # ---- 权重初始化（只对新加层）----
        self._init_new_weights()
    
    @torch.no_grad()
    def _make_layer4_stride1(self, layer4: nn.Sequential):
        import copy 
        """
        把 layer4 的第一个 block 的步长改为 1，并把 downsample 的步长也改为 1，
        从而保持空间分辨率 16x16 不变。兼容 BasicBlock 和 Bottleneck。
        """
        layer4 = copy.deepcopy(layer4)
        b0 = layer4[0]

        # 判定是否为 Bottleneck：Bottleneck 有 conv3，BasicBlock 没有
        is_bottleneck = hasattr(b0, 'conv3')

        # --- 主分支：把真正带 stride 的 3x3 conv 调成 1 ---
        if is_bottleneck:
            # torchvision 的 Bottleneck：stride 在 conv2
            if hasattr(b0, 'conv2') and getattr(b0.conv2, 'stride', None) != (1, 1):
                b0.conv2.stride = (1, 1)
            # timm 的一些实现可能把 stride 放在 conv1/conv3，这里兜底全设 1
            if hasattr(b0, 'conv1') and getattr(b0.conv1, 'stride', None) != (1, 1):
                b0.conv1.stride = (1, 1)
            if hasattr(b0, 'conv3') and getattr(b0.conv3, 'stride', None) != (1, 1):
                b0.conv3.stride = (1, 1)
        else:
            # BasicBlock：stride 在 conv1
            if hasattr(b0, 'conv1') and getattr(b0.conv1, 'stride', None) != (1, 1):
                b0.conv1.stride = (1, 1)
            # 兜底：conv2 本来就是 1，但也一并设
            if hasattr(b0, 'conv2') and getattr(b0.conv2, 'stride', None) != (1, 1):
                b0.conv2.stride = (1, 1)

        # --- 跳支：把 downsample 的 stride 也调成 1 ---
        if hasattr(b0, 'downsample') and b0.downsample is not None:
            ds0 = b0.downsample[0]  # 通常是 1x1 Conv
            if isinstance(ds0, nn.Conv2d) and getattr(ds0, 'stride', None) != (1, 1):
                ds0.stride = (1, 1)

        layer4[0] = b0
        return layer4

    @torch.no_grad()
    def _infer_stage3_channels(self):
        # torchvision resnet50: layer3 输出 1024 通道
        # resnet18: layer3 输出 256
        # 尝试从 layer3[-1] 的 convs 推断
        m = list(self.layer3.modules())
        # 找到最后一个 Conv2d 的 out_channels
        for mm in reversed(m):
            if isinstance(mm, nn.Conv2d):
                return mm.out_channels
        return 1024

    @torch.no_grad()
    def _infer_stage4_channels(self):
        m = list(self.layer4.modules())
        for mm in reversed(m):
            if isinstance(mm, nn.Conv2d):
                return mm.out_channels
        return 2048

    def _init_new_weights(self):
        for m in [self.adapter, self.fusion, self.bridge]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                    mod.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
                    if hasattr(mod, 'weight') and mod.weight is not None:
                        nn.init.ones_(mod.weight)
                    if hasattr(mod, 'bias') and mod.bias is not None:
                        nn.init.zeros_(mod.bias)

    def forward(self, x):
        # --- stem ---
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        x = self.maxpool(x)

        # --- res stages 1..3 ---
        feat1 = self.layer1(x)     # 56x56
        feat2 = self.layer2(feat1) # 28x28
        feat3 = self.layer3(feat2) # 14x14 (resnet18)
        feat4 = self.layer4(feat3) # 14x14 (resnet18)

        # --- upsample stage3 to 16x16 & adapter ---
        
        feat4_up = self.upsample(feat4)                 # [B, c4, 16, 16]

        adapter_feat = self.adapter(feat4_up)           # [B, teacher_dim, 16, 16]  <-- 用于蒸馏
        # print("adapter_feat:",adapter_feat.max(),adapter_feat.min())

        # --- fusion (teacher -> student) ---
        fused = self.fusion(feat4_up, adapter_feat)     # [B, c4, 16, 16]

        # --- bridge before layer4 ---
        fused = self.bridge(fused)                      # [B, c4, 16, 16]

        # --- 分类头（可选池化）---
        logits = self.pool_head(fused)


        feats = {}
        feats["feats"] = [
            F.relu(stem),
            F.relu(feat1),
            F.relu(feat2),
            F.relu(feat3),
            F.relu(feat4),
        ]
        feats["preact_feats"] = [stem, feat1, feat2, feat3, feat4]
        feats["adapter_feat"] = adapter_feat # 用于残差部分特征蒸馏
        feats["hca_feat"] = fused # 用于 heterogeneous class attention蒸馏
        return logits, feats




@register_model
def resnet18_proj_dim_768(pretrained=False,num_classes=100, up_size=7,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    import torchvision.models as tv

    # 以 ResNet-18 为例
    backbone = tv.resnet18(pretrained=False)
    if pretrained:
        backbone.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    model  = ResNet(
        backbone=backbone,
        num_classes=num_classes,
        teacher_dim=768,        # DINOv2-S: 384；如果是 DINOv2-B：768
        up_size=up_size,
        adapter='bottleneck',   # 思路1：更强 Adapter
        fusion='xattn',        # 思路2：'concat' 残差融合；或 'xattn' 轻量 Cross-Attn
        bridge='se',            # 思路3：'none' / 'dilated' / 'se'
        pool_mode='avg',       # 思路5：'avg' / 'gap_gmp' / 'attn'
    )
    
    return model

@register_model
def resnet18_proj_dim_384(pretrained=False,num_classes=100, up_size=7,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    import torchvision.models as tv

    # 以 ResNet-18 为例
    backbone = tv.resnet18(pretrained=False)
    if pretrained:
        backbone.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    model  = ResNet(
        backbone=backbone,
        num_classes=num_classes,
        teacher_dim=384,        # DINOv2-S: 384；如果是 DINOv2-B：768
        up_size=up_size,
        adapter='bottleneck',   # 思路1：更强 Adapter
        fusion='xattn',        # 思路2：'concat' 残差融合；或 'xattn' 轻量 Cross-Attn
        bridge='se',            # 思路3：'none' / 'dilated' / 'se'
        pool_mode='avg',       # 思路5：'avg' / 'gap_gmp' / 'attn'
    )
    
    return model


def resnet34_vcaV2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model

@register_model
def resnet50_proj_dim768(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101_vcaV2(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152_vcaV2(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model
