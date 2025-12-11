from timm.models.resnet import ResNet

from .registry import register_method

_target_class = ResNet




# custom_forward.py（关键片段）
# custom_forward.py（关键片段）
# @register_method
# def forward_features(self, x, requires_feat=False):
#     """
#     返回到 layer3 的输出以及分阶段特征列表（不包含 layer4 的输出）
#     这样 AT 可以拿到 feat3，然后自己做 adapter/fusion 再调用 layer4。
#     """
#     feat = []
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.act1(x)
#     x = self.maxpool(x)

#     x = self.layer1(x)
#     feat.append(x)          # feat[0] -> stage1
#     x = self.layer2(x)
#     feat.append(x)          # feat[1] -> stage2
#     x = self.layer3(x)
#     feat.append(x)          # feat[2] -> stage3  <-- 我们要的 distill_feat

#     # DO NOT run layer4 here
#     # 返回 (x, feat) 其中 x == feat3（layer3 的输出）
#     return (x, feat) if requires_feat else x


# @register_method
# def forward(self, x, requires_feat=False):
#     """
#     如果需要 features（requires_feat=True），先拿到 layer3，再完整跑 layer4 与 head（原有行为）。
#     否则按普通 forward 流程。
#     """
#     if requires_feat:
#         x, feat = self.forward_features(x, requires_feat=True)  # x 为 layer3 输出
#         # 现在正常地把 layer4 执行，得到 feat4，保持原来 forward 的语义
#         x = self.layer4(x)                # feat4
#         feat.append(x)
#         x = self.forward_head(x, pre_logits=True)
#         feat.append(x)                    # pre-logits
#         x = self.fc(x)
#         return x, feat
#     else:
#         x = self.forward_features(x, requires_feat=False)  # 返回 layer3 输出
#         x = self.layer4(x)
#         x = self.forward_head(x)
#         return x

@register_method
def forward_stage1(self, x, requires_feat=False):
    """
    返回到 layer3 的输出以及分阶段特征列表（不包含 layer4 的输出）
    这样 AT 可以拿到 feat3，然后自己做 adapter/fusion 再调用 layer4。
    """
    feat = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    feat.append(x)          # feat[0] -> stage1
    x = self.layer2(x)
    feat.append(x)          # feat[1] -> stage2
    x = self.layer3(x)
    feat.append(x)          # feat[2] -> stage3  <-- 我们要的 distill_feat

    # DO NOT run layer4 here
    # 返回 (x, feat) 其中 x == feat3（layer3 的输出）
    return (x, feat) if requires_feat else x


@register_method
def forward_stage2(self, fusion_input, requires_feat=False):

        x = self.layer4(fusion_input)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    feat.append(x)
    x = self.layer2(x)
    feat.append(x)
    x = self.layer3(x)
    feat.append(x)
    x = self.layer4(x)
    feat.append(x)

    return (x, feat) if requires_feat else x


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.fc(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'resnet18':
        if stage == 1:
            index = 0
            shape = (64, 56, 56)
        elif stage == 2:
            index = 1
            shape = (128, 28, 28)
        elif stage == 3:
            index = 2
            shape = (256, 14, 14)
        elif stage == 4:
            index = 3
            shape = (512, 7, 7)
        elif stage == -1:
            index = -1
            shape = 512
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'resnet34':
        if stage == 1:
            index = 0
            shape = (64, 56, 56)
        elif stage == 2:
            index = 1
            shape = (128, 28, 28)
        elif stage == 3:
            index = 2
            shape = (256, 14, 14)
        elif stage == 4:
            index = 3
            shape = (512, 7, 7)
        elif stage == -1:
            index = -1
            shape = 512
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'resnet50':
        if stage == 1:
            index = 0
            shape = (256, 56, 56)
        elif stage == 2:
            index = 1
            shape = (512, 28, 28)
        elif stage == 3:
            index = 2
            shape = (1024, 14, 14)
        elif stage == 4:
            index = 3
            shape = (2048, 7, 7)
        elif stage == -1:
            index = -1
            shape = 2048
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'resnet101':
        if stage == 1:
            index = 0
            shape = (256, 56, 56)
        elif stage == 2:
            index = 1
            shape = (512, 28, 28)
        elif stage == 3:
            index = 2
            shape = (1024, 14, 14)
        elif stage == 4:
            index = 3
            shape = (2048, 7, 7)
        elif stage == -1:
            index = -1
            shape = 2048
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
