import os
import datetime
import torch
import transforms
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from network_files import RetinaNet
from my_dataset import VOC2012DataSet
from train_utils import train_eval_utils as utils
import torchstat as stat

def create_model(num_classes, device):
    # 创建retinanet_res50_fpn模型
    # skip P2 because it generates too many anchors (according to their paper)
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256),
                                     trainable_layers=3)
    model = RetinaNet(backbone, num_classes)

    # 载入预训练权重
    # https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
    weights_dict = torch.load("./backbone/retinanet_resnet50_fpn.pth", map_location=device)
    # 删除分类器部分的权重，因为自己的数据集类别与预训练数据集类别(91)不一定致，如果载入会出现冲突
    del_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))

    return model

backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256),
                                     trainable_layers=3)

model = create_model(num_classes=2, device="cpu")
stat(backbone,(3,224,224))