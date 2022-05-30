import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import RetinaNet
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from draw_box_utils import draw_box
import math
import torch.nn as nn
import numpy as np
import scipy.misc
import cv2
fmap_block = list()
input_block = list()
def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)

def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        scipy.misc.imsave(str(index)+".png", feature_map[index-1])
    plt.show()
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup  # oup：论文Fig2(a)中output的通道数
        init_channels = math.ceil(oup/ratio)  # init_channels: 在论文Fig2(b)中,黄色部分的通道数
                                                # ceil函数：向上取整，
                                                # ratio：在论文Fig2(b)中，output通道数与黄色部分通道数的比值
        new_channels = init_channels*(ratio-1)  # new_channels: 在论文Fig2(b)中，output红色部分的通道数

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                                                #1//2=0
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(   # 黄色部分所用的普通的卷积运算，生成红色部分
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                                                # 3//2=1；groups=init_channel 组卷积极限情况=depthwise卷积
            nn.BatchNorm2d(new_channels),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)         # torch.cat: 在给定维度上对输入的张量序列进行连接操作
                                                # 将黄色部分和红色部分在通道上进行拼接
        return out[:,:self.oup,:,:]

def create_model(num_classes):
    # resNet50+fpn+retinanet
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # 注意：不包含背景
    model = create_model(num_classes=1)
    model.backbone.body.layer1._modules['0'].conv2 = GhostModule(inp=64,oup=64)
    model.backbone.body.layer1._modules['1'].conv2 = GhostModule(inp=64, oup=64)
    model.backbone.body.layer1._modules['2'].conv2 = GhostModule(inp=64, oup=64)







    # load train weights
    train_weights = "./save_weights/resNetFpn-model-19.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)
    print(model)


    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    original_img = Image.open("test7.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()






        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.18,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        original_img.save("test_result.jpg")
        print(predict_boxes)
        print(predict_classes)
        print(predict_scores)
        print(category_index)



if __name__ == '__main__':
    main()

