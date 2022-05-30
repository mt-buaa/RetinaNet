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
from cv2 import cv2

#---------------------------------
from cv2 import cv2
import numpy as np
pic_path = 'C:/Users/MT/Desktop/test.jpg'  # 分割的图片的位置
pic_target = 'C:/Users/MT/Desktop/big/'  # 分割后的图片保存的文件夹
# 要分割后的尺寸
cut_width = 500
cut_length = 500
# 读取要分割的图片，以及其尺寸等数据
picture = cv2.imread(pic_path)
(width, length, depth) = picture.shape
# 预处理生成0矩阵
pic = np.zeros((cut_width, cut_length, depth))
# 计算可以划分的横纵的个数
num_width = int(width / cut_width)
num_length = int(length / cut_length)
# for循环迭代生成
for i in range(0, num_width):
    for j in range(0, num_length):
        pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
        result_path = pic_target + '{}_{}.jpg'.format(i + 1, j + 1)
        cv2.imwrite(result_path, pic)

print("done!!!")
#---------------------------------
fmap_block = list()
input_block = list()
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


    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
    model.eval()  # 进入验证模式
    path_qg = "C:/Users/MT/Desktop/big"
    jpg_list = [x for x in os.listdir(path_qg) if x.endswith(".jpg")]
    real_predict_boxes = []
    real_predict_classes = []
    real_predict_scores = []
    for num,item in enumerate(jpg_list):
        num1 = int(item.split(".")[0].split("_")[0])-1
        num2 = int(item.split(".")[0].split("_")[1])-1
        #print(str(num)+'/'+str(len(jpg_list)))
        original_img = Image.open(path_qg+'/'+item)
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)
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
            for x in predict_boxes:
                x[0] = x[0]+500*num2
                x[2] = x[2]+500*num2
                x[1] = x[1]+500*num1
                x[3] = x[3]+500*num1
                real_predict_boxes.append(x.tolist())
            for x in predict_scores:
                real_predict_scores.append(x.tolist())
                #real_predict_scores.append(x.tolist())
            for x in predict_classes:
                real_predict_classes.append(x.tolist())

    box_rel = np.array(real_predict_boxes)
    box_rel = box_rel.astype(np.float32)
    sco_rel = np.array(real_predict_scores)
    sco_rel = sco_rel.astype(np.float32)
    cls_rel = np.array(real_predict_classes)
    cls_rel = cls_rel.astype(np.int64)
    if len(real_predict_boxes) == 0:
        print("没有检测到任何目标!")
    big_img = Image.open("C:/Users/MT/Desktop/test.jpg")
    order = np.argsort(-sco_rel)
    #sco_rel_ordered = sco_rel.sort()
    sco_rel.sort()  # [1 2 3 4 6 8]
    sco_rel = abs(np.sort(-sco_rel))  # [8 6 4 3 2 1] 先取相反数排序，再加上绝对值得到原数组的降序
    print(order)
    print(len(box_rel))
    box_rel_ordered = box_rel
    for i,item in enumerate(order):
        box_rel_ordered[i]=box_rel[int(item)]
    #sco_rel.sort()  # [1 2 3 4 6 8]
    #sco_rel = abs(np.sort(-sco_rel))
    #index_sort = sco_rel.argsort()
    #index_sort.reverse()
    #print(type(index_sort))
    draw_box(big_img,
             box_rel_ordered,
             cls_rel,
             sco_rel,
             category_index,
             thresh=0.4,
             line_thickness=3)
    plt.imshow(big_img)
    plt.show()
    big_img=big_img.convert("RGB")
    big_img.save("C:/Users/MT/Desktop/big_img_result_ty_new.jpg")
    #print(box_rel)
    #print(type(sco_rel))
    #print(cls_rel)
    #print(category_index)
if __name__ == '__main__':
    main()

