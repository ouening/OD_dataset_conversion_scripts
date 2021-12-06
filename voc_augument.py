'''
VOC格式数据集离线扩增
参考链接：
[1] https://zhuanlan.zhihu.com/p/85292901
[2] 
'''
import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa
import argparse
sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，
ia.seed(1)

def read_xml_annotation(root, image_id):
    '''从xml标注文件所在路径读取标注框
    root: xml文件路径
    image_id: xml文件名称（带.xml后缀）
    '''

    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    return bndboxlist


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    '''
    root: 原始voc数据集中xml文件所在路径
    image_id：xml文件名，带后缀
    new_target: 新的bndbox:[[x1,y1,x2,y2],...[],[]]
    saveroot: 扩增数据集后xml文件的保存路径
    id: 新的xml标注文件名名称（不含后缀）
    '''
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    elem.text = (str("%d" % int(id)) + '.jpg')
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)  # 替换原来的值
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str("%d" % int(id)) + '.xml'))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages和Annotations这两个文件夹')
    parser.add_argument('--aug-dir',type=str, default='VOCAugumented',
        help='数据增强后保存的路径，默认在voc-root路径下创建一个文件夹Augument进行保存')
    parser.add_argument('--aug_num',type=int, default=3,
        help='每张图片进行扩增的次数')   
    parser.add_argument('--ext', type=str, default='.jpg', help='图像后缀，默认为.jpg')
    opt = parser.parse_args()

    ext = opt.ext
    IMG_DIR = os.path.join(opt.voc_root, "JPEGImages")
    XML_DIR = os.path.join(opt.voc_root, "Annotations")

    AUGUMENT = os.path.join(os.path.dirname(opt.voc_root), opt.aug_dir)
    if not os.path.exists(AUGUMENT):
        os.mkdir(AUGUMENT)

    # 存储增强后的XML文件夹路径
    AUG_XML_DIR = os.path.join(AUGUMENT, "Annotations")  
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = os.path.join(AUGUMENT, "JPEGImages")  
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUGLOOP = opt.aug_num  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强
    '''
    seq = iaa.Sequential([
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
    '''
    seq = iaa.Sequential([
                iaa.Sometimes(p=0.5,
                              #高斯模糊
                              then_list=[iaa.GaussianBlur(sigma=(0, 0.5))],
                              #锐化
                              else_list=[iaa.ContrastNormalization((0.15, 0.75), per_channel=True)]
                              ),  #以p的概率执行then_list的增强方法，以1-p的概率执行else_list的增强方法，其中then_list,else_list默认为None

                iaa.SomeOf(4,[
                    # 以下一共10个，随机选7个进行处理，也可以将7改为其他数值，继续对数据集进行扩充

                    # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                    # sometimes(iaa.OneOf([
                    #     iaa.EdgeDetect(alpha=(0, 0.7)),
                    #     iaa.DirectedEdgeDetect(
                    #         alpha=(0, 0.7), direction=(0.0, 1.0)
                    #     ),
                    # ])),

                    # 将RGB变成灰度图然后乘alpha加在原图上
                    # iaa.Grayscale(alpha=(0.0, 1.0)),

                    # 扭曲图像的局部区域
                    sometimes(iaa.PiecewiseAffine(scale=(0.001, 0.005))),

                    # 每个像素随机加减-10到10之间的数
                    iaa.Add((-10, 10), per_channel=0.5),

                    # 中值模糊
                    iaa.MedianBlur(k=1, name=None, deterministic=False, random_state=None),
                    #锐化
                    iaa.Sharpen(alpha=0, lightness=1, name=None, deterministic=False, random_state=None),
                    # 从最邻近像素中取均值来扰动。
                    iaa.AverageBlur(k=1, name=None, deterministic=False, random_state=None),
                    # 0-0.05的数值，分别乘以图片的宽和高为剪裁的像素个数，保持原尺寸
                    # iaa.Crop(percent=(0.01, 0.01)),

                    # iaa.Affine(
                    #     # 对图片进行仿射变化，scale缩放x,y取值，translate_percent左右上下移动
                    #     # rotate为旋转角度，shear为剪切取值范围0-360
                    #     scale={"x": (0.99, 1), "y": (0.99, 1)},
                    #     translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    #     rotate=(-1, 1),
                    #     shear=(-1, 1)),

                    # 20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # 随机去掉一些像素点, 即把这些像素点变成0。
                    iaa.Dropout(p=0, per_channel=False, name=None, deterministic=False, random_state=None),
                    # 浮雕效果
                    # iaa.Emboss(alpha=0, strength=2, name=None, deterministic=False, random_state=None),
                    # loc 噪声均值，scale噪声方差，50%的概率，对图片进行添加白噪声并应用于每个通道
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.3)],
                          ),
                ], random_order=True)  # 打乱定义图像增强的顺序

    number = 1 # 在原来图像数量基础上叠加
    for root, sub_folders, files in os.walk(XML_DIR):

        for name in tqdm(files):

            bndbox = read_xml_annotation(XML_DIR, name)
            # 首先将原来的图像和对应的标注文件复制到扩增目录下
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + ext), AUG_IMG_DIR)

            for epoch in range(AUGLOOP): # 在每个扩增循环里处理
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + ext))
                # sp = img.size
                img = np.asarray(img)
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0] # 对bbox扩增
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0] # 对图像进行扩增
                path = os.path.join(AUG_IMG_DIR,
                                    str("%d" % (len(files) + number)) + ext)

                # image_auged = bbs.draw_on_image(image_aug, size=0)
                Image.fromarray(image_aug).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                           len(files) + number)
                # print(str("%d" % (len(files) + number)) + ext)
                number += 1
                new_bndbox_list = []