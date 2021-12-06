'''
YOLO数据集扩增
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

ia.seed(1)


def yolo_to_voc_format(x_center, y_center, width, height, img_width, img_height):
    '''将yolo格式转换为voc格式，注意输入的x_center, y_center, width, height是str类型'''

    center_x = round(float(str(x_center).strip()) * img_width)
    center_y = round(float(str(y_center).strip()) * img_height)
    bbox_width = round(float(str(width).strip()) * img_width)
    bbox_height = round(float(str(height).strip()) * img_height)

    xmin = int(center_x - bbox_width / 2 )
    ymin = int(center_y - bbox_height / 2)
    xmax = int(center_x + bbox_width / 2)
    ymax = int(center_y + bbox_height / 2)

    return xmin,ymin,xmax,ymax


def voc_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    '''将voc格式转换为yolo格式'''
    dw = 1./(img_width) # 宽度缩放比例, size[0]为图像宽度width
    dh = 1./(img_height)
    x = (xmin + xmax)/2.0 - 1
    y = (ymin + ymax)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x_center = x*dw
    width = w*dw
    y_center = y*dh
    height = h*dh

    return x_center, y_center, width, height

def read_yolo_annotation(img_root, label_root, image_id):
    '''
    root：一般是labels, txt文件, <object-class> <x_center> <y_center> <width> <height>
    image_id是包含.txt后缀的文件名

    return:
        [xmin, ymin, xmax, ymax, label]
    '''
    img_width, img_height = Image.open(os.path.join(img_root, image_id[:-4] + ext)).size

    annos = [x for x in open(os.path.join(label_root, image_id)).readlines()]
    bndboxlist = []   # 存储标注框信息

    for anno in annos:  # 找到root节点下的所有country节点
        lb, x_center, y_center, width, height = anno.split(' ') # 注意得到的是str类型

        xmin,ymin,xmax,ymax = yolo_to_voc_format(x_center, y_center, width, height, img_width, img_height)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax, int(lb)])
        # print(bndboxlist)

    return bndboxlist

def change_yolo_annotation(img_root, label_root, image_id, new_target, saveroot, id):
    '''保存新的yolo标注
    img_root: 原数据集图像路径
    label_root: 原数据标注文件路径
    image_id：文件名，带.txt后缀
    new_target: new_bndbox_list:[[x1,y1,x2,y2,label],...[],[]]
    saveroot: 扩增后标注文件保存路径
    id: 扩增后保存标注文件名
    '''
    img_width, img_height = Image.open(os.path.join(img_root, image_id[:-4] + ext)).size

    new_annos = []
    for anno in new_target:
        xmin, ymin, xmax, ymax, label = anno
        x_center, y_center, width, height = voc_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height)
        label = str(label)
        x_center = str(x_center)
        y_center = str(y_center)
        width = str(width)
        height = str(height)+'\n'

        new_annos.append(' '.join((label,x_center,y_center, width, height)))

    # 存储新标注
    with open(os.path.join(saveroot, str("%d" % int(id)) + '.txt'),'w') as f:
        f.writelines(new_annos)


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
    parser.add_argument('--yolo-root', type=str, required=True, 
        help='YOLO格式数据集根目录，该目录下必须包含images和labels这两个文件夹')
    parser.add_argument('--aug-dir',type=str, default='YOLOAugumented',
        help='数据增强后保存的路径，默认在voc-root路径下创建一个文件夹YOLOAugumented进行保存')
    parser.add_argument('--aug_num',type=int, default=5,
        help='每张图片进行扩增的次数')   
    parser.add_argument('--ext', type=str, default='.png', help='图像后缀，默认为.png')

    opt = parser.parse_args()
    ext = opt.ext

    IMG_DIR = os.path.join(opt.yolo_root, "images")
    LABEL_DIR = os.path.join(opt.yolo_root, "labels")

    AUGUMENT = os.path.join(opt.yolo_root, opt.aug_dir)
    if not os.path.exists(AUGUMENT):
        os.mkdir(AUGUMENT)

    AUG_IMG_DIR = os.path.join(AUGUMENT, "images")  # 存储增强后的影像文件夹路径 
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    AUG_LABEL_DIR = os.path.join(AUGUMENT, "labels")  # 存储增强后的XML文件夹路径
    try:
        shutil.rmtree(AUG_LABEL_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_LABEL_DIR)

    AUGLOOP = opt.aug_num  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    number = 1
    for root, sub_folders, files in os.walk(LABEL_DIR):
        for name in tqdm(files):
            # [xmin, ymin, xmax, ymax, int(lb)]
            bndbox = read_yolo_annotation(IMG_DIR, LABEL_DIR, name) # 读取yolo标注文件，name是包含.txt后缀的文件名，注意：## 返回voc格式标注框 ##
            shutil.copy(os.path.join(LABEL_DIR, name), AUG_LABEL_DIR) # 复制标注文件
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + ext), AUG_IMG_DIR) # 复制图像

            for epoch in range(AUGLOOP):
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

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
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
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2, bndbox[i][4]]) # 注意保存标签

                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR,
                                    str("%d" % (len(files) + number)) + ext)
                # image_auged = bbs.draw_on_image(image_aug, size=0)
                Image.fromarray(image_aug).save(path)

                # 存储变化后的yolo标注文件
                change_yolo_annotation(img_root=IMG_DIR, 
                                        label_root=LABEL_DIR, 
                                        image_id=name,  # 带后缀
                                        new_target=new_bndbox_list, 
                                        saveroot=AUG_LABEL_DIR,
                                        id=len(files) + number)

                # print(str("%d" % (len(files) + number)) + '.png')
                number = number + 1
                new_bndbox_list = []