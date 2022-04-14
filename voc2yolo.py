'''
PASCAL VOC格式数据集转YOLO格式数据集
适合项目地址：
1. https://github.com/eriklindernoren/PyTorch-YOLOv3
2. https://github.com/ultralytics/yolov3/
3. https://github.com/AlexeyAB
4. https://github.com/ultralytics/yolov5/

该项目对自定义的数据集格式要求图片要有对应的txt格式标注文件，要求图片存放在images文件夹，标签存放在labels文件夹，例如：

data/custom/images/train.jpg
data/custom/labels/train.txt
yolo_classes.names
yolo_classes_ssd.names
trainval.txt
train.txt
val.txt

当然，images文件夹和labels这两个文件夹名称可以更改，但相应的也要在代码中做修改（PyTorch-YOLOV3项目）：
```utils/datasets.py: line 65
class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            ##            ^^^^^^ and ^^^^^^ 修改这两处的值
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
    ...
```
labels/train.txt的标注信息格式为：

label_idx x_center y_center width height（归一化数值）
label_idx x_center y_center width height（归一化数值）
...

trainval.txt,val.txt,test.txt文件每一行记录了图像数据所在的全路径，这几个文件和yolo_classes.names
会在U版和A版的YOLOv3/v4系列的*.data配置文件中使用。在U版的yolov5模型中，数据配置文件保存在data/*.yaml文件中，其示例内容如下：
```
# train and val data as 
# 1) directory: path/images/, 
# 2) file: path/images.txt, or 
# 3) list: [path1/images/, path2/images/]

train: /data/custom_yolo/trainval.txt
val: /data/custom_yolo/test.txt

# number of classes
nc: 2

# class names
names: ['person', 'bicycle']
```        
'''
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import pandas as pd
import numpy as np
from collections import Counter
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import shutil
from pathlib import Path
from imageio import imread

def counting_labels(anno_root,yolo_root):
    '''
    获取pascal voc格式数据集中的所有标签名
    anno_root: pascal标注文件路径，一般为Annotations
    '''
    all_classes = []
    
    for xml_file in os.listdir(anno_root):
        xml_file = os.path.join(anno_root, xml_file)
        # print(xml_file)
        xml = open(xml_file,encoding='utf-8')
        tree=ET.parse(xml)
        root = tree.getroot()
        for obj in root.iter('object'):
            
            class_ = obj.find('name').text.strip()
            all_classes.append(class_)
    
    print(Counter(all_classes))

    labels = list(set(all_classes))
    print('标签数据：', labels)
    print('标签长度：', len(labels))
    print('写入标签信息...{}'.format(os.path.join(yolo_root,'yolo_classes.names')))
    with open( os.path.join(yolo_root,'yolo_classes.names') , 'w') as f:
        for k in labels:
            f.write(k)
            f.write('\n')
    with open( os.path.join(yolo_root,'yolo_classes_ssd.names') , 'w') as f:
        for k in labels:
            f.write("\'"+k+"\'"+',')
            f.write('\n')
    return labels


def convert(size, box):
    dw = 1./(size[0]) # 宽度缩放比例, size[0]为图像宽度width
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h) # <x_center> <y_center> <width> <height>

def convert_annotation(anno_root:str, image_id, classes, dest_yolo_dir='YOLOLabels'):
    '''
    anno_root:pascal格式标注文件路径，一般为Annotations
    image_id：文件名（图片名和对应的pascal voc格式标注文件名是一致的）
    dest_yolo_dir：yolo格式标注信息目标保存路径，默认为opt.yolo_dir
    '''
    in_file = open( os.path.join(anno_root, image_id+'.xml'), encoding='utf-8')
    out_file = open(os.path.join(dest_yolo_dir, image_id+'.txt'), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    try:
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    except:
        img_path = Path(anno_root).parent.joinpath('JPEGImages', image_id+img_suffix)
        w,h = imread(img_path).shape[:2]

    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin,xmax,ymin,ymax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
        assert xmin<xmax and ymin<ymax and xmin>=0 and ymin>=0, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        b = (xmin,xmax,ymin,ymax)
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def gen_image_ids(jpeg_root):
    '''
    jpeg_root: JPEGImages文件夹路径
    '''
    img_ids = []

    for k in os.listdir(jpeg_root):
        img_ids.append(k) # 图片名，含后缀
    
    return img_ids

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

def check_files(ann_root, img_root):
    '''检测图像名称和xml标准文件名称是否一致，检查图像后缀'''
    if os.path.exists(ann_root):
        ann = Path(ann_root)
    else:
        raise Exception("标注文件路径错误")
    if os.path.exists(img_root):
        img = Path(img_root)
    else:
        raise Exception("图像文件路径错误")
    ann_files = []
    img_files = []
    img_exts = []
    for an, im in zip(ann.iterdir(),img.iterdir()):
        ann_files.append(an.stem)
        img_files.append(im.stem)
        img_exts.append(im.suffix)

    print('图像后缀列表：', np.unique(img_exts))
    if len(np.unique(img_exts)) > 1:
        # print('数据集包含多种格式图像，请检查！', np.unique(img_exts))
        raise Exception('数据集包含多种格式图像，请检查！', np.unique(img_exts))
    if set(ann_files)==set(img_files):
        print('标注文件和图像文件匹配')
    else:
        print('标注文件和图像文件不匹配')
    
    return np.unique(img_exts)[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含存储图像和标注文件的两个文件夹')
    parser.add_argument('--img_dir', type=str, required=False, 
        help='VOC格式数据集图像存储路径，如果不指定，默认为JPEGImages')
    parser.add_argument('--anno_dir', type=str, required=False, 
        help='VOC格式数据集标注文件存储路径，如果不指定，默认为Annotations')
    parser.add_argument('--yolo-dir',type=str, default='YOLOFormatData',
        help='yolo格式数据集保存路径，默认为VOC数据集相同路径下新建文件夹YOLODataset')
    parser.add_argument('--valid-ratio',type=float, default=0.3,
        help='验证集比例，默认为0.3')   
 
    opt = parser.parse_args()

    voc_root = opt.voc_root

    print('Pascal VOC格式数据集路径：', voc_root)
    if opt.img_dir is None:
        img_dir = 'JPEGImages'
    else:
        img_dir = opt.img_dir
    jpeg_root = os.path.join(voc_root, img_dir)
    if not os.path.exists(jpeg_root):
        raise Exception(f'数据集图像路径{jpeg_root}不存在！')
    
    if opt.anno_dir is None:
        anno_dir = 'Annotations'
    else:
        anno_dir = opt.anno_dir
    anno_root = os.path.join(voc_root,anno_dir)
    if not os.path.exists(anno_root):
        raise Exception(f'数据集图像路径{anno_root}不存在！')
    
    # 确定图像后缀
    img_suffix = check_files(anno_root, jpeg_root)
    assert img_suffix is not None, "请检查图像后缀是否正确！"
    print('图像后缀：', img_suffix)
    #  YOLO数据集存储路径，YOLOFormat
    dest_yolo_dir = os.path.join(str(Path(voc_root).parent), Path(voc_root).stem+opt.yolo_dir)
    # 
    image_ids = [x.name for x in Path(jpeg_root).iterdir()]

    print('数据集长度：', len(image_ids))

    if not os.path.exists(dest_yolo_dir):
        os.makedirs(dest_yolo_dir)    # 创建labels文件夹,存储yolo格式标注文件

    yolo_labels = os.path.join(dest_yolo_dir,'labels')
    create_dir(yolo_labels)
    yolo_images = os.path.join(dest_yolo_dir,'images')
    create_dir(yolo_images)

    classes = counting_labels(anno_root,dest_yolo_dir)
    print('数据类别：', classes)
    length = len(image_ids)

    for idx, img in enumerate(image_ids):
        sys.stdout.write('\r>> Converting image %d/%d' % (
                    idx + 1, length))
        sys.stdout.flush()
        image_id = img.split('.')[0]
        # print(image_id)
#        print('图像名称：', image_id)
        # 转换标签
        convert_annotation(anno_root, image_id, classes, dest_yolo_dir=yolo_labels)

        shutil.copy(os.path.join(voc_root, 'JPEGImages', img), yolo_images)

    ## 生成用于config/custom.data指定的训练训练集和验证集文件yolo_train.txt和yolo_valid.txt
    # 该文件的内容就是每行为图片数据在文件系统中的绝对路径
    
    ratio = opt.valid_ratio     # 验证集比例
    def write_txt(txt_path, data):
            '''写入txt文件'''
            with open(txt_path,'w') as f:
                for d in data:
                    f.write(str(d))
                    f.write('\n')
    
    # 所有yolo images名称
    files = [x.stem for x in Path(yolo_images).iterdir() if not x.stem.startswith('.')]

    print('数据集长度:',len(files))
    assert os.path.exists(os.path.join(voc_root, 'ImageSets/Main/trainval.txt'))
    if os.path.exists(os.path.join(voc_root, 'ImageSets/Main/trainval.txt')):
        print('\n使用Pascal VOC ImageSet信息分割数据集')
        trainval_file = os.path.join(voc_root, 'ImageSets/Main/trainval.txt')
        trainval_name = [i.strip() for i in open(trainval_file,'r').readlines()]
        trainval = [os.path.join(yolo_images,name+img_suffix) for name in trainval_name]

        train_file = os.path.join(voc_root, 'ImageSets/Main/train.txt')
        train_name = [i.strip() for i in open(train_file,'r').readlines()]
        train = [os.path.join(yolo_images,name+img_suffix) for name in train_name]

        val_file = os.path.join(voc_root, 'ImageSets/Main/val.txt')
        val_name = [i.strip() for i in open(val_file,'r').readlines()]
        val = [os.path.join(yolo_images,name+img_suffix) for name in val_name]

        test_file = os.path.join(voc_root, 'ImageSets/Main/test.txt')
        test_name = [i.strip() for i in open(test_file,'r').readlines()]
        test = [os.path.join(yolo_images,name+img_suffix) for name in test_name]
        
        print('训练集数量: ',len(train_name))
        print('训练集验证集数量: ',len(trainval_name))
        print('验证集数量: ',len(val_name))
        print('测试集数量: ',len(test_name))

    else:
        print('\n随即划分YOLO数据集')

        trainval, test = train_test_split(files, test_size=ratio)
        train, val = train_test_split(trainval,test_size=0.2)
        print('训练集数量: ',len(train))
        print('验证集数量: ',len(val))
        print('测试集数量: ',len(test))

    # 写入各个txt文件
    trainval_txt = os.path.join(dest_yolo_dir,'trainval.txt')
    write_txt(trainval_txt, trainval)

    train_txt = os.path.join(dest_yolo_dir,'train.txt')
    write_txt(train_txt, train)

    val_txt = os.path.join(dest_yolo_dir,'val.txt')
    write_txt(val_txt, val)

    test_txt = os.path.join(dest_yolo_dir,'test.txt')
    write_txt(test_txt, test)
