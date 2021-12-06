'''
Roboflow支持导出voc,yolo,coco等多种数据格式，但是导出的图像名称以及组织形式发生了改变，导致image_id和id名称不一致，在yolov5验证的时候会出现一些问题
本脚本将roboflow导出的Pascal VOC格式数据集转成满足yolov5验证时使用的coco格式数据集（yolov5在训练的时候用yolo格式，但是在验证的时候为了保证指标的
一致性，yolov5还提供了pycocotools接口，这就需要对应的coco格式数据集。

适用项目：
1. yolov5
2. mmdetection2

roboflow导出的voc数据集按照如下方式进行组织（train，test和valid的文件命名不重叠）：
roboflow_vocdata/
    -train/         # 训练集
        -1_jpg.rf.679262612f32c8ad16ce6546f276a1c1.jpg
        -1_jpg.rf.679262612f32c8ad16ce6546f276a1c1.xml
        -2_jpg.rf.1c8ad7446f202205729e6ba1164ee310.jpg
        -2_jpg.rf.1c8ad7446f202205729e6ba1164ee310.xml
        -...
    -test/           # VOC数据集ImageSets
        -3_jpg.rf.0a2534752963e87fe7c0cf4a8de33a9c.jpg
        -3_jpg.rf.0a2534752963e87fe7c0cf4a8de33a9c.xml
        -4_jpg.rf.47e001dc6cbce6aa26b8d12726453f38.jpg
        -4_jpg.rf.47e001dc6cbce6aa26b8d12726453f38.xml
        -...
    -valid/           # VOC数据集图像存储路径
        -5_jpg.rf.7911892930670d44d21ae8db3c525171.jpg
        -5_jpg.rf.7911892930670d44d21ae8db3c525171.xml
        -6_jpg.rf.bf0a49d44a8e3c631c7d0050dfd406ac.jpg
        -6_jpg.rf.bf0a49d44a8e3c631c7d0050dfd406ac.xml
        -...
    -README.roboflow.txt

可以看出文件命名已经被改变，pycocotools对coco格式的标注信息限制比较多，这种命名形式下image_id和id名称不一致，在调用相关api的时候会报错。


转换后的coco数据集组织形式为：
├───annotations
├──────instances_train.json
├──────instances_test.json
├──────instances_val.json
├───test
├──────3_jpg.rf.0a2534752963e87fe7c0cf4a8de33a9c.jpg
├──────...
├───train
├──────1_jpg.rf.679262612f32c8ad16ce6546f276a1c1.jpg
├──────...
└───val
├──────5_jpg.rf.7911892930670d44d21ae8db3c525171.jpg
├──────...

'''
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:19:29 2021

@author: gaoya
"""
import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
# check_requirements(['pycocotools'])
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import shutil

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

def ToPascalFormat(roboflow_voc_root):
    # roboflow_voc_root = r"D:\Datasets\cotterpins\cotterpin.v1-640_mosaic_3x.voc"
    root = Path(roboflow_voc_root)

    create_dir(os.path.join(roboflow_voc_root, ''))
    imgs_dict = {'train':[], 'test':[], 'valid':[]}
    xmls_dict = {'train':[], 'test':[], 'valid':[]}

    for subdir in root.iterdir():
        if subdir.is_dir():
            for file in tqdm(subdir.iterdir()):
                if file.suffix == '.xml':
                    print(file)
                    xmls_dict[subdir.stem].append(str(file))
            
                if file.suffix == '.jpg':
                    print(file)
                    imgs_dict[subdir.stem].append(str(file))
    #%%
    imgs = imgs_dict['train'] + imgs_dict['test'] + imgs_dict['valid']
    xmls = xmls_dict['train'] + xmls_dict['test'] + xmls_dict['valid']

    img_names = [ Path(k).stem for k in imgs]
    xml_names = [ Path(k).stem for k in xmls]
    assert (img_names==xml_names)
    # 映射字典
    map_dict = {k:v for (v,k) in enumerate(img_names)}

