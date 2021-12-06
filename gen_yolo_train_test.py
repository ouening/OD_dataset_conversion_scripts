'''
YOLO格式数据集生成train.txt,val.txt,trainval.ttx和test.txt
'''
from pathlib import Path
import os
import sys
# from voc2coco import voc_root
import xml.etree.ElementTree as ET
import random
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import shutil

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

def write_txt(txt_path, data):
    '''写入txt文件'''
    with open(txt_path,'w') as f:
        for d in data:
            f.write(str(d))
            f.write('\n')
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-root', type=str, required=True, 
        help='YOLO格式数据集根目录，该目录下必须包含images和labels这两个文件夹')
    parser.add_argument('--from_voc',type=bool, default=False, 
        help='从VOC数据集中的ImageSets/Main文件夹下提取')  
    parser.add_argument('--voc-root',type=str,
        help='VOC数据集路径，需要包含ImageSets/Main文件夹')  
    parser.add_argument('--test-ratio',type=float, default=0.2,
        help='测试集比例，默认为0.2')  
    parser.add_argument('--ext', type=str, default='.png', 
        help='YOLO图像数据后缀，注意带"." ' ) 
    opt = parser.parse_args()

    yolo_root = opt.yolo_root
    print('YOLO格式数据集路径：', yolo_root)

    ANNO = os.path.join(yolo_root, 'labels')
    JPEG = os.path.join(yolo_root, 'images')
    
    if opt.from_voc:
        print('从VOC数据集中分割数据集')
        if not opt.voc_root:
            raise Exception('需要提供VOC格式路径')
        voc_root = opt.voc_root
        voc_sets = os.path.join(voc_root,'ImageSets/Main')
        if not os.path.exists(voc_sets):
            raise Exception('VOC数据集不存在ImageSets/Main路径')
        else:
            file_lists = list(Path(voc_sets).iterdir())
            for file in file_lists:
                img_ids = [x.strip() for x in open(file,'r').readlines()]
                img_full_path = [os.path.join(JPEG, img_id+opt.ext) for img_id in img_ids]
                file_to_write = os.path.join(yolo_root,file.name)
                write_txt(file_to_write, img_full_path)
    else:
        print('从YOLO数据集中按比例随机分割数据集')
        p = Path(JPEG)
        files = []
        for file in p.iterdir():
            # name,sufix = file.name.split('.')
            if file.name.split('.')[1]==opt.ext[1:]:
                files.append(str(file))
            # print(name, sufix)
        print('数据集长度:',len(files))
        files = shuffle(files)
        ratio = opt.test_ratio
        trainval, test = train_test_split(files, test_size=ratio)
        train, val = train_test_split(trainval, test_size=0.2)
        print('训练集数量: ',len(train))
        print('验证集数量: ',len(val))
        print('测试集数量: ',len(test))

        
        # 写入各个txt文件
        trainval_txt = os.path.join(yolo_root,'trainval.txt')
        write_txt(trainval_txt, trainval)

        train_txt = os.path.join(yolo_root,'train.txt')
        write_txt(train_txt, train)

        val_txt = os.path.join(yolo_root,'val.txt')
        write_txt(val_txt, val)

        test_txt = os.path.join(yolo_root,'test.txt')
        write_txt(test_txt, test)
    