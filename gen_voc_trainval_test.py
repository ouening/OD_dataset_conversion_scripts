'''
Pascal VOC格式数据集生成ImageSets/Main/train.txt,val.txt,trainval.ttx和test.txt
'''
from pathlib import Path
import os
import sys
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages和Annotations这两个文件夹')
    parser.add_argument('--test-ratio',type=float, default=0.2,
        help='验证集比例，默认为0.3')   
    opt = parser.parse_args()

    voc_root = opt.voc_root
    print('Pascal VOC格式数据集路径：', voc_root)

    xml_file = []
    img_files = []
    voc_anno = os.path.join(voc_root, 'Annotations')
    
    voc_jpeg = os.path.join(voc_root, 'JPEGImages')
    
    voc_img_set = os.path.join(voc_root, 'ImageSets')
    try:
        shutil.rmtree(voc_img_set)
    except FileNotFoundError as e:
        a = 1
    mkdir(voc_img_set)

    ImgSetsMain = os.path.join(voc_img_set, 'Main')
    try:
        shutil.rmtree(ImgSetsMain)
    except FileNotFoundError as e:
        a = 1
    mkdir(ImgSetsMain)

    files = [x.stem for x in Path(voc_jpeg).iterdir() if not x.stem.startswith('.')]
    print(files[:10])
    print('>>>随机划分VOC数据集')
    print('数据集长度:',len(files))
    files = shuffle(files)
    ratio = opt.test_ratio
    trainval, test = train_test_split(files, test_size=ratio)
    train, val = train_test_split(trainval,test_size=0.2)
    print('训练集数量: ',len(train))
    print('验证集数量: ',len(val))
    print('测试集数量: ',len(test))

    def write_txt(txt_path, data):
        '''写入txt文件'''
        with open(txt_path,'w') as f:
            for d in data:
                f.write(str(d))
                f.write('\n')
    # 写入各个txt文件
    trainvaltest_txt = os.path.join(ImgSetsMain,'trainvaltest.txt')
    write_txt(trainvaltest_txt, files)
    
    trainval_txt = os.path.join(ImgSetsMain,'trainval.txt')
    write_txt(trainval_txt, trainval)

    train_txt = os.path.join(ImgSetsMain,'train.txt')
    write_txt(train_txt, train)

    val_txt = os.path.join(ImgSetsMain,'val.txt')
    write_txt(val_txt, val)

    test_txt = os.path.join(ImgSetsMain,'test.txt')
    write_txt(test_txt, test)
    