# -*- coding:utf-8 -*-
'''
pascal voc格式数据集转csv格式，适用项目：
1. https://github.com/fizyr/keras-retinanet
2. https://github.com/yhenon/pytorch-retinanet

这种csv格式的数据集更灵活，只要利用不同的机器学习框架提供的数据接口编译一个数据加载类即可，而且csv文本方便处理。
本程序会生成3个文件：train_csv_annotations.csv,val_csv_annotations.csv和csv_classes.csv，
其中train_csv_annotations.csv和val_csv_annotations.csv的内容格式为：
path/to/image.jpg,x1,y1,x2,y2,class_name，例如：

/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
...
注意是每行一个标注信息，上面例子中img_003表示不含需要训练识别的区域(region of interest,ROI)

注意需要绝对路径。
csv_classes.csv的内容格式为：
class_name,id
例如：
cow,0
cat,1
bird,2
...

'''
import csv
import shutil
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
# 获取当前文件所在文件夹
dirname = os.path.dirname(os.path.abspath(__file__))
print('当前工作路径：',dirname)

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

class PascalVOC2CSV(object):
    def __init__(self, voc_root, xml, imgs, ratio, 
                trainvaltest_ann='trainvaltest_csv_annotations.csv',
                trainval_ann='trainval_csv_annotations.csv',
                train_ann='train_csv_annotations.csv',
                val_ann='val_csv_annotations.csv',
                test_ann='test_csv_annotations.csv',
                classes_path='csv_classes.csv', ):
        '''
        :param voc_root: VOC数据集根目录
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param jpgs: 所有图像的文件路径
        :param train_ann_path: 训练集标注信息
        :param val_ann_path: 验证集标注信息
        :param classes_path: classes_path

        返回值：
        在voc_root根目录生成三个文件：train_csv_annotations.csv,val_csv_annotations.csv和csv_classes.csv
        '''
        self.xml = xml
        self.imgs = imgs
        csv_root = os.path.join(voc_root,'CSVDataset')
        create_dir(csv_root)
        self.trainvaltest_ann = os.path.join(csv_root, trainvaltest_ann)
        self.trainval_ann = os.path.join(csv_root, trainval_ann)
        self.train_ann = os.path.join(csv_root, train_ann)
        self.val_ann = os.path.join(csv_root, val_ann)
        self.test_ann = os.path.join(csv_root, test_ann)
        

        self.classes_path = os.path.join(csv_root, classes_path)
        self.label=[]
        self.annotations=[]
        self.ratio = ratio
        self.data_transfer()
        self.write_file()
        self.valid=None
        self.train=None
 
    def data_transfer(self):
        for num, (xml_file, img_file) in enumerate( zip(self.xml, self.imgs)):
            try:
                # print(xml_file)
                # 进度输出
                sys.stdout.write('\r>> Converting image %d/%d' % (
                    num + 1, len(self.xml)))
                sys.stdout.flush()

                xml = open(xml_file,encoding='utf-8')
                tree=ET.parse(xml)
                root = tree.getroot()
                self.filename = img_file

                for obj in root.iter('object'):
                    
                    self.supercategory = obj.find('name').text.strip()
                    if self.supercategory not in self.label:
                        self.label.append(self.supercategory)
                    
                    xmlbox = obj.find('bndbox') # 进一步在bndbox寻找
                    x1 = int(xmlbox.find('xmin').text)
                    y1 = int(xmlbox.find('ymin').text)
                    x2 = int(xmlbox.find('xmax').text)
                    y2 = int(xmlbox.find('ymax').text)
                    assert x1 < x2 and y1 < y2, 'x1 must be less than x2 and y1 must be less than y2'
                    self.annotations.append(
                                [os.path.join(os.path.join(dirname, 'JPEGImages'),self.filename), 
                                x1,y1,x2,y2,
                                self.supercategory])
        
            except:
                continue
        # print(self.annotations[:10])
        # k = int(len(self.annotations) * self.ratio) # ratio是验证集比例
        print('\n按照比例：{:.2f}:{:.2f} 划分训练集和测试集...'.format(1-self.ratio, self.ratio))
        
        self.trainval, self.test = train_test_split(self.annotations, test_size=self.ratio)
        self.train, self.val = train_test_split(self.trainval, test_size=0.2)
        print('训练集数量：', len(self.train))
        print('验证集数量：', len(self.val))
        print('测试集数量：', len(self.test))
        sys.stdout.write('\n')
        sys.stdout.flush()
 
    def write_file(self,):
        print(f'写入全部数据集:{self.trainvaltest_ann}')
        with open(self.trainvaltest_ann, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.annotations)
        print(f'写入训练集:{self.trainval_ann}')
        with open(self.trainval_ann, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.trainval)
        print(f'写入训练集:{self.train_ann}')
        with open(self.val_ann, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.train)
        print(f'写入验证集:{self.train_ann}')
        with open(self.val_ann, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.val)
        print(f'写入测试集:{self.test_ann}')
        with open(self.test_ann, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(self.test)

        class_name=sorted(self.label)
        print('标签名称：', class_name)
       
        print('标签长度：', len(class_name))
        class_=[]
        for num,name in enumerate(class_name):
            class_.append([name,num])
        print(f'写入标签文件:{self.classes_path}...')
        with open(self.classes_path, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, dialect='excel')
            csv_writer.writerows(class_)

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
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages和Annotations这两个文件夹')
    parser.add_argument('--img_dir', type=str, required=False, 
        help='VOC格式数据集图像存储路径，如果不指定，默认为JPEGImages')
    parser.add_argument('--anno_dir', type=str, required=False, 
        help='VOC格式数据集标注文件存储路径，如果不指定，默认为Annotations')
    parser.add_argument('--valid-ratio',type=float, default=0.3,
        help='验证集比例，默认为0.3')   
    opt = parser.parse_args()

    voc_root = opt.voc_root
    print('Pascal VOC格式数据集路径：', voc_root)

    xml_file = []
    img_files = []
    
    if opt.img_dir is None:
        img_dir = 'JPEGImages'
    else:
        img_dir = opt.img_dir
    JPEG = os.path.join(voc_root, img_dir)
    
    if opt.anno_dir is None:
        anno_dir = 'Annotations'
    else:
        anno_dir = opt.anno_dir
    ANNO = os.path.join(voc_root, anno_dir)

    check_files(ANNO, JPEG)

    for k in os.listdir(JPEG):
        '''
        以图片所在路径进行遍历
        '''
        img_files.append( os.path.join(JPEG, k))
        xml_file.append( os.path.join(ANNO, k[:-4]+'.xml'))
    
    PascalVOC2CSV(voc_root, xml_file, img_files, ratio=opt.valid_ratio)
