'''
计算Pascal voc格式数据集的相关信息，包括：
1）各类别数量，可视化结果
2）各图片长宽大小，平均大小，可视化结果
3）锚框长宽大小，平均大小，可视化结果
4）锚框k-means聚类结果
5）各锚框占该锚框所在图中的面积比，可视化结果
6）无参考系的图像质量评估

'''

import xml.etree.ElementTree as ET
import pickle
import os
import sys
from os import listdir, getcwd
from os.path import join
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 13
plt.rc('axes', unicode_minus=False)
plt.rc('axes', unicode_minus=False)
# plt.style.use(['science','ieee'])
from collections import Counter
from pathlib import Path
import argparse
from tqdm import tqdm
import seaborn as sns
from matplotlib.transforms import Bbox
from imageio import imread

def load_dataset(xml_list, anno_root, savefig=True,img_name=''):
    '''计算标签
    xml_list: xml标注文件名称列表
    anno_root: 标注文件路径
    '''
    xml_info = []
    length = len(xml_list)
    for idx, xml_file in enumerate(xml_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (
                    idx+1, length))
        sys.stdout.flush()
        # print(xml_file)
        xml = open(xml_file, encoding='utf-8')        
        tree=ET.parse(xml)
        root = tree.getroot()
        try:
            filename = root.find('filename').text
        except:
            filename = root.find('frame').text

        try:
            # 图片高度
            height = int(root.findtext("./size/height"))
            # 图片宽度
            width = int(root.findtext("./size/width"))
        except:
            img_path = Path(anno_root).parent.joinpath('JPEGImages', filename+img_suffix)         
            width, height = imread(str(img_path)).shape[:2]
        
        for obj in root.iter('object'):
            xmin = int(float(obj.findtext("bndbox/xmin")))
            ymin = int(float(obj.findtext("bndbox/ymin")))
            xmax = int(float(obj.findtext("bndbox/xmax")))+1
            ymax = int(float(obj.findtext("bndbox/ymax")))+1
            assert xmin < xmax and ymin < ymax, 'xmin, ymin, xmax, ymax must be in ascending order'
            value = (
                filename,
                width,
                height,
                obj.find('name').text.strip(),
                xmin,
                ymin,
                xmax,
                ymax
            )
            if value[3]=='paches':
                print(xml_file)
            xml_info.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    xml_df = pd.DataFrame(xml_info, columns=column_name)
    xml_df['box_width'] = xml_df['xmax']-xml_df['xmin']
    xml_df['box_height'] = xml_df['ymax'] - xml_df['ymin']
    
    def color(df):
        if df['class']=='lost':
            return 0
        else:
            return 1
    def area(df):
        return (df['box_width']*df['box_height'])/(df['width']*df['height'])
    xml_df['box_area'] = xml_df.apply(area, axis=1)
    
    count = dict(Counter(xml_df['class']))
    print()
    print('标签类别：\n', np.unique(xml_df['class']))
    print('标签名称及数量：\n', xml_df['class'].value_counts())
    print('总标签数量：', len(count))
        
    ## 图像大小
    print('平均图像大小(宽度×高度)：{:.2f}×{:.2f}'.format(np.mean(xml_df['width']), 
        np.mean(xml_df['height'])))
    ## 锚框大小
    print('平均锚框大小(宽度×高度)：{:.2f}×{:.2f}'.format(np.mean(xml_df['xmax']-xml_df['xmin']), 
        np.mean(xml_df['ymax']-xml_df['ymin'])))

    df_group = xml_df.groupby('class')
    for cls, df in df_group:
        print('类别：', cls)
        print('平均锚框大小(宽度×高度)：{:.2f}×{:.2f}'.format(np.mean(df['xmax']-df['xmin']), 
        np.mean(df['ymax']-df['ymin'])))

    if savefig:
        
        plt.figure(figsize=(9,9))
        plt.subplot(2,2,1)
        count = dict(Counter(xml_df['class']))
        classes = list(count.keys())
        df = pd.Series(list(count.values()), index=count.keys())
        df = df.sort_values(ascending=True)
        df.plot(kind='bar',alpha=0.75, rot=0)
        # plt.xticks(rotation=90)
        plt.ylabel('number of instances')
        # plt.title('Distribution of different classes')

        plt.subplot(2,2,2)
        # plt.hist(xml_df['box_area']*100, bins=100,)
        # x直方图
        plt.hist((xml_df['xmax']+xml_df['xmin'])/2.0/xml_df['width'], bins=50,)
        # plt.title('Histogram plot of x')
        plt.xlabel('x')
        plt.ylabel('frequency')

        plt.subplot(2,2,3)
        plt.hist((xml_df['ymax']+xml_df['ymin'])/2.0/xml_df['height'], bins=50, )
        # plt.title('Histogram Plot of Box Areas')
        plt.xlabel('y')
        plt.ylabel('frequency')

        plt.subplot(2,2,4)
        for c in classes:
            df_ = xml_df[xml_df['class']==c][['box_width','box_height']]
            plt.scatter(df_['box_width']/xml_df['width'], df_['box_height']/xml_df['height'],label=c)

        plt.xlabel('w')
        plt.ylabel('h')
        # plt.title('Scatter Plot of Boxes')
        plt.legend(loc='best')

        plt.savefig(os.path.join(voc_stat,f'{img_name}_output.png'), dpi=800,bbox_inches='tight', pad_inches=0.0)


    return xml_df

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=False, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages,Annotationshe ImageSets这3个文件夹，\
            在ImageSets文件夹下还要有Main/trainval.txt等文件')
    parser.add_argument('--img_dir', type=str, required=False, 
        help='VOC格式数据集图像存储路径，如果不指定，默认为JPEGImages')
    parser.add_argument('--anno_dir', type=str, required=False, 
        help='VOC格式数据集标注文件存储路径，如果不指定，默认为Annotations')

    opt = parser.parse_args()

    voc_root = opt.voc_root

    # check image root and annotation root
    if opt.img_dir is None:
            img_dir = 'JPEGImages'
    else:
        img_dir = opt.img_dir
    voc_jpeg = os.path.join(voc_root, img_dir)
    if not os.path.exists(voc_jpeg):
        raise Exception(f'数据集图像路径{voc_jpeg}不存在！')

    if opt.anno_dir is None:
        anno_dir = 'Annotations'
    else:
        anno_dir = opt.anno_dir
    voc_anno = os.path.join(voc_root, anno_dir)  # 
    if not os.path.exists(voc_anno):
        raise Exception(f'数据集图像路径{voc_anno}不存在！')
    # check image set root
    if not os.path.exists(os.path.join(voc_root,'ImageSets/Main')):
        raise Exception("$VOC_ROOT/ImageSets/Main doesn't exist, please generate them using script voc2coco.py")
    
    voc_stat = os.path.join(voc_root, 'VOC统计信息')
    create_dir(voc_stat)

    anno_root = os.path.join(voc_root,'Annotations')
    # image suffix
    img_suffix = set([x.suffix for x in Path(voc_jpeg).iterdir()])
    if len(img_suffix) != 1:
        raise Exception('VOC数据集中JPEGImages文件夹中的图片格式不一致')
    img_suffix = img_suffix.pop()

    print("=========统计所有数据信息============")
    all_xml_list = list(Path(anno_root).iterdir())
    df = load_dataset(all_xml_list, anno_root)
    df.to_csv(os.path.join(voc_stat,'all_info.csv'), index=False)

    for data_type in ['train', 'trainval', 'val', 'test']:
        print(f"\n\n=========统计{data_type}数据信息============")
        txt = os.path.join(voc_root, f'ImageSets/Main/{data_type}.txt')
        if not os.path.exists(txt):
            print(f'文件ImageSets/Main/{data_type}.txt不存在!')
            continue
        # xml_files = [x.strip() for x in open(txt,'r').readlines()]
        xml_files = [x.replace('\n','') for x in open(txt,'r').readlines()]
        xml_list = [os.path.join(anno_root, xml_name+'.xml') for xml_name in xml_files]
        df = load_dataset(xml_list, anno_root, savefig=True, img_name=data_type)
        df.to_csv(os.path.join(voc_stat,f'{data_type}_info.csv'), index=False)
