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
from collections import Counter
from pathlib import Path
import argparse
from tqdm import tqdm
import seaborn as sns

def load_dataset(xml_list, anno_root, savefig=True):
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
        xml = open(xml_file, encoding='utf-8')        
        tree=ET.parse(xml)
        root = tree.getroot()
        # 图片高度
        height = int(root.findtext("./size/height"))
      
        # 图片宽度
        width = int(root.findtext("./size/width"))
    
        filename = root.find('filename').text
        for obj in root.iter('object'):
            
            value = (
                filename,
                width,
                height,
                obj.find('name').text.strip(),
                int(obj.findtext("bndbox/xmin")),
                int(obj.findtext("bndbox/ymin")),
                int(obj.findtext("bndbox/xmax")),
                int(obj.findtext("bndbox/ymax"))
            )
            if value[3]=='paches':
                print(xml_file)
            xml_info.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    xml_df = pd.DataFrame(xml_info, columns=column_name)
    xml_df['anchor_width'] = xml_df['xmax']-xml_df['xmin']
    xml_df['anchor_height'] = xml_df['ymax'] - xml_df['ymin']
    
    def color(df):
        if df['class']=='lost':
            return 0
        else:
            return 1
    def area(df):
        return (df['anchor_width']*df['anchor_height'])/(df['width']*df['height'])
    xml_df['anchor_area'] = xml_df.apply(area, axis=1)
    # xml_df['color'] = xml_df.apply(color, axis=1)

    count = dict(Counter(xml_df['class']))
    print('\n标签名称及数量：\n', xml_df['class'].value_counts())
    print('总标签数量：', len(count))
    if savefig:
        # plt.figure(figsize=(12, 15))
        plt.figure()
        df = pd.Series(list(count.values()), index=count.keys())
        df = df.sort_values(ascending=True)
        df.plot.barh()
        file_path = os.path.join(voc_stat,'各类别数量占比.png')
        print('图像存储路径：',file_path)
        plt.xlabel('number of instances')
        plt.title('Distribution of different classes')
        plt.savefig( file_path, dpi=300,bbox_inches = 'tight')
        print('保存类别数量分布结果')
        # plt.show()

        plt.figure()
        sns.distplot(a=xml_df['anchor_area'], bins=100, hist=True, kde=True, rug=False, )
        plt.title('Histogram Plot of Bounding Boxes')
        plt.savefig( os.path.join(voc_stat,'锚框面积占比.png'), dpi=300,bbox_inches = 'tight')
        print('保存锚框面积占比结果')
        # plt.show()

        plt.figure()
        sns.scatterplot(x='anchor_width', y='anchor_height', hue='class', data=xml_df)
        plt.title('Scatter Plot of Bounding Boxes')
        plt.savefig( os.path.join(voc_stat,'锚框长宽比例分布.png'), dpi=300,bbox_inches = 'tight')
        print('保存锚框长宽比例分布结果')
        # plt.show()

        plt.figure()
        plt.hist(xml_df['anchor_width']*xml_df['anchor_height'], bins=50)
        plt.title('Histogram Plot of Bounding Areas')
        plt.savefig( os.path.join(voc_stat,'锚框面积直方图分布.png'), dpi=300,bbox_inches = 'tight')
        print('保存锚框面积直方图分布结果')
        # plt.show()

        # plt.figure()
        # sns.scatterplot(x='width', y='height', hue='class', data=xml_df)
        # plt.title('Scatter Plot of Images')
        # plt.savefig( os.path.join(voc_root,'图像长宽比例分布.png'), dpi=300,bbox_inches = 'tight')

        # plt.figure()
        # plt.hist(xml_df['width']*xml_df['height'], bins=50)
        # plt.title('Histogram Plot of Image Areas')
        # plt.savefig( os.path.join(voc_root,'图像面积直方图分布.png'), dpi=300,bbox_inches = 'tight')
        # plt.show()
        
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

    return xml_df

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages,Annotationshe ImageSets这3个文件夹，\
            在ImageSets文件夹下还要有Main/trainval.txt等文件')
    
    opt = parser.parse_args()

    voc_root = opt.voc_root

    if not os.path.exists(os.path.join(voc_root,'ImageSets/Main')):
        raise Exception("$VOC_ROOT/ImageSets/Main doesn't exist, please generate them using script voc2coco.py")
    
    voc_stat = os.path.join(voc_root, 'VOC统计信息')
    create_dir(voc_stat)

    anno_root = os.path.join(voc_root,'Annotations')
   
    print("=========统计所有数据信息============")
    all_xml_list = list(Path(anno_root).iterdir())
    df = load_dataset(all_xml_list, anno_root)
    df.to_csv(os.path.join(voc_stat,'all_info.csv'), index=False)

    for data_type in ['train', 'trainval', 'val', 'test']:
        print(f"\n\n=========统计{data_type}数据信息============")
        txt = os.path.join(voc_root, f'ImageSets/Main/{data_type}.txt')
        if not os.path.exists(txt):
            raise Exception(f'文件ImageSets/Main/{data_type}.txt不存在!')
        xml_files = [x.strip() for x in open(txt,'r').readlines()]
        xml_list = [os.path.join(anno_root, xml_name+'.xml') for xml_name in xml_files]
        df = load_dataset(xml_list, anno_root, savefig=False)
        df.to_csv(os.path.join(voc_stat,f'{data_type}_info.csv'), index=False)


