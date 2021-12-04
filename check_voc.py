'''
检查voc数据集
'''
from pathlib import Path
import os
import argparse
import numpy as np

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages和Annotations这两个文件夹')
    parser.add_argument('--img_dir', type=str, required=False, 
        help='VOC格式数据集图像存储路径，如果不指定，默认为JPEGImages')
    parser.add_argument('--anno_dir', type=str, required=False, 
        help='VOC格式数据集标注文件存储路径，如果不指定，默认为Annotations')
    opt = parser.parse_args()
    
    print('Pascal VOC格式数据集路径：', opt.voc_root)

    if opt.img_dir is None:
        img_dir = 'JPEGImages'
    else:
        img_dir = opt.img_dir
    IMG_DIR = os.path.join(opt.voc_root, img_dir)
    if not os.path.exists(IMG_DIR):
        raise Exception(f'数据集图像路径{IMG_DIR}不存在！')

    if opt.anno_dir is None:
        anno_dir = 'Annotations'
    else:
        anno_dir = opt.anno_dir
    XML_DIR = os.path.join(opt.voc_root, anno_dir)
    if not os.path.exists(XML_DIR):
        raise Exception(f'数据集图像路径{XML_DIR}不存在！')

    check_files(XML_DIR, IMG_DIR)