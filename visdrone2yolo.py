'''
将visdrone数据集转换为yolo格式，visdrone标注数据的格式为：
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
该数据集的类别总共有11类
ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), 
awning-tricycle(8), bus(9), motor(10), others(11)

yolo格式为：
class x_center y_center width height（归一化数值）
'''
import os
from pathlib import Path
from PIL import Image
import csv
from tqdm import tqdm
import argparse
import numpy as np
import shutil
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2) * dw
    y = (box[1] + box[3] / 2) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def check_files(ann_root, img_root):
    '''检测图像名称和xml标准文件名称是否一致，检查图像后缀
    return：
        返回图像后缀
    '''
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


def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--visdrone-root', type=str, required=True, 
        help='visDrone数据集根目录，该目录下必须包含annotations和images的两个文件夹')
    
    parser.add_argument('--yolo-label-dir', type=str, default=None,
                        help='directory to save yolo label.')

    opt = parser.parse_args()

    vis_dir = opt.visdrone_root
    vis_img_dir = os.path.join(vis_dir, 'images') # visdrone图像存储路径
    vis_anno_dir = os.path.join(vis_dir, 'annotations') # visdrone标注文件存储路径
    # 检查数据集
    img_suffix = check_files(vis_anno_dir, vis_img_dir)   

    if opt.yolo_label_dir is None:
        yolo_label_dir = os.path.join(vis_dir,'labels')
        if not os.path.exists(yolo_label_dir):
            os.makedirs(yolo_label_dir)
    else:
        yolo_label_dir = opt.yolo_label_dir
    print('YOLO标签存储路径：', yolo_label_dir)

    total_imgs = len(os.listdir(vis_anno_dir))
    annos = Path(vis_anno_dir).iterdir()

    for anno in tqdm(annos, total=total_imgs):
        ans = ''
        # print(anno)
        if anno.suffix != '.txt':
            continue
        # load image
        with Image.open(os.path.join(vis_img_dir,anno.stem+img_suffix)) as Img:
            img_size = Img.size
        # read annotation file
        # print(img_size)
        with open(os.path.join(vis_anno_dir, str(anno)),) as f:
            lines = f.readlines()
            save_path = os.path.join(yolo_label_dir,anno.stem+'.txt') # path to save yolo format annotation
            for line in lines:
                row = line.strip().split(',')
                if row[4] == '0': 
                    continue
                bb = convert(img_size, tuple(map(int, row[:4])))
                ans = ans + str(int(row[5])-1) + ' ' + ' '.join(str(a) for a in bb) + '\n'
                with open(save_path, 'w') as outfile:
                    outfile.write(ans)
            # outfile.close()
