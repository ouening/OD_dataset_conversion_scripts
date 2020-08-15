from pathlib import Path
import os
import argparse

def check_files(ann_root, img_root):
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
    for an, im in zip(ann.iterdir(),img.iterdir()):
        ann_files.append(an.stem)
        img_files.append(im.stem)

    if set(ann_files)==set(img_files):
        print('标注文件和图像文件匹配')
    else:
        print('标注文件和图像文件不匹配')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages和Annotations这两个文件夹')
    opt = parser.parse_args()
    
    IMG_DIR = os.path.join(opt.voc_root, "JPEGImages")
    XML_DIR = os.path.join(opt.voc_root, "Annotations")

    check_files(XML_DIR, IMG_DIR)