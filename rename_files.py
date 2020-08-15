from pathlib import Path
import os
# from voc2coco import renamed_jpeg
from tqdm import tqdm
import shutil
import argparse

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    else:
        shutil.rmtree(ROOT) # 先删除，再创建
        os.mkdir(ROOT)

def rename_files(img_root, ann_root, img_ext='.png', ann_ext='.txt'):
    '''对图片和对应的标签进行重命名，支持voc格式和yolo格式(注意ann_ext后缀)'''
    p1 = Path(img_root)
    p2 = Path(ann_root)
    renamed_img = os.path.join(p1.parent, 'RenamedImg')
    create_dir(renamed_img)
    renamed_anno = os.path.join(p2.parent, 'RenamedAnno')
    create_dir(renamed_anno)

    imgs, annos = [], []
    for img, anno in zip(p1.iterdir(),p2.iterdir()):
        imgs.append(img.name.split('.')[0]) # 这里用'.'进行分割，因此要保证文件名中只有区分后缀的一个小数点
        annos.append(anno.name.split('.')[0])
    imgs= sorted(imgs)
    annos = sorted(annos)
    # print(imgs[:10], annos[:10])
    # print((set(imgs)|set(annos))-set(imgs)&set(annos))
    print(len(imgs), len(annos))
    print(set(imgs)-set(annos))
    print(set(annos)-set(imgs))
    assert set(imgs)==set(annos) # 检查图片文件名和标签文件名是否一致

    LENGTH = len(imgs) 
    print('图像数量：', LENGTH)
    for new_num, id in tqdm(zip(range(1,LENGTH+1), imgs), total=LENGTH):
        src_img_path = os.path.join(img_root, id+img_ext) # 原始Pascal格式数据集的图像全路径
        dst_img_path = os.path.join(renamed_img, str(new_num)+img_ext) # coco格式下的图像存储路径
        shutil.copy(src_img_path, dst_img_path) 

        src_xml_path = os.path.join(ann_root, id+ann_ext) # 原始Pascal格式数据集的图像全路径
        dst_xml_path = os.path.join(renamed_anno, str(new_num)+ann_ext) # coco格式下的图像存储路径
        shutil.copy(src_xml_path, dst_xml_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, 
        help='数据集图像存储路径')
    parser.add_argument('--anno',type=str, required=True,
        help='数据集标注文件存储路径')  
    
    opt = parser.parse_args()

    img_root = opt.img
    ann_root = opt.anno

    rename_files(img_root, ann_root, img_ext='.png', ann_ext='.txt')