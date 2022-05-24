'''
Pascal VOC格式数据集转COCO格式数据集
适用项目：
1. https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
2. mmdetection2

数据集按照如下方式进行组织：
datasets/
    -Annotations/         # VOC格式标注存储路径
        -*.xml
    -ImageSets/           # VOC数据集ImageSets
        -Main/
            -train.txt
            -trainval.txt
            -val.txt
            -test.txt
    -JPEGImages/           # VOC数据集图像存储路径
        -*.jpg
    -CocoFormat/           # 本脚本生成的coco格式数据集默认存储位置
        -trainval/         # COCO trainval图像路径
            -*.jpg
        -train/            # COCO train图像路径
            -*.jpg
        -val/              # COCO val图像路径
            -*.jpg
        -test/             # COCO test图像路径
            -*.jpg
        -annotations       # COCO json标注文件路径
            -instances_trainval.json
            -instances_train.json
            -instances_val.json
            -instances_test.json
##============== 重要通告 ===============##
笔者在使用一些COCO格式目标检测模型的时候，发现如果image_id不为int型的话会有很多问题，例如在使用torchvison中的
COCODetection时会遇到错误：

path = coco.loadImgs(img_id)[0]['file_name']

File "python\lib\site-packages\pycocotools\coco.py", line 230, in loadImgs
return [self.imgs[id] for id in ids]

File "python\lib\site-packages\pycocotools\coco.py", line 230, in <listcomp>
return [self.imgs[id] for id in ids]

KeyError: '0'

原因是image_id值是[]，直接报错，因此需要考虑将VOC格式下的文件名全部重命名为数字后再进行转换，使用参数选项--rename即可

'''
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import shutil
import json
from typing import Dict, List
from tqdm import tqdm
import re
from collections import Counter
from imageio import imread
def get_label2id(labels_path: str) -> Dict[str, int]:
    '''
    id is 1 start
    '''
    with open(labels_path, 'r') as f:
        labels_str = f.read().strip().split('\n')
    labels_ids = list(range(1, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_image_info(ann_path, annotation_root, extract_num_from_imgid=True):
    '''
    ann_path：标注文件全路径
    annotation_root：xml对根内容进行解析后的内容
    extract_num_from_imgid：是否从imageid中提取数字，对于COCO格式数据集最好使用True选项，将image_id转换为整型
    '''
    img_name = os.path.basename(ann_path)
    img_id = os.path.splitext(img_name)[0]
    filename = img_id+ext

    if extract_num_from_imgid and isinstance(img_id, str):
        # 采用正则表达式，支持转换的文件命名：0001.png, cls_0021.png, cls0123.jpg, 00123abc.png等
        img_id = int(re.findall(r'\d+', img_id)[0])

    try:
        size = annotation_root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
    except:
        img_path = Path(ann_path).parent.parent.joinpath('JPEGImages', filename)
        width, height = imread(str(img_path)).shape[:2]

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def counting_labels(anno_root: str):
    '''
    获取pascal voc格式数据集中的所有标签名
    anno_root: pascal标注文件路径，一般为Annotations
    '''
    all_classes = []
    
    for xml_file in os.listdir(anno_root):
        xml_file = os.path.join(anno_root, xml_file)
        # print(xml_file)
        xml = open(xml_file,) # encoding='utf-8'
        tree=ET.parse(xml)
        root = tree.getroot()
        for obj in root.iter('object'):
            
            class_ = obj.find('name').text.strip()
            all_classes.append(class_)
    
    print(Counter(all_classes))

    labels = sorted(list(set(all_classes)))
    print('标签数据：', labels)
    print('标签长度：', len(labels))
    print('写入标签信息...{}'.format(os.path.join(opt.voc_root,'labels.txt')))
    with open( os.path.join(opt.voc_root,'labels.txt') , 'w') as f:
        for k in labels:
            f.write(k)
            f.write('\n')

def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name').strip()
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        # 起始点是左上角，按照顺时针方向
        'segmentation': [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]] 
    }
    return ann

def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        # print(a_path)
        img_info = get_image_info(ann_path=a_path,
                                annotation_root=ann_root,
                                extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'electrical_fittings', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)

def create_dir(ROOT:str):
    if not os.path.exists(ROOT):
        os.makedirs(ROOT)

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
    for an in ann.iterdir():
        ann_files.append(an.stem)

    for im in img.iterdir():
        img_files.append(im.stem)
        img_exts.append(im.suffix)

    if not len(ann_files)==len(img_files):
        raise Exception("图像数据和标注数据数量不一致！")
    
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
        help='VOC格式数据集根目录，该目录下必须包含存储图像和标注文件的两个文件夹，例如官方格式下有JPEGImages和Annotations两个文件夹')
    parser.add_argument('--img_dir', type=str, required=False, 
        help='VOC格式数据集图像存储路径，如果不指定，默认为JPEGImages')
    parser.add_argument('--anno_dir', type=str, required=False, 
        help='VOC格式数据集标注文件存储路径，如果不指定，默认为Annotations')
    parser.add_argument('--coco-dir', type=str, default='CocoFormatData', 
        help='COCO数据集存储路径，默认为VOC数据集相同路径下新建文件夹CocoDataset')
    parser.add_argument('--test-ratio',type=float, default=0.2,
        help='验证集比例，默认为0.3')   
    parser.add_argument('--rename',type=bool, default=False,
        help='是否对VOC数据集进行数字化重命名')  
    parser.add_argument('--label-file', type=str, required=False,
                        help='path to label list.')

    opt = parser.parse_args()

    voc_root = opt.voc_root
    print('Pascal VOC格式数据集路径：', voc_root)

    xml_file = []
    img_files = []

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

    ext = check_files(voc_anno, voc_jpeg) # 检查图像后缀
    assert ext is not None, "请检查图像后缀是否正确！"
    print()
    ##============================##
    ##   对文件进行数字化重命名，需要对ImageSets/Main下的分割数据做相应修改
    ##============================##
    if opt.rename==True:
        renamed_jpeg = os.path.join(voc_root,'RenamedJPEGImages')
        create_dir(renamed_jpeg)
        renamed_xml = os.path.join(voc_root,'RenamedAnnotations')
        create_dir(renamed_xml)

        p1 = Path(voc_jpeg)
        p2 = Path(voc_anno)
        imgs = sorted([x.stem for x in p1.iterdir() if not x.stem.startswith('.')])
        annos = sorted([x.stem for x in p2.iterdir() if not x.stem.startswith('.')])

        assert imgs==annos

        # 非规则名称与数字id的映射字典{'a'：0, 'b':1, ...}
        names_to_id_dict = {k:v for (v,k) in enumerate(imgs)}
         
        print('图像数量：', len(imgs))
        for name, id in tqdm(names_to_id_dict.items()):
            src_img_path = os.path.join(voc_jpeg, name+ext) # 原始Pascal格式数据集的图像全路径
            # print(src_img_path)
            dst_img_path = os.path.join(renamed_jpeg, str(id)+ext) # coco格式下的图像存储路径
            # print(dst_img_path)
            shutil.copy2(src_img_path, dst_img_path) 

            src_xml_path = os.path.join(voc_anno, name+'.xml') # 原始Pascal格式数据集的图像全路径
            dst_xml_path = os.path.join(renamed_xml, str(id)+'.xml') # coco格式下的图像存储路径
            shutil.copy2(src_xml_path, dst_xml_path)
        
        voc_jpeg = renamed_jpeg     # 将重命名后的图像路径赋值给JPEG
        voc_anno = renamed_xml      # 将重命名后的标注路径赋值给ANNO

    ImgSets = os.path.join(voc_root, 'ImageSets')
    if not os.path.exists(ImgSets):
        os.mkdir(ImgSets)
    ImgSetsMain = os.path.join(ImgSets,'Main')

    create_dir(ImgSetsMain)

    #== COCO 数据集路径
    coco_root = os.path.join(str(Path(voc_root).parent), Path(voc_root).stem + opt.coco_dir) # pascal voc转coco格式的存储路径
    if os.path.exists(coco_root):
        shutil.rmtree(coco_root)
        create_dir(coco_root)

    txt_files = ['trainvaltest','train','val','trainval','test']

    coco_dirs = [] 
    for dir_ in txt_files:
        DIR = os.path.join(coco_root, dir_)
        coco_dirs.append(DIR)
        create_dir(DIR)

    coco_anno = os.path.join(coco_root, 'annotations') # coco标注文件存放路径
    create_dir(coco_anno)

    # 利用VOC ImageSets数据划分信息，注意ImageSets/main/train.txt文件只记录图片名称，没有后缀
    # 所有图片名称
    files = [x.stem for x in Path(voc_jpeg).iterdir() if not x.stem.startswith('.')]
    # files = list(Path(voc_jpeg).iterdir())
    # files = [str(x).replace('.jpg','') for x in files]
    print('数据集长度:',len(files))
    assert os.path.exists(os.path.join(voc_root, 'ImageSets/Main/trainval.txt'))
    if os.path.exists(os.path.join(voc_root, 'ImageSets/Main/trainval.txt')):
        
        print('>>>使用ImageSet信息分割数据集')
        trainval_file = os.path.join(voc_root, 'ImageSets/Main/trainval.txt')
        if opt.rename:
            trainval = [names_to_id_dict[i.strip()] for i in open(trainval_file,'r').readlines()]
        else:
            # trainval = [i.strip() for i in open(trainval_file,'r').readlines()] 
            trainval = [i.replace('\n','') for i in open(trainval_file,'r').readlines()]
        # trainval = [os.path.join(os.path.join(coco_root,'trainval'),name) for name in trainval_name]

        train_file = os.path.join(voc_root, 'ImageSets/Main/train.txt')
        if opt.rename:
            train = [names_to_id_dict[i.strip()] for i in open(train_file,'r').readlines()]
        else:
            # train = [i.strip() for i in open(train_file,'r').readlines()]
            train = [i.replace('\n','') for i in open(train_file,'r').readlines()]
        # train = [os.path.join(os.path.join(coco_root,'train'),name) for name in train_name]

        val_file = os.path.join(voc_root, 'ImageSets/Main/val.txt')
        if opt.rename:
            val = [names_to_id_dict[i.strip()] for i in open(val_file,'r').readlines()]
        else:
            val = [i.replace('\n','')  for i in open(val_file,'r').readlines()]
        # val = [os.path.join(os.path.join(coco_root,'val'),name) for name in val_name]

        test_file = os.path.join(voc_root, 'ImageSets/Main/test.txt')
        if opt.rename:
            test = [names_to_id_dict[i.strip()] for i in open(test_file,'r').readlines()]
        else:
            test = [i.replace('\n','')  for i in open(test_file,'r').readlines()]
        # test = [os.path.join(os.path.join(coco_root,'test'),name) for name in test_name]
        
        print('>>>训练集数量: ',len(train))
        print('>>>训练集验证集数量: ',len(trainval))
        print('>>>验证集数量: ',len(val))
        print('>>>测试集数量: ',len(test))

    else:
        print('>>>随机划分COCO数据集')
        files = shuffle(files)
        ratio = opt.test_ratio
        trainval, test = train_test_split(files, test_size=ratio)
        train, val = train_test_split(trainval,test_size=0.2)
        print('训练集数量: ',len(train))
        print('验证集数量: ',len(val))
        print('测试集数量: ',len(test))

    def write_txt(txt_path, data):
        with open(txt_path,'w') as f:
            for d in data:
                f.write(str(d))
                f.write('\n')
    
    # 写入各个txt文件
    datas = [files, train, val, trainval, test]

    for txt, data in zip(txt_files, datas):
        txt_path = os.path.join(ImgSetsMain, txt+'.txt')
        write_txt(txt_path, data)

    # 遍历xml文件，得到所有标签值，并且保存为labels.txt
    if opt.label_file:
        print('从自定义标签文件读取！')
        labels = opt.label_file
    else:
        print('从xml文件自动处理标签！')
        counting_labels(voc_anno)
        labels = os.path.join(voc_root, 'labels.txt')

    if not os.path.isfile(labels):
        raise Exception('需要提供数据集标签文件路径，用于按顺序转换数值id，如果没有，需要手动创建！')
    
    label2id = get_label2id(labels_path=labels)
    print('标签值及其对应的编码值：',label2id)

    for name,imgs,coco_dir in tqdm(zip(txt_files,datas,coco_dirs)):
        
        annotation_paths = []
        # [1] copy image files
        for img in imgs:
            # print(img)
            annotation_paths.append(os.path.join(voc_anno, str(img)+'.xml'))
            src_img_path = os.path.join(voc_jpeg, str(img)+ext) # 原始Pascal格式数据集的图像全路径
            # print(src_img_path)
            dst_img_path = os.path.join(coco_dir, str(img)+ext) # coco格式下的图像存储路径
            # print(dst_img_path)
            
            shutil.copy2(src_img_path, dst_img_path) 

        # [2] convert xml to coco json format files
        convert_xmls_to_cocojson(
                    annotation_paths=annotation_paths,
                    label2id=label2id,
                    output_jsonpath=os.path.join(coco_anno, f'instances_{name}.json'),
                    # img_ids = imgs
                    extract_num_from_imgid=True      # 一定注意这里，COCO格式数据集image_id需要整型，可以从图片名称中抽取id号
                    )
