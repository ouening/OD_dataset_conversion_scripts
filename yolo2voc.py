'''
本脚本将YOLO格式的数据集转换成VOC格式，数据集组织格式为：

|--data       # 数据集根目录
|----images   # 存储图像数据
|------0.png
|------1.png
|------...
|----labels   # 存储yolo格式的标注数据：class x_center y_center width height，class索引从0开始，xywh的范围是（0，1）
|------0.txt
|------1.txt
|------...
|----classes.txt

'''
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path


def yolo_to_voc_format(x_center, y_center, width, height, img_width, img_height):
    '''将yolo格式转换为voc格式'''

    center_x = round(float(str(x_center).strip()) * img_width)
    center_y = round(float(str(y_center).strip()) * img_height)
    bbox_width = round(float(str(width).strip()) * img_width)
    bbox_height = round(float(str(height).strip()) * img_height)

    xmin = int(center_x - bbox_width / 2 )
    ymin = int(center_y - bbox_height / 2)
    xmax = int(center_x + bbox_width / 2)
    ymax = int(center_y + bbox_height / 2)

    return xmin,ymin,xmax,ymax


def voc_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    '''将voc格式转换为yolo格式'''
    dw = 1./(img_width) # 宽度缩放比例, size[0]为图像宽度width
    dh = 1./(img_height)
    x = (xmin + xmax)/2.0 - 1
    y = (ymin + ymax)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x_center = x*dw
    width = w*dw
    y_center = y*dh
    height = h*dh

    return x_center, y_center, width, height

def read_yolo_annotation(img_root, label_root, image_id):
    '''root：一般是labels, txt文件, <object-class> <x_center> <y_center> <width> <height>
    image_id是包含.txt后缀的文件名

    return:
        [xmin, ymin, xmax, ymax, label], pascal format
    '''
    img_width, img_height = Image.open(os.path.join(img_root, image_id[:-4] + ext)).size

    annos = [x for x in open(os.path.join(label_root, image_id)).readlines()]
    bndboxlist = []

    for anno in annos:  # 找到root节点下的所有country节点
        lb, x_center, y_center, width, height = anno.split(' ') 

        xmin,ymin,xmax,ymax = yolo_to_voc_format(x_center, y_center, width, height, img_width, img_height)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax, int(lb)])
        # print(bndboxlist)

    return bndboxlist

def write_xml(img_root, bndbox, save_root, name):   
    '''
    img_root: yolo images root
    bndbox: [[xmin, ymin, xmax, ymax, label],[...],...], pascal format
    save_root: directory to save xml file
    name: note: with .txt
    '''
    img_width, img_height = Image.open(os.path.join(img_root, name[:-4] + ext)).size

    root = ET.Element('annotation')                             #创建Annotation根节点
    ET.SubElement(root, 'filename').text = name[:-4] + ext         #创建filename子节点（带后缀）
    sizes = ET.SubElement(root,'size')                          #创建size子节点            
    ET.SubElement(sizes, 'width').text = str(img_width)                 #没带脑子直接写了原图片的尺寸......
    ET.SubElement(sizes, 'height').text = str(img_height)
    ET.SubElement(sizes, 'depth').text = '3'                    #图片的通道数：img.shape[2]
    for box in bndbox:
        xmin, ymin, xmax, ymax, label = box
        label_class = classes_idx[label]                        # 数字标签转换为字符串标签
        
        objects = ET.SubElement(root, 'object')                 #创建object子节点
        ET.SubElement(objects, 'name').text = label_class        
                                                                       #的类别名
        ET.SubElement(objects, 'pose').text = 'Unspecified'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects,'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
    tree = ET.ElementTree(root)

    filepath = os.path.join( save_root, name[:-4]+'.xml')
    tree.write(filepath, encoding='utf-8')


def mkdir(path:str):
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

def check_files(img_root):
    '''检测图像名称和xml标准文件名称是否一致，检查图像后缀'''
    
    if os.path.exists(img_root):
        img = Path(img_root)
    else:
        raise Exception("图像文件路径错误")
    img_exts = []
    for im in img.iterdir():
        img_exts.append(im.suffix)

    print('图像后缀列表：', np.unique(img_exts))
    if len(np.unique(img_exts)) > 1:
        # print('数据集包含多种格式图像，请检查！', np.unique(img_exts))
        raise Exception('数据集包含多种格式图像，请检查！', np.unique(img_exts))
    
    return np.unique(img_exts)[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-root', type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含images和labels这两个文件夹，以及classes.txt标签名文件')
    parser.add_argument('--voc-outdir',type=str, default='VOCFormatData',
        help='Pascal VOC格式数据集存储路径，默认为yolo数据集同级目录下新建文件夹VOCFormatData')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例，（0,1）之间的浮点数')
    
    opt = parser.parse_args()
    assert os.path.exists(opt.yolo_root) # 确保数据集存在

    yolo_img = os.path.join(opt.yolo_root, "images")
    yolo_label = os.path.join(opt.yolo_root, "labels")
    ext = check_files(yolo_img) #检查文件后缀

    voc_root = os.path.join(str(Path(opt.yolo_root).parent), Path(opt.yolo_root).stem + opt.voc_outdir)
    if not os.path.exists(voc_root):
        os.mkdir(voc_root)
    # 读取标签名
    assert os.path.isfile(os.path.join(opt.yolo_root, 'classes.txt')), "请检查YOLO数据集根目录下的标签文件classes.txt，\
        若没有，需自行构建（每行一个类别，索引从0开始，和yolo的txt标注文件里的标签匹配！"
    classes = [x.strip() for x in open( os.path.join(opt.yolo_root, 'classes.txt'),'r', encoding='utf-8').readlines()]
    print('>>>数据集标签：', classes)
    classes_dict = {} # 字典{'class1':0, 'class2':1, ...}
    classes_idx = {} # 字典：{0:'class0', 1:'class1', ...}
    for k, v in enumerate(classes):
        classes_dict[v] = k
        classes_idx[k] = v
        
    # 创建存储图像路径
    voc_jpeg = os.path.join(voc_root, "JPEGImages")  # 存储
    try:
        shutil.rmtree(voc_jpeg)
    except FileNotFoundError as e:
        a = 1
    mkdir(voc_jpeg)
    # 创建存储xml标签路径
    voc_anno = os.path.join(voc_root, "Annotations")  # 存储XML文件夹路径
    try:
        shutil.rmtree(voc_anno)
    except FileNotFoundError as e:
        a = 1
    mkdir(voc_anno)

    # 创建存储train.txt, trainval.txt, test.txt标签路径
    voc_set_dir = os.path.join(voc_root, "ImageSets/Main")  # 
    if not os.path.exists(voc_set_dir):
        os.makedirs(voc_set_dir)

    for root, sub_folders, files in os.walk(yolo_label):
        for name in tqdm(files): # name是包含.txt后缀的文件名
            # [xmin, ymin, xmax, ymax, int(lb)]
            bndbox = read_yolo_annotation(yolo_img, yolo_label, name) # 读取yolo标注文，注意：## 返回voc格式标注框 ##
            shutil.copy(os.path.join(yolo_img, name[:-4] + ext), voc_jpeg) # 复制图像至VOC图像存储路径
            write_xml(img_root=yolo_img, 
                        bndbox=bndbox,
                        save_root=voc_anno,
                        name=name
                        )       
    
    # 所有图片名称
    files = [x.stem for x in Path(voc_jpeg).iterdir() if not x.stem.startswith('.')]

    # 利用已有的yolo数据划分信息 ${YOLO-ROOT}/trainval.txt, train.txt, val.txt, test.txt
    if os.path.exists(os.path.join(opt.yolo_root, 'trainval.txt')) and \
        os.path.exists(os.path.join(opt.yolo_root, 'train.txt')) and \
        os.path.exists(os.path.join(opt.yolo_root, 'val.txt')) and \
            os.path.exists(os.path.join(opt.yolo_root, 'test.txt')):
        print('>>> 使用YOLO已有划分数据分割train,val和test')
        trainval = [Path(x).stem for x in open(os.path.join(opt.yolo_root, 'trainval.txt')).readlines()]
        train = [Path(x).stem for x in open(os.path.join(opt.yolo_root, 'train.txt')).readlines()]
        val = [Path(x).stem for x in open(os.path.join(opt.yolo_root, 'val.txt')).readlines()]
        test = [Path(x).stem for x in open(os.path.join(opt.yolo_root, 'test.txt')).readlines()]
    else:
        
        print('>>>随即划分train,val和test')
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
    trainval_txt = os.path.join(voc_set_dir,'trainval.txt')
    write_txt(trainval_txt, trainval)

    train_txt = os.path.join(voc_set_dir,'train.txt')
    write_txt(train_txt, train)

    val_txt = os.path.join(voc_set_dir,'val.txt')
    write_txt(val_txt, val)

    test_txt = os.path.join(voc_set_dir,'test.txt')
    write_txt(test_txt, test)
