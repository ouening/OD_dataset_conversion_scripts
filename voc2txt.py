'''
本脚本处理的数据集格式适用程序项目：
1. https://github.com/Tianxiaomo/pytorch-YOLOv4
2. https://github.com/YunYang1994/tensorflow-yolov3
3. https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3

数据集格式：
# train.txt
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
...
...
'''

import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['fault']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.split('/')[-1].strip().split('.')[0].replace(' ','') for line in txt]
	
    with open(os.path.join(data_path,anno_path), 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.png')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=str, required=True, 
        help='VOC格式数据集根目录，该目录下必须包含JPEGImages，Annotations和ImageSets这三个文件夹')
    parser.add_argument("--train_annotation", default="voc_train.txt")
    parser.add_argument("--test_annotation",  default="voc_test.txt")
    opt = parser.parse_args()

    if os.path.exists(os.path.join( opt.voc_root, opt.train_annotation)):
        os.remove(os.path.join(opt.voc_root, opt.train_annotation))
    if os.path.exists(os.path.join( opt.voc_root, opt.test_annotation)):
        os.remove(os.path.join(opt.voc_root, opt.test_annotation))

    # trainval包括训练和验证，在此全部当作训练集使用
    num1 = convert_voc_annotation(opt.voc_root, 'trainval', opt.train_annotation, False)
    
    num2 = convert_voc_annotation(opt.voc_root, 'test', opt.test_annotation, False)
    print('=> The number of image for train is: %d\nThe number of image for test is:%d' %(num1, num2))


