'''
tf2.x 版本转换PASCAL VOC至tfrecord格式
1. 使用labelimg等标注工具制作pascal voc格式数据集，注意：图像存储在JPEGImages文件夹，xml标注文件存储在Annotations文件夹
2. 将xml格式转换成csv格式，本脚本使用xml_to_csv函数已经在内部实现
3. 将csv转成TFrecord格式，注意tf1.x版本和tf2.x版本接口是不一样的

参考链接：https://www.pythonf.cn/read/109620

注意事项：对于自定义数据集，需要指定labels列表
'''
from __future__ import division  
from __future__ import print_function  
from __future__ import absolute_import  
  
import os  
import io  
import pandas as pd  
import tensorflow as tf  
  
from PIL import Image  
# from object_detection.utils import dataset_util  
from collections import namedtuple, OrderedDict  
from tqdm import tqdm
import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
# flags = tf.app.flags  
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')  
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')  
# FLAGS = flags.FLAGS  
# TO-DO replace this with label map  
# labels = ['cow', 'tvmonitor', 'car', 'aeroplane', 'sheep', 
# 'motorbike', 'train', 'chair', 'person', 'sofa', 
# 'pottedplant', 'diningtable', 'horse', 'bottle', 
# 'boat', 'bus', 'bird', 'bicycle', 'cat', 'dog']

# 根据自定义数据集修改该列表
labels = ['raccoon']

def class_text_to_int(row_label):
    return labels.index(row_label)+1
  
def split(df, group):  
    data = namedtuple('data', ['filename', 'object'])  
    gb = df.groupby(group)  
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]  
  
def xml_to_csv(xml_anno, data_type):
    '''
    xml_anno: pascal voc标准文件路径
    data_type:['trainvaltest','train','val','trainval','test']
    '''
    xml_list = []
    # xml_files = []
    txt_file = str(Path(xml_anno).parent/'ImageSets/Main'/f'{data_type}.txt')
    xml_files = [os.path.join(xml_anno, k.strip()+'.xml') for k in open(txt_file,'r').readlines()]
    # for xml_file in glob.glob(xml_anno + '/*.xml'):
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = opt.format.encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax':  tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax':tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example
  
  
def main(voc_root, output_name):  
    img_path = os.path.join(voc_root, 'JPEGImages')
    # examples = pd.read_csv(csv_input)
    imgset_path = os.path.join(voc_root, 'ImageSets/Main')
    if not os.path.exists(imgset_path):
        raise Exception('ImageSets/Main文件夹不存在，请通过脚本生成相应的文件！')
    txt_files = ['trainvaltest.txt','train.txt','val.txt','trainval.txt','test.txt']

    valid_txt = []
    for k in txt_files:
        txt = os.path.join(imgset_path, k)
        if os.path.exists(txt):
            valid_txt.append(k[:-4])

    if valid_txt:
        print(valid_txt)
    else:
        raise Exception('ImageSets/Main文件夹下不存在train.txt等文件，请检查数据集！')
    
    for data_type in valid_txt:
        output_path = output_name + f'_{data_type}.tfrecord'
        output_path = os.path.join(voc_root, output_path)  
        writer = tf.io.TFRecordWriter(output_path)  
        examples = xml_to_csv(os.path.join(voc_root, 'Annotations'), data_type)
        grouped = split(examples, 'filename')  

        for group in tqdm(grouped):  
            tf_example = create_tf_example(group, img_path)  
            writer.write(tf_example.SerializeToString())    
    
        writer.close()  
        print('Successfully created the TFRecords: {}'.format(output_path))  
  
if __name__ == '__main__':  
    # tf.app.run()
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=str, required=True, help="PASCAL VOC 数据集路径，包含JPEGImages和Annotations两个文件夹")
    parser.add_argument("--output_name", type=str, default="voc2020", help="tfrecord文件名称,默认保存在VOC根路径")
    parser.add_argument("--format", type=str, default="jpg", help="图像格式")
    opt = parser.parse_args()
    main(opt.voc_root, opt.output_name)