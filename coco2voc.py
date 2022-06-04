'''
COCO数据集转VOC
`pip install git+https://github.com/microsoft/computervision-recipes.git@master#egg=utils_cv`

参考：https://github.com/microsoft/computervision-recipes/blob/master/utils_cv/detection/references/anno_coco2voc.py

coco instance标注格式转换为voc格式
'''

import argparse, json
import cytoolz
from lxml import etree, objectify
import os, re
# from utils_cv.detection.data import coco2voc
from pathlib import Path
import argparse

def instance2xml_base(anno):
    # anno: a dict type containing annotation infomation
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/{}'.format(anno['category_id'])),
        E.filename(anno['file_name']),
        E.size(
            E.width(anno['width']),
            E.height(anno['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def parse_instance(content, outdir):
    categories = {d['id']: d['name'] for d in content['categories']}

    # EDITED - make sure image_id is of type int (and not of type string)
    for i in range(len(content['annotations'])):
        content['annotations'][i]['image_id'] = int(content['annotations'][i]['image_id'])

    # EDITED - save all annotation .xml files into same sub-directory
    anno_dir = os.path.join(outdir, "annotations")
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations'])))
    
    # convert category id to name
    for instance in merged_info_list:
        assert 'category_id' in instance, f"WARNING: annotation error: image {instance['file_name']} has a rectangle without a 'category_id' field."
        instance['category_id'] = categories[instance['category_id']]

    # group by filename to pool all bbox in same file
    img_filenames = {}
    names_groups = cytoolz.groupby('file_name', merged_info_list).items()
    for index, (name, groups) in enumerate(names_groups):
        print(f"Converting annotations for image {index} of {len(names_groups)}: {name}")
        assert not name.lower().startswith(("http:","https:")), "Image seems to be a url rather than local. Need to set 'download_images' = False"

        anno_tree = instance2xml_base(groups[0])
        # if one file have multiple different objects, save it in each category sub-directory
        filenames = []
        for group in groups:
            filename = os.path.splitext(name)[0] + ".xml"

            # EDITED - save all annotations in single folder, rather than separate folders for each object 
            #filenames.append(os.path.join(outdir, re.sub(" ", "_", group['category_id']), filename)) 
            filenames.append(os.path.join(anno_dir, filename))

            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))

        for filename in filenames:
            etree.ElementTree(anno_tree).write(filename, pretty_print=True)

def coco2voc(anno_file, output_dir, anno_type):
    '''对原本代码进行裁剪，只支持instance标注格式'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    content = json.load(open(anno_file, 'r'))
    
    if anno_type == 'instance':
        # EDITED - save all annotations in single folder, rather than separate folders for each object 
        # make subdirectories
        # sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]   #EDITED
        # for sub_dir in sub_dirs:
        #     sub_dir = os.path.join(output_dir, str(sub_dir))
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)
        parse_instance(content, output_dir)
    else:
        raise Exception('格式不符合，请检查！')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-path', type=str, required=True, 
        help='COCO .json 格式文件路径')
    parser.add_argument('--out-dir', type=str, 
        help='VCO格式标注文件存储根路径，例如-out-dir=VOC,该路径下会自动生成VOC/annotations文件夹，该文件夹下存储.xml格式标注文件')
    parser.add_argument('--anno-type',type=str, default="instance",
        help='"instance" for rectangle annotation, or "keypoint" for keypoint annotation.')   

    opt = parser.parse_args()

    if opt.out_dir is None:
        out_dir = Path(opt.anno_path).parent.parent / 'VOCAnnotations'
    else:
        out_dir = opt.out_dir
    coco2voc(anno_file=opt.anno_path, output_dir=out_dir, anno_type=opt.anno_type)