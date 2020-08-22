'''
COCO数据集转VOC
`pip install git+https://github.com/microsoft/ComputerVision.git@master#egg=utils_cv`
'''

from utils_cv.detection.data import coco2voc
from pathlib import Path
import argparse

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
    coco2voc(anno_path=opt.anno_path, output_dir=out_dir, anno_type=opt.anno_type)