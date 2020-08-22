# OD_dataset_conversion_scripts
Object detection dataset conversion scripts

1. PASCAL VOC => YOLO: voc2yolo.py
2. YOLO => PASCAL VOC: yolo2voc.py
3. PASCAL VOC => COCO: voc2coco.py
4. COCO => PASCAL VOC
   
   Use `utils_cv.detection.data.coco2voc` to complete this conversion. The process is listed below:
   - Install **Microsoft utils_cv** package: `pip install git+https://github.com/microsoft/ComputerVision.git@master#egg=utils_cv`
   - Import fumction: `from utils_cv.detection.data import coco2voc`
   - Function Signature: 
    ```
    Signature:
    coco2voc(
        anno_path: str,
        output_dir: str,
        anno_type: str = 'instance',
        download_images: bool = False,
    ) -> None
    Docstring:
    Convert COCO annotation (single .json file) to Pascal VOC annotations
        (multiple .xml files).

    Args:
        anno_path: path to coco-formated .json annotation file
        output_dir: root output directory
        anno_type: "instance" for rectangle annotation, or "keypoint" for keypoint annotation.
        download_images: if true then download images from their urls.
    ```
5. PASCAL VOC => CSV: voc2csv.py
6. PASCAL VOC => TXT: voc2txt.py
7. PASCAL VOC dataset information: voc_dataset_information.py
8. PASCAL VOC Augmentation: voc_augument.py
9.  YOLO Augmentation: yolo_augument.py
10. Rename file names: rename_files.py
11. Generate VOC/ImageSets/Main/trainval.txt(train.txt,val.txt,test.txt): voc_gen_trainval_test.py
12. Cluster anchors used in YOLO series: anchor-cluster.py
