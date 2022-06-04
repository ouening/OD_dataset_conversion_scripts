import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
info = {"description": "DOTA dataset from WHU", "url": "http://caption.whu.edu.cn", "year": 2018, "version": "1.0"}
licenses = {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 1, "name": "Attribution-NonCommercial License"}
categories = []
cat_names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter','container-crane']
for i, catName in enumerate(cat_names, start=1):
    categories.append({"id": i, "name": "%s" % catName, "supercategory": "%s" % catName})

images = []
annotations = []
aug = "/home/lxy/dota/data/aug"
augmented = "/home/lxy/dota/data/augmented"
train_small = "/home/lxy/dota/data/train_small"
trainsplit_HBB = "/home/lxy/dota/data/trainsplit_HBB"
val_small = "/home/lxy/dota/data/val_small"
valsplit_HBB = r"D:\BaiduNetdiskDownload"
# dataset_path = [augmented, train_small, trainsplit_HBB, val_small, valsplit_HBB]
dataset_path = [valsplit_HBB]
imgid = 0
annid = 0
for path in dataset_path:
    img_path = os.path.join(path, "JPEGImages")
    label_path = os.path.join(path, "DOTA-v1.5_val_hbb")
    for file in tqdm(os.listdir(label_path)):
        img_name = file.replace("txt", "png")
        im = Image.open(os.path.join(img_path, img_name))
        w, h = im.size
        imgid += 1
        images.append({"license": 1, "file_name": "%s" % img_name, \
                       "height": h, "width": w, "id": imgid})
        
        f = open(os.path.join(label_path, file))
        for line in f.readlines():
            line = "".join(line).strip("\n").split(" ")
            # a bbox has 4 points, a category name and a difficulty
            if len(line) != 10:
                print(path, file)
            else:
                annid += 1
                catid = cat_names.index(line[-2]) + 1
                w_bbox = int(line[4][:-2]) - int(line[0][:-2])
                h_bbox = int(line[5][:-2]) - int(line[1][:-2])
                # bbox = [line[0], line[1], str(w_bbox)+'.0', str(h_bbox)+'.0']
                bbox = [np.double(line[0]),np.double(line[1]), np.double(w_bbox), np.double(h_bbox)]
                seg = [np.double(line[0]), np.double(line[1]), np.double(line[2]), np.double(line[3]),np.double(line[4]), np.double(line[5]), np.double(line[6]), np.double(line[7])]
                annotations.append({"id": annid, "image_id": imgid, "category_id": catid, \
                                    "segmentation": [seg], \
                                    "area": float(w_bbox*h_bbox), \
                                    "bbox": bbox, "iscrowd": 0})
                
        f.close()

my_json = {"info": info, "licenses": licenses, "images": images, "annotations": annotations, "categories": categories}

json_path = os.path.join(valsplit_HBB,'val1.json')
with open(json_path, "w+") as f:
    json.dump(my_json, f)
    print("writing json file done!")