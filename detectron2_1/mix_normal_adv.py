import json
import os
from pycocotools.coco import COCO
import random
import numpy as np
import argparse

def load_imgannot(cocodata, randomidx):
    img = cocodata.getImgIds(randomidx)
    img = cocodata.loadImgs(img)

    annot = []
    for i, idx in enumerate(randomidx):
        annIds = cocodata.getAnnIds(imgIds=img[i]['id'])
        anns = cocodata.loadAnns(annIds)
        annot.extend(anns)
        
#     print('%d instances are loaded'%len(randomidx))
    return img, annot


def relabel_img(img_dict, startid=0):
    
    combine_img = []
    map_dict = {}

    imgdict_c = img_dict.copy()
    for item in imgdict_c:
        map_dict[item['id']] = startid  # record mapping dict for later reference
        item_c = item.copy()
        item_c['id'] = startid
        combine_img.append(item_c)
        startid += 1
        
    return combine_img, map_dict

def relabel_annot(annotdict, mapdict, startid=0):
    
    combine_dict = []

    annotdict_c = annotdict.copy()
    for item in annotdict_c:
        item_c = item.copy()
        item_c['image_id'] = mapdict[item['image_id']]
        item_c['id'] = startid
        combine_dict.append(item_c)
        startid += 1
        
    return combine_dict


def main(args):
    # initialize COCO api for instance annotations
    coco=COCO(args.normal_path)
    coco_adv = COCO(args.adv_path)
    
    # get all images containing given categories
    catIds_logo = coco.getCatIds(catNms=['logo'])
    imgIds_logo = coco.getImgIds(catIds=catIds_logo)

    # get all images containing given categories
    catIds_logo_adv = coco_adv.getCatIds(catNms=['logo'])
    imgIds_logo_adv = coco_adv.getImgIds(catIds=catIds_logo_adv)
    
    # sample random image id: only sample from websites with identity logo
    random.seed(args.seed)
    random_normal = random.sample(imgIds_logo, args.normal_count) # Randomly sample 8000 normal training instances
    random_adv = random.sample(imgIds_logo_adv, args.adv_count) # Ranodmly sample 2000 adversarial training instances
    
    normal_img, normal_annot = load_imgannot(coco, random_normal)
    adv_img, adv_annot = load_imgannot(coco_adv, random_adv)
    
    # combine "images" field
    combine_img, map_dict_normal = relabel_img(normal_img, startid=0)
    combine_img_adv, map_dict_adv = relabel_img(adv_img, startid=len(normal_img))
    combine_img.extend(combine_img_adv)
    
    # combine "annotations" field
    combine_annot = relabel_annot(normal_annot, map_dict_normal)
    combine_annot_adv = relabel_annot(adv_annot, map_dict_adv, startid=len(combine_annot))
    combine_annot.extend(combine_annot_adv)
    
    # create new json dict
    combine_json = {
        "images": combine_img,  
        "annotations": combine_annot,
        "categories": [{"id": 1, "name": "box"}, {"id": 2, "name": "logo"}]
    }
    with open(args.save_path, 'w') as handle:
        json.dump(combine_json, handle)
        
    print("In total {total} training images are combined consists of {clean} clean examples and {adv} adversarial examples".format(total = len(combine_img), clean=len(normal_img), adv=len(adv_img)))
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normal-path",
        required=True,
        help="Path to ground-truth annotations json for clean training",
    )
    parser.add_argument(
        "--adv-path",
        required=True,
        help="Path to ground-truth annotations json for adv training",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Path to save the combined json",
    )
    parser.add_argument(
        "--normal-count",
        required=False,
        type=int,
        default = 8000,
        help="How many normal images to mix",
    )
    parser.add_argument(
        "--adv-count",
        required=False,
        type=int,
        default = 2000,
        help="How many adv images to mix",
    )
    parser.add_argument(
        "--seed",
        required=False,
        default = 1234,
        type=int,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    main(args)
