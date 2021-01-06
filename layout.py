from layout_matcher.layout_matcher_knn import bipartite_web
from layout_matcher.misc import load_yaml
from layout_matcher.heuristic import layout_heuristic

import os
import cv2
from element_detector import element_recognition
import torch
import numpy as np



def layout_config(cfg_dir:str, ref_dir:str, matched_brand:str, ele_model):
    
    '''
    Load layout configurations
    :param cfg_dir: path to cfg file
    :param ref_dir: reference layout folder
    :param matched_brand: prediction target from siamese
    :param ele_model: element recognition model
    :return cfg
    :return gt_coords_arr: reference layouts
    :return gt_files_arr: reference paths
    :return gt_shot_size_arr: reference shots sizes
    '''
    
    # load cfg
    cfg = load_yaml(cfg_dir)
    
    assert matched_brand in ['Amazon', 'Facebook', 'Google', 'Instagram', 'LinkedIn Corporation', 'ms_skype', 'Twitter, Inc.']
    
    #TODO: save pred coords or not? 
    gt_coords_arr = [] # save ref layout coords
    gt_files_arr = [] # save ref layout filename
    gt_shot_size_arr = [] # save ref layout screenshot size
    
    for template in os.listdir(os.path.join(ref_dir, matched_brand)):
        if template.startswith('.'): # skip hidden file
            continue
        img = cv2.imread(os.path.join(ref_dir, matched_brand, template))
        _, pred_boxes, _ = element_recognition(img, ele_model)
        pred_boxes = pred_boxes.numpy()
        gt_coords_arr.append(pred_boxes)
        gt_files_arr.append(os.path.join(ref_dir, matched_brand, template))
        gt_shot_size_arr.append(img.shape)
        del pred_boxes, img
        
    assert len(gt_files_arr) == len(gt_coords_arr) and len(gt_files_arr) == len(gt_shot_size_arr)
        
    return cfg, gt_coords_arr, gt_files_arr, gt_shot_size_arr
        
def layout_matcher(pred_boxes, img, 
                   gt_coords_arr, gt_files_arr, gt_shot_size_arr,
                   cfg):
    '''
    Run layout matcher
    :param pred_boxes: torch.Tensor|np.ndarray Nx4 bbox coords
    :param img: str|np.ndarray
    :param gt_coords_arr: reference layouts
    :param gt_files_arr: reference paths
    :param gt_shot_size_arr: reference shots sizes
    :param cfg
    :return max_s: maximum similarity
    :return max_site: matched site
    '''
    pred_boxes = pred_boxes.numpy() if isinstance(pred_boxes, torch.Tensor) else pred_boxes
    img = cv2.imread(img) if not isinstance(img, np.ndarray) else img
    shot_size = img.shape
    
    # If the number of reported boxes is less or equal to one, no point of continue
    if len(pred_boxes) <= 1:
        return 0, None
        
    # set initial similarity = 0
    max_s = 0
    max_site = None
    for j, gt_c in enumerate(gt_coords_arr):
        similarity, sim_mat, _, _, _, _,_, _ = \
                    bipartite_web(gt_c, pred_boxes, gt_shot_size_arr[j], shot_size, cfg)
        
        if similarity >= max_s:
            max_s = similarity
            max_site = gt_files_arr[j]

    return max_s, max_site