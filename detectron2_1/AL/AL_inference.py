import os
os.chdir('..')
from configs import get_cfg
from modelling import *
from AL_merge import *
from detectron2.data import build_detection_test_loader
from datasets import CustomizeMapper
import json
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup

import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import funcy

from itertools import groupby

import pickle

def setup(args):
    """
    Create configs and perform basic setups.
    """

    # Initialize the configurations
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Ensure it uses appropriate names and architecture  
    cfg.MODEL.ROI_HEADS.NAME = 'ROIHeadsAL'
    cfg.MODEL.META_ARCHITECTURE = 'ActiveLearningRCNN'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # lower this threshold to get more boxes

    cfg.freeze()
    default_setup(cfg, args)
    return cfg



class Inference:
    
    def __init__(self, cfg, dataset_name, gt_json_path, write_json_path):

        # Initialize model
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.eval()

        # Build_model will not load weights, have to load weights explicitly
        checkpointer = DetectionCheckpointer(self.model)  
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # Metadata of dataset
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
          
        # TODO: customize dataset mapper
        self.dataset_name = dataset_name
        dataset_mapper = CustomizeMapper(cfg, is_train=False) 
        self.data_loader= build_detection_test_loader(
            cfg, dataset_name, mapper=dataset_mapper
        )
        
        self.gt_json_path = gt_json_path
        self.write_json_path = write_json_path
        
    def _feature_embed(self, features):
        '''Obtain feature embeddings from backbone'''
        # Avgpool over channels
        features_heatmaps = [torch.mean(features[k], dim=1) for k in features.keys()]
        
        # Adaptive pooling 2D --> Tensor(shape 5x8x8)]
        pooled_heatmaps = torch.cat([F.adaptive_avg_pool2d(f_map[None, ...], output_size=(8, 8)) \
                                     .view(1, 8, 8).detach().cpu() \
                                     for f_map in features_heatmaps], dim=0)
        pooled_heatmaps = pooled_heatmaps.view(-1)
        
        return pooled_heatmaps
    
    def _create_instance_dicts(self, outputs, image_id):
        '''Create dict from output'''
        instance_dicts = []
        
        # Get different fields from outputs
        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_boxes = instances.pred_boxes
        scores = instances.scores.cpu().numpy()
        scores_al = instances.scores_al.cpu().numpy() if 'scores_al' in instances.get_fields().keys() else None
        
        # For each bounding box
        for i, box in enumerate(pred_boxes):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            width = x2 - x1
            height = y2 - y1

            # Category_id starts from 1
            category_id = int(pred_classes[i] + 1) 
            score = float(scores[i])
            score_al = float(scores_al[i]) if (scores_al is not None) else None
            
            i_dict = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height], #Boxes are in XYWH instead of XYXY
                "score": score,
                "score_al": score_al,
            }

            instance_dicts.append(i_dict)

        return instance_dicts
    
    def _save_inference_dicts(self, **kwargs):
        raise Exception ("Not Implemented")
        
    def run(self, **kwargs):
        raise Exception ("Not Implemented")
        
        
        
        
class ALInference(Inference):
    '''Save inference dict for any dataset'''

    def _save_inference_dicts(self, feature_refs):
        
        self.write_json_path = self.write_json_path

        # Save predictions in coco format
        coco_instances_results = []
        
        for i, data in tqdm(enumerate(self.data_loader)):
            
            file_name = data[0]['file_name']
            image_id = data[0]['image_id']
            
            with torch.no_grad():
                # Predict 
                outputs = self.model.forward_al(data, feature_refs)
                
                # Convert to coco predictions format
                instance_dicts = self._create_instance_dicts(outputs[0], image_id)
                coco_instances_results.extend(instance_dicts)
                               
            if i % 100 == 0:
                # Write intermediate results
                with open(self.write_json_path, "w") as f:
                    json.dump(coco_instances_results, f)

        with open(self.write_json_path, "w") as f:
            json.dump(coco_instances_results, f)            

    def run(self, feature_refs=None):
        self._save_inference_dicts(feature_refs)



        
class FeatureEmb(Inference):
    
    '''Save feature embedding reference list for training data'''
    
    def __init__(self, cfg, dataset_name, gt_json_path, write_json_path):
        
        super().__init__(cfg, dataset_name, gt_json_path, write_json_path)
        
        self.inference_dict = []
        
        # Fix this threshold
        self.prThre = .25
        self.rcThre = .25
        
        # TODO: remove this 
#         with open(self.write_json_path, 'rt', encoding='UTF-8') as f:
#             self.inference_dict = json.load(f)
    
    def _save_inference_dicts(self):
        '''Collect all output+feature embedding for training data'''
        
        pooled_heatmaps = []
        for i, data in tqdm(enumerate(self.data_loader)):
            
            file_name = data[0]['file_name']
            image_id = data[0]['image_id']
            
            with torch.no_grad():
                ## Preprocess image
                images = self.model.preprocess_image(data)

                # Get features
                features = self.model.backbone(images.tensor)

                # Get heatmap of shape (320,)
                pooled_heatmap = self._feature_embed(features)
                pooled_heatmaps.append({'image_id':image_id,
                                        'heatmap':pooled_heatmap})
                
                outputs = self.model(data)

            # Convert to coco predictions format
            instance_dicts = self._create_instance_dicts(outputs[0], image_id)
            
            self.inference_dict.extend(instance_dicts)
            
            if i % 100 == 0:
                Write intermediate results
                with open(self.write_json_path, 'wt', encoding='UTF-8') as f:
                    json.dump(self.inference_dict, f)
                print(len(pooled_heatmaps))  
                print(pooled_heatmaps[i]['heatmap'].shape)
            
#         Write results
        with open(self.write_json_path, 'wt', encoding='UTF-8') as f:
            json.dump(self.inference_dict, f)
        
        return pooled_heatmaps
        
    def _filter_embed(self, bad_ids, feature_embed_dict):
        '''Only get images with worse performance'''
        feature_embed_dict_filter = funcy.lremove(lambda a: a['image_id'] not in bad_ids, 
                                                  feature_embed_dict)
        return feature_embed_dict_filter
    
    def _aggregate_score(self, pr_rc_dict):
        '''Aggregate precision,recall scores for each image averaged over all object area range and all categories'''
        agg_dict = []
        count = 0
        all_image_ids = set([x['image_id'] for x in pr_rc_dict])

        for image_id in tqdm(all_image_ids):
            values = funcy.lfilter(lambda a: a['image_id'] == image_id, pr_rc_dict)
            precision_all = funcy.lmap(lambda a: a['precision'], values)
            recall_all = funcy.lmap(lambda a:a['recall'], values)
            agg_dict.append({'image_id': image_id,
                             'precision': np.mean(precision_all),
                             'recall': np.mean(recall_all)})

            count += 1
            if count % 1000 == 0:
                print(count)
                
        return agg_dict
    
    def _evaluate_inference(self):
        
        coco_gt = COCO(self.gt_json_path)
        coco_dt = coco_gt.loadRes(self.write_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        
        pr_rc_dict = []
        
        # Report 20 boxes
        maxDet = 20
        for evalImg in tqdm(coco_eval.evalImgs):

            # Ignore None 
            if evalImg is None:
                continue

            image_id = evalImg['image_id']
            category_id = evalImg['category_id']
            aRng = evalImg['aRng']

            # Sort detection scores in descending order
            dtScores = np.asarray(evalImg['dtScores'][:maxDet])   
            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]

            ## Matched index
            dtm  = evalImg['dtMatches'][:,:maxDet][:,inds][0]   

            ## Ignored indices -- 0 for our dataset (no hard example)
            dtIg = evalImg['dtIgnore'][:,:maxDet][:,inds][0]
            gtIg = evalImg['gtIgnore']

            ## Count the number of ground-truths
            npig = np.count_nonzero(gtIg==0)

            ## If no groud-truth, neither bad/good example, simply ignore
            if npig == 0:
                continue
            
            ## If no detection, precision=recall=0
            if len(dtm) == 0:
                rc = 0
                pr = 0

            else:
                ## Get precision, recall for single image
                tps = np.logical_and(dtm,                 np.logical_not(dtIg) )
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                tp_sum = np.cumsum(tps).astype(dtype=np.float)
                fp_sum = np.cumsum(fps).astype(dtype=np.float)

                rc = tp_sum[-1] / npig
                pr = tp_sum[-1] / (fp_sum[-1]+tp_sum[-1]+np.spacing(1))

            pr_rc_dict.append({'image_id':image_id,
                               'category_id': category_id,
                               'aRng':aRng,
                               'precision':pr,
                               'recall':rc})  
            
        pr_rc_dict_agg = self._aggregate_score(pr_rc_dict)
        print("Length of aggregated dict", len(pr_rc_dict_agg))
        
        # TODO: group by image id and compute precision/recall
        # Bad ids
        fp_list = funcy.lfilter(lambda a: a['precision'] < self.prThre, pr_rc_dict_agg)
        fp_ids = funcy.lmap(lambda a: a['image_id'], fp_list)
        print('Number of FPs:', len(fp_ids))
        
        fn_list = funcy.lfilter(lambda a: a['recall'] < self.rcThre, pr_rc_dict_agg)
        fn_ids = funcy.lmap(lambda a: a['image_id'], fn_list)
        print('Number of FNs:', len(fn_ids))
        
        bad_ids = list(set(fp_ids + fn_ids))
        print('Number of Bad ids:', len(bad_ids))
        return bad_ids
            
    def run(self):
        
        # Collect all inferences
        pooled_heatmaps = self._save_inference_dicts()
        print('%d inferences done'%len(pooled_heatmaps))

        # Get bad image ids
        bad_ids = self._evaluate_inference()
        
        # Filter bad ids
        pooled_heatmaps_filter = self._filter_embed(bad_ids, feature_embed_dict=pooled_heatmaps)
        print('%d bad instances left'%len(pooled_heatmaps_filter))
            
        # Return bad images with their feature embeddings
        return pooled_heatmaps_filter
    
    
    
    
if __name__ == '__main__':
    parser = default_argument_parser()

    # Extra Configurations for dataset names and paths
    parser.add_argument("--train_dataset_name",  required=True, help="The Training Dataset Name")
    parser.add_argument("--json_annotation_train", required=True, help="The path to the training set JSON annotation")
    
    parser.add_argument("--al_dataset_name", required=True, help="The AL Dataset Name")
    parser.add_argument("--json_annotation_AL", required=True, help="The path to the AL set prediction JSON annotation")
    
    parser.add_argument("--json_annotation_merge", required=True, help="The path to the merged set JSON annotation")


    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    
    # If feature embedding is the object scoring function
    if cfg.AL.OBJECT_SCORING == 'feature_emb':
        
        # Get bad feature embedding from training set
        save_emb = FeatureEmb(cfg, dataset_name=args.train_dataset_name, 
                              gt_json_path=args.json_annotation_train, 
                              write_json_path='coco_results_coco_2017_train_subset1_feature_emb.json')

        pooled_heatmaps_filter = save_emb.run()
        all_heatmaps = funcy.lmap(lambda a: a["heatmap"], pooled_heatmaps_filter)
        
        # Save temporary feature embedding reference file
        with open('coco_2017_train_subset1_bad_feature_emb.pkl', 'wb') as handle:
            pickle.dump(all_heatmaps, handle)
        
        all_heatmaps = torch.stack(all_heatmaps)
        
        AL_inference = ALInference(cfg, dataset_name=args.al_dataset_name, 
                                   gt_json_path=None, 
                                   write_json_path=args.json_annotation_AL)
        
        
        AL_inference.run(all_heatmaps)
        
        
        
    else:
        
        AL_inference = ALInference(cfg, dataset_name=args.al_dataset_name, 
                                   gt_json_path=None, 
                                   write_json_path=args.json_annotation_AL)
        AL_inference.run()
        
        