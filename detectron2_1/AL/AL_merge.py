
import funcy
import random
import json
import numpy as np
from tqdm import tqdm

class MergeDataset:
    
    def __init__(self, cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path):
        
        self.cfg = cfg
        # Read training data json file: {"images":[], "annotations":[]}
        with open(orig_dataset_json, 'rt', encoding='UTF-8') as f:
            self.orig_data_dict = json.load(f)
            
        # Read AL data json file: {"images":[], "annotations":[]}
        with open(al_dataset_json, 'rt', encoding='UTF-8') as f:
            self.al_data_dict = json.load(f)           
            
        # Read AL data prediction file: [{"image_id":, "bbox":, "score":, "category_id":, "score_al":}, {}]
        with open(al_pred_dict, 'rt', encoding='UTF-8') as f:
            self.al_pred_dict = json.load(f)
            
        # Merge dataset json path
        self.merge_dataset_path = merge_dataset_path
        
    def _merge_datadict(self, **kwargs):
        raise Exception ("Not Implemented")
            
    def run(self, **kwargs):
        raise Exception ("Not Implemented")
        
        
class SelectALDataset(MergeDataset):
    '''Merge original datadict with selected AL datadict'''
    def __init__(self, cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path):
        
        # Inherit all 
        super(SelectALDataset, self).__init__(cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path)
    
    def _aggregate_score(self):
        '''Aggregate proposals scores for single image'''
        count = 0
        all_image_ids = set([x['image_id'] for x in self.al_pred_dict])

        for image_id in all_image_ids:
        
            values = funcy.lfilter(lambda a: a['image_id'] == image_id, self.al_pred_dict)
            if self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'avg':
                agg_score = np.mean(funcy.lmap(lambda a: a['score_al'], values))
            elif self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'max':
                agg_score = np.max(funcy.lmap(lambda a: a['score_al'], values))
            elif self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'sum':
                agg_score = np.sum(funcy.lmap(lambda a: a['score_al'], values))
            elif self.cfg.AL.IMAGE_SCORE_AGGREGATION == 'random':
                agg_score = random.sample(funcy.lmap(lambda a: a['score_al'], values), 1)[0]

            for r in values:
                r['image_score'] = agg_score
                
            count += 1
            if count % 1000 == 0:
                print(count)
        

    def _select_topn(self, n):
        '''Get topn images with topn uncertainty score'''
        
        print('Start score aggregation') 
        if self.cfg.AL.OBJECT_SCORING == 'feature_emb': # for feature embedding method is already aggregted
            pass
        else:
            self._aggregate_score()
        print('Score aggregation finished')
        
        if self.cfg.AL.OBJECT_SCORING == 'feature_emb':
            image_scores = [fs['score_al'] for fs in self.al_pred_dict]
        else:
            image_scores = [fs['image_score'] for fs in self.al_pred_dict]
            
        image_ids = [fs['image_id'] for fs in self.al_pred_dict]
        
        # Remove duplicates: we only want to look at image scores 
        image_scores = list(dict.fromkeys(image_scores))
        image_ids = list(dict.fromkeys(image_ids))
        
        # Select TopN uncertain image ids
        print('Start finding topN image ids')
        sorted_image_scores = np.argsort(image_scores)[::-1] 
        print(sorted_image_scores)
        selected_image_ids = np.asarray(image_ids)[sorted_image_scores[:n]]
        
        al_select_images, al_select_annotations = self._select_dict_byid(selected_image_ids)
        return al_select_images, al_select_annotations
    
    
    def _select_dict_byid(self, selected_image_ids):
        print('Number of AL instances before filtering:', len(self.al_data_dict))
        al_select_images = funcy.lfilter(lambda a: a['id'] in selected_image_ids.tolist(), 
                                       self.al_data_dict["images"])
        
        al_select_annotations = funcy.lfilter(lambda a: a['image_id'] in selected_image_ids.tolist(), 
                                       self.al_data_dict["annotations"])     
        
        print('Number of AL instances after filtering:', len(al_select_images))
        print('Number of AL bboxes after filtering:', len(al_select_annotations))

        return al_select_images, al_select_annotations

        
    def _merge_datadict(self):
        '''Reindex datadict when merging two datadict'''
        datadict_orig = self.orig_data_dict
        al_select_images, al_select_annotations = self._select_topn(n=len(self.al_data_dict["images"])//4) # Select 1/4 of AL data
        
        # Directly merge because there is no image id conflict
        merged_dict = datadict_orig
        merged_dict['images'].extend(al_select_images)
        merged_dict['annotations'].extend(al_select_annotations)

        return merged_dict
    
    def run(self):
        merged_dict = self._merge_datadict()
        with open(self.merge_dataset_path, 'wt', encoding='UTF-8') as f:
            json.dump(merged_dict, f)
            
        return merged_dict
    
    
    
    
class PseudoALDataset(MergeDataset):
    '''Merge training dataset with pseudo labelled AL dataset'''
    def __init__(self, cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path):
        
        super(PseudoALDataset, self).__init__(cfg, orig_dataset_json, 
                 al_dataset_json, al_pred_dict, 
                 merge_dataset_path)

        
    def _generate_pseudo_dict(self):
        coco_images = []
        coco_annotations = []

        for _, image_dict in enumerate(self.al_pred_dict):

            instance_gt = funcy.lfilter(lambda a: a["id"] == image_dict["image_id"], 
                                        self.al_data_dict["images"])[0]
            
            instance_width = instance_gt['width']
            instance_height = instance_gt['height']
            instance_file_name = instance_gt['file_name']

            if image_dict["image_id"] not in [x["id"] for x in coco_images]:
                coco_image = {
                    "id": image_dict["image_id"],
                    "width": instance_width,
                    "height": instance_height,
                    "file_name": instance_file_name,
                }
                coco_images.append(coco_image)

            x1, y1, width, height = image_dict['bbox']
            category_id = image_dict['category_id']
            ann = {
                "area": width * height,
                "image_id": image_dict["image_id"],
                "bbox": [x1, y1, width, height],
                "category_id": category_id,
                "id": len(coco_annotations) + 1, # id for box, need to be continuous
                "iscrowd": 0
                }

            coco_annotations.append(ann)
        return coco_images, coco_annotations

        
    def _merge_datadict(self):
        '''Reindex datadict when merging two datadict'''
        datadict_orig = self.orig_data_dict
        coco_images, coco_annotations = self._generate_pseudo_dict() 
        
        # Directly merge because there is no image id conflict
        merged_dict = datadict_orig
        merged_dict['images'].extend(coco_images)
        merged_dict['annotations'].extend(coco_annotations)

        return merged_dict
    
    
    def run(self):
        merged_dict = self._merge_datadict()
        with open(self.merge_dataset_path, 'wt', encoding='UTF-8') as f:
            json.dump(merged_dict, f)
            
        return merged_dict
    