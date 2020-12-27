#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import detectron2.utils.comm as comm
import yaml
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader, build_batch_data_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetMapper
from detectron2_1.datasets import CustomizeMapper
from detectron2_1.engine import cushooks


from detectron2_1.configs import get_cfg
from detectron2_1.modelling import *

from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.samplers import TrainingSampler

from detectron2.utils.events import EventStorage
import json
from detectron2.utils.logger import log_every_n_seconds
import time
import datetime

# Implement evaluation here
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return COCOEvaluator(dataset_name, cfg, False, cfg.OUTPUT_DIR)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # Insert custom data loading logic
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=CustomizeMapper(cfg, is_train=True)
        )
    
    # Build training loader with batch size 1 and no shuffle
    @classmethod
    def build_cus_train_loader(cls, cfg):
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        dataset = DatasetFromList(dataset_dicts)
        mapper = CustomizeMapper(cfg, is_train=False) # set training = False 
        dataset = MapDataset(dataset, mapper)

        sampler = TrainingSampler(len(dataset), shuffle=False) # no shuffle
        
        # # This will be an infinite data loader 
        return len(dataset), build_batch_data_loader(
                dataset,
                sampler,
                1, # batch size equals 1
                aspect_ratio_grouping=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS)

    # Define loss recorder hook
    @classmethod
    def loss_recorder(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        cfg_copy = cfg.clone()
        
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running loss calculations on all training data ...")
        
        # Get customized training loader 
        len_dataset, data_loader = cls.build_cus_train_loader(cfg_copy)
        logger.info("Length of dataset is: {}".format(len_dataset))
        
        # Collect all losses 
        try:
            with open(cfg_copy.OUTPUT_DIR_LOSS, 'rt', encoding='UTF-8') as handle:
                result_dict = json.load(handle)
        except:
            result_dict = {}
            
        
        total = len_dataset
        total_compute_time = 0
        start_time = time.perf_counter()
        
        # Record loss
        with EventStorage() as storage:
    
            for i, data in enumerate(data_loader):
                start_compute_time = time.perf_counter()
                
                model.zero_grad() # zero-out gradients
                loss_dict = model(data) 
                losses = sum(loss_dict.values()).detach().cpu().item() # sum all 4 losses
                if data[0]['image_id'] in result_dict.keys():
                    result_dict[data[0]['image_id']].append(losses)
                else:
                    result_dict[data[0]['image_id']] = [losses]
                    
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = i + 1
                seconds_per_img = total_compute_time / iters_after_start

                # Log on every second
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - i - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Training done {}/{}. {:.4f} s / img. ETA={}".format(
                        i + 1, total, seconds_per_img, str(eta)
                    ), n=60,
                )
                    
                # Break the infinite data stream
                if i >= len_dataset - 1:
                    model.zero_grad()
                    break 
            comm.synchronize()
                    
        # Write loss dict 
        with open(cfg_copy.OUTPUT_DIR_LOSS, 'wt', encoding='UTF-8') as handle:
            json.dump(result_dict, handle)
        logger.info("Finish all loss calculations at step {} and written to {} \n".format(step, cfg_copy.OUTPUT_DIR_LOSS))

        return result_dict



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    # Can create custom configs fields here too
    cfg.merge_from_list(args.opts)
    
    # Ensure it uses appropriate names and architecture  
    cfg.MODEL.ROI_HEADS.NAME = 'ROIHeadsAL'
    cfg.MODEL.META_ARCHITECTURE = 'ActiveLearningRCNN'

    # Create directory if does not exist
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg



def load_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def main(args):
    cfg = setup(args)

    # Load cfg as python dict
    config = load_yaml(args.config_file)

    # If evaluation
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)

    # If training
    else:
        trainer = Trainer(cfg)
        # Load model weights (if specified)
        trainer.resume_or_load(resume=args.resume)
        if cfg.TEST.AUG.ENABLED:
            trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
            
        # Register customized hook -- loss logger
        trainer.register_hooks(
            [cushooks.CusEvalHook(cfg.SOLVER.CHECKPOINT_PERIOD_LOSS, lambda: trainer.loss_recorder(cfg, trainer.model))])  

        # Will evaluation be done at end of training?
        res = trainer.train()

    return res


if __name__ == "__main__":
    # Create a parser with some common arguments
    parser = default_argument_parser()
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



