import json
import os
import random
from typing import Any, Dict, List, Tuple, Callable

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.structures import Boxes, pairwise_iou
from detectron2.structures.image_list import ImageList
from detectron2.utils.visualizer import Visualizer
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
os.environ['QT_QPA_PLATFORM']='offscreen'


class DAGAttacker:
    def __init__(
        self,
        cfg: CfgNode,
        n_iter=200,
        gamma=0.5,
        nms_thresh=0.9,
        mapper: Callable = DatasetMapper,
    ):
        """Implements the DAG algorithm

        Parameters
        ----------
        cfg : CfgNode
            Config object used to train the model
        n_iter : int, optional
            Number of iterations to run the algorithm on each image, by default 150
        gamma : float, optional
            Perturbation weight, by default 0.5
        nms_thresh : float, optional
            NMS threshold of RPN; higher it is, more dense set of proposals, by default 0.9
        mapper : Callable, optional
            Can specify own DatasetMapper logic, by default DatasetMapper
        """
        self.n_iter = n_iter
        self.gamma = gamma

        # Modify config
        self.cfg = cfg.clone()  # cfg can be modified by model
        # To generate more dense proposals
        self.cfg.MODEL.RPN.NMS_THRESH = nms_thresh
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000

        # Init model
        self.model = build_model(self.cfg)
        self.model.eval()

        # Load weights
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        # Init dataloader on test dataset
        # mapper = BenignMapper(cfg, is_train=False)
        dataset_mapper = mapper(cfg, is_train=False)
        self.data_loader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST[0], mapper=dataset_mapper
        )

        self.device = self.model.device
        self.n_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # HACK Only specific for this dataset
        self.metadata.thing_classes = ["box", "logo"]
        # self.contiguous_id_to_thing_id = {
        #     v: k for k, v in self.metadata.thing_dataset_id_to_contiguous_id.items()
        # }

    def run_DAG(
        self,
        results_save_path="coco_instances_results.json",
        vis_save_dir=None,
        vis_conf_thresh=0.5,
    ):
        """Runs the DAG algorithm and saves the prediction results.

        Parameters
        ----------
        results_save_path : str, optional
            Path to save the results JSON file, by default "coco_instances_results.json"
        vis_save_dir : str, optional
            Directory to save the visualized bbox prediction images
        vis_conf_thresh : float, optional
            Confidence threshold for visualized bbox predictions, by default 0.5

        Returns
        -------
        Dict[str, Any]
            Prediction results as a dict
        """
        # Save predictions in coco format
        coco_instances_results = []

        # Runs on entire test dataset
        # For each image
        for i, batch in tqdm(enumerate(self.data_loader)):
            original_image = batch[0]["image"].permute(1, 2, 0).numpy()

            file_name = batch[0]["file_name"]
            basename = os.path.basename(file_name)
            image_id = batch[0]["image_id"]

            # Peform DAG attack
            print(f"[{i}/{len(self.data_loader)}] Attacking {file_name} ...")
#             self.attack_image(batch)
            perturbed_image, r_accum = self.attack_image(batch)
            
#             Permute axis into HxWxC and convert BGR to RGB for better display
            r_accum = r_accum[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1].astype(np.float32)
            r_accum = np.mean(r_accum, axis=-1)
            
            # Perform inference on perturbed image
            perturbed_image = (
                perturbed_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )
            outputs = self(perturbed_image)

            # Convert to coco predictions format
            instance_dicts = self._create_instance_dicts(outputs, image_id)
            coco_instances_results.extend(instance_dicts)

            if vis_save_dir:
                # Save adv predictions
                # Set confidence threshold
                instances = outputs["instances"]
                mask = instances.scores > vis_conf_thresh
                instances = instances[mask]

                v = Visualizer(perturbed_image[:, :, ::-1], self.metadata)
                vis_adv = v.draw_instance_predictions(instances.to("cpu")).get_image()

                # Save original predictions
                outputs = self(original_image)
                instances = outputs["instances"]
                mask = instances.scores > vis_conf_thresh
                instances = instances[mask]

                v = Visualizer(original_image[:, :, ::-1], self.metadata)
                vis_og = v.draw_instance_predictions(instances.to("cpu")).get_image()
                
                # Save single image
                cv2.imwrite(os.path.join(vis_save_dir, str(i)+'.jpg'), vis_og)
                cv2.imwrite(os.path.join(vis_save_dir, str(i)+'_adv.jpg'), vis_adv)
                plt.figure(figsize=(50, 50))
                sns.heatmap(r_accum, cmap='RdBu', vmin=-2, vmax=2, square=True, annot=False)
                plt.savefig(os.path.join(vis_save_dir, str(i)+'_noise.jpg'))
                plt.close()
                            
        # Save predictions as COCO results json format
        with open(results_save_path, "w") as f:
            json.dump(coco_instances_results, f)

        return coco_instances_results


    def attack_image(self, batched_inputs: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attack an image from the test dataloader

        Parameters
        ----------
        batched_inputs : List[Dict[str, Any]]
            A list containing a single element, the dataset_dict from the test dataloader

        Returns
        -------
        torch.Tensor
            [C, H, W], Perturbed image
        torch.Tensor
            [C, H, W], Accumulated noise
        """
        images = self.model.preprocess_image(batched_inputs)

        instances = batched_inputs[0]["instances"]

        # If no ground truth annotations, no choice but to skip
        if len(instances.gt_boxes) == 0:
            return self._post_process_image(images.tensor[0]), torch.zeros_like(images.tensor)

        # Preserve positive proposals
        target_boxes, target_labels = self._get_targets(batched_inputs)
        
        # Assign adv labels
        adv_labels = self._get_adv_labels(target_labels)

        # Get initial offset of xywh (after Fast-RCNN) 
        features = self.model.backbone(images.tensor)
        orig_logits, orig_proposal_delta = self._get_roi_heads_predictions(features, target_boxes)
        orig_proposal_delta = orig_proposal_delta.detach()
        
        # If no prediction, no choice but to skip
        if len(orig_logits) == 0:
            return self._post_process_image(images.tensor[0]), torch.zeros_like(images.tensor)
        
        # get the index where the predicted class belongs to foreground
        orig_startind = torch.argmax(orig_logits, dim=1)
        orig_cond = orig_startind < self.n_classes

        '''Initial attack '''
        images, r_accum = self._attack_first_round(images, 
                                                  features,
                                                  target_boxes,
                                                  orig_startind,
                                                  orig_cond,
                                                  orig_logits,
                                                  orig_proposal_delta)


        ''' Start DAG '''
        # Record gradients for image
        images.tensor.requires_grad = True
        
        for i in range(self.n_iter):
            features = self.model.backbone(images.tensor)
            logits, proposal_delta = self._get_roi_heads_predictions(features, target_boxes)

            # If no prediction, just break
            if len(logits) == 0:
                break
                
            # Predicted class
            startind = logits.argmax(dim=1)
            
            # Update active target set, i.e. filter for correctly predicted targets;
            active_cond = (startind == target_labels)
#             active_cond = (startind != adv_labels)
            
            '''Regression loss'''
            # get the index where the predicted class belongs to foreground
            cond = startind < self.n_classes

            assert (orig_cond.shape == cond.shape) 

            # only attack when both original image and perturbed image predict proposals as foreground and they fall in active set
            attack_cond = orig_cond & cond & active_cond
            print("In iteration %d:, "%i, "the number of attack regions is %d"%np.sum(attack_cond.detach().cpu().numpy()))
           
            _, orig_proposal_delta_f = self._filter_class(orig_startind, attack_cond, 
                                                          orig_logits, orig_proposal_delta)

            _, proposal_delta_f = self._filter_class(startind, attack_cond, 
                                                     logits, proposal_delta)    

            
            # If attack set is empty, break the loop
            if np.sum(attack_cond.detach().cpu().numpy()) == 0:
                break

            # Compute regression loss
            reg_loss = torch.norm(orig_proposal_delta_f-proposal_delta_f, p=2, dim=1).sum()
            
            '''Classification loss'''
            target_boxes_f = target_boxes[attack_cond]
            logits_f = logits[attack_cond]
            target_labels_f = target_labels[attack_cond]
            adv_labels_f = adv_labels[attack_cond]

            # Compute total loss
            target_loss = torch.mul(logits_f, torch.nn.functional.one_hot(target_labels_f, num_classes=self.n_classes+1)).sum()
            adv_loss = torch.mul(logits_f, torch.nn.functional.one_hot(adv_labels_f, num_classes=self.n_classes+1)).sum()

            # Make every target incorrectly predicted as the adversarial label
            cls_loss = adv_loss - target_loss
                            
            
            '''Combine 2 losses'''
            # Make two losses at the same scale
            dynamic_scale =  10**(abs(len(str(round(abs(reg_loss.item())))) - len(str(round(abs(cls_loss.item()))))))
            if len(str(round(abs(reg_loss.item())))) > len(str(round(abs(cls_loss.item())))):
                total_loss = reg_loss + cls_loss*dynamic_scale
            else:
                total_loss = reg_loss*dynamic_scale + cls_loss
                
            total_loss.backward()
            image_grad = images.tensor.grad.detach()
            # Apply perturbation on image
            with torch.no_grad():
                # Normalize grad
                image_perturb = (
                    self.gamma / image_grad.norm(float("inf"))
                ) * image_grad
                images.tensor += image_perturb
                r_accum += image_perturb

            # Zero gradients
            image_grad.zero_()
            self.model.zero_grad()
            
            print("In iteration %d: Regression Loss: %.4f, Classification Loss: %.4f, Total loss: %.4f"%(i, reg_loss.item(),
                                                                                                        cls_loss.item(),
                                                                                                        total_loss.item()))

        print(f"Done with attack. Total Iterations {i}")
        return self._post_process_image(images.tensor[0]), r_accum

    def _create_instance_dicts(
        self, outputs: Dict[str, Any], image_id: int
    ) -> List[Dict[str, Any]]:
        """Convert model outputs to coco predictions format

        Parameters
        ----------
        outputs : Dict[str, Any]
            Output dictionary from model output
        image_id : int

        Returns
        -------
        List[Dict[str, Any]]
            List of per instance predictions
        """
        instance_dicts = []

        instances = outputs["instances"]
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_boxes = instances.pred_boxes
        scores = instances.scores.cpu().numpy()

        # For each bounding box
        for i, box in enumerate(pred_boxes):
            box = box.cpu().numpy()
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            width = x2 - x1
            height = y2 - y1

            # HACK Only specific for this dataset
            category_id = int(pred_classes[i] + 1)
            # category_id = self.contiguous_id_to_thing_id[pred_classes[i]]
            score = float(scores[i])

            i_dict = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": score,
            }

            instance_dicts.append(i_dict)

        return instance_dicts
    
    
    def _attack_first_round(self, 
                          images: ImageList, 
                          features: Dict,
                          target_boxes: Boxes,
                          orig_startind: torch.Tensor,
                          orig_cond: torch.Tensor,
                          orig_logits: torch.Tensor,
                          orig_proposal_delta: torch.Tensor) -> Tuple[ImageList, torch.Tensor]:
        '''Launch initial attack for the regressor where the noise is randomly generated
        
        Parameters:
        images: test image
        orig_startind: original predicted classes for all proposals
        orig_cond: boolean tensor specify the non-background proposals
        orig_logits: original predicted classification scores(before softmax) for all proposals
        orig_proposal_delta: original predicted offset xywh for all proposals
        
        Returns:
        images: test image after attack
        r_accum: amount of perturbation 
        '''
        
        # randomly generate noise follow normal distribution
        noise = torch.normal(0, 0.1, size=images.tensor.shape).to(self.device)
        images.tensor += noise
        
        # Record gradients for image
        images.tensor.requires_grad = True
        features = self.model.backbone(images.tensor)
        logits, proposal_delta = self._get_roi_heads_predictions(features, target_boxes)
        
        # get the index where the predicted class belongs to foreground
        startind = torch.argmax(logits, dim=1)
        cond = startind < self.n_classes
        
        assert (orig_cond.shape) == (cond.shape)
        
        # only attack when both the original image and the perturbed image predict proposals as foreground 
        foreground_cond = orig_cond & cond
        _, orig_proposal_delta_f = self._filter_class(orig_startind, foreground_cond, 
                                                      orig_logits, orig_proposal_delta)
        
        _, proposal_delta_f = self._filter_class(startind, foreground_cond, 
                                                 logits, proposal_delta)    
        
        # Compute total loss
        reg_loss = torch.norm(orig_proposal_delta_f-proposal_delta_f, p=2, dim=1).sum()
        reg_loss.backward()
        image_grad = images.tensor.grad.detach()
        
        # Apply perturbation on image
        with torch.no_grad():
            # Normalize grad
            image_perturb = (
                self.gamma / image_grad.norm(float("inf"))
            ) * image_grad
            images.tensor += image_perturb
            r_accum = image_perturb # intialize r_accum

        # Zero-out gradients
        image_grad.zero_()
        self.model.zero_grad()
        return images, r_accum
        
    @torch.no_grad()
    def _post_process_image(self, image: torch.Tensor) -> torch.Tensor:
        """Process image back to [0, 255] range, i.e. undo the normalization

        Parameters
        ----------
        image : torch.Tensor
            [C, H, W]

        Returns
        -------
        torch.Tensor
            [C, H, W]
        """
        image = image.detach()
        image = (image * self.model.pixel_std) + self.model.pixel_mean

        return torch.clamp(image, 0, 255)
        
    def _get_adv_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Assign adversarial labels to a set of correct labels,
        i.e. randomly sample from incorrect classes.

        Parameters
        ----------
        labels : torch.Tensor
            [n_targets]

        Returns
        -------
        torch.Tensor
            [n_targets]
        """
        adv_labels = torch.zeros_like(labels)

        # FIXME Make this more efficient / vectorized?
        for i in range(len(labels)):
            # Include background class: self.n_classes
            incorrect_labels = [l for l in range(self.n_classes+1) if l != labels[i]]
            adv_labels[i] = random.choice(incorrect_labels)

        return adv_labels.to(self.device)

    @torch.no_grad()
    def _get_targets(
        self, batched_inputs: List[Dict[Any, Any]]
    ) -> Tuple[Boxes, torch.Tensor]:
        """Select a set of initial targets for the DAG algo.

        Parameters
        ----------
        batched_inputs : List[Dict[Any]]
            A list containing a single dataset_dict, transformed by a DatasetMapper.

        Returns
        -------
        Tuple[Boxes, torch.Tensor]
            target_boxes, target_labels
        """
        instances = batched_inputs[0]["instances"]
        gt_boxes = instances.gt_boxes.to(self.device)
        gt_classes = instances.gt_classes
#         print(gt_classes)
        
        images = self.model.preprocess_image(batched_inputs)

        # Get features
        features = self.model.backbone(images.tensor)

        # Get bounding box proposals
        proposals, _ = self.model.proposal_generator(images, features, None)
        proposal_boxes = proposals[0].proposal_boxes

        # Get proposal boxes' classification scores
        predictions = self._get_roi_heads_predictions(features, proposal_boxes)
        # Scores (softmaxed) for a single image, [n_proposals, n_classes + 1]
        scores = self.model.roi_heads.box_predictor.predict_probs(
            predictions, proposals
        )[0]

        return self._filter_positive_proposals(
            proposal_boxes, scores, gt_boxes, gt_classes
        )

    def _get_roi_heads_predictions(
        self, features: Dict[str, torch.Tensor], proposal_boxes: Boxes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Partial execution of prediction head
        Args：
         features: feature pyramid (P2, P3, P4, P5, P6)
         proposal_boxes: predicted RoIs after RPN
         
        Returns:
         logits: predicted classification scores before softmax NxC
         proposal_delta: predicted delta xywh Nx(4*C) (offset from RoIs)
        '''
        roi_heads = self.model.roi_heads
        features = [features[f] for f in roi_heads.box_in_features]
        box_features = roi_heads.box_pooler(features, [proposal_boxes])
        box_features = roi_heads.box_head(box_features)

        logits, proposal_deltas = roi_heads.box_predictor(box_features)
        del box_features

        return logits, proposal_deltas
    
    def _filter_class(
          self, 
          startind: torch.Tensor,
          cond: torch.Tensor, 
          logits: torch.Tensor,
          proposal_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Filter predictions with conditions, reshape predicted box
        Args:
         startind: predicted classes for all regions 
         cond: boolean tensors 
         logits: predicted classification scores before softmax NxC
         proposal_delta: predicted delta xywh Nx(4*C) (offset from RoIs)
         
        Returns:
         logits: filtered predicted classification scores N'xC
         regbox: filtered predicted boxes only for most probable class N'x4
        '''
        
        logits, proposal_delta, startind = logits[cond], proposal_delta[cond], startind[cond]
        
        # convert it into one-hot matrix
        onehot = torch.nn.functional.one_hot(startind, num_classes=self.n_classes)
        onehot_tile = onehot.repeat_interleave(4, dim=1)
        
        # get predicted bounding box only for the most probable class
        regbox = torch.mul(proposal_delta, onehot_tile)
        regbox = regbox[regbox!=0].view(-1, 4)
        
        return logits, regbox
        
    def _filter_positive_proposals(
        self,
        proposal_boxes: Boxes,
        scores: torch.Tensor,
        gt_boxes: Boxes,
        gt_classes: torch.Tensor,
    ) -> Tuple[Boxes, torch.Tensor, torch.Tensor]:
        """Filter for desired targets for the DAG algo

        Parameters
        ----------
        proposal_boxes : Boxes
            Proposal boxes directly from RPN
        scores : torch.Tensor
            Softmaxed scores for each proposal box
        gt_boxes : Boxes
            Ground truth boxes
        gt_classes : torch.Tensor
            Ground truth classes

        Returns
        -------
        Tuple[Boxes, torch.Tensor]
            filtered_target_boxes, corresponding_class_labels
        """
        n_proposals = len(proposal_boxes)

        proposal_gt_ious = pairwise_iou(proposal_boxes, gt_boxes)

        # For each proposal_box, pair with a gt_box, i.e. find gt_box with highest IoU
        # IoU with paired gt_box, idx of paired gt_box
        paired_ious, paired_gt_idx = proposal_gt_ious.max(dim=1)

        # Filter for IoUs > 0.1
        iou_cond = paired_ious > 0.1

        # Filter for score of proposal > 0.1
        # Get class of paired gt_box
        gt_classes_repeat = gt_classes.repeat(n_proposals, 1)
        paired_gt_classes = gt_classes_repeat[torch.arange(n_proposals), paired_gt_idx]
        
        # Get scores of corresponding class
        paired_scores = scores[torch.arange(n_proposals), paired_gt_classes]
        score_cond = paired_scores > 0.1

        # Filter for positive proposals and their corresponding gt labels
        cond = iou_cond & score_cond

        return proposal_boxes[cond], paired_gt_classes[cond].to(self.device)

    def __call__(self, original_image):
        """Simple inference on a single image

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
