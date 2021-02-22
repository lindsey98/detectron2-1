# Global configuration
from layout import *
from siamese import *
from element_detector import *
from credential import *


# element recognition model
ele_cfg, ele_model = element_config(rcnn_weights_path = 'output/website_lr0.001/model_final.pth', 
                                    rcnn_cfg_path='configs/faster_rcnn_web.yaml')

# CRP classifier
cls_model = credential_config(checkpoint='credential_classifier/output/website_finetune/websiteAL/WebsiteAL/FCMaxV2_0.05.pth.tar')

# siamese model
pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277, 
                                                weights_path='phishpedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='phishpedia/expand_targetlist/')
siamese_ts = 0.9 # threshold is 0.9 in phish-discovery

# brand-domain dictionary
domain_map_path = 'phishpedia/domain_map.pkl'

# layout templates
layout_cfg_dir = 'layout_matcher/configs.yaml'
layout_ref_dir = 'layout_matcher/layout_reference'
layout_ts = 0.4 # TODO: set this ts
