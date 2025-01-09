# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from .text_feature_gen import make_text_feature_generator
from .inference import make_roi_action_post_processor
from .loss import make_roi_action_loss_evaluator
from .metric import make_roi_action_accuracy_evaluator
from STCLIP.utils.comm import all_reduce

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import time

class ROIActionHead(torch.nn.Module):
    """
    Generic Action Head class.
    """

    def __init__(self, cfg, actionlist, actiondict, actiontoken, device):
        super(ROIActionHead, self).__init__()
        self.cfg = cfg
        self.text_feature_generator = make_text_feature_generator(cfg, actionlist, actiondict, actiontoken, device)
        self.post_processor = make_roi_action_post_processor(cfg)
        self.loss_evaluator = make_roi_action_loss_evaluator(cfg)
        self.accuracy_evaluator = make_roi_action_accuracy_evaluator(cfg)
        self.device = device

    def forward(self, boxes, objects=None, keypoints=None, extras={}, part_forward=-1):
        # In training stage, boxes are from gt.
        # In testing stage, boxes are detected by person detector
        assert not (self.training and part_forward >= 0)

        if self.training:
            proposals = self.loss_evaluator.sample_box(boxes)
            image_paths = []
            for box in proposals:
                image_root = "data/jhmdb/videos/"
                movie_id = box.get_field('movie_id')
                timestamp = box.get_field('timestamp')
                str_timestamp = str(timestamp).zfill(5)
                image_root = image_root + movie_id + '/' + str_timestamp + '.png'
                image_paths.append(image_root)
        else:
            proposals = boxes
            image_paths = []
            for i in range(len(extras['movie_ids'])):
               image_root = "data/jhmdb/videos/"
               movie_id = extras['movie_ids'][i]
               timestamp = extras['timestamps'][i]
               str_timestamp = str(timestamp).zfill(5)
               image_root = image_root + movie_id + '/' + str_timestamp + '.png'
               image_paths.append(image_root)

        # get label text feature and image feature, also take person feature,object feature
        text_features, x = self.text_feature_generator(image_paths, proposals)
                
        x = x / x.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        action_logits = torch.einsum("bd,bkd->bk", x, text_features)
        action_logits = action_logits / 0.07
                
        # get result confident score
        if not self.training:
            result = self.post_processor((action_logits,), boxes)
            return result, {}, {}, {}

        box_num = action_logits.size(0)
        box_num = torch.as_tensor([box_num], dtype=torch.float32, device=action_logits.device)
        all_reduce(box_num, average=True)

        loss_dict, loss_weight = self.loss_evaluator(
            [action_logits], box_num.item()
        )

        metric_dict = self.accuracy_evaluator(
            [action_logits], proposals, box_num.item(),
        )

        return (
            [None, None, None, None],
            loss_dict,
            loss_weight,
            metric_dict,
        )

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map

def build_roi_action_head(cfg, actionlist, actiondict, actiontoken, device):
    return ROIActionHead(cfg, actionlist, actiondict, actiontoken, device)
