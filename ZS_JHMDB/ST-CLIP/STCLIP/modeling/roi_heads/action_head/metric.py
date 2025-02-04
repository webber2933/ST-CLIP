# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from STCLIP.modeling.utils import cat


class ActionAccuracyComputation(object):
    def __init__(self, num_pose, num_object, num_person, cfg):
        
        all_label_file = open(cfg.ALL_LABEL, "r")
        data = all_label_file.read()
        self.original_labels = data.split("\n")
        self.original_labels = self.original_labels[:-1]
        all_label_file.close()
        
        train_label_file = open(cfg.TRAIN_LABEL, "r")
        data = train_label_file.read()
        self.our_train_labels = data.split("\n")
        self.our_train_labels = self.our_train_labels[:-1]
        train_label_file.close()
        
        self.num_pose = num_pose
        self.num_object = num_object
        self.num_person = num_person

    def logic_iou(self, pred, label):
        device = pred.device

        version = torch.__version__
        if eval('.'.join(version.split('.')[:2]))>=1.3:
            pred = pred.bool()
            label = label.bool()

        label_union = (pred | label).float().sum(dim=1)
        label_inter = (pred & label).float().sum(dim=1)
        replacer = torch.ones_like(label_union, device=device)
        zero_mask = label_union == 0
        label_inter = torch.where(zero_mask, replacer, label_inter)
        label_union = torch.where(zero_mask, replacer, label_union)
        return label_inter / label_union

    def __call__(self, class_logits, proposals, avg_box_num):
        class_logits = [logits.detach() for logits in class_logits]
        class_logits = cat(class_logits, dim=0)
        #assert class_logits.shape[1] == (self.num_pose + self.num_object + self.num_person), \
        #    "The shape of tensor class logits doesn't match total number of action classes."

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        our_labels = labels.clone()
        for i in range(len(our_labels)):
            for j in range(self.num_pose):
                our_labels[i][j] = labels[i][self.original_labels.index(self.our_train_labels[j])]


        metric_dict = {}
        if self.num_pose>0:
            pose_label = our_labels[:, :self.num_pose].argmax(dim=1)
            pose_pred = class_logits[:, :self.num_pose].argmax(dim=1)
            accuracy_pose_action = pose_label.eq(pose_pred).float().sum()
            metric_dict["accuracy_pose_action"] = accuracy_pose_action / avg_box_num

        interaction_label = our_labels[:, self.num_pose:]
        interaction_logits = class_logits[:, self.num_pose:]
        interaction_pred = interaction_logits.sigmoid() > 0.5

        if self.num_object>0:
            object_label = interaction_label[:, :self.num_object]
            object_pred = interaction_pred[:, :self.num_object]
            accuracy_object_interaction = self.logic_iou(object_pred, object_label)
            metric_dict["accuracy_object_interaction"] = accuracy_object_interaction.sum() / avg_box_num

        if self.num_person>0:
            person_label = interaction_label[:, self.num_object:]
            person_pred = interaction_pred[:, self.num_object:]
            accuracy_person_interaction = self.logic_iou(person_pred, person_label)
            metric_dict["accuracy_person_interaction"] = accuracy_person_interaction.sum() / avg_box_num

        return metric_dict


def make_roi_action_accuracy_evaluator(cfg):
    num_pose = cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES
    num_object = cfg.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES
    num_person = cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES
    return ActionAccuracyComputation(num_pose, num_object, num_person, cfg)