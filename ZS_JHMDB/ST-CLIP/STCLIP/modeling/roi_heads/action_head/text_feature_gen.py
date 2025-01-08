from torch import nn
from STCLIP.modeling import registry
import torch
import numpy as np
import os

import sys
sys.path.append('clip')
from clip import clip

from PIL import Image
import time

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

@registry.TEXT_FEATURE_GENERATOR.register("CLIPencoder")
class CLIPencoder(nn.Module):
    def __init__(self, cfg, actionlist, actiondict, actiontoken, device):
        super(CLIPencoder, self).__init__()

        # for text encoder
        self.actionlist = actionlist
        self.actiondict = actiondict
        self.actiontoken = actiontoken

        self.device = device
        self.clipmodel, self.preprocess = clip.load('ViT-B/16', device=self.device, jit=False, return_intermediate_text_feature=0)

        for name, param in self.clipmodel.named_parameters():
            if "projector" in name or "prompts_generator" in name or "temporal" in name or "lora" in name:
                pass
            else:
                param.requires_grad = False
        
        # memory len
        self.memory_before = 4
        self.memory_after = 4

        # adapter(2-layer FFN) for person feature
        self.person_dim = self.clipmodel.visual.class_embedding.shape[0]
        self.adapter = nn.Sequential(nn.Linear(self.person_dim, self.person_dim * 4),
                                     QuickGELU(),
                                     nn.Linear(self.person_dim * 4, self.person_dim))        
        self.ln_person = nn.LayerNorm(self.person_dim)

    def generate_person_feature_each_frame(self,path,proposals):
        images = []
        if proposals is None:
            return None
 
        image = Image.open(path).convert("RGB")
        box_num = len(proposals)
        for j in range(box_num):
            bbox = proposals.bbox
            
            x1 = round(bbox[j][0].item())
            y1 = round(bbox[j][1].item())
            x2 = round(bbox[j][2].item())
            y2 = round(bbox[j][3].item())
            
            # crop person or object
            wanted = image.crop((x1, y1, x2, y2))
            images.append(self.preprocess(wanted))


        image_input = torch.tensor(np.stack(images)).to(self.device)
        wanted_features = self.clipmodel.encode_image(image_input,False).float()

        return wanted_features
    
    def generate_image_feature(self,image_paths,proposals):
        images = []

        for i in range(len(image_paths)):
            path = image_paths[i]
            image = Image.open(path).convert("RGB")

            box_num = len(proposals[i])
            for _ in range(box_num):
                images.append(self.preprocess(image))
            
        image_input = torch.tensor(np.stack(images)).to(self.device)
        image_feature = self.clipmodel.encode_image(image_input).float()

        return image_feature

    def visual_prompt_person(self, image_paths, proposals, image_feature, tFeature):
        ret = None
        text_ret = None
        # for each frame
        for i in range(len(image_paths)):
            
            # if there are no person detected in this frame, we skip it
            if len(proposals[i]) == 0:
                continue

            movie_path = image_paths[i].rsplit('/', 1)[0]
            movie_len = len([name for name in os.listdir(movie_path) if os.path.isfile(os.path.join(movie_path, name))])
            timestamp = int(image_paths[i].split('/')[-1].split('.')[0]) # 00015.png -> 15
            if (timestamp > self.memory_before) and (timestamp + self.memory_after <= movie_len):
                memory_start = timestamp - self.memory_before
                memory_end = timestamp + self.memory_after
            elif timestamp <= self.memory_before:
                memory_start = 1
                memory_end = timestamp + self.memory_after + (self.memory_before - timestamp + 1)
            elif timestamp + self.memory_after > movie_len:
                memory_start = timestamp - self.memory_before - (timestamp + self.memory_after - movie_len)
                memory_end = movie_len

            # generate person feature for current frame
            person_feature = self.generate_person_feature_each_frame(image_paths[i],proposals[i])
            person_feature = person_feature + self.adapter(self.ln_person(person_feature))
            
            # gather current frame and neighboring frames
            images = []
            for t in range(memory_start,memory_end+1):
                if t >= 1:    
                    if t > movie_len:
                        break
                    str_timestamp = str(t).zfill(5)
                    path = movie_path + '/' + str_timestamp + '.png'
                    image = Image.open(path).convert("RGB")
                    images.append(self.preprocess(image))
                
            image_input = torch.tensor(np.stack(images)).to(self.device)

            text_features = tFeature.unsqueeze(0)
            interaction_feature,text_features = self.clipmodel.encode_image_with_prompts(image_input, person_feature, text_features)
            interaction_feature = interaction_feature.float()
            
            # all person in the current frame use the same text feature
            person_num = len(proposals[i])
            text_features = text_features.expand(person_num, -1, -1)
            text_features = text_features.float()

            # concat interaction feature for all frames
            if ret is None:
                ret = interaction_feature
            else:
                ret = torch.cat((ret,interaction_feature),0)

            # concat text feature for all frames
            if text_ret is None:
                text_ret = text_features
            else:
                text_ret = torch.cat((text_ret,text_features),0)
                
        ret = ret + image_feature

        return ret,text_ret

    # with input image_paths
    def forward(self, image_paths, proposals):

        actiontoken = clip.tokenize([a for a in self.actionlist]).cuda()
        tFeature = self.clipmodel.encode_text_original(actiontoken).float()

        image_feature = self.generate_image_feature(image_paths,proposals)

        interaction_feature,text_features = self.visual_prompt_person(image_paths,proposals,image_feature,tFeature)

        return text_features, interaction_feature 

def make_text_feature_generator(cfg, actionlist, actiondict, actiontoken, device):
    func = registry.TEXT_FEATURE_GENERATOR[cfg.MODEL.ROI_ACTION_HEAD.TEXT_FEATURE_GENERATOR]
    return func(cfg, actionlist, actiondict, actiontoken, device)
