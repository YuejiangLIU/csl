import os
import torch
from torch import nn
import torchvision

import pdb

class HeadAndEmbedding(nn.Module):
    def __init__(self, head):
        super(HeadAndEmbedding, self).__init__()
        self.head = head

    def forward(self, x):
        return x, self.head(x)


def _alexnet_replace_fc(model):
    model.classifier = HeadAndEmbedding(model.classifier)
    return model


def resnet50_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
    return model


def vitb8_dino():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    return model


def vits14_dino():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    return model


def vitb14_dino():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    return model


def alexnet(ckpt=None, num_outputs=1000):
    model = torchvision.models.alexnet(pretrained=True)
    if num_outputs != 1000:
        model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=num_outputs)
    if os.path.isfile(ckpt):
        print("=> loading checkpoint from '{}'".format(ckpt))  
        checkpoint = torch.load(ckpt, map_location="cpu")
        try:
            state_dict = checkpoint['state_dict']
            # Remove all instances of 'module.' from the key
            basic_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                basic_state_dict[new_key] = v
            msg = model.load_state_dict(basic_state_dict, strict=True)
        except Exception:
            msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    else:
        print("=> Load the official Pytorch pretrained checkpoint")  
    return _alexnet_replace_fc(model)


def alexnet1(ckpt=None, num_outputs=1000):
    model = torchvision.models.alexnet(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, num_outputs)
    )    
    if os.path.isfile(ckpt):
        print("=> Load checkpoint from '{}'".format(ckpt))  
        checkpoint = torch.load(ckpt, map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    else:
        print(f"=> Cannot find {ckpt}, load the official Pytorch pretrained checkpoint instead")  
    return _alexnet_replace_fc(model)


def probe(d=2048, n_classes=1000, n_hidden=0):
    if n_hidden == 0:
        model = nn.Linear(d, n_classes).cuda()
    elif n_hidden == 1:
        h = int(d/2)
        model = nn.Sequential(
                    nn.Linear(d, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, n_classes)
                ).cuda()
    elif n_hidden == 2:
        h = int(d/2)
        model = nn.Sequential(
                    nn.Linear(d, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, n_classes)
                ).cuda()
    else:
        raise NotImplementedError
    return model
