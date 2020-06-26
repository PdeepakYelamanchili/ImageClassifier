import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from get_predict_args import get_predict_args
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import argparse


def process_image(image):
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(Image.open(image))

def load_model(filepath):

    checkpoint = torch.load(filepath)
    model = getattr(models,'vgg16')(pretrained=True)
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image_path, checkpoint_path, top_k, gpu):
   
    with open('./cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print('prediction started for {}'.format(image_path))
    model = load_model(checkpoint_path)
    if gpu == True:
        # use GPU
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model.to(device)
    
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)
    log_probs = model.forward(torch_image)
    linear_probs = torch.exp(log_probs)
    top_probs, top_labels = linear_probs.topk(top_k)
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    
    flower_nms = [cat_to_name[str(c)] for c in top_labels]
    print('Probabilities for given image : ',  top_probs)
    print('Categories are:    ', flower_nms)   
    flower_num = image_path.split('/')[-2]
    title_ = cat_to_name[flower_num]
    print('True category for given image: ', title_)



if __name__ == "__main__":
    image_path, checkpoint_path, top_k, gpu = get_predict_args()
    # predict the image
    predict(image_path, checkpoint_path, top_k, gpu)
       