#!/usr/bin/python
from fastai import *
from fastai.vision import get_image_files
from fastai.vision import create_cnn, models
from fastai.vision import ImageDataBunch, imagenet_stats, get_transforms
import numpy as np
import torch

class Catsdogs(object): 
    def __init__(self):
        self.x = "hello"              
        path = untar_data(URLs.PETS)
        path_anno = path/'annotations'
        path_images = path/'images'
        fnames = get_image_files(path_images)

        np.random.seed(2)
        pat = re.compile(r'/([^/]+)_\d+.jpg$')
       
        bs = 6
        # create data loaderi
        self.data =ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
    
        self.learner = create_cnn(self.dataloader, models.resnet34, metrics=error_rate)

    def getLearner(self):
        """ return fastai Learner Object """ 
        return self.learner

if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = Catsdogs() 
    learner = model.getLearner()
