#!/usr/bin/python
from fastai import *
from fastai.vision import get_image_files
from fastai.vision import create_cnn, models
from fastai.vision import ImageDataBunch, imagenet_stats, get_transforms
import numpy as np

def main():
    path = untar_data(URLs.PETS)
    path_anno = path/'annotations'
    path_images = path/'images'
    fnames = get_image_files(path_images)

    np.random.seed(2)
    pat = re.compile(r'/([^/]+)_\d+.jpg$')
   
    bs = 6
    # create data loaderi
    data =ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(4)


if __name__ == '__main__':
    main()
