#encoding:utf-8
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models
import tifffile as tiff
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import get_loader, get_data_path
from metrics import scores
import os
def test(img_path):

    # Setup image

    # print ("Read Input Image from : {}".format(args.img_path))
    # img = misc.imread(args.img_path)
    img = tiff.imread(img_path)

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = torch.load(args.model_path)
    model.eval()


    model.cuda(0)
    images = Variable(img.cuda(0))

    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
    decoded = np.array(loader.decode_segmap(pred[0]),dtype=np.uint8)
    return decoded
    # print (np.unique(pred))
    # misc.imsave(args.out_path, decoded)
    # print ("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='/models/',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='Vaihingen',
                        help='Dataset to use [\'pascal, Vaihingen, Potsdam etc\']')
    # parser.add_argument('--img_path', nargs='?', type=str, default='/home/s405/pengcheng/project/pytorch-semseg-master/data/Vaihingen/image_data/train/image/area28_2_0_1.tif',
    #                     help='Path of the input image'
    args = parser.parse_args()
    test_path ='/data/Vaihingen/image_data/test/image/'
    save_path = '/data/Vaihingen/image_data/test/'
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True)
    model = torch.load(args.model_path)
    model.eval()
    model.cuda(0)
    img_list = os.listdir(test_path)
    for i in img_list:
        img = tiff.imread(test_path+i)
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= loader.mean
        img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()

        images = Variable(img.cuda(0))
        outputs = model(images)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        decoded = np.array(loader.decode_segmap(pred[0]), dtype=np.uint8)
        tiff.imsave(save_path+i,decoded)




