#encoding:utf-8
import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from DFCN.uitls import recursive_glob
from tqdm import tqdm
from torch.utils import data
import tifffile as tiff


class PotsdamLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=256):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 6
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        # self.mean = [0.47270723, 0.32049636, 0.31585335]
        # self.std = [0.21542274, 0.15453463,0.14744809]
        self.files = collections.defaultdict(list)
        for split in ["train", "validation", ]:
            file_list = os.listdir(self.root + 'image_data/'+ split + '/' + 'label/')
            self.files[split] = file_list

        if not os.path.isdir(self.root + '/image_data/train/pre_encoded'):
            self.setup(pre_encode=True)
        else:
            self.setup(pre_encode=False)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/image_data/' + self.split+ '/' + 'image/' + img_name
        lbl_path = self.root + '/image_data/train/' + 'pre_encoded/' + img_name[:-4] + '.png'

        img = tiff.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)
        lbl_c = np.unique(lbl)
        lbl_cls = [0,0,0,0,0,0]
        if 1 in lbl_c:
            lbl_cls[0] = 1
        if 2 in lbl_c:
            lbl_cls[1] = 1
        if 3 in lbl_c:
            lbl_cls[2] = 1
        if 4 in lbl_c:
            lbl_cls[3] = 1
        if 5 in lbl_c:
            lbl_cls[4] = 1
        if 6 in lbl_c:
            lbl_cls[5] = 1
        lbl_cls = np.array(lbl_cls)
        # lbl = np.array(lbl, dtype=np.int32)
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl


    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        # lbl[lbl == 255] = 0
        # lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    # def transform(self, img, lbl):
    #     img = (img.astype('float32') / 255.0-self.mean)/self.std
    #     img = img.transpose(2, 0, 1)
    #     # lbl[lbl == 255] = 0
    #     # lbl = lbl.astype(float)
    #     # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
    #     # lbl = lbl.astype(int)
    #     img = torch.from_numpy(img).float()
    #     lbl = torch.from_numpy(lbl).long()
    #     return img, lbl

    def get_pascal_labels(self):
        return np.asarray([[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0],[255, 0, 0]])

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.subplot(121)
            plt.imshow(rgb)
            plt.subplot(122)
            plt.imshow(temp)
            plt.show()
        else:
            return rgb

    def setup(self, pre_encode=False):
        target_path = self.root + '/image_data/train/pre_encoded/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if pre_encode:
            print("Pre-encoding segmentation masks...")
            for i in tqdm(self.files['train']):
                lbl_path = self.root+ '/image_data/train/label/' + i
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i[:-4] + '.png', lbl)

    def standardize(self, x):

        x -= np.mean(x)
        x /= np.std(x)

        return x


if __name__ == '__main__':
    local_path = '/data/Potsdam/'
    dst = potsdamLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=8)
    for i, data in enumerate(trainloader):
        imgs, labels, y_cls = data
        # rgb = dst.decode_segmap(labels.numpy()[i])
        # plt.imshow(np.array(rgb,dtype=np.uint8))
        # plt.show()
        # a = labels.numpy()[i]
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            print(type(y_cls))
            plt.imshow(img)
            plt.show()
            plt.imshow(np.array(dst.decode_segmap(labels.numpy()[i]),dtype=np.uint8))
            plt.show()
        break
