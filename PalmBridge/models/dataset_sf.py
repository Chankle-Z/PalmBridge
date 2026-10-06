# -*- coding:utf-8 -*-
import os

import PIL
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor.contiguous()


class MyDataset(data.Dataset):
    """
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    """

    def __init__(self, txt, transforms=None, train=True, image_size=128, out_channels=1):
        self.train = train

        self.image_size = image_size  # 128, 224
        self.out_channels = out_channels  # 1, 3

        self.text_path = txt

        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([

                    T.Resize((self.image_size,self.image_size)),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.out_channels)

                ])
            else:
                self.transforms = T.Compose([

                    T.Resize((self.image_size,self.image_size)),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        T.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=PIL.Image.BICUBIC, expand=False,
                                             center=(0.5 * self.image_size, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=PIL.Image.BICUBIC, expand=False,
                                             center=(0.0, 0.5 * self.image_size)),
                        ]),
                    ]),

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.out_channels)
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])

    def __getitem__(self, index):
        img_path_anchor = self.images_path[index]
        label_anchor = self.images_label[index]

        positive_index = np.random.choice(
            np.arange(len(self.images_label))[np.array(self.images_label) == label_anchor])
        negative_index = np.random.choice(
            np.arange(len(self.images_label))[np.array(self.images_label) != label_anchor])

        if self.train == True:
            while positive_index == index:
                positive_index = np.random.choice(
                    np.arange(len(self.images_label))[np.array(self.images_label) == label_anchor])

            while negative_index == index:
                negative_index = np.random.choice(
                    np.arange(len(self.images_label))[np.array(self.images_label) != label_anchor])

        else:
            positive_index = index
            negative_index = index

        img_path_positive = self.images_path[positive_index]

        anchor = Image.open(img_path_anchor).convert('L')
        anchor = self.transforms(anchor)

        positive = Image.open(img_path_positive).convert('L')
        positive = self.transforms(positive)

        negative = Image.open(self.images_path[negative_index]).convert('L')
        negative = self.transforms(negative)

        data = [anchor, positive, negative]
        label = [int(label_anchor), int(self.images_label[positive_index]), int(self.images_label[negative_index])]
        # print(data)
        # print(label)

        return data, label

    def __len__(self):
        return len(self.images_path)
