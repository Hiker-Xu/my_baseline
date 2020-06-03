#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import torch
from torch.utils.data import Dataset
import numpy as np


class SURREAL24(Dataset):
    def __init__(self, data_path, is_train=True):
        """
        :param data_path: path to dataset
        :param is_train: load train/test dataset
        """

        self.data_path = data_path
        self.is_train = is_train

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_meta, self.test_meta = [], []

        # loading data
        train_2d_file = 'train_2d_nor.npy'
        test_2d_file = 'test_2d_nor.npy'

        if self.is_train:
            # load train data
            self.train_3d = np.load(os.path.join(data_path, 'train_3d_nor.npy'))
            self.train_2d = np.load(os.path.join(data_path, train_2d_file))[:, 2:]
            num_f = self.train_2d.shape[0]
            assert self.train_2d.shape[0] == self.train_3d.shape[0], '(training) 3d & 2d shape not matched' 
            self.train_inp = self.train_2d.reshape(num_f, 36)
            #self.train_inp = self.train_2d.reshape(num_f, 50)
            #self.train_inp = self.train_2d.reshape(num_f, 32)
            self.train_out = self.train_3d.reshape(num_f, 54)
            #self.train_out = self.train_3d.reshape(num_f, 75)
            #self.train_out = self.train_3d.reshape(num_f, 48)

        else:
            # load test data
            self.test_3d = np.load(os.path.join(data_path, 'test_3d_nor.npy'))
            self.test_2d = np.load(os.path.join(data_path, test_2d_file))[:, 2:]
            num_f = self.test_2d.shape[0]
            assert self.test_2d.shape[0] == self.test_3d.shape[0], '(testing) 3d & 2d shape not matched' 
            self.test_inp = self.test_2d.reshape(num_f, 36)            
            #self.test_inp = self.test_2d.reshape(num_f, 50)
            #self.test_inp = self.test_2d.reshape(num_f, 32)
            self.test_out = self.test_3d.reshape(num_f, 54)
            #self.test_out = self.test_3d.reshape(num_f, 75)
            #self.test_out = self.test_3d.reshape(num_f, 48)
        
    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()
        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()

        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)  
