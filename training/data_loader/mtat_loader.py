# coding: utf-8
import os
import numpy as np
from torch.utils import data

class AudioFolder(data.Dataset):
    def __init__(self, root, trval='TRAIN', dataset=None, data_type=None, input_length=None):
        self.trval = trval
        self.root = root
        self.dataset = dataset
        self.data_type = data_type
        if data_type == 'spec':
            self.input_length = 256
        elif data_type == 'raw':
            self.input_length = 65610
        if input_length != 256:
            self.input_length = input_length
        self.get_songlist()
        self.binary = np.load(os.path.join(self.root, 'data/binary.npy'))

    def __getitem__(self, index):
        if self.data_type == 'spec':
            spec, tag_binary = self.get_spec(index)
            return spec.astype('float32'), tag_binary.astype('float32')
        elif self.data_type == 'raw':
            raw, tag_binary = self.get_raw(index)
            return raw.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        if self.trval == 'TRAIN':
            self.fl = np.load(os.path.join(self.root, 'data/train_new.npy'))
        elif self.trval == 'VALID':
            self.fl = np.load(os.path.join(self.root, 'data/valid_new.npy'))
        elif self.trval == 'TEST':
            self.fl = np.load(os.path.join(self.root, 'data/test_new.npy'))

    def get_spec(self, index):
        ix, fn = self.fl[index].split('\t')
        spec_path = os.path.join(self.root, 'spec', fn.split('/')[1][:-3]) + 'npy'
        spec = np.load(spec_path, mmap_mode='r')
        if self.trval == 'TRAIN' or self.trval == 'VALID':
            random_idx = int(np.floor(np.random.random(1) * ((29*16000/256)-self.input_length)))
            spec = np.array(spec[:, random_idx:random_idx+self.input_length])
        tag_binary = self.binary[int(ix)]
        return spec, tag_binary

    def get_raw(self, index):
        ix, fn = self.fl[index].split('\t')
        raw_path = os.path.join(self.root, 'raw', fn.split('/')[1][:-3]) + 'npy'
        raw = np.load(raw_path, mmap_mode='r')
        if self.trval == 'TRAIN' or self.trval == 'VALID':
            random_idx = int(np.floor(np.random.random(1) * ((29*16000)-self.input_length)))
            raw = np.array(raw[random_idx:random_idx+self.input_length])
        tag_binary = self.binary[int(ix)]
        return raw, tag_binary

    def __len__(self):
        return len(self.fl)


def get_audio_loader(root, batch_size, trval='TRAIN', num_workers=0, dataset=None, data_type=None, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, trval=trval, dataset=dataset, data_type=data_type, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

