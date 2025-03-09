# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:34:52 2024

@author: Habib
"""

import numpy as np
import os

path = r"D:\predrnn-pytorch"
filename = 'satelit_dataset.npy'

data = np.load(os.path.join(path, filename))


train = data[0:60, ...]
test = data[60:80, ...]


def npz_generate(raw, path, n_groups=2, clip_length=20):

    n_clips = int((raw.shape[0] / 20))
    
    # Initialize new clips array
    new_clips = np.zeros((n_groups, n_clips, 2), dtype=np.int32)



    # Create clips: Alternating start indices between groups
    for g in range(n_groups):
        for i in range(n_clips):
            start_idx = i * 20 + g * 10  # Alternating between group 1 (starts at 0) and group 2 (starts at 10)
            new_clips[g, i] = [start_idx, clip_length]
    
    new_clips[:, :, 1] = 10

    # Dims: Keep the dimensions the same as the original data
    new_dims = np.array([[1, 850, 2350]], dtype=np.int32)
    np.savez(path, 
             clips=new_clips, dims=new_dims, 
             input_raw_data=raw)

    return

train_path = os.path.join(path, 'data/1_train.npz')
test_path = os.path.join(path, 'data/1_valid.npz')

npz_generate(train, train_path)
npz_generate(test, test_path)


    
    



