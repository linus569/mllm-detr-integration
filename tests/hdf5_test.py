import h5py
import numpy as np
import os

train_set = None
val_set = None

print(os.path.exists("precomputed_img_final.hdf5"))

with h5py.File("precomputed_img_final.hdf5", 'r') as f:
    train_set = f['precomputed_train_img'][:]
    val_set = f['precomputed_val_img'][:]

print("train_set shape: ", train_set.shape)
print("val_set shape: ", val_set.shape)