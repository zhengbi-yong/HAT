import h5py
import numpy as np

fileName = 'psf01_00001.h5'
filePath = './datasets/LightField/LFDatasetWithNoise/train/'
h5f = h5py.File(filePath + fileName, 'r')

length = len(h5f.keys())
print(f"length: {length}")
for key in h5f.keys():
    print(f"*" * 40)
    print(f"name:{key}")
    print(f"type:{type(h5f[key])}")
    print(f"dtype:{h5f[key].dtype}")
    print(f"shape:{h5f[key].shape}")
    print(f"-" * 40)
