from os.path import isfile, join
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ranksums
from os import listdir
import glob
import shutil
from sklearn.model_selection import train_test_split
from nibabel.testing import data_path
import nibabel as nib
import gzip

#renaming
dir='/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task050_BrainPET'
path1='labelsTr'
path2='labelsTs'

files1 = glob.glob(join(dir, path1)+'/*')
files2 = glob.glob(join(dir, path2)+'/*')

files=files1+files2

filesall=glob.glob(dir+'/*/*')

for i in range(len(filesall)):
    os.rename(filesall[i], filesall[i][:-3])
    print(i)

#zip files

desktop = '/Users/carsten/Desktop'
file = 'brain_0000_0000.nii'
pathfile = join(desktop, file)


for i in range(len(filesall)):
    with open(filesall[i], 'rb') as f_in:
        with gzip.open(filesall[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filesall[i])
    print(i)
