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
import sys, os
import nibabel as ni
import nibabel.ecat as ecat
import numpy as np


#path = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/TMZ_MONITORING'
path = 'H:/'
v_files = glob.glob(path + '/*.v')

for count in range(len(v_files)):
    name = os.path.basename(v_files[count])
    name = name[:-2] + '.nii.gz'

    eimg = ecat.load(v_files[count])



    new = nib.Nifti1Image(data,affine)

    nib.save(new, name)
    print(name[:-4])
    print('progress: ', count + 1, ' / ', len(v_files))

