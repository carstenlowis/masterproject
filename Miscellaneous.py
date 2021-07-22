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

#renaming
dir='/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task050_BrainPET'
path1='labelsTr'
path2='labelsTs'

files1 = glob.glob(join(dir, path1)+'/*')
files2 = glob.glob(join(dir, path2)+'/*')

files=files1+files2

for i in range(len(files)):
    os.rename(files[i], files[i])
    print(i)

