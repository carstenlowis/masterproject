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

#import data
#win
#path = 'Z:\MITARBEITER\Lowis\nn-Unet'
#mac
path = '/Volumes/BTU/MITARBEITER/Lowis/nn-Unet'
imaging_path = join(path, 'nnUnet_imagingdata')
files = glob.glob(imaging_path+'/*/*.nii')

maskfiles = [file for file in files if file[-8:-3] == 'mask.']

patientids = []
for i in range(len(maskfiles)):
    patientids.append(maskfiles[i][len(path)+4:-9])

petfiles = []
for i in range(len(patientids)):
    if len(patientids[i]) == 9:
        petfiles.append(maskfiles[i][:-18]+patientids[i]+'_Sum_coreg.nii')
    else:
        petfiles.append(maskfiles[i][:-17] + patientids[i] + '_Sum_coreg.nii')

df_files = pd.DataFrame({'maskfiles': maskfiles,
                         'petfiles': petfiles})

df_files = df_files.sort_values(by='maskfiles')

#create directory
#taskname = 'Task001_BrainTumor'

#nnUNetpath = 'nnUNet_raw_data_base/nnUNet_raw_data'
#directory = join(path, nnUNetpath , taskname)

#os.makedirs(directory)
#os.makedirs(join(directory, 'imagesTr'))
#os.makedirs(join(directory, 'imagesTs'))
#os.makedirs(join(directory, 'labelsTr'))

#copy data in new directory

#create filenames
img_names = []
lbl_names = []
for i in range(len(patientids)):
