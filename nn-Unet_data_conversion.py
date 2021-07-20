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

#import data
#win
path = 'H:\data_nnUnet'
imaging_path = join(path, 'nnUnet_imagingdata')
files = glob.glob(imaging_path+'\*\*.nii')
#mac
#path = '/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet'
#imaging_path = join(path, 'nnUnet_imagingdata')
#files = glob.glob(imaging_path+'/*/*.nii')

maskfiles = [file for file in files if file[-8:-3] == 'mask.']

patientids = []
for i in range(len(maskfiles)):
    patientids.append(maskfiles[i][len(imaging_path)+4:-9])

petfiles = []
for i in range(len(patientids)):
    petfiles.append(maskfiles[i][:-(len(patientids[i])+9)] + patientids[i] + '_Sum_coreg.nii')

df_files1 = pd.DataFrame({'maskfiles': maskfiles,
                         'petfiles': petfiles})

df_files1 = df_files1.sort_values(by='maskfiles', ignore_index=True)

#create directory
taskname = 'Task050_BrainPET'

nnUNetpath = 'nnUNet_raw_data_base/nnUNet_raw_data'
directory = join(path, nnUNetpath , taskname)

os.makedirs(directory)
os.makedirs(join(directory, 'imagesTr'))
os.makedirs(join(directory, 'imagesTs'))
os.makedirs(join(directory, 'labelsTr'))
os.makedirs(join(directory, 'labelsTs'))

#copy data in new directory

#create filenames
petnames = []
masknames = []
for i in range(len(patientids)):
    petnames.append('brain_'+'{0:04}'.format(i)+'_'+'0000.nii.gz')
    masknames.append('la_'+'{0:04}'.format(i)+'_'+'0000.nii.gz')

df_files2 = pd.DataFrame({'masknames': masknames,
                          'petnames': petnames})

df_files = df_files1.join(df_files2)


#divide in test and train
train ,test = train_test_split(list(range(len(patientids))),test_size=0.2, random_state=42)

#copy in new directory

for i in range(len(patientids)):
    if i in train:
        shutil.copyfile(df_files['maskfiles'].iloc[i], join(directory, 'labelsTr', df_files['masknames'].iloc[i]))
        shutil.copyfile(df_files['petfiles'].iloc[i], join(directory, 'imagesTr', df_files['petnames'].iloc[i]))
    else:
        shutil.copyfile(df_files['maskfiles'].iloc[i], join(directory, 'labelsTs', df_files['masknames'].iloc[i]))
        shutil.copyfile(df_files['petfiles'].iloc[i], join(directory, 'imagesTs', df_files['petnames'].iloc[i]))
    print(i, '/', len(patientids))

#create dataset.json
#generate_dataset_json(join(directory, 'dataset.json'), join(directory, 'imagesTr'), join(directory, 'imagesTs'), ('PET_sum'),
#                          {0: 'background', 1: 'tumor'}, 'brain_metastases')