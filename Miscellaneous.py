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
dir='/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task004_Hippocampus'
ssd='D:/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task050_BrainPET'
path1='labelsTr'
path2='labelsTs'

files1 = glob.glob(join(ssd, path1)+'/*')
files2 = glob.glob(join(ssd, path2)+'/*')

files=files1+files2

filesall2=glob.glob(ssd+'/*/*')

for i in range(len(files)):
    os.rename(files[i], files[i][:-16]+'brain'+files[i][-14:-9]+'.nii')
    print(i)

#hippocampus
files=glob.glob(dir+'/*/*')
for i in range(len(files)):
    os.rename(files[i], files[i][:-7]+'_0000'+files[i][-7:])
#nnUNet_train 3d_fullres nnUNetTrainerV2 Task0004_Hippocampus FOLD --npz
#zip files

desktop = '/Users/carsten/Desktop'
file = 'brain_0000_0000.nii'
pathfile = join(desktop, file)


for i in range(len(filesall)):
    with open(filesall[i], 'rb') as f_in:
        with gzip.open(filesall[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(i)

for i in range(len(filesall)):
    os.remove(filesall[i])
    print(i)


#nochmal uberprufen
# brain_0048
# brain_0047
# brain_0045
# brain_0038
# brain_0039
# nan = 0, normalisierung, voxelgrossenverteilung, dice coefficient selber berechnen

#nan to 0
dir='D:/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data'
#dir='/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data'
task = 'Task051_BrainPET'
fileslabelTr = glob.glob(join(dir, task)+'/labelsTr/*')
filesimageTr = glob.glob(join(dir, task)+'/imagesTr/*')
fileslabelTs = glob.glob(join(dir, task)+'/labelsTs/*')
filesimageTs = glob.glob(join(dir, task)+'/imagesTs/*')

files_all = fileslabelTr + filesimageTr + fileslabelTs + filesimageTs

for i in range(len(files_all)):
    loaded_file = nib.load(files_all[i])
    array = loaded_file.get_fdata()
    inds = np.where(np.isnan(array))
    array[inds] = 0
    img = nib.Nifti1Image(array, loaded_file.affine, loaded_file.header)
    nib.save(img, files_all[i])
    print(i)

#some statistics
#import data
path = 'Z:/MITARBEITER/Lowis/data_nnUnet'
#path = 'H:/data_nnUnet'
#path = '/Users/carsten/Desktop/data_nnUnet_test'
#path = '/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet'
imaging_path = join(path, 'nnUnet_imagingdata')
files = glob.glob(imaging_path+'/*/*.nii')


#search for wanted files
maskfiles = [file for file in files if 'mask' in file]

patientids = []
for i in range(len(maskfiles)):
    a = maskfiles[i][55:maskfiles[i][55:].find('_')+55]
    if a[-1].isnumeric():
        patientids.append(a[:-1])
    else:
        patientids.append(a)

patientids = list(set(patientids))


male = 0
female = 0
for i in range(len(patientids)):
    if 'M' in patientids[i][4:]:
        male = male + 1
    else:
        female = female + 1

#copy some data

source = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/BIOPSY_STUDY'
dest = 'H:/BIOPSY_STUDY_seg'

voi_files = glob.glob(source+'/*/_nii/*.voi')
nii_files = glob.glob(source+'/*/_nii/*T1KM.nii')
mask_files = glob.glob(dest+'/*/*Mask__*_T1KM.nii.gz')

for i in range(len(voi_files)):
    os.mkdir(dest + '/' + voi_files[i][53:56])
    shutil.copyfile(voi_files[i], dest + '/' + voi_files[i][53:56] + '/' + voi_files[i][62:])
    shutil.copyfile(nii_files[i], dest + '/' + nii_files[i][53:56] + '/' + nii_files[i][62:])

for i in range(len(mask_files)):
    shutil.copyfile(mask_files[i], source+'/'+mask_files[i][20:23]+'/_nii/'+'mask1_'+mask_files[i][-22:])
    print(i,'/', len(mask_files))


for i in range(len(mask_files)):
    with open(mask_files[i], 'rb') as f_in:
        with gzip.open(mask_files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(mask_files[i])
    print('zip_progress: ', i+1, '/', len(mask_files))