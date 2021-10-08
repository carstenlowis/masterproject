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
import re

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


for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('progress: ', i + 1, ' / ', len(files))

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

#read filenames
files = glob.glob('/Volumes/btu-ai/data/_Temp/PET_DATA_FOR_SEGMENTATION/TMZ_MONITORING/*.v')
a = []
b = []
for i in range(len(files)):
    files[i] = os.path.basename(files[i])
    a.append(files[i][3:13])
    b.append(files[i][0:10])

names = []
for i in range(len(files)):
    if a[i][0] == 'F':
        names.append(a[i])
    if b[i][0] == 'F':
        names.append(b[i])

for i in range(len(names)):
    string = names[i]
    names[i] = string.replace('_', '1')

for i in range(len(names)):
    if names[i][-1] == '-':
        string = names[i]
        names[i] = string.replace('-', '1')

#search
path = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/TMZ_MONITORING'
nii_masks = glob.glob(path+'/nii_masks/*.nii.gz')

#rename
for count in range(len(nii_masks)):
    start = nii_masks[count].find('voi_')
    ent = nii_masks[0].find('_F', 65) +1
    new = nii_masks[count][:start] + nii_masks[count][end:]
    if 'followup' in new:
        new = new.replace('_followup', '2')
    if 'baseline' in new:
        new = new.replace('_baseline', '')

    os.rename(nii_masks[count], new)

    print(os.path.basename(nii_masks[count])[:-7])
    print('progress: ', count + 1, ' / ', len(nii_masks))







files = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/METS_FIRST_STUDY_RADIOMICS/*.nii')

print(files[0])
for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('progress: ', i + 1, ' / ', len(files))


files = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/MONITORING_PARTIAL_RESECTION/*.nii')

print(files[0])
for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('progress: ', i + 1, ' / ', len(files))


files = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/PSEUDOPROGRESSION_RADIOMICS/*.nii')

print(files[0])
for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('progress: ', i + 1, ' / ', len(files))


files = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/REGORAFENIB/*.nii')

print(files[0])
for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('progress: ', i + 1, ' / ', len(files))


files = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/TMZ_MONITORING/*.nii')

print(files[0])
for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('progress: ', i + 1, ' / ', len(files))


for i in range(len(overview['ID.5'])):
	overview['ID.5'][i]=overview['ID.5'][i][:3] + '-' + overview['ID.5'][i][5:]


#rename images and masks
old = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/patient_data/MONITORING_PARTIAL_RESECTION/masks/*.nii.gz')

new = []
for i in range(len(old)):
    id = old[i][110:120]
    if id[-1].isdigit() == False:
        id = id[:-1] + '1'

    if bool(re.search(r'_', id)) == True:
        id = id[:3] + '-' + id[4:]

    if bool(re.search(r'-', id)) == False:
        id = id[:3] + '-' + id[5:]

    n = '1'
    if old[i][old[i].find('Tum')+3].isdigit() ==True:
        n = old[i][old[i].find('Tum')+3]

    print(n)
    #if old[i][110].isdigit() == True:
    #    n = old[i][110]

    new.append(os.path.dirname(old[i]) + '/' + id + '_MONITORING_PARTIAL_RESECTION_mask'+n+'.nii.gz')

for i in range(len(new)):
    os.rename(old[i], new[i])



masks = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/nnunet/Task070_segm/labelsTr/*mask.nii.gz')
mapath =  'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/nnunet/Task070_segm/labelsTr/'
images =[]
impath = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/nnunet/Task070_segm/imagesTr/'
for i in range(len(masks)):
    images.append(impath + os.path.basename(masks[i])[:-11] + 'image.nii.gz')

for i in range(len(masks)):
    os.rename(images[i], impath + 'brainseg_' + "{:04n}".format(i) + '_0000.nii.gz')
    os.rename(masks[i], mapath + 'brainseg_' + "{:04n}".format(i) + '.nii.gz')
    print(os.path.basename(masks[i])[:-11])


#read excel and find not working masks
data = pd.read_excel('Z:/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task070_segm/copylog.xlsx')

masks = glob.glob('Z:/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task070_segm/labelsTr_old/*.nii.gz')
masks = sorted(masks)
images = glob.glob('Z:/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task070_segm/imagesTr/*.nii.gz')
images = sorted(images)
path = 'Z:/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data/Task070_segm/labelsTr/'
for i in range(len(masks)):
    m = nib.load(masks[i])
    p = nib.load(images[i])
    m = np.array(m.dataobj)
    m = nib.Nifti1Image(m, p.affine, p.header)
    nib.save(m, path + os.path.basename(masks[i]))
    print('progress: ', i + 1, '/', len(images))
