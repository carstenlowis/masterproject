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
import gzip
import nibabel as nib

#import data
path = 'Z:/MITARBEITER/Lowis/data_nnUnet'
#path = 'H:/data_nnUnet'
#path = '/Users/carsten/Desktop/data_nnUnet_test'
#path = '/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet'
imaging_path = join(path, 'nnUnet_imagingdata')
files = glob.glob(imaging_path+'/*/*.nii')


#search for wanted files
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

#create directory for task
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
task = 'brain'
petnames = []
masknames = []
for i in range(len(patientids)):
    petnames.append(task+'_'+'{0:04}'.format(i)+'_'+'0000.nii')
    masknames.append(task+'_'+'{0:04}'.format(i)+'.nii')

df_files2 = pd.DataFrame({'masknames': masknames,
                          'petnames': petnames})

df_files = df_files1.join(df_files2)


#divide in test and train
train, test = train_test_split(list(range(len(patientids))), test_size=0.2, random_state=42)

#copy in new directory
copylog = pd.DataFrame(columns = ['mask_origin', 'mask_destination', 'image_origin', 'image_destination'])

for i in range(len(patientids)):
    mask_origin = df_files['maskfiles'].iloc[i]
    image_origin = df_files['petfiles'].iloc[i]
    mask_destination = join(directory, 'labelsTr', df_files['masknames'].iloc[i])
    image_destination = join(directory, 'imagesTr', df_files['petnames'].iloc[i])
    if i in train:
        shutil.copyfile(mask_origin, mask_destination)
        shutil.copyfile(image_origin, image_destination)
    else:
        shutil.copyfile(mask_origin, mask_destination)
        shutil.copyfile(image_origin, image_destination)
    temp_log = pd.DataFrame([[mask_origin, mask_destination, image_origin, image_destination]], columns = ['mask_origin', 'mask_destination', 'image_origin', 'image_destination'], index = i)
    copylog.append(temp_log)
    print('copy_progress: ', i+1, '/', len(patientids))

#compress files .nii to .nii.gz
allfiles = glob.glob(directory+'/*/*')

for i in range(len(allfiles)):
    with open(allfiles[i], 'rb') as f_in:
        with gzip.open(allfiles[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(allfiles[i])
    print('zip_progress: ', i+1, '/', len(allfiles))

#set NAN to 0
allfiles = glob.glob(directory+'/*/*')

for i in range(len(allfiles)):
    loaded_file = nib.load(allfiles[i])
    array = loaded_file.get_fdata()
    inds = np.where(np.isnan(array))
    array[inds] = 0
    img = nib.Nifti1Image(array, loaded_file.affine, loaded_file.header)
    nib.save(img, allfiles[i])
    print('remove_NAN: ', i + 1, '/', len(allfiles))

#create dataset.json
generate_dataset_json(join(directory, 'dataset.json'), join(directory, 'imagesTr'), join(directory, 'imagesTs'), ('PET_sum',),
                          {0: 'background', 1: 'tumor'}, 'brain_metastases')

print('finished')