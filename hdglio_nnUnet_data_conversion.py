from os.path import join

import pandas as pd

import glob
import shutil
from sklearn.model_selection import train_test_split


#import datanames
dir = 'H:/hdglio_data'
files=glob.glob(dir+'/*/*/*')
des = ''

flair, mprage, km, t2, mask = ([] for i in range(5))

for i in range(len(files)):
    #flair
    if files[i][-25:] == '_FLAIR_to_T1_hdbet.nii.gz':
        flair.append(files[i])
    #mprage
    if files[i][-27:] == '_MPRAGE_STRUCT_hdbet.nii.gz':
        mprage.append(files[i])
    #km
    if files[i][-24:] == '_T1KM_to_T1_hdbet.nii.gz':
        km.append(files[i])
    #t2
    if files[i][-22:] == '_T2_to_T1_hdbet.nii.gz':
        t2.append(files[i])
    #mask
    if files[i][-21:] == '_hd_glio_Check.nii.gz':
        mask.append(files[i])

#train test split
train, test = train_test_split(list(range(len(flair))), test_size=0.2, random_state=42)

#copy data to new destinatiion
mask_origin, mask_destination = ([] for i in range(2))

for i in range(len(flair)):
    #train
    mask_origin.append(mask[i])
    if i in train:
        mask_dest = 'H:/Task060_hdglio/' + 'labelsTr/' + 'hdglio_' + "{:04n}".format(i) + '.nii.gz'
        shutil.copyfile(flair[i], 'H:/Task060_hdglio/' + 'imagesTr/' + 'hdglio_' + "{:04n}".format(i) + '_0000.nii.gz')
        shutil.copyfile(mprage[i], 'H:/Task060_hdglio/' + 'imagesTr/' + 'hdglio_' + "{:04n}".format(i) + '_0001.nii.gz')
        shutil.copyfile(km[i], 'H:/Task060_hdglio/' + 'imagesTr/' + 'hdglio_' + "{:04n}".format(i) + '_0002.nii.gz')
        shutil.copyfile(t2[i], 'H:/Task060_hdglio/' + 'imagesTr/' + 'hdglio_' + "{:04n}".format(i) + '_0003.nii.gz')
        shutil.copyfile(mask[i], mask_dest)
    #test
    else:
        mask_dest = 'H:/Task060_hdglio/' + 'labelsTs/' + 'hdglio_' + "{:04n}".format(i) + '.nii.gz'
        shutil.copyfile(flair[i], 'H:/Task060_hdglio/' + 'imagesTs/' + 'hdglio_' + "{:04n}".format(i) + '_0000.nii.gz')
        shutil.copyfile(mprage[i], 'H:/Task060_hdglio/' + 'imagesTs/' + 'hdglio_' + "{:04n}".format(i) + '_0001.nii.gz')
        shutil.copyfile(km[i], 'H:/Task060_hdglio/' + 'imagesTs/' + 'hdglio_' + "{:04n}".format(i) + '_0002.nii.gz')
        shutil.copyfile(t2[i], 'H:/Task060_hdglio/' + 'imagesTs/' + 'hdglio_' + "{:04n}".format(i) + '_0003.nii.gz')
        shutil.copyfile(mask[i], mask_dest)
    mask_destination.append(mask_dest)
    print('copy_progress: ', i+1, '/', len(flair))

copylog = pd.DataFrame({'mask_origin': mask_origin, 'mask_destination': mask_destination})
copylog.to_csv('H:/Task060_hdglio/copylog.csv', index=False)

#create dataset.json
generate_dataset_json(join('H:/Task060_hdglio', 'dataset.json'), ('H:/Task060_hdglio' + '/imagesTr'), ('H:/Task060_hdglio' + '/imagesTs'), ('flair','mprage','km','t2'),
                          {0: 'background', 1: 'contrast_agent', 2: 'suspicious_tissue'}, 'hdglio')




