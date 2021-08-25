from os.path import isfile, join
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
from collections import Counter

path = 'H:/Task053_BrainPET/'


images = glob.glob(path + 'Images/*')
truth = glob.glob(path + 'Truth/*')
predictions = glob.glob(path + 'Predictions/*')


dices=[]

for i in range(len(images)):
    t = nib.load(truth[i])
    t = np.array(t.dataobj)
    t = t[:,:,:,0]
    p = nib.load(predictions[i])
    p = np.array(p.dataobj)
    sum= t + p
    one = np.count_nonzero(sum==1)
    two = np.count_nonzero(sum==2)
    dices.append( (2 * two)/(2 * two + one) )
    print(images[i])
    print('dice: ', dices[i])
    print('size truth: ', np.count_nonzero(t==1))
    print('size predictions: ', np.count_nonzero(p==1))
    print('number of identical voxel: ', two)
    image =nib.load(images[i])
    combined_mask = t + 2 * p
    combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
    nib.save(combined_mask, path+'combined/'+truth[i][-17:])

