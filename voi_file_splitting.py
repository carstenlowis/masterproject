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

path = 'H:/Pmod_test/'
voi_file = glob.glob(path+'/*.voi')

for count in range(len(voi_file)):
    #copy data into folder and converion from .voi to .txt
    copy_path = voi_file[count][:-4]
    os.mkdir(copy_path)
    #read data
    orig = open(voi_file[count], 'r')
    orig = orig.read()
    orig = orig.splitlines()
    #number of VOIs
    n = int(orig[15])

    #create header, bottom and find positions of the VOIs and names
    header = orig[:18]
    header [15] = '1'
    voi_index = []
    bottom = []
    names = []
    for i in range(len(orig)):
        if '</UID>' in orig[i][-6:]:
            voi_index.append(i)

            names.append(orig[i+1][:orig[i+1].find('#')])
        elif '#END OF VOIS DEFINITION' in orig[i]:
            voi_index.append(i-1)
            bottom.append(orig[i - 1])
            bottom.append(orig[i])

    #create new txt files according to the number of VOIs
    for i in range(n):
        temp = orig[voi_index[i]:voi_index[i+1]-1]
        new = header + temp + bottom
        name = voi_file[count][len(path):-4] + '_VOI_' + names[i][:-1] + '.voi'

        f = open(copy_path + '/' + name, 'x')
        f.write('\n'.join(new))
        f.close()

