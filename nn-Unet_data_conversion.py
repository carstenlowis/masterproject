from os.path import isfile, join
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
#path = 'Z:\MITARBEITER\Lowis\nn-Unet\nnUnet_imagingdata'
#mac
path = '/Volumes/BTU/MITARBEITER/Lowis/nn-Unet/nnUnet_imagingdata'
files = glob.glob(path+'/*/*.nii')

maskfiles = [file for file in files if file[-8:-3] == 'mask.']

patientids = []
for i in range(len(maskfiles)):



path2 = '_Patiententabelle_Serial_Imaging_BM_anonymized_07072021.xlsx'
 = pd.read_excel(join(path1, path2))