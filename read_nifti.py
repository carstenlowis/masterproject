import SimpleITK as sitk
from os.path import join
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ranksums
import pylab
import nibabel as nib
import math

#import data
#win
#path1 = 'Z:\MITARBEITER\Lowis\imaging_data_btu'
#mac
dir='/Volumes/BTU/MITARBEITER/Lowis/data_nnUnet/nnUNet_raw_data_base/nnUNet_raw_data'
task = 'Task050_BrainPET'
fileslabelTr = glob.glob(join(dir, task)+'/labelsTr/*')
filesimageTr = glob.glob(join(dir, task)+'/imagesTr/*')
fileslabelTs = glob.glob(join(dir, task)+'/labelsTs/*')
filesimageTs = glob.glob(join(dir, task)+'/imagesTs/*')



for i in range(len(fileslabelTs)):
	mask = nib.load(fileslabelTs[i]).get_data()
	min = mask.min()
	if math.isnan(min):
		print(i,'nan')
	else:
		print(i,'no nan')

for i in range(len(filesimagesTs)):
	mask = nib.load(filesimagesTs[i]).get_data()
	max = mask.max()
	print(max)
#	if max == 1:
#		print(i,'max is 1')
#	else:
#		print(i,'max is not 1')