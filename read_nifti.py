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

#import data
path1 = 'Z:\MITARBEITER\Lowis\imaging_data_btu'
path2 = '01_FE1CP014M3\MRI'
file = 'FE1-014M3_MRI_KM_coreg.ii'
ni_data = join(path1, path2, file)

test = sitk.ReadImage(ni_data)