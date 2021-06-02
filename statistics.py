from os.path import join
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

#import data
path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = '_Patiententabelle_Serial_Imaging_BM_anonymized_20052021.xlsx'
exceldata = pd.read_excel(join(path, file))