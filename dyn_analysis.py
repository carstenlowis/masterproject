from os.path import join
import pandas as pd
import numpy as np

#import data
path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = 'dyn_exceltest.xlsx'
exceldata = pd.read_excel(join(path, file))