import os.path as join
import pandas as pd
import seaborn as sns
import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt


testa = pd.DataFrame(
    np.array([[[result[i]['TN'][np.argmax(result[i]['multiplication'])], result[i]['FP'][np.argmax(result[i]['multiplication'])]] for i in range(3)]]),
    index=['AUC', 's_mann_whitney'],
    columns=['T0', 'T0-12', 'T>12'])

test = pd.DataFrame(
    np.array([auc[:], slist[:]]),
    index=['AUC', 's_mann_whitney'],
    columns=['T0', 'T0-12', 'T>12'])