import os.path as join
import pandas as pd
import seaborn as sns
import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt

exceldata = pd.read_excel(r'/Volumes/BTU/MITARBEITER/Lowis/_Patiententabelle_Serial_Imaging_BM_anonymized_20052021.xlsx')
PET_ID = exceldata[["PET_ID", "T16_SUV_mean", "Ground_Truth"]]

mask = []
for x in range(0, len(PET_ID)):
    if (PET_ID.iloc[x][0][-1].isdigit() == False) and (len(PET_ID.iloc[x][0]) != 3):
        mask.append(True)
    else:
        mask.append(False)

T16_SUV_mean_list1 = PET_ID[mask].dropna()

#confusion matrix
thresholds = T16_SUV_mean_list1.sort_values(by=['T16_SUV_mean']).T16_SUV_mean.values.tolist()

truth = []
for x in range(0, len(T16_SUV_mean_list1)):
    if T16_SUV_mean_list1.Ground_Truth.iloc[x] == 'RI':
        truth.append(0)
    else:
        truth.append(1)

predictions = np.zeros((len(T16_SUV_mean_list1),len(thresholds)))
for x in range(0, len(thresholds)):
    for i in range(0, len(T16_SUV_mean_list1)):
        if ( T16_SUV_mean_list1.iloc[i]['T16_SUV_mean'] >= thresholds[x] ):
            predictions[i][x] = 1
        else:
            predictions[i][x] = 0

cm_array=np.zeros((len(thresholds),2,2))
for x in range(0, len(thresholds)):
    cm_array[x] = confusion_matrix(truth,predictions[:,x])

#roc-curve


fpr=[]
tpr=[]
for x in range(0, len(cm_array)):
    tpr.append(cm_array[x, 0, 0] / (cm_array[x, 0, 0] + cm_array[x, 0, 1]))
    fpr.append(cm_array[x, 1, 0] / (cm_array[x, 1, 1] + cm_array[x, 1, 0]))

areaundercurve = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr,label='ROC curve (area = %0.2f)' % areaundercurve)
plt.plot([0,1],[0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: T16_SUV_mean')
plt.legend(loc="lower right")
