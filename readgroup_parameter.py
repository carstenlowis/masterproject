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

#divide data in groups T0, T0-12, T>12

groupdataT0 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0' ], axis=1).transpose()
groupdataT012 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0-12' ], axis=1).transpose()
groupdataT12 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T>12' ], axis=1).transpose()
groupname = ['T0', 'T0-12', 'T>12']

#give a specific parameter

parameter = 'T16_TBR_mean' #can be changed to every parameter, the excel table contains

group=[groupdataT0[[parameter, 'Ground_Truth']].dropna(), groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]
#group[0] is T0, group[1] is T0-12, group[2] is T>12

#Analysis group T0
cm_array, result, bestthreshold, areaundercurve = ([[]] * 3 for i in range(4)) #analysis data for each group is stored in these lists

for count in range(3):  #analysis for the three different groups

    thresholds = group[count].sort_values(by=[parameter])[parameter]

    truth = (group[count]['Ground_Truth'] != 'RI').astype(int)

    predictions = [(group[count][parameter] >= threshold).astype(int) for threshold in thresholds]

    cm_array[count] = np.zeros((len(thresholds), 2, 2))
    for i, _ in enumerate(thresholds):
        cm_array[count][i] = metrics.confusion_matrix(truth, predictions[i])

    dic = {'threshold': [], 'tpr': [], 'fpr': [], 'spe': [], 'multiplication': []}

    for i, cm in enumerate(cm_array[count]):
        tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        dic['tpr'].append(tpr)
        fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        dic['fpr'].append(fpr)
        spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        dic['spe'].append(spe)
        dic['threshold'].append(thresholds.iloc[i])
        dic['multiplication'].append(tpr*spe)


    result[count] = pd.DataFrame(dic)

    bestthreshold[count] = result[count]['threshold'][np.argmax(result[count]['multiplication'])]

    auc = metrics.auc(result[count]['fpr'], result[count]['tpr'])
    areaundercurve[count] = auc

#plot

gs = gridspec.GridSpec(3, 1)
plt.figure()
for i in range(3):
    ax = plt.subplot(gs[i])
    plt.plot(result[i]['fpr'].values, result[i]['tpr'].values, label='ROC curve (area = %0.2f and best threshold = %0.2f)' %(areaundercurve[i], bestthreshold[i]))
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(groupname[i] + ' - ' + parameter)
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

