from os.path import join
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind_from_stats

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

parameter = 'T16_SUV_mean' #can be changed to every parameter, the excel table contains

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
plt.figure(1)
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

#statistical operations

group_mean, group_std, groupRelapse, groupRI, groupRelapse_mean, groupRelapse_std, groupRI_mean, groupRI_std, t2, p2 = ([] for i in range(10))

for count in range(3):  #divide groups in RI and Relapse for ttest

    groupRI.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'RI'], axis=1).transpose())
    groupRelapse.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'Relapse'], axis=1).transpose())

for count in range(3):  #calculate mean, std of the different groups

    group_mean.append(np.mean(group[count][parameter]))
    group_std.append(np.std(group[count][parameter]))
    groupRelapse_mean.append(np.mean(groupRelapse[count][parameter]))
    groupRelapse_std.append(np.std(groupRelapse[count][parameter]))
    groupRI_mean.append(np.mean(groupRI[count][parameter]))
    groupRI_std.append(np.std(groupRI[count][parameter]))

#boxplot

gs = gridspec.GridSpec(3, 1)
plt.figure(2)
sns.set_theme(style="whitegrid")
for i in range(3):
    ax = plt.subplot(gs[i])
    sns.boxplot(x = group[i][parameter])

plt.show()

sns.set_theme(style="whitegrid")
sns.boxplot(x = group[0][parameter])

#ttest between RI and Relapsed

for count in range(3):  #calculate t and p of the different groups
    t, p = ttest_ind_from_stats(groupRelapse_mean[count], groupRelapse_std[count], groupRelapse[count][parameter].size,
                              groupRI_mean[count], groupRI_std[count], groupRI[count][parameter].size,
                              equal_var = False)

    t2.append(t)
    p2.append(p)

