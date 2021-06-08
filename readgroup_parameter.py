from os.path import join
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from scipy import stats

#import data
path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = '_Patiententabelle_Serial_Imaging_BM_anonymized_07062021.xlsx'
exceldata = pd.read_excel(join(path, file))

#divide data in groups T0, T0-12, T>12

groupdataT0 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0' ], axis=1).transpose()
groupdataT012 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0-12' ], axis=1).transpose()
groupdataT12 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T>12' ], axis=1).transpose()

#give a specific parameter

parameter = 'T16_SUV_mean' #can be changed to every parameter, the excel table contains

if len(groupdataT0[parameter].dropna()) != 0:
    groupname = ['T0', 'T0-12', 'T>12']
    group=[groupdataT0[[parameter, 'Ground_Truth']].dropna(), groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]
else:
    groupname = ['T0-12', 'T>12']
    group=[groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]
#group[0] is T0, group[1] is T0-12, group[2] is T>12 or (if T0 is not available) T0-12 is group[0] and T>12 ist group[1]

#Analysis group T0
cm_array, result, bestthreshold, youden, auc = ([[]] * len(group) for i in range(5)) #analysis data for each group is stored in these lists

for count in range(len(group)):  #analysis for the different groups

    thresholds = group[count].sort_values(by=[parameter])[parameter]

    truth = (group[count]['Ground_Truth'] != 'RI').astype(int)

    predictions = [(group[count][parameter] >= threshold).astype(int) for threshold in thresholds]

    cm_array[count] = np.zeros((len(thresholds), 2, 2))
    for i, _ in enumerate(thresholds):
        cm_array[count][i] = metrics.confusion_matrix(truth, predictions[i])

    dic = {'threshold': [], 'tpr': [], 'fpr': [], 'spe': [], 'multiplication': [], 'youden': [], 'TN': [], 'FP': [], 'FN': [], 'TP': []}

    for i, cm in enumerate(cm_array[count]):
        tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        dic['tpr'].append(tpr)
        fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        dic['fpr'].append(fpr)
        spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        dic['spe'].append(spe)
        dic['threshold'].append(thresholds.iloc[i])
        dic['multiplication'].append(tpr*spe)
        dic['youden'].append(tpr+fpr-1)
        dic['TN'].append(cm[0, 0])
        dic['FP'].append(cm[0, 1])
        dic['FN'].append(cm[1, 0])
        dic['TP'].append(cm[1, 1])


    result[count] = pd.DataFrame(dic)

    youden[count] = result[count]['threshold'][np.argmax(result[count]['youden'])]
    bestthreshold[count] = result[count]['threshold'][np.argmax(result[count]['multiplication'])]

    areaundercurve = metrics.auc(result[count]['fpr'], result[count]['tpr'])
    auc[count] = areaundercurve

#plot

for i in range(len(group)):
    plt.figure(groupname[i])
    plt.plot(result[i]['fpr'].values, result[i]['tpr'].values, label='ROC curve (area = %0.2f, best threshold = %0.2f, n=%0.0f)' %(auc[i], bestthreshold[i], len(result[i]['tpr'])))
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(groupname[i] + ' - ' + parameter)
    plt.legend(loc="lower right")

plt.show()

#statistical operations

k2RIlist, pRIlist, k2Relapselist, pRelapselist, group_mean, group_std, groupRelapse, groupRI, groupRelapse_mean, groupRelapse_std, groupRI_mean, groupRI_std, t2, p2 = ([] for i in range(14))

for count in range(len(group)):  #divide groups in RI and Relapse for ttest
    groupRI.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'RI'], axis=1).transpose())
    groupRelapse.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'Relapse'], axis=1).transpose())

for count in range(len(group)):  #calculate mean, std of the different groups
    group_mean.append(np.mean(group[count][parameter]))
    group_std.append(np.std(group[count][parameter]))
    groupRelapse_mean.append(np.mean(groupRelapse[count][parameter]))
    groupRelapse_std.append(np.std(groupRelapse[count][parameter]))
    groupRI_mean.append(np.mean(groupRI[count][parameter]))
    groupRI_std.append(np.std(groupRI[count][parameter]))

#boxplot

for i in range(3):
    textstr = '\n'.join((
        r'RI',
        r'n = %.0f' % (len(groupRI[i]),),
        r'mean = %.2f' % (groupRI_mean[i],),
        r'std = %.2f' % (groupRI_std[i],),
        r'',
        r'Relapse',
        r'n = %.0f' % (len(groupRelapse[i]),),
        r'mean = %.2f' % (groupRelapse_mean[i],),
        r'std = %.2f' % (groupRelapse_std[i],)))

    DF = pd.DataFrame({'RI': groupRI[i][parameter], 'Relapse': groupRelapse[i][parameter]})
    ax = DF[['RI', 'Relapse']].plot(kind = 'box', title = 'Group ' + groupname[i], showmeans = True)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top')
    plt.show()

#mann whitney u test between RI and Relapsed


#ttest between RI and Relapsed

for count in range(len(group)):  #calculate t and p of the different groups
    #k2RI, pRI = stats.normaltest(groupRI[count][parameter])
    #k2Relapse, pRelapse = stats.normaltest(groupRelapse[count][parameter])
    t, p = ttest_ind_from_stats(groupRelapse_mean[count], groupRelapse_std[count], groupRelapse[count][parameter].size,
                              groupRI_mean[count], groupRI_std[count], groupRI[count][parameter].size,
                              equal_var = False)

    #k2RIlist.append(k2RI)
    #pRIlist.append(pRI)
    #k2Relapselist.append(k2Relapse)
    #pRelapselist.append(pRelapse)
    t2.append(t)
    p2.append(p)

#excel output

output = pd.DataFrame(
    np.array([[auc[i] for i in range(3)],
              [bestthreshold[i] for i in range(3)], [result[i]['TN'][np.argmax(result[i]['multiplication'])] for i in range(3)], [result[i]['FP'][np.argmax(result[i]['multiplication'])] for i in range(3)],
              [result[i]['FN'][np.argmax(result[i]['multiplication'])] for i in range(3)], [result[i]['TP'][np.argmax(result[i]['multiplication'])] for i in range(3)],
              [youden[i] for i in range(3)], [result[i]['TN'][np.argmax(result[i]['youden'])] for i in range(3)], [result[i]['FP'][np.argmax(result[i]['youden'])] for i in range(3)],
              [result[i]['FN'][np.argmax(result[i]['youden'])] for i in range(3)], [result[i]['TP'][np.argmax(result[i]['youden'])] for i in range(3)],
              [group_mean[i] for i in range(3)], [group_std[i] for i in range(3)], [len(group[i]) for i in range(3)],
              [groupRI_mean[i] for i in range(3)], [groupRI_std[i] for i in range(3)], [len(groupRI[i]) for i in range(3)],
              [groupRelapse_mean[i] for i in range(3)], [groupRelapse_std[i] for i in range(3)], [len(groupRelapse[i]) for i in range(3)]]),
    index=['AUC',
            'Best_Threshold', 'Best_Threshold_TN', 'Best_Threshold_FP',
            'Best_Threshold_FN', 'Best_Threshold_TP',
            'Youden_Threshold', 'Youden_Threshold_TN', 'Youden_Threshold_FP',
            'Youden_Threshold_FN', 'Youden_Threshold_TP',
            'mean', 'std', 'n',
            'RI_mean', 'RI_std', 'RI_n',
            'Relapse_mean', 'Relapse_std', 'Relapse_n'],
    columns=['T0', 'T0-12', 'T>12'])

#pathout = '/Volumes/BTU/MITARBEITER/Lowis/results/'
pathout = '/Users/robin/Desktop/results'
fileout = 'output.xlsx'
dataout = join(pathout, fileout)

writer = pd.ExcelWriter(dataout, engine='openpyxl', mode='a')
output.to_excel(writer, sheet_name=parameter)
writer.save()


