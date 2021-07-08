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
path = 'Z:\MITARBEITER\Lowis'
file = '_Patiententabelle_Serial_Imaging_BM_anonymized_07072021.xlsx'
exceldata = pd.read_excel(join(path, file))

#divide data in groups T0, T0-12, T>12

groupdataT0 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0' ], axis=1).transpose()
groupdataT012 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0-12' ], axis=1).transpose()
groupdataT12 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T>12' ], axis=1).transpose()

#give a specific parameter and drop NA

parameter = 'Dyn_2k0_slope' #can be changed to every parameter, the excel table contains

if len(groupdataT0[parameter].dropna()) != 0:
    groupname = ['T0', 'T0-12', 'T>12']
    group = [groupdataT0[[parameter, 'Ground_Truth']].dropna(), groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]
else:
    groupname = ['T0-12', 'T>12']
    group = [groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]

#normalization for negative values?
if len(groupname) == 3:
    group_min = [min(group[0][parameter]), min(group[1][parameter]), min(group[2][parameter])]
    group[0][parameter] = group[0][parameter] - group_min[0]
    group[1][parameter] = group[1][parameter] - group_min[1]
    group[2][parameter] = group[2][parameter] - group_min[2]
else:
    group_min = [min(group[0][parameter]), min(group[1][parameter])]
    group[0][parameter] = group[0][parameter] - group_min[0]
    group[1][parameter] = group[1][parameter] - group_min[1]


#Drop zeros ?

#for i in range(len(groupname)):
#    group[i]=group[i][group[i][parameter] != 0]

#group[0] is T0, group[1] is T0-12, group[2] is T>12 or (if T0 is not available) T0-12 is group[0] and T>12 ist group[1]

#Analysis group T0
cm_array, result, bestthreshold, youden, auc = ([[]] * len(group) for i in range(5)) #analysis data for each group is stored in these lists

for count in range(len(group)):  #analysis for the different groups

    thresholds = group[count].sort_values(by=[parameter])[parameter]

    truth = (group[count]['Ground_Truth'] != 'RI').astype(int)

#<> ?

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
        dic['youden'].append(tpr+spe-1)
        dic['TN'].append(cm[0, 0])
        dic['FP'].append(cm[0, 1])
        dic['FN'].append(cm[1, 0])
        dic['TP'].append(cm[1, 1])


    result[count] = pd.DataFrame(dic)

    youden[count] = result[count]['threshold'][np.argmax(result[count]['youden'])]
    bestthreshold[count] = result[count]['threshold'][np.argmax(result[count]['multiplication'])]

    areaundercurve = metrics.auc(result[count]['fpr'], result[count]['tpr'])
    auc[count] = areaundercurve

#statistical operations

group_mean, group_std, groupRelapse, groupRI, groupRelapse_mean, groupRelapse_std, groupRI_mean, groupRI_std, slist, plist= ([] for i in range(10))

for count in range(len(group)):  #divide groups in RI and Relapse
    groupRI.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'RI'], axis=1).transpose())
    groupRelapse.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'Relapse'], axis=1).transpose())

for count in range(len(group)):  #calculate mean, std of the different groups
    group_mean.append(np.mean(group[count][parameter]))
    group_std.append(np.std(group[count][parameter]))
    groupRelapse_mean.append(np.mean(groupRelapse[count][parameter]))
    groupRelapse_std.append(np.std(groupRelapse[count][parameter]))
    groupRI_mean.append(np.mean(groupRI[count][parameter]))
    groupRI_std.append(np.std(groupRI[count][parameter]))

#normalization for negative values?
bestthreshold

#

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

#boxplot

for i in range(len(groupname)):
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

#histogramm



#mann whitney u test between RI and Relapsed

for count in range(len(group)):  #calculate t and p of the different groups
    s, p = ranksums(groupRI[count][parameter], groupRelapse[count][parameter])

    slist.append(s)
    plist.append(p)

#excel output

bestthreshold_TN, bestthreshold_FP, bestthreshold_FN, bestthreshold_TP, youdenthreshold_TN, youdenthreshold_FP, youdenthreshold_FN, youdenthreshold_TP, n, nRI, nRelapse = ([] for i in range(11))

for i in range(len(groupname)):
    bestthreshold_TN.append(result[i]['TN'][np.argmax(result[i]['multiplication'])])
    bestthreshold_FP.append(result[i]['FP'][np.argmax(result[i]['multiplication'])])
    bestthreshold_FN.append(result[i]['FN'][np.argmax(result[i]['multiplication'])])
    bestthreshold_TP.append(result[i]['TP'][np.argmax(result[i]['multiplication'])])
    youdenthreshold_TN.append(result[i]['TN'][np.argmax(result[i]['youden'])])
    youdenthreshold_FP.append(result[i]['FP'][np.argmax(result[i]['youden'])])
    youdenthreshold_FN.append(result[i]['FN'][np.argmax(result[i]['youden'])])
    youdenthreshold_TP.append(result[i]['TP'][np.argmax(result[i]['youden'])])
    n.append(len(group[i]))
    nRI.append(len(groupRI[i]))
    nRelapse.append(len(groupRelapse[i]))

output = pd.DataFrame(
    np.array([auc, slist, plist,
              bestthreshold, bestthreshold_TN, bestthreshold_FP,
              bestthreshold_FN, bestthreshold_TP,
              youden, youdenthreshold_TN, youdenthreshold_FP,
              youdenthreshold_FN, youdenthreshold_TP,
              group_mean, group_std, n,
              groupRI_mean, groupRI_std, nRI,
              groupRelapse_mean, groupRelapse_std, nRelapse]),
    index=['AUC', 's_mann_whitney', 'p_mann_whitney',
            'Best_Threshold', 'Best_Threshold_TN', 'Best_Threshold_FP',
            'Best_Threshold_FN', 'Best_Threshold_TP',
            'Youden_Threshold', 'Youden_Threshold_TN', 'Youden_Threshold_FP',
            'Youden_Threshold_FN', 'Youden_Threshold_TP',
            'mean', 'std', 'n',
            'RI_mean', 'RI_std', 'RI_n',
            'Relapse_mean', 'Relapse_std', 'Relapse_n'],
    columns=groupname)

#pathout = '/Volumes/BTU/MITARBEITER/Lowis/results/'
#pathout = '/Users/robin/Desktop/results'
#fileout = 'output.xlsx'
#dataout = join(pathout, fileout)

#writer = pd.ExcelWriter(dataout, engine='openpyxl', mode='a')
#output.to_excel(writer, sheet_name=parameter)
#writer.save()

print(output)