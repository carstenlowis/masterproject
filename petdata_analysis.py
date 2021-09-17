from os.path import join
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ranksums

#settings
#'T1k6_TBR_mean'T1k6_rSUV_100'T2k0_Volume'd_T16_Volume'Dyn_max100vx_slope'
parameter = 'T2k0_Volume' #can be changed to every parameter, the excel table contains
#groundtruth = 'Ground_Truth' #'Ground_Truth' or 'Ground_Truth_only_largest_metastase'
groundtruth = 'Ground_Truth_only_largest_metastase' #'Ground_Truth' or 'Ground_Truth_only_largest_metastase'
#drop_0 = 'without_zeros' #drop zeros? 'with_zeros' or 'without_zeros'
drop_0 = 'with_zeros' #drop zeros? 'with_zeros' or 'without_zeros'
thresh = '>' #'<'>'  > normaly
#saveoutput = 'y' #save output? 'y' or 'n'
saveoutput = 'n' #save output? 'y' or 'n'

#import data
#win
path = 'Z:/MITARBEITER/Lowis/'
#mac
#path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = '_Patiententabelle_Serial_Imaging_BM_anonymized_07092021.xlsx'
exceldata = pd.read_excel(join(path, file))
#output path
fileout = 'out_' + groundtruth + '_' + parameter + '.xlsx'
pathout = path + 'results' + '/' + groundtruth + '_' + parameter + '_' + drop_0
dataout = pathout + '/' + fileout

#divide data in groups T0, T0-12, T>12

groupdataT0 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0' ], axis=1).transpose()
groupdataT012 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0-12' ], axis=1).transpose()
groupdataT12 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T>12' ], axis=1).transpose()
groupdatalast = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['Ground_Truth_last_measurement']) if type(x) == str ], axis=1).transpose()

#drop NA

if len(groupdataT0[parameter].dropna()) != 0:
    groupname = ['T0', 'T0-12', 'T>12', 'last_measurement']
    group = [groupdataT0[[parameter, groundtruth, 'time_since_first_scan']].dropna(), groupdataT012[[parameter, groundtruth, 'time_since_first_scan']].dropna(), groupdataT12[[parameter, groundtruth, 'time_since_first_scan']].dropna(), groupdatalast[[parameter, groundtruth, 'time_since_first_scan']].dropna()]
else:
    groupname = ['T0-12', 'T>12', 'last_measurement']
    group = [groupdataT012[[parameter, groundtruth, 'time_since_first_scan']].dropna(), groupdataT12[[parameter, groundtruth, 'time_since_first_scan']].dropna(), groupdatalast[[parameter, groundtruth, 'time_since_first_scan']].dropna()]

#normalization for negative values?
#if len(groupname) == 3:
#    group_min = [min(group[0][parameter]), min(group[1][parameter]), min(group[2][parameter])]
#    group[0][parameter] = group[0][parameter] - group_min[0]
#    group[1][parameter] = group[1][parameter] - group_min[1]
#    group[2][parameter] = group[2][parameter] - group_min[2]
#else:
#    group_min = [min(group[0][parameter]), min(group[1][parameter])]
#    group[0][parameter] = group[0][parameter] - group_min[0]
#    group[1][parameter] = group[1][parameter] - group_min[1]


#Drop zeros ?
if drop_0 == 'without_zeros':
    for i in range(len(groupname)):
        group[i]=group[i][group[i][parameter] != 0]

#group[0] is T0, group[1] is T0-12, group[2] is T>12 or (if T0 is not available) T0-12 is group[0] and T>12 ist group[1]

#Analysis group T0
cm_array, result, bestthreshold, youden, auc = ([[]] * len(group) for i in range(5)) #analysis data for each group is stored in these lists

for count in range(len(group)):  #analysis for the different groups

    thresholds = group[count].sort_values(by=[parameter])[parameter]

    truth = (group[count][groundtruth] != 'RI').astype(int)

    if thresh == '>':
        predictions = [(group[count][parameter] >= threshold).astype(int) for threshold in thresholds]

    elif thresh == '<':
        predictions = [(group[count][parameter] <= threshold).astype(int) for threshold in thresholds]

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

group_mean, group_std, groupRelapse, groupRI, groupRelapse_mean, groupRelapse_std, groupRI_mean, groupRI_std, slist, plist, t_mean, t_std= ([] for i in range(12))

for count in range(len(group)):  #divide groups in RI and Relapse
    groupRI.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count][groundtruth]) if x == 'RI'], axis=1).transpose())
    groupRelapse.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count][groundtruth]) if x == 'Relapse'], axis=1).transpose())

for count in range(len(group)):  #calculate mean, std of the different groups
    group_mean.append(np.mean(group[count][parameter]))
    group_std.append(np.std(group[count][parameter]))
    groupRelapse_mean.append(np.mean(groupRelapse[count][parameter]))
    groupRelapse_std.append(np.std(groupRelapse[count][parameter]))
    groupRI_mean.append(np.mean(groupRI[count][parameter]))
    groupRI_std.append(np.std(groupRI[count][parameter]))
    t_mean.append(np.mean(group[count]['time_since_first_scan']))
    t_std.append(np.std(group[count]['time_since_first_scan']))

#normalization for negative values?

#

#plot

for i in range(len(group)):
    plt.figure(groupname[i]+'_plot')
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




#mann whitney u test between RI and Relapsed

for count in range(len(group)):  #calculate s and p of the different groups
    s, p = ranksums(groupRI[count][parameter], groupRelapse[count][parameter])

    slist.append(s)
    plist.append(p)

#excel output

bestthreshold_TN, bestthreshold_FP, bestthreshold_FN, bestthreshold_TP, youdenthreshold_TN, youdenthreshold_FP, youdenthreshold_FN, youdenthreshold_TP, n, nRI, nRelapse, accuracy, specificity = ([] for i in range(13))

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
    accuracy.append((bestthreshold_TP[i]+bestthreshold_TN[i])/(bestthreshold_TP[i]+bestthreshold_TN[i]+bestthreshold_FP[i]+bestthreshold_FN[i]))
    specificity.append(bestthreshold_TN[i]/(bestthreshold_TN[i]+bestthreshold_FP[i]))

output = pd.DataFrame(
    np.array([auc, slist, plist,
              bestthreshold, bestthreshold_TN, bestthreshold_FP,
              bestthreshold_FN, bestthreshold_TP,
              youden, youdenthreshold_TN, youdenthreshold_FP,
              youdenthreshold_FN, youdenthreshold_TP,
              group_mean, group_std, n,
              groupRI_mean, groupRI_std, nRI,
              groupRelapse_mean, groupRelapse_std, nRelapse,
              t_mean, t_std, accuracy, specificity]),
    index=['AUC', 's_mann_whitney', 'p_mann_whitney',
            'Best_Threshold', 'Best_Threshold_TN', 'Best_Threshold_FP',
            'Best_Threshold_FN', 'Best_Threshold_TP',
            'Youden_Threshold', 'Youden_Threshold_TN', 'Youden_Threshold_FP',
            'Youden_Threshold_FN', 'Youden_Threshold_TP',
            'mean', 'std', 'n',
            'RI_mean', 'RI_std', 'RI_n',
            'Relapse_mean', 'Relapse_std', 'Relapse_n',
            'time_mean', 'time_std', 'accuracy', 'specificity'],
    columns=groupname)


if saveoutput == 'y':
    os.makedirs(pathout)
    output.to_excel(dataout)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        pdfname = 'fig_out_' + groundtruth + '_' + parameter + '_' + str(i) + '.pdf'
        fig.savefig(join(pathout, pdfname), format='pdf')




print(output)

#new plot
font = {'size'   : 15}

plt.rc('font', **font)

fig, axes = plt.subplots(ncols = 3, figsize=(12, 4))
fig.tight_layout()
DF1 = pd.DataFrame({'T0': groupRI[0][parameter], 'T0-12': groupRI[1][parameter], 'T>12': groupRI[2][parameter]})
DF2 = pd.DataFrame({'T0': groupRelapse[0][parameter], 'T0-12': groupRelapse[1][parameter], 'T>12': groupRelapse[2][parameter]})


DF1.plot(ax=axes[0], kind='box', title='Radiation necrosis')
DF2.plot(ax=axes[1], kind='box', title='Tumor recurrence')
axes[0].set_ylim((-0.2, np.max(exceldata[parameter])+0.2))
axes[1].set_ylim((-0.2, np.max(exceldata[parameter])+0.2))
axes[0].set_ylabel('Volume [mL]')
axes[1].set_ylabel('Volume [mL]')

axes[2].plot(result[0]['fpr'].values, result[0]['tpr'].values)
axes[2].plot(result[1]['fpr'].values, result[1]['tpr'].values)
axes[2].plot(result[2]['fpr'].values, result[2]['tpr'].values)
axes[2].plot([0, 1], 'k--')
axes[2].set_title('ROC curves')
axes[2].legend(['T0', 'T0-12', 'T>12'], loc="lower right", frameon = False)
axes[2].set_xlabel('False positive rate')
axes[2].set_ylabel('True positive rate')
axes[2].set_ylim((-0.05, 1.05))
axes[2].set_xlim((-0.05, 1.05))
fig.set_figheight(4.28)

fig.savefig(('Z:MITARBEITER/Lowis/results/' + parameter + '.eps'), format='eps')
