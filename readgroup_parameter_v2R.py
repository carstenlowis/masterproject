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
path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = '_Patiententabelle_Serial_Imaging_BM_anonymized_15062021.xlsx'
exceldata = pd.read_excel(join(path, file))

#divide data in groups T0, T0-12, T>12

groupdataT0 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0' ], axis=1).transpose()
groupdataT012 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T0-12' ], axis=1).transpose()
groupdataT12 = pd.concat([exceldata.iloc[i] for i, x in enumerate(exceldata['d_time']) if x == 'T>12' ], axis=1).transpose()

#give a specific parameter

parameter = 'T16_TBR_mean' #can be changed to every parameter, the excel table contains

if len(groupdataT0[parameter].dropna()) != 0:
    groupname = ['T0', 'T0-12', 'T>12']
    groups=[groupdataT0[[parameter, 'Ground_Truth']].dropna(), groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]
else:
    groupname = ['T0-12', 'T>12']
    groups=[groupdataT012[[parameter, 'Ground_Truth']].dropna(), groupdataT12[[parameter, 'Ground_Truth']].dropna()]
#group[0] is T0, group[1] is T0-12, group[2] is T>12 or (if T0 is not available) T0-12 is group[0] and T>12 ist group[1]

### Old version with comments ###

#Analysis group T0
# cm_array, result, bestthreshold, youden, auc = ([[]] * len(groups) for i in range(5)) #analysis data for each group is stored in these lists
#
# # cm_array = {'T0': [], 'T0-12': [], 'T>12': []}
# # ich habe count durch i ersetzt und group in groups umbenannt.
# # Es ist immer sinnvoll den variablen namen aussagekräftige namen zu geben (z.B. ob es sich um einzelne oder mehrere gruppen handelt)
# # 1) Äußere Loop über die einzelnen Gruppen (Zeitpunkte)
# for i, group in enumerate(groups):  #analysis for the different groups
#     # thresholds = group.sort_values(by=[parameter])[parameter]
#     thresholds = group[parameter].sort_values()
#
#     truth = (group['Ground_Truth'] != 'RI').astype(int)
#
#     ###
#     # Mach es nicht zu verwirrend. Definier lieber mal die ein oder andere Variable mehr.
#     values = group[parameter]
#     ###
#
#     # predictions = [(group[i][parameter] >= threshold).astype(int) for threshold in thresholds]
#     #
#     # cm_array[count] = np.zeros((len(thresholds), 2, 2))
#     # for i, _ in enumerate(thresholds):
#     #     cm_array[count][i] = metrics.confusion_matrix(truth, predictions[i])
#
#     ###
#     # Hier kannst du dir glaube ich sparen den array erst mit zeros zu erzeugen.
#     # Du kannst auch dir auch eine for-loop sparen wenn du nur einmal über alle thresholds iterierst z.B. so:
#     for threshold in thresholds:
#         prediction = values >= threshold.astype(int)
#         cm_array[i].append(metrics.confusion_matrix(truth, prediction))
#     ###
#
#     dic = {'threshold': [], 'tpr': [], 'fpr': [], 'spe': [], 'multiplication': [], 'youden': [], 'TN': [], 'FP': [], 'FN': [], 'TP': []}
#
#     # 2) Innere Loop über die einzelnen Confusion matritzen der individuellen thresholds
#     for j, cm in enumerate(cm_array[i]):
#         tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # Ich würde einzelne Funktionen für Sensitivität etc. erstellen. So finde ich es schwierig zu lesen.
#         dic['tpr'].append(tpr)
#         fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
#         dic['fpr'].append(fpr)
#         spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#         dic['spe'].append(spe)
#         dic['threshold'].append(thresholds.iloc[j])
#         dic['multiplication'].append(tpr*spe)
#         dic['youden'].append(tpr+spe-1)
#         dic['TN'].append(cm[0, 0])
#         dic['FP'].append(cm[0, 1])
#         dic['FN'].append(cm[1, 0])
#         dic['TP'].append(cm[1, 1])
#
#     result[i] = pd.DataFrame(dic)
#
#     youden[i] = result[i]['threshold'][np.argmax(result[i]['youden'])]
#     bestthreshold[i] = result[i]['threshold'][np.argmax(result[i]['multiplication'])]
#
#     areaundercurve = metrics.auc(result[i]['fpr'], result[i]['tpr'])
#     auc[i] = areaundercurve

#%% Alternativer Vorschlag ### dein code ist gut und funktioniert ja auch.
# hier mal eine Anregung um die dinge besser zu lesen..


def sensitivity(cm):
    """
    Calculates sensitivity for a given confusion matrix
    Sensitivity = True Positives / (True Positives + False Negative) (Amount of correctly identified diseases subjects)
    :param cm: confusion matrix with index (0, 0): True Negatives, (0, 1): ...
    :return: sensitivity
    """
    tp = cm[1, 1]
    fn = cm[1, 0]
    sen = tp / (tp + fn)
    return sen


def specificity(cm):
    """
    Calculates specificity for a given confusion matrix
    Specificity = True Negatives / (True Negatives + False Positives) (Amount of correctly identified healthy subjects)
    :param cm: ...
    :return: ...
    """
    tn = cm[0, 0]
    fp = cm[0, 1]
    spe = tn / (tn + fp)
    return spe


result = {'group': [], 'predictions': [], 'best_threshold': [], 'auc': []} # 'fpr': [], 'spe': [], 'multiplication': [], 'youden': []}

group_names = ['T0', 'T0-12', 'T>12']
for i, group in enumerate(groups):
    thresholds = group[parameter].sort_values()
    truth = (group['Ground_Truth'] != 'RI').astype(int)
    values = group[parameter]

    result_one_group = {'group': [], 'threshold': [], 'sen': [], 'spe': [], 'fpr': [], 'multiplication': []} # 'fpr': [], 'spe': [], 'multiplication': [], 'youden': []}
    for threshold in thresholds:
        result_one_group['group'].append(group_names[i])
        result_one_group['threshold'].append(threshold)
        prediction = (values >= threshold).astype(int)
        cm = metrics.confusion_matrix(truth, prediction)
        sen = sensitivity(cm)
        result_one_group['sen'].append(sen)
        spe = specificity(cm)
        result_one_group['spe'].append(spe)
        result_one_group['fpr'].append(1 - spe)

        # ... usw. für die anderen Parameter
        result_one_group['multiplication'].append(sen*spe)


    result['group'].append(group_names[i])
    result['predictions'].append(result_one_group)
    index_best_result = np.argmax(result_one_group['multiplication'])
    result['best_threshold'].append(result_one_group['threshold'][index_best_result])

    result['auc'].append(metrics.auc(result_one_group['fpr'], result_one_group['sen']))
    # ...

#%%
#plot

# for i in range(len(group)):
#     plt.figure(groupname[i])
#     plt.plot(result[i]['fpr'].values, result[i]['tpr'].values, label='ROC curve (area = %0.2f, best threshold = %0.2f, n=%0.0f)' %(auc[i], bestthreshold[i], len(result[i]['tpr'])))
#     plt.plot([0, 1], [0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(groupname[i] + ' - ' + parameter)
#     plt.legend(loc="lower right")
#
# plt.show()


#%% PLOTTING ###
# ich würde immer mit plt.subplots() arbeiten meiner Erfahrung nach das einfachste um mehrere Abbildungen darzustellen
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
axs = axs.ravel()
for i, ax in enumerate(axs):
    ax.plot(np.array(result['predictions'][i]['fpr']), np.array(result['predictions'][i]['sen']),
            label=f'AUC: {round(result["auc"][i], 3)}')
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('-'.join([result['group'][i], parameter]))
    ax.legend(loc="lower right")

plt.show()

#%%

#statistical operations

# das ist leider viel zu unübersichtlich. Schreib dann Code immer so wie wenn du in 6 Monaten nochmal drauf guckst
# und jedem Gedankengang folgen kannst (hier helfen Sturktur und einfache Variablen Bezeichnungen)

# k2RIlist, pRIlist, k2Relapselist, pRelapselist, group_mean, group_std, groupRelapse, groupRI, groupRelapse_mean, groupRelapse_std, groupRI_mean, groupRI_std, t2, p2 , slist, plist= ([] for i in range(16))
#
# for count in range(len(group)):  #divide groups in RI and Relapse for ttest
#     groupRI.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'RI'], axis=1).transpose())
#     groupRelapse.append(pd.concat([group[count].iloc[i] for i, x in enumerate(group[count]['Ground_Truth']) if x == 'Relapse'], axis=1).transpose())
#
# for count in range(len(group)):  #calculate mean, std of the different groups
#     group_mean.append(np.mean(group[count][parameter]))
#     group_std.append(np.std(group[count][parameter]))
#     groupRelapse_mean.append(np.mean(groupRelapse[count][parameter]))
#     groupRelapse_std.append(np.std(groupRelapse[count][parameter]))
#     groupRI_mean.append(np.mean(groupRI[count][parameter]))
#     groupRI_std.append(np.std(groupRI[count][parameter]))


#%% Boxplot Beispiel
# Hier kannst du noch mit den parametern rumspielen und einen schönen plot erstellen
# einfach mal in der seaborn documentation gucken
# Hier kannst du auch wieder mit subplots mehrere Abbildungen erstellen

plt.figure(figsize=(3, 5))
sns.boxplot(data=group, x='Ground_Truth', y=parameter)
sns.swarmplot(data=group, x='Ground_Truth', y=parameter, color='.25')
plt.show()

# Wieso kommen soviele Werte mit 0 vor??

#%% Divide in two groups
group_names = ['T0', 'T0-12', 'T>12']
for i, group in enumerate(groups):
    values = group[parameter]
    RI_values = group[group['Ground_Truth'] == 'RI'][parameter]
    Relapse_values = group[group['Ground_Truth'] == 'Relapse'][parameter]

    # ... statistische Tests ...
    _, p_val = ranksums(RI_values, Relapse_values)

# Ich würde mir auch mal die Verteilungen der Wert anschauen mit einfachen histogrammen

#%%

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

for count in range(len(group)):  #calculate t and p of the different groups
    s, p = ranksums(groupRI[count][parameter], groupRelapse[count][parameter])

    slist.append(s)
    plist.append(p)

##ttest between RI and Relapsed

#for count in range(len(group)):  #calculate t and p of the different groups
    #k2RI, pRI = stats.normaltest(groupRI[count][parameter])
    #k2Relapse, pRelapse = stats.normaltest(groupRelapse[count][parameter])
    #t, p = ttest_ind_from_stats(groupRelapse_mean[count], groupRelapse_std[count], groupRelapse[count][parameter].size,
                              #groupRI_mean[count], groupRI_std[count], groupRI[count][parameter].size,
                              #equal_var = False)

    #k2RIlist.append(k2RI)
    #pRIlist.append(pRI)
    #k2Relapselist.append(k2Relapse)
    #pRelapselist.append(pRelapse)
    #t2.append(t)
    #p2.append(p)

#excel output

# das sieht super kompliziert aus. Würde ich dringend vereinfachen.
output = pd.DataFrame(
    np.array([[auc[i] for i in range(3)], [slist[i] for i in range(3)], [plist[i] for i in range(3)],
              [bestthreshold[i] for i in range(3)], [result[i]['TN'][np.argmax(result[i]['multiplication'])] for i in range(3)], [result[i]['FP'][np.argmax(result[i]['multiplication'])] for i in range(3)],
              [result[i]['FN'][np.argmax(result[i]['multiplication'])] for i in range(3)], [result[i]['TP'][np.argmax(result[i]['multiplication'])] for i in range(3)],
              [youden[i] for i in range(3)], [result[i]['TN'][np.argmax(result[i]['youden'])] for i in range(3)], [result[i]['FP'][np.argmax(result[i]['youden'])] for i in range(3)],
              [result[i]['FN'][np.argmax(result[i]['youden'])] for i in range(3)], [result[i]['TP'][np.argmax(result[i]['youden'])] for i in range(3)],
              [group_mean[i] for i in range(3)], [group_std[i] for i in range(3)], [len(group[i]) for i in range(3)],
              [groupRI_mean[i] for i in range(3)], [groupRI_std[i] for i in range(3)], [len(groupRI[i]) for i in range(3)],
              [groupRelapse_mean[i] for i in range(3)], [groupRelapse_std[i] for i in range(3)], [len(groupRelapse[i]) for i in range(3)]]),
    index=['AUC', 's_mann_whitney', 'p_mann_whitney',
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

print()