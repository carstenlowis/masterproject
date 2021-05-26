from os.path import join
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = '_Patiententabelle_Serial_Imaging_BM_anonymized_20052021.xlsx'
exceldata = pd.read_excel(join(path, file))
PET_ID = exceldata[["PET_ID", "T16_SUV_mean", "Ground_Truth"]]

#%% 1)
# bitte in Zukunft mehr kommentieren!

# deine version
mask = []
for x in range(0, len(PET_ID)):
    if (PET_ID.iloc[x][0][-1].isdigit() == False) and (len(PET_ID.iloc[x][0]) != 3):
        mask.append(True)
    else:
        mask.append(False)

T16_SUV_mean_list1 = PET_ID[mask].dropna()

# verbesserungs vorschlag
mask = []
for i, pid in enumerate(PET_ID['PET_ID']):  # enumerate erzeugt: index, item
    if not (pid[-1].isdigit()) and (len(pid) != 3):
        mask.append(True)
    else:
        mask.append(False)

T16_SUV_mean_list1 = PET_ID[mask].dropna()

# alternative version
SUV_mean = pd.concat([PET_ID.iloc[i] for i, x in enumerate(PET_ID['PET_ID']) if not x[-1].isdigit() and len(x) != 3],
                     axis=1).transpose()


#%% 2)
#confusion matrix

# deine version
thresholds = T16_SUV_mean_list1.sort_values(by=['T16_SUV_mean']).T16_SUV_mean.values.tolist()
# verbesserungs vorschlag
thresholds = T16_SUV_mean_list1.sort_values(by=['T16_SUV_mean'])['T16_SUV_mean']

truth = []
for x in range(0, len(T16_SUV_mean_list1)):
    if T16_SUV_mean_list1.Ground_Truth.iloc[x] == 'RI':
        truth.append(0)
    else:
        truth.append(1)

# verbesserungs vorschlag
truth = []
for i, x in enumerate(T16_SUV_mean_list1['Ground_Truth']):  # ich würde immer i für den Index nehmen
    if x == 'RI':
        truth.append(0)
    else:
        truth.append(1)

# alternative version
truth = (T16_SUV_mean_list1['Ground_Truth'] != 'RI').astype(int)


#%% 3)
# deine version
predictions = np.zeros((len(T16_SUV_mean_list1), len(thresholds)))
for x in range(0, len(thresholds)):
    for i in range(0, len(T16_SUV_mean_list1)):
        if ( T16_SUV_mean_list1.iloc[i]['T16_SUV_mean'] >= thresholds[x] ):
            predictions[i][x] = 1
        else:
            predictions[i][x] = 0

# Verbesserungs Vorschlag
predictions = np.zeros((len(T16_SUV_mean_list1), len(thresholds)))
for i, threshold in enumerate(thresholds):
    for j, suv_mean in enumerate(T16_SUV_mean_list1['T16_SUV_mean']):
        if suv_mean >= threshold:
            predictions[j][i] = 1
        else:
            predictions[j][i] = 0

# Alternative
predictions = [(T16_SUV_mean_list1['T16_SUV_mean'] >= threshold).astype(int) for threshold in thresholds]


# deine version
cm_array=np.zeros((len(thresholds), 2, 2))
for x in range(0, len(thresholds)):
    cm_array[x] = metrics.confusion_matrix(truth, predictions[:, x])

# Verbesserungs Vorschlag
cm_array = np.zeros((len(thresholds), 2, 2))
for i, _ in enumerate(thresholds):
    cm_array[i] = metrics.confusion_matrix(truth, predictions[i])


#%% 4)
#roc-curve

# deine version
fpr=[]
tpr=[]
for x in range(0, len(cm_array)):
    tpr.append(cm_array[x, 1, 1] / (cm_array[x, 1, 1] + cm_array[x, 1, 0]))
    fpr.append(cm_array[x, 0, 1] / (cm_array[x, 0, 1] + cm_array[x, 0, 0]))

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# kann es sein dass du dich vertan hast. Lies nochmal welche TN, TP, FN, FP sind.

areaundercurve = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % areaundercurve)
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: T16_SUV_mean')
plt.legend(loc="lower right")
plt.show()


# Vebesserungs Vorschlag
# vielleicht noch irgendwie sowas dass du ein ergebniss als tabelle hast:

dic = {'threshold': [], 'tpr': [], 'fpr': [], 'spe': [], 'multiplication': []}

for i, cm in enumerate(cm_array):
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    dic['tpr'].append(tpr)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    dic['fpr'].append(fpr)
    spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    dic['spe'].append(spe)
    dic['threshold'].append(thresholds[i])
    dic['multiplication'].append(tpr*spe)

result = pd.DataFrame(dic)  # jetzt kannst du dann direkt auch den besten threshold rauslesen

bestthreshold=result['threshold'][np.argmax(result['multiplication'])]

areaundercurve = metrics.auc(result['fpr'], result['tpr'])

plt.plot(result['fpr'].values, result['tpr'].values, label='ROC curve (area = %0.2f)' % areaundercurve)
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve: T16_SUV_mean')
plt.legend(loc="lower right")
plt.show()
