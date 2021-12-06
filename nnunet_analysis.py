from os.path import isfile, join
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ranksums
from os import listdir
from scipy import stats
import glob
import shutil
from sklearn.model_selection import train_test_split
from nibabel.testing import data_path
import nibabel as nib
import gzip
from collections import Counter
import SimpleITK as sitk
import six
from radiomics import featureextractor, getTestCase

path = '/Users/carsten/Desktop/'
#path should contain folders: truths, predictions, images, combined
paramPath = '/Users/carsten/PycharmProjects/masterproject/params.yaml'

path_truth = path + '072_segm_groundtruth'
path_predict = path + '072_segm_predictions'
path_image = path + '072_segm_images'

images = glob.glob(path_image + '/*.nii.gz')
images = sorted(images)
truths = glob.glob(path_truth + '/*.nii.gz')
truths = sorted(truths)
predictions = glob.glob(path_predict + '/*.nii.gz')
predictions = sorted(predictions)

equal1 = []
for i in range(len(truths)):
    equal1.append(os.path.basename(truths[i]))

equal2 = []
for i in range(len(predictions)):
    equal2.append(os.path.basename(predictions[i]))

equal = list(set(equal1+equal2))
equal = sorted(equal)

truths_equal = []
predictions_equal = []
images_equal = []
for i in range(len(equal)):
    truths_equal.append(path_truth + '/' + equal[i])
    predictions_equal.append(path_predict + '/' + equal[i])
    images_equal.append(path_image + '/' + equal[i][:4]+'.nii.gz')

#result = pd.DataFrame(index=['dice', 'overlap_voxel', 'image_voxel',
#                             't_voxel', 't_elongation', 't_flatness', 't_leastaxislength', 't_majoraxislength',
#                             't_minoraxislength', 't_max3Ddiameter', 't_sphericity', 't_surfacearea', 't_surfacevolumeratio',
#                             'p_voxel', 'p_elongation', 'p_flatness', 'p_leastaxislength', 'p_majoraxislength',
#                             'p_minoraxislength', 'p_max3Ddiameter', 'p_sphericity', 'p_surfecearea', 'p_surfacevolumeratio'])

o = 0
tp = []
fp = []
fn = []
feature_names = []
extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
temp = extractor.execute(truths_equal[0], truths_equal[0])
for key in temp.keys():
    feature_names.append(key)

result_t = pd.DataFrame(index=temp)
result_p = pd.DataFrame(index=temp)

result = pd.DataFrame(index=['dice', 'overlap_vx', 'image_vx', 'truth_vx', 'prediction_vx'])

for i in range(len(equal)):
#for i in range(30):
    if images_equal[i] not in images:
        print(equal[i] + ' error1')
        if truths_equal[i] not in truths:
            print(equal[i] + ' error2')
        elif predictions_equal[i] not in predictions:
            print(equal[i] + ' error3')
    elif truths_equal[i] not in truths:
        fp.append(i)
        print(equal[i] + ' false positive')
        if images_equal[i] not in images:
            print(equal[i] + ' error4')
        elif predictions_equal[i] not in predictions:
            print(equal[i] + ' error5')
    elif predictions_equal[i] not in predictions:
        fn.append(i)
        print(equal[i] + ' false negative')
        if truths_equal[i] not in truths:
            print(equal[i] + ' error6')
        elif images_equal[i] not in images:
            print(equal[i] + ' error7')
    elif predictions_equal[i] in predictions and truths_equal[i] in truths and images_equal[i] in images:
        t = nib.load(truths_equal[i])
        t = np.array(t.dataobj)
        p = nib.load(predictions_equal[i])
        p = np.array(p.dataobj)
        if len(t.shape)==4:
            t = t[:, :, :, 0]
        if len(p.shape)==4:
            p = p[:, :, :, 0]
        sum = t + p
        one = np.count_nonzero(sum==1)
        two = np.count_nonzero(sum==2)
        dice = ((2 * two)/(2 * two + one))
        print(equal[i] + ' true positive')
        #print('dice: ', dice)
        #print('size truth: ', np.count_nonzero(t==1))
        #print('size predictions: ', np.count_nonzero(p==1))
        #print('number of identical voxel: ', two)
        image = nib.load(images_equal[i])
        combined_mask = t + 2 * p
        combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
        #nib.save(combined_mask, path+'combined/'+os.path.basename(truths_equal[i]))

        #shape features
        #print(np.count_nonzero(t == 1))
        #print(np.count_nonzero(p == 1))
        if np.count_nonzero(p == 1) == 0:
            extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
            result_truth = extractor.execute(images_equal[i], truths_equal[i])
            result_prediction = result_truth
            print(equal[i] + ' error8')
            for key in result_prediction.keys():
                result_prediction[key]=0
                result_truth = extractor.execute(images_equal[i], truths_equal[i])
                print(equal[i] + ' error9')
        else:
            extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
            result_truth = extractor.execute(images_equal[i], truths_equal[i])
            result_prediction = extractor.execute(images_equal[i], predictions_equal[i])


        #output

        features = []
        for name in feature_names:
            features.append(result_truth[name])

        result_t.insert(o, equal[i], features, True)

        features = []
        for name in feature_names:
            features.append(result_prediction[name])

        result_p.insert(o, equal[i], features, True)

        result.insert(o, equal[i], [dice, two, p.shape[0]*p.shape[1]*p.shape[2], np.count_nonzero(t==1), np.count_nonzero(p==1)], True)

        o = o + 1
        tp.append(i)

    print('progress:', i + 1, '/', len(equal))

result_t = result_t.drop(feature_names[0:22])
result_p = result_p.drop(feature_names[0:22])

result_t = [result, result_t]
result_p = [result, result_p]

result_t = pd.concat(result_t)
result_p = pd.concat(result_p)

dices = result.to_numpy()[0]
print('DICE = ', np.mean(dices), ' ± ', np.std(dices))
print('sample size: ', i + 1)
print('true positive: ', len(tp))
print('false positive: ', len(fp))
print('false negative: ', len(fn))

y = list(result_t.to_numpy()[0])

for i in range(len(result_t.index)-1):
    rp = stats.pearsonr(dices, result_t.to_numpy()[i+1])
    print(result_t.index[i+1], 'r, p = ', rp)
    x = list(result_t.to_numpy()[i+1])
    #plt.scatter(x, y)
    #plt.show()

result_t.to_csv(path + 'resultnew_nnUNet072.csv', index=True)

#fn, fp
result_fn = pd.DataFrame(index=temp)
o = 0
for num in fn:
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
    result_temp = extractor.execute(images_equal[num], truths_equal[num])

    features = []
    for name in feature_names:
        features.append(result_temp[name])

    result_fn.insert(o, equal[num], features, True)
    o = o + 1

result_fn = result_fn.drop(feature_names[0:22])

result_fp = pd.DataFrame(index=temp)
o = 0
for num in fp:
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
    result_temp = extractor.execute(images_equal[num], predictions_equal[num])

    features = []
    for name in feature_names:
        features.append(result_temp[name])

    result_fp.insert(o, equal[num], features, True)
    o = o + 1

result_fp = result_fp.drop(feature_names[0:22])

#small and big tumours
vol = result_t.iloc[[13]].values.tolist()
median_vol = np.median(vol)

result_big = pd.DataFrame(index=result_t.index.values.tolist())
result_small = pd.DataFrame(index=result_t.index.values.tolist())

for column in result_t.columns:
    if result_t.loc['original_shape_MeshVolume', column] <= median_vol:
        result_small[column] = result_t.loc[:, column]
    elif result_t.loc['original_shape_MeshVolume', column] > median_vol:
        result_big[column] = result_t.loc[:, column]

#plots
parameter = 'original_shape_MeshVolume'
#'original_shape_Sphericity'
#'original_shape_MeshVolume'
name = 'volume'
unit = ' (ml)'
data_conv = 0.001

font = {'size': 15}

plt.rc('font', **font)

big_data = [ e * data_conv for e in result_big.loc[parameter].values.tolist()]
small_data = [ e * data_conv for e in result_small.loc[parameter].values.tolist()]
truth_data = [ e * data_conv for e in result_t.loc[parameter].values.tolist()]
prediction_data = [ e * data_conv for e in result_p.loc[parameter].values.tolist()]
if parameter in result_fn.index.tolist():
    fn_data = [ e * data_conv for e in result_fn.loc[parameter].values.tolist()]
    fp_data = [ e * data_conv for e in result_fp.loc[parameter].values.tolist()]

#bland altmann
def relative_bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = (data1 - data2)/data1           # Difference between data1 and data2 in relation to data1
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

plt.figure(1)
bland_altman_plot(truth_data, prediction_data)
plt.ylabel('Difference between true ' + name + ' and predicted ' + name + unit)
plt.xlabel('True ' + name + unit)

#plt.savefig((path + 'figures/' + 'bland_altman_plot.pdf'), format = 'pdf', bbox_inches = 'tight')

plt.figure(2)
relative_bland_altman_plot(truth_data, prediction_data)
plt.ylabel('Ratio between true and predicted ' + name + 's (%)')
plt.xlabel('True ' + name + unit)

#plt.savefig((path + 'figures/' + 'relative_bland_altman_plot.pdf'), format = 'pdf', bbox_inches = 'tight')

plt.figure(3)
plt.plot(truth_data, prediction_data, '.')
plt.plot([np.min(truth_data+prediction_data), np.max(truth_data+prediction_data)], [np.min(truth_data+prediction_data), np.max(truth_data+prediction_data)], '--' ,color = 'black')
plt.show()
plt.xlabel('True ' + name + unit)
plt.ylabel('Predicted ' + name + unit)

#plt.savefig((path + 'figures/' + 'true_predicted.pdf'), format = 'pdf', bbox_inches = 'tight')

#swarm plot
swarm_list1 = truth_data + fn_data + fp_data
swarm_list2 = []
for i in range(len(prediction_data)):
    swarm_list2.append('Detected lesions')
for i in range(len(fn_data)):
    swarm_list2.append('Missed lesions')
for i in range(len(fp_data)):
    swarm_list2.append('False positiv')

plt.figure(4)
swarm_data = pd.DataFrame({"Lesions": swarm_list2, parameter: swarm_list1})
ax = sns.swarmplot(x = "Lesions", y = parameter, data = swarm_data)
plt.xlabel('')
plt.ylabel(name.capitalize() + unit)

#plt.savefig((path + 'figures/' + 'swarm.pdf'), format = 'pdf', bbox_inches = 'tight')

#Statistics dependent on the Parameter
stats_data = swarm_data.sort_values(by = [parameter], ascending = False)
stats_data = stats_data.reset_index(drop = True)

s1 = 1
s2 = 1
s3 = 1
sum_tp = []
sum_fn = []
sum_fp = []
sen = []
acc = []
for i in range(len(stats_data)):
    if stats_data['Lesions'][i] == 'Detected lesions':
        sum_tp.append(s1)
        s1 = s1 + 1
        if i == 0:
            sum_fn.append(0)
            sum_fp.append(0)
        else:
            sum_fn.append(sum_fn[i-1])
            sum_fp.append(sum_fp[i-1])
    elif stats_data['Lesions'][i] == 'Missed lesions':
        sum_fn.append(s2)
        s2 = s2 + 1
        if i == 0:
            sum_tp.append(0)
            sum_fp.append(0)
        else:
            sum_tp.append(sum_tp[i-1])
            sum_fp.append(sum_fp[i-1])
    elif stats_data['Lesions'][i] == 'False positiv':
        sum_fp.append(s2)
        s3 = s3 + 1
        if i == 0:
            sum_tp.append(0)
            sum_fn.append(0)
        else:
            sum_tp.append(sum_tp[i-1])
            sum_fn.append(sum_fn[i-1])

    sen.append(sum_tp[i]/(sum_tp[i]+sum_fn[i]))
    acc.append(sum_tp[i]/(sum_tp[i]+sum_fp[i]+sum_fn[i]))

stats_data.insert(2, 'tp', sum_tp)
stats_data.insert(3, 'fn', sum_fn)
stats_data.insert(4, 'fp', sum_fp)
stats_data.insert(5, 'sensitivity', sen)
stats_data.insert(6, 'accuracy', acc)

stats_data = stats_data.sort_values(by = [parameter])
stats_data = stats_data.reset_index(drop = True)

plt.figure(5)
plt.plot(stats_data[parameter].tolist(), stats_data['accuracy'].tolist(), label = 'Accuracy')
plt.plot(stats_data[parameter].tolist(), stats_data['sensitivity'].tolist(), label = 'Sensitivity')
plt.show()
plt.xlabel(name.capitalize() + unit)
plt.ylabel('Accuracy and Sensitivity')
plt.legend()

#plt.savefig((path + 'figures/' + 'accuracy_sensitivity.pdf'), format = 'pdf', bbox_inches = 'tight')


