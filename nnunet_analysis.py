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

result = pd.DataFrame(index=['dice', 'overlap_voxel', 'image_voxel',
                             't_voxel', 't_elongation', 't_flatness', 't_leastaxislength', 't_majoraxislength',
                             't_minoraxislength', 't_max3Ddiameter', 't_sphericity', 't_surfacearea', 't_surfacevolumeratio',
                             'p_voxel', 'p_elongation', 'p_flatness', 'p_leastaxislength', 'p_majoraxislength',
                             'p_minoraxislength', 'p_max3Ddiameter', 'p_sphericity', 'p_surfecearea', 'p_surfacevolumeratio'])

o = 0
tp = 0
fp = 0
fn = 0
#for i in range(len(equal)):
for i in range(93):
    if images_equal[i] not in images:
        print(equal[i] + ' error1')
        if truths_equal[i] not in truths:
            print(equal[i] + ' error2')
        elif predictions_equal[i] not in predictions:
            print(equal[i] + ' error3')
    elif truths_equal[i] not in truths:
        fp = fp + 1
        print(equal[i] + ' false positive')
        if images_equal[i] not in images:
            print(equal[i] + ' error4')
        elif predictions_equal[i] not in predictions:
            print(equal[i] + ' error5')
    elif predictions_equal[i] not in predictions:
        fn = fn + 1
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
        print(equal[i])
        #print('dice: ', dice)
        #print('size truth: ', np.count_nonzero(t==1))
        #print('size predictions: ', np.count_nonzero(p==1))
        #print('number of identical voxel: ', two)
        image =nib.load(images_equal[i])
        combined_mask = t + 2 * p
        combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
        #nib.save(combined_mask, path+'combined/'+os.path.basename(truths_equal[i]))

        #shape features
        print(np.count_nonzero(t == 1))
        print(np.count_nonzero(p == 1))
        if np.count_nonzero(p == 1) == 0:
            extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
            result_truth = extractor.execute(truths_equal[i], truths_equal[i])
            result_prediction = result_truth
            for key in result_prediction.keys():
                result_prediction[key]=0
                result_truth = extractor.execute(truths_equal[i], truths_equal[i])
        else:
            extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
            result_truth = extractor.execute(truths_equal[i], truths_equal[i])
            result_prediction = extractor.execute(predictions_equal[i], predictions_equal[i])

        features = [dice, two, p.shape[0] * p.shape[1] * p.shape[2],    #dice, overlap, image voxel
                    np.count_nonzero(t == 1), result_truth['original_shape_Elongation'],  #truth features
                    result_truth['original_shape_Flatness'], result_truth['original_shape_LeastAxisLength'],
                    result_truth['original_shape_MajorAxisLength'], result_truth['original_shape_MinorAxisLength'],
                    result_truth['original_shape_Maximum3DDiameter'], result_truth['original_shape_Sphericity'],
                    result_truth['original_shape_SurfaceArea'], result_truth['original_shape_SurfaceVolumeRatio'],
                    np.count_nonzero(p == 1), result_prediction['original_shape_Elongation'],   #prediction features
                    result_prediction['original_shape_Flatness'], result_prediction['original_shape_LeastAxisLength'],
                    result_prediction['original_shape_MajorAxisLength'], result_prediction['original_shape_MinorAxisLength'],
                    result_prediction['original_shape_Maximum3DDiameter'], result_prediction['original_shape_Sphericity'],
                    result_prediction['original_shape_SurfaceArea'], result_prediction['original_shape_SurfaceVolumeRatio'],
                    ]
        #output
        result.insert(o, equal[i], features, True)
        o = o + 1
        print(equal[i] + ' true positive')
        tp = tp + 1

    print('progress:', i + 1, '/', len(equal))

print('DICE: ', np.mean(dices), ' ± ', np.std(dices))
print('sample size: ', i + 1)
print('true positive: ', tp)
print('false positive: ', fp)
print('false negative: ', fn)


dices = result.to_numpy()[0]
for i in range(len(result.index)-1):
    rp = stats.pearsonr(dices, result.to_numpy()[i+1])
    print(result.index[i+1], 'r, p = ', rp)


#result.to_excel(path + 'result.xlsx', index=True)
