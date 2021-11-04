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

path = '/Users/carsten/Desktop/nnunet_test/lowis/'
#path should contain folders: truths, predictions, images, combined
paramPath = '/Users/carsten/PycharmProjects/masterproject/params.yaml'


images = glob.glob(path + 'images/*')
images = sorted(images)
truths = glob.glob(path + 'truths/*')
truths = sorted(truths)
predictions = glob.glob(path + 'predictions/*')
predictions = sorted(predictions)

result = pd.DataFrame(index=['dice', 'overlap_voxel', 'image_voxel',
                             't_voxel', 't_elongation', 't_flatness', 't_leastaxislength', 't_majoraxislength',
                             't_minoraxislength', 't_max3Ddiameter', 't_sphericity', 't_surfacearea', 't_surfacevolumeratio',
                             'p_voxel', 'p_elongation', 'p_flatness', 'p_leastaxislength', 'p_majoraxislength',
                             'p_minoraxislength', 'p_max3Ddiameter', 'p_sphericity', 'p_surfecearea', 'p_surfacevolumeratio'])

for i in range(len(truths)):
    t = nib.load(truths[i])
    t = np.array(t.dataobj)
    p = nib.load(predictions[i])
    p = np.array(p.dataobj)
    if len(t.shape)==4:
        t = t[:, :, :, 0]
    if len(p.shape)==4:
        p = p[:, :, :, 0]
    sum= t + p
    one = np.count_nonzero(sum==1)
    two = np.count_nonzero(sum==2)
    dice = ((2 * two)/(2 * two + one))
    #print(images[i])
    #print('dice: ', dice)
    #print('size truth: ', np.count_nonzero(t==1))
    #print('size predictions: ', np.count_nonzero(p==1))
    #print('number of identical voxel: ', two)
    image =nib.load(images[i])
    combined_mask = t + 2 * p
    combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
    nib.save(combined_mask, path+'combined/'+os.path.basename(truths[i]))

    #shape features
    if np.count_nonzero(p==1) == 0:
        extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
        result_truth = extractor.execute(images[i], truths[i])
        result_prediction = result_truth
        for key in result_prediction.keys():
            result_prediction[key]=0
        result_truth = extractor.execute(images[i], truths[i])
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
        result_truth = extractor.execute(images[i], truths[i])
        result_prediction = extractor.execute(images[i], predictions[i])

    features = [dice, two, p.shape[0]*p.shape[1]*p.shape[2],    #dice, overlap, image voxel
                np.count_nonzero(t==1), result_truth['original_shape_Elongation'],  #truth features
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
    result.insert(i, os.path.basename(truths[i]), features, True)
    print(os.path.basename(truths[i]))
    print('progress:', i+1, '/', len(truths))

result.to_excel(path + 'result.xlsx', index=True)
