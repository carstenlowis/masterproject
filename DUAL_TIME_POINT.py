

files = glob.glob('H:/DUAL_TIME_POINT/*/*.nii')

for i in range(len(files)):
    with open(files[i], 'rb') as f_in:
        with gzip.open(files[i]+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('zip_progress: ', i+1, '/', len(files))


masks = glob.glob('H:/DUAL_TIME_POINT/masks/*_Tum_*.nii.gz')
masks1 = glob.glob('H:/DUAL_TIME_POINT/masks/*_Tum1_*.nii.gz')
masks2 = glob.glob('H:/DUAL_TIME_POINT/masks/*_Tum2_*.nii.gz')
masks3 = glob.glob('H:/DUAL_TIME_POINT/masks/*_Tum3_*.nii.gz')

all_masks = masks + masks1 + masks2 + masks3

for i in range(len(masks1)):
    m1 = nib.load(masks1[i])
    m1 = np.array(m1.dataobj)
    m2 = nib.load(masks2[i])
    m2 = np.array(m2.dataobj)
    image = nib.load(masks[i])
    combined_mask = m1 + m2
    combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
    nib.save(combined_mask, path+os.path.basename(masks[i])[:-8]+'1.nii.gz')
    print('progress: ', i + 1, '/', len(masks))

all_masks = glob.glob('H:/DUAL_TIME_POINT/masks/*.nii.gz')
for i in range(len(all_masks)):
    name = all_masks[i][:25]+all_masks[i][50:53]+'-'+all_masks[i][55:60]+all_masks[i][all_masks[i].find('_Tum'):all_masks[i].find('_Tum')+5]+'_.nii.gz'
    os.rename(all_masks[i], name)

all_images = glob.glob('H:/DUAL_TIME_POINT/images/*.nii.gz')
for i in range(len(all_images)):
    name = all_images[i][:26]+all_images[i][56:59]+'-'+all_images[i][61:66]+'_.nii.gz'
    os.rename(all_images[i], name)

#combine masks
masks = glob.glob('H:/DUAL_TIME_POINT/masks/*.nii.gz')
path = 'H:/DUAL_TIME_POINT/masks_temp/'

for i in range(len(masks)):
    m = nib.load(masks[i])
    m = np.array(m.dataobj)
    image = nib.load(masks[i])
    n = list(set(list(m.flatten())))
    if len(n) > 2:
        print(masks[i])
        for c in range(len(n)-2):
            inds = np.where(np.isnan(c+2))
            m[inds] = 1
    inds = np.where(np.isnan(m))
    m[inds] = 0
    m = nib.Nifti1Image(m, image.affine, image.header)
    nib.save(m, path+os.path.basename(masks[i]))
    print('progress: ', i + 1, '/', len(masks))

masks1 = glob.glob('H:/DUAL_TIME_POINT/masks/*1_.nii.gz')
masks2 = glob.glob('H:/DUAL_TIME_POINT/masks/*2_.nii.gz')
masks3 = glob.glob('H:/DUAL_TIME_POINT/masks/*3_.nii.gz')

path = 'H:/DUAL_TIME_POINT/masks_temp/'
masks = masks3
#combine masks
for i in range(len(masks)):
    m1 = nib.load(masks[i][:-9]+'1_.nii.gz')
    m1 = np.array(m1.dataobj)
    m2 = nib.load(masks[i])
    m2 = np.array(m2.dataobj)
    image = nib.load(masks[i])
    combined_mask = m1 + m2
    print(np.max(combined_mask))
    combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
    nib.save(combined_mask, path+os.path.basename(masks[i])[:-9]+'1_.nii.gz')
    print('progress: ', i + 1, '/', len(masks))
