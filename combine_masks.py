


masks1 = glob.glob('H:/temp/*1.nii.gz')
masks2 = glob.glob('H:/temp/*2.nii.gz')
masks3 = glob.glob('H:/temp/*3.nii.gz')
masks4 = glob.glob('H:/temp/*4.nii.gz')
masks5 = glob.glob('H:/temp/*5.nii.gz')

path = 'H:/temp/comb/'
masks = masks2
#combine masks
for i in range(len(masks)):
    m1 = nib.load(masks[i][:-8]+'1.nii.gz')
    m1 = np.array(m1.dataobj)
    m2 = nib.load(masks[i])
    m2 = np.array(m2.dataobj)
    image = nib.load(masks[i])
    combined_mask = m1 + m2
    combined_mask = nib.Nifti1Image(combined_mask, image.affine, image.header)
    nib.save(combined_mask, path+os.path.basename(masks[i])[:-8]+'1.nii.gz')
    print('progress: ', i + 1, '/', len(masks))

#set max to 1 and nan to 0

masks = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/masks/*1.nii.gz')
path = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/masks/'

for i in range(len(masks)):
    m = nib.load(masks[i])
    m = np.array(m.dataobj)
    image = nib.load(masks[i])
    n = list(set(list(m.flatten())))
    if len(n) > 2:
        print(masks[i])
        for i in range(len(n)-2):
            inds = np.where(np.isnan(i+2))
            m[inds] = 1
    inds = np.where(np.isnan(m))
    m[inds] = 0
    m = nib.Nifti1Image(m, image.affine, image.header)
    nib.save(m, path+os.path.basename(masks[i])[:-8]+'.nii.gz')
    print('progress: ', i + 1, '/', len(masks))

#remove 4th dim
masks = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/masks/*mask.nii.gz')
path = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/images_with_masks/'
for i in range(len(masks)):
    m = nib.load(masks[i])
    m = np.array(m.dataobj)
    image = nib.load(masks[i])

    if len(m.shape)==4:
        m = m[:, :, :, 0]
        print(masks[i])

    m = nib.Nifti1Image(m, image.affine, image.header)
    nib.save(m, path+os.path.basename(masks[i]))
    print('progress: ', i + 1, '/', len(masks))

#add images, remove nan and 4th dim
masks = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/images_with_masks/*mask.nii.gz')
images =[]
orig = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/images\\'
for i in range(len(masks)):
    images.append(orig + os.path.basename(masks[i])[:-11] + 'image.nii.gz')

actualimages = glob.glob('Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/images/*image.nii.gz')

path = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/_nnUNet_Lowis/All_PET_sum_and_masks/imagesnew/'
for i in range(len(images)):
    if images[i] in actualimages:
        m = nib.load(images[i])
        m = np.array(m.dataobj)
        image = nib.load(images[i])

        if len(m.shape)==4:
            m = m[:, :, :, 0]
            print(images[i])

        inds = np.where(np.isnan(m))
        m[inds] = 0

        m = nib.Nifti1Image(m, image.affine, image.header)
        nib.save(m, path+os.path.basename(images[i]))

    else:
        print('doesnt work: ', os.path.basename(images[i]))

    print('progress: ', i + 1, '/', len(images))


