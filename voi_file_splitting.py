import os
import glob

path = 'Y:/data/_Temp/PET_DATA_FOR_SEGMENTATION/BIOPSY_STUDY'
voi_file = glob.glob(path+'/*/*/*.voi')

dir = '/seperated_voi'
os.mkdir(path + dir)

for count in range(len(voi_file)):
    #read data
    orig = open(voi_file[count], 'r')
    orig = orig.read()
    orig = orig.splitlines()
    #number of VOIs
    n = int(orig[15])

    #create header, bottom and find positions of the VOIs and names
    header = orig[:18]
    header [15] = '1'
    voi_index = []
    bottom = []
    names = []
    for i in range(len(orig)):
        if 'voi_time_list_index voi_direction' in orig[i]:
            voi_index.append(i)
            names.append(orig[i+1][:orig[i+1].find('#')])
        elif '#END OF VOIS DEFINITION' in orig[i]:
            voi_index.append(i-1)
            bottom.append(orig[i - 1])
            bottom.append(orig[i])

    #create new txt files according to the number of VOIs
    for i in range(n):
        temp = orig[voi_index[i]:voi_index[i+1]]

        if '#VOI TIME LIST NUMBER' in temp[-1]:
            del temp[-1]

        new = header + temp + bottom
        name = os.path.basename(voi_file[count])[:-4] + '_VOI_' + names[i][:-1] + '.voi'\

        if len(temp) > 9:
            if 'operation' in temp[10]:
                name = name[:-4] + '_incompatible.voi'

        new [19] = names[i] + '# voi_name'

        f = open(path + dir + '/' + name, 'x')
        f.write('\n'.join(new))
        f.close()

    print(os.path.basename(voi_file[count])[:-4])
    print('progress: ', count + 1, ' / ', len(voi_file))
