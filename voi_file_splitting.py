import os
import glob

path = 'H:/Pmod_test/'
voi_file = glob.glob(path+'/*.voi')

os.mkdir(path + 'voi/')

for count in range(len(voi_file)):
    #converion from .voi to .txt
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
        if '</UID>' in orig[i][-6:]:
            voi_index.append(i)
            names.append(orig[i+1][:orig[i+1].find('#')])
        elif '#END OF VOIS DEFINITION' in orig[i]:
            voi_index.append(i-1)
            bottom.append(orig[i - 1])
            bottom.append(orig[i])

    #create new txt files according to the number of VOIs
    for i in range(n):
        temp = orig[voi_index[i]:voi_index[i+1]-1]
        new = header + temp + bottom
        name = voi_file[count][len(path):-4] + '_VOI_' + names[i][:-1] + '.voi'
        new [19] = names[i] + '# voi_name'

        f = open(path + '/voi/' + name, 'x')
        f.write('\n'.join(new))
        f.close()

