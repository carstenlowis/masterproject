from os.path import join
import pandas as pd
import numpy as np

#import data
path = '/Volumes/BTU/MITARBEITER/Lowis/'
file = 'dyn_exceltest.xlsx'
data = pd.read_excel(join(path, file))

#delete unwanted information

data = data.dropna()
data = data[data.iloc[:, 1] != 'File [string]']

#create a more practical dataframe
dyndata=pd.DataFrame(np.array(data['Unnamed: 10'][0:16]), columns=['time'])

columns=[]
for i in range(int(len(data)/16)):
    columns.append(join(data.iloc[i*16][9],data.iloc[i*16][1][0:11]))
    for n in range(16):
        count=i*16+n
        dyndata = dyndata.append({columns[i]:data.iloc[count][11]}, ignore_index=True)

dyndata = dyndata.apply(lambda x: pd.Series(x.dropna().values))

#linear fit
coef=[]
for i in range(1, dyndata.shape[1]):
    coef.append(np.polyfit(dyndata.iloc[:, 0].values.tolist(), dyndata.iloc[:, i].values.tolist(), 1))

result=pd.DataFrame(coef, index=columns, columns=['slope', 'y axis intercept'])
print(result)



