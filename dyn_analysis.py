from os.path import join
import pandas as pd
import numpy as np

#import data
path = 'Z:\MITARBEITER\Lowis'
file = 'dyn_excel.xlsx'
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
start = 9
coef=[]
for i in range(1, dyndata.shape[1]):
    coef.append(np.polyfit(dyndata.iloc[start:, 0].values.tolist(), dyndata.iloc[start:, i].values.tolist(), 1))

for i in range(len(coef)):
    coef[i][0] = coef[i][0] * 3600

result=pd.DataFrame(coef, index=columns, columns=['slope', 'y axis intercept'])

#
print(result)

#plot
#for i in range(6):
#    plt.figure(columns[i])
#    plt.plot(dyndata.iloc[:, 0].values.tolist(), dyndata.iloc[:, i+1].values.tolist())

result.to_excel("dyn_result.xlsx")
