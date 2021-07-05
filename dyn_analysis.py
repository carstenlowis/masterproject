from os.path import join
import pandas as pd
import numpy as np
from collections import OrderedDict

#import data
path = 'Z:\MITARBEITER\Lowis'
file = 'dyn_excel.xlsx'
data = pd.read_excel(join(path, file))

#delete unwanted information

data = data.dropna()
data = data[data.iloc[:, 1] != 'File [string]']

#create a more practical dataframe
dyndata=pd.DataFrame(np.array(data['Unnamed: 10'][0:20]), columns=['datarange'])

columns=[]
for i in range(len(data)):
    columns.append(join(data.iloc[i][1][0:11], data.iloc[i][9]))

columns = list(OrderedDict.fromkeys(columns))

result = []
for i in columns:
    result.append(join('time',i))
    result.append(i)

columns = result

dyndata = pd.DataFrame(columns=[columns])


for i in range(len(data)):
    tempcolumn=[join('time',join(data.iloc[i][1][0:11], data.iloc[i][9])), join(join(data.iloc[i][1][0:11], data.iloc[i][9]))]
    tempdf = pd.DataFrame(np.array([[data.iloc[i][10], data.iloc[i][11]]]), columns=[tempcolumn])
    dyndata = pd.concat([dyndata, tempdf])
    #for performance
    if i%100 == 0:
        dyndata = dyndata.apply(lambda x: pd.Series(x.dropna().values))
        print((i/len(data))*100,'%')

dyndata = dyndata.apply(lambda x: pd.Series(x.dropna().values))

#linear fit
start = 9
length = int(dyndata.shape[1]/2)
coef=[]
for i in range(length):
    end = dyndata.iloc[0:, i].last_valid_index()
    coef.append(np.polyfit(dyndata.iloc[start:end, i+length].values.tolist(), dyndata.iloc[start:end, i].values.tolist(), 1))

for i in range(len(coef)):
    coef[i][0] = coef[i][0] * 3600

result=pd.DataFrame(coef, index=dyndata.columns.values[:length], columns=['slope', 'y axis intercept'])

#
print(result)

#plot
#for i in range(6):
#    plt.figure(str(dyndata.columns.values[i]))
#    plt.plot(dyndata.iloc[:, i+length].values.tolist(), dyndata.iloc[:, i].values.tolist())

result.to_excel("dyn_result.xlsx")
