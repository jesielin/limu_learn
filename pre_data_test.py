import os
import pandas as pd
import torch

# os.makedirs(os.path.join('.','data'),exist_ok=True)
data_file = os.path.join('.','data','house_tiny.csv')
#
# with open(data_file,'w',encoding='utf-8') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')


data = pd.read_csv(data_file)
print(data)

for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:  # 检查是否为数值类型
        data[col] = data[col].fillna(data[col].mean())
    # elif data[col].dtype == 'object':  # 对象类型，可能是字符串或类别
        # data[col] = data[col].fillna(data[col].mode().iloc[0])
    # 其他类型的列可以按需添加填充规则

print(data[data.columns[1]])

inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
# print(inputs,outputs)
inputs = pd.get_dummies(inputs,dummy_na=True).astype(int)
print(inputs)

inputs,outputs = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(inputs,outputs)
