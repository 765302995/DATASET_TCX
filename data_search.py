import os

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    folder_path = './Data/Dataset'
    list1 = os.listdir(folder_path)
    # 获取文件
    file_list = [file for file in list1 if os.path.isfile(os.path.join(folder_path, file))]
    data = pd.DataFrame()
    for files in tqdm(file_list):  # 数据集列表筛选
        df1 = pd.read_excel(folder_path + '/' + files, sheet_name='Water quality dataset')
        # Index  Date  Longitude  Latitude  Chl-a(μg/L)  TSM(mg/L)  SDD(cm) Temp(°C)
        data_1 = df1[['Chl-a(μg/L)', 'TSM(mg/L)', 'SDD(cm)', 'Temp(°C)']]
        data = pd.concat([data, data_1], axis=0)
        # print(data_1)
    data = data[data['Chl-a(μg/L)'] < 200]
    data = data[data['TSM(mg/L)'] > 1]
    # data = data.drop([1, 97, 115, 122, 123, 175, 177, 179])
    print(data['Chl-a(μg/L)'].max(), data['Chl-a(μg/L)'].min(), data['Chl-a(μg/L)'].mean(), data['Chl-a(μg/L)'].std())
    print(data['TSM(mg/L)'].max(), data['TSM(mg/L)'].min(), data['TSM(mg/L)'].mean(), data['TSM(mg/L)'].std())
    print(data['SDD(cm)'].max(), data['SDD(cm)'].min(), data['SDD(cm)'].mean(), data['SDD(cm)'].std())
    print(data['Temp(°C)'].max(), data['Temp(°C)'].min(), data['Temp(°C)'].mean(), data['Temp(°C)'].std())






