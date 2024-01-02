# This is a sample Python script.
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def srf_function(wave, band, srf):
    max = np.max(wave)
    min = np.min(wave)
    ind1 = 400 - min
    v1 = []
    for SRF_ind in srf.keys():
        t1 = 0
        t2 = 0
        k = 0
        for i in range(ind1, len(band)):
            t2 = t2 + srf[SRF_ind][i - ind1]
            t1 = t1 + band[i] * srf[SRF_ind][i - ind1]
        v1.append(round(t1 / t2, 6))
    return v1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder_path = './Data/Dataset'
    list1 = os.listdir(folder_path)
    # 获取文件
    file_list = [file for file in list1 if os.path.isfile(os.path.join(folder_path, file))]

    satellite = 'HJ2A'
    sensor = 'CCD1'
    srf_path = './SRF/' + satellite + '.json'
    with open(srf_path, "r") as json_file:
        srf_data = json.load(json_file)
        srf = srf_data[sensor]['SRF']
    for files in tqdm(file_list):  # 数据集列表筛选
        name = files.split('_')[0]
        lake_name = name.split(' ')[1]
        time = files.split('_')[1]
        data_name = lake_name + '_' + time
        # print('\n', files)
        df = pd.read_excel(folder_path + '/' + files, sheet_name='Water spectral dataset')
        column_labels = df.columns.tolist()
        index_labels = df.index.tolist()
        wave = column_labels[4:]
        ind = [int(i) for i in wave]
        spectrum = []
        for i in index_labels:
            band = [df.loc[i, j] for j in wave]
            temp = srf_function(ind, band, srf)
            spectrum.append(temp)
        col = [satellite + '_' + str(j + 1) for j in range(len(temp))]
        data = pd.DataFrame(spectrum, columns=col, index=index_labels)
        data_dir = './Data/HJ_simulation.xlsx'
        with pd.ExcelWriter(data_dir, engine='openpyxl', mode='a') as writer:
            # 写入DataFrame到新的工作表
            data.to_excel(writer, sheet_name=data_name, index=False)

        print(data_dir)
    # writer.save()
