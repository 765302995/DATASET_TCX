import numpy as np
import pandas as pd
import os
from tqdm import tqdm


if __name__ == '__main__':
    folder_path = './Data/Dataset'
    list1 = os.listdir(folder_path)
    file_list = [file for file in list1 if os.path.isfile(os.path.join(folder_path, file))]
    data_count = pd.DataFrame(columns=['name', 'count'])
    i = 1
    for files in tqdm(file_list):

        df1 = pd.read_excel(folder_path + '\\' + files, sheet_name='Water quality dataset')
        df2 = pd.read_excel(folder_path + '\\' + files, sheet_name='Water spectral dataset')
        c1 = df1.shape[0]
        c2 = df2.shape[0]
        if c1 == c2:
            alist = [files, c1]
            data_count.loc[i] = alist
            i = i + 1
        else:
            print(files)

    count_dir = 'D:\\ScientificData\\Data\\data_count.xlsx'
    data_count.to_excel(count_dir, sheet_name='data_count', index=True)

