import os
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tqdm import tqdm


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data_dir = './Data/train_data.xlsx'
    test_data_dir = './Data/test_data.xlsx'
    train_data = pd.read_excel(train_data_dir)
    test_data = pd.read_excel(test_data_dir)
    train_data = train_data[(train_data['SDD(cm)'] > 10)]
    test_data = test_data[(test_data['SDD(cm)'] > 10)]
    col = ['HJ2A_1', 'HJ2A_2', 'HJ2A_3', 'HJ2A_4', 'HJ2A_5', 'SDD(cm)']
    x_train = train_data.loc[:, col[:5]]
    y_train = train_data.loc[:, col[5]]
    x_test = test_data.loc[:, col[:5]]
    y_test = test_data.loc[:, col[5]]

    # Random_Forest_Regressor
    rf_regressor = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=3)
    rf_regressor.fit(x_train, y_train)

    # Training dataset prediction
    y_pred = rf_regressor.predict(x_train)

    # Draw a scatter plot
    max_value = np.max([y_train.max(), y_pred.max()])
    min_value = np.min([y_train.min(), y_pred.min()])
    x_ = np.arange(min_value, max_value, 10)

    coef = np.polyfit(y_train.values, y_pred, 1)
    a = float(coef[0])
    b = float(coef[1])
    y_ = a * x_ + b
    text2 = 'y=' + str(np.round(a, 2)) + 'x + ' + str(np.round(b, 2))
    xy = np.vstack([y_train.values, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = y_train.values[idx], y_pred[idx], z[idx]

    plt.figure(figsize=(4.5, 4.5))
    left = 0.15
    bottom = 0.15
    width = 0.8
    height = 0.8
    plt.subplots_adjust(left=left, bottom=bottom, right=left + width, top=bottom + height)
    R2 = np.around(r2_score(y_train, y_pred), 3)
    MAPE = np.around(mean_absolute_percentage_error(y_train, y_pred), 3)
    RMSE = np.around(np.sqrt(mean_squared_error(y_train, y_pred)), 3)
    print('R2 = ', np.around(r2_score(y_train, y_pred), 3))
    print('MAPE = ', np.around(mean_absolute_percentage_error(y_train, y_pred), 3))
    print('MAE = ', np.around(mean_absolute_error(y_train, y_pred), 3))
    print('RMSE = ', np.around(np.sqrt(mean_squared_error(y_train, y_pred)), 3))
    text3 = text2 + '\n' + '  R2  = ' + str(np.round(R2, 3)) + '\n' + 'MAPE = ' + str(np.round(MAPE, 3)) + '\n' \
            + 'RMSE = ' + str(np.round(RMSE, 3))
    plt.plot([0, max_value], [0, max_value], 'k', linewidth=1.5, label='1:1')
    # plt.plot(y_train, y_pred, 'o', color='dodgerblue', alpha=0.7, markersize=6, label='Train dataset')
    plt.scatter(x, y, c=z, s=30, cmap='jet', label='Train dataset')
    plt.plot(x_, y_, 'r', linewidth=2)

    fontdict = {'family': 'Times New Roman', 'size': 12, 'style': 'normal'}
    plt.text(max_value * 0.7, max_value * 0.04, text3,
             fontdict=fontdict)
    plt.xlabel('Measured SDD(cm)', fontdict=fontdict)
    plt.ylabel('Derived SDD(cm)', fontdict=fontdict)
    plt.xticks(fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.legend(loc='best', prop=fontdict, frameon=False)
    plt.axis([0, max_value, 0, max_value])
    plt.savefig('./picture/sdd_train_data.png', dpi=300)
    plt.show()

    # testing dataset prediction
    y_hat = rf_regressor.predict(x_test)

    # Draw a scatter plot
    plt.figure(figsize=(4.5, 4.5))
    left = 0.15
    bottom = 0.15
    width = 0.8
    height = 0.8
    plt.subplots_adjust(left=left, bottom=bottom, right=left + width, top=bottom + height)
    R2 = np.around(r2_score(y_test, y_hat), 3)
    MAPE = np.around(mean_absolute_percentage_error(y_test, y_hat), 3)
    RMSE = np.around(np.sqrt(mean_squared_error(y_test, y_hat)), 3)
    print('R2 = ', np.around(r2_score(y_test, y_hat), 3))
    print('MAE = ', np.around(mean_absolute_error(y_test, y_hat), 3))
    print('MAPE = ', np.around(mean_absolute_percentage_error(y_test, y_hat), 3))
    print('RMSE = ', np.around(np.sqrt(mean_squared_error(y_test, y_hat)), 3))
    max_value = np.max([y_test.max(), y_hat.max()])
    min_value = np.min([y_test.min(), y_hat.min()])
    x_ = np.arange(min_value, max_value, 10)

    coef = np.polyfit(y_test.values, y_hat, 1)
    a = float(coef[0])
    b = float(coef[1])
    y_ = a * x_ + b

    xy = np.vstack([y_test.values, y_hat])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = y_test.values[idx], y_hat[idx], z[idx]

    text2 = 'y=' + str(np.round(a, 2)) + 'x + ' + str(np.round(b, 2))
    text3 = text2 + '\n' + '  R2  = ' + str(np.round(R2, 3)) + '\n' + 'MAPE = ' + str(np.round(MAPE, 3)) + '\n' \
            + 'RMSE = ' + str(np.round(RMSE, 3))

    plt.plot([0, max_value], [0, max_value], 'k', linewidth=1.5, label='1:1')
    plt.scatter(x, y, c=z, s=30, cmap='jet', label='Test dataset')
    plt.plot(x_, y_, 'r', linewidth=2)

    # 设置图例
    fontdict = {'family': 'Times New Roman', 'size': 12, 'style': 'normal'}
    plt.text(max_value * 0.7, max_value * 0.04, text3,
             fontdict=fontdict)
    plt.xlabel('Measured SDD(cm)', fontdict=fontdict)
    plt.ylabel('Derived SDD(cm)', fontdict=fontdict)
    plt.xticks(fontname='Times New Roman', fontsize=10)
    plt.yticks(fontname='Times New Roman', fontsize=10)
    plt.legend(loc='best', prop=fontdict, frameon=False)
    plt.axis([0, max_value, 0, max_value])
    plt.savefig('./picture/sdd_test_data.png', dpi=300)
    plt.show()
