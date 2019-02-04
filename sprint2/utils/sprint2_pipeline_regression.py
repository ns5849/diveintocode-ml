from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import math


param_default =  {
            "split_test_size": 0.25,
            "split_random_state": 42,
            "normalization_on": True,
            "roc_plot_off": False,
            "cv_on": False,
            "cv_split": 5,
            "cv_random_state" : None,
            "cv_shuffle" : True,
            "grid_search_on": False,
            "grid_search_samples": "ALL",
        }


def plot_result(result, test_target, test_feature, feature_value, target_value):

    num_of_row = math.ceil(len(feature_value) / 2)
    num_of_col = 2 if len(feature_value) > 1 else 1

    fig = plt.figure(figsize=(15, (5 * num_of_row)))
    #fig = plt.figure()
    ax = [0 for _ in range(0, len(feature_value))]
    for i in range(0, len(feature_value)):
        row = math.floor(i / 2)
        col = i % 2
        ax[i] = plt.subplot2grid((num_of_row, num_of_col), (row, col))
        ax[i].set_title('{} vs {}'.format(feature_value[i], target_value))
        ax[i].set_xlabel('{}'.format(feature_value[i]))
        ax[i].set_ylabel('{}'.format(target_value))
        ax[i].grid(True)
        ax[i].scatter(test_feature[:, i], result, s=50, color="Red", marker='*', label="predicted")
        ax[i].scatter(test_feature[:, i], test_target.values, s=30, color="Blue", marker='+', label="train data")
        ax[i].legend(fontsize=15)

    plt.show()


def train_model(model, train_feature, train_target, test_feature, test_target):
    model.fit(train_feature, train_target)
    result = model.predict(test_feature)
    mse = mean_squared_error(test_target, result)

    return result, mse


def pipeline_regression(model, X, Y, target_value, feature_value, params=param_default, params_grid={}):
    """
    Parameter
    ---------------
    model : 利用するライブラリのオブジェクト
    X : data(feature)
    Y : data(target)
    target_value :目的変数名
    feature_value :　説明変数名
    pos_label : 目的変数 Positive label指定
    Params :  この関数のパラメーター達
    params_grid : GridSearchのパラメータ達

    Return
    ---------------
    fpr, tpr, thresholds


    メモ：予測結果の出力が目的なので、Cross validationは削除した。
    """

    # 1.Train , test dataに分割
    print('Split data train & test')
    train_feature, test_feature, train_target, test_target = train_test_split(X, Y,
                                                                        test_size=params.get('split_test_size'),
                                                                        random_state=params.get('split_random_state'))
    print('元データ数：{}　学習データ数：{}　検証データ数：{}'
          .format(len(Y), len(train_target), len(test_target)))

    # 2-1.前処理
    if params.get('normalization_on') == True:
        print('Normalize feature data')
        scaler = StandardScaler()
        scaler.fit(train_feature)
        train_feature = scaler.transform(train_feature)
        scaler.fit(test_feature)
        test_feature = scaler.transform(test_feature)

    # 2-2.GridSearch
    if params.get('grid_search_on') == True:
        data_feature = pd.DataFrame(data=train_feature, columns=[feature_value])
        data_target = pd.DataFrame(data=train_target, columns=[target_value])
        data_feature.reset_index(drop=True, inplace=True)
        data_target.reset_index(drop=True, inplace=True)
        data = pd.concat([data_feature, data_target], axis=1)
        if params.get('grid_search_samples') == 'All':
            num_of_grid_search_sample = len(data.index)
        else:
            num_of_grid_search_sample = params.get('grid_search_samples')
        tmp = data.sample(n=num_of_grid_search_sample)
        print('Run GridSearch with {} samples'.format(num_of_grid_search_sample))

        grid_search = GridSearchCV(
            model,
            param_grid=params_grid.get('hyper_param'),
            cv=params_grid.get('grid_search_param', {}).get('grid_search_cv'))

        grid_search.fit(tmp[feature_value].values, tmp[target_value].values)
        model.set_params(**grid_search.best_params_)
        print('Set best params ', grid_search.best_params_)

    # 3.学習
    result, mse = train_model(model, train_feature, train_target, test_feature, test_target)
    print("mean square error={}".format(np.sqrt(mse)))

    # 4.Plot
    if params.get('plot_on') == True:
        plot_result(result, test_target, test_feature, params.get('plot_feature_value'), target_value)

    return result, test_target
