from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
import pandas as pd


param_default =  {
            "split_test_size" : 0.25,
            "split_random_state" : 42,
            "normalization_on": True,
            "roc_plot_off": False,
            "cv_on": False,
            "cv_split": 5,
            "cv_random_state" : None,
            "cv_shuffle" : True,
            "grid_search_on" : False,
            "grid_search_samples" : "ALL",
            "result_format" : "predict_as_binary"
        }


def plot_roc(fpr, tpr):
    plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.show()


def train_model(model, train_feature, train_target, test_feature, test_target, positive_label, result_format):
    model.fit(train_feature, train_target)

    if result_format == 'predict_as_probability':
        print('Result Format = proba')
        result = model.predict_proba(test_feature)
        # rocを計算
        fpr, tpr, thresholds = metrics.roc_curve(test_target, result[:, 1], pos_label=positive_label)
        auc = metrics.auc(fpr, tpr)
        print("AUC={:.5f}".format(auc))
    elif result_format == 'predict_as_binary':
        print('Result Format = binary')
        result = model.predict(test_feature)
        fpr = 0
        tpr = 0
        auc = 0
    else:
        print('Result Format = ', result_format)
        fpr = 0
        tpr = 0
        auc = 0
        result = 0

    return result, fpr, tpr, auc


def pipeline_classifier(model, X, Y, target_value, feature_value, pos_label=1, params=param_default, params_grid={}):
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

    # 学習データ、テストデータに分ける
    print('Split data train & test')
    train_feature, test_feature, train_target, test_target = train_test_split(X, Y,
                                                                        test_size=params.get('split_test_size'),
                                                                        random_state=params.get('split_random_state'))
    print('元データ数：{}　学習データ数：{}　検証データ数：{}'
          .format(len(Y), len(train_target), len(test_target)))
    #train_feature = train_data[feature_value].values
    #train_target = train_data[target_value].values
    #test_feature = test_data[feature_value].values
    #test_target = test_data[target_value].values

    if params.get('normalization_on') == True:
        print('Normalize feature data')
        scaler = StandardScaler()
        scaler.fit(train_feature)
        train_feature = scaler.transform(train_feature)
        scaler.fit(test_feature)
        test_feature = scaler.transform(test_feature)

    if params.get('grid_search_on') == True:
        # 時間がかかるのでtrain dataを絞る (サンプリングのやり方はあとで再度考える)
        data_feature = pd.DataFrame(data=train_feature, columns=feature_value)
        data_target = pd.DataFrame(data=train_target, columns=[target_value])
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

    # 学習と予測
    result, fpr, tpr, auc = train_model(
        model,
        train_feature, train_target,
        test_feature, test_target,
        pos_label,
        params.get('result_format')
    )

    # rocをplot
    if params.get('roc_plot_on') == True:
        plot_roc(fpr, tpr)

    return result, test_target, fpr, tpr, auc
