# ライブラリ・モジュールを読み込む
import numpy as np


def train_test_split(x, y, test_size):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何分割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """

    data = np.concatenate([x,y], axis=1) # 上と同じ結果になります
    num_of_row = data.shape[0]

    np.random.shuffle(data)
    #print(data)
    train = data[0:(int)(num_of_row * test_size)]
    test = data[(int)(num_of_row * test_size):num_of_row]
    #train, test = np.vsplit(data, (int)(num_of_row * test_size))
    #print("train")
    #print(train)
    #print("test")
    #print(test)

    x_train = train[:,0:data.shape[1] - 1]
    y_train = train[:,-1]

    x_test = test[:,0:data.shape[1] - 1]
    y_test = test[:,-1]

    return x_train, x_test, y_train, y_test
