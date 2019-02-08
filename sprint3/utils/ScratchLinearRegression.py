import numpy as np
import matplotlib.pyplot as plt


class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, bias, verbose):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.loss_theta = 0
        self.theta = 0

    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """

        # 1-1. データを shape(n_feature, n_samples)へ整形, θをshape(n_feature, 1)へ整形　h=dot( (θ)t, X)とするため。
        train_feature = X.T
        train_target = y.T
        test_feature = X_val.T
        test_target = y_val.T

        # 1-2. バイアスを追加 if self.bias = True
        # train_feature, test_featureの特徴量方向(index方向)にバイアス成分を追加 X = 1,1,1...,1
        if self.bias == True:
            bias = np.array([1 for _ in range(train_feature.shape[1])])
            train_feature = np.vstack((bias, train_feature))
            bias = np.array([1 for _ in range(test_feature.shape[1])])
            test_feature = np.vstack((bias, test_feature))

        # 1-3. 準備 適当な値でθを定義(random)、特徴量分用意
        THETA_INIT_MIN = 1
        THETA_INIT_MAX = 10
        self.theta = np.random.randint(THETA_INIT_MIN, THETA_INIT_MAX, train_feature.shape[0])
        # self.theta = np.array([0.1 for _ in range(train_feature.shape[0])])
        self.theta = np.reshape(self.theta, (len(self.theta), 1))
        self.loss_theta = np.zeros((self.iter, (len(self.theta))))
        print("Initial theta:\n{}".format(self.theta))

        # 2. 最急降下法(Loopをiter回だけ回す)　
        for i in range(0, self.iter):
            self.theta = self._gradient_descent(train_feature, train_target)

            if self.verbose:
                # verboseをTrueにした際は学習過程を出力
                self.loss[i] = self._compute_cost(train_feature, train_target)
                self.loss_theta[i] = self.theta.T.reshape(-1)

                if len(test_target) != 0:
                    self.val_loss[i] = self._compute_cost(test_feature, test_target)
                    # Accuracy

        tmp = self._test_predict(train_feature)
        print("Theta:\n{}".format(self.theta))
        print("Feature:\n{}".format(train_feature))
        print("Target:\n{}".format(train_target))
        print("Result:\n{}".format(tmp))

        return

    def predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        # 入力値を整形
        X = X.T
        if self.bias == True:
            bias = np.array([1 for _ in range(X.shape[1])])
            X = np.vstack((bias, X))

        # _linear_hypothesis(self, X)を使って計算
        return self._linear_hypothesis(X)

    def MSE(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        mse : numpy.float
          平均二乗誤差
        """

        delta = (y_pred - y) ** 2
        mse = np.sum(delta, axis=1) / delta.shape[0]

        return mse

    def _test_predict(self, X):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        # _linear_hypothesis(self, X)を使って計算
        return self._linear_hypothesis(X)

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          線形の仮定関数による推定結果

        """

        return np.dot(self.theta.T, X)

    def _compute_cost(self, X, y):
        """
        平均二乗誤差を計算する。MSEは共通の関数を作っておき呼び出す

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
          次の形のndarray, shape (1,)
          平均二乗誤差
        """

        #y_pred = self.predict(X)
        y_pred = self._linear_hypothesis(X)
        return self.MSE(y_pred, y)

    def _gradient_descent(self, X, y):
        """
        説明を記述
        """
        delta = self._linear_hypothesis(X) - y
        return self.theta - (self.lr * np.dot(X, delta.T) / X.shape[1])

    def plot_loss(self):
        plt.title("iter vs loss")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.plot(self.loss, "r", label="loss")
        plt.plot(self.val_loss, "b", label="val_loss")
        plt.legend()
        plt.yscale("Log")
        plt.show()

    def plot_theta_loss(self, num_of_feature):
        plt.title("theta vs loss")
        plt.xlabel("theta (Feature={})".format(num_of_feature))
        plt.ylabel("loss")
        plt.scatter(self.loss_theta[:, num_of_feature], self.loss, s=50, marker='*', color='r')
        plt.legend()
        plt.show()
