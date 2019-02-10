import numpy as np
import matplotlib.pyplot as plt


class ScratchClfEvaluation:
    #分類結果の評価クラス
    #AND, OR, NAND, XORを組み合わせてTP, TN, FP, FNを求め各指標値を計算
    #入力データは0, 1のみ有効（関数を呼び出す側で規格化する必要がある）

    def __init__():
        pass

    def _AND_logic(self, x, y):
        tmp = x * 0.5 + y * 0.5 - 0.7
        return (tmp >= 0).astype(int)

    def _OR_logic(self, x, y):
        tmp = x * 0.5 + y * 0.5 - 0.2
        return (tmp >= 0).astype(int)

    def _NAND_logic(self, x, y):
        tmp = x * (-0.5) + y * (-0.5) + 0.7
        return (tmp >= 0).astype(int)

    def _XOR_logic(self, x, y):
        tmp1 = self._NAND_logic(x, y)
        tmp2 = self._OR_logic(x, y)
        return self._AND_logic(tmp1, tmp2)

    def _cal_tp(self, ans, pred, pos_label=1):
        if pos_label == 1:
            x = ans
            y = pred
        elif pos_label == 0:
            x = self._NAND_logic(ans, ans)
            y = self._NAND_logic(pred, pred)

        tmp = self._AND_logic(x, y)
        return np.sum(tmp == 1)

    def _cal_fp(self, ans, pred, pos_label=1):
        if pos_label == 1:
            x = ans
            y = pred
        elif pos_label == 0:
            x = self._NAND_logic(ans, ans)
            y = self._NAND_logic(pred, pred)

        tmp1 = self._XOR_logic(x, y)
        tmp = self._AND_logic(tmp1, y)
        return np.sum(tmp == 1)

    def _cal_tn(self, ans, pred, pos_label=1):
        if pos_label == 0:
            x = ans
            y = pred
        elif pos_label == 1:
            x = self._NAND_logic(ans, ans)
            y = self._NAND_logic(pred, pred)

        tmp = self._AND_logic(x, y)
        return np.sum(tmp == 1)

    def _cal_fn(self, ans, pred, pos_label=1):
        if pos_label == 1:
            x = ans
            y = pred
        elif pos_label == 0:
            x = self._NAND_logic(ans, ans)
            y = self._NAND_logic(pred, pred)

        tmp1 = self._XOR_logic(x, y)
        tmp2 = self._AND_logic(tmp1, x)
        return np.sum(tmp2 == 1)

    def eval_tpr(self, ans, pred, pos_label=1):
        sum_of_pos = np.sum(ans == int(pos_label))
        tpr = self._cal_tp(ans, pred, pos_label) / sum_of_pos
        return tpr

    def eval_fpr(self, ans, pred, pos_label=1):
        sum_of_neg = np.sum(ans != int(pos_label))
        fpr = self._cal_fp(ans, pred, pos_label) / sum_of_neg
        return fpr

    def eval_accuracy(self, ans, pred, pos_label=1):
        numer = self._cal_tp(ans, pred) + self._cal_tn(ans, pred)
        denom = self._cal_tp(ans, pred) + self._cal_tn(ans, pred) + self._cal_fp(ans, pred) + self._cal_fn(ans, pred)

        return numer / denom

    def eval_precision(self, ans, pred, pos_label=1):
        numer = self._cal_tp(ans, pred)
        denom = self._cal_tp(ans, pred) + self._cal_fp(ans, pred)

        return numer / denom

    def eval_recall(self, ans, pred, pos_label=1):
        numer = self._cal_tp(ans, pred)
        denom = self._cal_tp(ans, pred) + self._cal_fn(ans, pred)

        return numer / denom

    def eval_f1(self, ans, pred, pos_label=1):
        numer = 2 * self.eval_precision(ans, pred) * self.eval_recall(ans, pred)
        denom = self.eval_precision(ans, pred) + self.eval_recall(ans, pred)

        return numer / denom

    def view_result(self, ans, pred, pos_label=1):
        print("tp={} tn={} fp={} fn={}".format(self._cal_tp(ans, pred),
                                               self._cal_tn(ans, pred),
                                               self._cal_fp(ans, pred),
                                               self._cal_fn(ans, pred)))


class ScratchLogisticRegression(ScratchClfEvaluation):
    """
    ロジスティック回帰のスクラッチ実装

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
    C : float
      正則化項パラメータ

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """

    def __init__(self, num_iter, lr, bias, verbose, C=0):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.C = C
        self.verbose = verbose
        # 損失を記録する配列を用意
        self.accuracy = 0
        self.theta = 0
        self.theta_cal_log = 0
        self.cross_entropy = np.zeros(self.iter)
        self.cross_entropy_val = np.zeros(self.iter)
        self.label0_val = 0
        self.label1_val = 0

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
        train_target = y.reshape(1, len(y))
        test_feature = X_val.T
        test_target = y_val.reshape(1, len(y_val))

        # 1-2. Yを0 or 1に規格化
        train_target_label1_val = max(y)
        train_target_label0_val = min(y)
        test_target_label1_val = max(y_val)
        test_target_label0_val = min(y_val)
        train_target = (train_target - train_target_label0_val) / (train_target_label1_val - train_target_label0_val)
        test_target = (test_target - test_target_label0_val) / (test_target_label1_val - test_target_label0_val)
        self.label0_val = train_target_label0_val
        self.label1_val = train_target_label1_val

        # 1-3. バイアスを追加 if self.bias = True
        # train_feature, test_featureの特徴量方向(index方向)にバイアス成分を追加 X = 1,1,1...,1
        if self.bias == True:
            bias = np.array([1 for _ in range(train_feature.shape[1])])
            train_feature = np.vstack((bias, train_feature))
            bias = np.array([1 for _ in range(test_feature.shape[1])])
            test_feature = np.vstack((bias, test_feature))

        # 1-4. 準備 適当な値でθを定義(random)、特徴量分用意
        THETA_INIT_MIN = 1
        THETA_INIT_MAX = 10
        self.theta = np.random.randint(THETA_INIT_MIN, THETA_INIT_MAX, train_feature.shape[0]) / THETA_INIT_MAX
        self.theta = np.reshape(self.theta, (len(self.theta), 1))
        self.theta_cal_log = np.zeros((self.iter, (len(self.theta))))
        print("Initial theta:\n{}".format(self.theta))

        # 2. 最急降下法(Loopをiter回だけ回す)　
        for i in range(0, self.iter):
            self.theta = self._gradient_descent(train_feature, train_target)
            self.cross_entropy[i] = self._compute_cost(train_feature, train_target)
            self.cross_entropy_val[i] = self._compute_cost(test_feature, test_target)
            self.theta_cal_log[i] = self.theta.T.reshape(-1)

        print("Theta:\n", self.theta)
        print("Feature:\n", train_feature)
        print("Target:\n", self._standval_to_actval(train_target))
        print("Result:\n", self._test_predict(train_feature))

        return

    def predict_proba(self, X):
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
        return self._logistic_hypothesis(X)

    def predict(self, X, threshold=0.5):
        """
        ロジスティック回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """

        # 入力値を整形
        X = X.T
        if self.bias == True:
            bias = np.array([1 for _ in range(X.shape[1])])
            X = np.vstack((bias, X))

        # _linear_hypothesis(self, X)を使って計算
        prediction = self._logistic_hypothesis(X)

        # 元のラベルに戻す
        prediction = (prediction >= threshold).astype(int)
        prediction = self._standval_to_actval(prediction)

        return prediction.reshape(-1)

    def save_param(self, file_path):
        #指定されたファイル名で重みθを保存する
        np.savez(file_path, theta=self.theta)

        return

    def recall_param(self, file_path):
        # 指定されたnpzファイルから読み込みこんだ重みθを設定する。
        param = np.load(file_path)
        self.theta = param['theta']
        print("recall param=\n", param['theta'])

        return

    def cal_accuracy(self, y_pred, y):
        return self.eval_accuracy(self._actval_to_standval(y_pred), self._actval_to_standval(y))

    def cal_precision(self, y_pred, y):
        return self.eval_precision(self._actval_to_standval(y_pred), self._actval_to_standval(y))

    def cal_recall(self, y_pred, y):
        return self.eval_recall(self._actval_to_standval(y_pred), self._actval_to_standval(y))

    def cal_f1(self, y_pred, y):
        return self.eval_f1(self._actval_to_standval(y_pred), self._actval_to_standval(y))

    def view_summary(self, y_pred, y):
        return self.view_result(self._actval_to_standval(y_pred), self._actval_to_standval(y))

    def cal_cross_entropy(self, y_pred, y):
        #クロスエントロピーを計算する
        pos_proba = -1 * y * np.log(y_pred)
        neg_proba = -1 * (1 - y) * np.log((1 - y_pred))
        cross_entropy = (pos_proba.sum(axis=1) + neg_proba.sum(axis=1)) / y.shape[1]

        return cross_entropy

    def _test_predict_proba(self, X):
        """
        ロジスティック回帰を確率を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """

        # _linear_hypothesis(self, X)を使って計算
        return self._logistic_hypothesis(X)

    def _test_predict(self, X, threshold=0.5):
        """
        ロジスティック回帰を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """

        # _logistic_hypothesis(self, X)を使って計算
        prediction = self._logistic_hypothesis(X)

        # 確率から元のラベルへ変換する
        prediction = (prediction >= threshold).astype(int)
        prediction = self._standval_to_actval(prediction)

        return prediction.reshape(-1)

    def _logistic_hypothesis(self, X):
        """
        ロジスティック回帰の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
          次の形のndarray, shape (n_samples, 1)
          仮定関数による推定結果

        """
        sigmoid = 1 / (1 + np.exp(-1 * np.dot(self.theta.T, X)))

        return sigmoid

    def _compute_cost(self, X, y):
        """
        クロスエントロピーを計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
          次の形のndarray, shape (1,)
          クロスエントロピーの計算結果
        """

        # y_pred = self.predict(X)
        y_pred = self._logistic_hypothesis(X)
        return self.cal_cross_entropy(y_pred, y)

    def _gradient_descent(self, X, y):
        """
        説明を記述
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        y : 次の形のndarray, shape (n_samples, 1)
          正解値

        Returns
        -------
        """
        delta = self._logistic_hypothesis(X) - y
        theta_step = (self.lr * np.dot(X, delta.T) / X.shape[1])
        if self.bias == True:
            theta_step[1:theta_step.shape[0]] += (self.C / X.shape[1]) * self.theta[1:theta_step.shape[0]]
        else:
            theta_step += (self.C / X.shape[1]) * self.theta

        return self.theta - theta_step

    def _standval_to_actval(self, x):
        #0,1へ規格化されたラベルを元データのラベルに戻す
        return x * (self.label1_val - self.label0_val) + self.label0_val

    def _actval_to_standval(self, x):
        # データラベルを0,1へ規格化する
        return (x - self.label0_val) / (self.label1_val - self.label0_val)

    def plot_cross_entropy(self):
        #学習毎のクロスエントロピーの計算結果をプロットする
        plt.title("iter vs loss")
        plt.xlabel("iter")
        plt.ylabel("cross entropy")
        plt.plot(self.cross_entropy, "r", label="cross entropy")
        plt.plot(self.cross_entropy_val, "b", label="val cross entropy")
        plt.legend()
        # plt.yscale("Log")
        plt.show()

    def plot_theta_cal_log(self, num_of_feature):
        #学習毎の重みθの計算結果をプロットする
        plt.title("theta vs loss")
        plt.xlabel("theta (Feature={})".format(num_of_feature))
        plt.ylabel("cross entropy")
        plt.scatter(self.theta_cal_log[:, num_of_feature], self.cross_entropy, s=50, marker='*', color='r')
        plt.legend()
        plt.show()
