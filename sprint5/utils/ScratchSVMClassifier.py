import numpy as np
import matplotlib.pyplot as plt
import time


class ScratchClfEvaluation:
    #分類結果の評価クラス
    #AND, OR, NAND, XORを組み合わせてTP, TN, FP, FNを求め各指標値を計算
    #入力データは0, 1のみ有効（関数を呼び出す側で規格化する必要がある）

    def __init__():
        pass

    def _format(self, x):
        label1_val = np.max(x)
        label0_val = np.min(x)
        return (x - label0_val) / (label1_val - label0_val)

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
        ans = self._format(ans)
        pred = self._format(pred)
        sum_of_pos = np.sum(ans == int(pos_label))
        tpr = self._cal_tp(ans, pred, pos_label) / sum_of_pos
        return tpr

    def eval_fpr(self, ans, pred, pos_label=1):
        ans = self._format(ans)
        pred = self._format(pred)
        sum_of_neg = np.sum(ans != int(pos_label))
        fpr = self._cal_fp(ans, pred, pos_label) / sum_of_neg
        return fpr

    def eval_accuracy(self, ans, pred, pos_label=1):
        ans = self._format(ans)
        pred = self._format(pred)
        numer = self._cal_tp(ans, pred) + self._cal_tn(ans, pred)
        denom = self._cal_tp(ans, pred) + self._cal_tn(ans, pred) + self._cal_fp(ans, pred) + self._cal_fn(ans, pred)

        return numer / denom

    def eval_precision(self, ans, pred, pos_label=1):
        ans = self._format(ans)
        pred = self._format(pred)
        numer = self._cal_tp(ans, pred)
        denom = self._cal_tp(ans, pred) + self._cal_fp(ans, pred)

        return numer / denom

    def eval_recall(self, ans, pred, pos_label=1):
        ans = self._format(ans)
        pred = self._format(pred)
        numer = self._cal_tp(ans, pred)
        denom = self._cal_tp(ans, pred) + self._cal_fn(ans, pred)

        return numer / denom

    def eval_f1(self, ans, pred, pos_label=1):
        ans = self._format(ans)
        pred = self._format(pred)
        numer = 2 * self.eval_precision(ans, pred) * self.eval_recall(ans, pred)
        denom = self.eval_precision(ans, pred) + self.eval_recall(ans, pred)

        return numer / denom

    def view_result(self, ans, pred, pos_label=1):
        print("tp={} tn={} fp={} fn={}".format(self._cal_tp(ans, pred),
                                               self._cal_tn(ans, pred),
                                               self._cal_fp(ans, pred),
                                               self._cal_fn(ans, pred)))


class ScratchSVMClassifier(ScratchClfEvaluation):
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

    def __init__(self, num_iter, lr, bias=True, verbose=False, threshold=1e-5, hit_vector_cnt_threshold=5,
                 kernel='linear', gamma=1, theta0=0, pow_d=1):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.threshold = threshold
        self.bias = bias
        if hit_vector_cnt_threshold >= 2:
            self.hit_vector_cnt_threshold = hit_vector_cnt_threshold
        else:
            self.hit_vector_cnt_threshold = 2
        self.kernel = kernel
        self.verbose = verbose
        self.lam = 0
        self.sp_vector = 0
        self.num_of_feature = 0
        self.num_of_samples = 0
        self.label0_val = 0
        self.label1_val = 0
        self.gamma = gamma
        self.theta0 = theta0
        self.pow_d = pow_d
        self.start_time = [0 for _ in range(8)]

    def fit(self, arg_X, arg_y, arg_X_val=None, arg_y_val=None):
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
        X = np.copy(arg_X)
        y = np.copy(arg_y)

        X = X.T
        y = y.reshape(1, len(y))

        if arg_X_val is not None:
            X_val = np.copy(arg_X_val)
            X_val = X_val.T

        if arg_y_val is not None:
            y_val = np.copy(arg_y_val)
            y_val = y_val.reshape(1, len(y_val))

        # 1-2. バイアスを追加 if self.bias = True
        # train_feature, test_featureの特徴量方向(index方向)にバイアス成分を追加 X = 1,1,1...,1
        if self.bias == True:
            bias = np.array([1 for _ in range(X.shape[1])])
            X = np.vstack((bias, X))

            if arg_X_val is not None:
                bias = np.array([1 for _ in range(X_val.shape[1])])
                X_val = np.vstack((bias, X_val))

        self.num_of_feature = X.shape[0]
        self.num_of_samples = X.shape[1]

        # 1-3. Yを-1 or 1に規格化
        self.label1_val = np.max(y)
        self.label0_val = np.min(y)
        y = self._actval_to_standval(y)

        if arg_y_val is not None:
            y_val = self._actval_to_standval(y_val)

        # 1-4. 準備 サポートベクターの検出のためのFeature+Target行列を用意
        data_source = np.concatenate((X, y), axis=0)
        HIT_LABEL_CNT_THRESHOLD = (int)(self.hit_vector_cnt_threshold / 2)

        # 1-5. 準備 適当な値でλを定義(random)、サンプル数分準備
        LAMBDA_INIT_MIN = 1
        LAMBDA_INIT_MAX = 10
        LAMBDA_INIT_SCALE = 1e-07
        self.lam = np.random.randint(LAMBDA_INIT_MIN, LAMBDA_INIT_MAX, X.shape[1]) * LAMBDA_INIT_SCALE
        self.lam = np.reshape(self.lam, (1, len(self.lam)))
        self.lam_cal_log = np.zeros((1, (len(self.lam))))
        print("Initial lambda:\n{}".format(self.lam))

        # 2. 最急降下法(Loopをiter回だけ回す)　
        self._start_timer(1)
        is_fit = False
        for i in range(0, self.iter):
            self.lam = self._gradient_descent(X, y)
            if self.hit_vector_cnt_threshold <= np.sum(self.lam > self.threshold):
                # 学習データからサポートベクターを抜き出す
                # data_source = np.concatenate([data_source, self.lam], axis=0)
                selector = self.lam * np.ones((data_source.shape[0], 1))
                sp_vector = data_source[selector > self.threshold]
                sp_vector = sp_vector.reshape(data_source.shape[0], (int)(len(sp_vector) / data_source.shape[0]))
                # 抜き出したサポートベクターが+/-の両方含んでいる場合ループを抜けて計算を終了。サポートベクターをメンバ変数に保存
                label_p_cnt = np.sum([sp_vector[sp_vector.shape[0] - 1] == 1])
                label_n_cnt = np.sum([sp_vector[sp_vector.shape[0] - 1] == -1])

                if label_p_cnt >= HIT_LABEL_CNT_THRESHOLD & label_n_cnt >= HIT_LABEL_CNT_THRESHOLD:
                    print("Loop count={}".format(i))
                    print("Support vector=\n{}".format(sp_vector))
                    print("Labe count 1:[{}] -1:[{}]".format(label_p_cnt, label_n_cnt))
                    self.sp_vector = sp_vector
                    self.lam = self.lam[self.lam > self.threshold]
                    print("accuracy:", self.cal_accuracy(self._test_predict(X), y))

                    is_fit = True

                    break

        self._stop_timer(1)
        return is_fit

    def predict(self, arg_X):
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
        X = np.copy(arg_X)
        X = X.T

        # バイアスを追加 if self.bias = True
        # train_feature, test_featureの特徴量方向(index方向)にバイアス成分を追加 X = 1,1,1...,1
        if self.bias == True:
            bias = np.array([1 for _ in range(X.shape[1])])
            X = np.vstack((bias, X))

        x_sn = self.sp_vector[0:self.sp_vector.shape[0] - 1]
        x_sn = x_sn.reshape((self.num_of_feature, x_sn.shape[1]))
        y_sn = self.sp_vector[self.sp_vector.shape[0] - 1].reshape((1, x_sn.shape[1]))
        lam = np.reshape(self.lam, (1, len(self.lam)))
        tmp1 = self._svm_kernel_function(X, x_sn)
        # print("lam=",self.lam.shape)
        # print("y_sn=",y_sn.shape)
        # print("tmp1=",tmp1.shape)
        tmp2 = lam * y_sn * tmp1
        result = np.sum(tmp2, axis=1)
        result[result < 0] = -1
        result[result >= 0] = 1
        result = result.astype('int8').T

        return self._standval_to_actval(result)

    # def cal_confusion_matrix(self, y_pred, y):

    def cal_accuracy(self, y_pred, y):
        return self.eval_accuracy(y_pred, y)

    def cal_precision(self, y_pred, y):
        return self.eval_precision(y_pred, y)

    def cal_recall(self, y_pred, y):
        return self.eval_recall(y_pred, y)

    def cal_f1(self, y_pred, y):
        return self.eval_f1(y_pred, y)

    def _test_predict(self, X):
        """
        SVMを使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """

        x_sn = self.sp_vector[0:self.sp_vector.shape[0] - 1]
        x_sn = x_sn.reshape((self.num_of_feature, x_sn.shape[1]))
        y_sn = self.sp_vector[self.sp_vector.shape[0] - 1].reshape((1, x_sn.shape[1]))
        lam = np.reshape(self.lam, (1, len(self.lam)))
        tmp1 = self._svm_kernel_function(X, x_sn)
        tmp2 = lam * y_sn * tmp1
        result = np.sum(tmp2, axis=1)
        result[result < 0] = -1
        result[result >= 0] = 1

        return result.astype('int8').T

    def _svm_kernel_function(self, X1, X2):
        """
        SVM kernel関数
        dot(X1(転置), X2)を計算

        Parameters
        ----------
        X : 次の形のndarray, shape (n_features, n_samples)

        Returns
        -------
          次の形のndarray, shape (n_samples, n_samples)
        """
        #ans = np.dot(X1.T, X2)

        if self.kernel == 'linear':
            ans = np.dot(X1.T, X2)
        elif self.kernel == 'rbf':
            ans = self.gamma * (np.dot(X1.T, X2) + self.theta0)**self.pow_d
        else:
            ans = 0

        return ans

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

        tmp1 = y.T * y * self.lam * self._svm_kernel_function(X, X)
        delta = 1 - (np.sum(tmp1, axis=0))
        delta = delta.reshape(len(delta), 1)
        result = self.lam + self.lr * delta.T
        # 計算の都合で<0の結果は0に置き換える
        result[result < 0] = 0

        return result

    def _standval_to_actval(self, x):
        tmp = x
        tmp[tmp == 1] = self.label1_val
        tmp[tmp == -1] = self.label0_val

        return tmp

    def _actval_to_standval(self, x):
        tmp = x
        tmp[tmp == self.label0_val] = -1
        tmp[tmp == self.label1_val] = 1

        return tmp

    def plot_boundary(self, feature, target, index_of_x1, index_of_x2):
        x_sn = self.sp_vector[0:self.sp_vector.shape[0] - 1]
        x_sn = x_sn.reshape((self.num_of_feature, x_sn.shape[1]))
        y_sn = self.sp_vector[self.sp_vector.shape[0] - 1].reshape((1, x_sn.shape[1]))
        sp_theta = self.lam * y_sn * x_sn
        sp_theta = np.sum(sp_theta, axis=1)
        print("sp_theta=\n", sp_theta)

        if self.bias == True:
            b = sp_theta[0]
        else:
            b = 0

        y = -1 * (b + sp_theta[index_of_x1]) / sp_theta[index_of_x2] * feature[:, index_of_x1]
        plt.title("Boundary")
        plt.xlabel("Feature index={}".format(index_of_x1))
        plt.ylabel("Feature index={}".format(index_of_x2))
        plt.plot(feature[:, index_of_x1], y, label="Logistic boundary")
        plt.scatter(feature[:, index_of_x1], feature[:, index_of_x2], c=target)

        return

    def view_sp_vector(self):
        # print(self.sp_vector)
        return self.sp_vector

    def _start_timer(self, timer_index):
        self.start_time[timer_index] = time.time()

    def _stop_timer(self, timer_index):
        if self.start_time[timer_index] != 0:
            print( "Timer_index={}  Elapsed time={}".format(timer_index, time.time() - self.start_time[timer_index]))
        else:
            print("Timer isn't started")
