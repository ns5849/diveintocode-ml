import numpy as np

class ScratchKMeans():

    def __init__(self, k, max_loop=1000, settle_distance=0, settle_repeat=5, use_SSE=False):
        # ハイパーパラメータを属性として記録
        self.k = k  # クラスタの数
        self.iter = max_loop  # クラスタリングする際の最大loop数
        self.label_info = 0  # 各データのラベル　　row=1 col=Sample
        self.center_info = 0  # クラスタの受信座標 row=Feature Col=クラスタ(クラスタの数)
        self.distance_info = 0  # row=クラスタ col=Sample
        self.SSE_log = np.zeros(self.iter)
        self.silhouette_log = 0
        self.settle_distance = settle_distance #クラスタの収束条件。中心と重心の距離差がこの値以下になったら計算終了
        self.settle_repeat_count = settle_repeat
        self.use_SSE = use_SSE

    def _cal_distance(self, base_point, data):
        """
        Input
        base_point : ndarray shape(Feature, 基準点)　基準点の座標
        data : ndarray shape(Feature, Sample)

        Return
        distance : ndarray shape(基準点, Sample)
        """

        distance = np.zeros((base_point.shape[1], data.shape[1]))
        for i in range(base_point.shape[1]):
            diff_vector = data - base_point[:, i].reshape((base_point.shape[0], 1))
            distance[i] = np.linalg.norm(diff_vector, axis=0)

        return distance

    def _cal_center(self, data):
        """
        Input
        data : ndarray shape(Feature, Sample)

        Return
        center : shape(Feature, 1)　重心の座標
        """

        num_of_sample = data.shape[1]
        center_point = np.sum(data, axis=1) / num_of_sample

        return center_point

    def fit(self, feature_data):
        """
        Input
        feature_data : ndarray shape(Sample, Feature)

        Return
        result : ndarray shape(1, Sample) 各サンプルのラベル
        """

        # 1.データ整形 shape(Feature, Sample)へ整形
        feature_data = np.copy(feature_data)
        feature_data = feature_data.T

        # 2.インスタンス変数(table)を初期化
        self.label_info = np.zeros((1, feature_data.shape[1]))  # row=クラスタ col=Sample
        self.center_info = np.zeros((feature_data.shape[0], self.k))  # クラスタの受信座標 row=Feature Col=クラスタ(クラスタの数)
        self.distance_info = np.zeros((self.k, feature_data.shape[1]))  # row=クラスタ col=Sample　
        self.silhouette_log = np.zeros((self.iter, feature_data.shape[1]))

        # 3.ランダムにk個の点を選択
        np.random.seed(0)
        for i in range(self.k):
            index = np.random.randint(feature_data.shape[1])
            self.center_info[:, i] = feature_data[:, index]

        # 4.Loop
        MAX_RETRY = 3
        settle_count = 0
        for i in range(self.iter):
            # 1.重心からの距離を計算　（if distance_info.any(0) => 重心位置をもっとも離れているデータ点に移動して再度距離計算）
            self.distance_info = self._cal_distance(self.center_info, feature_data)

            # 2.ラベルを降る
            prev_label_info = np.copy(self.label_info)
            self.label_info = np.argmin(self.distance_info, axis=0)
            settle_count += 1 if (prev_label_info == self.label_info).all() else 0

            # 3.重心計算(次のループの重心となる)
            prev_center_info = np.copy(self.center_info)
            for n in range(self.k):
                mask = self.label_info == n
                mask = (mask * np.ones((feature_data.shape[0], 1))).astype(bool)
                clustered_data = feature_data[mask].reshape((feature_data.shape[0], np.sum(mask[0])))
                self.center_info[:, n] = self._cal_center(clustered_data)

            # 4.計算終了条件のチェック
            #同じデータ点Xnについて同じクラスタが連続で指定回数以上振られたら終了
            if settle_count > self.settle_repeat_count:
                print("COMP! Condition1 Loop:{}".format(i))
                break
            #クラスタの中心と計算された重心が指定した距離範囲に収まったら終了
            sample_count = 0
            for index in range(self.center_info.shape[1]):
                row = self.center_info.shape[0]
                point_a = self.center_info[:, index].reshape((row, 1))
                point_b = prev_center_info[:, index].reshape((row, 1))
                if self._cal_distance(point_a, point_b) < self.settle_distance:
                    #print("cal:{} condition:{}".format(self._cal_distance(point_a, point_b), self.settle_distance))
                    sample_count += 1

            if sample_count == self.center_info.shape[1]:
                print("COMP! Condition2 Loop:{}".format(i))
                break

        result = self.label_info
        return result

    def view_result(self):
        return self.label_info

    def cal_SSE(self, feature_data):
        feature_data = np.copy(feature_data)
        feature_data = feature_data.T
        return self._SSE(self.center_info, self.label_info, feature_data)

    def cal_silhouette(self, feature_data):
        feature_data = np.copy(feature_data)
        feature_data = feature_data.T
        return self._cal_silhouette_coff(feature_data)

    def _SSE(self, base_point, label_info, data):
        SSE = 0
        for i in range(base_point.shape[1]):
            #クラス iに属するデータを選択　クラスiのSSEを計算
            mask = label_info == i
            mask = (mask * np.ones((data.shape[0], 1))).astype(bool)
            tmp = data[mask].reshape((mask.shape[0], np.sum(mask[0])))
            diff_vector = tmp - base_point[:, i].reshape((base_point.shape[0], 1))
            SSE += np.sum(np.linalg.norm(diff_vector, axis=0))

        return SSE

    def _cal_silhouette_coff(self, data):

        s_n = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            #anを計算
            mask = self.label_info == self.label_info[i]
            mask = (mask * np.ones((data.shape[0], 1))).astype(bool)
            tmp = data[mask].reshape((mask.shape[0], np.sum(mask[0])))
            distance = self._cal_distance(data[:, i].reshape((mask.shape[0], 1)), tmp)
            a_n = np.average(distance)

            #bnを計算
            #一番近いクラスタを重心の位置から決定
            label_index = self.label_info[i]
            base = self.center_info[:, label_index]
            distance = self._cal_distance(base.reshape((len(base), 1)), self.center_info)
            nearest_index = np.argmin(distance[distance != 0])
            #データ点から最近棒クラスタのデータの平均距離を算出
            mask = self.label_info == nearest_index
            mask = (mask * np.ones((data.shape[0], 1))).astype(bool)
            tmp = data[mask].reshape((mask.shape[0], np.sum(mask[0])))
            distance = self._cal_distance(data[:, i].reshape((mask.shape[0], 1)), tmp)
            b_n = np.average(distance)

            #Snを算出
            s_n[i] = (b_n - a_n) / max(a_n, b_n)

        return s_n