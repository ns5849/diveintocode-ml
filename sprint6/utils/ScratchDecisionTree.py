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


class ScratchDecisionTreeClassifier(ScratchClfEvaluation):
    """
    決定木のスクラッチ実装
    各Nodeに対して作られるBranchは２つ

    Parameters
    ----------
    max_depth : int
      イテレーション数

    Attributes
    ----------
    self.node_ID : 次の形のndarray, shape (depth, index)
      NodeのID（深さと横方向）
    self.dt_feat　次の形のndarray, shape (depth, index)
      分類に用いる特徴量
    self.dt_threshold 次の形のndarray, shape (depth, index)
    　分類のための閾値
    """

    def __init__(self, max_depth, verbose=False):
        # ハイパーパラメータを属性として記録
        self.max_depth = max_depth
        self.label_list = 0  # クラスのラベルを格納 ２種類なら２, 3種類なら3つ入る
        self.node_id = np.array([[None for _ in range(2 ** i)] for i in range(max_depth + 1)])
        self.node_feature_index = np.array([[None for _ in range(2 ** i)] for i in range(max_depth + 1)])
        self.node_threshold = np.array([[None for _ in range(2 ** i)] for i in range(max_depth + 1)])
        self.node_label = np.array([[None for _ in range(2 ** i)] for i in range(max_depth + 1)])
        self.node_label[0][0] = 0
        self.clf_comp = np.array([[True for _ in range(2 ** i)] for i in range(max_depth + 1)])
        self.clf_comp[0][0] = False

    def cal_accuracy(self, y_pred, y):
        return self.eval_accuracy(y_pred, y)

    def cal_precision(self, y_pred, y):
        return self.eval_precision(y_pred, y)

    def cal_recall(self, y_pred, y):
        return self.eval_recall(y_pred, y)

    def cal_f1(self, y_pred, y):
        return self.eval_f1(y_pred, y)

    def _cal_gini_impurity(self, target_data):
        """
        Gini不純度を計算

        target : ndarray shape(1, sample)
        """
        tmp = 0
        count_all = len(target_data)
        for i in range(len(self.label_list)):
            count = np.sum([target_data == self.label_list[i]])
            tmp += (count / count_all) ** 2

        return 1 - tmp

    def _cal_info_gain(self, target_data_left, target_data_right):
        """
        情報利得を計算

        target_data_left : ndarray shape(1, sample(left))
        target_data_right : ndarray shape(1, sample(right))
        """
        comp = False

        # 親データのジニ不純度計算のためright, leftのデータを合体
        target_data_left.reshape(1, len(target_data_left))
        target_data_right.reshape(1, len(target_data_right))
        target_data = np.concatenate((target_data_left, target_data_right), axis=0)
        # 親、右、左のジニ不純度を計算
        count_parent = len(target_data_left) + len(target_data_right)
        gini_parent = self._cal_gini_impurity(target_data)
        coff_left = len(target_data_left) / count_parent
        coff_right = len(target_data_right) / count_parent
        tmp = self._cal_gini_impurity(target_data_left) * coff_left
        tmp += self._cal_gini_impurity(target_data_right) * coff_right
        # print("Gini left={}  Gini right={}".format(self._cal_gini_impurity(target_data_left), self._cal_gini_impurity(target_data_right)))
        if tmp == 0:
            comp = True

        return gini_parent - tmp, comp

    def _make_branch(self, feature_data, target_data):
        """
        データを分割（情報利得が最小）

        feature : ndarray shape(feature, sample)
        target : ndarray shape(1, sample)
        """
        # データが空の場合は処理をしない。returnはNone, None
        if target_data is None:
            return None, None, None

        # 分ける必要があるかどうか判断(ラベルが１つしかなければ分ける必要なし)
        if 1 == len(np.unique(target_data)):
            return None, None

        # 初期値を代入
        info_max = 0
        feature_index = 0
        feature_threshold = feature_data[0, 0]
        # 情報利得が最小になる分割を見つける
        for i in range(feature_data.shape[0]):
            for threshold in feature_data[i]:
                mask = feature_data[i] <= threshold
                mask = mask.reshape(1, len(mask))
                target_left = target_data[mask]
                target_right = target_data[~mask]
                info_gain, clf_comp = self._cal_info_gain(target_left, target_right)
                if info_max < info_gain:
                    # print("info_max={} Threshold={} mask={}".format(info_gain, threshold, mask))
                    info_max = info_gain
                    feature_index = i
                    feature_threshold = threshold
                    # 情報利得が1の場合、ラベルは完全に分類されている。

        return feature_index, feature_threshold

    def _return_majority_label(self, target_data):

        #print("target_data=", target_data)
        label0_count = np.sum([target_data == self.label_list[0]])
        label1_count = np.sum([target_data == self.label_list[1]])
        # print("0 count", label0_count)
        # print("1 count", label1_count)
        majority_label = self.label_list[0] if label0_count >= label1_count else self.label_list[1]
        # print("majority=", majority_label)

        return majority_label

    def fit(self, feature_data, target_data):
        """
        学習データから各Nodeの特徴量、閾値を求めメンバ変数に保存

        feature_data :　shape(Sample, Feature)
        target_data : shape(Sample, 1)
        """

        # 引数を分離するため新しく変数を定義
        feature_data = np.copy(feature_data)
        target_data = np.copy(target_data)

        # 関数内のデータフォーマットへ変換 shape(Feature, Sample)
        feature_data = feature_data.T
        target_data = target_data.reshape(1, len(target_data))

        # クラスが何種類あるか数える
        self.label_list = np.unique(target_data)

        # ルートデータを設定(feature_data, target_data)
        i = 0
        node_feature_data = np.array(np.array([[i * n for n in range(2 ** i)] for i in range(self.max_depth + 1)]))
        node_target_data = np.array(np.array([[i * n for n in range(2 ** i)] for i in range(self.max_depth + 1)]))
        node_feature_data[0][0] = feature_data
        node_target_data[0][0] = target_data

        # 深さ方向のループ
        for depth in range(self.max_depth):
            # 各階層の横方向の広がり分のループ
            for index in range(len(self.node_id[depth])):
                # Node ID
                self.node_id[depth][index] = index
                # このNode（親データ）からBranchを作るための特徴量、閾値を決定
                #print("Loop = d:{} i:{}".format(depth, index))
                p_feature_data = node_feature_data[depth][index]
                p_target_data = node_target_data[depth][index]
                feature_index, feature_threshold = self._make_branch(p_feature_data, p_target_data)
                # feature_index, feature_threshold = NoneならLoopをskip
                if feature_index is None:
                    continue

                    # このNodeで決定した特徴量(feature_index)と閾値をリストに保存
                self.node_feature_index[depth][index] = feature_index
                self.node_threshold[depth][index] = feature_threshold
                # 次のNode（次の層の親データ）を作成しリストへ保存
                next_depth = depth + 1
                next_index_left = 2 * index
                next_index_right = 2 * index + 1
                mask_array = p_feature_data[feature_index] <= feature_threshold
                mask_array = mask_array.reshape(1, len(mask_array))
                mask_array = np.tile(mask_array, (p_feature_data.shape[0], 1))
                mask_row = mask_array[0].reshape((1, len(mask_array[0])))
                num_row = mask_array.shape[0]
                num_col_T = np.sum([mask_row == True])
                num_col_F = np.sum([mask_row == False])

                # 次の層のデータを格納
                if num_col_T != 0:
                    node_feature_data[next_depth][next_index_left] = p_feature_data[mask_array].reshape(
                        (num_row, num_col_T))
                    node_target_data[next_depth][next_index_left] = p_target_data[mask_row].reshape((1, num_col_T))
                    self.node_label[next_depth][next_index_left] = self._return_majority_label(
                        node_target_data[next_depth][next_index_left])
                else:
                    node_feature_data[next_depth][next_index_left] = None
                    node_target_data[next_depth][next_index_left] = None
                    self.node_label[next_depth][next_index_left] = None
                if num_col_F != 0:
                    node_feature_data[next_depth][next_index_right] = p_feature_data[~mask_array].reshape(
                        (num_row, num_col_F))
                    node_target_data[next_depth][next_index_right] = p_target_data[~mask_row].reshape((1, num_col_F))
                    self.node_label[next_depth][next_index_right] = self._return_majority_label(
                        node_target_data[next_depth][next_index_right])
                else:
                    node_feature_data[next_depth][next_index_right] = None
                    node_target_data[next_depth][next_index_right] = None
                    self.node_label[next_depth][next_index_right] = None

                # 分けた結果ラベルが１種類に分別されているかどうか判断
                if 1 == len(np.unique(node_target_data[next_depth][next_index_left])):
                    self.clf_comp[next_depth][next_index_left] = True
                else:
                    self.clf_comp[next_depth][next_index_left] = False
                if 1 == len(np.unique(node_target_data[next_depth][next_index_right])):
                    self.clf_comp[next_depth][next_index_right] = True
                else:
                    self.clf_comp[next_depth][next_index_right] = False

        return True

    def predict(self, feature_data):
        """
        fitした結果（特徴量、閾値からラベルを予測する。閾値より小さい場合は左、大きい場合は右へ進む）
        """

        # 引数を分離するため新しく変数を定義
        feature_data = np.copy(feature_data)
        # 関数内のデータフォーマットへ変換 shape(Feature, Sample)
        feature_data = feature_data.T
        # 次の層のNode indexを入れるリストを用意(1, sample) ==> 最終的に行き着いたindexのlabeが答えとなる
        answer = np.array([0 for _ in range(feature_data.shape[1])])

        # 深さ方向のループ
        for sample_index in range(feature_data.shape[1]):
            next_node_index = 0
            for depth in range(self.max_depth):
                index = next_node_index
                # 各Nodeの閾値と比較　次のノードを決める
                if feature_data[self.node_feature_index[depth][index], sample_index] <= self.node_threshold[depth][
                    index]:
                    next_node_index = 2 * index
                else:
                    next_node_index = 2 * index + 1

                # Comp flagがTrueならリーフ
                if self.clf_comp[depth + 1][next_node_index] == True:
                    break

            if self.clf_comp[depth + 1][next_node_index] == True:
                answer[sample_index] = self.node_label[depth + 1][next_node_index]
            else:
                answer[sample_index] = self.node_label[depth + 1][next_node_index]

        return answer
