import numpy as np


class ScratchClfEvaluation:

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
        tmp = self._AND_logic(tmp1, tmp2)
        return tmp

    def _cal_tp(self, test_target, result, pos_label=1):
        if pos_label == 1:
            x = test_target
            y = result
        elif pos_label == 0:
            x = self._NAND_logic(test_target, test_target)
            y = self._NAND_logic(result, result)

        tmp = self._AND_logic(x, y)
        return np.sum(tmp == 1)

    def _cal_fp(self, test_target, result, pos_label=1):
        if pos_label == 1:
            x = test_target
            y = result
        elif pos_label == 0:
            x = self._NAND_logic(test_target, test_target)
            y = self._NAND_logic(result, result)

        tmp1 = self._XOR_logic(x, y)
        tmp = self._AND_logic(tmp1, y)
        return np.sum(tmp == 1)

    def _cal_tn(self, test_target, result, pos_label=1):
        if pos_label == 0:
            x = test_target
            y = result
        elif pos_label == 1:
            x = self._NAND_logic(test_target, test_target)
            y = self._NAND_logic(result, result)

        tmp = self._AND_logic(x, y)
        return np.sum(tmp == 1)

    def _cal_fn(self, test_target, result, pos_label=1):
        if pos_label == 1:
            x = test_target
            y = result
        elif pos_label == 0:
            x = self._NAND_logic(test_target, test_target)
            y = self._NAND_logic(result, result)

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
        print("tp={} tn={} fp={} fn={}".format(self._cal_tp(ans, pred),
                                               self._cal_tn(ans, pred),
                                               self._cal_fp(ans, pred),
                                               self._cal_fn(ans, pred)))

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