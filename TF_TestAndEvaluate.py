"""用于评价问答
根据两个文件
    QA Pair(question, answer, label)文件、
    QA Pair的打分（cosin similarity）文件

    NLPCC2016采用了三种指标：（1）Mean Reciprocal Rank（MPP）、（2）Accuracy、（3）Average F1
     此处令：正解只有一个，那么指标1、2计算出来的结果相等
"""

import sys
import codecs


class Evaluator(object):

    # ｛问题、｛答案、得分｝｝ 字典
    qIndex2aIndex2aScore = {}
    qIndex2aIndex2aLabel = {}

    # 错误测试答案tuple list（问题、答案、得分）
    wrong_ans_list = []

    def __init__(self, qaPairFile, scoreFile, metrics_save_file):
        """
        :param qaPairFile (q,a,label)
        :param scoreFile 运用模型打完分的文件
        :param metrics_save_file (MRR、Accuracy、F值)三个评价指标
        """
        self.loadData(qaPairFile, scoreFile)
        # 可能有多个答案的情况Smax - Si <= r(0.3)也也被是为正确答案
        self.r = 0.3
        self.mrr = 0.0
        self.accuracy = 0.0
        self.f_score = 0.0
        self.similarity_thredhold = 0.5  # 一个答案的相似度至少得大于0.5
        self.qaPairFile = qaPairFile
        self.scoreFile = scoreFile
        self.metrics_save_file = metrics_save_file

    def loadData(self, qaPairFile, scoreFile):
        qaPairLines = codecs.open(qaPairFile, 'r', 'utf-8').readlines()
        scoreLines = open(scoreFile).readlines()
        print("len_qa=%d, len_score_file=%d" %(len(qaPairLines), len(scoreLines)))
        #assert len(qaPairLines) == len(scoreLines)

        for idx in range(len(scoreLines)):
            qaLine = qaPairLines[idx].strip()
            qaLineArr = qaLine.split('\t')
            assert len(qaLineArr) == 3
            question = qaLineArr[0]
            answer = qaLineArr[1]
            label = int(qaLineArr[2])
            score = float(scoreLines[idx])

            if question not in self.qIndex2aIndex2aScore:
                self.qIndex2aIndex2aScore[question] = {}
                self.qIndex2aIndex2aLabel[question] = {}
            self.qIndex2aIndex2aLabel[question][answer] = label
            self.qIndex2aIndex2aScore[question][answer] = score

    def evaluation_matric(self):
        """问答的三个评价指标计算，平均倒数排名、准确率、F值
        """
        # 目前设为1个
        A = []
        total_ques_num = 0
        mrr_sum = 0
        accuracy_sum = 0
        f_score_sum = 0
        for ques, ques_ans_scores in self.qIndex2aIndex2aScore.items():
            # [('hometown', 9), ('my', 5), ('go', 3), ('lets', 2), ('to', -2)] return tuple list
            sorted_ans_scores = sorted(ques_ans_scores.items(), key=lambda d: d[1], reverse=True)
            Candidates = self.get_candidate_ans(sorted_ans_scores)

            mrr_sum = self.sum_mrr(ques, mrr_sum, Candidates)
            accuracy_sum = self.sum_accuracy_num(ques, accuracy_sum, Candidates)
            f_score_sum = self.sum_f_score_num(ques, f_score_sum, Candidates)
            total_ques_num += 1
            Candidates = []

        assert total_ques_num > 0

        self.mrr = float(mrr_sum) / total_ques_num
        self.accuracy = float(accuracy_sum) / total_ques_num
        self.f_score = float(f_score_sum) / total_ques_num

    def get_candidate_ans(self, ans_scorers):
        assert list == type(ans_scorers) and ans_scorers
        max_ans_score = ans_scorers[0]
        for i in range(1, len(ans_scorers)):
            if max_ans_score[1] - ans_scorers[i][1] > self.r:
                return ans_scorers[:i]
        # 只有一个候选答案情况
        return ans_scorers[:1]

    def sum_mrr(self, ques, mrr_num, condidates):
        assert len(condidates) > 0
        rank_i = 0
        for i in range(len(condidates)):
            if self.qIndex2aIndex2aLabel[ques][condidates[i][0]] == 1:
                rank_i = i + 1
                break
        if rank_i == 0:
            return 0 + mrr_num
        return mrr_num + (float(1) / rank_i)

    def sum_f_score_num(self, ques, f_score_num, candidates):
        true_candidate_num = 0
        # candidates的元素类型是tuple元组
        for candi in candidates:
            if self.qIndex2aIndex2aLabel[ques][candi[0]] == 1:
                true_candidate_num += 1
        return f_score_num + 2 * true_candidate_num / (len(candidates) + 1)   # 暂时A长度设为1

    def sum_accuracy_num(self, ques, accuracy_sum, candidates):
        # candidates的元素类型是tuple元组
        for candi in candidates:
            if self.qIndex2aIndex2aLabel[ques][candi[0]] == 1:
                accuracy_sum += 1
                return accuracy_sum
        # 候选答案没有一个正确
        return 0 + accuracy_sum

    def evaluate(self):
        self.evaluation_matric()
        with open(self.metrics_save_file, 'w') as f:
            f.write(" MRR: %.3f \n Accuracy_1: %.3f \n F_Score: %.3f"
                    % (self.mrr, self.accuracy, self.f_score))
        f.close()
        print("测试集评价完毕，见文件：" + metrics_save_file)

if __name__ == '__main__':

    #  bash 运行版本
    if len(sys.argv) != 4:
        print("格式错误： 测试集问答对、测试集打分文件、问答三指标写入文件")
        exit(-1)
    qApairFile = sys.argv[1]
    scoreFile = sys.argv[2]
    metrics_save_file = sys.argv[3]

    test_res = Evaluator(qApairFile, scoreFile, metrics_save_file)
    test_res.evaluate()
