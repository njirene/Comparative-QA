# 数据处理

from collections import defaultdict
import jieba
import numpy as np

"""Training set：指定一个Question只有一个正确答案，同时对应20个错误答案
为平衡训练，暂采取 正例：反例 = 1 ：1进行训练（即讲正确答案复制19次）
   Testing set：一个Question给出20个候选答案，其中可以给定多个正确答案
"""


def load_embedding(embedding_file, embedding_dim):
    """加载预训练好的embedding，一般是txt文件
    :param embedding_file 训练好的embedding文本文件
    :param embedding_dim embedding维度
    :returns embedding矩阵，词汇表字典((key,value)= (vocabulary, index))
    """
    embedding_matrix = []
    word2vocabulary_index = defaultdict(list)
    word2vocabulary_index['UNKNOWN'] = 0  # 未登录词

    embedding_matrix.append([0] * embedding_dim) # 设置未登录词的embedding
    with open(embedding_file, 'r', encoding="utf-8") as f:
        f.readline()
        for line in f.readlines():
            line_array = line.split(" ")  # embedding以空格分隔，如the 0.232 0.87323.....
            line_array = line_array[:len(line_array)-1]
            embedding = [float(val) for val in line_array[1:]]
            word2vocabulary_index[line_array[0]] = len(word2vocabulary_index) # 从1开始，0为未登录词
            embedding_matrix.append(embedding)
    word2vocabulary_index['UNKNOWN'] = 0  # 未登录词
    return embedding_matrix, word2vocabulary_index



def sentence2embedding_index(sentence, word2vocabulary_index, max_len, oov_set):
    """将分好词的句子中映射为词典索引列表(亦为embedding索引列表)
       此处会处理UNKNOWN词，或者句子可以忽视数字、特殊符号等
    :param sentence 分好词的句子（词list）
    :param word2vocabulary_index 词典
    :param max_len 句子最大长度，大于max_len则后面词语丢弃（此处采用定长表示）
    :param oov_set set用于存储未登录词
    :return sentence在词典索引列表
    """
    index = []
    i = 0
    # 将未登录词的index设为0
    unknow_index = word2vocabulary_index.get('UNKNOWN', 0)
    for word in jieba.cut(sentence):
        if word in word2vocabulary_index:
            index.append(word2vocabulary_index[word])
        else:
            index.append(unknow_index)   # 此处未登录词全用一个相同的embedding表示
            oov_set.add(word)
        i += 1
        if i >= max_len:
            break
    # 此处先将问题和答案设为定长max_len
    if i < max_len:
        index.extend([0] * (max_len - i))
    return index


# 加载训练集
def load_training_data(file_path, word2embedding_index, sentence_max_len, oov_set):
    """加载训练集
        训练集行格式：question \t positive ans \t negative ans
    :param word2embedding_index 词典
    :param file_path 文件路径
    :param sentence_max_len 问题或答案最大长度、该长度后的词语丢弃（本实验设置等长）
    :param oov_set 未登录词记录表
    :return questions、positive ans 、negative ans
    """
    questions, positive_ans, negative_ans = [], [], []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line_array = line.split("\t")
            if len(line_array) != 3:  # 格式不正确
                continue
            ques_embedding_idx = sentence2embedding_index(line_array[0], word2embedding_index, sentence_max_len,
                                                          oov_set)
            positive_ans_embedding_idx = sentence2embedding_index(line_array[1], word2embedding_index, sentence_max_len,
                                                                  oov_set)
            negative_ans_embedding_idx = sentence2embedding_index(line_array[2], word2embedding_index, sentence_max_len,
                                                                  oov_set)

            questions.append(ques_embedding_idx)
            positive_ans.append(positive_ans_embedding_idx)
            negative_ans.append(negative_ans_embedding_idx)

        return questions, positive_ans, negative_ans


def data_iter(questions, posi_ans, nega_ans, batch_size):
    """batch迭代器， (问题，正解，错解)即(q, a+, a-)
    """
    sample_len = len(questions)
    # 策略：丢掉最后不足batch_size的数据，batch_size采取定长
    batch_num = int(sample_len / batch_size)

    for batch_i in range(batch_num):
        # per batch,
        pair_i = batch_i * batch_size

        question_iter = questions[pair_i: pair_i + batch_size]
        positive_answer_iter = posi_ans[pair_i: pair_i + batch_size]
        negative_answer_iter = nega_ans[pair_i: pair_i + batch_size]

        yield np.array(question_iter), np.array(positive_answer_iter), np.array(negative_answer_iter)


# 加载测试集与训练集格式不同
# （训练集采取question \t positive ans \t negative ans），测试集格式 为question \t answer \t label(1或0)
def load_test_data(file_path, word2embedding_index, sentence_max_len, oov_set):
    """加载训练集
        训练集行格式：question \t answer \t label(1或0)
    :param word2embedding_index 词典
    :param file_path 文件路径
    :param sentence_max_len 问题或答案最大长度、该长度后的词语丢弃（本实验设置等长）
    :return questions、answers、label
    """
    questions, ans, labels = [], [], []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line_array = line.split("\t")
            if len(line_array) != 3:   # 格式不正确
                continue
            ques_embedding_idx = sentence2embedding_index(line_array[0], word2embedding_index, sentence_max_len, oov_set)
            ans_embedding_idx = sentence2embedding_index(line_array[1], word2embedding_index, sentence_max_len, oov_set)
            label = line_array[2]

            questions.append(ques_embedding_idx)
            ans.append(ans_embedding_idx)
            labels.append(label)

        return questions, ans, labels


# 测试集迭代器
def test_data_iter(questions, ans, labels, batch_size):
    """batch迭代器， (问题，测试答案)
    """
    sample_len = len(questions)
    # 策略：丢掉最后不足batch_size的数据，batch_size采取定长
    batch_num = int(sample_len / batch_size)

    for batch_i in range(batch_num):
        # per batch,
        pair_i = batch_i * batch_size

        question_iter = questions[pair_i: pair_i + batch_size]
        answer_iter = ans[pair_i: pair_i + batch_size]
        label_iter = labels[pair_i: pair_i + batch_size]

        yield question_iter, answer_iter, label_iter
