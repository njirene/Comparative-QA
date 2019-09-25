"""引入此文件的初衷：
当md文件达到5M大小左右时，jupyter加载会特别的卡，甚至加载不出来，
 可能原因是：文件有太多输出

 此版本为可运行的版本
"""
import tensorflow as tf
from TF_LSTMQa_Att_Multilayer import LSTMQa
import TF_DataParser as data_parser
import time
import sys

# embedding默认
embedding_file = "data/wiki.zh.vec"
embeddingSize = 300  # embedding维度
rnnSize = 100  # LSTM cell中隐藏层神经元的个数
margin = 0.1  # M is constant margin

unrollSteps = 25  # 句子中的最大词汇数目
# 问句、答案采取定长
sentence_max_len, embedding_dim = 25, 300
# 取决于词向量维度
input_dim, hidden_dim, output_dim, margin = 300, 200, 300, 0.3

max_grad_norm = 5
dropout = 1.0
# 学习速度、学习速度下降速度、学习速度下降次数
learningRate, lrDownRate, lrDownCount = 0.4, 0.5, 4

batch_size, learning_rate, epochs = 50, 0.3, 20

try_device = "/cpu:0"

"""接收四个参数
"""
if len(sys.argv) != 4:
    print("格式错误： 模型路径、测试集路径、测试打分结果存储文件")
    exit(-1)
modelSaveFile = sys.argv[1]
test_file = sys.argv[2]
score_saved_test_file = sys.argv[3]

"""加载训练集、测试集
"""
print("加载词向量，大概需要1分30秒左右...........")
start_time = time.time()
embedding, word2idx = data_parser.load_embedding(embedding_file, embeddingSize)
print("词向量加载完成...........")
print("耗时：%.2f 分钟" %((time.time() - start_time) / 60))


# 定义未登录词集合
oov_set = set()

# 加载测试集
qTest, aTest, tLabels = data_parser.load_test_data(test_file, word2idx, sentence_max_len, oov_set)
# #################加载训练集、测试集#########################

# 配置tensorflow
with tf.Graph().as_default(), tf.device(try_device):
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf).as_default() as sess:
        # 加载LSTM NET
        print("加载LSTM NET 开始")
        start_time = time.time()
        # 优化学习速率的参数
        globalStep = tf.Variable(0, name="global_step", trainable=False)
        # init
        lstm = LSTMQa(batch_size, sentence_max_len, embedding, embeddingSize, rnnSize, margin)
        # 获取训练变量列表
        tvars = tf.trainable_variables()
        # 剪枝
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
        # 存储Graph中所有的变量
        saver = tf.train.Saver()

        # 加载模型
        saver.restore(sess, modelSaveFile)

        print("初始LSTM网络耗时：%.2f 分钟" % ((time.time() - start_time) / 60))

        # 给测试集打分
        # 测试模型
        with open(score_saved_test_file, 'w') as file:
            for question, answer, _ in data_parser.test_data_iter(qTest, aTest, tLabels, batch_size):
                feed_dict = {
                    lstm.inputTestQuestions: question,
                    lstm.inputTestAnswers: answer,
                    lstm.keep_prob: dropout
                }
                _, scores = sess.run([globalStep, lstm.result], feed_dict)
                for score in scores:
                    file.write("%.9f" % score + '\n')
            file.close()
        print("测试集打分结束：" + score_saved_test_file)

#usage command: python TF_UseTrainedModel2ScoreTestFile.py result2/savedModel.model baidu_sousuo438_plus_nlpcc1000_nospace test.scores

