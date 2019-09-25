"""引入此文件的初衷：
当md文件达到5M大小左右时，jupyter加载会特别的卡，甚至加载不出来，
 可能原因是：文件有太多输出(输出打印了10万行)
"""
import tensorflow as tf
from TF_LSTMQa import LSTMQa
import TF_DataParser as data_parser
import time

print(tf.__version__)

"""初始化所有输入变量
"""
# 参数
embedding_file = "data/wiki.zh.vec"
#embedding_file = "../data/trained_wiki_wordembbedding20epoch.model.txt"
#embedding_file = "data/news_12g_baidubaike_20g_novel_90g_embedding_64.txt"
#embedding_file = "data/zh_wiki_embedding_20_epoch.model.txt"

training_pairs_file = "data/nlpcc2016_final_training_pairs.txt"
develop_test_file = "data/nlpcc2016_dev_test1000_set.txt"
test_file = "data/baidu_sousuo438_plus_nlpcc1000"
#test_file = "../data/nlpcc2016_test1000_set.txt"

# 结果文件
modelSaveFile = "result1/savedModel.model"
scored_test_set_file = "result1/scored_test_set_file.score"
scored_develop_test_set_file = "result1/scored_develop_test_set_file.score"

# 输出日志文件
output_log_file = "result1/output.log"
embeddingSize = 300  # embedding维度
#embeddingSize = 64  # embedding维度

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

# 打开日志文件，准备写入
log_f = open(output_log_file, 'a')

"""加载训练集、测试集
"""
print("加载词向量，大概需要1分30秒左右...........")
log_f.write("加载词向量，大概需要1分30秒左右...........\n")

start_time = time.time()
embedding, word2idx = data_parser.load_embedding(embedding_file, embeddingSize)
print("词向量加载完成...........")
print("耗时：%.2f 分钟" %((time.time() - start_time) / 60))
log_f.write("词向量加载完成,耗时：%.2f 分钟\n" % ((time.time() - start_time) / 60))


print("加载训练集，大概需要3分钟左右............")
log_f.write("加载训练集，大概需要3分钟左右............\n")

# 加载训练集
start_time = time.time()
# 定义未登录词集合
oov_set = set()
questions, positive_ans, negative_ans = data_parser.load_training_data(training_pairs_file,
                                                                       word2idx, sentence_max_len, oov_set)
print("训练集加载完成...........")
print("耗时：%.2f 分钟" %((time.time() - start_time) / 60))
log_f.write("训练集加载完成,耗时：%.2f 分钟\n" % ((time.time() - start_time) / 60))


# 创建训练集迭代器
tqs, tta, tfa = [], [], []
for question_iter, positive_ans_iter, negative_ans_iter in \
        data_parser.data_iter(questions, positive_ans, negative_ans, batch_size):
    tqs.append(question_iter), tta.append(positive_ans_iter), tfa.append(negative_ans_iter)

# 加载开发测试集（总训练集中的1000条）
qDevelop, aDevelop, lDevelop = data_parser.load_test_data(develop_test_file, word2idx, sentence_max_len, oov_set)

# 加载测试集
qTest, aTest, tLabels = data_parser.load_test_data(test_file, word2idx, sentence_max_len, oov_set)
# #################加载训练集、测试集#########################


epochs = 5
# 配置tensorflow
with tf.Graph().as_default(), tf.device(try_device):
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf).as_default() as sess:
        # 加载LSTM NET
        print("加载LSTM NET 开始")
        log_f.write("加载LSTM NET 开始\n")
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

        # 加载训练集、测试集以及生成迭代器
        # 加载训练集
        print("初始LSTM网络耗时：%.2f 分钟" % ((time.time() - start_time) / 60))
        log_f.write("初始LSTM网络耗时：%.2f 分钟\n" % ((time.time() - start_time) / 60))

        # 开始训练模型
        model_train_start_time = time.time()

        print("开始训练，全部训练过程大约需要12小时")
        log_f.write("开始训练，全部训练过程大约需要12小时\n")
        sess.run(tf.global_variables_initializer())  # 初始化所有变量

        lr = learningRate
        k = 0
        for i in range(lrDownCount):
            start_time = time.time()
            k += 1
            # 实例化一个优化器
            optimizer = tf.train.GradientDescentOptimizer(lr)
            optimizer.apply_gradients(zip(grads, tvars))
            trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)

            for epoch in range(epochs):
                for question, trueAnswer, falseAnswer in zip(tqs, tta, tfa):
                    startTime = time.time()
                    # 传入变量参数列表
                    feed_dict = {
                        lstm.inputQuestions: question,
                        lstm.inputTrueAnswers: trueAnswer,
                        lstm.inputFalseAnswers: falseAnswer,
                        lstm.keep_prob: dropout
                    }
                    _, step, _, _, loss = \
                        sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss], feed_dict)
                    timeUsed = time.time() - startTime
                    print("step:", step, "loss:", loss, "time:", timeUsed)
            saver.save(sess, modelSaveFile)

            print("第%d轮训练结束耗时：%.2f 小时" % (k, ((time.time() - start_time) / 3600)))
            log_f.write("第%d轮训练结束耗时：%.2f 小时\n" % (k, ((time.time() - model_train_start_time) / 3600)))
            lr *= lrDownRate

        log_f.write("模型训练完毕，总耗时：%.2f 小时\n" % ((time.time() - model_train_start_time) / 3600))

        start_test_file_time = time.time()
        log_f.write("开始给测试文件 %s 打分\n" % scored_test_set_file)

        # 测试模型
        with open(scored_test_set_file, 'w') as file:
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

        log_f.write("测试文件打分完毕，耗时：%.2f 小时\n" % ((time.time() - start_test_file_time) / 3600))
        log_f.close()