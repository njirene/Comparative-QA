
import tensorflow as tf
from TF_LSTMQa import LSTMQa
import TF_DataParser as data_parser

import time


# 参数
embedding_file = "/data/trained_wiki_wordembbedding20epoch.model.txt"

training_pairs_file = "/data/nlpcc2016_training_pairs.txt"
develop_test_file = "/data/nlpcc2016_develop_test_set.txt"
test_file = "/data/nlpcc2016_test_set.txt"

# 结果文件
modelSaveFile = "result/savedModel.model"
scored_test_set_file = "result/scored_test_set_file.score/"
scored_develop_test_set_file = "result/scored_develop_test_set_file.score"

embeddingSize = 50

rnnSize = 100  # LSTM cell中隐藏层神经元的个数
margin = 0.1  # M is constant margin

unrollSteps = 100  # 句子中的最大词汇数目
# 问句、答案采取定长
sentence_max_len, embedding_dim = 50, 300
# 取决于词向量维度
input_dim, hidden_dim, output_dim, margin = 300, 200, 300, 0.3

max_grad_norm = 5
dropout = 1.0
# 学习速度、学习速度下降速度、学习速度下降次数
learningRate, lrDownRate, lrDownCount = 0.4, 0.5, 4

batch_size, learning_rate, epochs = 50, 0.3, 20

try_device = "/gpu:0"


# #################加载训练集、测试集#########################
# 加载embedding
embedding, word2idx = data_parser.load_embedding(embedding_file, embeddingSize)

print("加载训练集中............")
# 加载训练集
questions, positive_ans, negative_ans = data_parser.load_training_data(training_pairs_file,
                                                                       word2idx, sentence_max_len)
# 创建训练集迭代器
tqs, tta, tfa = [], [], []
for question_iter, positive_ans_iter, negative_ans_iter in \
        data_parser.data_iter(questions, positive_ans, negative_ans, batch_size):
    tqs.append(question_iter), tta.append(positive_ans_iter), tfa.append(negative_ans_iter)
print("训练集加载完成！")

# 加载开发测试集（总训练集中的1000条）
qDevelop, aDevelop, lDevelop = data_parser.load_test_data(develop_test_file, word2idx, sentence_max_len)

# 加载测试集
qTest, aTest, tLabels = data_parser.load_test_data(test_file, word2idx, sentence_max_len)
# #################加载训练集、测试集#########################


# 测试集打分，包括开发测试集、测试集
def use_model2score_test_set(score_file, qTest, aTest, testLabels, batch_size, sess):
    with open(score_file, 'w') as file:
        for question, answer in data_parser.test_data_iter(qTest, aTest, testLabels, batch_size):
            feed_dict = {
                lstm.inputTestQuestions: question,
                lstm.inputTestAnswers: answer,
                lstm.keep_prob: dropout
            }
            _, scores = sess.run([globalStep, lstm.result], feed_dict)
            for score in scores:
                file.write("%.9f" % score + '\n')


# 配置tensorflow
with tf.Graph().as_default(), tf.device(try_device):
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf).as_default() as sess:
        # 加载LSTM NET
        print("加载LSTM NET 开始")

        # 优化学习速率的参数
        globalStep = tf.Variable(0, name="global_step", trainable=False)
        # init
        lstm = LSTMQa(batch_size, unrollSteps, embedding, embeddingSize, rnnSize, margin)
        # 获取训练变量列表
        tvars = tf.trainable_variables()
        # 剪枝
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
        # 存储Graph中所有的变量
        saver = tf.train.Saver()

        # 加载训练集、测试集以及生成迭代器

        # 开始训练模型
        print("开始训练，全部训练过程大约需要12小时")
        sess.run(tf.global_variables_initializer())  # 初始化所有变量

        lr = learningRate
        for i in range(lrDownCount):
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
        lr *= lrDownRate

        # 测试模型
        print("正在进行测试，大约需要三分钟...")
        use_model2score_test_set(scored_develop_test_set_file, qDevelop, aDevelop, lDevelop, batch_size, sess)
        use_model2score_test_set(scored_test_set_file, qTest, aTest, tLabels, batch_size, sess)

        # GAME OVER








