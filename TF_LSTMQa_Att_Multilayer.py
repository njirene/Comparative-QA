
"""
基于LSTM Attention机制的地理中文问答
"""

import tensorflow as tf

embedding_error_file="embedding_log"


class LSTMQa(object):
    """基于lstm的中文地理知识库问答
    实现
    """
    def __init__(self, batchSize, unrollSteps, embeddings, embeddingSize, rnnSize, margin):
        self.batchSize = batchSize
        self.unrollSteps = unrollSteps
        self.embeddings = embeddings
        self.embeddingSize = embeddingSize
        self.rnnSize = rnnSize
        self.margin = margin

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.unrollSteps])

        self.h = 2 * rnnSize   # h 是basic lstm单元的输出长度

        # 设置word embedding层
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            try:
                tfEmbedding = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            except:
                with open(embedding_error_file, 'w') as file:
                    file.write("embedding_len=%d" % (len(self.embeddings)))
                exit(-1)
            questions = tf.nn.embedding_lookup(tfEmbedding, self.inputQuestions)
            trueAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTrueAnswers)
            falseAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputFalseAnswers)

            testQuestions = tf.nn.embedding_lookup(tfEmbedding, self.inputTestQuestions)
            testAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTestAnswers)

        # 建立LSTM网络
        with tf.variable_scope("LSTM_scope", reuse=tf.AUTO_REUSE):
            questions_out = self.biLSTMCell(questions, self.rnnSize)
            #print("questions_out:OOOOOO", questions_out) # (Batch, T, H)
            # print("questions_max_pooling:", self.max_pooling(questions_out)) (?, H)
            question_out_maxpool = tf.nn.tanh(self.max_pooling(questions_out))
            # print("question_out_maxpool:", question_out_maxpool) (?, H)
        with tf.variable_scope("LSTM_scope", reuse=tf.AUTO_REUSE):
            trueAnswer1 = self.biLSTMCell(trueAnswers, self.rnnSize)
            falseAnswer1 = self.biLSTMCell(falseAnswers, self.rnnSize)
            # trueAnswer2 = tf.nn.tanh(self.max_pooling(trueAnswer1))
            testQuestion1 = self.biLSTMCell(testQuestions, self.rnnSize)
            testQuestion2 = tf.nn.tanh(self.max_pooling(testQuestion1))
            testAnswer1 = self.biLSTMCell(testAnswers, self.rnnSize)

        # --------------------------------------------------------------
        #            ------- Attention层------------
        # --------------------------------------------------------------
        # 初始化attention层参数,运用IBM Watson的Attention参数设置
        h = self.h
        with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
            w_am = tf.get_variable('w_am', shape=[h, h], initializer=tf.random_normal_initializer())
            w_qm = tf.get_variable('w_qm', shape=[h, h], initializer=tf.random_normal_initializer())
            w_att = tf.get_variable('w_att', shape=[1, h], initializer=tf.random_normal_initializer())

        trueAnswer2 = self.create_ans_attention_emb(question_out_maxpool, trueAnswer1, w_am, w_qm, w_att)
        falseAnswer2 = self.create_ans_attention_emb(question_out_maxpool, falseAnswer1, w_am, w_qm, w_att)
        # falseAnswer2 = tf.nn.tanh(self.max_pooling(falseAnswer1))


        testAnswer2 = self.create_ans_attention_emb(testQuestion2, testAnswer1, w_am, w_qm, w_att)
        # testAnswer2 = tf.nn.tanh(self.max_pooling(testAnswer1))


        self.trueCosSim = self.getCosineSimilarity(question_out_maxpool, trueAnswer2)
        self.falseCosSim = self.getCosineSimilarity(question_out_maxpool, falseAnswer2)
        self.loss = self.getLoss(self.trueCosSim, self.falseCosSim, self.margin)

        self.result = self.getCosineSimilarity(testQuestion2, testAnswer2)

    def create_ans_attention_emb(self, ques_out, raw_ans_out, w_am, w_qm, w_att):
        """根据Attention机制对答案重新编码
        :param ques_out 问题
        :param raw_ans_out 原始答案编码
        """
        # output_a = tf.concat([output_fw_a, output_bw_a], axis=-1)
        #   返回的结果就是concat过的，这一点与bidirectional_dynamic_rnn不一样
        # 将(Batch,T,H)转换为(H,Batch*T),使其可以和w_am相乘
        ans_shaped2mul = tf.transpose(tf.reshape(raw_ans_out, (-1, self.h)))
        print("ans_shape=", tf.shape(ans_shaped2mul))
        # Eij(就是M(a,q))的实现方法 M(a,q) = tanh(W(am)h(a) + W(qm)Oq)
        # S(a,q) = Softmax(W(ms)M(a,q)); _ha = h(a)S(a,q);
        print("tf.transpose(ques_out):", tf.shape(tf.transpose(ques_out)))  #shape=(200, ?)

        mul_ans = tf.matmul(w_am, ans_shaped2mul)
        mul_ques = tf.matmul(w_qm, tf.transpose(ques_out))
        # mul_ans (H, Batch*T), mul_ques (H, Batch),需要做对齐
        mul_ques = tf.tile(mul_ques, [1, self.unrollSteps])

        e_ij = tf.tanh(mul_ans + mul_ques)
        # print("e_ij:", e_ij) shape=(200, ?)
        s_aq = tf.matmul(w_att, e_ij)  # -> 1xbt
        s_aq = tf.squeeze(s_aq)  # -> bt
        s_aq = tf.reshape(s_aq, [-1, self.unrollSteps]) # bxt

        #applying softmax
        s_aq = tf.nn.softmax(s_aq) # -> bxt
        #answer vectors updated with attention
        final_attentioned_ans = tf.multiply(raw_ans_out, tf.expand_dims(s_aq, -1))
        print("tf.expand_dims(s_aq, -1)", tf.expand_dims(s_aq, -1))
        max_pooled_ans = self.max_pooling(final_attentioned_ans)
        return tf.nn.tanh(max_pooled_ans)

    @staticmethod
    def biLSTMCell(x, hiddenSize):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        layer_num = 2  # 两��~Blstm

        def _single_cell():
            _cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
            _cell = tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob=1.0)
            return _cell

        #fw_cell = tf.contrib.rnn.MultiRNNCell(cells=[_single_cell() for _ in range(layer_num)], state_is_tuple = True)
        #bw_cell = tf.contrib.rnn.MultiRNNCell(cells=[_single_cell() for _ in range(layer_num)], state_is_tuple = True)
        with tf.name_scope('LSTM_scope'):
            fw_cell = tf.contrib.rnn.MultiRNNCell(cells=[_single_cell() for _ in range(layer_num)], state_is_tuple = True)
            bw_cell = tf.contrib.rnn.MultiRNNCell(cells=[_single_cell() for _ in range(layer_num)], state_is_tuple = True)

        #lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        #lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        # ** 2.dropout
        keep_prob = 0.5
        #lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        #lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        # layer_num = 3  # 两��~Blstm
        # multi_cell_fw = tf.contrib.rnn.MultiRNNCell(cells=[lstm_fw_cell for _ in range(layer_num)], state_is_tuple=True)
        # multi_cell_bw = tf.contrib.rnn.MultiRNNCell(cells=[lstm_bw_cell for _ in range(layer_num)], state_is_tuple=True)
        # ** 4.�~H~]��~K�~J��~@~A
        # batch_size = 50
        # initial_state_fw = multi_cell_fw.zero_state(batch_size, tf.float32)
        # initial_state_bw = multi_cell_bw.zero_state(batch_size, tf.float32)
        try:
            # output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, input_x,initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
            output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, input_x, dtype=tf.float32)
        except:
            print("my god, exception comes, input_x")
            print(input_x)
            print(x.get_shape())
        print("No Exception, print current output")
        print(output)
        output = tf.stack(output)
        # print("self.h", self.h)
        # print("output_raw", output) 维度��~Z(T,batch,H)
        output = tf.transpose(output, [1, 0, 2])
        # print("output_transpose", output)维度(Batch,T,H)

        # self.h = 2 * lstm_fw_cell.output_size
        return output

    @staticmethod
    def oldbiLSTMCell(x, hiddenSize):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, forget_bias=1.0, state_is_tuple=True)
        # ** 2.dropout
        keep_prob = 0.5
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        layer_num = 3  # 两层lstm
        multi_cell_fw = tf.contrib.rnn.MultiRNNCell(cells=[lstm_fw_cell for _ in range(layer_num)], state_is_tuple=True)
        multi_cell_bw = tf.contrib.rnn.MultiRNNCell(cells=[lstm_bw_cell for _ in range(layer_num)], state_is_tuple=True)
        # ** 4.初始状态
        batch_size = 50
        initial_state_fw = multi_cell_fw.zero_state(batch_size, tf.float32)
        initial_state_bw = multi_cell_bw.zero_state(batch_size, tf.float32)  
        try:
            output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(multi_cell_fw, multi_cell_bw, input_x,initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
        except:
            print("input_x")
            print(input_x)                                     
                                                               
            print("lstm_fw_cell")                              
            print(lstm_fw_cell)                                
                                                               
            print("lstm_bw_cell")                              
            print(lstm_fw_cell)                                
                                                               
            print("multi_cell_fw")                             
            print(multi_cell_fw)                               
                                                               
            print("multi_cell_bw")                             
            print(multi_cell_bw)                              
            print(multi_cell_bw.compute_output_shape())
            print(input_x)
            print(x)
        output = tf.stack(output)
        # print("self.h", self.h)
        # print("output_raw", output) 维度：(T,batch,H)
        output = tf.transpose(output, [1, 0, 2])
        # print("output_transpose", output)维度(Batch,T,H)

        # self.h = 2 * lstm_fw_cell.output_size

        return output

    @staticmethod
    def getCosineSimilarity(q, a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.div(mul, tf.multiply(q1, a1))
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def getLoss(trueCosSim, falseCosSim, margin):
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss


