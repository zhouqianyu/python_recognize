from deep_learning import *

LEANING_RATE = 0.001  # 初始学习率
LEANING_RATE_DECAY = 0.995  # 学习率衰减

REGULARIZATION_RATE = 0.00001  # 正则化系数
MOVING_AVERAGE_DECAY = 0.9995  # 滑动平均模型参数

TRAINING_EPOCH = 100  # 迭代次数(完整过一遍数据)

NUMCEP = 26  # mfcc系数个数
N_CONTEXT = 9  # 上下文个数
BATCH_SIZE = 4

N_HIDDEN_1 = 1024
N_HIDDEN_2 = 1024
N_HIDDEN_3 = 1024 * 2
N_CELL = 100
N_HIDDEN_5 = 200

KEEP_DROPOUT_RATE = 0.95
RELU_CLIP = 20  # 避免梯度爆炸进行梯度裁剪

STDDEV = 0.046875


class VrModel:
    def __init__(self, is_training, batch_size, vacab_size):
        self.is_training = is_training
        n_mfcc = (2 * N_CONTEXT + 1) * NUMCEP
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, n_mfcc], name='inputs')
        self.targets = tf.sparse_placeholder(tf.int32, name='targets')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length')
        self.keep_dropout = tf.placeholder(tf.float32)

        self.global_step = tf.Variable(0, trainable=False)

        input_tensors = tf.transpose(self.inputs, [1, 0, 2])  # 将inputs转化为时序优先序列

        with tf.variable_scope('bi-rnn'):
            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(N_CELL, forget_bias=1.0, state_is_tuple=True)
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, input_keep_prob=self.keep_dropout)
            lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(N_CELL, forget_bias=1.0, state_is_tuple=True)
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, input_keep_prob=self.keep_dropout)

            outputs, self.state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                                  cell_bw=lstm_cell_bw, inputs=input_tensors,
                                                                  sequence_length=self.seq_length, time_major=True,
                                                                  dtype=tf.float32)
            outputs = tf.concat(outputs, 2)  # 连接两个神经元输出结果
            layer4 = tf.reshape(outputs, [-1, 2 * N_CELL])

        # 连向输出层
        with tf.variable_scope('fc6'):
            w6 = variable_on_cpu([N_HIDDEN_5, vacab_size], 'w6', tf.random_normal_initializer(stddev=STDDEV))
            b6 = variable_on_cpu([vacab_size], 'b6', tf.random_normal_initializer(stddev=STDDEV))
            layer6 = tf.matmul(layer4, w6) + b6
            if is_training:
                # tf.add_to_collection('losses', regularizer(w6))
                tf.summary.histogram('b6', b6)
                tf.summary.histogram('w6', w6)

        logits = tf.reshape(layer6, [-1, batch_size, vacab_size])  # 转化成时序优先输出
        if is_training:
            # moving_average_op = moving_average.apply(tf.trainable_variables())
            self.avg_loss = tf.reduce_mean(tf.nn.ctc_loss(self.targets, logits, self.seq_length))
            tf.summary.scalar('loss', self.avg_loss)
            self.train = tf.train.AdamOptimizer(learning_rate=LEANING_RATE).\
                minimize(self.avg_loss, global_step=self.global_step)

            # with tf.control_dependencies([moving_average_op, train]):
            #     self.train_op = tf.no_op('train')

        # 使用ctc_decoder进行解码

        self.decode, log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_length, merge_repeated=False)
        # 计算与正确结果之间的编辑距离
        self.distance = tf.reduce_mean(tf.edit_distance(tf.cast(self.decode[0], tf.int32), self.targets))  # 开始速度超级慢

        # 其中decode[0] 为一个稀疏矩阵

    def run(self, sess, dict_map, merged=None, eval=False):
        if self.is_training:
            _, avg_loss, global_step, rs, = sess.run([self.train,
                                                     self.avg_loss, self.global_step, merged], feed_dict=dict_map)
            return avg_loss, global_step, rs
        else:
            if eval:
                result, distance = sess.run([self.decode[0], self.distance], feed_dict=dict_map)
                return result, distance
            else:
                result = sess.run(self.decode[0], feed_dict=dict_map)
                return result


if __name__ == '__main__':
    print(tf.__version__)
