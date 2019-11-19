import numpy as np
import tensorflow as tf

from data.data_utils import *

class TextRNN(object):
    """文本分类，TextRNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_sen_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.keep_prob = tf.where(self.is_training, config.dropout_keep_prob, 1.0)

        self.rnn()

    def rnn(self):
        # lstm核
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
        # gru核
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)
        # 为每一个rnn核后面加一个dropout层
        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        """RNN模型"""
        # 词向量映射
        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([self.config.vocab_size, self.config.embedding_dim])
            embedding = tf.get_variable("embedding", initializer=init_embeddings, dtype=tf.float32, trainable=self.config.update_w2v)
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 获取batch
    def get_batches(self, x, y=None, batch_size=64, is_shuffle=True):
        for index in batch_index(len(x), batch_size, is_shuffle=is_shuffle):
            n = len(index)
            feed_dict = {
                self.input_x: [x[i] for i in index]
            }
            if y is not None:
                feed_dict[self.input_y] = [y[i] for i in index]
            yield feed_dict, n
    
    # 对一个batch训练
    def train_on_batch(self, sess, feed):
        feed[self.is_training]=True
        _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict=feed)
        return _loss, _acc

    # 对一个batch验证
    def val_on_batch(self, sess, feed):
        feed[self.is_training]=False
        _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict=feed)
        return _loss, _acc

    # 对一个batch预测
    def predict_on_batch(self, sess, feed, prob=True):
        feed[self.is_training]=False
        result = tf.argmax(self.logits, 1)
        if prob:
            result = tf.nn.softmax(logits=self.logits, dim=1)

        res = sess.run(result, feed_dict=feed)
        return res

    # 预测输入x
    def predict(self, sess, x, prob=False):
        y_pred = []
        for _feed, _ in self.get_batches(x, batch_size=self.config.batch_size, is_shuffle=False):
            _y_pred = self.predict_on_batch(sess, _feed, prob)
            y_pred += _y_pred.tolist()
        return np.array(y_pred)

    def evaluate(self, sess, x, y):
        """评估在某一数据集上的准确率和损失"""
        num = len(x)
        total_loss = 0.0
        total_acc = 0.0
        for _feed, _n in self.get_batches(x, y, batch_size=self.config.batch_size):
            loss, acc = self.val_on_batch(sess, _feed)
            total_loss += loss * _n
            total_acc += acc * _n
        return total_loss / num, total_acc / num
