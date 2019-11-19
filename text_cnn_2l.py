import numpy as np
import tensorflow as tf

from data.data_utils import *

class TextCNN2L(object):
    """文本分类，TextCNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_sen_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.keep_prob = tf.where(self.is_training, config.dropout_keep_prob, 1.0)

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([self.config.vocab_size, self.config.embedding_dim])
            embedding = tf.get_variable("embedding", initializer=init_embeddings, dtype=tf.float32, trainable=self.config.update_w2v)
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            self.embedding_inputs = tf.expand_dims(self.embedding_inputs, 3)

        with tf.variable_scope('CNN_Layer1'):
            # 添加卷积层做滤波
            conv1 = tf.contrib.layers.convolution2d(self.embedding_inputs
                                                    ,self.config.num_filters
                                                    ,[self.config.window_size, self.config.embedding_dim]
                                                    ,padding='VALID')
            # 添加RELU非线性
            conv1 = tf.nn.relu(conv1) 
            # 最大池化
            pool1 = tf.nn.max_pool(conv1
                                   ,ksize=[1, self.config.pooling_window, 1, 1]
                                   ,strides=[1, self.config.pooling_stride, 1, 1]
                                   ,padding='SAME')
            # 对矩阵进行转置，以满足形状
            pool1 = tf.transpose(pool1, [0, 1, 3, 2])
        with tf.variable_scope('CNN_Layer2'):
            # 第2卷积层
            conv2 = tf.contrib.layers.convolution2d(pool1
                                                    ,self.config.num_filters
                                                    ,[self.config.window_size, self.config.num_filters]
                                                    ,padding='VALID') 
            # 抽取特征
            pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
        
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            #fc = tf.layers.dense(pool2, self.config.hidden_dim, name='fc1')
            #fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            #h_drop = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(pool2, self.config.num_classes, name='fc2')
            
            # 预测类别
            self.y_pred_class = tf.argmax(tf.nn.softmax(self.logits), 1, output_type=tf.int32)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y))
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1, output_type=tf.int32), self.y_pred_class)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


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