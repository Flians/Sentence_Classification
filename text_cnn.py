import numpy as np
import tensorflow as tf

from data.data_utils import *

class TextCNN(object):
    """文本分类，TextCNN模型"""

    def __init__(self, config, embeddings=None):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.max_sen_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.keep_prob = tf.where(self.is_training, config.dropout_keep_prob, 1.0)

        self.cnn(embeddings)

    def cnn(self, embeddings):
        """CNN模型"""
        # 词向量映射
        with tf.name_scope("embedding"):
            if embeddings is not None:
                self.embedding=tf.Variable(embeddings,dtype=tf.float32,trainable=self.config.update_w2v)
            else:
                init_embeddings = tf.random_uniform([self.config.vocab_size, self.config.embedding_dim])
                print(tf.shape(init_embeddings))
                self.embedding = tf.get_variable("embedding", initializer=init_embeddings, dtype=tf.float32, trainable=self.config.update_w2v)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            print()
            print(tf.shape(self.embedding_inputs))
            self.embedding_inputs = tf.expand_dims(self.embedding_inputs, -1)

        # 卷积操作
        pooled_outputs = []
        with tf.name_scope("cnn"):
            for filter_size in self.config.filter_sizes:
                # 卷积
                conv = tf.layers.conv2d(
                    self.embedding_inputs,
                    filters=self.config.num_filters,
                    kernel_size=[filter_size, self.config.embedding_dim],
                    strides=(1, 1),
                    padding="VALID",
                    activation=tf.nn.relu)
                # 最大池化
                pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size=[self.config.max_sen_len - filter_size + 1, 1],
                    strides=(1, 1),
                    padding="VALID")
                pooled_outputs.append(pool)

            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.config.num_filters * len(self.config.filter_sizes)])
        
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            #fc = tf.layers.dense(h_pool_flat, self.config.hidden_dim, name='fc1')
           # fc = tf.contrib.layers.dropout(fc, self.keep_prob)
          #  h_drop = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(h_drop, self.config.num_classes, name='fc2')
            #print(tf.shape(self.logits))
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