import os
import time
import argparse
from datetime import timedelta
import tensorflow as tf

import numpy as np
from sklearn import metrics

import re
import jieba
import pandas as pd

import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook

from data.data_utils import *
from text_rnn import TextRNN
from text_cnn import TextCNN
from run_text_cnn import TextCNN_Run
from text_cnn_2l import TextCNN2L
from naive_bayes import naive_bayes
from text_nn import TextNN

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="textcnn",
                    help="textcnn | textrnn | textcnn_run")
parser.add_argument("--set", type=str, default="train",
                    help="train | test | predict")
args = parser.parse_args()

models = ['textcnn', 'textrnn', 'textcnn_run','textcnn_2l','bayes', 'text_nn']
sets = ['train', 'test', 'predict']

class TextConfig(object):
    """Text配置参数"""
    update_w2v = True      # 是否在训练中更新w2v
    embedding_dim = 32     # 词向量维度
    max_sen_len = 50       # 序列长度 1145
    vocab_size = 131577     # 词汇大小 56366

    num_classes = 2         # 类别数
    num_filters = 256       # 卷积核数目
    filter_sizes = [3,4,5]  # 卷积核尺寸
    kernel_size=4
    window_size=20          #感受野大小
    pooling_window=4
    pooling_stride=2

    num_layers= 2           # 隐藏层层数
    hidden_dim = 256        # 隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次,即遍历整个训练样本的次数

    print_per_batch = 100   # 每多少轮输出一次结果
    save_per_batch = 10     # 每多少轮存入tensorboard
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    max_df=0.95  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉
    min_df=2     # 在低于这一数量的文档中出现的关键词（过于独特），去除掉
    n_dim = 68268 #数据维度 0.8 3 50934
    gamma = 0.1

    save_dir = './checkpoints/' # 训练模型保存的地址
    data_path = './data/comment.csv'
    stopwords_path = './data/stopwords.txt'
    train_path = './data/train.txt'
    dev_path = './data/val.txt'
    test_path = './data/test.txt'
    word_dict_path = './data/word_dict.txt'
    pre_word2vec_path = './data/wiki_word2vec_50.bin'
    word2vec_path = './data/word_vecs.txt'


# 定义时间函数，供计算模型迭代时间使用
def get_time_dif(start_time):
    """当前距初始时间已花费的时间"""
    end_time = time.time()
    diff = end_time - start_time
    return timedelta(seconds=int(round(diff)))

# 训练
def train(word_dict, model):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/'+args.model
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(os.path.join(config.save_dir, args.model)):
        os.makedirs(os.path.join(config.save_dir, args.model))

    # 载入训练集与验证集
    start_time = time.time()
    print('加载train语料库========================')
    train_df = pd.read_csv(config.train_path, sep='\t', header=None, names=["score", "content"], encoding='utf-8')
    x_train, y_train = build_word_dataset(
        train_df, word_dict, config.stopwords_path, max_sen_len=config.max_sen_len, num_class=config.num_classes)
    print('加载Validation语料库===================')
    val_df = pd.read_csv(config.dev_path, sep='\t', header=None, names=["score", "content"], encoding='utf-8')
    x_val, y_val = build_word_dataset(
        val_df, word_dict, config.stopwords_path, max_sen_len=config.max_sen_len, num_class=config.num_classes)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    print('Training and evaluating===============')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        for train_feed, train_n in model.get_batches(x_train, y_train, batch_size=config.batch_size):

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                train_feed[model.is_training]=True
                s = sess.run(merged_summary, feed_dict=train_feed)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每config.print_per_batch轮次输出在训练集和验证集上的性能
                loss_train, acc_train = model.train_on_batch(sess, train_feed)
                loss_val, acc_val = model.evaluate(sess, x_val, y_val)
                
                if acc_val >= best_acc_val:
                    # 保存在验证集上性能最好的模型
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=os.path.join(config.save_dir, args.model, 'best_text_cnn_model'))
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train Acc: {2:>7.4%},' + \
                        ' Val Loss: {3:>6.4}, Val Acc: {4:>7.4%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            # 运行优化
            train_feed[model.is_training]=True
            sess.run(model.optimizer, feed_dict=train_feed)  
            total_batch += 1
            
            if total_batch - last_improved > config.require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


# 模型测试
def test(word_dict,model):
    print('加载test语料库=========================')
    start_time = time.time()
    test_df = pd.read_csv(config.test_path, sep='\t', header=None, names=["score", "content"], encoding='utf-8')
    x_test, y_test = build_word_dataset(
        test_df, word_dict, config.stopwords_path, max_sen_len=config.max_sen_len, num_class=config.num_classes)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        print('加载model==========================')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        print('Testing============================')
        loss_test, acc_test = model.evaluate(sess, x_test, y_test)
        msg = 'Test Loss: {0:>6.4}, Test Acc: {1:>7.4%}'
        print(msg.format(loss_test, acc_test))
        # 预测x
        y_pred = model.predict(sess, x_test)

    y_cls = np.argmax(y_test, 1)

    # 评估
    print("Precision, Recall and F1-Score=====")
    print(metrics.classification_report(y_cls, y_pred, target_names=categories))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_cls, y_pred)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


# 预测
def sent_to_id(inputs_x, word_dict):
    """
    将语句进行分词，然后将词语转换为word_to_id中的id编码
    :param inputs_x: 句子：列表的形式
    :return: 用id表征的语句
    """
    # 读取停用词
    stopwords = [line.strip() for line in open(config.stopwords_path, encoding='utf-8').readlines()]
    # 只保留中文，且繁体字转为简体字
    inputs_x = Traditional2Simplified(re.sub(r'[^\u4e00-\u9fa5^A-Z^a-z^\']', ' ', inputs_x)).lower().strip()
    token = jieba.lcut(inputs_x)
    sentence = [i for i in token if i not in stopwords]
    sentence = [word_dict.get(w, word_dict["<unk>"]) for w in sentence]
    sentence = sentence[:config.max_sen_len]
    if len(sentence) < config.max_sen_len:
        sentence += [word_dict['<pad>']] * \
            (config.max_sen_len - len(sentence))
    return [sentence]

def predict(word_dict, label=True):
    """
    :param x: 语句列表
    :param label: 是否以分类标签的形式：pos或neg输出
    :return: 情感预测结果
    """
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if args.model=='textrnn':
            model = TextRNN(config)
        elif args.model=='textcnn':
            model = TextCNN(config)
        elif args.model == 'textcnn_run':
            model = TextCNN_Run(config)
        else:
            model = TextCNN2L(config)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            input_x = input("请输入：")
            if input_x=='-1':
                break
            x = sent_to_id(input_x,word_dict)
            y = model.predict(sess, x, prob=not label)

            if label:
                y = [categories[w] for w in y.tolist()]
            print('预测结果:',y)

def nn_train():
    # 载入训练集与验证集
    start_time = time.time()
    print('加载train语料库========================')
    train_df = pd.read_csv(config.train_path, sep='\t', header=None, names=["score", "content"], encoding='utf-8')

    print('加载Validation语料库===================')
    val_df = pd.read_csv(config.dev_path, sep='\t', header=None, names=["score", "content"], encoding='utf-8')

    x_train, y_train, x_val, y_val, vectorizer = build_tfidvec_dataset(train_df, val_df, config.stopwords_path, max_df=config.max_df, min_df=config.min_df, num_class=config.num_classes)
    config.n_dim = len(vectorizer.get_feature_names())

    model = TextNN(config)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    """
    train_dataloader, _train_len = model.dataloader(x_train,y_train,config.batch_size,shuffle=True,num_workers=1)
    val_dataloader, _val_len = model.dataloader(x_val,y_val,config.batch_size,shuffle=False,num_workers=1)

    model = model.double()
    # 保存准确度最高的模型
    best_model = copy.deepcopy(model)
    best_accuracy = 0.0

    for epoch in range(config.num_epochs):
        model.exp_lr_scheduler.step()
        # 训练
        model.train()
        loss_total = 0
        st = time.time()
        # train_dataloader 加载数据集
        for data, label in tqdm_notebook(train_dataloader):
            output = model(data)
            # 计算损失
            loss = model.losser(output, label)
            model.optimizer.zero_grad()
            # 反向传播
            loss.backward()
            model.optimizer.step()
            loss_total += loss.item()

        # 输出损失、训练时间等
        print('epoch {}/{}:'.format(epoch, config.num_epochs))
        print('training loss: {}, time resumed {}s'.format(
            loss_total/_train_len, time.time()-st))
        # 验证
        model.eval()
        loss_total = 0
        st = time.time()

        correct = 0
        for data, label in val_dataloader:
            output = model(data)
            loss = model.losser(output, label)
            loss_total += loss.item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
        # 如果准确度取得最高，则保存准确度最高的模型
        if correct/_val_len > best_accuracy:
            best_model = copy.deepcopy(model)
        print('val loss: {}, time resumed {}s, accuracy: {}'.format(
            loss_total/_val_len, time.time()-st, correct/_val_len))
        
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # 模型保存
    with open(os.path.join(config.save_dir,'best_nn_model.pkl'), 'wb') as file:
        save = {
            'tfidfVectorizer' : vectorizer,
            'nn_model' : best_model
        }
        pickle.dump(save, file)
    

def nn_test():
    # 加载模型
    with open(os.path.join(config.save_dir,'best_nn_model.pkl'), 'rb') as file:
        tfidf_model = pickle.load(file)
        tfidfVectorizer = tfidf_model['tfidfVectorizer']
        nn_model = tfidf_model['nn_model']

    print('加载test语料库=========================')
    start_time = time.time()
    test_df = pd.read_csv(config.test_path, sep='\t', header=None, names=["score", "content"], encoding='utf-8')
    x_test, y_test = load_tfidvec_dataset(test_df, config.stopwords_path, tfidfVectorizer)
    test_dataloader, _test_len = nn_model.dataloader(x_test,y_test,config.batch_size,shuffle=False,num_workers=1)
    print('test语料库=========================')
    nn_model.eval()
    all_pred = []
    all_label = []
    for data, label in test_dataloader:
        # 评估
        _, predicted = torch.max(nn_model(data).data, 1)
        all_pred += predicted.numpy().tolist()
        all_label += label.numpy().tolist()
    
    msg = 'Test Acc: {0:>7.4%}'
    print(msg.format(sum([all_label[i]==all_pred[i] for i in range(_test_len)])/_test_len))
    print("Precision, Recall and F1-Score=====")
    print(metrics.classification_report(all_label, all_pred, target_names=categories))
    # 混淆矩阵
    print("Confusion Matrix...")
    print(metrics.confusion_matrix(all_label, all_pred))
    print()

def nn_predict():
    # 加载模型
    with open(os.path.join(config.save_dir,'best_nn_model.pkl'), 'rb') as file:
        tfidf_model = pickle.load(file)
        tfidfVectorizer = tfidf_model['tfidfVectorizer']
        nn_model = tfidf_model['nn_model']
    
    nn_model.eval()
    # 读取停用词
    stopwords = [line.strip() for line in open(config.stopwords_path, encoding='utf-8').readlines()]
    while True:
        input_x = input("请输入：")
        if input_x=='-1':
            break
        # 只保留中文，且繁体字转为简体字
        input_x_ = Traditional2Simplified(re.sub(r'[^\u4e00-\u9fa5^A-Z^a-z^\']', ' ', input_x)).lower().strip()
        token = jieba.lcut(input_x_)
        sentence = ' '.join([i for i in token if i not in stopwords])

        # 转化为特征向量
        one_test_data = tfidfVectorizer.transform([sentence])

        # 转化为 pytorch 输入的 Tensor 数据，squeeze(0) 增加一个 batch 维度
        one_test_data = torch.from_numpy(one_test_data.toarray()).unsqueeze(0)
        # 使用准确度最好的模型预测，softmax 处理输出概率，取得最大概率的下标
        pred = torch.argmax(F.softmax(nn_model(one_test_data), dim=1))
        print('预测结果:',categories[pred])

if __name__ == "__main__":
    if args.set in sets and args.model in models:
        print('配置模型参数=============================')
        config = TextConfig()
        categories, cat2id = class_to_id(is_2c=config.num_classes==2)
        if args.model == 'bayes':
            naive_bayes(config.data_path,categories)
        else:
            if os.path.exists(config.word_dict_path) and os.path.exists(config.train_path) and os.path.exists(config.dev_path) and os.path.exists(config.test_path):
                print('加载word_dict===========================')
                word_dict = load_word_dict(config.word_dict_path)
            else:
                print('划分数据集==============================')
                train_df,val_df,test_df = unique_divide(config.data_path, out_path='./data', is_2c=config.num_classes==2)
                print('加载word_dict===========================')
                word_dict,max_sen_len,vocab_size = build_word_dict(train_df,val_df,config.word_dict_path, config.stopwords_path, config.vocab_size)
                #config.max_sen_len = max_sen_len
            #config.vocab_size = len(word_dict)
            
            print('加载word_vecs===========================')
            # word2vecs = load_corpus_word2vec(config.word2vec_path)
            # word2vecs = build_word2vec(config.pre_word2vec_path, word_dict, config.word2vec_path)        
            # word2vec = pd.read_csv(config.word2vec_path, sep='\t', header=None, names=['word','id'], encoding='utf-8')

            if args.model=='textrnn':
                model = TextRNN(config)
            elif args.model=='textcnn':
                model = TextCNN(config)
            elif args.model == 'textcnn_run':
                model = TextCNN_Run(config)
            elif args.model == 'textcnn_2l':
                model = TextCNN2L(config)

            if args.set == "train":
                if args.model == 'text_nn':
                    nn_train()
                else:
                    train(word_dict,model)
            elif args.set == "test":
                if args.model == 'text_nn':
                    nn_test()
                else:
                    test(word_dict,model)
            elif args.set == "predict":
                if args.model == 'text_nn':
                    nn_predict()
                else:
                    # 使用训练所得模型进行电影评论分析 label :one-hot表示pos-[1, 0], neg-[0, 1]
                    tf.reset_default_graph()
                    predict(word_dict, label=True)
    else:
        raise NotImplementedError()
