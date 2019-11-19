
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer

from snownlp import SnowNLP

# 清洗数据,通过jieba分词
def word_clean(mytext):
    return ' '.join(jieba.lcut(mytext))

def get_sentiment(text):
    return 1 if SnowNLP(text).sentiments >0.5 else 0

# 使用贝叶斯预测分类
def naive_bayes(data_path, categories):
    df = pd.read_csv(data_path, header=None, names=['score_comment'])
    data_unique = df.drop_duplicates()
    split_df = pd.DataFrame(data_unique.score_comment.str.split('\t').tolist(), columns=["score", "comment"])

    x=split_df[['comment']]
    x['cutted_comment'] = x.comment.apply(word_clean)
    if len(categories)==2:
        split_df['score']=split_df['score'].map(lambda s:1 if int(s[0])>2 else 0)
    y=split_df.score

    #划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    nb = MultinomialNB()
    # 读取停用词
    stopwords = [line.strip() for line in open('data/stopwords.txt', encoding='utf-8').readlines()]
    max_df=0.8  # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉
    min_df=3    # 在低于这一数量的文档中出现的关键词（过于独特），去除掉
    vect=CountVectorizer(max_df=max_df,min_df=min_df,stop_words=frozenset(stopwords))
    # 利用管道顺序连接工作
    pipe=make_pipeline(vect,nb)

    #交叉验证的准确率
    cross_result=cross_val_score(pipe,x_train.cutted_comment,y_train,cv=5,scoring='accuracy').mean()
    print('交叉验证的准确率：'+str(cross_result))

    #进行预测
    pipe.fit(x_train.cutted_comment,y_train)
    y_pred = pipe.predict(x_test.cutted_comment)

    # 评估
    print("Precision, Recall and F1-Score=====")
    print(metrics.classification_report(y_test, y_pred, target_names=categories))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)

    print('\nPython自带情感分析方法结果:')

    # 评估
    y_pred_snownlp = [get_sentiment(sentence) for sentence in x_test.cutted_comment]
    msg = 'Test Acc: {0:>7.4%}'
    sum_equal = 0
    i=0
    for label in y_test:
        sum_equal += label==y_pred_snownlp[i]
        i+=1
        
    print(msg.format(sum_equal/len(y_pred_snownlp)))
    print("Precision, Recall and F1-Score=====")
    print(metrics.classification_report(y_test, y_pred_snownlp, target_names=categories))
    # 混淆矩阵
    print("Confusion Matrix...")
    print(metrics.confusion_matrix(y_test, y_pred_snownlp))