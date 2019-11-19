import os
import re
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import gensim
from interval import Interval

try:
    from langconv import *
except ImportError:
    from data.langconv import *

#os.chdir(r'cnn_sentence_classification\data')

def clear_comment(path='./data'):
    files = ['best','new','popular']
    files = ['popular']
    for file in files:
        df = pd.read_csv(os.path.join(path,'comment_full_'+file+'.csv'), header=None, names=['username','score','comment','pub_time','votes'], encoding='utf-8')
        df['comment'] = df['comment'].map(lambda com: re.sub(r'\s+',' ', str(com)))
        #将df中score列所有空值赋值为'null',并删掉
        df['score']=df['score'].fillna('NaN')
        df=df[~df['score'].isin(['NaN'])]
        df.drop(['username','pub_time','votes'], axis=1, inplace=True)
        df['score'] =df['score'].astype(int)
        # socre设为0-4
        df['score'] =df['score'].map(lambda score: score-1)
        df.to_csv(os.path.join(path,'comment_' + file+ '.csv'), sep='\t', index=False, header=False)

    if os.path.exists(os.path.join(path,'comment.csv')):
        os.remove(os.path.join(path,'comment.csv'))
    with open(os.path.join(path,'comment.csv'), 'a+', encoding='utf-8') as cfile:
        for file in files:
            temp = open(os.path.join(path,'comment_' + file + '.csv'),'r', encoding='utf-8').readlines()
            cfile.writelines(temp)

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

# 将 socre 和 content写入out_path文件
def df_to_csv(socre, content, out_path):
    data = {'score':socre, 'content':content}
    df_data = pd.DataFrame(data, columns=['score', 'content'])
    df_data.to_csv(out_path, sep='\t', index=False, header=False)
    return df_data

# 去除csv文件中的重复行
# 将近10W条数据划分：测试集1W,验证集2W,剩余近7W数据归为训练集
# 将score>2的归为一类:正向评价；score<=2的归为一类：负面评价。{4:"力荐",3:"推荐",2:"还行",1:"较差",0:"很差"}
# 返回test,train,val
def unique_divide(file_csv, out_path='./data', is_2c=True):
    df = pd.read_csv(file_csv, header=None, names=['score_content'])
    data_unique = df.drop_duplicates()
    # data_unique.to_csv(file_csv)
    split_df = pd.DataFrame(data_unique.score_content.str.split('\t').tolist(), columns=["score", "content"])

    # 只保留中文和英文，且繁体字转为简体字,英文小写
    split_df['content'] = split_df['content'].map(lambda com: Traditional2Simplified(re.sub(r'[^\u4e00-\u9fa5^A-Z^a-z^\']', ' ', com)).lower().strip())
    split_df['content'] = split_df['content'].map(lambda com: re.sub(r'\s+', ' ', com).strip())
    split_df.info()
    print(split_df['score'].value_counts(normalize=True,ascending=True))
    # 删除评论字数小于60的
    # split_df = split_df[~(split_df.content.str.len() < 60)]
    print('\nafter deleted================\n')
    # 读取停用词
    stopwords = [line.strip() for line in open('data/stopwords.txt', encoding='utf-8').readlines()]
    zoom_5_100 = Interval(5, 50)
    # 同时对每条评论使用jieba分词，去掉停用词，去掉词数小于5的行
    rows=[i for i in split_df.index if len([j for j in jieba.lcut(str(split_df.iat[i,1]).strip()) if j not in stopwords and j.strip()]) not in zoom_5_100]
    split_df=split_df.drop(rows,axis=0)
    split_df.info()
    print(split_df['score'].value_counts(normalize=True,ascending=True))

    # 将score>2的归为一类:正向评价；score<=2的归为一类：负面评价
    # 5分类,{4:"力荐",3:"推荐",2:"还行",1:"较差",0:"很差"}
    if is_2c:
        split_df['score']=split_df['score'].map(lambda s:1 if int(s[0])>2 else 0)
        # split_df['score'] = np.where(split_df['score'] > 2, 1, 0)
        # 数据预览
        # split_df.head()
    
    #将数据分为输入数据和输出数据
    score = split_df.iloc[:, 0:1].values
    content = split_df.iloc[:, 1].values

    #划分训练集和测试集    
    score_tv, score_test, content_tv, content_test = train_test_split(score, content, test_size=0.1, random_state=20)
    #划分训练集和验证集
    score_train, score_val, content_train, content_val = train_test_split(score_tv, content_tv, test_size=0.1, random_state=20)


    test  = df_to_csv([s[0] for s in score_test], np.array(content_test).flatten(), os.path.join(out_path,'test.txt'))
    train = df_to_csv([s[0] for s in score_train], np.array(content_train).flatten(), os.path.join(out_path, 'train.txt'))
    val   = df_to_csv([s[0] for s in score_val], np.array(content_val).flatten(), os.path.join(out_path,'val.txt'))
    # print(train_2c[train_2c['score']>0].count())
    print('train:',str(train[train['score']>0].size/train.size))
    print('val:',str(val[val['score']>0].size/val.size))
    print('test:',str(test[test['score']>0].size/test.size))
    return train,val,test

# 获取分类类别和对应词典
# {4:"力荐",3:"推荐",2:"还行",1:"较差",0:"很差"}
def class_to_id(is_2c=True):
    classes = ['neg', 'pos']
    if not is_2c:
        classes = ["很差","较差","还行","推荐","力荐"]
    class_to_id = dict(zip(classes, range(len(classes))))
    return classes,class_to_id

# 构建词汇表
def build_word_dict(train_df, val_df, out_path, stopwords_path, vocab_size=0):
    # 读取停用词
    stopwords = [line.strip() for line in open(stopwords_path, encoding='utf-8').readlines()]
    df = [train_df["content"], val_df["content"]]
    max_sen_len = 0
    sum_sen_len = 0
    words = list()
    for contents in df:
        for content in contents:
            if content.strip():
                token = jieba.lcut(content.strip())
                token = [i for i in token if i not in stopwords and i.strip()]
                # print(token)
                sum_sen_len += len(token)
                max_sen_len = max(max_sen_len, len(token))
                for word in token:
                    words.append(word)
    print('>>> ave_sen_len:'+str(sum_sen_len/(len(df[0])+len(df[0]))))
    print('>>> max_sen_len:'+str(max_sen_len))
    # 统计词频
    word_counter = collections.Counter(words)
    if vocab_size!=0 and len(word_counter) > vocab_size-3:
        word_counter=word_counter.most_common(vocab_size-3)
    else:
        word_counter = word_counter.most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)
    vocab_size = len(word_dict)
    print('>>> vocab_size:'+str(vocab_size))

    with open(out_path, 'w', encoding='utf-8') as wd:
        for w in word_dict:
            wd.write(w+'\t')
            wd.write(str(word_dict[w]))
            wd.write('\n')
    return word_dict,max_sen_len,vocab_size

# 加载词汇表
def load_word_dict(word_dict_path):
    word_dict_df = pd.read_csv(word_dict_path, sep='\t', header=None, names=['word','id'], encoding='utf-8')
    word_dict = dict()
    for indexs in word_dict_df.index:
        word_dict[word_dict_df.loc[indexs].values[0]] = word_dict_df.loc[indexs].values[1]
    #print(word_dict)
    return word_dict
    

# 构建word2vec
def build_word2vec(fname, word_dict, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word_dict: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word_dict.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word_dict.keys():
        try:
            word_vecs[word_dict[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

def load_corpus_word2vec(path):
    """加载语料库word2vec词向量,相对wiki词向量相对较小"""
    word2vec = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = [float(w) for w in line.strip().split()]
            word2vec.append(sp)
    return np.asarray(word2vec)

# 加载语料库：train/val/test
# x为构成一条评论的词所对应的id。 y为one-hot表示pos-[0, 1], neg-[1, 0]
# {4:"力荐"-[0,0,0,0,1],3:"推荐"-[0,0,0,1,0],2:"还行"-[0,0,1,0,0],1:"较差"-[0,1,0,0,0],0:"很差"-[0,0,0,0,0]} 
def build_word_dataset(df_data, word_dict, stopwords_path, max_sen_len=870, num_class=2):
    # Shuffle dataframe
    df = df_data.sample(frac=1)
    # 读取停用词
    stopwords = [line.strip() for line in open(stopwords_path, encoding='utf-8').readlines()]
    x = [[j for j in jieba.lcut(df.iat[i,1].strip()) if j not in stopwords and str(j).strip()] for i in df.index]
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:max_sen_len], x))
    x = list(map(lambda d: d + (max_sen_len - len(d)) * [word_dict["<pad>"]], x))

    y = list(map(lambda d: [1 if i==d else 0 for i in range(num_class)], list(df["score"])))

    return x, y

def cut_dataset(df_data, stopwords_path):
    # 读取停用词
    stopwords = [line.strip() for line in open(stopwords_path, encoding='utf-8').readlines()]
    # 通过jieba分词
    def word_clean(content):
        return ' '.join([j for j in jieba.lcut(content.strip()) if j not in stopwords and j.strip()])
    df_data['content_cut'] = df_data.content.apply(word_clean)
    score = df_data.iloc[:, 0].values
    content = df_data.iloc[:, 2].values
    return content,score

def build_tfidvec_dataset(train_df, val_df, stopwords_path, max_df=0.95, min_df=2, num_class=2):
    train_content, train_score = cut_dataset(train_df, stopwords_path)
    val_content, val_score = cut_dataset(val_df, stopwords_path)
    # 数据的TF-IDF信息计算
    # sublinear_tf=True 时生成一个近似高斯分布的特征，可以提高大概1~2个百分点
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, smooth_idf=True, sublinear_tf=True)
    # 对数据训练
    train_vec_data = vectorizer.fit_transform(train_content)
    # 训练完成之后对验证数据转换
    val_vec_data = vectorizer.transform(val_content)
    n_dim = len(vectorizer.get_feature_names())
    print("关键词个数："+str(n_dim))
    return train_vec_data, train_score, val_vec_data, val_score, vectorizer


def load_tfidvec_dataset(df_data, stopwords_path, vectorizer):
    test_content, test_score = cut_dataset(df_data, stopwords_path)
    # 训练完成之后对测试数据转换
    test_vec_data = vectorizer.transform(test_content)
    return test_vec_data, test_score


def  batch_index(length, batch_size, is_shuffle=True):
    """
    生成批处理样本序列id.
    :param length: 样本总数
    :param batch_size: 批处理大小
    :param is_shuffle: 是否打乱样本顺序
    :return:
    """
    index = [idx for idx in range(length)]
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(np.ceil(length / batch_size))):
        yield index[i * batch_size:(i + 1) * batch_size]

if __name__ == "__main__":    
    # clear_comment('temp')
    # train,val,test = unique_divide('./comment.csv', out_path='', is_2c=True)
    # word_dict,max_sen_len,vocab_size = build_word_dict(train,val,"temp/word_dict.txt", './stopwords.txt')
    # word_vecs = build_word2vec('./wiki_word2vec_50.bin', word_dict, './word_vecs.txt')
    # x,y=build_word_dataset(train, word_dict, './stopwords.txt', max_sen_len=max_sen_len+1, num_class=2)
    # cla,class2id=class_to_id(is_2c=True)
    # print(class2id)
    
    train_df = pd.read_csv('./train.txt', sep='\t', header=None, names=["score", "content"], encoding='utf-8')
    val_df = pd.read_csv('./val.txt', sep='\t', header=None, names=["score", "content"], encoding='utf-8')
    #word_dict,max_sen_len,vocab_size = build_word_dict(train_df,val_df,"./word_dict.txt", './stopwords.txt')
    build_tfidvec_dataset(train_df, './stopwords.txt', max_df=0.95, min_df=2, num_class=2)
