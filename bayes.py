import numpy as np
from numpy import *
import operator
#feedparser的安装，直接在anaconda prompt中输入conda install feedparser
#词表转为向量函数
def load_dataset():
    #邮件中可能含有的词汇
    posting_List=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #分类向量,1表示侮辱性文字，0表示正常言论
    class_vectors = [0, 1, 0, 1, 0, 1]
    return posting_List,class_vectors

def createvocablist(dataset):
    #创建一个空的词汇集合
    vocab_set = set([])
    for doucument in dataset:
        vocab_set = vocab_set | set(doucument)#按位与，创建两个的并集
    return list(vocab_set)

#词集模型，将每一个词的出现与否作为一个特征，每个词只能出现一次
def setofwords_vector(vocab_list, input_set):#字符转向量
    return_vector = [0]*len(vocab_list)#创建一个vocab_list长度的0列表
    #遍历input_set列表，如果Word在其中就将其设为1
    for word in input_set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] = 1
        else:
            print("The world:%s is not in my vocabulary!" % word)
        
    return return_vector

#词袋模型bayes，每个词可以出现多次
def bag_bayes(vocab_list,input_set):
    return_vector = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list: 
            #增加遇到的单词的对应值
            return_vector[vocab_list.index(word)] += 1
    return return_vector

#朴素贝叶斯分类器训练函数
def train_bayes(train_matrix,train_category):
    #训练数据的组数
    number_train = len(train_matrix)
    #这些训练数据所组成的词汇的单词量
    number_words = len(train_matrix[0])
    #侮辱性种类和比数据总量，train_category里面只含有0或1，所以求和相加的实际上只有1
    p_abusive = sum(train_category)/float(number_train)
    #防止其中有个类别的概率为0和防止下溢，将zeros换为ones分母设为2
    p0_num = ones(number_words)#初始化非侮辱性文档的字数
    p1_num = ones(number_words)#初始化侮辱性文档的字数
    #初始化分母
    p0_total = 2.0
    p1_total = 2.0

    for i in range(number_train):
        #遍历number_train中所有文档，如果出现含有侮辱性词汇的文档，p1向量数加1，在所有文档中侮辱性总词汇加1
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_total += sum(train_matrix[i])
        #相反就将非侮辱性词汇加1，和所有文档中的非侮辱性总词汇加1
        else:
            p0_num += train_matrix[i]
            p0_total +=sum(train_matrix[i])
    #计算侮辱性词汇出现的概率
    p1_vector = log(p1_num/p1_total)
    #计算非侮辱性词汇出现的概率
    p0_vector = log(p0_num/p0_total)

    return p0_vector, p1_vector,p_abusive

#朴素贝叶斯分类函数
def classify_bayes(vector_classify, p0_vector, p1_vector,p_class):

    p1 = sum(vector_classify * p1_vector) + log(p_class)
    p0 = sum(vector_classify * p0_vector) + log(1-p_class)
    if p1 > p0:
        return 1
    else:
        return 0


#用正则表达式分隔邮件
def text_parse(bigString):#接受一个大字符串并将其转换为字符串列表
    import re
    list_token = re.split(r'\W*', bigString)
    return [tok.lower() for tok in list_token if len(tok) > 2]#去掉少于两位字符的字符串

#过滤垃圾邮件
def spam_test():
    doc_list = []; class_list = []; full_text = []
    for i in range(1,26):#遍历spam
       #将每条邮件都分隔分解开 
        word_list = text_parse(open('email/spam/%d.txt' % i, "rb").read().decode('GBK','ignore'))#修改，将一些不符合的标志忽略掉
        doc_list.append(word_list)#逐个增加分解开的邮件
        full_text.extend(word_list)#将doc_list列表加到full_text中
        class_list.append(1)#分类列表加1
        #将ham也解析
        word_list =  text_parse(open('email/spam/%d.txt' % i, "rb").read().decode('GBK','ignore'))#修改，将一些不符合的标志忽略掉
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = createvocablist(doc_list)#将所有邮件的文字合并成一个不重复的列表
    #训练列表1-49
    training_set = list(range(50))#python3以后要采用list()创建列表
    test_set = []
    #选取10个邮件用于测试
    for i in list(range(10)):
        #随机选取10个邮件，并将其加到test_set列表中
        rand_index = int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_index])
        #删掉原列表中被选取的邮件
        del(training_set[rand_index])
    
    training_matrix = []
    training_classes = []

    for doc_index in training_set:#training_set还有40封邮件
        #将字符转为向量
        training_matrix.append(bag_bayes(vocab_list,doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    p0v,p1v,p_spam = train_bayes(array(training_matrix),array(training_classes))
    
    #初始化错误率
    error_count = 0

    for doc_index in test_set:
        word_vector = bag_bayes(vocab_list,doc_list[doc_index])
        if classify_bayes(array(word_vector),p0v,p1v,p_spam) != class_list[doc_index]:
            error_count += 1
    print("classification error",doc_list[doc_index])
    print('the error rate is: ',float(error_count)/len(test_set))

#RSS源分类器以及高频词去除函数
#统计词汇表里每个词出现的次数
def cal_freq_words(vocab_list,full_text):
    #初始化字典
    frequency_dict = {}
    for token in vocab_list:
        frequency_dict[token] = full_text.count(token)#.count()用于统计某个字符在字符串里出现的次数
    #排序
    sorted_frequent_words = sorted(frequency_dict.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_frequent_words[:30]

def local_words(feed1,feed0):#使用两个RSS源作为参数
    #初始化参数
    import feedparser

    doc_list = []; class_list = []; full_text = []
    min_len = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(min_len):
        #切分文本
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    #创建一个合并的词汇表
    vocab_list = createvocablist(doc_list)
    #前30频率最高的词汇
    top30_words = cal_freq_words(vocab_list,full_text)
    #移除vocab_list里的前30个高频词
    for words in top30_words:
        if words[0] in vocab_list:
            vocab_list.remove(words[0])
    
    training_set =list(range(2*min_len))
    test_set = []

    #在训练集中随机选取20个数用于测试，并删掉
    for i in list(range(20)):
        rand_index = int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    training_matrix = []
    training_classes = []
    #将词汇转为向量
    for doc_index in training_set:
        training_matrix.append(bag_bayes(vocab_list,doc_list[doc_index]))
        training_classes.append(class_list[doc_index])
    #朴素叶贝斯分类器训练函数
    p0v,p1v,p_spam = train_bayes(array(training_matrix),array(training_classes))
    error_count = 0#初始化错误率
    #遍历测试表进行测试
    for doc_index in test_set:
        word_vector = bag_bayes(vocab_list,doc_list[doc_index])
        if classify_bayes(array(word_vector),p0v,p1v,p_spam) != class_list[doc_index]:
            error_count += 1
    print("the error rate is:",float(error_count)/len(test_set))
    return vocab_list,p0v,p1v

