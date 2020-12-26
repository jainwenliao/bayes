import bayes
from numpy import *
import sys, os, feedparser
import feedparser


ny = feedparser.parse('http://newyork.craigslist.org/org/stp/index.rss')
print(ny)
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

vocab_list,psf,pny = bayes.local_words(ny,sf)
print(vocab_list,psf,pny)
#封装函数
def testing_bayes():
    list_posts, list_classes = bayes.load_dataset()#加载训练库里的文档和标签
    myvocablist = bayes.createvocablist(list_posts)#将所有文档保存为不重复的列表
    #创建一个空集合
    train_matrix = []
    for posting_doc in list_posts:
        train_matrix.append(bayes.setofwords_vector(myvocablist,posting_doc))

    p0v,p1v,pab = bayes.train_bayes(array(train_matrix),array(list_classes))
    test_entry = ['love','my','dalmation']
    this_doc = array(bayes.setofwords_vector(myvocablist,test_entry))
    print(test_entry,"classifiedd as: ", bayes.classify_bayes(this_doc,p0v,p1v,pab))
    test_entry = ['stupid','garbage']
    this_doc = array(bayes.setofwords_vector(myvocablist,test_entry))
    print(test_entry,"classifiedd as: ", bayes.classify_bayes(this_doc,p0v,p1v,pab))


