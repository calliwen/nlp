#!/usr/bin/env python
#coding:utf-8
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

import numpy as np
import re
import jieba
jieba.load_userdict("./data/define_dict.txt")


def del_stop_words(word_str):
    stop_words = [ "了", "吗", "的",  "嘛",  "么", "哼", "？", "，", "。" ]
    res = ""
    for word in word_str.rstrip().split():
        if word not in stop_words:
            res += word + " "
    return res.rstrip()

def share_word(q1_list, q2_list):
    """ q1_list:  list type
        q2_list:  list type
        return :  share_word, list type
    """
    share_word = []
    if len(q1_list) <= len(q2_list) :
        for word in q1_list:
            if word in q2_list:
                share_word.append( word )
    else:
        for word in q2_list:
            if word in q1_list:
                share_word.append( word )
    return share_word

def all_word(q1_set, q2_set):
    return " ".join(q1_set|q2_set)


# 
def clever_participle(str_1, str_2):
    str_q1 = re.sub( r'[ ]+', ' ', " ".join( jieba.lcut( str_1, cut_all=False, HMM=True ) ) )
    str_q2 = re.sub( r'[ ]+', ' ', " ".join( jieba.lcut( str_2, cut_all=False, HMM=True ) ) )
    q1_list =  del_stop_words(str_q1).split()  
    q2_list =  del_stop_words(str_q2).split()  
    ###  求 share_word 时，最好按照 最少词的 question中对应词的顺序排列。可能会用到 share_word 中的顺序信息。
    share_word_1 = share_word(q1_list, q2_list)
    if len(q1_list) < len(q2_list) :
        for word in q1_list:
            jieba.suggest_freq(word, tune=True)
        q2_list_tmp = set( jieba.lcut( str_2, cut_all=False, HMM=True )  )
        share_word_2 = share_word(q1_list, q2_list_tmp)
        for word in q1_list:
            jieba.suggest_freq(word, tune=False)
        if len(share_word_1) >= len(share_word_2):
            return q1_list, q2_list, share_word_1
        else:
            return q1_list, q2_list_tmp, share_word_2
    else:
        for word in q2_list:
            jieba.suggest_freq(word, tune=True)
        q1_list_tmp = set( jieba.lcut( str_1, cut_all=False, HMM=True )  )
        share_word_3 = share_word(q1_list_tmp, q2_list)
        for word in q2_list:
            jieba.suggest_freq(word, tune=False)
        if len(share_word_1) >= len(share_word_3):
            return q1_list, q2_list, share_word_1
        else:
            return q1_list_tmp, q2_list, share_word_3

# participle word for words

def participle_train(source_file, target_file):
    q1_part = []
    q2_part = []
    for line in source_file.readlines():
        line_list = line.rstrip().split('\t')
        """
            id = line_list[0]
            q1 = line_list[1]
            q2 = line_list[2]
            label = line_list[3]
        """
        if len(line_list)<3:
            continue
        elif len(line_list)==3:
            print( " len : 3 " )
        q1_p_list, q2_p_list, share_list = clever_participle(line_list[1], line_list[2])
        q1_p_str = del_stop_words( " ".join(q1_p_list) )
        q2_p_str = del_stop_words( " ".join(q2_p_list) )

        if len(share_list)==0:
            share_str = str("Miss")
        else:
            share_str = del_stop_words( " ".join(share_list) )
        q1_part.append(q1_p_str)
        q2_part.append(q2_p_str)
        #   id+'\t'+q1-part+'\t'+q2-part+'\t'+share-part+'\t'+label+'\n'
        # write_str for test
        write_str = line_list[0]+'\t'+q1_p_str+'\t'+q2_p_str+'\t'+share_str+'\t'+line_list[3]+'\n'
        target_file.write( write_str  )
    source_file.close()
    target_file.close()
    return q1_part, q2_part


def participle_test(source_file, target_file):
    q1_part = []
    q2_part = []

    index = 0
    for line in source_file.readlines():
        index +=1
        if index<20:
            print( index )
            # print( line )
        line_list = line.rstrip().split('\t')
        """
            id = line_list[0]
            q1 = line_list[1]
            q2 = line_list[2]
            label = line_list[3]
        """
        # if len(line_list)==4:
        #     pass
        #     print(" len : 4 ")
        q1_p_list, q2_p_list, share_list = clever_participle(line_list[1], line_list[2])
        q1_p_str = del_stop_words( " ".join(q1_p_list) )
        q2_p_str = del_stop_words( " ".join(q2_p_list) )
        if len(share_list)==0:
            share_str = str("Miss")
        else:
            share_str = del_stop_words( " ".join(share_list) )
        q1_part.append(q1_p_str)
        q2_part.append(q2_p_str)
        #   id+'\t'+q1-part+'\t'+q2-part+'\t'+share-part+'\t'+label+'\n'
        # write_str for test
        write_str = line_list[0]+'\t'+q1_p_str+'\t'+q2_p_str+'\t'+share_str+'\t'+'\n'
        target_file.write( write_str  )
    source_file.close()
    target_file.close()
    return q1_part, q2_part


path = str("./data/")

train = open( path + "atec_nlp_sim_train.csv", 'r+')
train_part_file = open( path + "train_participle_word.txt", 'w+')
train_q1_part, train_q2_part = participle_train(train, train_part_file)

# python main.py INPUT_PATH OUTPUT_PATH
# sys.argv[0] : './main.py'
# sys.argv[1] : 'INPUT_PATH'
# sys.argv[2] : 'OUTPUT_PATH'
INPUT_PATH = sys.argv[1]  # Predict_data

INPUT_PATH = INPUT_PATH.rstrip('\r')
# print( "\n" )
# print( "\n" )
# print( "\n" )
print( "----INPUT_PATH: ",INPUT_PATH )
# print( "\n" )
# print( "\n" )
# print( "\n" )

test = open( INPUT_PATH, 'r+' )
test_part_file = open(path + "test_participle_word.txt", 'w+')
test_q1_part, test_q2_part = participle_test(test, test_part_file)
print( "participle done!" )


# Word2Vec
from gensim.models import Word2Vec
texts = np.hstack( [ train_q1_part, train_q2_part, test_q1_part, test_q2_part ] )

texts =  [ [ word for word in doc.rstrip().split() ] for doc in texts  ]
model = Word2Vec(texts, size=100, window=3, min_count=0, workers=4)
model_name = str("./data/model/train_model_size100.bin")
model.save(model_name)
del model
print( "Word2Vec done!" )
