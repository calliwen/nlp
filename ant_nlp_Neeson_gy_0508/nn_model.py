#!/usr/bin/env python
# coding:utf-8
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, concatenate, multiply, Merge, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers.recurrent import GRU, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.callbacks import Callback
from gensim.models import Word2Vec
import gc  


def get_train_data( source_file ):
    id_list = []
    q1_part = []
    q2_part = []
    share = []
    label = []
    all_lines = source_file.readlines()
    for line in all_lines:
        line_list = line.rstrip().split('\t')
        if len(line_list)<4:
            continue
        #   id+'\t'+q1-part+'\t'+q2-part+'\t'+share-part+'\t'+label+'\n'
        """
            id = line_list[0]
            q1 = line_list[1]
            q2 = line_list[2]
            share = line_list[3]
            label = line_list[4]
        """
        id_list.append(line_list[0])
        q1_part.append(line_list[1])
        q2_part.append(line_list[2])
        share.append(line_list[3])
        label.append( int(line_list[4]) ) 
    source_file.close()
    print( "id_list len: ", len(id_list) )
    print( "q1_part len: ", len(q1_part) )
    print( "q2_part len: ", len(q2_part) )
    print( "share len: ", len(share) )
    print( "label len: ", len(label) )
    
    return id_list, q1_part, q2_part, share, label


def get_test_data( source_file ):
    id_list = []
    q1_part = []
    q2_part = []
    share = []
    all_lines = source_file.readlines()
    index = 1
    for line in all_lines:
        line_list = line.rstrip().split('\t')
        if len(line_list)<=3:
            print( " <=3  " )
            print( str(index)+ line )
            # continue
        if index <5:
            print( len(line_list) )
        index +=1
        #   id+'\t'+q1-part+'\t'+q2-part+'\t'+share-part+'\t'+label+'\n'
        """
            id = line_list[0]
            q1 = line_list[1]
            q2 = line_list[2]
            share = line_list[3]
            label = line_list[4]
        """

        id_list.append(line_list[0])
        q1_part.append(line_list[1])
        q2_part.append(line_list[2])
        share.append(line_list[3])
    source_file.close()
    print( "id_list len: ", len(id_list) )
    print( "q1_part len: ", len(q1_part) )
    print( "q2_part len: ", len(q2_part) )
    print( "share len: ", len(share) )
    
    return id_list, q1_part, q2_part, share




def pad2seq(train, test, max_len=12 ):
    tokenizer = Tokenizer(num_words=None, filters='？?')
    list_sentences_train = np.hstack( [ train[1], train[2] ] )
    tokenizer.fit_on_texts(list(list_sentences_train))
    
    train_q1_seq = tokenizer.texts_to_sequences(train[1])
    train_q2_seq = tokenizer.texts_to_sequences(train[2])
    train_share_seq = tokenizer.texts_to_sequences( [ str(doc) for doc in train[3]])
    
    train_q1_pad_seq = pad_sequences( train_q1_seq, maxlen=max_len )
    train_q2_pad_seq = pad_sequences( train_q2_seq, maxlen=max_len )
    train_share_pad_seq = pad_sequences( train_share_seq, maxlen = max_len )
    
    test_q1_seq = tokenizer.texts_to_sequences( test[1]  )
    test_q2_seq = tokenizer.texts_to_sequences( test[2] )
    test_share_seq = tokenizer.texts_to_sequences( [ str(doc) for doc in test[3] ] )
    
    test_q1_pad_seq = pad_sequences( test_q1_seq, maxlen=max_len )
    test_q2_pad_seq = pad_sequences( test_q2_seq, maxlen=max_len )
    test_share_pad_seq = pad_sequences( test_share_seq, maxlen = max_len )
    
    print( "train_q1_pad_seq len: ", len(train_q1_pad_seq)  )
    print( "train_q2_pad_seq len: ", len(train_q2_pad_seq)  )
    print( "train_share_pad_seq len: ", len(train_share_pad_seq)  )

    print( "test_q1_pad_seq len: ", len(test_q1_pad_seq) )
    print( "test_q2_pad_seq len: ", len(test_q2_pad_seq) )
    print( "test_share_pad_seq len: ", len(test_share_pad_seq) )
    return train_q1_pad_seq, train_q2_pad_seq, train_share_pad_seq,  \
            test_q1_pad_seq, test_q2_pad_seq, test_share_pad_seq, tokenizer
    

def loadEmbeddingMatrix( tokenizer, model, vec_size=100 ):
    embeddings_index = dict()
    for word in model.wv.vocab:
        embeddings_index[word] = model.wv[word]
    # print( "Loaded %s wrod vectors." %( len( embeddings_index ) )   )
    embed_size = vec_size
    nb_words = len( tokenizer.word_index )  
    # print( "nb_words: ", nb_words )    # nb_words:  5798
    # nb_words = len(model.wv.vocab)  # 7979
    gc.collect()
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
    gc.collect()
    embeddedCount = 0
    for word, i in tokenizer.word_index.items():
        i -= 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
            embeddedCount+=1
    # print('total embedded:',embeddedCount,'common words')
    del(embeddings_index)
    gc.collect()
    return embedding_matrix


def q1_q2_share_rnn_model():
    input_q1 = Input(shape=[max_len])
    input_q2 = Input(shape=[max_len])
    input_share = Input(shape=[max_len])
    embedding_layer = Embedding( len(tokenizer.word_index)+1, 
                                  embedding_matrix.shape[1], 
                                  weights=[embedding_matrix], 
                                  trainable=False
                                )
    q1 = Bidirectional(LSTM(128))(embedding_layer(input_q1) ) 
    q2 = Bidirectional(LSTM(128))(embedding_layer(input_q2) )
    share = Bidirectional(LSTM(128))(embedding_layer(input_share) )
    res = multiply( [q1, q2, share] )
    res = BatchNormalization()(res)
    
    dense1_layer = Dense( 16, activation='relu', name='dense_1' )(res)
    
    output = Dense( 1, activation='sigmoid', name='output_layer' )(dense1_layer)
    model = Model(inputs=[input_q1, input_q2, input_share], outputs=output)
    print( model.summary() )
    return model

path = str("./data/")
train_part_file = open( path + "train_participle_word.txt", 'r+')
test_part_file = open( path + "test_participle_word.txt", 'r+')
train_id, train_q1, train_q2, train_share, train_label = get_train_data( train_part_file )
test_id, test_q1, test_q2, test_share = get_test_data( test_part_file )

x_train = np.vstack( [ train_id, train_q1, train_q2, train_share ] )
y_train = train_label

x_test = np.vstack( [test_id, test_q1, test_q2, test_share] )

# Padding sequences
max_len = 12
train_q1_pad_seq, train_q2_pad_seq, train_share_pad_seq, test_q1_pad_seq, test_q2_pad_seq, test_share_pad_seq, tokenizer \
    = pad2seq( x_train, x_test, max_len=max_len )

# load Word2Vec model
model_name = str("./data/model/train_model_size100.bin") 
model = Word2Vec.load(model_name)
embedding_matrix = loadEmbeddingMatrix( tokenizer, model, vec_size=100 )

# Train rnn_model
model = q1_q2_share_rnn_model()
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['acc'])
model.fit(  [ train_q1_pad_seq, train_q2_pad_seq, train_share_pad_seq ], 
            y_train, 
            batch_size=32, 
            epochs=1,
            verbose=2 )

# Get the threshold
pre = model.predict( [ train_q1_pad_seq, train_q2_pad_seq, train_share_pad_seq ], batch_size=32, verbose=2)
top =  sum(y_train)
import heapq
nlarg = heapq.nlargest(top, pre)
threshold = min(nlarg)[0]
print("threshold: ", threshold )

# Predict
res_pred = model.predict( [ test_q1_pad_seq, test_q2_pad_seq, test_share_pad_seq ], batch_size=32, verbose=2)
print( len(res_pred) )
res = [ 1 if val >= threshold else 0 for val in res_pred ]

# Write result
Index = x_test[0]

# python main.py INPUT_PATH OUTPUT_PATH
# sys.argv[0] : './main.py'
# sys.argv[1] : 'INPUT_PATH'
# sys.argv[2] : 'OUTPUT_PATH'
OUTPUT_PATH = sys.argv[1]
OUTPUT_PATH = OUTPUT_PATH.rstrip('\r')


print( len(Index) )
print( len(res) )
with open( OUTPUT_PATH, "w" ) as f:
    for i, pre in zip( list(Index), list(res) ):
        f.write( str(i)+"\t"+str(pre)+"\n" )

