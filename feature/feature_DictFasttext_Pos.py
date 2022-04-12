#%%
import os
import sys
import sklearn_crfsuite
import random
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as psg
import nltk
import io

from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from tqdm import tqdm
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from gensim . models import FastText
from gensim.models import word2vec
import joblib
# %%
def loadInputFile(path):
    trainingset = list()  # store trainingset [content,content,...]
    # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    position = list()
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            # annot= article_id, start_pos, end_pos, entity_text, entity_type
            annot = annot.split('\t')
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions
# %%

def CRFFormatData(trainingset, position, path):
    if (os.path.isfile(path)):
        os.remove(path)
    outputfile = open(path, 'a', encoding='utf-8')

    # output file lines
    count = 0  # annotation counts in each content
    tagged = list()
    for article_id in range(len(trainingset)):
        trainingset_split = list(trainingset[article_id])
        while '' or ' ' in trainingset_split:
            if '' in trainingset_split:
                trainingset_split.remove('')
            else:
                trainingset_split.remove(' ')
        start_tmp = 0
        for position_idx in range(0, len(position), 5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type = position[position_idx+4]
                    if start_pos == 0:
                        token = list(
                            trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    else:
                        token = list(trainingset[article_id][0:start_pos])
                        whole_token = trainingset[article_id][0:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue

                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                        token = list(
                            trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type = position[position_idx+4]
                    if start_pos < start_tmp:
                        continue
                    else:
                        token = list(
                            trainingset[article_id][start_tmp:start_pos])
                        whole_token = trainingset[article_id][start_tmp:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                    token = list(trainingset[article_id][start_pos:end_pos])
                    whole_token = trainingset[article_id][start_pos:end_pos]
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(' ', '')) == 0:
                            continue
                        # BIO states
                        if token[0] == '':
                            if token_idx == 1:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                        else:
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type

                        output_str = token[token_idx] + ' ' + label + '\n'
                        outputfile.write(output_str)
                    start_tmp = end_pos

        token = list(trainingset[article_id][start_tmp:])
        whole_token = trainingset[article_id][start_tmp:]
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(' ', '')) == 0:
                continue

            output_str = token[token_idx] + ' ' + 'O' + '\n'
            outputfile.write(output_str)

        count = 0

        output_str = '\n'
        outputfile.write(output_str)
        ID = trainingset[article_id]

        if article_id % 10 == 0:
            print('Total complete articles:', article_id)
        # if article_id == 120 :
        #     break   

    # close output file
    outputfile.close()
# %%

def CRF(x_train, y_train, x_test, y_test):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=500,
        all_possible_transitions=True,
        verbose=True
    )
    crf.fit(x_train, y_train)
    joblib.dump(crf, "crf_1227.pkl")

    y_pred = crf.predict(x_test)
    y_pred_mar = crf.predict_marginals(x_test)

    labels = list(crf.classes_)
    labels.remove('O')
    f1score = metrics.flat_f1_score(
        y_test, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(labels, key=lambda name: (
        name[1:], name[0]))  # group B and I results
    print(flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3))
    return y_pred, y_pred_mar, f1score
# %%
# 斷詞
def Token(sentence_list):
    # Download data
    # data_utils.download_data("./")
    
    # Load model without GPU
    ws = WS("./data")

    word_sentence_list = ws(sentence_list)
    # print(word_sentence_list)
    del ws
    return word_sentence_list

# %%
def Word2VectorModel(data_list):
    sentences = data_list
    # sentences = traindata_list
    # train model
    model = FastText( sentences , size =300 , window =3 , min_count = 1 , iter = 500, min_n =3, max_n =6, word_ngrams =0)
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    model.save('model/fasttext.model')
    return model,words
# %%
def Dataset(data_path):
    r"""
    load `train.data` and separate into a list of labeled data of each text
    return:
    data_list: a list of lists of tuples, storing tokens and labels (wrapped in tuple) of each text in `train.data`
    traindata_list: a list of lists, storing training data_list splitted from data_list
    testdata_list: a list of lists, storing testing data_list splitted from data_list
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)

    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line,
    # and generate data_list of `train data` and data_list of `development/test data` by this function
    traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = train_test_split(data_list,
                                                                                                          article_id_list,
                                                                                                          test_size=0.33,
                                                                                                          random_state=42)

    return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list

# %%
def Word2Vector(data_list, embedding_dict):
    r"""
    look up word vectors
    turn each word into its pretrained word vector
    return a list of word vectors corresponding to each token in train.data
    """
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector = np.random.rand(*(list(embedding_dict.values())[0].shape))

    for idx_list in range(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0]  # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)
    return embedding_list

# def Word2Vector(data_list, embedding_dict):
#     embedding_list = list()
#     couut = 0
#     # No Match Word (unknown word) Vector in Embedding
#     unk_vector=np.random.rand(100)
#     record = False
#     for idx_list in range(len(data_list)):
#         embedding_list_tmp = list()
#         for idx_tuple in range(len(data_list[idx_list])):
#             key = data_list[idx_list][idx_tuple][0] # token

#             if record == True:
#                 record = False
#             elif key in embedding_dict:
#                 value = embedding_dict[key]
#             else:
#                 key = data_list[idx_list][idx_tuple][0]+data_list[idx_list][idx_tuple+1][0]
#                 if key in embedding_dict:
#                     value = embedding_dict[key] 
#                     record = True
#                 else:
#                     # print('nnn')
#                     couut+=1
#                     value = unk_vector
#             embedding_list_tmp.append(value)
#         embedding_list.append(embedding_list_tmp)
#         # print(couut)
#     return embedding_list

# %%
def Feature(embed_list):
    r"""
    input features: pretrained word vectors of each token
    return a list of feature dicts, each feature dict corresponding to each token
    """
    feature_list = list()
    for idx_list in range(len(embed_list)):
        feature_list_tmp = list()
        for idx_tuple in range(len(embed_list[idx_list])):
            feature_dict = dict()
            for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                feature_dict['dim_' + str(idx_vec+1)
                             ] = embed_list[idx_list][idx_tuple][idx_vec]
            feature_list_tmp.append(feature_dict)
        feature_list.append(feature_list_tmp)
    return feature_list


def Preprocess(data_list):
    r"""
    Get the labels of each tokens in train.data.
    Return a list of lists of labels.
    """
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    return label_list

# %%
def load_word_vector():
    # Load pretrained word vectors.
    # Get a dict of tokens (key) and their pretrained word vectors (value).
    # Pretrained word2vec CBOW word vector: https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
    # dim = 0
    # word_vecs = {}
    # # Open pretrained word vector file.
    # with open('./baseline/cna.cbow.cwe_p.tar_g.512d.0.txt',encoding="utf-8") as f:
    #     for line in f:
    #         tokens = line.strip().split()

    #         # There 2 integers in the first line: vocabulary_size, word_vector_dim.
    #         if len(tokens) == 2:
    #             dim = int(tokens[1])
    #             continue

    #         word = tokens[0]
    #         vec = np.array([float(t) for t in tokens[1:]])
    #         word_vecs[word] = vec
    fname = 'D:/Alia/Downloads/109-1/aidea_data/testfast/cc.zh.300.vec/cc.zh.300.vec'
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    word_vecs = {}
    print(n,d)
    # print(type(fin))
    count=0
    for line in fin:
        tokens = line.rstrip().split(' ')
        vec = np.array([float(t) for t in tokens[1:]])
        word_vecs[tokens[0]] = vec
        count+=1
        if count==200000:
            break
    # return data
    print('vocabulary_size: ', len(word_vecs), ' word_vector_dim: ', vec.shape)
    return word_vecs

# %%
def add_Feature(old_feature, add_feature, feature_name):
    new_feature = old_feature
    for i in range(len(old_feature)):
        for k in range(len(old_feature[i])):
            new_feature[i][k][feature_name] = add_feature[i][k]
    return new_feature
# %%

def get_pos_feature(train_data):
    seg_list = [psg.cut(i) for i in train_data]
    pos_feature = []
    for dialogue_pos in seg_list:
        pos_feature.append([])
        for word, pos in dialogue_pos:
            pos_feature[-1].extend([pos] * len(word))
    return pos_feature


def get_pos_feature_uni(train_data):
    seq_list = []
    for dia in train_data:
        seq_list.append([])
        for word in dia:
            seq_list[-1].append(word)
    pos_tag = []
    for i in seq_list:
        pos_tag.append(nltk.pos_tag(i))
    pos_feature = []
    for dia in pos_tag:
        pos_feature.append([])
        for item in dia:
            pos_feature[-1].append(item[1])
    return pos_feature

# %%

def FormatOutput(y_pred):
    r"""
    Format data.
    """
    output = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id in range(len(y_pred)):
        pos = 0
        start_pos = None
        end_pos = None
        entity_text = None
        entity_type = None
        for pred_id in range(len(y_pred[test_id])):
            if y_pred[test_id][pred_id][0] == 'B':
                start_pos = pos
                entity_type = y_pred[test_id][pred_id][2:]
            elif start_pos is not None and y_pred[test_id][pred_id][0] == 'I' and y_pred[test_id][pred_id+1][0] == 'O':
                end_pos = pos
                entity_text = ''.join([testdata_list[test_id][position][0]
                                       for position in range(start_pos, end_pos+1)])
                line = str(testdata_article_id_list[test_id])+'\t'+str(
                    start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
                output += line+'\n'
            pos += 1
    output_path = 'output.tsv'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output

# %%

def Dataset_dev(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data=f.readlines()#.encode('utf-8').decode('utf-8-sig')
    data_list = list()
    data_list_tmp =list()
    article_id_list=list()
    idx=0
    
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx+=1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)
    
    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line, 
    # and generate data_list of `train data` and data_list of `development/test data` by this function
   # traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list=train_test_split(data_list,
                                                                                                    #article_id_list,
                                                                                                    #test_size=0.33,
                                                                                                   # random_state=42)
    
    return data_list,article_id_list#, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list


# %%
if __name__ == '__main__':
    # Set random seed.
    np.random.seed(25)
    random.seed(25)

    # Load data.
    file_path =  './train_1_update.txt' #'./train_2.txt' #'./train_腳色統一_120.txt'  # #SampleData_deid.txt
    trainingset, position, mentions = loadInputFile(file_path)

    # Format data.
    data_path = './sample.data'
    CRFFormatData(trainingset, position, data_path)

    # Load pretrained word vector.
    word_vecs = load_word_vector()

    # cut_trainingset = Token(trainingset)
    # w2v_model,words = Word2VectorModel(cut_trainingset)

    # Load formated data and split data.
    data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = Dataset(
        data_path)

    # Load Word Embedding
    # trainembed_list = Word2Vector(traindata_list, w2v_model)
    # testembed_list = Word2Vector(testdata_list, w2v_model) 
    trainembed_list = Word2Vector(traindata_list, word_vecs)
    testembed_list = Word2Vector(testdata_list, word_vecs)
    del word_vecs

    # CRF - Train Data (Augmentation Data)
    x_train = Feature(trainembed_list)
    y_train = Preprocess(traindata_list)

    # CRF - Test Data (Golden Standard)
    x_test = Feature(testembed_list)
    y_test = Preprocess(testdata_list)

    # Load untokenization train/test data.
    origin_traindata = [trainingset[i] for i in traindata_article_id_list]
    origin_testdata = [trainingset[i] for i in testdata_article_id_list]

    # Add POS feature.
    # pos_feature = get_pos_feature_uni(origin_traindata)
    pos_feature = get_pos_feature(origin_traindata)
    x_train = add_Feature(x_train, pos_feature, 'POS')

    pos_feature = get_pos_feature(origin_testdata)
    x_test = add_Feature(x_test, pos_feature, 'POS')

    # Train model.
    y_pred, y_pred_mar, f1score = CRF(x_train, y_train, x_test, y_test)

    # Print f1score
    print(f1score)

    # Format output.
    # output = FormatOutput(y_pred)
    # print(output)

# %%
input_path = 'test.txt' #'development_2.txt'
output_path = 'test.data'

fi = open(output_path,"w", encoding='utf8')
with open(input_path,"r", encoding='utf8') as f:
    file_text=f.read()#.encode('utf-8').decode('utf-8-sig')
articles=file_text.split('\n\n--------------------\n\n')[:-1]
origin_testdata = []
for each_article in articles:
     #words = each_article[1].split("")
     #for word in each_article[1]:
     words = each_article.split('\n')
     origin_testdata.append(words[1])
     for word in words[1]:
            fi.write(word)
            fi.write(" O")
            fi.write('\n')
     fi.write('\n')  
f.close()
fi.close()

# %%
word_vecs = load_word_vector()
data_list,article_id_list = Dataset_dev(output_path)
test_emb_list = Word2Vector(data_list,word_vecs)  #Word2Vector(traindata_list, w2v_model)
del word_vecs
x_test = Feature(test_emb_list)
# %%
pos_feature = get_pos_feature(origin_testdata)
x_test = add_Feature(x_test, pos_feature, 'POS')
# print(data_list[0][0])
# %%

crf = joblib.load("crf_1227.pkl")
y_pred = crf.predict(x_test)
y_pred_mar = crf.predict_marginals(x_test)
# print(y_pred)
# %%
output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
for test_id in range(len(y_pred)):
    pos=0
    start_pos=None
    end_pos=None
    entity_text=None
    entity_type=None
    for pred_id in range(len(y_pred[test_id]) - 1):
        if y_pred[test_id][pred_id][0]=='B':
            start_pos=pos
            entity_type=y_pred[test_id][pred_id][2:]
        elif start_pos is not None and y_pred[test_id][pred_id][0]=='I' and y_pred[test_id][pred_id+1][0]=='O':
            end_pos=pos
            entity_text=''.join([data_list[test_id][position][0] for position in range(start_pos,end_pos+1)])
            line=str(article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)+'\t'+entity_text+'\t'+entity_type
            output+=line+'\n'
        pos+=1
# %%
output_path='output_fasttext_1227.tsv'
with open(output_path,'w',encoding='utf-8') as f:
    f.write(output)
print(output)


# %%
