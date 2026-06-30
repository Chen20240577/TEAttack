import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

root = '../'
k = 30


def get_topk_index(k, arr):
    top_k = k
    array = arr
    top_k_index = array.argsort()[::-1][0:top_k]
    return top_k_index  # return top-k use list[]


print('read embedding....')

word2vec = Word2Vec.load(root + 'word2vec_model/node_w2v_code_64.model').wv
vocab = word2vec.vocab
max_token = word2vec.vectors.shape[0]
embedding_dim = word2vec.vectors.shape[1]
embeddings = np.zeros((max_token + 1, embedding_dim))
embeddings[:max_token] = word2vec.vectors
print('end read embedding..')

print('read all var and extract embedding..')
data_all_var = pd.read_pickle(root + 'datasets/var_for_allCode.pkl')
all_var_list = list(data_all_var['all vars'].tolist()[0])
max_var = len(all_var_list)
embeddings_allvar = np.zeros((max_var, embedding_dim))
index_all_var = []
for item in all_var_list:
    if item in vocab:
        index_all_var.append(vocab[item].index)
    else:
        index_all_var.append(max_token)

for i in range(max_var):
    embeddings_allvar[i] = embeddings[index_all_var[i]]
print('read and extract end')

print('read every var and formalparameter for every code')
data_every_var = pd.read_pickle(root + 'datasets/var_for_everyCode.pkl')
every_var_list = data_every_var['variable'].tolist()

# formalParameter
formalParameter_for_every_code = pd.read_pickle \
    (root + 'datasets/formalParameter_for_everyCode.pkl')
formalParameter_list = formalParameter_for_every_code['formal_parameters'].tolist()

print('select top k nearest var')
nearest_list = []
var_embed = np.zeros((1, embedding_dim))
count = 0
for every_code in every_var_list:
    nearest_dict = {}
    mask_index_list = []
    formalParameter_every_code = formalParameter_list[count]
    count = count + 1
    print('ok' + str(count))
    for var in every_code:
        if var in all_var_list:
            var_index_in_all_var = all_var_list.index(var)
            mask_index_list.append(var_index_in_all_var)
    if formalParameter_every_code != []:
        for item in formalParameter_every_code:
            if item in all_var_list:
                mask_index_list.append(all_var_list.index(item))

    allcan_list = []

    print("目标:{}个替换词".format(k))
    for var in every_code:
        n_list = []
        var_index = vocab[var].index if var in vocab else max_token
        var_embed[0] = embeddings[var_index]
        cos_dist = cosine_similarity(embeddings_allvar, var_embed)
        for item in mask_index_list:
            cos_dist[item][0] = -1

        cos_dist = cos_dist.reshape(max_var)

        k_count = 2
        max_search_iterations = max_var * 2
        iteration_count = 0
        found_count = 0  # 记录实际找到的替换词数量

        while len(n_list) < k and iteration_count < max_search_iterations:
            iteration_count += 1
            top_k_index = get_topk_index(k_count, cos_dist)

            if (all_var_list[top_k_index[-1]] in allcan_list) == False:
                n_list.append(all_var_list[top_k_index[-1]])
                allcan_list.append(all_var_list[top_k_index[-1]])
                found_count += 1  # 成功找到一个替换词

            k_count = k_count + 2

        # 打印每个变量实际找到的替换词数量
        print(f'"{var}" : {found_count}')

        nearest_dict[var] = n_list
    nearest_list.append(nearest_dict)

index = [i for i in range(len(nearest_list))]
nearest_pd = pd.DataFrame({'id': index, 'nearest_k': nearest_list})
nearest_pd.to_pickle('./var_name/code_nearest_top' + str(k) + '.pkl')
print('end')
