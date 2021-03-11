#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:deepwalk-master
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: embed_nodes.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/09/13 
"""

from gensim.models import Word2Vec
from collections import defaultdict
import pickle
import pandas as pd

user_item_relation = pd.read_csv('../../Amazon_Music/refine/user_item.relation', header=None, sep=',')
new = user_item_relation.sort_values(2)
print(user_item_relation)

users = set(user_item_relation[0])
# with open('./user_history.txt', 'a') as f:
with open('../../Amazon_Music/path/user_history/user_history.txt', 'a') as f:
    for user in users:
        this_user = user_item_relation[user_item_relation[0] == user].sort_values(2)
        path = [user] + this_user[1].tolist()
        for s in path:
            f.write(str(s) + ' ')
        f.write('\n')


edges = set()
# with open('./user_history.txt', 'r') as f:
with open('../../Amazon_Music/path/user_history/user_history.txt', 'r') as f:
    for line in f.readlines():
        s = line.split()
        uid = s[0]
        node_list = [int(x) for x in s[1:]]
        for i in range(len(node_list) - 1):
            if node_list[i] <= node_list[i + 1]:
                t = (node_list[i], node_list[i + 1])
            else:
                t = (node_list[i + 1], node_list[i])
            edges.add(t)

print(len(edges))
edges_id = defaultdict(int)
id_edges = defaultdict(tuple)
for i, edge in enumerate(edges):
    edges_id[edge] = i
    id_edges[i] = edge
pickle.dump(edges_id, open('../../Amazon_Music/path/user_history/user_history.edges2id', 'wb'))
pickle.dump(id_edges, open('../../Amazon_Music/path/user_history/user_history.id2edges', 'wb'))


# 2


edges_id = pickle.load(open('../../Amazon_Music/path/user_history/user_history.edges2id', 'rb'))
edge_path = []

with open('../../Amazon_Music/path/user_history/user_history.txt', 'r') as f:
    for line in f.readlines():
        path = []
        node_list = [int(x) for x in line.split()[1:]]
        for i in range(len(node_list) - 1):
            if node_list[i] <= node_list[i + 1]:
                t = (node_list[i], node_list[i + 1])
            else:
                t = (node_list[i + 1], node_list[i])
            path.append(edges_id[t])
        edge_path.append(path)

with open('../../Amazon_Music/path/user_history/user_history_edge_path.txt', 'a') as f:
    for path in edge_path:
        print(len(path),path)
        for s in path:
            print(s)
            f.write(str(s) + ' ')
        f.write('\n')



walks = []

with open('../../Amazon_Music/path/user_history/user_history_edge_path.txt', 'r') as f:
    for line in f:
        walks.append(line.split())
# print(walks)

print("Training...")
model = Word2Vec(walks, size=100, window=3, min_count=0, sg=1, hs=1,
                 workers=4)
# model.wv (item_item) 2200* 100
model.wv.save_word2vec_format('../../Amazon_Toys/path/user_history/item_item.wv')
