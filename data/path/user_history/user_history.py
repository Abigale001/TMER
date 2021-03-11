#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:Hongxu_ICDM
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: user_history.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2020/05/20

input:
user_history.txt，超过N个以上的，选取前N个，剩下的留作测试集，小于N个的，直接pass掉，
item_item.wv
node.wv (user.feature,item.feature)

output:
user_id: features(2N-1,100)
training_links,
testing_links
训练集合，测试集合
"""
import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

bridge = 2
train = 4
test = 6

history = []
testing = []
training = []

"""
把nodewv转化为dic,或者是tensor
"""
# nodewv = '../../Amazon_Music/node.wv'
nodewv = '../../Amazon_Music/node_GAT.embeddings'

nodewv_dic = defaultdict(torch.Tensor)
with open(nodewv, 'r') as f:
    f.readline()
    for line in f:
        s = line.split()
        nodeid = int(s[0])
        fea = [float(x) for x in s[1:]]
        nodewv_dic[nodeid] = torch.Tensor(fea)

print("node.feature done")
print(len(nodewv_dic)) #26333
user_history_edges2id = pickle.load(open('../../Amazon_Music/path/user_history/user_history.edges2id', 'rb'))

"""
转化为 dic,tensor
"""

##################################
item_item_wv_dic = defaultdict(torch.Tensor)
with open('../../Amazon_Music/path/user_history/item_item.wv', 'r') as f:
    f.readline()
    for line in f:
        s = line.split()
        item_item_id = int(s[0])
        fea = [float(x) for x in s[1:]]
        item_item_wv_dic[item_item_id] = torch.Tensor(fea)
print("item_item.feature done")
print(len(item_item_wv_dic)) #36558
# exit(0)
###################################
# item_item_wv_dic_att = pickle.load(open('../../../representations/li_user_item_dic_att.wv', 'rb'))  # defautdic #len 2200
# print(len(item_item_wv_dic_att))
###################################


user_history_wv = defaultdict(torch.Tensor)
with open('../../Amazon_Music/path/user_history/user_history.txt', 'r') as f: # 2200
    for line in f:
        s = line.split()
        uid = int(s[0])
        item_history = [int(x) for x in s[1:]]
        if len(item_history) < (bridge+train):
            continue
        else:
            # 划分 train 和 test
            training.append([uid] + item_history[bridge:(bridge+train)])
            testing.append([uid] + item_history[train:])

            """
            training
            item1.feature,(item2,item2).feature,item2.feature...
            """
            ## old user_history
            # feature = nodewv_dic[item_history[0]].reshape((1, -1))  # first item feature
            # for i in range(N - 1):
            #     item1 = item_history[i]
            #     item2 = item_history[i + 1]
            #     print(f'item1: {item1}')
            #     print(f'item2: {item2}')
            #     if item1 > item2:
            #         t = (item2, item1)
            #     else:
            #         t = (item1, item2)
            #
            #     edge_id = user_history_edges2id[t]
            #     ii = item_item_wv_dic[edge_id].reshape((1, -1))
            #     feature = torch.cat([feature, ii, nodewv_dic[item2].reshape((1, -1))], 0)
            # user_history_wv[uid] = feature
            # print(feature.shape)

#user_history_wv 包括 history item / user->...>current item /
pickle.dump(training, open('./training', 'wb'))
pickle.dump(testing, open('./testing', 'wb'))
pickle.dump(user_history_wv, open('./user_history.wv', 'wb'))
