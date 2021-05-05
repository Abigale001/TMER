#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:deepwalk-master
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: embed_nodes.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/09/13 
"""



import random

from data.path import simple_walks as serialized_walks
from gensim.models import Word2Vec
import pickle
import torch
from collections import defaultdict
from pathlib import Path

if __name__ == '__main__':

    number_walks = 10
    walk_length = 6 # length of path
    workers = 2
    representation_size = 100
    window_size = 3
    output = '../Amazon_Music/node.wv'
    G = pickle.load(open('../Amazon_Music/graph.nx', 'rb')) #node 包括 user/item/brand/category/also_bought
    walks_filebase = "../Amazon_Music/path/node_path/walks.txt"
    nodewv = '../Amazon_Music/nodewv.dic'
    print("Number of nodes: {}".format(G.number_of_nodes()))
    print("Number of edges: {}".format(G.number_of_edges()))
    print("number_walks: {}".format(number_walks))
    num_walks = G.number_of_nodes() * number_walks
    print("Number of walks: {}".format(num_walks))
    data_size = num_walks * walk_length
    print("Data size (walks*length): {}".format(data_size))

    print(type(G))
    Path("../Amazon_Music/path/node_path").mkdir(parents=True, exist_ok=True)

    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                      path_length=walk_length, num_workers=workers, alpha=0.1,
                                                      rand=random.Random(100), always_rebuild=True)  # , r=args.r)
    # walk_files = ["../Amazon_Music/path/node_path/walks.txt.0", "../Amazon_Music/path/node_path/walks.txt.1"]
    walks = serialized_walks.WalksCorpus(walk_files)


    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
                     workers=workers)

    model.wv.save_word2vec_format(output)

    nodewv_dic = defaultdict(torch.Tensor)
    with open(output, 'r') as f:
        f.readline()
        for line in f:
            s = line.split()
            nodeid = int(s[0])
            fea = [float(x) for x in s[1:]]
            nodewv_dic[nodeid] = torch.Tensor(fea)

    pickle.dump(nodewv_dic, open(nodewv, 'wb'))
