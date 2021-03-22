#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:deepwalk-master
@author:xiangguosun 
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: simple_walks.py 
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1 
@time: 2018/09/14 
"""

from io import open
from os import path
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from six.moves import zip

from data.path import graph

current_graph = None


def count_words(file):
    c = Counter()
    with open(file, 'r') as f:
        for l in f:
            words = l.strip().split()
            c.update(words)
    return c


def count_textfiles(files, workers=1):
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c


def count_textfiles(f):
    if path.isfile(f):  # Test whether a path is a regular file
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0

def count_lines(f):
    if path.isfile(f):  # Test whether a path is a regular file
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0

def _write_walks_to_disk(args):
    global current_graph

    num_paths, path_length, alpha, rand, f, G = args

    # G = current_graph
    with open(f, 'w') as fout:

        for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                     alpha=alpha, rand=rand):
            # print(walk)
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    return f



def write_walks_to_disk(G, filebase, num_paths, path_length, num_workers, alpha=0, rand=random.Random(0),
                            always_rebuild=True):
    global current_graph

    current_graph = G
    files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_workers))]

    expected_size = len(G)
    args_list = []
    files = []

    paths_per_worker = [len(list(filter(lambda z: z != None, [y for y in x])))
                        for x in graph.grouper(int(num_paths / num_workers)
                                               if num_paths % num_workers == 0
                                               else int(num_paths / num_workers) + 1,
                                               range(1, num_paths + 1))]
    # print(f'write_walks_to_disk current_graph:{type(current_graph)}')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            if always_rebuild or size != (ppw * expected_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_, G))
            else:
                files.append(file_)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files


class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()
