#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

from six.moves import range, zip_longest
import random


def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        返回一些较短的随机游走路径
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
        random.Random()：产生0-1之间的随机浮点数
        请注意：这里的随机游走路径未必是连续的，有可能是走着走着突然回到起点接着走
    """

    if start:
        path = [start]
    else:
        # Sampling is uniform w.r.t V, and not w.r.t E

        path = [rand.choice(list(G.nodes()))]

    while len(path) < path_length:
        cur = path[-1]
        if len(G[cur]) > 0:  # 当cur有邻居时
            if rand.random() >= alpha:  # 这是一个经典的概率编程的语句
                """
                这条语句成立的概率是1-alpha
                """
                samples = list(G[cur])
                if len(samples) > 0:
                    path.append(rand.choice(samples))
                else:
                    path.append(start)
                """
                上面这段代码是为了使deepwalk与我们ada_random_walk在同一个起跑线上做比较，因为我们的ada_random_walk
                在这里用了这样的采样（尽管我们也可以不采样，但是运算时间会更大，采样后，30个进程，需要运行30分钟，
                不采样可能会运行更长）
                """

                # path.append(rand.choice(G[cur]))
            else:
                path.append(path[0])  # 这里应该写的是path.append(cur)吧？
                """
                以概率1-alpha从当前节点继续向前走，或者以alpha的概率restart
                """
        else:
            break
    return [str(node) for node in path]


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        # 这条语句将下面的random walk过程重复了num_paths次，
        # 相当于每一个节点做num_paths次random walk
        rand.shuffle(nodes)
        for node in nodes:  # 这条语句对图所有的节点各自进行一次random walk
            walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

    return walks


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        count = 0
        for node in nodes:
            count = count + 1
            yield random_walk(G, path_length, rand=rand, alpha=alpha, start=node)
