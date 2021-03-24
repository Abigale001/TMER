import sys
sys.path.append('../../')
import argparse
import random
import numpy as np
import time
import pickle
import torch
from collections import defaultdict
from itertools import chain
from pathlib import Path


class UIPath:
    def __init__(self, **kargs):
        self.metapath_list = kargs.get('metapath_list')
        self.ui_dict = dict()
        self.iu_dict = dict()
        self.ic_dict = dict()
        self.ci_dict = dict()
        self.ib_dict = dict()
        self.bi_dict = dict()
        self.usize = kargs.get('usize')
        self.isize = kargs.get('isize')
        self.csize = kargs.get('csize')
        self.bsize = kargs.get('bsize')
        self.uibi_outfile = ''
        self.uibici_outfile = ''
        self.uici_outfile = ''
        self.uicibi_outfile = ''
        self.embeddings = np.zeros((self.usize + self.isize + self.csize + self.bsize, 100))
        self.user_embedding = np.zeros((self.usize, 100))
        self.item_embedding = np.zeros((self.isize, 100))
        self.category_embedding = np.zeros((self.csize, 100))
        self.brand_embedding = np.zeros((self.bsize, 100))
        print('Begin to load data')
        start = time.time()

        self.load_embedding(kargs.get('node_emb_dic'))

        self.load_ui(kargs.get('ui_relation_file'))
        self.load_ic(kargs.get('ic_relation_file'))
        self.load_ib(kargs.get('ib_relation_file'))
        self.outputfolder = (kargs.get('outputfolder'))
        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))
        self.path_list = list()
        self.metapath_based_randomwalk()

    def load_embedding(self, embfile):
        nodewv_dic = pickle.load(open(embfile, 'rb'))
        nodewv_tensor = []
        all_nodes = list(range(len(nodewv_dic.keys())))
        for node in all_nodes:
            nodewv_tensor.append(nodewv_dic[node].numpy())
        nodewv_tensor = torch.Tensor(nodewv_tensor)
        self.embeddings = nodewv_tensor
        self.user_embedding = nodewv_tensor[:1450, :]
        self.item_embedding = nodewv_tensor[1450:12907, :]
        self.category_embedding = nodewv_tensor[12907:13336, :]
        self.brand_embedding = nodewv_tensor[13336:14522, :]

    def load_user_embedding(self, ufile):
        with open(ufile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.user_embedding[i][j] = float(arr[j + 1])

    def load_item_embedding(self, ifile):
        with open(ifile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.item_embedding[i][j] = float(arr[j + 1])

    def load_type_embedding(self, tfile):
        with open(tfile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.type_embedding[i][j] = float(arr[j + 1])

    def metapath_based_randomwalk(self):

        pair_list = []

        for u in range(0, self.usize):
            for i in self.ui_dict[u]:
                pair_list.append([u, i])
        print('load pairs finished num = ', len(pair_list))
        for metapath in self.metapath_list:
            print(metapath)
            if metapath == 'uibi':
                self.uibi_outfile = open(self.outputfolder + 'uibi.paths', 'w')
            if metapath == 'uibici':
                self.uibici_outfile = open(self.outputfolder + 'uibici.paths', 'w')
            if metapath == 'uici':
                self.uici_outfile = open(self.outputfolder + 'uici.paths', 'w')
            if metapath == 'uicibi':
                self.uicibi_outfile = open(self.outputfolder + 'uicibi.paths', 'w')

            ctn = 0
            t1 = time.time()
            avg = 0
            for u, i in pair_list:
                ctn += 1
                # print u, m
                if ctn % 10000 == 0:
                    print('%d [%.4f]\n' % (ctn, time.time() - t1))

                # 4 user metapaths
                if metapath == 'uici':
                    path = self.walk_uici(u, i)
                elif metapath == 'uibi':
                    path = self.walk_uibi(u, i)
                elif metapath == 'uicibi':
                    path = self.walk_uicibi(u, i)
                elif metapath == 'uibici':
                    path = self.walk_uibici(u, i)
                else:
                    print('unknow metapath.')
            if metapath == 'uibi':
                self.uibi_outfile.close()
            if metapath == 'uibici':
                self.uibici_outfile.close()
            if metapath == 'uici':
                self.uici_outfile.close()
            if metapath == 'uicibi':
                self.uicibi_outfile.close()

    def get_sim(self, u, v):
        return u.dot(v) / ((u.dot(u) ** 0.5) * (v.dot(v) ** 0.5))

    def walk_uici(self, s_u, e_i):
        limit = 10
        i_list = []
        for i in self.ui_dict[s_u]:
            sim = self.get_sim(self.embeddings[s_u], self.embeddings[i])
            i_list.append([i, sim])
        i_list.sort(key=lambda x: x[1], reverse=True)
        i_list = i_list[:min(limit, len(i_list))]

        c_list = []
        for c in self.ic_dict.get(e_i, []):
            c_list.append([c, 1])

        ic_list = []
        for i in i_list:
            for c in c_list:
                ii = i[0]
                cc = c[0]
                if ii in self.ic_dict and cc in self.ic_dict[ii] and ii != e_i:
                    sim = i[1]
                    if sim > 0.5:
                        ic_list.append([ii, cc, sim])
        ic_list.sort(key=lambda x: x[2], reverse=True)
        ic_list = ic_list[:min(5, len(ic_list))]

        if (len(ic_list) == 0):
            return
        self.uici_outfile.write(str(s_u) + ',' + str(e_i) + '\t' + str(len(ic_list)))
        for ic in ic_list:
            path = [str(s_u), str(ic[0]), str(ic[1]), str(e_i)]
            self.uici_outfile.write('\t' + ' '.join(path))
        self.uici_outfile.write('\n')

    def walk_uibi(self, s_u, e_i):

        limit = 10
        i_list = []
        for i in self.ui_dict[s_u]:
            sim = self.get_sim(self.embeddings[s_u], self.embeddings[i])
            i_list.append([i, sim])
        i_list.sort(key=lambda x: x[1], reverse=True)
        i_list = i_list[:min(limit, len(i_list))]

        b_list = []
        for b in self.ib_dict.get(e_i, []):
            b_list.append([b, 1])

        ib_list = []
        for i in i_list:
            for b in b_list:
                ii = i[0]
                bb = b[0]
                if ii in self.ib_dict and bb in self.ib_dict[ii] and ii != e_i:
                    sim = i[1]
                    if sim > 0.5:
                        ib_list.append([ii, bb, sim])
        ib_list.sort(key=lambda x: x[2], reverse=True)
        ib_list = ib_list[:min(5, len(ib_list))]

        if (len(ib_list) == 0):
            return
        self.uibi_outfile.write(str(s_u) + ',' + str(e_i) + '\t' + str(len(ib_list)))
        for ib in ib_list:
            path = [str(s_u), str(ib[0]), str(ib[1]), str(e_i)]
            self.uibi_outfile.write('\t' + ' '.join(path))
        self.uibi_outfile.write('\n')

    def walk_uicibi(self, start, end):
        path = [str(start)]

        # u - i
        # print start
        if start not in self.ui_dict:
            return None
        index = np.random.randint(len(self.ui_dict[start]))
        i = self.ui_dict[start][index]
        path.append(str(i))
        # i - c
        if i not in self.ic_dict:
            return None
        index = np.random.randint(len(self.ic_dict[i]))
        c = self.ic_dict[i][index]
        path.append(str(c))

        # c - i
        if c not in self.ci_dict:
            return None
        index = np.random.randint(len(self.ci_dict[c]))
        i = self.ci_dict[c][index]
        path.append(str(i))

        # i - b
        if i not in self.ib_dict:
            return None
        index = np.random.randint(len(self.ib_dict[i]))
        b = self.ib_dict[i][index]
        path.append(str(b))

        # b - i
        # print path
        if b not in self.bi_dict:
            return None
        if end not in self.bi_dict[b]:
            return None
        path.append(str(end))
        write_path = str(start) + ',' + str(end) + '\t' + '1' + '\t' + ' '.join(path)
        self.uicibi_outfile.write(write_path)
        self.uicibi_outfile.write('\n')
        return ' '.join(path)

    def walk_uibici(self, start, end):
        path = [str(start)]

        # u - i
        # print start
        if start not in self.ui_dict:
            return None
        index = np.random.randint(len(self.ui_dict[start]))
        i = self.ui_dict[start][index]
        path.append(str(i))
        # i - b
        if i not in self.ib_dict:
            return None
        index = np.random.randint(len(self.ib_dict[i]))
        b = self.ib_dict[i][index]
        path.append(str(b))

        # b - i
        if b not in self.bi_dict.keys():
            return None
        index = np.random.randint(len(self.bi_dict[b]))
        i = self.bi_dict[b][index]
        path.append(str(i))

        # i - c
        if i not in self.ic_dict:
            return None
        index = np.random.randint(len(self.ic_dict[i]))
        c = self.ic_dict[i][index]
        path.append(str(c))

        # c - i
        # print path
        if c not in self.ci_dict:
            return None
        if end not in self.ci_dict[c]:
            return None
        path.append(str(end))

        write_path = str(start) + ',' + str(end) + '\t' + '1' + '\t' + ' '.join(path)
        self.uibici_outfile.write(write_path)
        self.uibici_outfile.write('\n')
        return ' '.join(path)


    def load_ib(self, ibfile):
        item_brand_data = open(ibfile, 'r').readlines()
        for item_brand_ele in item_brand_data:
            item_brand_ele_list = item_brand_ele.strip().split(',')
            item = int(item_brand_ele_list[0])
            brand = int(item_brand_ele_list[1])
            if item not in self.ib_dict.keys():
                self.ib_dict[item] = [brand]
            else:
                self.ib_dict[item].append(brand)

            if brand not in self.bi_dict.keys():
                self.bi_dict[brand] = [item]
            else:
                self.bi_dict[brand].append(item)

    def load_ic(self, icfile):
        item_category_data = open(icfile, 'r').readlines()
        for item_category_ele in item_category_data:
            item_category_ele_list = item_category_ele.strip().split(',')
            item = int(item_category_ele_list[0])
            category = int(item_category_ele_list[1])
            if item not in self.ic_dict.keys():
                self.ic_dict[item] = [category]
            else:
                self.ic_dict[item].append(category)

            if category not in self.ci_dict.keys():
                self.ci_dict[category] = [item]
            else:
                self.ci_dict[category].append(item)

    def load_ui(self, uifile):
        user_item_data = open(uifile, 'r').readlines()
        for user_item_ele in user_item_data:
            user_item_ele_list = user_item_ele.strip().split(',')
            user = int(user_item_ele_list[0])
            item = int(user_item_ele_list[1])
            if item not in self.iu_dict.keys():
                self.iu_dict[item] = [user]
            else:
                self.iu_dict[item].append(user)

            if user not in self.ui_dict.keys():
                self.ui_dict[user] = [item]
            else:
                self.ui_dict[user].append(item)

class IIPath:
    def __init__(self, **kargs):
        self.metapath_list = kargs.get('metapath_list')
        # self.metapath_list = ['uibi', 'uibici', 'uici', 'uicibi', 'ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
        self.outputfolder = kargs.get('outputfolder')

        self.ui_dict = dict()
        self.iu_dict = dict()
        self.ic_dict = dict()
        self.ci_dict = dict()
        self.ib_dict = dict()
        self.bi_dict = dict()
        self.all_ii_direct = set()
        self.usize = kargs.get('usize')
        self.isize = kargs.get('isize')
        self.csize = kargs.get('csize')
        self.bsize = kargs.get('bsize')

        self.embeddings = np.zeros((self.usize+self.isize+self.csize+self.bsize, 100))

        print('Begin to load data')
        start = time.time()

        self.load_embedding(kargs.get('node_emb_dic'))


        self.load_ui(kargs.get('ui_relation_file'))
        self.load_ic(kargs.get('ic_relation_file'))
        self.load_ib(kargs.get('ib_relation_file'))
        self.load_all_ii_direct(kargs.get('user_history_file'))

        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))
        self.path_list = list()

        self.metapath_based_randomwalk()


    def load_embedding(self, embfile):
        nodewv_dic = pickle.load(open(embfile, 'rb'))
        nodewv_tensor = []
        all_nodes = list(range(len(nodewv_dic.keys())))
        for node in all_nodes:
            nodewv_tensor.append(nodewv_dic[node].numpy())
        nodewv_tensor = torch.Tensor(nodewv_tensor)
        self.embeddings = nodewv_tensor
        print('############################')
        print(self.embeddings.shape)


    def load_all_ii_direct(self, user_history_f):
        with open(user_history_f, 'r') as f:  # 2200
            for line in f:
                s = line.split()
                item_history = [int(x) for x in s[1:]]
                item_num = len(item_history)
                for index in range(item_num-1):
                    i1 = item_history[index]
                    i2 = item_history[index+1]
                    self.all_ii_direct.add((i1,i2))
        print(f'Here are {len(self.all_ii_direct)} item-item paths in data.')

    def load_ib(self, ibfile):
        item_brand_data = open(ibfile, 'r').readlines()
        for item_brand_ele in item_brand_data:
            item_brand_ele_list = item_brand_ele.strip().split(',')
            item = int(item_brand_ele_list[0])
            brand = int(item_brand_ele_list[1])
            if item not in self.ib_dict.keys():
                self.ib_dict[item] = [brand]
            else:
                self.ib_dict[item].append(brand)

            if brand not in self.bi_dict.keys():
                self.bi_dict[brand] = [item]
            else:
                self.bi_dict[brand].append(item)

    def load_ic(self, icfile):
        item_category_data = open(icfile, 'r').readlines()
        for item_category_ele in item_category_data:
            item_category_ele_list = item_category_ele.strip().split(',')
            item = int(item_category_ele_list[0])
            category = int(item_category_ele_list[1])
            if item not in self.ic_dict.keys():
                self.ic_dict[item] = [category]
            else:
                self.ic_dict[item].append(category)

            if category not in self.ci_dict.keys():
                self.ci_dict[category] = [item]
            else:
                self.ci_dict[category].append(item)

    def load_ui(self, uifile):
        user_item_data = open(uifile, 'r').readlines()
        for user_item_ele in user_item_data:
            user_item_ele_list = user_item_ele.strip().split(',')
            user = int(user_item_ele_list[0])
            item = int(user_item_ele_list[1])
            if item not in self.iu_dict.keys():
                self.iu_dict[item] = [user]
            else:
                self.iu_dict[item].append(user)

            if user not in self.ui_dict.keys():
                self.ui_dict[user] = [item]
            else:
                self.ui_dict[user].append(item)

    def ifInIIpairs(self, startItem, endItem):
        if (startItem, endItem) in self.all_ii_direct:
            return True
        else:
            return False

    def save_icibi(self, all_item_ids, start_time, outfile):
        limit = 5
        for i in all_item_ids:
            try:
                c_list = self.ic_dict[i]
            except KeyError:
                continue

            c_list = self.get_top_k(c_list, i, limit)
            for c in c_list:
                try:
                    i2_list = self.ci_dict[c]
                    if i in i2_list: i2_list.remove(i)
                except KeyError:
                    continue


                for i2 in i2_list:
                    try:
                        b_list = self.ib_dict[i2]
                    except KeyError:
                        continue
                    for b in b_list:
                        try:
                            i3_list = self.bi_dict[b]
                            if i in i3_list: i3_list.remove(i)
                            if i2 in i3_list: i3_list.remove(i2)
                        except KeyError:
                            continue
                        for i3 in i3_list:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(c) + ' ' + str(i2) + ' ' + str(b) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')

    def save_ibici(self, all_item_ids, start_time, outfile):
        limit = 5
        for i in all_item_ids:
            try:
                b_list = self.ib_dict[i]
            except KeyError:
                continue

            b_list = self.get_top_k(b_list, i, limit)
            for b in b_list:
                try:
                    i2_list = self.bi_dict[b]
                except KeyError:
                    continue

                i2_list = self.get_top_k(i2_list, b, limit)
                for i2 in i2_list:
                    try:
                        c_list = self.ic_dict[i2]
                    except KeyError:
                        continue
                    for c in c_list:
                        try:
                            i3_list = self.ci_dict[c]
                        except KeyError:
                            continue
                        for i3 in i3_list:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(b) + ' ' + str(i2) + ' ' + str(c) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')


    def get_sim(self, u, v):
        return u.dot(v) / ((u.dot(u) ** 0.5) * (v.dot(v) ** 0.5))

    def get_top_k(self, c_list, i, limit):
        i_c1_sim_list = []
        for c1 in c_list:
            # cal sim
            sim = self.get_sim(self.embeddings[c1], self.embeddings[i])
            i_c1_sim_list.append([c1, sim])
        i_c1_sim_list.sort(key=lambda x: x[1], reverse=True)
        i_c1_sim_list = i_c1_sim_list[:min(limit, len(i_c1_sim_list))]
        c_list = [c1 for c1, sim in i_c1_sim_list]
        return c_list

    def save_icici(self, all_item_ids, start_time, outfile):
        limit = 5
        start_time = time.time()
        for i in all_item_ids:
            try:
                c_list = self.ic_dict[i]
            except KeyError:
                continue
            c_list = self.get_top_k(c_list, i, limit)
            for c1 in c_list:
                try:
                    i_list2 = self.ci_dict[c1]
                except KeyError:
                    continue
                i_list2 = self.get_top_k(i_list2, c1, limit)

                for i2 in i_list2:
                    try:
                        c_list2 = self.ic_dict[i2]
                    except KeyError:
                        continue

                    c_list2 = self.get_top_k(c_list2, i2, limit)
                    for c2 in c_list2:
                        try:
                            i_list3 = self.ci_dict[c2]
                        except KeyError:
                            continue

                        for i3 in i_list3:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(c1) + ' ' + str(i2) + ' ' + str(c2) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')



    def save_ibibi(self, all_item_ids, start_time, outfile):
        # i -> b
        for i in all_item_ids:
            try:
                b_list = self.ib_dict[i]
            except KeyError:
                continue

            # b -> i
            for b1 in b_list:
                try:
                    i2_list = self.bi_dict[b1]
                    if i in i2_list: i2_list.remove(i)
                except KeyError:
                    continue

                # i -> b
                for i2 in i2_list:
                    try:
                        b_list2 = self.ib_dict[i2]
                        if b1 in b_list2: b_list2.remove(b1)
                    except KeyError:
                        continue

                    # b -> i
                    for b2 in b_list2:
                        try:
                            i3_list = self.bi_dict[b2]
                            if i in i3_list: i3_list.remove(i)
                            if i2 in i3_list: i3_list.remove(i2)
                        except KeyError:
                            continue
                        for i3 in i3_list:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(b1) + ' ' + str(i2) + ' ' + str(b2) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')

    def save_iuiui(self, all_item_ids, start_time, outfile):
        for i in all_item_ids:
            try:
                u_list1 = self.iu_dict[i]
            except KeyError:
                continue

            for u1 in u_list1:
                try:
                    i2_list = self.ui_dict[u1]
                except KeyError:
                    continue
                for i2 in i2_list:
                    try:
                        u_list2 = self.iu_dict[i2]
                    except KeyError:
                        continue
                    for u2 in u_list2:
                        try:
                            i3_list = self.ui_dict[u2]
                        except KeyError:
                            continue
                        for i3 in i3_list:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(u1) + ' ' + str(i2) + ' ' + str(u2) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')

    def save_iciui(self, all_item_ids, start_time, outfile):
        for i in all_item_ids:
            try:
                c_list = self.ic_dict[i]
            except KeyError:
                continue

            for c1 in c_list:
                try:
                    i2_list = self.ci_dict[c1]
                except KeyError:
                    continue
                for i2 in i2_list:
                    try:
                        u_list = self.iu_dict[i2]
                    except KeyError:
                        continue
                    for u in u_list:
                        try:
                            i3_list = self.ui_dict[u]
                        except KeyError:
                            continue
                        for i3 in i3_list:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(c1) + ' ' + str(i2) + ' ' + str(u) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')

    def save_ibiui(self, all_item_ids, start_time, outfile):
        for i in all_item_ids:
            try:
                b_list = self.ib_dict[i]
            except KeyError:
                continue

            for b in b_list:
                try:
                    i2_list = self.bi_dict[b]
                except KeyError:
                    continue
                for i2 in i2_list:
                    try:
                        u_list = self.iu_dict[i2]
                    except KeyError:
                        continue
                    for u in u_list:
                        try:
                            i3_list = self.ui_dict[u]
                        except KeyError:
                            continue
                        for i3 in i3_list:
                            if self.ifInIIpairs(i, i3):
                                path = str(i) + ' ' + str(b) + ' ' + str(i2) + ' ' + str(u) + ' ' + str(i3)
                                path_id = str(i) + ',' + str(i3)
                                write_content = path_id + '\t' + path + '\n'
                                outfile.write(write_content)
            this_time = time.time() - start_time
            print(f'processed item: {i}, time: {this_time}')



    def metapath_based_randomwalk(self):

        all_item_ids = list(self.iu_dict.keys())
        all_item_ids.sort()
        start_time = time.time()

        print(self.metapath_list)
        for metapath in self.metapath_list:
            print(metapath)
            outfile = open(self.outputfolder + metapath + '.paths', 'w')
            print(f'outfile name = {self.outputfolder}{metapath}.paths')
            if metapath == 'icibi':
                self.save_icibi(all_item_ids, start_time, outfile)
            if metapath == 'ibici':
                self.save_ibici(all_item_ids, start_time, outfile)
            if metapath == 'icici':
                self.save_icici(all_item_ids, start_time, outfile)
            if metapath == 'ibibi':
                self.save_ibibi(all_item_ids, start_time, outfile)
            if metapath == 'iuiui':
                self.save_iuiui(all_item_ids, start_time, outfile)
            if metapath == 'iciui':
                self.save_iciui(all_item_ids, start_time, outfile)
            if metapath == 'ibiui':
                self.save_ibiui(all_item_ids, start_time, outfile)

            outfile.close()

def load_all_ii_direct(user_history_f):
    all_ii_direct = set()
    with open(user_history_f, 'r') as f:  # 2200
        for line in f:
            s = line.split()
            item_history = [int(x) for x in s[1:]]
            item_num = len(item_history)
            for index in range(item_num - 1):
                i1 = item_history[index]
                i2 = item_history[index + 1]
                all_ii_direct.add((i1, i2))
    print(f'Here are {len(all_ii_direct)} item-item paths in data.')



def form_ii_paths(user_history_file, metapaths_folder, output_filename, metapaths):
    ii_path_dict = defaultdict()
    ibibi_path_dict = defaultdict()
    ibici_path_dict = defaultdict()
    ibiui_path_dict = defaultdict()
    icici_path_dict = defaultdict()
    icibi_path_dict = defaultdict()
    iciui_path_dict = defaultdict()
    iuiui_path_dict = defaultdict()
    # for each dict, randomly get a value
    random_one_ibibi_path_dict = defaultdict()
    random_one_ibici_path_dict = defaultdict()
    random_one_ibiui_path_dict = defaultdict()
    random_one_icici_path_dict = defaultdict()
    random_one_icibi_path_dict = defaultdict()
    random_one_iciui_path_dict = defaultdict()
    random_one_iuiui_path_dict = defaultdict()
    load_all_ii_direct(user_history_file)

    all_instances = 0

    for metapath in metapaths:
        this_metapath_instances = 0
        path_filename = metapaths_folder + metapath + '.paths'
        with open(path_filename, 'r') as mpf:
            for line in mpf:
                ii, path = line.strip().split('\t')
                if ii not in ii_path_dict.keys():
                    ii_path_dict[ii] = []
                if ii not in globals()[metapath + '_path_dict']:
                    globals()[metapath + '_path_dict'][ii] = []
                ii_path_dict[ii].append(path)
                globals()[metapath + '_path_dict'][ii].append(path)
                all_instances += 1
                this_metapath_instances += 1
    print(len(ibibi_path_dict))
    print(len(ibici_path_dict))
    print(len(ibiui_path_dict))
    print(len(icici_path_dict))
    print(len(icibi_path_dict))
    print(len(iciui_path_dict))
    print(len(iuiui_path_dict))



    for metapath in metapaths:
        for key, values in globals()[metapath + '_path_dict'].items():
            random_index = random.choice(values)
            globals()['random_one_' + metapath + '_path_dict'][key] = random_index


    random_one_all_path_dict = defaultdict(list)
    for k, v in chain(random_one_ibibi_path_dict.items(), random_one_ibici_path_dict.items(),
                      random_one_ibiui_path_dict.items(), random_one_icici_path_dict.items(),
                      random_one_icibi_path_dict.items(), random_one_iciui_path_dict.items(),
                      random_one_iuiui_path_dict.items()):
        random_one_all_path_dict[k].append(v)

    avg_randomed = 0
    output_f = open(output_filename, 'a+')
    for key, paths in random_one_all_path_dict.items():
        num_paths = len(paths)
        avg_randomed += num_paths
        paths_str = '\t'.join(path for path in paths)
        output_f.write(key + '\t' + str(num_paths) + '\t' + paths_str + '\n')
    avg_randomed = avg_randomed / 36654
    print(avg_randomed)



def gen_instances(args):
    refinefolder = args.refinefolder
    ic_relation_file = refinefolder + 'item_category.relation'
    ib_relation_file = refinefolder + 'item_brand.relation'
    ui_relation_file = refinefolder + 'user_item.relation'
    node_emb_dic = args.basedatafolder + 'nodewv.dic'
    user_history_file = args.userhistoryfile
    usize = args.usize
    isize = args.isize
    csize = args.csize
    bsize = args.bsize
    ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    # ii_metapahts_list = ['iuiui']
    ii_metapahts_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
    outputinstancesfolder = args.outputinstancesfolder
    output_filename = outputinstancesfolder + 'ii_random_form.paths'

    UIPath(ib_relation_file=ib_relation_file, ic_relation_file=ic_relation_file,
                           ui_relation_file=ui_relation_file,
                           metapath_list=ui_metapaths_list,node_emb_dic=node_emb_dic,
           usize=usize, isize=isize, csize=csize, bsize=bsize,
           outputfolder=outputinstancesfolder)
    IIPath(ib_relation_file=ib_relation_file, ic_relation_file=ic_relation_file,
                           ui_relation_file=ui_relation_file, user_history_file=user_history_file,
           node_emb_dic=node_emb_dic, metapath_list=ii_metapahts_list,
           usize=usize, isize=isize, csize=csize, bsize=bsize,
           outputfolder=outputinstancesfolder)
    form_ii_paths(user_history_file, outputinstancesfolder, output_filename, ii_metapahts_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate user item and item item instances')
    parser.add_argument('basedatafolder', type=str, default='../Amazon_Music/',
                        nargs='?', help='this data base folder')
    parser.add_argument('outputinstancesfolder', type=str, default='../Amazon_Music/path/all_ui_ii_instance_paths/',
                        nargs='?', help='this instances folder')
    parser.add_argument('refinefolder', type=str, default='../Amazon_Music/refine/', nargs='?',
                        help='output to refine folder')
    parser.add_argument('userhistoryfile', type=str, default='../Amazon_Music/path/user_history/user_history.txt', nargs='?',
                        help='user history file')
    parser.add_argument('usize', type=int, default=1450, nargs='?')
    parser.add_argument('isize', type=int, default=11457, nargs='?')
    parser.add_argument('csize', type=int, default=429, nargs='?')
    parser.add_argument('bsize', type=int, default=1185, nargs='?')

    args = parser.parse_args()

    Path(args.outputinstancesfolder).mkdir(parents=True, exist_ok=True)
    gen_instances(args)
