#encoding=utf-8
import torch.tensor
from collections import defaultdict
import pickle
import pandas as pd

def load_node_tensor(filename):
    nodewv_dic = pickle.load(open(filename, 'rb'))
    nodewv_tensor = []
    all_nodes = list(range(len(nodewv_dic.keys())))
    for node in all_nodes:
        nodewv_tensor.append(nodewv_dic[node].numpy())
    nodewv_tensor = torch.Tensor(nodewv_tensor)
    return nodewv_tensor

def instance_paths_to_dict(path_file) -> dict:
    user_item_paths_relation_uibi = open(path_file,'r').readlines() #4463
    ui_pairs = []
    ui_paths_dict = {}
    for user_item_paths_relation in user_item_paths_relation_uibi:
        ui, pathnum, path_list_str = user_item_paths_relation.strip().split('\t',2)
        path_list = path_list_str.split('\t') #pathnum 是 list的长度
        (user, item) = ui.split(',')
        ui_pair = (int(user), int(item))
        ui_pairs.append(ui_pair)
        if ui_pair not in ui_paths_dict.keys():
            for index, path in enumerate(path_list):
                path = path.strip().split(' ')
                path_list[index] = path
            ui_paths_dict[ui_pair] = path_list
    # print(ui_paths_dict)
    # exit(0)
    return ui_paths_dict

def get_instance_paths(path_file) -> list:
    user_item_paths_relation_uibi = open(path_file, 'r').readlines()
    paths_list = []
    for user_item_paths_relation in user_item_paths_relation_uibi:
        ui, pathnum, path_list_str = user_item_paths_relation.strip().split('\t', 2)
        path_list = path_list_str.split('\t')  # pathnum 是 list的长度
        for index, path in enumerate(path_list):
            path = path.strip().split(' ')
            path_list[index] = path
            paths_list.append(path)
    # print(len(paths_list))
    # print(paths_list)
    # exit(0)
    return paths_list

def load_ui_seq_relation(uifile):
    ui_dict = {}
    user_item_data = open(uifile, 'r').readlines()
    for line in user_item_data:
        line_list = line.strip().split(' ')
        user = int(line_list[0])
        item_list = [int(item) for item in line_list[1:]]
        ui_dict[user] = item_list
    return ui_dict


def load_item_item_wv(filename):
    item_item_wv_dic = defaultdict(torch.Tensor)
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            s = line.split()
            item_item_id = int(s[0])
            fea = [float(x) for x in s[1:]]
            item_item_wv_dic[item_item_id] = torch.Tensor(fea)
    return item_item_wv_dic


def load_ui_metapath_instances_emb(metapath_list, ui_metapath_emb_folder, user_num, ui_dict, user_item_direct_emb):
    ui_metapath = []
    for metapath in metapath_list:
        ui_metapath_emb_file = ui_metapath_emb_folder+metapath+'.wv'
        ui_metapath.append(pickle.load(open(ui_metapath_emb_file, 'rb'))) # [dict, ...] dict:{(u,i):}

    ui_instances_embs = defaultdict()
    for u in range(user_num):
        this_user_ui_instances_embs = defaultdict() # {(u,i):[array1, array2,,,], (u,i):[,,]}
        for i in ui_dict[u]:
            for ele in ui_metapath:
                # print(ele)
                # exit(0)
                if (u,i) in ele.keys():
                    if (u,i) not in this_user_ui_instances_embs.keys():
                        this_user_ui_instances_embs[(u,i)] = ele[(u,i)]
                    else:
                        for list_ele in ele[(u,i)]:
                            this_user_ui_instances_embs[(u,i)].append(list_ele)
            # if there is no this path in metapath list, use user-item path emb
            if (u,i) not in this_user_ui_instances_embs.keys():
                this_user_ui_instances_embs[(u,i)] = user_item_direct_emb[u].unsqueeze(0)
            else:
                this_user_ui_instances_embs[(u, i)] = torch.Tensor(this_user_ui_instances_embs[(u, i)])
        assert len(ui_dict[u]) == len(this_user_ui_instances_embs)
        ui_instances_embs[u] = this_user_ui_instances_embs
    return ui_instances_embs


def load_ii_metapath_instances_emb(metapath_emb_folder, user_num, ui_dict, item_item_direct_emb, edges_id_dict):
    ii_metapath_emb_file = metapath_emb_folder + 'ii_random_form.wv'
    ii_metapath_emb = pickle.load(open(ii_metapath_emb_file, 'rb')) # [dict, ...] dict:{(u,i):}

    ii_instances_embs = defaultdict()
    for u in range(user_num):
        this_user_ii_instances_embs = defaultdict()  # {(u,i):[array1, array2,,,], (u,i):[,,]}
        num_item = len(ui_dict[u])
        for i_index in range(num_item - 1):
            i1 = ui_dict[u][i_index]
            i2 = ui_dict[u][i_index + 1]

                # print(ele)
                # exit(0)
            if (i1, i2) in ii_metapath_emb.keys():
                if (i1, i2) not in this_user_ii_instances_embs.keys():
                    this_user_ii_instances_embs[(i1, i2)] = ii_metapath_emb[(i1, i2)]
                else:
                    for list_ele in ii_metapath_emb[(i1, i2)]:
                        this_user_ii_instances_embs[(i1, i2)].append(list_ele)
            if (i1, i2) in this_user_ii_instances_embs.keys():
                this_user_ii_instances_emb_tensor = torch.Tensor(this_user_ii_instances_embs[(i1, i2)])
                this_user_ii_instances_embs[(i1, i2)] = this_user_ii_instances_emb_tensor
                continue
            if (i1, i2) not in this_user_ii_instances_embs.keys():  # if there is no this path in metapath list, use item-item path emb
                this_user_ii_instances_embs[(i1, i2)] = item_item_direct_emb[edges_id_dict[(i1, i2)]].unsqueeze(0)
        assert (num_item - 1) == len(this_user_ii_instances_embs)
        ii_instances_embs[u] = this_user_ii_instances_embs
    return ii_instances_embs

def load_train_test_data(train_file, test_file):

    train_df = pd.read_csv(train_file, header=None, sep=',')
    train_data = torch.LongTensor(train_df.to_numpy())

    test_df = pd.read_csv(test_file, header=None, sep=',')
    test_data = torch.LongTensor(test_df.to_numpy())  # torch.Size([28074, 7])

    return train_data, test_data
