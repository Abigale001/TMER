#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@project:Hongxu_ICDM
@file: meta_path_instances_representation.py
@time: 2020/06/08
"""
import sys
sys.path.append("../../../")
from gensim.models import Word2Vec
from data.data_utils import *
import torch
from torch import nn
import numpy as np
import pickle

"""
ui_path_vectors:{

}
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Autoencoder(nn.Module):
    def __init__(self, d_in=2000, d_hid=800, d_out=100):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(True),
            nn.Linear(d_hid, d_out),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_hid),
            nn.ReLU(True),
            nn.Linear(d_hid, d_in),
            nn.ReLU(True))

    def forward(self, x):
        self.embeddings = self.encoder(x)
        xx = self.decoder(self.embeddings)
        return xx

    def save_embeddings(self):
        return self.embeddings

def instance_emb(metapath_file, output_file):
    walks = get_instance_paths(metapath_file)
    path_dict = instance_paths_to_dict(metapath_file)

    print("Training...")
    model = Word2Vec(walks, size=100, window=3, min_count=0, sg=1, hs=1,
                     workers=1)

    # mean pooling
    ui_path_vectors = {}
    for ui, ui_paths in path_dict.items():
        for path in ui_paths:
            nodes_vectors = []
            for nodeid in path:
                nodes_vectors.append(model.wv[nodeid])
            nodes_np = np.array(nodes_vectors)
            path_vector = np.mean(nodes_np, axis=0)
            if ui not in ui_path_vectors.keys():
                ui_path_vectors[ui] = [path_vector]
            else:
                ui_path_vectors[ui].append(path_vector)
    pickle.dump(ui_path_vectors, open(output_file, 'wb'))


if __name__ == '__main__':
    ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    ii_metapaths_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
    # ii form
    data_base_folder = '../../Amazon_Music/'
    # embed ui paths
    for metapath in ui_metapaths_list:
        metapath_file = data_base_folder + 'path/all_ui_ii_instance_paths/' + metapath + '.paths'
        output_file = data_base_folder + 'path/meta_path_instances_representation/' + metapath + '.wv'
        instance_emb(metapath_file, output_file)
    # embed ii paths
    ii_instance_file = data_base_folder + 'path/all_ui_ii_instance_paths/ii_random_form.paths'
    output_ii_emb_file = data_base_folder + 'path/meta_path_instances_representation/ii_random_form.wv'
    # we randomly select 1 path from 7 item-item instances to generate 'this ii_random_form.wv', and then the following attention
    instance_emb(ii_instance_file, output_ii_emb_file)


