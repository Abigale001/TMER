#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:Hongxu_ICDM
@author:yicongli
@contact:liyicong123@outlook.com
@file: att.py
@time: 2020/06/10
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ItemAttention(nn.Module):
    def __init__(self, latent_dim, att_size):
        super(ItemAttention, self).__init__()

        self.itemItemPathAttention = PathAttention(att_size=att_size, latent_dim=latent_dim)

        self.dense = nn.Linear(in_features=latent_dim * 2, out_features=att_size)
        nn.init.xavier_normal_(self.dense.weight.data)
        self.lam = lambda x: F.softmax(x, dim=1)

    def forward(self, item_input, ii_path):
        # calculate attention weights among item paths as importance
        # calculate final ii_path emb
        # calculate ii_path and item_input with attention
        one_ii_path = self.itemItemPathAttention(item_input, ii_path)
        inputs = torch.cat((item_input, one_ii_path), 1)
        output = self.dense(inputs)
        output = torch.relu(output)
        atten = self.lam(output)

        output = item_input * atten
        return output

class UserAttention(nn.Module):
    def __init__(self, latent_dim, att_size):
        super(UserAttention, self).__init__()

        self.dense = nn.Linear(in_features=latent_dim * 2, out_features=att_size)
        nn.init.xavier_normal_(self.dense.weight.data)
        self.lam = lambda x: F.softmax(x, dim=1)

    def forward(self, user_input, ui_path):
        inputs = torch.cat((user_input, ui_path), 1)
        output = self.dense(inputs)
        output = torch.relu(output)
        atten = self.lam(output)
        output = user_input * atten
        return output

class PathAttention(nn.Module):

    def __init__(self, att_size, latent_dim):
        super(PathAttention, self).__init__()

        self.att_size = att_size
        self.latent_dim = latent_dim

        self.dense_layer_1 = nn.Linear(in_features=latent_dim * 2, out_features=att_size)
        self.dense_layer_2 = nn.Linear(in_features=att_size, out_features=1)
        nn.init.xavier_normal_(self.dense_layer_1.weight.data)
        nn.init.xavier_normal_(self.dense_layer_2.weight.data)

        self.lam1 = lambda x, index: x[:, index, :]
        self.lam2 = lambda x: F.softmax(x, dim=1)
        self.lam3 = lambda path_latent, atten: torch.sum(path_latent * torch.unsqueeze(atten, -1), 1)

    def forward(self, item_latent, path_latent):
        path_num = path_latent.shape[-2]

        path = self.lam1(path_latent, 0) # get the first path (1,100)
        inputs = torch.cat((item_latent, path), 1)
        output = self.dense_layer_1(inputs)
        output = F.relu(output)

        output = self.dense_layer_2(output)
        output = F.relu(output)


        for i in range(1, path_num):
            path = self.lam1(path_latent, i) # get the i-th path
            inputs = torch.cat((item_latent, path), 1)

            tmp_output = self.dense_layer_1(inputs)
            tmp_output = F.relu(tmp_output)

            tmp_output = self.dense_layer_2(tmp_output)
            tmp_output = F.relu(tmp_output)

            output = torch.cat((output, tmp_output), 1) # paths att score

        atten = self.lam2(output) # normalized paths att score

        output = self.lam3(path_latent, atten)

        return output





class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(model_dim, num_heads*d_k)
        self.w_ks = nn.Linear(model_dim, num_heads*d_k)
        self.w_vs = nn.Linear(model_dim, num_heads*d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0/(model_dim+d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0/(model_dim+d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0/(model_dim+d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(num_heads*d_v, model_dim)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

class Self_Attention_Network(nn.Module):
    def __init__(self, user_item_dim, num_heads=8, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.self_att = MultiHeadAttention(
            model_dim=user_item_dim, num_heads=num_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.feed_forward = nn.Linear(user_item_dim,user_item_dim)

    def forward(self, slf_att_input):
        slf_att_output, att = self.self_att(q=slf_att_input,k=slf_att_input,v=slf_att_input)
        slf_att_output = self.feed_forward(slf_att_output)
        return slf_att_output

class uiPathAtt(nn.Module):
    def __init__(self, user_item_dim, num_heads=8, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.self_att = MultiHeadAttention(
            model_dim=user_item_dim, num_heads=num_heads, d_k=d_k, d_v=d_v, dropout=dropout)
        self.feed_forward = nn.Linear(user_item_dim,user_item_dim)
        self.gMaxPooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, slf_att_input):
        slf_att_output, att = self.self_att(q=slf_att_input,k=slf_att_input,v=slf_att_input)
        slf_att_output = self.feed_forward(slf_att_output)
        maxpool_input = slf_att_output.permute(0,2,1)
        after_maxPooling = self.gMaxPooling(maxpool_input)
        output = after_maxPooling.permute(0, 2, 1)
        return output


class Maxpooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.gMaxPooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, input_tensor):
        maxpool_input = input_tensor.permute(0, 2, 1)
        after_maxPooling = self.gMaxPooling(maxpool_input)
        output = after_maxPooling.permute(0, 2, 1)
        return output
