import os
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict as ddict
from collections import Counter
import numpy as np
import pandas as pd
import torch
import pickle
import sqlite3
import re
import networkx as nx
from torch.nn.utils.rnn import pad_sequence
import pygtrie
import argparse
import sys
import torch.nn.functional as F




def read(configs, cur, name):
    inputs = []
    targets = []
    candidates = []

    # 执行sql创建表
    sql = "select * from retrieve_query where flag='{}'".format(name)
    cursor = cur.execute(sql)
    for row in cursor:
        inputs.append(eval(row[1]))
        targets.append(int(row[2]))
        candidates.append(eval(row[3]))

    with open(configs.dataset_path + '/' + configs.dataset + '/corpus_verb_mapping.pkl', 'rb') as f:
        corpus = pickle.load(f)

    configs.n_ent = len(corpus)
    return inputs, targets, candidates, corpus

def read_sample(configs, name):
    datasets = []
    con = sqlite3.connect(configs.dataset_path + '/' + configs.dataset + '/' + 'db.db')
    # 获取cursor对象
    cur = con.cursor()
    # 执行sql创建表
    sql = "select * from retrieve_query where flag='{}'".format(name)
    cursor = cur.execute(sql)
    for row in cursor:
        datasets.append(row)

    # with open(configs.dataset_path + '/' + configs.dataset + '/corpus_verb_mapping.pkl', 'rb') as f:
    #     corpus = pickle.load(f)
    return datasets

def read_name(configs):
    with open(configs.dataset_path + '/' + configs.dataset + '/corpus_verb_mapping.pkl', 'rb') as f:
        ent_name_list = pickle.load(f)
    return ent_name_list



def load_factruples(dataset_path, dataset, fileName):
    with open(os.path.join(dataset_path, dataset, fileName), 'r') as fr:
        quadrupleList = []
        for id, line in enumerate(fr):
            if id == 0:
                continue
            line_split = line.split()
            head = int(line_split[0])
            rel = int(line_split[2])
            tail = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])

    return np.array(quadrupleList)


def get_ground_truth(all_input, all_target):
    ground_truth = ddict(list)
    for idx, input in enumerate(all_input):
        target = all_target[idx]
        ground_truth[tuple(input)].append(target)

    return ground_truth

def get_next_token_dict(configs, ent_token_ids_in_trie, prefix_trie):
    neg_candidate_mask = []
    next_token_dict = {(): [32099] * configs.n_ent}
    for ent_id in tqdm(range(configs.n_ent)):
        rows, cols = [0], [32099]
        input_ids = ent_token_ids_in_trie[ent_id]
        for pos_id in range(1, len(input_ids)):
            cur_input_ids = input_ids[:pos_id]
            if tuple(cur_input_ids) in next_token_dict:
                cur_tokens = next_token_dict[tuple(cur_input_ids)]
            else:
                seqs = prefix_trie.keys(prefix=cur_input_ids)
                cur_tokens = [seq[pos_id] for seq in seqs]
                next_token_dict[tuple(cur_input_ids)] = Counter(cur_tokens)
            cur_tokens = list(set(cur_tokens))
            rows.extend([pos_id] * len(cur_tokens))
            cols.extend(cur_tokens)
        sparse_mask = sp.coo_matrix(([1] * len(rows), (rows, cols)), shape=(len(input_ids), configs.vocab_size), dtype=np.long)
        neg_candidate_mask.append(sparse_mask)
    return neg_candidate_mask, next_token_dict

def construct_prefix_trie(ent_token_ids_in_trie):
    trie = pygtrie.Trie()
    for input_ids in ent_token_ids_in_trie:
        trie[input_ids] = True
    return trie

def batchify(output_dict, key, padding_value=None, return_list=False):
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    if padding_value is None:
        tensor_out = torch.stack(tensor_out, dim=0)
    else:
        tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
    return tensor_out

def _get_performance(ranks, dataset, next_step):
    ranks = np.array(ranks, dtype=np.float)
    out = dict()
    out['mr'] = ranks.mean(axis=0)
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    if next_step > 1:
        out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    else:
        out['hit10'] = 1.00
    # if dataset == 'NELL':
    #     out['hit5'] = np.sum(ranks <= 5, axis=0) / len(ranks)
    return out

def get_performance(model, tail_ranks):
    tail_out = _get_performance(tail_ranks, model.configs.dataset, model.configs.next_step)
    mr = np.array([tail_out['mr']])
    mrr = np.array([tail_out['mrr']])
    hit1 = np.array([tail_out['hit1']])
    hit3 = np.array([tail_out['hit3']])
    hit10 = np.array([tail_out['hit10']])

    if model.configs.dataset == 'NELL':
        val_mrr = tail_out['mrr'].item()
        model.log('val_mrr', val_mrr)
        hit5 = np.array([tail_out['hit5']])
        perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@5': hit5, 'hit@10': hit10}
    else:
        val_mrr = mrr.mean().item()
        model.log('val_mrr', val_mrr)
        perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail ranking'])
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf