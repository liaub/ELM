import torch
import pytorch_lightning as pl
from transformers import T5Tokenizer
from helper import batchify
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, List

class Generation_TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, trainsets, name_list_dict, prefix_trie_dict):
        self.configs = configs
        self.trainsets = trainsets
        self.tokenizer = tokenizer
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.neg_candidate_mask = prefix_trie_dict['neg_candidate_mask']

    def __len__(self):
        return len(self.trainsets[0])

    def __getitem__(self, index):
        input, target, candidate = self.trainsets[0][index], self.trainsets[1][index], self.trainsets[2][index]
        input_txt = 'Query:('
        for idx, ent in enumerate(input):
            if idx + 1 == len(input):
                input_txt += self.original_ent_name_list[ent] + ', [MASK]).'
            else:
                input_txt += self.original_ent_name_list[ent] + ', '

        if self.configs.style == 0: # event prediction
            input_txt += ' Canidate:('
            for idx, candi in enumerate(candidate):
                if idx + 1 == len(candidate):
                    input_txt += self.original_ent_name_list[candi] + ').'
                else:
                    input_txt += self.original_ent_name_list[candi] + ', '
        target_txt = '<extra_id_0>' + self.original_ent_name_list[target] + '<extra_id_1>'

        tokenized_src = self.tokenizer(input_txt, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(target_txt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask
        target_names = self.ent_name_list[target]

        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'tgt_ids': target,
            'target_names': target_names,
            'event_chains': input,
            'candidate': candidate
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['tgt_ids'] = [dt['tgt_ids'] for dt in data]
        agg_data['event_chains'] = batchify(data, 'event_chains', return_list=True)
        agg_data['candidate'] = batchify(data, 'candidate', return_list=True)
        return agg_data

class Generation_TestDataset(Dataset):
    def __init__(self, configs, tokenizer, testsets, name_list_dict, prefix_trie_dict):  # mode: {tail, head}
        self.configs = configs
        self.testsets = testsets
        self.tokenizer = tokenizer
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.next_step = name_list_dict['next_step']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.neg_candidate_mask = prefix_trie_dict['neg_candidate_mask']

    def __len__(self):
        return len(self.testsets[0])

    def __getitem__(self, index):
        input, target = self.testsets[0][index], self.testsets[1][index]
        step = self.next_step - 1
        event_chains = input[:-step]
        targets = input[-step:] + [target]
        # input_txt = 'Query:('
        # for idx, ent in enumerate(input):
        #     if idx + 1 == len(input):
        #         input_txt += self.original_ent_name_list[ent] + ', [MASK]).'
        #     else:
        #         input_txt += self.original_ent_name_list[ent] + ', '
        #
        # if self.configs.style == 0:  # event prediction
        #     input_txt += ' Canidate:('
        #     for idx, candi in enumerate(candidate):
        #         if idx + 1 == len(candidate):
        #             input_txt += self.original_ent_name_list[candi] + ').'
        #         else:
        #             input_txt += self.original_ent_name_list[candi] + ', '
        # target_txt = '<extra_id_0>' + self.original_ent_name_list[target] + '<extra_id_1>'
        #
        # tokenized_src = self.tokenizer(input_txt, max_length=self.configs.src_max_length, truncation=True)
        # source_ids = tokenized_src.input_ids
        # source_mask = tokenized_src.attention_mask
        # tokenized_tgt = self.tokenizer(target_txt, max_length=self.configs.train_tgt_max_length, truncation=True)
        # target_ids = tokenized_tgt.input_ids
        # target_mask = tokenized_tgt.attention_mask
        # target_names = self.ent_name_list[target]
        #
        # out = {
        #     'source_ids': source_ids,
        #     'source_mask': source_mask,
        #     'target_ids': target_ids,
        #     'target_mask': target_mask,
        #     'tgt_ids': target,
        #     'target_names': target_names,
        #     'event_chains': input,
        #     'candidate': candidate
        # }

        out = {
            'event_chains': event_chains,
            'targets': targets
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['event_chains'] = batchify(data, 'event_chains', return_list=True)
        agg_data['targets'] = batchify(data, 'targets', return_list=True)
        return agg_data

class GenerationDataModule(pl.LightningDataModule):
    def __init__(self, configs, trainsets, validsets, testsets, name_list_dict, prefix_trie_dict, running_model='train_model'):
        super().__init__()
        self.configs = configs
        self.trainsets = trainsets
        self.validsets = validsets
        self.testsets = testsets
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.prefix_trie_dict = prefix_trie_dict
        self.running_model = running_model

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = Generation_TrainDataset(self.configs, self.tokenizer, self.trainsets, self.name_list_dict, self.prefix_trie_dict)
        self.valid_both = Generation_TestDataset(self.configs, self.tokenizer, self.validsets, self.name_list_dict, self.prefix_trie_dict)
        self.test_both = Generation_TestDataset(self.configs, self.tokenizer, self.testsets, self.name_list_dict, self.prefix_trie_dict)


    def train_dataloader(self):

        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_both,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_both.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_both,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_both.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)

        return test_loader


class Evaluation_TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, trainsets):
        self.configs = configs
        self.trainsets = trainsets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.trainsets)

    def __getitem__(self, index):
        input, target, candidate, neg_start_id, neg_end_id = self.trainsets[index][1], self.trainsets[index][2], self.trainsets[index][3], self.trainsets[index][4], self.trainsets[index][5]

        out = {
            'input': input,
            'target': target,
            'candidate': candidate,
            'neg_start_id': neg_start_id,
            'neg_end_id': neg_end_id,
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['input'] = [dt['input'] for dt in data]
        agg_data['target'] = [dt['target'] for dt in data]
        agg_data['candidate'] = [dt['candidate'] for dt in data]
        agg_data['neg_start_id'] = [dt['neg_start_id'] for dt in data]
        agg_data['neg_end_id'] = [dt['neg_end_id'] for dt in data]
        return agg_data

class Evaluation_TestDataset(Dataset):
    def __init__(self, configs, tokenizer, testsets):  # mode: {tail, head}
        self.configs = configs
        self.testsets = testsets
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.testsets)

    def __getitem__(self, index):
        input, target, candidate, neg_start_id, neg_end_id = self.testsets[index][1], self.testsets[index][2], \
        self.testsets[index][3], self.testsets[index][4], self.testsets[index][5]
        out = {
            'input': input,
            'target': target,
            'candidate': candidate,
            'neg_start_id': neg_start_id,
            'neg_end_id': neg_end_id,
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['input'] = [dt['input'] for dt in data]
        agg_data['target'] = [dt['target'] for dt in data]
        agg_data['candidate'] = [dt['candidate'] for dt in data]
        agg_data['neg_start_id'] = [dt['neg_start_id'] for dt in data]
        agg_data['neg_end_id'] = [dt['neg_end_id'] for dt in data]
        return agg_data

class EvaluationDataModule(pl.LightningDataModule):
    def __init__(self, configs, trainsets, validsets, testsets, running_model='train_model'):
        super().__init__()
        self.configs = configs
        self.trainsets = trainsets
        self.validsets = validsets
        self.testsets = testsets
        # ent_name_list, rel_name_list .type: list
        self.running_model = running_model

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = Evaluation_TrainDataset(self.configs, self.tokenizer, self.trainsets)
        self.valid_both = Evaluation_TestDataset(self.configs, self.tokenizer, self.validsets)
        self.test_both = Evaluation_TestDataset(self.configs, self.tokenizer, self.testsets)


    def train_dataloader(self):

        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_both,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_both.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_both,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_both.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)

        return test_loader
