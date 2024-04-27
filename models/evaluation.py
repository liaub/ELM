import os
import re
import sqlite3
import random
import pickle
import networkx as nx
import numpy as np
import torch
import time
from multiprocessing.pool import ThreadPool
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score, recall_score
from models.modified_model.modified_T5 import ModifiedT5ForConditionalGeneration
from transformers.optimization import Adafactor


class EvalutionFinetuner(pl.LightningModule):
    def __init__(self, configs, tokenizer, ent_id_list,  ent_name_list, cuda):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.tokenizer = tokenizer
        self.ent_id_list = ent_id_list
        self.ent_name_list = ent_name_list
        self.T5ForConditionalGeneration = ModifiedT5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
        self.history = {'perf': ..., 'loss': []}
        self.Initialize_random_seeds()
        self.cuda = cuda


    def Initialize_random_seeds(self):
        # 初始化随机种子
        negnum = 0
        self.con = sqlite3.connect(self.configs.dataset_path + '/' + self.configs.dataset + '/' + 'db.db')
        # 获取cursor对象
        self.cur = self.con.cursor()
        # 执行sql创建表
        sql = "select COUNT(*) as num from negative_data"
        cursor = self.cur.execute(sql)
        for row in cursor:
            negnum = row[0]

        # 初始化随机种子
        self.neg_seed = np.ones((negnum,)) * self.configs.epochs
        # 随机数量
        self.sampling_num = 1


    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = self.batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = self.batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = self.batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = self.batchify(data, 'target_mask', padding_value=0)
        return agg_data

    def batchify(self, output_dict, key, padding_value=None, return_list=False):
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

    def postive_function(self,  chain, target, out):
        # 正样本构建
        input_text = 'predict correlation: ('
        for idx, entity in enumerate(eval(chain)):
            if idx + 1 == len(eval(chain)):
                input_text += self.ent_name_list[entity] + ') | '
            else:
                input_text += self.ent_name_list[entity] + ', '

        input_text += "({ch})".format(ch=self.ent_name_list[int(target)])
        target_text = '<extra_id_0>yes<extra_id_1>'
        tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask
        out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                    'target_mask': target_mask})
        return out


    def negative_sampling(self, num, neighbor, neg_start_id, neg_end_id, chain, out):
        # 采样过程
        if num < len(neighbor):
            seed_list = self.neg_seed[neg_start_id: neg_end_id]
            index = np.arange(len(neighbor))
            weights = seed_list / np.sum(seed_list)
            sampling_ids = np.random.choice(a=index, size=num, p=weights, replace=False)
            sampling_data = neighbor[sampling_ids]
            for idx in sampling_ids:
                seed_list[idx] = seed_list[idx] - 1

            self.neg_seed[neg_start_id: neg_end_id] = seed_list
        else:
            sampling_data = neighbor

        input_text = 'predict correlation: ('
        for idx, entity in enumerate(eval(chain)):
            if idx + 1 == len(eval(chain)):
                input_text += self.ent_name_list[entity] + ') | '
            else:
                input_text += self.ent_name_list[entity] + ', '
        for data in sampling_data:
            input_text += "({ch})".format(ch=self.ent_name_list[int(data)])
            target_text = '<extra_id_0>no<extra_id_1>'
            tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
            source_ids = tokenized_src.input_ids
            source_mask = tokenized_src.attention_mask
            tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
            target_ids = tokenized_tgt.input_ids
            target_mask = tokenized_tgt.attention_mask
            out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                        'target_mask': target_mask})
        return out


    def training_step(self, batched_data, batch_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        input = batched_data['input']
        target = batched_data['target']
        # candidate = batched_data['candidate']
        neg_start_id = batched_data['neg_start_id']
        neg_end_id = batched_data['neg_end_id']
        out = []
        for idx, ipt in enumerate(input):
            sql = "select * from negative_data where id >={} and id<{}".format(neg_start_id[idx] + 1, neg_end_id[idx] + 1)
            cursor = self.cur.execute(sql)
            negative = np.array([row[1] for row in cursor])
            # 正样本构建
            out = self.postive_function(ipt, target[idx], out)
            # 负采样过程
            out = self.negative_sampling(self.sampling_num, negative, neg_start_id[idx], neg_end_id[idx], ipt, out)

        agg_data = self.collate_fn(out)

        # target_ids, target_mask, labels: .shape: (batch_size, padded_seq_len)
        source_ids = agg_data['source_ids'].to(self.cuda)
        source_mask = agg_data['source_mask'].to(self.cuda)
        target_ids = agg_data['target_ids'].to(self.cuda)
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100

        # ent_rel .shape: (batch_size, 2)
        source_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(source_ids)

        # 掩码过程
        if self.configs.seq_dropout > 0.:
            rand = torch.rand_like(source_mask.float())
            dropout = torch.logical_not(rand < self.configs.seq_dropout).long().type_as(source_mask)
            dropout[source_ids == 9689] = 1
            dropout[source_ids == 18712] = 1
            dropout[source_ids == 10] = 1
            dropout[source_ids == 41] = 1
            dropout[source_ids == 61] = 1
            dropout[source_ids == 6] = 1
            indices = torch.where(source_ids==1820)[1]
            for id, index in enumerate(indices):
                dropout[id][index:] = 1
            # dropout[source_ids == 1820] = 1
            # dropout[source_ids == 1] = 1
            source_mask = source_mask * dropout


        # batch_size, seq_len, model_dim = inputs_emb.shape
        output = self.T5ForConditionalGeneration(inputs_embeds=source_emb, attention_mask=source_mask, labels=labels, output_hidden_states=True)
        if self.configs.train_style == 1:
            # 方案一：Use only generated loss
            loss = torch.mean(output.loss)
        elif self.configs.train_style == 2:
            # 方案二：Logarithmic contrast loss
            gloss = torch.mean(output.loss)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[:, 1, yes_token_id]
            yprob = yes_score[torch.where(labels[:, 1] == 4273)]  # 4273->yes
            nprob = yes_score[torch.where(labels[:, 1] != 4273)]  # 150->no
            yloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    yloss += torch.log(1 + torch.exp((npb - ypb) * self.configs.lamda))
            yloss = yloss / count
            loss = yloss + gloss
        else:
            # 方案三：Max contrast loss
            gloss = torch.mean(output.loss)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[:, 1, yes_token_id]
            yprob = yes_score[torch.where(labels[:, 1] == 4273)]  # 4273->yes
            nprob = yes_score[torch.where(labels[:, 1] != 4273)]  # 150->no
            yloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    margin = 1
                    yloss = torch.max(torch.tensor([0, npb - ypb + margin]))
            yloss = yloss / count

            loss = yloss + gloss
        self.history['loss'].append(loss.detach().item())
        self.log('val_loss', loss, on_step=True)
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        input = batched_data['input']
        target = batched_data['target']
        # candidate = batched_data['candidate']
        neg_start_id = batched_data['neg_start_id']
        neg_end_id = batched_data['neg_end_id']
        out = []
        for idx, ipt in enumerate(input):
            sql = "select * from negative_data where id >={} and id<{}".format(neg_start_id[idx] + 1,
                                                                               neg_end_id[idx] + 1)
            cursor = self.cur.execute(sql)
            negative = np.array([row[1] for row in cursor])
            # 正样本构建
            out = self.postive_function(ipt, target[idx], out)
            # 负采样过程
            out = self.negative_sampling(self.sampling_num, negative, neg_start_id[idx], neg_end_id[idx], ipt, out)

        agg_data = self.collate_fn(out)

        source_ids = agg_data['source_ids'].to(self.cuda)
        source_mask = agg_data['source_mask'].to(self.cuda)
        target_ids = agg_data['target_ids'].to(self.cuda)
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100

        # generated_text .type: list(str) .len: batch_size * num_beams
        if self.configs.running_model == "test_model":  # 测试准确率
            generated_text, clue_label = self.decode(source_ids, source_mask, target_ids)
            y_true = []
            y_pred = []
            for idx in range(len(clue_label)):
                if clue_label[idx] == "yes":
                    y_true.append(1)
                else:
                    y_true.append(0)
            for idx in range(len(generated_text)):
                if generated_text[idx] == "yes":
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            out = {'y_true': y_true, 'y_pred': y_pred}
        else:  # 存储有效的路径信息
            # 1.找有效实体
            generated_text, scores = self.decode(source_ids, source_mask, target_ids)
            # 2.根据有效实体在图中寻找路径
            data = []
            for idx, gt in enumerate(generated_text):
                query = input[idx]
                tgt = target[idx]
                data.append({"query": query, "target": tgt, "score": scores[idx], "flag": gt})
            # if labels[idx, 1].item() == 4273:
            #     data.append({"query": str(triple), "clue": clue, "score": scores[idx], "flag": "Y"})
            # else:
            #     data.append({"query": str(triple), "clue": clue, "score": scores[idx], "flag": "N"})
            out = data
        return out

    def decode(self, input_ids, input_mask, target_ids):
        def _extract(generated_text):
            compiler = re.compile(r'<extra_id_0>(.*)<extra_id_1>')
            extracted_text = []
            for text in generated_text:
                match = compiler.search(text)
                if match is None:
                    # text = text.strip().lstrip('<pad> <extra_id_0>')
                    extracted_text.append(text.strip())
                else:
                    extracted_text.append(match.group(1).strip())
            return extracted_text

        inputs_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(input_ids)
        input_mask = input_mask

        outputs = self.T5ForConditionalGeneration.generate(inputs_embeds=inputs_emb,
                                                           attention_mask=input_mask,
                                                           max_length=self.configs.eval_tgt_max_length,
                                                           return_dict_in_generate=True,
                                                           output_scores=True,)

        # outputs = self.T5ForConditionalGeneration(inputs_embeds=inputs_emb, attention_mask=input_mask)

        raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs['sequences'])
        generated_text = _extract(raw_generated_text)
        if self.configs.running_model == "test_model":
            raw_label = self.trainer.datamodule.tokenizer.batch_decode(target_ids)
            tgt_label = _extract(raw_label)
        else:
            scores = outputs['scores'][1]  # [seq_len, batch_size, num_labels]
            tgt_label = []
            for score in scores:
                probs = F.softmax(score, dim=-1)
                yes_token_id = self.tokenizer.encode("yes")[0]
                yes_score = probs[yes_token_id]
                tgt_label.append(yes_score.item())

        return generated_text, tgt_label


    def validation_epoch_end(self, outs):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        if self.configs.running_model == "test_model":
            report_f1 = []
            report_acc = []
            for out in outs:
                f1 = f1_score(out['y_true'], out['y_pred'])
                acc = recall_score(out['y_true'], out['y_pred'])
                report_f1.append(f1)
                report_acc.append(acc)
            f1_value = np.sum(np.array(report_f1)) / len(report_f1)
            acc_value = np.sum(np.array(report_acc)) / len(report_acc)
            print("F1值为:{}".format(f1_value))
            print("准确率为:{}".format(acc_value))
        else:
            # storing results
            filename = self.configs.contextual_fact + '/' + self.configs.dataset + '/effective_path'
            with open(filename, 'wb') as fo:
                pickle.dump(outs, fo)
                fo.close()
            print("successful, building history store!")
            print("有效信息识别完成，已经存储到{}".format(filename))

    def test_step(self, batched_data, batch_idx):

        return self.validation_step(batched_data, batch_idx)


    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        if self.configs.optim == 'Adafactor':
            optim = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.configs.lr)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optim
