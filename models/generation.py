import os
import re
import pickle
import random
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import sqlite3
import pytorch_lightning as pl
from models.modified_model.modified_T5 import ModifiedT5ForConditionalGeneration
from transformers.optimization import Adafactor
from collections import Counter
from helper import get_performance


class GenerationFinetuner(pl.LightningModule):
    def __init__(self, configs, tokenizer, name_list_dict, prefix_trie_dict=None, evalution_model=None, cuda=None):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.tokenizer = tokenizer
        self.ent_name_list = name_list_dict['ent_name_list']
        self.ent_id_list = name_list_dict['ent_id_list']
        self.event_ids = name_list_dict['event_ids']
        self.prefix_trie = prefix_trie_dict['prefix_trie']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.next_token_dict = prefix_trie_dict['next_token_dict']
        self.all_ground_truth = name_list_dict['all_ground_truth']
        self.evalution_model = evalution_model
        self.T5ForConditionalGeneration = ModifiedT5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
        self.conn = sqlite3.connect(configs.dataset_path + '/' + configs.dataset + '/' + 'db.db')
        # 获取cursor对象
        self.cursor = self.conn.cursor()
        self.cuda = cuda


        self.history = {'perf': ..., 'loss': []}

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


    def training_step(self, batched_data, batch_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        '''
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['tgt_ids'] = [dt['tgt_ids'] for dt in data]
        agg_data['event_chain'] = [dt['event_chain'] for dt in data]
        '''
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        target_ids = batched_data['target_ids']
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100
        source_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(src_ids)
        output = self.T5ForConditionalGeneration(inputs_embeds=source_emb, attention_mask=src_mask, labels=labels, output_hidden_states=True)
        # 你的工作
        '''
        loss = loss1 + loss2 + torch.mean(output.loss)
        '''
        loss = torch.mean(output.loss)
        self.history['loss'].append(loss.detach().item())
        self.log('val_loss', loss, on_step=True)
        return {'loss': loss}
    def validation_step(self, batched_data, batch_idx):
        '''
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['tgt_ids'] = [dt['tgt_ids'] for dt in data]
        agg_data['event_chains'] = [dt['event_chains'] for dt in data]
        '''
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        ranks = []
        batched_event_chains = batched_data['event_chains']
        batched_targets = batched_data['targets']
        for (event_chains, targets) in zip(batched_event_chains, batched_targets):
            self.event_chains = event_chains
            rank = 0
            for id, target in enumerate(targets):
                # 超过预测step, 自动跳出
                if id + 1 > self.configs.next_step:
                    break
                input_txt = 'Query:('
                for idx, ent in enumerate(self.event_chains):
                    if idx + 1 == len(self.event_chains):
                        input_txt += self.ent_name_list[ent] + ', [MASK]).'
                    else:
                        input_txt += self.ent_name_list[ent] + ', '
                out = []
                ent_ids = self.event_ids.copy()
                ent_ids.remove(target)
                ent_ids = np.array(ent_ids)
                sampling_info = ent_ids[np.random.choice(len(ent_ids), self.configs.candi_count-1, replace=False)]
                candidate = np.append(sampling_info, [target])
                np.random.shuffle(candidate)
                if self.configs.style == 0:  # event prediction
                    input_txt += ' Canidate:('
                    for idx, candi in enumerate(candidate):
                        if idx + 1 == len(candidate):
                            input_txt += self.ent_name_list[candi] + ').'
                        else:
                            input_txt += self.ent_name_list[candi] + ', '

                target_txt = '<extra_id_0>' + self.ent_name_list[target] + '<extra_id_1>'

                tokenized_src = self.tokenizer(input_txt, max_length=self.configs.src_max_length, truncation=True)
                source_ids = tokenized_src.input_ids
                source_mask = tokenized_src.attention_mask
                tokenized_tgt = self.tokenizer(target_txt, max_length=self.configs.train_tgt_max_length, truncation=True)
                target_ids = tokenized_tgt.input_ids
                target_mask = tokenized_tgt.attention_mask
                target_names = self.ent_name_list[target]

                out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                            'target_mask': target_mask})

                agg_data = self.collate_fn(out)

                src_ids = agg_data['source_ids'].to(self.cuda)
                src_mask = agg_data['source_mask'].to(self.cuda)
                target_ids = agg_data['target_ids'].to(self.cuda)
                self.tgt_ids = target
                labels = target_ids.clone()
                labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100

                # generated_text .type: list(str) .len: batch_size * num_beams
                generated_text, sequences_scores = self.decode(src_ids, src_mask)
                if self.configs.using_evaluation == True:
                    finy = []
                    for i in range(len(generated_text)):
                        predict_tgt = generated_text[i]
                        predict_score = sequences_scores[i].item()
                        # 评估机制
                        eval_text, eval_score = self.evalution_model.reason(self.event_chains, predict_tgt)
                        score = 0.3*predict_score + 0.7*eval_score
                        finy.append({"tgt_name": predict_tgt, "score": score})

                    sorted_dicts = sorted(finy, key=lambda x: x['score'], reverse=True)
                    generated_text = []
                    sequences_scores = []
                    if self.configs.style == 0:
                        for dicts in sorted_dicts:
                            if self.ent_id_list[dicts["tgt_name"]] in candidate:
                                generated_text.append(dicts["tgt_name"])
                                sequences_scores.append(dicts["score"])
                    else:
                        for r, dicts in enumerate(sorted_dicts):
                            generated_text.append(dicts["tgt_name"])
                            sequences_scores.append(dicts["score"])
                else:
                    sequences_scores.tolist()
                    template_generated_text = []
                    template_sequences_scores = []
                    if self.configs.style == 0:
                        for (gt_name, seq_score) in zip(generated_text, sequences_scores):
                            if self.ent_id_list[gt_name] in candidate:
                                template_generated_text.append(gt_name)
                                template_sequences_scores.append(seq_score)
                    generated_text = template_generated_text
                    sequences_scores = template_sequences_scores

                # 筛选rank点
                prd_tgt = None
                if target_names != generated_text[0]:
                    if target_names in generated_text:
                        for gt_name in generated_text:
                            if gt_name == target_names:
                                prd_tgt = target_names
                                rank += 1
                                break
                            else:
                                rank += 1
                    else:
                        rank = 1000
                else:
                    prd_tgt = target_names
                    if rank == 0 and id + 1 == self.configs.next_step:
                        rank = 1

                    # if id + 1 == len(targets):
                    #     print("保存最终结果")
                    #     rank = 1
                    # else:
                    #     print("不保存")
                #     if target_names == generated_text[0]:

                # if generated_text[0]
                # sql = "INSERT INTO predict_results (query, predict, target, score, rank, flag) VALUES ('{}', '{}', '{}', '{}', {}, '{}')".format(candi)
                # self.cursor.execute(sql)
                # hr_key = tuple(self.event_chains)
                # scores =sequences_scores
                # all_gt_ids = self.all_ground_truth[hr_key]
                # all_gt_seqs = [self.ent_name_list[ids] for ids in all_gt_ids]
                # ## get rank
                # if target_names in generated_text:
                #     top_entities = set()
                #     rank = 1
                #     tobreak = False
                #     for j, text in enumerate(generated_text):
                #         score = scores[j]
                #         if text == target_names:
                #             ranks.append(rank)
                #             tobreak = True
                #             sql = "INSERT INTO predict_results (query, predict, target, score, rank, flag) VALUES ('{}', '{}', '{}', '{}', {}, '{}')".format(
                #                 str(self.event_chains), text, target_names, score, j + 1, 'Generator')
                #             self.cursor.execute(sql)
                #
                #         else:
                #             sql = "INSERT INTO predict_results (query, predict, target, score, rank, flag) VALUES ('{}', '{}', '{}', '{}', {}, '{}')".format(
                #                 str(self.event_chains), text, target_names, score, j + 1, 'Generator')
                #             self.cursor.execute(sql)
                #
                #         if not tobreak:
                #             if text in set(self.ent_name_list) and (text not in all_gt_seqs) and (text not in top_entities):
                #                 top_entities.add(text)
                #                 rank += 1
                # else:
                #     value = random.randint(self.configs.num_beams + 1, self.configs.n_ent)
                #     ranks.append(value)
                #     for j, text in enumerate(generated_text):
                #         score = scores[j]
                #         sql = "INSERT INTO predict_results (query, predict, target, score, rank, flag) VALUES ('{}', '{}', '{}', '{}', {}, '{}')".format(
                #             str(self.event_chains), text, target_names, score, j + 1, 'Generator')
                #         self.cursor.execute(sql)
                if rank <= 10:
                    if prd_tgt != None:
                        self.event_chains = self.event_chains + [self.ent_id_list[prd_tgt]]
                    else:
                        rank = 1000
                        break
                else:
                    break

            ranks.append(rank)

        out = {'ranks': ranks}
        return out
    def decode(self, src_ids, src_mask):
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

        def _next_candidate(batch_idx, input_ids):
            hr_key = tuple(self.event_chains)
            all_gt_ids = self.all_ground_truth[hr_key]
            # ent_token_ids_in_trie = self.ent_token_ids_in_trie_with_descrip if configs.tgt_descrip_max_length > 0 else self.ent_token_ids_in_trie
            ent_token_ids_in_trie = self.ent_token_ids_in_trie
            all_gt_seq = [tuple(ent_token_ids_in_trie[ids]) for ids in all_gt_ids]

            pred_ids = tuple(ent_token_ids_in_trie[self.tgt_ids])

            input_ids = input_ids.tolist()
            if input_ids[0] == 0:
                input_ids = input_ids[1:]

            if tuple(input_ids) in self.next_token_dict:
                if len(input_ids) == 0:
                    return [32099]
                if input_ids[-1] == 32098:
                    return [1]
                next_tokens = self.next_token_dict[tuple(input_ids)]
                all_gt_seq = [seq for seq in all_gt_seq if tuple(seq[: len(input_ids)]) == tuple(input_ids)]
                gt_next_tokens = Counter([seq[len(input_ids)] for seq in all_gt_seq if len(input_ids) < len(seq)])
                if tuple(pred_ids[: len(input_ids)]) == tuple(input_ids) and len(input_ids) < len(pred_ids):
                    pred_id = Counter([pred_ids[len(input_ids)]])
                else:
                    pred_id = Counter([])
                next_tokens = list(set(next_tokens - gt_next_tokens + pred_id))
                return next_tokens
            else:
                return []

        num_beam_groups = self.configs.num_beam_groups if self.configs.decoder == 'diverse_beam_search' else 1
        diversity_penalty = self.configs.diversity_penalty if self.configs.decoder == 'diverse_beam_search' else 0.
        prefix_allowed_tokens_fn = lambda batch_idx,input_ids: _next_candidate(batch_idx,input_ids) if self.configs.use_prefix_search else None

        source_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(src_ids)
        outputs = self.T5ForConditionalGeneration.generate(inputs_embeds=source_emb,
                                                           attention_mask=src_mask,
                                                           return_dict_in_generate=True,
                                                           num_return_sequences=self.configs.num_beams,
                                                           max_length=self.configs.eval_tgt_max_length,
                                                           diversity_penalty=diversity_penalty,
                                                           num_beam_groups=num_beam_groups,
                                                           prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                           num_beams=self.configs.num_beams,
                                                           bos_token_id=0,
                                                           output_scores=True,)

        # sequences = outputs['sequences']
        sequences_scores = F.softmax(outputs['sequences_scores'], dim=-1)
        # outputs.scores
        # scores = outputs['scores'][1]  # [seq_len, batch_size, num_labels]
        raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs.sequences)
        generated_text = _extract(raw_generated_text)


        assert len(generated_text) == self.configs.num_beams * len(src_ids)
        return generated_text, sequences_scores

    def validation_epoch_end(self, outs):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            # self.log('val_mrr', 0.00)
            return

        # if self.current_epoch + 1 == self.configs.epoch:
        # print("当前epoch为{}".format(self.current_epoch))
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        pred_tail_out = outs
        agg_tail_out = dict()
        for out in pred_tail_out:
            for key, value in out.items():
                if key in agg_tail_out:
                    agg_tail_out[key] += value
                else:
                    agg_tail_out[key] = value

        tail_ranks = agg_tail_out['ranks']
        del agg_tail_out['ranks']

        perf = get_performance(self, tail_ranks)

        print(perf)

    def test_step(self, batched_data, batch_idx):
        # print("test_step==============================")
        return self.validation_step(batched_data, batch_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        if self.configs.optim == 'Adafactor':
            optim = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.configs.lr)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optim


class EvalutionFinetuner(pl.LightningModule):
    def __init__(self, configs, tokenizer, ent_id_list,  ent_name_list, cuda):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.tokenizer = tokenizer
        self.ent_id_list = ent_id_list
        self.ent_name_list = ent_name_list
        self.cuda = cuda
        self.T5ForConditionalGeneration = ModifiedT5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
        self.history = {'perf': ..., 'loss': []}


    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = self.batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = self.batchify(data, 'source_mask', padding_value=0)
        # agg_data['target_ids'] = self.batchify(data, 'target_ids', padding_value=0)
        # agg_data['target_mask'] = self.batchify(data, 'target_mask', padding_value=0)
        # agg_data['event_chains'] = [dt['event_chains'] for dt in data]
        # agg_data['tgt_ids'] = [dt['tgt_ids'] for dt in data]
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


    def decode(self, input_ids, input_mask):
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
                                                           output_scores=True, )

        # outputs = self.T5ForConditionalGeneration(inputs_embeds=inputs_emb, attention_mask=input_mask)

        raw_generated_text = self.tokenizer.batch_decode(outputs['sequences'])
        predict_text = _extract(raw_generated_text)
        # if self.configs.running_model == "test_model":
        #     raw_label = self.trainer.datamodule.tokenizer.batch_decode(target_ids)
        #     tgt_label = _extract(raw_label)
        # else:
        scores = outputs['scores'][1]  # [seq_len, batch_size, num_labels]
        predict_score = []
        for score in scores:
            probs = F.softmax(score, dim=-1)
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[yes_token_id]
            predict_score.append(yes_score.item())

        return predict_text, predict_score
    def reason(self, event_chains, predict_tgt):
        out = []
        input_txt = 'predict correlation: ('
        for idx, entity in enumerate(event_chains):
            if idx + 1 == len(event_chains):
                input_txt += self.ent_name_list[entity] + ') | '
            else:
                input_txt += self.ent_name_list[entity] + ', '

        input_txt += "({ch})".format(ch=predict_tgt)

        tokenized_src = self.tokenizer(input_txt, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        # tokenized_tgt = self.tokenizer(target_txt, max_length=self.configs.train_tgt_max_length, truncation=True)
        # target_ids = tokenized_tgt.input_ids
        # target_mask = tokenized_tgt.attention_mask
        # target_names = self.ent_name_list[target]

        out.append({'source_ids': source_ids, 'source_mask': source_mask})

        agg_data = self.collate_fn(out)

        source_ids = agg_data['source_ids'].to(self.cuda)
        source_mask = agg_data['source_mask'].to(self.cuda)

        predict_text, predict_score = self.decode(source_ids, source_mask)

        return predict_text[0], predict_score[0]
        # source_ids = agg_data['source_ids'].to(self.cuda)
        # source_mask = agg_data['source_mask'].to(self.cuda)
        # target_ids = agg_data['target_ids'].to(self.cuda)
        # labels = target_ids.clone()
        # labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100
        # generated_text, scores = self.decode(source_ids, source_mask, target_ids)
