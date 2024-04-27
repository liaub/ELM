import pickle
import numpy as np
import sqlite3
class event_predict():
    def __init__(self):
        self.dataset = "NYT"
        self.next_step = 1
        self.candi_count = 5  # 候选集的个数，如果是-1代表所有实体都是候选集
        self.dataset_dir = "./data/" + self.dataset
        self.train_data = self.load_sequences('train_sequence.pkl')
        self.test_data = self.load_sequences('test_sequence.pkl')
        self.dev_data = self.load_sequences('dev_sequence.pkl')
        self.corpus = self.load_corpus('corpus_verb_mapping.pkl')
        self.event_ids = [_ for _ in range(len(self.corpus))]
        self.conn = sqlite3.connect(self.dataset_dir + '/db.db')
        self.cursor = self.conn.cursor()
        # 删除之前的数据
        self.cursor.execute("DELETE from negative_data")
        self.conn.commit()
        self.cursor.execute("DELETE from retrieve_query")
        self.conn.commit()
        self.cursor.execute("update sqlite_sequence set seq = 0 where name = 'negative_data'")
        self.conn.commit()
        self.cursor.execute("update sqlite_sequence set seq = 0 where name = 'retrieve_query'")
        self.conn.commit()


    def load_sequences(self, name):
        with open(self.dataset_dir + '/' +name, 'rb') as f:
            sequence = pickle.load(f)

        return sequence

    def load_corpus(self, name):
        with open(self.dataset_dir + '/' +name, 'rb') as f:
            corpus = pickle.load(f)
        return corpus


    def generate_datasets(self, name):
        if name == 'train':
            sequence = self.train_data + self.dev_data
            event_ids = self.event_ids
        elif name == 'eval':
            sequence = self.test_data
            event_ids = self.event_ids
        else:
            sequence = self.test_data
            event_ids = self.event_ids
            self.next_step = 1

        # sequence = sequence[0:1000]

        neg_idx = 0
        neg_record = 0
        for sp in range(self.next_step):
            step = sp + 1  # 预测的步数
            event_seqs = np.array(sequence)[:, :-step]
            labels = np.array(sequence)[:, -step]
            for idx, lab in enumerate(labels):
                input = event_seqs[idx]
                target = lab
                if self.candi_count > 0:
                    ent_ids = event_ids.copy()
                    ent_ids.remove(lab)
                    ent_ids = np.array(ent_ids)
                    sampling_info = ent_ids[np.random.choice(len(ent_ids), self.candi_count-1, replace=False)]
                    list_b = np.append(sampling_info, [lab])
                else:
                    list_b = event_ids
                np.random.shuffle(list_b)
                # candidate.append(list_b)
                for candi in list_b:
                    sql = "INSERT INTO negative_data (entity) VALUES ('{}')".format(candi)
                    self.cursor.execute(sql)
                    neg_idx = neg_idx + 1
                if self.candi_count > 0:
                    candidate = list_b
                else:
                    candidate = [-1]
                sql = "INSERT INTO retrieve_query (query,target,canidate,neg_start_idx,neg_end_idx,flag) VALUES ('{}','{}','{}',{},{},'{}')".format(
                    str(input.tolist()), target, str(candidate.tolist()), neg_record, neg_idx, name)
                self.cursor.execute(sql)
                neg_record = neg_idx

                print("第{}步, 总共{}条数据, 正在处理第{}条".format(sp+1, len(labels), idx+1))

        if name == 'test':
            self.conn.commit()
            print("数据插入成功")
            self.conn.close()




    def load_datasets(self, name):
        with open(self.dataset_dir + '/' +name, 'rb') as f:
            sequence = pickle.load(f)

        return sequence


    def convert_sequences(self, name):
        sequences = []
        if name == 'train':
            dataset = self.train_data
        else:
            dataset = self.test_data

        for idx, (data, label) in enumerate(zip(dataset[0], dataset[1])):
            conbinx = data + [label]
            sequences.append(conbinx)
            self.corpus_verb_mapping_list = self.corpus_verb_mapping_list + conbinx
            print("共{},正在处理第{}个".format(len(dataset[0]), idx+1))

        with open(self.dataset_dir + '/{}_sequence.pkl'.format(name), 'wb') as fo:
            pickle.dump(sequences, fo)
            fo.close()

    def generate_corpus_verb_mapping(self):
        self.corpus_verb_mapping_list = set(self.corpus_verb_mapping_list)
        new_corpus_verb_mapping_list = []
        for corpus in self.corpus_verb_mapping_list:
            item_name = 'Item {}'.format(corpus)
            new_corpus_verb_mapping_list.append(item_name)

        with open(self.dataset_dir + '/corpus_verb_mapping.pkl', 'wb') as fo:
            pickle.dump(new_corpus_verb_mapping_list, fo)
            fo.close()


if __name__ == '__main__':
    event = event_predict()
    event.generate_datasets('train')
    event.generate_datasets('eval')
    event.generate_datasets('test')








