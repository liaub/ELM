import os
import datetime
import argparse
from datetime import datetime
import pickle
import numpy as np
import  os
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Tokenizer
from transformers import T5Config
from models.evaluation import EvalutionFinetuner
from data import EvaluationDataModule
from helper import read_sample, read_name,  get_next_token_dict, construct_prefix_trie, get_ground_truth
from callbacks import PrintingCallback


def main():
    ## read triples
    trainsets = read_sample(configs, 'train')
    testsets = read_sample(configs, 'eval')
    # testsets = trainsets
    validsets = testsets
    # all_input = train_input + test_input
    # all_target = train_target + test_target

    ## construct name list
    original_ent_name_list = read_name(configs)
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in original_ent_name_list], max_length=configs.train_tgt_max_length, truncation=True).input_ids

    ent_name_list = tokenizer.batch_decode([tokens[1:-2] for tokens in ent_token_ids_in_trie])
    ent_id_list = {ent: idx for idx, ent in enumerate(ent_name_list)}

    filename = 'lm-evaluate-{epoch:02d}-{val_loss:.4f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=configs.save_dir,
        filename=filename,
        mode='min'
    )

    printing_callback = PrintingCallback()

    gpu = [int(configs.gpu)] if torch.cuda.is_available() else 0
    trainer_params = {
        'gpus': gpu,
        'limit_train_batches': 0.001,  # 限制训练模型的batch数
        'max_epochs': configs.epochs,  # 1000
        'checkpoint_callback': True,  # True
        'logger': False,  # TensorBoardLogger
        'num_sanity_val_steps': 0,  # 模型训练开始前，提前验证模型是否能跑起来
        'check_val_every_n_epoch': 1, # 每n个epoch验证模型
        'enable_progress_bar': True, # 使用进度条
        'callbacks': [
            printing_callback,
            checkpoint_callback
        ],
    }
    trainer = pl.Trainer(**trainer_params)

    kw_args = {
        'ent_id_list': ent_id_list,
        'ent_name_list': ent_name_list,
        'cuda': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }


    if configs.model_path == '' and configs.running_model == 'train_model':

        datamodule = EvaluationDataModule(configs, trainsets, validsets, testsets, running_model='train_model')
        print('train_model datamodule construction done.', flush=True)
        model = EvalutionFinetuner(configs, tokenizer, **kw_args)
        trainer.fit(model, datamodule)
        model_path = checkpoint_callback.best_model_path
        print('training best model path:', model_path, flush=True)

    else:
        model_path = configs.model_path
        model_name = configs.model_name
        datamodule = EvaluationDataModule(configs, trainsets, validsets, testsets, running_model='test_model')
        model = EvalutionFinetuner.load_from_checkpoint(model_path + model_name, strict=False, configs=configs, **kw_args)
        trainer.test(model, dataloaders=datamodule)


if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='./data')
    parser.add_argument('-dataset', dest='dataset', default='NYT', help='Dataset to use, NYT, ICEWS14')
    parser.add_argument('-model', default='T5Finetuner', help='Model Name')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')

    parser.add_argument('-pretrained_model', type=str, default='./models/t5-base', help='')
    parser.add_argument('-train_style', default=2, type=int, help='Training style->1:Use only generated loss; 2:Logarithmic contrast loss, 3:Max contrast loss')
    parser.add_argument('-batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-train_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-eval_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.0005, help='Starting Learning Rate')
    parser.add_argument('-lamda', default=15, type=int, help='Boundary interval constant of positive and negative samples')
    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    # parser.add_argument('-model_path', dest='model_path', default='/home/shu/product/liaub/INSEP/checkpoint/NYT-train_model/', help='The path for reloading models')
    # parser.add_argument('-model_name', dest='model_name', default='lm-evaluate-epoch=01-val_loss=0.0993.ckpt', help='The path for reloading models')
    parser.add_argument('-optim', default='Adam', type=str, help='')
    parser.add_argument('-skip_n_val_epoch', default=1000, type=int, help='Using train process')
    # parser.add_argument('-skip_n_val_epoch', default=0, type=int, help='Using test process')
    parser.add_argument('-seq_dropout', default=0., type=float, help='0-1, a larger value indicates a larger mask')
    parser.add_argument('-running_model', type=str, default='train_model', help='[train_model, test_model, predict_model]')
    configs = parser.parse_args()
    configs.vocab_size = T5Config.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = T5Config.from_pretrained(configs.pretrained_model).d_model
    if configs.save_dir == '' and configs.running_model == 'train_model': # if train and valid, makedires else not makedires
        # configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model-' + str(datetime.now())) # use liunx
        configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model')  # use windows
        os.makedirs(configs.save_dir, exist_ok=True)

    # if configs.error_store == '' and configs.running_model == 'test_model': # if train and valid, makedires else not makedires
    #     # configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model-' + str(datetime.now())) # use liunx
    #     configs.error_store = os.path.join('./checkpoint', configs.dataset + '-error_store')  # use windows
    #     os.makedirs(configs.error_store, exist_ok=True)

    pl.seed_everything(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()
