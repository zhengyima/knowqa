import random
from dataclasses import dataclass

import datasets
from typing import Union, List, Tuple, Dict

import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import os

class BERTPretrainedMLMDataset(Dataset):
    def __init__(
            self,
            train_file,
            tokenizer: PreTrainedTokenizer,
            dataset_script_dir,
            dataset_cache_dir,
            mode
    ):
        # train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        print("start loading datasets, train_files: ", train_files)        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        # self.args = args
        self.mode = mode
        if mode == 'train':
            self.nlp_dataset = datasets.load_dataset(
                f'{dataset_script_dir}/json.py',
                data_files=train_files,
                ignore_verifications=False,
                cache_dir=dataset_cache_dir,
                features=datasets.Features({
                    "masked_lm_labels": [datasets.Value("string")],
                    "masked_lm_positions": [datasets.Value("int32")],
                    "tokens":[datasets.Value("string")],
                    "tokens_idx":[datasets.Value("int32")],
                    "masked_lm_labels_idxs":[datasets.Value("int32")],
                })
            )['train']
        elif mode == "test":
            self.nlp_dataset = datasets.load_dataset(
                f'{dataset_script_dir}/json.py',
                data_files=train_files,
                ignore_verifications=False,
                cache_dir=dataset_cache_dir,
                features=datasets.Features({
                    "masked_lm_positions": [datasets.Value("int32")],
                    "tokens":[datasets.Value("string")],
                    "tokens_idx":[datasets.Value("int32")],
                    "qid":datasets.Value("string")
                })
            )['train']            

        self.total_len = len(self.nlp_dataset)

        print("loading dataset ok! len of dataset,", self.total_len,'mode', mode)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        # print(data)

        encoding = self.tok.encode_plus(data['tokens_idx'], padding='max_length', max_length=128)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']


        labels = np.array([-100] * len(input_ids)) # [-100, ..., -100] len: 200
        if self.mode == "train":
            masked_lm_positions = data['masked_lm_positions']  # [2, 6]
            masked_lm_labels = data['masked_lm_labels_idxs']
            labels[masked_lm_positions] = masked_lm_labels
        data = {
            "input_ids": np.array(list(input_ids)),
            "attention_mask": np.array(list(attention_mask)),
            "labels": np.array(list(labels)), # [5, 9, 11]
        }

        return data