import csv
import os
import json
import mindspore
import pandas as pd
from mindnlp.transformers import BertForSequenceClassification, BertModel, BertTokenizer
from mindnlp._legacy.amp import auto_mixed_precision
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, context
from mindnlp._legacy.engine import Trainer, Evaluator
from mindnlp._legacy.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp._legacy.metrics import Accuracy


class SentimentDataset:
    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def load_accusation_id_map(self, filename):
        # 使用pandas读取CSV文件
        df = pd.read_csv(filename)
        # 将DataFrame转换为字典，其中'Accusation'列作为键，'id'列作为值
        accusation_id_map = df.set_index('Accusation')['id'].to_dict()
        return accusation_id_map

    def _load(self):
        labelss = self.load_accusation_id_map('../data/label1.csv')
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                case = json.loads(line.strip())  # 解析每一行作为一个单独的JSON对象
                # if case['meta']['accusation'][0] in labelss:

                label = labelss[case['meta']['accusation'][0]]
                text_a = case['fact']
                self._labels.append(label)
                self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)


def process_dataset(source, tokenizer, max_seq_len=64, batch_size=64, shuffle=True):
    is_ascend = mindspore.get_context('device_target') == 'gpu'

    column_names = ["label", "text_a"]

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    type_cast_op = transforms.TypeCast(mindspore.int32)

    def tokenize_and_pad(text):
        if is_ascend:
            tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
        else:
            # tokenized = tokenizer(text)
            tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
        return tokenized['input_ids'], tokenized['attention_mask']

    # map dataset
    dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a",
                          output_columns=['input_ids', 'attention_mask'])
    dataset = dataset.map(operations=[type_cast_op], input_columns="label", output_columns='labels')
    # batch dataset
    if is_ascend:
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                             'attention_mask': (None, 0)})

    return dataset


tokenizer = BertTokenizer.from_pretrained('../pretrained/')

dataset_train = process_dataset(
    SentimentDataset('E:/Data/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_train.json'), tokenizer)
dataset_val = process_dataset(
    SentimentDataset("E:/Data/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_valid.json"), tokenizer)
dataset_test = process_dataset(
    SentimentDataset("E:/Data/CAIL2018_ALL_DATA/final_all_data/exercise_contest/data_test.json"), tokenizer,
    shuffle=False)

model = BertForSequenceClassification.from_pretrained('../pretrained/')
param_dict = mindspore.load_checkpoint('./checkpoint/bert_emotect_best.ckpt')
model = auto_mixed_precision(model, 'O1')

metric = Accuracy()
# define callbacks to save checkpoints
ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='bert_emotect', epochs=1, keep_checkpoint_max=203)
best_model_cb = BestModelCallback(save_path='checkpoint', ckpt_name='bert_emotect_best', auto_load=True)

evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
evaluator.run(tgt_columns="labels")


