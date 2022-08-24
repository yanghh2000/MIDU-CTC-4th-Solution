#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

import torch
from src import logger
from src.baseline.modeling import ModelingCtcBert
from src.baseline.tokenizer import CtcTokenizer
from src.baseline.ctc_vocab.config import VocabConf


class PredictorCtc:
    def __init__(
        self,
        in_model_dirs,
        ctc_label_vocab_dir='src/baseline/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
        batch_size=32,
        keep_bias=0,
        model_weights=None
    ):
        self.batch_size = batch_size
        self.in_model_dirs = in_model_dirs

        self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)

        self.use_cuda = use_cuda
        self.keep_bias = keep_bias
        self.model_weights = model_weights

        if self.use_cuda and torch.cuda.is_available():
            if cuda_id is not None:
                torch.cuda.set_device(cuda_id)
        #     self.model.cuda()
        #     self.model.half()
        # self.model.eval()

        # 多个模型集成
        self.models = []
        # self.tokenizers = []

        for path in in_model_dirs:
            model = ModelingCtcBert.from_pretrained(path)
            # print('================bert================')
            # print(model.bert)
            # print('======================ModelingCtcBert======================')
            # print(model)
            # print('================loaded_state_dict_keys================')
            # loaded_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')
            # loaded_state_dict_keys = loaded_state_dict.keys()
            # print(loaded_state_dict_keys)
            # print('================expected_state_dict_keys================')
            # model_state_dict = model.state_dict()
            # expected_keys = model_state_dict.keys()
            # print(expected_keys)
            # missing_keys = list(set(expected_keys) - set(loaded_state_dict_keys))
            # print('missing keys: ', missing_keys)
            # unused_keys = list(set(loaded_state_dict_keys) - set(expected_keys))
            # print('unused keys: ', unused_keys)

            logger.info('model loaded from dir {}'.format(path))
            model.cuda()
            model.half()
            model.eval()
            self.models.append(model)
            # self.tokenizers.append(tokenizer)
        self.tokenizer = CtcTokenizer.from_pretrained(in_model_dirs[0])      # 使用同一个词表

        try:
            self._start_vocab_id = self.tokenizer.vocab['[START]']
        except KeyError:
            self._start_vocab_id = self.tokenizer.vocab['[unused1]']

    def load_label_dict(self, ctc_label_vocab_dir):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def id_list2ctag_list(self, id_list) -> list:

        return [self._id2ctag[i] for i in id_list]

    @torch.no_grad()
    def predict(self, texts, return_topk=1):
        batch_size = self.batch_size
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = texts
        outputs = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]

            batch_texts = [' ' + t for t in batch_texts]  # 开头加一个占位符
            inputs = self.tokenizer(batch_texts,
                                    return_tensors='pt')

            # 把 ' ' 换成 _start_vocab_id
            inputs['input_ids'][..., 1] = self._start_vocab_id
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            preds_all_models = []
            for model in self.models:           # 模型集成
                d_preds, c_preds, loss = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                )

                c_preds = torch.softmax(c_preds[:, 1:, :], dim=-1)  # 从cls后面开始
                preds_all_models.append(c_preds)

            c_preds_ensemble = torch.zeros_like(c_preds)

            if self.model_weights is None:
                self.model_weights = [1/len(preds_all_models)] * len(preds_all_models)
            for c_preds, weight in zip(preds_all_models, self.model_weights):
                c_preds_ensemble += weight * c_preds / sum(self.model_weights) # 多个模型集成，计算概率平均

            batch_size, sequence_length, _ = c_preds_ensemble.size()
            if self.keep_bias:          # 给$KEEP增加一个置信度bias
                prob_offset = [self.keep_bias] + [0] * (VocabConf.correct_vocab_size - 1)       # $KEEP 标签index为0
                prob_offset = torch.FloatTensor(prob_offset).repeat((batch_size, sequence_length, 1)).cuda()
                c_preds_ensemble += prob_offset

            recall_top_k_probs, recall_top_k_ids = c_preds_ensemble.topk(
                k=return_topk, dim=-1, largest=True, sorted=True)
            recall_top_k_probs = recall_top_k_probs.tolist()
            recall_top_k_ids = recall_top_k_ids.tolist()
            recall_top_k_chars = [[self.id_list2ctag_list(
                char_level) for char_level in sent_level] for sent_level in recall_top_k_ids]
            batch_texts = [['']+list(t)[1:] for t in batch_texts]  # 占位符
            batch_outputs = [list(zip(text, top_k_char, top_k_prob)) for text, top_k_char, top_k_prob in zip(
                batch_texts, recall_top_k_chars, recall_top_k_probs)]
            outputs.extend(batch_outputs)
        return outputs

    @staticmethod
    def output2text(output):

        pred_text = ''
        for src_token, pred_token_list, pred_prob_list in output:
            pred_token = pred_token_list[0]
            pred_prob = pred_prob_list[0]
            # if 'UNK' in pred_token:         # 将UNK标签替换为KEEP，否则UNK标签默认删除对应token
            #     pred_token = '$KEEP'
            if '$KEEP' in pred_token:
                pred_text += src_token
            elif pred_prob < 0:         # 纠错标签概率小于0.2，不纠错（选择KEEP）
                pred_text += src_token
            elif '$DELETE' in pred_token:
                continue
            elif '$REPLACE' in pred_token:
                pred_text += pred_token.split('_')[-1]
            elif '$APPEND' in pred_token:
                pred_text += src_token+pred_token.split('_')[-1]

        return pred_text