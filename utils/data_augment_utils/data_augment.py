import os
import sys
proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_path)

from tqdm import tqdm
import numpy as np
import argparse
import re
import json
import pypinyin
from pypinyin import pinyin
from utils.data_augment_utils.data_augment_dicts.dataAUG_util import load_json_file, load_pkl_file
from utils.data_augment_utils.text_clean import traditional2simplified
from multiprocessing import Pool
# import torch
# ctx = torch.multiprocessing.get_context("spawn")
# import hanlp
# tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH, devices=0)
import pynlpir
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/yanghh/MuCGEC-main/plm/chinese-roberta-wwm-ext')
from utils.data_augment_utils.data_augment_dicts.dataAUG_util import is_sent_ok
from functools import partial

confusion_sets = None

def get_synonyms_or_simChar(input_word, confusion_dict, is_word_same_len=False):
    '''
    随机从大词林的近义词中采样替换
    :param input_word:
    :param confusion_dict:
    :param is_same_len: 是否取同样长度的近义词
    :return:
    '''
    index_ids = confusion_dict['index_id']
    index_vals = confusion_dict['index_val']
    tgt_token = None
    if input_word in index_ids:
        index_id = index_ids[input_word]
        index_val = index_vals[str(index_id)]
        if is_word_same_len:
            index_val = [word for word in index_val if len(word)==len(input_word)]
        tgt_token = np.random.choice(index_val)
    return tgt_token

def get_homo_pinyin(input_token, input_item, homoPY_confusion_set, is_word_same_len=False):
    '''
    随机从词级别同音混淆集中采样替换

    :param input_word:
    :param homoPY_confusion_set:
    :return:
    '''
    tgt_token = None
    if input_item in homoPY_confusion_set:
        same_pin_yin = homoPY_confusion_set[input_item]
        if is_word_same_len:
            same_pin_yin = [word for word in same_pin_yin if len(word) == len(input_token)]
        tgt_token = np.random.choice(same_pin_yin)
    return tgt_token

def get_pinyin_index(input_token, charORword:str):
    """得到输入token 的拼音混淆集索引表示：eg: 字：zi; 喜欢：xi_huan"""
    pinyin_index = None
    _pinyin = pinyin(input_token, style=pypinyin.NORMAL)
    # 如果创建的是word dict,则只保留词语(2字以上)
    if charORword == "word":
        if len(_pinyin) > 1:
            pinyin_index = '_'.join([py[0] for py in _pinyin])
    else:
        pinyin_index = _pinyin[0][0]
    return pinyin_index


def replace(input_token, grain, confusion_sets, is_word_same_len=False):
    '''
    进行替换操作

    word_grain: 误用（近义词）； 拼写（同音词）
    char_grain: 形似（形似字）； 音似（音似字）

    :param input_token: 需要替换的token
    :param grain: 替换粒度 word or char
    :param confusion_sets: 各种混淆集
    :return:
    '''
    tgt_token = None
    if grain == 'word':
        if is_word_same_len:
            specific_edit = np.random.choice(['misuse', 'spell'], p=[0, 1])     # 构造拼写错误数据
        else:
            specific_edit = np.random.choice(['misuse', 'spell'], p=[0.1, 0.9])
        if specific_edit == 'misuse':
            synonyms_dict = confusion_sets['synonyms']
            tgt_token = get_synonyms_or_simChar(input_token, synonyms_dict, is_word_same_len=is_word_same_len)
        elif specific_edit == 'spell':
            word_homoPY_dict = confusion_sets['word_homoPY']
            pinyin_index = get_pinyin_index(input_token, charORword='word')
            tgt_token = get_homo_pinyin(input_token, pinyin_index, word_homoPY_dict, is_word_same_len=is_word_same_len)
    elif grain == 'char':
        specific_edit = np.random.choice(['sim_char', 'homoPY'], p=[0.1, 0.9])
        if specific_edit == 'sim_char':
            simChar_dict = confusion_sets['simChar']
            tgt_token = get_synonyms_or_simChar(input_token, simChar_dict)
        elif specific_edit == 'homoPY':
            char_homoPY_dict = confusion_sets['char_homoPY']
            pinyin_index = get_pinyin_index(input_token, charORword='char')
            tgt_token = get_homo_pinyin(input_token, pinyin_index, char_homoPY_dict)
    else:
        raise ValueError("params grain must in [word, char]")

    return tgt_token


def append(input_seq, src_token, src_ind, tgt_ind, grain, confusion_sets):
    '''
    进行添加增强

    char_level: 啰嗦： 将当前字复制并插入到附近某个位置
    word_level: 啰嗦(50%)： 将当前单词复制并插入到附近某个位置
                语义重复(50%)： 随机从大词林的近义词中采样插入
    :param input_seq: 输入子序列
    :param src_token: 进行增加操作的原始词语
    :param tgt_ind: 子序列中进行增加的目标位置
    :param grain: 添加操作粒度
    :param confusion_set: 混淆集合
    :return:
    '''
    if grain == "char":
        specific_edit = np.random.choice(['verbose', 'semantic_repetitions'], p=[1.0, 0])
    else:
        specific_edit = np.random.choice(['verbose', 'semantic_repetitions'], p=[0.5, 0.5])

    if specific_edit == 'verbose':
        input_seq[tgt_ind] += src_token
    else:
        synonyms_dict = confusion_sets['synonyms']
        tgt_token = get_synonyms_or_simChar(src_token, synonyms_dict)
        if tgt_token:
            input_seq[src_ind] += tgt_token  # 语义重复型添加：添加在原始词后
    return input_seq


def augment_step(sub_seq:list, confusion_sets:dict, grain:str, fix_edit:str, is_word_same_len=False):
    '''
    进行一次增强

    :param tokens: 分词后的句子
    :param tgt_index: 进行增强的索引位置 token_index
    :param confusion_sets: 所需的所有混淆集
    :return:
    '''
    pattern = re.compile('^[\u4e00-\u9fa5]+')
    if grain == 'char':
        chinese_tokens = {ind:token for ind, token in enumerate(sub_seq) if pattern.match(token)}
    else:
        chinese_tokens = {ind:token for ind, token in enumerate(sub_seq) if pattern.match(token) and len(token) > 1}
    if chinese_tokens:
        src_ind = np.random.choice(list(chinese_tokens.keys()))
        src_token = chinese_tokens[src_ind]
        # 获取原词附近的某个进行添加或乱序的目标位置 暂定为：5—gram
        central_ind = list(chinese_tokens.keys()).index(src_ind)
        left_interval_value = central_ind - 2 if central_ind - 2 > 0 else 0
        right_interval_value = central_ind + 2 if central_ind + 2 < len(sub_seq) else len(sub_seq)
        tgt_ind = np.random.choice(
            list(chinese_tokens.keys())[left_interval_value: right_interval_value])  # 保证添加的token 在原词附近

        edit_choose = np.random.choice(['replace', 'delete', 'append', 'move'], p=[0.7, 0.15, 0.1, 0.05])
        if fix_edit:
            edit_choose = fix_edit

        if edit_choose == 'replace':
            tgt_token = replace(src_token, grain, confusion_sets, is_word_same_len=is_word_same_len)
            if tgt_token:
                sub_seq[src_ind] = tgt_token
            return sub_seq
        elif edit_choose == 'delete':
            if grain == 'char':
                sub_seq[src_ind] = ''
            else:
                sub_seq[src_ind] = src_token.replace(np.random.choice(list(src_token)), '')  # 随机从当前单词中删除 1 个字
        elif edit_choose == 'append':
            sub_seq = append(sub_seq, src_token, src_ind, tgt_ind, grain, confusion_sets)
        elif edit_choose == 'move':  # 随机将当前字/词与附近某个字交换位置
            tgt_token = sub_seq[tgt_ind]
            sub_seq[src_ind] = tgt_token
            sub_seq[tgt_ind] = src_token
    return sub_seq


def augment(input_data, word_prob=0.5, fix_grain=None, fix_edit=None, is_word_same_len=False):
    """增强的main函数"""
    if not is_sent_ok(input_data):
        return input_data, None

    src_data = input_data
    src_data = traditional2simplified(src_data.strip())
    words = pynlpir.segment(src_data, pos_tagging=False)

    # 含有超过10个词语的句子，每10个词语为一区间，做一次增强
    data_augment = []
    if len(words) > 10:
        for i in range(0, len(words), 10):
            grain_choose = 'word' if np.random.random() < word_prob else 'char'
            sub_words_seq = words[i: i + 10]
            # sub_chars_seq = tokenizer.tokenize(''.join(sub_words_seq))    # 不能用tokenizer，会产生[UNK]
            sub_chars_seq = list(''.join(sub_words_seq))
            if fix_grain:
                grain_choose = fix_grain

            if grain_choose == 'word':
                sub_seq = sub_words_seq
            else:
                sub_seq = sub_chars_seq
            sub_seq_aug = augment_step(sub_seq, confusion_sets, grain_choose, fix_edit, is_word_same_len=is_word_same_len)

            data_augment.extend(sub_seq_aug)

    else:  # 不到10个词语的句子，随机位置做一次增强
        grain_choose = 'word' if np.random.random() < word_prob else 'char'
        if fix_grain:
            grain_choose = fix_grain

        if grain_choose == 'word':
            sub_seq = words
        else:
            # sub_seq = tokenizer.tokenize(''.join(words))
            sub_seq = list(''.join(words))

        data_augment = augment_step(sub_seq, confusion_sets, grain_choose, fix_edit, is_word_same_len=is_word_same_len)

    # print(f"{grain_choose}_level")
    # print(f"原始句子：{src_data}\n增强后的：{''.join(data_augment)}")
    return src_data, ''.join(data_augment)


def augment_test(input_data:list, confusion_set:dict, fix_grain=None, fix_edit=None):
    '''
    增强程序测试

    :param input_data: ['要有信心,就要客观的正视自己,了解考试内容和要求。','',] 测试样例
    :param confusion_set: 混淆集
    :param fix_grain: 增强的粒度 word / char
    :param fix_edit: 增强的操作 replace/ delete/ append/ move
    :return:
    '''
    for inp_dt in input_data:
        res = augment(inp_dt, confusion_set, fix_grain, fix_edit)
        print(res)


def data_aug(args):
    np.random.seed(123)

    data_file = os.path.join(args.input_dir, f'{args.file_index}th.pkl')
    out_file = os.path.join(args.output_dir, f'{args.file_index}.txt')

    # print('input file: ', data_file)
    # print('output file: ', out_file)

    src_dataset = load_pkl_file(data_file)

    write_list = []

    afunc = partial(augment, word_prob=args.word_prob, fix_grain=None, fix_edit=args.fix_edit, is_word_same_len=args.is_word_same_len)
    print('word_prob: ', args.word_prob)
    print('fix_edit: ', args.fix_edit)
    print('is_word_same_len: ', args.is_word_same_len)

    # for src in tqdm(src_dataset):
    #     src_data, aug_sent = afunc(src)
    #     if aug_sent:
    #         write_list.append(aug_sent + '\t' + src_data + '\n')
    # print(len(write_list))
    with Pool(args.worker_num) as pool:
        for src_data, aug_sent in pool.imap(afunc, tqdm(src_dataset), chunksize=8):
            if aug_sent:
                write_list.append(aug_sent + '\t' + src_data + '\n')
    with open(out_file, 'w', encoding='utf-8') as tgt_f:
        tgt_f.writelines(write_list)

def check_gec_data(data_file):
    '''
    检查数据 去除含有字母或数字或空格多于10%的句子 后的比例
    :param data_file:
    :return:
    '''
    count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            sents = line.strip()
            if sents:
                if is_sent_ok(sents.split('\t')[1]):
                    count += 1
        print(count / len(lines))

def check_csc_data(data_file):
    '''
    检查数据 去除含有字母或数字或空格多于10%的句子 后的比例
    :param data_file:
    :return:
    '''
    count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            sents = line.strip()
            if sents:
                s = sents.split('\t')
                src, trg = s[0], s[1]
                if len(src) == len(trg):
                    count += 1
                else:
                    print(src, trg, sep='\t')
        print(count / len(lines))

def futher_clean_gec_data(data_file):
    '''
    删除含有字母或数字或空格多于10%的句子
    :param data_file:
    :return:
    '''
    result = []
    count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            sents = line.strip()
            if sents:
                try:
                    if is_sent_ok(sents.split('\t')[1]):
                        count += 1
                        result.append(line)
                except Exception as e:
                    print(sents)
        print(count / len(lines))
    with open(f'{data_file}.clean', 'w', encoding='utf-8') as o:
        o.writelines(result)

def futher_clean_csc_data(data_file):
    '''
    删除含有字母或数字或空格多于10%的句子
    :param data_file:
    :return:
    '''
    result = []
    count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            sents = line.strip()
            if sents:
                try:
                    s = sents.split('\t')
                    src, trg = s[0], s[1]
                    if len(src) == len(trg):
                        count += 1
                        result.append(line)
                except Exception as e:
                    print(sents)
        print(count / len(lines))
    with open(f'{data_file}.clean', 'w', encoding='utf-8') as o:
        o.writelines(result)

def load_confusion_dict(args):
    synonyms_dict = load_json_file(args.synonyms_dict)
    char_homoPY_dict = load_json_file(args.char_homoPY_confusion_file)
    word_homoPY_dict = load_json_file(args.word_homoPY_confusion_file)
    simChar_dict = load_json_file(args.sim_char_file)

    confusion_sets = {'synonyms':synonyms_dict, 'word_homoPY':word_homoPY_dict,
                      'simChar':simChar_dict, 'char_homoPY':char_homoPY_dict}

    return confusion_sets

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global confusion_sets
    confusion_sets = load_confusion_dict(args)        # 加载所有混淆集字典

    # gec 数据增强
    pynlpir.open()      # 中科院分词nlpir

    if args.aug_type == 'gec':
        # 增强拼写错误数据时，augment函数word_prob为0.3，fix_edit为'replace'，is_word_same_len为True
        # 增强语法错误数据时，word_prob为0.5，fix_edit为None，is_word_same_len为False
        args.word_prob = 0.5
        args.fix_edit = None
        args.is_word_same_len = False

        for i in range(1, args.file_num + 1):
            args.file_index = i
            data_aug(args)
            futher_clean_gec_data(os.path.join(args.output_dir, f'{i}.txt'))    # 清洗增强文本
            check_gec_data(os.path.join(args.output_dir, f'{i}.txt.clean'))

            if os.path.exists(os.path.join(args.output_dir, f'{i}.txt')):
                os.remove(os.path.join(args.output_dir, f'{i}.txt'))

    else:
        # csc 数据增强
        args.word_prob = 0.2
        args.fix_edit = 'replace'
        args.is_word_same_len = True
        for i in range(1, args.file_num + 1):
            args.file_index = i
            data_aug(args)
            futher_clean_csc_data(os.path.join(args.output_dir, f'{i}.txt'))
            check_csc_data(os.path.join(args.output_dir, f'{i}.txt.clean'))

            if os.path.exists(os.path.join(args.output_dir, f'{i}.txt')):
                os.remove(os.path.join(args.output_dir, f'{i}.txt'))

    pynlpir.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        help='Path to the train data',
                        default='data/news2016zh/split_train')
    parser.add_argument('--output_dir',
                        help='Path to the target data',
                        default='data/news2016zh/aug_train_gec')
    parser.add_argument('--file_num',
                        help='切分文件个数',
                        type=int,
                        default=10)
    parser.add_argument('--file_index',
                        default='1')
    parser.add_argument('--synonyms_dict',
                        help='同义词典文件',
                        default='utils/data_augment_utils/data_augment_dicts/confusion_sets/synonyms.json')
    parser.add_argument('--char_homoPY_confusion_file',
                        help='字符同音混淆集',
                        default='utils/data_augment_utils/data_augment_dicts/confusion_sets/homoPY_char.json')
    parser.add_argument('--word_homoPY_confusion_file',
                        help='词语同音混淆集',
                        default='utils/data_augment_utils/data_augment_dicts/confusion_sets/homoPY_word.json')
    parser.add_argument('--sim_char_file',
                        help='同形字混淆集',
                        default='utils/data_augment_utils/data_augment_dicts/confusion_sets/simChar.json')
    parser.add_argument('--worker_num',
                        help='进程数',
                        type=int,
                        default=16)
    parser.add_argument('--aug_type',
                        help='数据增强类型',
                        type=str,
                        choices=['gec', 'csc'],
                        default='gec')

    args = parser.parse_args()

    main(args)

    # from opencc import OpenCC
    #
    # cc = OpenCC("t2s")
    # import json
    # data = json.load(open('data/preliminary_a_data/preliminary_train.json', 'r', encoding='utf-8'))
    # src_lines = [d['source'] for d in data]
    # trg_lines = [d['target'] for d in data]
    #
    # write_lines = [cc.convert(src.strip()) + '\t' + cc.convert(trg.strip()) + '\n' for src, trg in zip(src_lines, trg_lines)]
    #
    # with open('data/preliminary_a_data/preliminary_train.txt', 'w', encoding='utf-8') as f:
    #     f.writelines(write_lines)
    #
    # check_gec_data('data/preliminary_a_data/preliminary_train.txt')
    # futher_clean_gec_data(f'data/preliminary_a_data/preliminary_train.txt')
    # check_gec_data('data/preliminary_a_data/preliminary_train.txt.clean')

    # with open('data/preliminary_a_data/preliminary_train.txt.clean', 'r', encoding='utf-8') as f:
    #     lines1 = f.readlines()
    #
    # with open('data/preliminary_a_data/preliminary_train_my_aug.txt.clean', 'r',
    #           encoding='utf-8') as f:
    #     lines2 = f.readlines()
    #
    # all = lines1 + lines2
    #
    # with open('data/preliminary_a_data/preliminary_train_all.txt.clean', 'w', encoding='utf-8') as o:
    #     o.writelines(all)


