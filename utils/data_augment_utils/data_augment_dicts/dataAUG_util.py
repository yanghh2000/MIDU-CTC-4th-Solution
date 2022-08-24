import os
import sys
proj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_path)

import re
import json
import pickle
import argparse
from copy import deepcopy
from utils.data_augment_utils.text_clean import stringQ2B, remove_illegal_chars, is_sent_ok, split_sent, traditional2simplified
import pypinyin
from pypinyin import pinyin
from utils.helpers import write_lines
from tqdm import tqdm
from collections import defaultdict
from utils.data_augment_utils.helpers import get_allFiles_in_dir

def get_homophones_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(chr(i))
    return result


def get_homophones_by_pinyin(input_pinyin):
    """
    根据拼音取同音字
    :param input_pinyin:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(chr(i))
    return result


def read_weixin_corpus(data_file, max_len, output_dir, num):
    '''
    读取weixin corpus并抽取content 写入tgt_file
    :param data_file: 原始weixin语料
    :param max_len: 需要的文本最大长度
    :param tgt_file: 写入的目标文件
    :return:
    '''
    wx_dataset = []
    line_count = 0
    print("reading file.....................")
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            content = json.loads(line.strip())['content']
            if content:
                # sents = content.split('\n')
                # for sent in sents:
                #     sent = stringQ2B(sent)
                #     sent = remove_illegal_chars(sent)
                #     sent = traditional2simplified(sent)
                #     if is_sent_ok(sent):
                #         wx_dataset.append(sent)
                content = content.replace('\n', '')
                line_count += 1
                content = stringQ2B(content)
                content = remove_illegal_chars(content)
                if len(content) > max_len:
                    _content = content.split('。')
                    for _cont in _content:
                        _cont += '。'
                        if is_sent_ok(_cont):
                            if len(_cont) > max_len:
                                split_sents = split_sent(_cont, min_len=10)
                                if split_sents:
                                    wx_dataset.extend(split_sents)
                            else:
                                wx_dataset.append(_cont)
                else:
                    if is_sent_ok(content):
                        wx_dataset.append(content)

    print('data size before split: ', line_count)
    print(f"data size after split::{len(wx_dataset)}")

    # tgt_f = open(tgt_file, 'wb')
    # pickle.dump(wx_dataset, tgt_f)

    # split big dataset into smalls
    sub_corpus_size = len(wx_dataset) // num
    print("spliting dataset into small ones..............")
    for i in tqdm(range(num)):
        sub_corpus = wx_dataset[i * sub_corpus_size: (i + 1) * sub_corpus_size]
        print('subset size: ', len(sub_corpus))
        sub_corpus_f = open(os.path.join(output_dir, f'{i + 1}th.pkl'), 'wb')
        pickle.dump(sub_corpus, sub_corpus_f)

def read_and_split_corpus(data_file, max_len, output_dir, num):
    '''
    读取weixin corpus并抽取content 写入tgt_file
    :param data_file: 原始weixin语料
    :param max_len: 需要的文本最大长度
    :param tgt_file: 写入的目标文件
    :return:
    '''
    wx_dataset = []
    line_count = 0

    with open(data_file, 'r', encoding='utf-8') as f:
        print("reading file.....................")
        for line in tqdm(f):
            content = json.loads(line.strip())['content']
            if content:
                line_count += 1
                content = stringQ2B(content)
                content = remove_illegal_chars(content)
                if len(content) > max_len:
                    _content = content.split('。')
                    for _cont in _content:
                        _cont += '。'
                        if is_sent_ok(_cont):
                            if len(_cont) > max_len:
                                split_sents = split_sent(_cont, min_len=10)
                                if split_sents:
                                    wx_dataset.extend(split_sents)
                            else:
                                wx_dataset.append(_cont)
                else:
                    if is_sent_ok(content):
                        wx_dataset.append(content)

    print('data size before split: ', line_count)
    print(f"data size after split::{len(wx_dataset)}")

    # write_lines(os.path.join(output_dir, 'all.txt'), wx_dataset, mode='w')

    # split big dataset into smalls
    sub_corpus_size = len(wx_dataset) // num
    print("spliting dataset into small ones..............")
    for i in tqdm(range(num)):
        sub_corpus = wx_dataset[i * sub_corpus_size: (i + 1) * sub_corpus_size]
        print('subset size: ', len(sub_corpus))
        sub_corpus_f = open(os.path.join(output_dir, f'{i + 1}th.pkl'), 'wb')
        pickle.dump(sub_corpus, sub_corpus_f)


def split_corpus(input_file, output_dir, num):
    """手动多线程 pool下 hanlp 会出问题？"""
    input_f = open(input_file, 'rb')
    input_dataset = pickle.load(input_f)
    sub_corpus_size = int(len(input_dataset) / num)
    print("spliting dataset into small ones..............")
    for i in range(num):
        if i == num-1:
            sub_corpus = input_dataset[i * sub_corpus_size: ]
        else:
            sub_corpus = input_dataset[i*sub_corpus_size: (i+1)*sub_corpus_size]
        sub_corpus_f = open(os.path.join(output_dir, f'{i+1}th.pkl'), 'wb')
        pickle.dump(sub_corpus, sub_corpus_f)


def get_corpus_dict(data_file, output_file, min_word_freq, charORword:str, batch_size:int=None):
    '''
    获得语料对应的词典 或 字典

    :param data_file:
    :param min_word_freq: 最小统计词频
    :param charORword: choice '_char' | 'word'
    :return:
    '''
    input_f = open(data_file, 'rb')
    input_dataset = pickle.load(input_f)
    statis_dict = defaultdict(int)
    print("creating dict......................")
    batch_input = []
    for line in tqdm(input_dataset):
        if line.strip():
            batch_input.append(line.strip())
            if len(batch_input) == batch_size:
                if charORword == 'word':
                    batch_tokens = tokenizers[charORword](batch_input)
                    for bt_tokens in batch_tokens:
                        for token in bt_tokens:
                            statis_dict[token] += 1
                else:
                    tokens = tokenizers[charORword].tokenize(batch_input[0])
                    for token in tokens:
                        statis_dict[token] += 1
                batch_input = []

    corpus_dict = {item[0]: item[1] for item in sorted(statis_dict.items(), key=lambda x: x[1], reverse=True)
                    if item[1] >= min_word_freq}
    print(f"corpus dict size:{len(corpus_dict)}")

    output_f = open(output_file, 'w', encoding='utf-8')
    json.dump(corpus_dict, output_f, ensure_ascii=False)


def merge_dict(data_dir):
    """合并拆分数据集对应的词典"""
    all_files = get_allFiles_in_dir(data_dir)
    merged_dict_file = os.path.join(data_dir, 'merged_dict.json')
    sub_dict_files = [i for i in all_files if i.endswith('json')]
    merged_dict = defaultdict(int)
    print('merging sub_corpus dict..............')
    for sub_dict_f in sub_dict_files:
        sub_dict = open(sub_dict_f, 'r')
        _sub_dict = json.load(sub_dict)
        for _key in _sub_dict:
            merged_dict[_key] += _sub_dict[_key]

    merged_dict_f = open(merged_dict_file, 'w', encoding='utf-8')
    _merged_dict = {item[0]: item[1] for item in sorted(merged_dict.items(), key=lambda x: x[1], reverse=True)}
    print(f"final dict size:{len(_merged_dict)}")
    json.dump(_merged_dict, merged_dict_f, ensure_ascii=False)


def convert_char_homoPY_confusionSet_format(pinyin_confusion_set_file, norm_file):
    """将已有的字符拼音混淆集合 格式统一为： {pinyin:[chars]...} 同时考虑模糊音：zhang zang"""
    pinyin_confusion_set_f = open(pinyin_confusion_set_file, 'r', encoding='utf-8')
    pinyin_confusion_set = json.load(pinyin_confusion_set_f)
    new_pinyin_confusion_set = defaultdict(list)
    for _char in pinyin_confusion_set:
        _pinyin = pinyin(_char, style=pypinyin.NORMAL)
        assert len(_pinyin) == 1
        _pinyin_str = _pinyin[0][0]
        new_pinyin_confusion_set[_pinyin_str] = list(set(new_pinyin_confusion_set[_pinyin_str].extend(pinyin_confusion_set[_char])))

    new_pinyin_confusion_set = merge_fuzzySounds_for_char(new_pinyin_confusion_set)

    norm_f = open(norm_file, 'w', encoding='utf-8')
    json.dump(new_pinyin_confusion_set, norm_f, ensure_ascii=False)

    return new_pinyin_confusion_set


def merge_fuzzySounds_for_char(pinyin_confusion_set:dict):
    """将字符拼音混淆集中模糊发音合并"""
    fuzzySounds = defaultdict(str)
    pattern = re.compile('^[a-z]{1}h', re.I)
    pinyins = list(pinyin_confusion_set.keys())
    for _py in pinyins:
        if pattern.match(_py):
            meta_py = _py.replace('h', '')
            if meta_py in pinyins:
                fuzzySounds[_py] = meta_py
    # 模糊音 合并混淆集
    new_pinyin_confusion_set = deepcopy(pinyin_confusion_set)
    for item in fuzzySounds.items():
        src = item[0]
        tgt = item[1]
        new_pinyin_confusion_set[src] = pinyin_confusion_set[tgt] + pinyin_confusion_set[src]
        new_pinyin_confusion_set[tgt] = new_pinyin_confusion_set[src]
    return new_pinyin_confusion_set


def convert_synonyms_dict_format(existing_synonyms_dict_file:str, norm_format_file:str):
    """规范化同义词存储格式&合并已有词典"""
    # input_format: Aa01A05= 匹夫 个人
    _file = open(existing_synonyms_dict_file, 'r', encoding='utf-8')
    # 双层索引
    index_dict = defaultdict(int)  # {‘我’：index_id,}
    synonyms_dict = defaultdict(list)  # {index_id:[我的同义词list]}
    hierarch_synonyms_dict = defaultdict()
    for ind, line in enumerate(_file.readlines()):
        items = line.strip().split()
        for word in items[1:]:
            index_dict[word] = ind
        synonyms_dict[ind] = items[1:]
    hierarch_synonyms_dict['index_id'] = index_dict
    hierarch_synonyms_dict['index_val'] = synonyms_dict
    norm_format_f = open(norm_format_file, 'w', encoding='utf-8')
    json.dump(hierarch_synonyms_dict, norm_format_f, ensure_ascii=False)

    return hierarch_synonyms_dict


def convert_simChar_dict_format(existing_simChar_dict_file:str, norm_format_file:str):
    """规范化同形字字典"""
    # input_format: 龚	龛	詟	垄	陇
    _file = open(existing_simChar_dict_file, 'r', encoding='utf-8')
    # 双层索引
    index_dict = defaultdict(int)  # {‘我’：index_id,}
    simChar_dict = defaultdict(list)  # {index_id:[我的同义词list]}
    hierarch_simChar_dict = defaultdict()
    for ind, line in enumerate(_file.readlines()):
        items = line.strip().split()
        for word in items:
            index_dict[word] = ind
        simChar_dict[ind] = items
    hierarch_simChar_dict['index_id'] = index_dict
    hierarch_simChar_dict['index_val'] = simChar_dict
    norm_format_f = open(norm_format_file, 'w', encoding='utf-8')
    json.dump(hierarch_simChar_dict, norm_format_f, ensure_ascii=False)

    return hierarch_simChar_dict


def create_homoPY_confusionSet(corpus_dict_file, output_file, existing_confusion_set_file=None, charORword:str=None):
    """基于语料中出现过的词语 进行同音归类"""
    corpus_dict_f = open(corpus_dict_file, 'r', encoding='utf-8')
    corpus_dict = json.load(corpus_dict_f)
    print(f'{charORword} dict size:{len(corpus_dict)}')
    pinyin_dict = defaultdict(set)
    pattern = re.compile('^[\u4e00-\u9fa5]+')
    for item in corpus_dict:
        if pattern.match(item):
            _pinyin = pinyin(item, style=pypinyin.NORMAL)
            # 如果创建的是word dict,则只保留词语(2字以上)
            if charORword == "word":
                if len(_pinyin) > 1:
                    _pinyin_str = '_'.join([py[0] for py in _pinyin])
                else:
                    continue
            else:
                _pinyin_str = _pinyin[0][0]
            pinyin_dict[_pinyin_str].add(item)

    output_f = open(output_file, 'w', encoding='utf-8')
    # 如果已有部分同音混淆集 则进行补充
    if existing_confusion_set_file:
        existing_confusion_set_f = open(existing_confusion_set_file, 'r', encoding='utf-8')
        existing_confusion_set = json.load(existing_confusion_set_f)
        for i in existing_confusion_set:
            if i in pinyin_dict:
                pinyin_dict[i] = set.union(pinyin_dict[i], existing_confusion_set[i])
            else:
                pinyin_dict[i] = set(existing_confusion_set[i])
        print(f"已有同音混淆集规模：{len(existing_confusion_set)}")

    for _py in pinyin_dict:
        pinyin_dict[_py] = list(pinyin_dict[_py])
    if charORword == 'char':
        pinyin_dict = merge_fuzzySounds_for_char(pinyin_dict)

    print(f"现有同音混淆集规模：{len(pinyin_dict)}")
    json.dump(pinyin_dict, output_f, ensure_ascii=False)
    return pinyin_dict


def traditional2simplified_forDict(dict_files:dict, tgt_dir:str):
    """最初创建词典时忘记进行繁体-->简体转化 对已创建的词典 进行补救"""
    def for_homoPY_dict(dict_file, new_file):
        dict_f = open(dict_file, 'r', encoding='utf-8')
        org_dict = json.load(dict_f)
        new_dict = defaultdict(list)
        for _pinyin in org_dict:
            # print(_pinyin, org_dict[_pinyin])
            new_dict[_pinyin] = list(set([traditional2simplified(word) for word in org_dict[_pinyin]]))
        new_f = open(new_file, 'w', encoding='utf-8')
        json.dump(new_dict, new_f, ensure_ascii=False)

    def for_hierarchi_dict(dict_file, new_file):
        dict_f = open(dict_file, 'r', encoding='utf-8')
        org_dict = json.load(dict_f)
        new_dict = defaultdict()

        new_index_id = {traditional2simplified(_index):org_dict['index_id'][_index] for _index in org_dict['index_id']}
        new_index_val = {_index:[traditional2simplified(word) for word in org_dict['index_val'][_index]] for _index in
                          org_dict['index_val']}
        new_dict['index_id'] = new_index_id
        new_dict['index_val'] = new_index_val
        new_f = open(new_file, 'w', encoding='utf-8')
        json.dump(new_dict, new_f, ensure_ascii=False)

    for_homoPY_dict(dict_file=dict_files['homoPY_char'], new_file=os.path.join(tgt_dir, 'homoPY_char.json'))
    for_homoPY_dict(dict_file=dict_files['homoPY_word'], new_file=os.path.join(tgt_dir, 'homoPY_word.json'))
    for_hierarchi_dict(dict_file=dict_files['simChar'], new_file=os.path.join(tgt_dir, 'simChar.json'))
    for_hierarchi_dict(dict_file=dict_files['synonyms'], new_file=os.path.join(tgt_dir, 'synonyms.json'))


def load_pkl_file(file)->dict:
    data_f = open(file, 'rb')
    pkl_data = pickle.load(data_f)
    return pkl_data


def load_json_file(file)->dict:
    data_f = open(file, 'r', encoding='utf-8')
    json_data = json.load(data_f)
    return json_data


def merge_augment_sub_datasets(augment_dir, tgt_file):
    all_sub_dataset_files = get_allFiles_in_dir(augment_dir)
    tgt_f = open(tgt_file, 'w', encoding='utf-8')
    for sub_file in all_sub_dataset_files:
        sub_f = open(sub_file, 'r', encoding='utf-8')
        sub_dataset = sub_f.readlines()
        for _data in sub_dataset:
            _data = json.loads(_data.strip())

            _data['source'] = remove_illegal_chars(''.join(_data['source']))
            _data['target'] = remove_illegal_chars(_data['target'])
            # print(_data)
            _data = json.dumps(_data)
            tgt_f.write(f"{_data}\n")


# def create_simChar_confusion_set(corpus_char_dict_file, output_file, existing_confusion_set_file=None):
#     """基于语料的字典 构建字形混淆集 拓展已有混淆集""" #需要借助Faspell中的char_sim模块 计算字形相似度
#     corpus_char_dict_f = open(corpus_char_dict_file, 'r', encoding='uft-8')
#     corpus_char_dict = json.load(corpus_char_dict_f)


def main(args):
    # step1-7 可以顺序连续执行，step8 需要执行data_augment.py后才可执行

    # step1: 从原始数据中抽取需要的内容，并做清洗
    read_weixin_corpus(args.data_file, 200, args.target_file)
    # step2: 为手动多线程做准备 拆分整个大数据集为10份
    split_corpus(args.target_file, args.split_corpus_dir, num=10)
    # step3: 得到word-dict 并合并
    get_corpus_dict(data_file=args.target_file, output_file=args.word_dict_file, min_word_freq=5, charORword='word', batch_size=512)
    merge_dict(args.split_corpus_dir)
    # step4: 得到char-dict
    get_corpus_dict(data_file=args.target_file, output_file=args.char_dict_file, min_word_freq=5, charORword='char', batch_size=1)
    # step5: 基于前人的同义词及字形相似词表 转化为我们加载需要的格式
    convert_synonyms_dict_format('./同义词库.txt', './synonyms.json')
    convert_simChar_dict_format('./same_stroke.txt', './simChar.json')
    # step6: 基于当前语料构建的词表及字表， 构建同音词混淆集以及同音字混淆集
    create_homoPY_confusionSet(args.char_dict_file, '../../data/confusion_sets/homoPY_char.json', charORword='char')
    create_homoPY_confusionSet(args.word_dict_file, '../../data/confusion_sets/homoPY_word.json', charORword='word')
    # step7: 将已有的混淆集合中繁体字转化为简体
    traditional2simplified_forDict(dict_files={'homoPY_char':'../../data/confusion_sets/homoPY_char.json',
                                               'homoPY_word':'../../data/confusion_sets/homoPY_word.json',
                                               'simChar':'../../data/confusion_sets/simChar.json',
                                               'synonyms':'../../data/confusion_sets/synonyms.json'},
                                   tgt_dir='../../data/confusion_sets')
    # step8: 合并增强后的所有子数据集
    # merge_augment_sub_datasets(augment_dir=args.augment_dir,
    #                            tgt_file=os.path.join(args.augment_dir, 'merged_augment_data.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        help='Path to the train data',
                        default='data/news2016zh/news2016zh_valid.json')
    parser.add_argument('--output_dir',
                        help='Path to split data',
                        default='data/news2016zh/split_train')
    parser.add_argument('--split_file_num',
                        type=int,
                        default=10)
    # parser.add_argument('--target_file',
    #                     help='Path to the target data',
    #                     default='/data/shitianyuan/corpus/general_domain/weixin_corpus/content.pkl')
    # parser.add_argument('--split_corpus_dir',
    #                     help='Path to the split corpus data',
    #                     default='/data/shitianyuan/corpus/general_domain/weixin_corpus/split_content')
    # parser.add_argument('--char_dict_file',
    #                     help='Path to the corpus dict data',
    #                     default='/data/shitianyuan/corpus/general_domain/weixin_corpus/char_dict.json')
    # parser.add_argument('--word_dict_file',
    #                     help='Path to the corpus dict data',
    #                     default='/data/shitianyuan/corpus/general_domain/weixin_corpus/split_content/merged_dict.json')
    # parser.add_argument('--augment_dir',
    #                     help='dir of augment sub-dataset ',
    #                     default='/data/shitianyuan/corpus/general_domain/weixin_corpus/split_content/augment_file')
    # parser.add_argument('--cuda_device',
    #                     help='The number of GPU',
    #                     type=int,
    #                     default=0)

    args = parser.parse_args()

    # import hanlp
    # tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH, devices=args.cuda_device)
    #
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('/data/yanghh/MuCGEC-main/plm/chinese-struct-bert-large')
    #
    # tokenizers = {'word': tok_fine, 'char': tokenizer}

    # main(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    read_and_split_corpus(data_file=args.data_file, max_len=128, output_dir=args.output_dir, num=args.split_file_num)

    # data = load_pkl_file(os.path.join(output_dir, '1th.pkl'))
    # print(len(data))
    # for i in range(5):
    #     print(data[i])
    #     print(len(data[i]))