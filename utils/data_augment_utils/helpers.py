import os

from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
import argparse
import numpy as np
from tqdm import tqdm
import Levenshtein
import json
from utils.data_augment_utils.text_clean import remove_illegal_chars

VOCAB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR",
                  "dist_labels": '|ED_DIST|'}
REPLACEMENTS = {
    "''": '"',
    '--': '—',
    '`': "'",
    "'ve": "' ve",
}


def get_allFiles_in_dir(data_dir):
    filePaths = []  # 存储目录下的所有文件名，含路径
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return filePaths


def get_verb_form_dicts():
    path_to_dict = os.path.join(VOCAB_DIR, "verb-form-vocab.txt")
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = word2
    return encode, decode


ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()


def get_target_sent_by_edits(source_tokens, edits):
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)


def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens

    target_line = " ".join(tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()


def convert_using_case(token, smart_action):
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


def convert_using_plural(token, smart_action):
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":
            return source_token
        # deal with case
        if transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        if transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        if transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        if transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token


def read_parallel_lines(fn1, fn2, test_size=None):
    lines1 = read_lines(fn1)
    lines2 = read_lines(fn2)
    assert len(lines1) == len(lines2), print(len(lines1), len(lines2))
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    if test_size:
        out_lines1 = out_lines1[0: test_size]
        out_lines2 = out_lines2[0: test_size]
    return out_lines1, out_lines2


def read_pair_data_V2(file, keep_id=False):
    input_dataset = read_json_file(file)
    source, target, ids = [], [], []
    print("reading sub-dataset...........")
    for input_data in tqdm(input_dataset):
        input_data['source'] = remove_illegal_chars(''.join(input_data['source']))
        input_data['target'] = remove_illegal_chars(input_data['target'])

        # _source, _target = seg_func(input_data)
        _source = input_data['source']
        _target = input_data['target']

        if not _source.strip() or not _target.strip():
            continue
        else:
            source.append(_source)
            target.append(_target)

            if keep_id:
                ids.append(input_data['ID'])

    return source, target, ids


def read_json_file(json_file):
    json_f = open(json_file, 'r', encoding='utf-8')
    json_dataset = []
    for json_data in json_f:
        if json_data.strip():
            json_dataset.append(json.loads(json_data.strip()))
    return json_dataset


def read_lines(fn):
    if not os.path.exists(fn):
        print('data path not exist!!')
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip()]


def write_lines(fn, lines, mode='w', add_id:bool=False):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        for ind, line in enumerate(lines):
            if add_id:
                data_id = ''.join((8-len(str(ind+1)))*['0']) + str(ind+1)
                f.write(f"{data_id}\t{line}\n")
            else:
                f.write(f"{line}\n")


def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)


def encode_verb_form(original_word, corrected_word):
    decoding_request = original_word + "_" + corrected_word
    decoding_response = ENCODE_VERB_DICT.get(decoding_request, "").strip()
    if original_word and decoding_response:
        answer = decoding_response
    else:
        answer = None
    return answer


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'bert-large' and not lowercase:
        return 'bert-large-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'roberta-large':
        return 'roberta-large'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
    if transformer_name == 'xlnet-large':
        return 'xlnet-large-cased'
    if transformer_name == 'bert-base-chinese':
        return '/data/shitianyuan/bert-base-chinese'


def remove_double_tokens(sent):
    tokens = sent.split(' ')
    deleted_idx = []
    for i in range(len(tokens) -1):
        if tokens[i] == tokens[i + 1]:
            deleted_idx.append(i + 1)
    if deleted_idx:
        tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
    return ' '.join(tokens)


def normalize(sent):
    sent = remove_double_tokens(sent)
    for fr, to in REPLACEMENTS.items():
        sent = sent.replace(fr, to)
    return sent.lower()


def write2txt(dataset:list, data_path:str, with_id:bool=False):
    dataset_f = open(data_path, 'w', encoding='utf-8')
    dataset_size = 0
    for data in dataset:
        if isinstance(data, np.ndarray):
            data = data[0].strip()
        if not with_id:
            dataset_size += 1
            id = ''.join((9 - len(str(dataset_size))) * ['0']) + str(dataset_size)
            dataset_f.write(f"{id}\t{data}\n")
        else:
            dataset_f.write(f"{data}\n")


def labelset_cover_rate(pretrain_labels_file, current_labels_file, uncover_labels_file):
    '''
    计算两个label 集合之间的重叠率

    '''
    pretrain_label_f = open(pretrain_labels_file, 'r', encoding='utf-8')
    current_labels_f = open(current_labels_file, 'r', encoding='utf-8')
    uncover_labels_f = open(uncover_labels_file, 'w', encoding='utf-8')

    pretrain_labels = [dt.strip() for dt in pretrain_label_f.readlines()]
    current_labels = [dt.strip() for dt in current_labels_f.readlines()]
    cover_items = [label for label in current_labels if label in pretrain_labels]
    uncover_labels = [label for label in current_labels if label not in pretrain_labels]
    print(len(cover_items) / len(current_labels))
    print(f"uncover labels num: {len(uncover_labels)}")
    for i in uncover_labels:
        uncover_labels_f.write(f"{i}\n")

def filter_uncoverLabel_data(uncover_labels_file, test_data_file, test_label_file, filtered_test_file):
    '''
    过滤测试集中对应有uncover label的数据

    :return:
    '''
    uncover_labels_f = open(uncover_labels_file, 'r', encoding='utf-8')
    uncover_labels = [uncover_label.strip() for uncover_label in uncover_labels_f.readlines()]
    test_labels_f = open(test_label_file, 'r', encoding='utf-8')
    filtered_data_ids = []
    test_data_f = open(test_data_file, 'r', encoding='utf-8')
    filtered_test_f = open(filtered_test_file, 'w', encoding='utf-8')
    for line in test_labels_f:
        terms = line.strip().split(',')
        terms = [t.strip() for t in terms]
        pid = terms[0]

        if len(terms) == 2 and terms[-1] == '-1':
            continue

        if (len(terms) - 2) % 4 == 0:
            error_num = int((len(terms) - 2) / 4)
            for i in range(error_num):
                loc, typ, wrong, correct = terms[i * 4 + 1: (i + 1) * 4 + 1]
                norm_label = None
                if typ == '冗余' or typ == "语义重复":
                    norm_label = '$DELETE'
                elif typ == "别字":
                    norm_label = f"$REPLACE_{correct}"
                elif typ == '缺失':
                    norm_label = f"$APPEND_{correct}"
                elif typ == "句式杂糅" or typ == '乱序':
                    filtered_data_ids.append(pid)
                    break
                if not norm_label:
                    raise ValueError(f"漏掉错误类型：{typ}")

                if norm_label in uncover_labels:
                    filtered_data_ids.append(pid)
                    break

        else:
            continue

    print(f"需要从测试集中过滤 {len(filtered_data_ids)} 条数据。")
    for test_data in test_data_f.readlines():
        items = test_data.strip().split('\t')
        pid = items[0]
        if pid in filtered_data_ids:
            continue
        else:
            filtered_test_f.write(f"{pid}\t{items[1]}\n")


def sent2chars(src_file, tgt_file, weights_name):
    '''
    将句子保存为：char1 char2 ... 格式

    :param src_file:
    :param tgt_file:
    :param args:
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(weights_name)
    src_f = open(src_file, 'r', encoding='utf-8')
    tgt_f = open(tgt_file, 'w', encoding='utf-8')
    for src_dt in src_f.readlines():
        if src_dt.startswith('pid'):
            src_sent = src_dt.strip().split('\t')[1]
        else:
            src_sent = src_dt.strip()

        tgt_dt = ' '.join(tokenizer.tokenize(src_sent))
        tgt_f.write(f"{tgt_dt}\n")


def convert_NLPCC_gold2sent(test_input_file, gold_file, gold_norm_file, gold_text_file, gold_edit_file):
    '''
    将NLPCC 测试数据中的gold 转化为纠错后的文本 以及相应的edit labels

    :return:
    '''
    test_input_f = open(test_input_file, 'r', encoding='utf-8')
    gold_f = open(gold_file, 'r', encoding='utf-8')
    correct_items_f = open(gold_norm_file, 'r', encoding='utf-8')
    gold_text_f = open(gold_text_file, 'w', encoding='utf-8')
    gold_edit_f = open(gold_edit_file, 'w', encoding='utf-8')

    test_inputs = test_input_f.readlines()
    gold_sets = []
    for line in gold_f.readlines():
        if line.strip():
            items = line.strip().split()
            if items[0] == 'S':
                gold_sets.append(items[1:])

    correct_items = correct_items_f.readlines()
    test_dataset_size = len(correct_items)
    print(f'test dataset size: {test_dataset_size}')
    assert len(correct_items) == len(gold_sets) == len(test_inputs)
    for ind, items in tqdm(enumerate(zip(gold_sets, correct_items, test_inputs))):
        gold_set = items[0]
        _correct_items = eval(items[1])[0]
        test_input = items[2].strip()
        for _correct_item in reversed(_correct_items):
            s_ind, e_ind, src_sub, tgt_sub = _correct_item
            tgt_chars = tgt_sub[0].split()
            gold_set = gold_set[0: s_ind] + tgt_chars + gold_set[e_ind:]
        tgt_sent = ''.join(gold_set)
        gold_text_f.write(tgt_sent+'\n')

        edits = Levenshtein.opcodes(test_input, tgt_sent)

        # result = []
        # for edit in edits:
        #     if "。" in tgt_sent[edit[3]:edit[4]]:  # rm 。
        #         continue
        #     if edit[0] == "insert":
        #         result.append((str(edit[1]), "缺失", "", tgt_sent[edit[3]:edit[4]]))
        #     elif edit[0] == "replace":
        #         result.append((str(edit[1]), "别字", test_input[edit[1]:edit[2]], tgt_sent[edit[3]:edit[4]]))
        #     elif edit[0] == "delete":
        #         result.append((str(edit[1]), "冗余", test_input[edit[1]:edit[2]], ""))
        #
        # out_line = ""
        # for res in result:
        #     out_line += ', '.join(res) + ', '
        # test_pid = ''.join(((len(str(test_dataset_size)) + 1) - len(str(ind))) * ['0']) + str(ind)
        #
        # if out_line:
        #     normal_form = test_pid + ', ' + out_line.strip()
        # else:
        #     normal_form = test_pid + ', -1'
        # gold_edit_f.write(f"{normal_form}\n")


def addID2text(input_file, output_file, seg=False, weight_name=None):
    tokenizer = AutoTokenizer.from_pretrained(weight_name)

    input_f = open(input_file, 'r', encoding='utf-8')
    output_f = open(output_file, 'w', encoding='utf-8')
    input_data = input_f.readlines()
    input_data_size = len(input_data)
    for ind, inp_dt in enumerate(input_data):
        pid = ''.join(((len(str(input_data_size)) + 1) - len(str(ind))) * ['0']) + str(ind)
        if seg:
            inp_segs = ' '.join(tokenizer.tokenize(inp_dt.strip()))
            output_f.write(f"{pid}\t{inp_segs}\n")
        else:
            output_f.write(f"{pid}\t{inp_dt.strip()}\n")

def main(args):
    '''
    将test data 分字后保存 用于预测输入

    :param args:
    :return:
    '''
    filter_uncoverLabel_data(uncover_labels_file=args.uncover_labels_file,
                             test_data_file=args.test_data_file,
                             test_label_file=args.test_label_file,
                             filtered_test_file=args.filtered_test_file)
    sent2chars(args.filtered_test_file, args.test_input, args.weights_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--uncover_labels_file',
                        help='Path to the train data',
                        default='../data/vocabulary/uncover_labels.txt')  # required=True
    parser.add_argument('--test_data_file',
                        help='dataset name',
                        default='../data/CTC2021/qua_input.txt')  # required=True
    parser.add_argument('--test_label_file',
                        help='dataset name',
                        default='../data/CTC2021/qua.solution')  # required=True
    parser.add_argument('--filtered_test_file',
                        help='dataset name',
                        default='../data/CTC2021/test_filtered.txt')  # required=True
    parser.add_argument('--test_input',
                        help='dataset name',
                        default='../data/CTC2021/test_input_char.txt')  # required=True
    parser.add_argument('--weights_name',
                        help='pretrained model name or path',
                        default='/data/shitianyuan/PLMs/chinese-struct-bert-large')  # required=True
    args = parser.parse_args()

    # main(args)
    # convert_NLPCC_gold2sent('../data/NLPCC2018/NLPCC_testdata/source.txt',
    #                         '../data/NLPCC2018/NLPCC_testdata/gold/gold.0',
    #                         '../data/NLPCC2018/NLPCC_testdata/gold/gold.norm',
    #                         '../data/NLPCC2018/NLPCC_testdata/gold/gold.text',
    #                         '../data/NLPCC2018/NLPCC_testdata/gold/gold.edits')
    addID2text('../data/NLPCC2018/NLPCC_testdata/source.txt',
               '../data/NLPCC2018/NLPCC_testdata/source_seg.txt',
               True,
               '/data/shitianyuan/PLMs/chinese-struct-bert-large')