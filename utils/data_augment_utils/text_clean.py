import six
import re
import pypinyin
from pypinyin import pinyin
from collections import Counter
from utils.data_augment_utils.data_augment_dicts.langconv import Converter


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()


def remove_illegal_chars(strs):
    """去除空字符 非法字符"""
    strs = re.sub('[\s]+|[^a-zA-Z0-9\u4e00-\u9fa5+——！，。？、~@#￥%……&*（）\.\!\/<>“”,$%^*(+\"\']+'
                  '|^[+——！，。？、~@#￥%……&*\.\!\/,$%^*]+', '', strs.strip())
    return strs


def traditional2simplified(sentence):
    """
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    """
    return Converter('zh-hans').convert(sentence)


def is_sent_ok(sent:str, min_len=10)->bool:
    """句子中含有的数字或者字母是否过多 句子本身是否过短"""
    sent_len = len(sent)
    if sent.strip():
        if sent_len < min_len:
            return False
        if len(re.findall('[a-z0-9\s]', sent, re.I)) / sent_len > 0.1:
            return False
        else:
            return True
    else:
        return False


def split_sent(sent, min_len=None):
    """拆分原始长句"""
    sub_sents = []
    puncs = re.findall('[！。？~……\.\!;；]+', sent)
    if puncs:
        puncs_statis = {item[0]:item[1] for item in sorted(Counter(puncs).items(), key=lambda x: x[1], reverse=True)}
        split_punc = list(puncs_statis.keys())[0]
        for sub_sent in sent.split(split_punc):
            if min_len:
                if is_sent_ok(sub_sent, min_len=min_len):
                    sub_sents.append(sub_sent + split_punc)
            else:
                sub_sents.append(sub_sent + split_punc)
        return sub_sents
    else:
        return [sent]


if __name__ == "__main__":
    str1 = ',为...了保证活动通信畅通，请记好18937788989、18623770733、15938818399、13507639651电话，QQ群179921554，微信18937788989，微信公众号0377天涯户外请注意防晒。请携带防晒用品，霜、遮阳伞、遮阳帽等。'
    res = remove_illegal_chars(str1)
    res = split_sent(res, min_len=10)
    print(res)