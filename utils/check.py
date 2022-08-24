import json
import difflib

data1 = json.load(open('/data/yanghh/MiduCTC-competition/final_data/final_train+val.json', 'r', encoding='utf-8'))
data2 = json.load(open('/data/yanghh/MiduCTC-competition/final_data/final_val.json', 'r', encoding='utf-8'))
src1_list = [d['source'] for d in data1]
src2_list = [d['source'] for d in data2]

print(len(src1_list))
print(len(src2_list))
scores = [(difflib.SequenceMatcher(None, s2, s1).quick_ratio(), s1, s2) for s1 in src1_list for s2 in src2_list]
print(len(scores))
scores.sort(key=lambda x : x[0], reverse=True)
print(scores[:10])


